import logging
from datetime import datetime, timedelta
import pandas as pd

import numpy as np
from scipy.stats import norm as _scipy_norm

# Minimum predicted sigma (°F) after residual reframing. Residual targets have
# std ~5°F vs ~18°F absolute; models can produce near-zero sigma on easy cases.
# This floor prevents overconfident Kelly sizing on any single forecast.
# Must match RESIDUAL_SIGMA_FLOOR in backtest/runner.py.
RESIDUAL_SIGMA_FLOOR = 2.0

# Clamp the returned fair value away from 0/1. Low-variance coastal stations
# (e.g. KSFO, KLAX) can produce raw probabilities ≥ 0.99 that calibrate to
# exactly 1.0 — a "zero loss probability" fair value drives overconfident Kelly
# sizing on thin edge cushions. Bounding to [0.02, 0.98] preserves a residual
# uncertainty margin without affecting any in-range forecast.
FAIR_VALUE_FLOOR = 0.02
FAIR_VALUE_CEIL = 0.98

from ingestion.asos import fetch_daily_tmax_history
from ingestion.gefs import fetch_latest_gefs_run, detect_new_run, FORECAST_HOURS
from ingestion.ecmwf import fetch_latest_ecmwf_run
from ingestion.nbm import fetch_latest_nbm
from config.series import is_low_temp_series
from processing.asos_history import compute_daily_residuals, build_asos_history_df
from processing.bias_correction import BiasCorrectionRegistry, get_lead_bucket, get_season
from processing.features import build_feature_matrix, get_feature_columns
from models.spread_inflation import apply_spread_inflation_from_stats
from config.stations import ALL_ICAO
from db.forecast_log import record_forecast_tmax, get_forecast_tmax
from db.models import ForecastRun
from db.session import get_session

logger = logging.getLogger(__name__)

STATIONS = ALL_ICAO
RESOLUTION_WINDOW_DAYS = 7

# Below this overall GEFS coverage, the scheduler should keep retrying
# ingestion (cheap — fetch_latest_gefs_run only re-downloads missing files)
# rather than waiting for the next 6-hourly GEFS cycle.
RETRY_COVERAGE_THRESHOLD = 0.9

# Below this per-lead-bucket GEFS coverage, skip updating fair values for
# tickers in that bucket rather than quoting off empty/zero-filled features.
GATE_COVERAGE_THRESHOLD = 0.5


class EnsembleStrategy:
    def __init__(self, shared_state, model_registry, kalshi_client, settings) -> None:
        self._state = shared_state
        self._registry = model_registry
        self._client = kalshi_client
        self._settings = settings
        self._bias_registry = BiasCorrectionRegistry()
        self._last_run_time: datetime = datetime.min
        self.last_gefs_coverage_pct: float = 1.0

    def needs_retry(self) -> bool:
        """True if the last cycle's GEFS coverage was low enough that the
        scheduler should retry ingestion before the next 6-hourly cycle."""
        return self.last_gefs_coverage_pct < RETRY_COVERAGE_THRESHOLD

    @staticmethod
    def _compute_coverage_by_bucket(gefs_raw: dict, stations: list[str]) -> dict[str, float]:
        """Fraction of expected (member × station × lead-hour) GEFS records
        received, broken down by lead bucket (D1-2, D3-4, D5-7)."""
        expected = {"D1-2": 0, "D3-4": 0, "D5-7": 0}
        received = {"D1-2": 0, "D3-4": 0, "D5-7": 0}
        for station in stations:
            for lead_hour in FORECAST_HOURS:
                bucket = get_lead_bucket(lead_hour)
                expected[bucket] += 31
                received[bucket] += len(gefs_raw.get(station, {}).get(lead_hour, []))
        return {
            bucket: (received[bucket] / expected[bucket] if expected[bucket] else 0.0)
            for bucket in expected
        }

    @staticmethod
    def _record_tomorrows_forecast(feature_df: "pd.DataFrame", today=None) -> None:
        """Persist each station's GEFS D+1 (lead_hour=24) Tmax forecast as the
        prediction for tomorrow, so a future cycle can diff it against the
        observed Tmax to populate obs_minus_model_lag1/2/3."""
        target_date = (today or datetime.utcnow().date()) + timedelta(days=1)
        d1_rows = feature_df[feature_df["lead_hour"] == 24]
        for _, row in d1_rows.iterrows():
            record_forecast_tmax(row["station"], target_date, float(row["gefs_tmax_mean"]))

    @staticmethod
    async def _build_live_asos_history(stations: list[str], lookback_days: int = 7) -> "pd.DataFrame":
        """Build asos_history (obs_minus_model residuals) from recent observed
        Tmax and previously-persisted GEFS D+1 forecasts."""
        residuals_by_station: dict[str, dict] = {}
        for station in stations:
            actual_tmax = await fetch_daily_tmax_history(station, days=lookback_days)
            if not actual_tmax:
                continue
            forecast_tmax = get_forecast_tmax(station, list(actual_tmax.keys()))
            residuals = compute_daily_residuals(actual_tmax, forecast_tmax)
            if residuals:
                residuals_by_station[station] = residuals
        return build_asos_history_df(residuals_by_station)

    @staticmethod
    def _compute_fair_value(
        X: "pd.DataFrame",
        ngboost_model,
        qrf_model,
        residual_model,
        blender,
        calibrator,
        threshold: float,
        ecmwf_offset: float = 0.0,
    ) -> dict:
        mu, sigma = ngboost_model.predict_distribution(X)

        if residual_model is not None:
            try:
                mu = mu + residual_model.predict(X)
            except Exception:
                pass

        # Shift residual predictions to absolute temperature scale.
        # Models train on (actual_tmax - ecmwf_tmax); ecmwf_offset converts back.
        mu = mu + ecmwf_offset

        if "gefs_tmax_std" in X.columns and "gefs_tmax_range" in X.columns:
            std_arr = X["gefs_tmax_std"].fillna(0).values
            range_arr = X["gefs_tmax_range"].fillna(0).values
            _, sigma = apply_spread_inflation_from_stats(mu, sigma, std_arr, range_arr)

        # Enforce minimum sigma: residual reframing shrinks the target std from
        # ~18°F to ~5°F, which can produce overconfident distributions on easy
        # forecasts and drive excessive Kelly bet sizing.
        sigma = np.maximum(sigma, RESIDUAL_SIGMA_FLOOR)

        raw_prob = float(1.0 - _scipy_norm.cdf(threshold, loc=mu[0], scale=max(sigma[0], 0.01)))

        if qrf_model is not None and blender is not None:
            try:
                # QRF also predicts in residual space; shift threshold to match
                qrf_prob = float(qrf_model.predict_prob_above(X, threshold - ecmwf_offset)[0])
                raw_prob = float(
                    blender.weights["ngboost"] * raw_prob
                    + blender.weights["qrf"] * qrf_prob
                )
            except Exception:
                pass

        if calibrator is not None:
            cal_prob, ci_lo, ci_hi = calibrator.calibrate(raw_prob)
            ci_width = ci_hi - ci_lo
        else:
            cal_prob = raw_prob
            ci_width = 0.1

        cal_prob = min(FAIR_VALUE_CEIL, max(FAIR_VALUE_FLOOR, float(cal_prob)))

        return {"raw_prob": raw_prob, "cal_prob": float(cal_prob), "ci_width": float(ci_width)}

    async def run_cycle(self) -> None:
        logger.info("EnsembleStrategy: starting cycle")
        try:
            gefs_raw = await fetch_latest_gefs_run()
            ecmwf_raw = await fetch_latest_ecmwf_run()
            nbm_raw = await fetch_latest_nbm()
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            self._state.post_alert("ingestion_error", f"Data ingestion failed: {exc}")
            return

        # Check GEFS member coverage and alert on significant gaps
        total_expected = len(STATIONS) * len(FORECAST_HOURS) * 31
        total_received = sum(
            len(gefs_raw.get(s, {}).get(lh, []))
            for s in STATIONS for lh in FORECAST_HOURS
        )
        coverage_pct = total_received / total_expected if total_expected > 0 else 0
        self.last_gefs_coverage_pct = coverage_pct
        if coverage_pct < 0.5:
            self._state.post_alert(
                "gefs_coverage",
                f"GEFS member coverage low: {total_received}/{total_expected} "
                f"({coverage_pct:.0%}) — forecasts may be unreliable",
            )
        elif coverage_pct >= 0.9:
            self._state.clear_alerts("gefs_coverage")

        bucket_coverage = self._compute_coverage_by_bucket(gefs_raw, STATIONS)

        # Apply Kalman bias correction to GEFS member temperatures
        for station in STATIONS:
            station_data = gefs_raw.get(station, {})
            for lead_hour, member_list in station_data.items():
                lead_bucket = get_lead_bucket(lead_hour)
                season = get_season(datetime.utcnow().month)
                corrector = self._bias_registry.get_corrector(station, lead_bucket, season)
                for member in member_list:
                    tf = member.get("temp_f", float("nan"))
                    import math
                    if not math.isnan(tf):
                        member["temp_f"] = corrector.correct(tf)

        try:
            asos_history = await self._build_live_asos_history(STATIONS)
        except Exception as exc:
            logger.warning("Failed to build live ASOS history: %s", exc)
            asos_history = pd.DataFrame()

        feature_df = build_feature_matrix(
            gefs_data=gefs_raw,
            ecmwf_data=ecmwf_raw,
            asos_history=asos_history,
            nbm_data=nbm_raw,
            station_meta=None,
        )

        if feature_df.empty:
            logger.warning("Empty feature matrix — skipping order updates")
            return

        try:
            self._record_tomorrows_forecast(feature_df)
        except Exception as exc:
            logger.warning("Failed to record forecast log: %s", exc)

        feature_cols = get_feature_columns()
        available_cols = [c for c in feature_cols if c in feature_df.columns]

        # Registry stores dicts keyed by station for per-station models
        ngboost_by_station  = self._registry.get("ngboost") or {}
        qrf_by_station      = self._registry.get("qrf") or {}
        residual_by_station = self._registry.get("residual") or {}
        blender             = self._registry.get("blender")
        calibrators         = self._registry.get("calibrators") or {}

        if not ngboost_by_station:
            logger.warning("No trained NGBoost models in registry — skipping cycle")
            return

        active_tickers = await self.fetch_active_temperature_tickers()

        for ticker in active_tickers:
            try:
                station = self._ticker_to_station(ticker)
                threshold = self._ticker_to_threshold(ticker)
                horizon = self._ticker_to_horizon(ticker)
                if station is None or threshold is None:
                    continue

                station_rows = feature_df[feature_df["station"] == station]
                if station_rows.empty:
                    continue

                # Pick the row whose lead_hour is closest to the ticker's horizon
                target_lead = horizon * 24
                if "lead_hour" in station_rows.columns:
                    idx = (station_rows["lead_hour"] - target_lead).abs().argmin()
                    closest_row = station_rows.iloc[[idx]]
                else:
                    closest_row = station_rows.iloc[[0]]
                X = closest_row[available_cols].fillna(0.0)
                ecmwf_offset = float(
                    closest_row["ecmwf_tmax"].iloc[0]
                    if "ecmwf_tmax" in closest_row.columns
                    else 0.0
                )

                lead_bucket = "D1-2" if horizon <= 2 else "D3-4" if horizon <= 4 else "D5-7"
                model_key = f"{station}_{lead_bucket}"

                ngboost_model  = ngboost_by_station.get(model_key) or ngboost_by_station.get(station)
                qrf_model      = qrf_by_station.get(model_key) or qrf_by_station.get(station)
                residual_model = residual_by_station.get(model_key) or residual_by_station.get(station)
                if ngboost_model is None:
                    logger.debug("No model for station %s lead_bucket %s", station, lead_bucket)
                    continue

                if bucket_coverage.get(lead_bucket, 1.0) < GATE_COVERAGE_THRESHOLD:
                    logger.info(
                        "Skipping %s — %s GEFS coverage too low (%.0f%%)",
                        ticker, lead_bucket, bucket_coverage.get(lead_bucket, 0.0) * 100,
                    )
                    continue

                cal_key = f"{station}_{lead_bucket}"
                calibrator = calibrators.get(cal_key)

                try:
                    fv = self._compute_fair_value(
                        X=X,
                        ngboost_model=ngboost_model,
                        qrf_model=qrf_model,
                        residual_model=residual_model,
                        blender=blender,
                        calibrator=calibrator,
                        threshold=threshold,
                        ecmwf_offset=ecmwf_offset,
                    )
                except Exception as exc:
                    logger.warning("Model inference failed for %s: %s", ticker, exc)
                    continue

                cal_prob = fv["cal_prob"]
                ci_width = fv["ci_width"]

                self._state.update_fair_a(ticker, cal_prob, ci_width, horizon)

                with get_session() as db:
                    db.add(ForecastRun(
                        station=station,
                        model_source="blend",
                        run_time=datetime.utcnow(),
                        lead_time_hours=horizon * 24,
                        mu=fv["raw_prob"],
                        sigma=ci_width,
                        calibrated_prob=cal_prob,
                        ci_lower=cal_prob - ci_width / 2,
                        ci_upper=cal_prob + ci_width / 2,
                        threshold=threshold,
                    ))

                logger.info("Updated %s: fair_a=%.3f ci=%.3f", ticker, cal_prob, ci_width)

            except Exception as exc:
                logger.error("Failed processing ticker %s: %s", ticker, exc)

        self._last_run_time = datetime.utcnow()
        logger.info("EnsembleStrategy cycle complete; updated %d tickers", len(active_tickers))

    async def fetch_active_temperature_tickers(self) -> list[str]:
        """
        Fetch open temperature markets by querying each known series directly.
        The live Kalshi API does not reliably tag temperature markets with a
        category, so category-based filtering returns 0 results.
        """
        tickers: list[str] = []
        try:
            for series in self._SERIES_TO_STATION:
                if is_low_temp_series(series):
                    continue
                try:
                    data = await self._client._request(
                        "GET", "/markets",
                        params={"series_ticker": series, "status": "open", "limit": 50},
                    )
                    for market in data.get("markets", []):
                        ticker = market.get("ticker", "")
                        if self._ticker_to_station(ticker) is not None:
                            tickers.append(ticker)
                except Exception:
                    pass  # series may not exist on this environment
            logger.info("Found %d active temperature tickers across %d series",
                        len(tickers), len(self._SERIES_TO_STATION))
        except Exception as exc:
            logger.error("Failed to fetch active tickers: %s", exc)
        return tickers

    def detect_new_model_run(self, last_run_ts: datetime) -> bool:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, detect_new_run(last_run_ts))
                    return future.result(timeout=15)
            return loop.run_until_complete(detect_new_run(last_run_ts))
        except Exception as exc:
            logger.warning("Run detection check failed: %s", exc)
            return False

    # Series prefix → ASOS station. Matches what Kalshi actually uses.
    _SERIES_TO_STATION: dict[str, str] = {
        "KXHIGHCHI":  "KORD", "KXLOWTCHI":  "KORD",
        "KXHIGHNY":   "KLGA", "KXHIGHNY0":  "KLGA", "KXLOWTNYC":  "KLGA",
        "KXHIGHLAX":  "KLAX", "KXLOWTLAX":  "KLAX",
        "KXHIGHMIA":  "KMIA", "KXLOWTMIA":  "KMIA",
        "KXHIGHOU":   "KIAH", "KXHIGHHOU":  "KIAH", "KXHOUHIGH":  "KIAH",
        "KXLOWTHOU":  "KIAH", "KXHIGHTHOU": "KIAH",
        "KXHIGHPHIL": "KPHL", "KXLOWTPHIL": "KPHL",
        "KXHIGHATL":  "KATL", "KXHIGHTATL": "KATL", "KXLOWTATL":  "KATL",
        "KXHIGHAUS":  "KAUS", "KXLOWTAUS":  "KAUS",
        "KXDENHIGH":  "KDEN", "KXHIGHDEN":  "KDEN", "KXLOWTDEN":  "KDEN",
        "KXHIGHTPHX": "KPHX", "KXLOWTPHX":  "KPHX",
        "KXHIGHTSFO": "KSFO", "KXLOWTSFO":  "KSFO",
        "KXHIGHTSEA": "KSEA", "KXLOWTSEA":  "KSEA",
        "KXHIGHTBOS": "KBOS", "KXLOWTBOS":  "KBOS",
        "KXHIGHTDAL": "KDFW", "KXLOWTDAL":  "KDFW",
        "KXHIGHTDC":  "KDCA", "KXLOWTDC":   "KDCA",
        "KXHIGHTLV":  "KLAS", "KXLOWTLV":   "KLAS",
        "KXHIGHTMIN": "KMSP", "KXLOWTMIN":  "KMSP",
        "KXHIGHTOKC": "KOKC", "KXLOWTOKC":  "KOKC",
        "KXHIGHTSATX":"KSAT", "KXLOWTSATX": "KSAT",
        "KXHIGHTNOLA":"KMSY", "KXLOWTNOLA": "KMSY",
    }

    @classmethod
    def _ticker_to_station(cls, ticker: str) -> str | None:
        # Ticker format: KXHIGHCHI-26MAY31-T81 → series=KXHIGHCHI
        series = ticker.split("-")[0]
        return cls._SERIES_TO_STATION.get(series)

    @staticmethod
    def _ticker_to_threshold(ticker: str) -> float | None:
        import re
        # Format: ...-T81 or ...-B80.5 (T=above, B=below)
        m = re.search(r"-[TB]([\d.]+)$", ticker)
        if m:
            return float(m.group(1))
        return None

    @staticmethod
    def _ticker_to_horizon(ticker: str) -> int:
        import re
        from datetime import date as date_type
        # Format: KXHIGHCHI-26MAY31-T81 → date part = 26MAY31 = May 31 2026
        parts = ticker.split("-")
        if len(parts) >= 2:
            try:
                tdate = datetime.strptime(parts[1], "%y%b%d").date()
                delta = (tdate - date_type.today()).days
                return max(1, delta)
            except ValueError:
                pass
        return 1
