import logging
from datetime import datetime
import pandas as pd

from ingestion.gefs import fetch_latest_gefs_run, detect_new_run
from ingestion.ecmwf import fetch_latest_ecmwf_run
from ingestion.nbm import fetch_latest_nbm
from processing.bias_correction import BiasCorrectionRegistry, get_lead_bucket, get_season
from processing.features import build_feature_matrix, get_feature_columns
from config.stations import ALL_ICAO
from db.models import ForecastRun
from db.session import get_session

logger = logging.getLogger(__name__)

STATIONS = ALL_ICAO
RESOLUTION_WINDOW_DAYS = 7


class EnsembleStrategy:
    def __init__(self, shared_state, model_registry, kalshi_client, settings) -> None:
        self._state = shared_state
        self._registry = model_registry
        self._client = kalshi_client
        self._settings = settings
        self._bias_registry = BiasCorrectionRegistry()
        self._last_run_time: datetime = datetime.min

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
        from ingestion.gefs import FORECAST_HOURS
        total_expected = len(STATIONS) * len(FORECAST_HOURS) * 31
        total_received = sum(
            len(gefs_raw.get(s, {}).get(lh, []))
            for s in STATIONS for lh in FORECAST_HOURS
        )
        coverage_pct = total_received / total_expected if total_expected > 0 else 0
        if coverage_pct < 0.5:
            self._state.post_alert(
                "gefs_coverage",
                f"GEFS member coverage low: {total_received}/{total_expected} "
                f"({coverage_pct:.0%}) — forecasts may be unreliable",
            )
        elif coverage_pct >= 0.9:
            self._state.clear_alerts("gefs_coverage")

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

        feature_df = build_feature_matrix(
            gefs_data=gefs_raw,
            ecmwf_data=ecmwf_raw,
            asos_history=pd.DataFrame(),
            regime_labels=pd.Series(dtype=float),
            nbm_data=nbm_raw,
            station_meta=None,
        )

        if feature_df.empty:
            logger.warning("Empty feature matrix — skipping order updates")
            return

        feature_cols = get_feature_columns()
        available_cols = [c for c in feature_cols if c in feature_df.columns]

        # Registry stores dicts keyed by station for per-station models
        ngboost_by_station = self._registry.get("ngboost") or {}
        qrf_by_station     = self._registry.get("qrf") or {}
        blender            = self._registry.get("blender")
        calibrators        = self._registry.get("calibrators") or {}

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

                ngboost_model = ngboost_by_station.get(station)
                qrf_model     = qrf_by_station.get(station)
                if ngboost_model is None:
                    logger.debug("No model for station %s", station)
                    continue

                try:
                    ng_prob = ngboost_model.predict_prob_above(X, threshold)
                    if qrf_model is not None and blender is not None:
                        qrf_prob = qrf_model.predict_prob_above(X, threshold)
                        blended_prob = blender.blend_probs(ng_prob, qrf_prob)
                    else:
                        blended_prob = ng_prob
                except Exception as exc:
                    logger.warning("Model inference failed for %s: %s", ticker, exc)
                    continue

                raw_prob = float(blended_prob[0])
                lead_bucket = "D1-2" if horizon <= 2 else "D3-4" if horizon <= 4 else "D5-7"
                cal_key = f"{station}_{lead_bucket}"
                calibrator = calibrators.get(cal_key)
                if calibrator is not None:
                    cal_prob, ci_lo, ci_hi = calibrator.calibrate(raw_prob)
                    ci_width = ci_hi - ci_lo
                else:
                    cal_prob = raw_prob
                    ci_width = 0.1

                self._state.update_fair_a(ticker, cal_prob, ci_width, horizon)

                with get_session() as db:
                    db.add(ForecastRun(
                        station=station,
                        model_source="blend",
                        run_time=datetime.utcnow(),
                        lead_time_hours=horizon * 24,
                        mu=cal_prob,
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
        "KXHIGHCHI": "KORD", "KXLOWTCHI": "KORD",
        "KXHIGHNY":  "KLGA", "KXHIGHNY0": "KLGA", "KXLOWNYC":  "KLGA",
        "KXHIGHLAX": "KLAX", "KXLOWTLAX": "KLAX",
        "KXHIGHMIA": "KMIA", "KXLOWMIA":  "KMIA",
        "KXHIGHOU":  "KIAH", "KXHIGHHOU": "KIAH", "KXLOWTHOU": "KIAH",
        "KXHIGHPHIL":"KPHL", "KXLOWPHIL": "KPHL",
        "KXHIGHATL": "KATL", "KXLOWTATL": "KATL",
        "KXHIGHAUS": "KAUS", "KXLOWTAUS": "KAUS",
        "KXDENHIGH": "KDEN", "KXHIGHDEN": "KDEN", "KXLOWDEN":  "KDEN",
        "KXHIGHTPHX":"KPHX", "KXLOWTPHX": "KPHX",
        "KXHIGHTSFO":"KSFO", "KXLOWTSFO": "KSFO",
        "KXHIGHTSEA":"KSEA", "KXLOWTSEA": "KSEA",
        "KXHIGHTBOS":"KBOS", "KXLOWTBOS": "KBOS",
        "KXHIGHTDAL":"KDFW", "KXLOWTDAL": "KDFW",
        "KXHIGHTDC": "KDCA", "KXLOWTDC":  "KDCA",
        "KXHIGHTLV": "KLAS", "KXLOWTLV":  "KLAS",
        "KXHIGHTMIN":"KMSP", "KXLOWTMIN": "KMSP",
        "KXHIGHTOKC":"KOKC", "KXLOWTOKC": "KOKC",
        "KXHIGHTSATX":"KSAT","KXLOWTSATX":"KSAT",
        "KXHIGHTNOLA":"KMSY","KXLOWTNOLA":"KMSY",
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
