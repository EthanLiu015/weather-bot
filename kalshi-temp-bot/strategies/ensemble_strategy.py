import logging
from datetime import datetime
import pandas as pd

from ingestion.gefs import fetch_latest_gefs_run, detect_new_run
from ingestion.ecmwf import fetch_latest_ecmwf_run
from ingestion.qc import qc_metar_list
from processing.downscaling import downscale_gefs_to_station, STATION_META
from processing.bias_correction import BiasCorrectionRegistry, get_lead_bucket, get_season
from processing.features import build_feature_matrix, get_feature_columns
from db.models import ForecastRun
from db.session import get_session

logger = logging.getLogger(__name__)

STATIONS = ["KORD", "KJFK", "KLAX"]
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
        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            return

        gefs_by_station = {}
        for station in STATIONS:
            try:
                gefs_by_station[station] = downscale_gefs_to_station(
                    gefs_data=gefs_raw.get(station, {}),
                    station=station,
                )
            except Exception as exc:
                logger.warning("Downscaling failed for %s: %s", station, exc)
                gefs_by_station[station] = {}

        # Bias correct each station/lead
        gefs_corrected = {}
        for station in STATIONS:
            df = gefs_by_station.get(station)
            if df is None or (hasattr(df, 'empty') and df.empty):
                gefs_corrected[station] = {}
                continue
            corrected_members = {}
            for member, mdf in gefs_raw.get(station, {}).items():
                corrected_df = mdf.copy()
                for idx, row in mdf.iterrows():
                    lead = row.get("lead_hour", 24)
                    lead_bucket = get_lead_bucket(lead)
                    season = get_season(datetime.utcnow().month)
                    corrector = self._bias_registry.get_corrector(station, lead_bucket, season)
                    if "t2m" in corrected_df.columns:
                        corrected_df.at[idx, "t2m"] = corrector.correct(row.get("t2m", float("nan")))
                corrected_members[member] = corrected_df
            gefs_corrected[station] = corrected_members

        gefs_for_features = {s: gefs_raw.get(s, {}) for s in STATIONS}
        feature_df = build_feature_matrix(
            gefs_data=gefs_for_features,
            ecmwf_data=ecmwf_raw,
            asos_history=pd.DataFrame(),
            regime_labels=pd.Series(dtype=float),
            station_meta=None,
        )

        if feature_df.empty:
            logger.warning("Empty feature matrix — skipping order updates")
            return

        feature_cols = get_feature_columns()
        available_cols = [c for c in feature_cols if c in feature_df.columns]

        try:
            ngboost_model = self._registry.get("ngboost")
            qrf_model = self._registry.get("qrf")
            blender = self._registry.get("blender")
            calibrators = self._registry.get("calibrators", {})
        except Exception as exc:
            logger.warning("Model registry lookup failed: %s", exc)
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

                closest_row = station_rows.iloc[[0]]
                X = closest_row[available_cols].fillna(0.0)

                try:
                    ng_prob = ngboost_model.predict_prob_above(X, threshold)
                    qrf_prob = qrf_model.predict_prob_above(X, threshold)
                    blended_prob = blender.blend_probs(ng_prob, qrf_prob)
                except Exception as exc:
                    logger.warning("Model inference failed for %s: %s", ticker, exc)
                    continue

                raw_prob = float(blended_prob[0])
                cal_key = f"{station}_D{min(horizon, 5)}"
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
        try:
            markets = await self._client.get_markets(status="open", category="temperature")
            tickers = []
            for market in markets:
                ticker = market.get("ticker", "")
                station_match = any(s.replace("K", "") in ticker for s in STATIONS)
                if station_match:
                    tickers.append(ticker)
            return tickers[:50]
        except Exception as exc:
            logger.error("Failed to fetch active tickers: %s", exc)
            return []

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

    @staticmethod
    def _ticker_to_station(ticker: str) -> str | None:
        for station in STATIONS:
            shortname = station[1:]
            if shortname in ticker:
                return station
        return None

    @staticmethod
    def _ticker_to_threshold(ticker: str) -> float | None:
        import re
        m = re.search(r"(\d+)", ticker)
        if m:
            return float(m.group(1))
        return None

    @staticmethod
    def _ticker_to_horizon(ticker: str) -> int:
        from datetime import date
        import re
        m = re.search(r"(\d{8})", ticker)
        if m:
            try:
                tdate = datetime.strptime(m.group(1), "%Y%m%d").date()
                delta = (tdate - date.today()).days
                return max(1, delta)
            except ValueError:
                pass
        return 1
