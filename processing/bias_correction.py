import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

SEASONS = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
LEAD_BUCKETS = ["D1-2", "D3-4", "D5-7"]
LEAD_BUCKET_HOUR_RANGES = [("D1-2", 0, 48), ("D3-4", 49, 96), ("D5-7", 97, 999)]


def get_lead_bucket(lead_time_hours: int) -> str:
    if lead_time_hours <= 48:
        return "D1-2"
    elif lead_time_hours <= 96:
        return "D3-4"
    return "D5-7"


def get_season(month: int) -> str:
    return SEASONS.get(month, "DJF")


class KalmanBiasCorrector:
    def __init__(self, process_noise: float = 0.1, obs_noise: float = 1.5) -> None:
        self.process_noise = process_noise
        self.obs_noise = obs_noise
        self._bias_estimate: float = 0.0
        self._error_variance: float = 1.0

    def update(self, residual: float) -> float:
        # Kalman predict step
        predicted_variance = self._error_variance + self.process_noise
        # Kalman update step
        kalman_gain = predicted_variance / (predicted_variance + self.obs_noise)
        self._bias_estimate = self._bias_estimate + kalman_gain * (residual - self._bias_estimate)
        self._error_variance = (1 - kalman_gain) * predicted_variance
        return self._bias_estimate

    def correct(self, model_pred: float) -> float:
        return model_pred - self._bias_estimate

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "process_noise": self.process_noise,
            "obs_noise": self.obs_noise,
            "bias_estimate": self._bias_estimate,
            "error_variance": self._error_variance,
        }
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "KalmanBiasCorrector":
        with open(path) as f:
            state = json.load(f)
        corrector = cls(
            process_noise=state["process_noise"],
            obs_noise=state["obs_noise"],
        )
        corrector._bias_estimate = state["bias_estimate"]
        corrector._error_variance = state["error_variance"]
        return corrector


class BiasCorrectionRegistry:
    def __init__(
        self,
        persist_dir: str = "data/bias_correctors",
        process_noise: float = 0.1,
        obs_noise: float = 1.5,
    ) -> None:
        self._persist_dir = Path(persist_dir)
        self._process_noise = process_noise
        self._obs_noise = obs_noise
        self._correctors: dict[tuple[str, str, str], KalmanBiasCorrector] = {}

    def _key(self, station: str, lead_bucket: str, season: str) -> tuple[str, str, str]:
        return (station, lead_bucket, season)

    def get_corrector(self, station: str, lead_bucket: str, season: str) -> KalmanBiasCorrector:
        key = self._key(station, lead_bucket, season)
        if key not in self._correctors:
            path = self._persist_dir / f"{station}_{lead_bucket}_{season}.json"
            if path.exists():
                try:
                    self._correctors[key] = KalmanBiasCorrector.load(str(path))
                    return self._correctors[key]
                except Exception as exc:
                    logger.warning("Failed to load corrector %s: %s; creating fresh", path, exc)
            self._correctors[key] = KalmanBiasCorrector(
                process_noise=self._process_noise,
                obs_noise=self._obs_noise,
            )
        return self._correctors[key]

    def update_all(self, new_obs: dict) -> None:
        for (station, lead_bucket, season), residual in new_obs.items():
            corrector = self.get_corrector(station, lead_bucket, season)
            corrector.update(residual)

    def initialize_from_history(
        self,
        features_df,
        window_days: int = 60,
    ) -> None:
        """
        Seed each Kalman corrector from historical residuals (model_proxy - actual_tmax).
        Feeds the last `window_days` of residuals in chronological order so the
        corrector state reflects recent model bias rather than starting cold at 0.
        """
        import pandas as pd

        if features_df.empty or "actual_tmax" not in features_df.columns:
            logger.warning("Cannot initialize bias correctors: missing features or actual_tmax")
            return

        cutoff = features_df["date"].max() - pd.Timedelta(days=window_days)
        recent = features_df[features_df["date"] >= cutoff].sort_values("date")

        count = 0
        for _, row in recent.iterrows():
            station = row["station"]
            lead_hour = int(row["lead_hour"])
            actual = row["actual_tmax"]
            model_proxy = row.get("gefs_tmax_mean", float("nan"))

            import math
            if math.isnan(actual) or math.isnan(model_proxy):
                continue

            residual = model_proxy - actual  # positive = model runs warm
            month = row["date"].month if hasattr(row["date"], "month") else pd.Timestamp(row["date"]).month
            season = get_season(month)
            lead_bucket = get_lead_bucket(lead_hour)
            corrector = self.get_corrector(station, lead_bucket, season)
            corrector.update(residual)
            count += 1

        logger.info(
            "Initialized %d bias correctors from %d recent residuals (last %d days)",
            len(self._correctors), count, window_days,
        )

    def persist(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        for (station, lead_bucket, season), corrector in self._correctors.items():
            path = self._persist_dir / f"{station}_{lead_bucket}_{season}.json"
            corrector.save(str(path))
        logger.info("Persisted %d bias correctors", len(self._correctors))
