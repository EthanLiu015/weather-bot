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
    def __init__(self, persist_dir: str = "data/bias_correctors") -> None:
        self._persist_dir = Path(persist_dir)
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
            self._correctors[key] = KalmanBiasCorrector()
        return self._correctors[key]

    def update_all(self, new_obs: dict) -> None:
        for (station, lead_bucket, season), residual in new_obs.items():
            corrector = self.get_corrector(station, lead_bucket, season)
            corrector.update(residual)

    def persist(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        for (station, lead_bucket, season), corrector in self._correctors.items():
            path = self._persist_dir / f"{station}_{lead_bucket}_{season}.json"
            corrector.save(str(path))
        logger.info("Persisted %d bias correctors", len(self._correctors))
