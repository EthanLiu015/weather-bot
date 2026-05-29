import pickle
import logging
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

MIN_SAMPLES = 100


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None
        self._raw_probs: np.ndarray | None = None
        self._outcomes: np.ndarray | None = None
        self._reliability_slope_val: float | None = None

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> None:
        self._raw_probs = raw_probs
        self._outcomes = outcomes
        if len(raw_probs) < MIN_SAMPLES:
            logger.warning("Only %d samples — below minimum %d; calibrator will use raw output", len(raw_probs), MIN_SAMPLES)
            self._iso = None
            return
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw_probs, outcomes)

        calibrated = self._iso.predict(raw_probs)
        reg = LinearRegression().fit(raw_probs.reshape(-1, 1), calibrated)
        self._reliability_slope_val = float(reg.coef_[0])
        logger.info("Calibrator fitted; reliability slope=%.3f", self._reliability_slope_val)

    def calibrate(self, raw_prob: float) -> tuple[float, float, float]:
        if self._iso is None:
            return raw_prob, max(0.0, raw_prob - 0.1), min(1.0, raw_prob + 0.1)
        cal = float(self._iso.predict([raw_prob])[0])
        ci_lo, ci_hi = self.bootstrap_ci(raw_prob)
        return cal, ci_lo, ci_hi

    def bootstrap_ci(self, raw_prob: float, n_bootstrap: int = 1000) -> tuple[float, float]:
        if self._raw_probs is None or self._outcomes is None or len(self._raw_probs) < MIN_SAMPLES:
            return max(0.0, raw_prob - 0.1), min(1.0, raw_prob + 0.1)
        rng = np.random.default_rng(42)
        n = len(self._raw_probs)
        boot_preds = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            rp_b = self._raw_probs[idx]
            oc_b = self._outcomes[idx]
            try:
                iso_b = IsotonicRegression(out_of_bounds="clip")
                iso_b.fit(rp_b, oc_b)
                pred = float(iso_b.predict([raw_prob])[0])
                boot_preds.append(pred)
            except Exception:
                pass
        if not boot_preds:
            return max(0.0, raw_prob - 0.1), min(1.0, raw_prob + 0.1)
        return float(np.percentile(boot_preds, 5)), float(np.percentile(boot_preds, 95))

    def reliability_slope(self) -> float:
        return self._reliability_slope_val if self._reliability_slope_val is not None else float("nan")

    def brier_score(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> float:
        if self._iso is not None:
            cal_probs = self._iso.predict(raw_probs)
        else:
            cal_probs = raw_probs
        return float(brier_score_loss(outcomes, cal_probs))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "iso": self._iso,
                "raw_probs": self._raw_probs,
                "outcomes": self._outcomes,
                "reliability_slope": self._reliability_slope_val,
            }, f)

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance._iso = data["iso"]
        instance._raw_probs = data["raw_probs"]
        instance._outcomes = data["outcomes"]
        instance._reliability_slope_val = data.get("reliability_slope")
        return instance
