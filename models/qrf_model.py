import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from quantile_forest import RandomForestQuantileRegressor
import properscoring as ps

logger = logging.getLogger(__name__)

DEFAULT_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


class QRFTemperatureModel:
    def __init__(self, n_estimators: int = 500, min_samples_leaf: int = 20) -> None:
        self._n_estimators = n_estimators
        self._min_samples_leaf = min_samples_leaf
        self._model: RandomForestQuantileRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._model = RandomForestQuantileRegressor(
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X.values, y.values)
        logger.info("QRF fitted on %d samples with %d estimators", len(X), self._n_estimators)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: list[float] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        preds = self._model.predict(X.values, quantiles=quantiles)
        cols = [f"q{int(q*100)}" for q in quantiles]
        return pd.DataFrame(preds, columns=cols, index=X.index)

    def predict_prob_above(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        q_df = self.predict_quantiles(X)
        quantiles = DEFAULT_QUANTILES
        probs = np.zeros(len(X))
        for i in range(len(X)):
            row_vals = q_df.iloc[i].values
            try:
                interp = interp1d(
                    row_vals,
                    quantiles,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(0.0, 1.0),
                )
                cdf_val = float(interp(threshold))
                probs[i] = 1.0 - cdf_val
            except Exception:
                probs[i] = 0.5
        return probs

    def predict_full_cdf(self, X: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
        result = {}
        for t in thresholds:
            result[f"cdf_{t}"] = self.predict_prob_above(X, t)
        return pd.DataFrame(result, index=X.index)

    def log_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        q_df = self.predict_quantiles(X)
        median_pred = q_df["q50"].values
        crps_scores = ps.crps_ensemble(y.values, q_df.values)
        return float(np.mean(crps_scores))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("QRF model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "QRFTemperatureModel":
        instance = cls()
        with open(path, "rb") as f:
            instance._model = pickle.load(f)
        return instance
