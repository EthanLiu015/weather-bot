import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal

logger = logging.getLogger(__name__)


class NGBoostTemperatureModel:
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        natural_gradient: bool = True,
    ) -> None:
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._natural_gradient = natural_gradient
        self._model: NGBRegressor | None = None

    def _build_model(self) -> NGBRegressor:
        base = DecisionTreeRegressor(max_depth=4)
        return NGBRegressor(
            Dist=Normal,
            Base=base,
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            natural_gradient=self._natural_gradient,
            verbose=False,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        split = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]
        self._model = self._build_model()
        self._model.fit(
            X_tr.values,
            y_tr.values,
            X_val=X_val.values,
            Y_val=y_val.values,
            early_stopping_rounds=20,
        )
        logger.info("NGBoost fitted on %d samples, validated on %d", len(X_tr), len(X_val))

    def predict_distribution(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        dist = self._model.pred_dist(X.values)
        mu = dist.loc
        sigma = dist.scale
        return mu, sigma

    def predict_prob_above(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        mu, sigma = self.predict_distribution(X)
        return 1.0 - stats.norm.cdf(threshold, loc=mu, scale=sigma)

    def predict_full_cdf(self, X: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
        mu, sigma = self.predict_distribution(X)
        result = {}
        for t in thresholds:
            result[f"cdf_{t}"] = 1.0 - stats.norm.cdf(t, loc=mu, scale=sigma)
        return pd.DataFrame(result, index=X.index)

    def log_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        scores = self._model.score(X.values, y.values)
        return float(np.mean(scores))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("NGBoost model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "NGBoostTemperatureModel":
        instance = cls()
        with open(path, "rb") as f:
            instance._model = pickle.load(f)
        return instance
