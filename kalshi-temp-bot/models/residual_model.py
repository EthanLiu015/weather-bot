import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)


class ResidualModel:
    def __init__(self, station: str) -> None:
        self.station = station
        self._model: lgb.LGBMRegressor | None = None

    def fit(self, X_residual: pd.DataFrame, residuals: pd.Series) -> None:
        split = int(len(X_residual) * 0.8)
        X_tr, X_val = X_residual.iloc[:split], X_residual.iloc[split:]
        y_tr, y_val = residuals.iloc[:split], residuals.iloc[split:]

        self._model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False), lgb.log_evaluation(period=-1)],
        )
        logger.info("ResidualModel for %s fitted on %d samples", self.station, len(X_tr))

    def predict(self, X_residual: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict(X_residual)

    def feature_importance(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return pd.Series(
            self._model.feature_importances_,
            index=self._model.feature_name_,
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"station": self.station, "model": self._model}, f)
        logger.info("ResidualModel saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ResidualModel":
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(station=data["station"])
        instance._model = data["model"]
        return instance
