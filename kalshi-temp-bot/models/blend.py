import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelBlender:
    def __init__(self) -> None:
        self._ngboost_weight: float = 0.5
        self._qrf_weight: float = 0.5

    def compute_weights_from_log_scores(
        self,
        ngboost_log_score: float,
        qrf_log_score: float,
    ) -> None:
        # Softmax over log-scores — better (higher) log-score gets more weight
        scores = np.array([ngboost_log_score, qrf_log_score])
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum()
        self._ngboost_weight = float(weights[0])
        self._qrf_weight = float(weights[1])
        logger.info(
            "Blend weights: NGBoost=%.3f, QRF=%.3f",
            self._ngboost_weight,
            self._qrf_weight,
        )

    def blend_probs(
        self,
        ngboost_prob: np.ndarray,
        qrf_prob: np.ndarray,
    ) -> np.ndarray:
        return self._ngboost_weight * ngboost_prob + self._qrf_weight * qrf_prob

    def blend_mu_sigma(
        self,
        ngboost_mu: np.ndarray,
        ngboost_sigma: np.ndarray,
        qrf_mu: np.ndarray,
        qrf_sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        blended_mu = self._ngboost_weight * ngboost_mu + self._qrf_weight * qrf_mu
        # Variance of a mixture: E[var] + Var[mean]
        blended_var = (
            self._ngboost_weight * ngboost_sigma**2
            + self._qrf_weight * qrf_sigma**2
            + self._ngboost_weight * self._qrf_weight * (ngboost_mu - qrf_mu)**2
        )
        return blended_mu, np.sqrt(blended_var)

    @property
    def weights(self) -> dict[str, float]:
        return {"ngboost": self._ngboost_weight, "qrf": self._qrf_weight}
