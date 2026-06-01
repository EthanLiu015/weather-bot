import logging
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

from backtest.leakage_audit import audit_no_leakage
from backtest.track_a import track_a_metrics
from backtest.track_b import simulate_pnl
from backtest.report import BacktestReport, FoldResult
from models.ngboost_model import NGBoostTemperatureModel
from models.calibration import IsotonicCalibrator
from processing.features import get_feature_columns

logger = logging.getLogger(__name__)


class BacktestRunner:
    def __init__(
        self,
        settings,
        start_date: date,
        end_date: date,
        train_window_years: int = 3,
    ) -> None:
        self._settings = settings
        self._start = start_date
        self._end = end_date
        self._train_window_years = train_window_years
        self._kalshi_prices = self._load_kalshi_prices()

    def run(self) -> BacktestReport:
        report = BacktestReport()
        test_month = self._start + relativedelta(years=self._train_window_years)

        while test_month <= self._end:
            train_start = test_month - relativedelta(years=self._train_window_years)
            train_end = test_month - timedelta(days=1)
            logger.info("Running fold: train %s → %s, test month %s", train_start, train_end, test_month)

            try:
                fold = self._run_fold(train_start, train_end, test_month)
                report.folds.append(fold)
            except Exception as exc:
                logger.error("Fold %s failed: %s", test_month, exc)

            test_month += relativedelta(months=1)

        logger.info("Backtest complete: %d folds", len(report.folds))
        return report

    def _run_fold(self, train_start: date, train_end: date, test_month: date) -> FoldResult:
        feature_cols = get_feature_columns()

        train_df = self._load_historical_features(train_start, train_end)
        test_df = self._load_historical_features(test_month, test_month + relativedelta(months=1) - timedelta(days=1))

        if train_df.empty or test_df.empty:
            logger.warning("Empty data for fold %s — returning dummy metrics", test_month)
            return FoldResult(
                fold_month=test_month,
                crps=float("nan"),
                mae=float("nan"),
                brier_score=float("nan"),
                reliability_slope=float("nan"),
                simulated_pnl_usd=0.0,
                num_simulated_trades=0,
                edge_above_threshold_pct=0.0,
            )

        ok, issues = audit_no_leakage(train_df, test_df, date_col="date", train_end=train_end)
        if not ok:
            raise ValueError(f"Leakage detected in fold {test_month}: {issues}")

        avail_cols = [c for c in feature_cols if c in train_df.columns]
        target_col = "actual_tmax"

        if target_col not in train_df.columns:
            logger.warning("No target column in fold %s — skipping model training", test_month)
            return FoldResult(
                fold_month=test_month,
                crps=float("nan"), mae=float("nan"), brier_score=float("nan"),
                reliability_slope=float("nan"), simulated_pnl_usd=0.0,
                num_simulated_trades=0, edge_above_threshold_pct=0.0,
            )

        X_train = train_df[avail_cols].fillna(0.0)
        y_train = train_df[target_col]
        X_test = test_df[avail_cols].fillna(0.0)
        y_test = test_df[target_col]

        ngb = NGBoostTemperatureModel(n_estimators=200, learning_rate=0.05)
        ngb.fit(X_train, y_train)
        mu_test, sigma_test = ngb.predict_distribution(X_test)

        threshold = float(y_train.mean())
        prob_forecasts = ngb.predict_prob_above(X_test, threshold)
        outcomes = (y_test > threshold).astype(float).values

        calibrator = IsotonicCalibrator()
        prob_train = ngb.predict_prob_above(X_train, threshold)
        outcomes_train = (y_train > threshold).astype(float).values
        calibrator.fit(prob_train, outcomes_train)
        if calibrator._iso is not None:
            cal_probs = calibrator._iso.predict(prob_forecasts)
        else:
            cal_probs = prob_forecasts

        metrics = track_a_metrics(
            prob_forecasts=cal_probs,
            mu_forecasts=mu_test,
            sigma_forecasts=sigma_test,
            observations=y_test.values,
            outcomes=outcomes,
        )

        # Simulate trades at D+1 only — one decision per (station, date) market
        if "lead_hour" in test_df.columns:
            d1_mask = test_df["lead_hour"] == 24
            d1_test = test_df[d1_mask].copy()
            trade_probs = cal_probs[d1_mask]
            trade_outcomes = outcomes[d1_mask]
        else:
            d1_test = test_df.copy()
            trade_probs = cal_probs
            trade_outcomes = outcomes

        # Market mid = climatological P(Tmax > threshold | station, month) from training data.
        # An efficient market prices close to historical frequency; our edge comes from
        # deviating from this when model has skill. Much more realistic than random uniform.
        market_mids = self._climatological_mids(
            train_df=train_df,
            test_df=d1_test,
            threshold=threshold,
            target_col=target_col,
        )

        sim = simulate_pnl(
            model_probs=trade_probs,
            market_mids=market_mids,
            outcomes=trade_outcomes,
            min_edge=self._settings.MIN_EDGE_CENTS / 100.0,
        )

        return FoldResult(
            fold_month=test_month,
            crps=metrics["crps"],
            mae=metrics["mae"],
            brier_score=metrics["brier_score"],
            reliability_slope=metrics["reliability_slope"],
            simulated_pnl_usd=sim["simulated_pnl_usd"],
            num_simulated_trades=sim["num_simulated_trades"],
            edge_above_threshold_pct=sim["edge_above_threshold_pct"],
        )

    @staticmethod
    def _load_kalshi_prices() -> pd.DataFrame:
        path = "data/historical/kalshi_prices.parquet"
        try:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            logger.info("Loaded %d real Kalshi market prices (%s → %s)",
                        len(df), df["date"].min(), df["date"].max())
            return df
        except FileNotFoundError:
            logger.info("No kalshi_prices.parquet found — using climatological mids")
            return pd.DataFrame()

    def _get_market_mid(self, station: str, date_: date, threshold: float) -> float | None:
        """Return real Kalshi D+1 mid for (station, date, threshold) if available."""
        if self._kalshi_prices.empty:
            return None
        mask = (
            (self._kalshi_prices["station"] == station) &
            (self._kalshi_prices["date"] == date_) &
            (self._kalshi_prices["market_type"] == "above")
        )
        candidates = self._kalshi_prices[mask]
        if candidates.empty:
            return None
        # Pick the contract whose threshold is closest to ours
        closest = candidates.iloc[(candidates["threshold"] - threshold).abs().argmin()]
        mid = closest["d1_mid"]
        if pd.isna(mid) or mid <= 0 or mid >= 1:
            return None
        return float(mid)

    def _climatological_mids(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        threshold: float,
        target_col: str,
        noise_std: float = 0.02,
    ) -> np.ndarray:
        """
        Compute market mid for each test row as P(Tmax > threshold | station, month)
        estimated from the training set. This is the efficient-market baseline price —
        a rational market participant with only historical frequency data would price here.

        Small Gaussian noise (std=2¢) simulates bid-ask spread and market irrationality.
        """
        rng = np.random.default_rng(42)
        mids = np.full(len(test_df), 0.5)
        real_price_count = 0

        train_df = train_df.copy()
        train_df["_month"] = pd.to_datetime(train_df["date"]).dt.month

        for i, (_, row) in enumerate(test_df.iterrows()):
            station = row.get("station")
            row_date = row["date"] if isinstance(row["date"], date) else pd.to_datetime(row["date"]).date()
            month = pd.to_datetime(row["date"]).month

            # Prefer real Kalshi market prices when available
            real_mid = self._get_market_mid(station, row_date, threshold)
            if real_mid is not None:
                mids[i] = real_mid
                real_price_count += 1
                continue

            # Fall back to climatological probability
            hist = train_df[
                (train_df["station"] == station) &
                (train_df["_month"] == month)
            ][target_col].dropna()

            if len(hist) >= 10:
                clim_prob = float((hist > threshold).mean())
                # Clip to tradeable range — Kalshi markets rarely go below 5¢ or above 95¢
                clim_prob = float(np.clip(clim_prob, 0.05, 0.95))
            else:
                clim_prob = 0.5

            noise = rng.normal(0, noise_std)
            mids[i] = float(np.clip(clim_prob + noise, 0.02, 0.98))

        if real_price_count > 0:
            logger.debug("Market mids: %d real Kalshi prices, %d climatological",
                         real_price_count, len(test_df) - real_price_count)
        return mids

    def _load_historical_features(self, start: date, end: date) -> pd.DataFrame:
        hist_path = "data/historical/features.parquet"
        try:
            df = pd.read_parquet(hist_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df[(df["date"] >= start) & (df["date"] <= end)].copy()
            return df
        except FileNotFoundError:
            logger.warning("Historical features not found at %s — returning empty", hist_path)
            return pd.DataFrame()
        except Exception as exc:
            logger.error("Error loading historical features: %s", exc)
            return pd.DataFrame()
