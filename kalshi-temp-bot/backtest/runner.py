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
from models.qrf_model import QRFTemperatureModel
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
        mu_tr, _ = ngb.predict_distribution(X_train)
        prob_train = ngb.predict_prob_above(X_train, threshold)
        outcomes_train = (y_train > threshold).astype(float).values
        calibrator.fit(prob_train, outcomes_train)
        cal_probs = np.array([calibrator.calibrate(p)[0] for p in prob_forecasts])

        metrics = track_a_metrics(
            prob_forecasts=cal_probs,
            mu_forecasts=mu_test,
            sigma_forecasts=sigma_test,
            observations=y_test.values,
            outcomes=outcomes,
        )

        market_mids = np.random.uniform(0.2, 0.8, size=len(cal_probs))
        sim = simulate_pnl(
            model_probs=cal_probs,
            market_mids=market_mids,
            outcomes=outcomes,
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
