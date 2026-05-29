"""
One-time script: train all models from scratch after bootstrap_history.py.

Usage: python scripts/initial_train.py
"""
import logging
from datetime import date
from dateutil.relativedelta import relativedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATIONS = ["KORD", "KJFK", "KLAX"]
HIST_DIR = Path("data/historical")


def load_feature_data() -> pd.DataFrame:
    feat_path = HIST_DIR / "features.parquet"
    if feat_path.exists():
        return pd.read_parquet(feat_path)
    logger.warning("features.parquet not found — using empty DataFrame")
    return pd.DataFrame()


def main() -> None:
    from config.settings import get_settings
    from db.session import init_db
    settings = get_settings()
    init_db(settings.DB_URL)

    logger.info("Loading historical feature data...")
    df = load_feature_data()

    if df.empty:
        logger.error("No historical data found. Run scripts/bootstrap_history.py first.")
        return

    # Step 2: Run backtest to get validation metrics and calibration datasets
    end_date = date.today()
    start_date = end_date - relativedelta(years=4)
    logger.info("Running walk-forward backtest %s → %s", start_date, end_date)

    from backtest.runner import BacktestRunner
    runner = BacktestRunner(settings=settings, start_date=start_date, end_date=end_date)
    report = runner.run()
    summary = report.summary()
    logger.info("Backtest summary: %s", summary)

    report.to_csv("data/backtest_results.csv")
    report.to_html("data/backtest_report.html")

    # Step 3: Train final models on ALL available data
    from processing.features import get_feature_columns
    from models.ngboost_model import NGBoostTemperatureModel
    from models.qrf_model import QRFTemperatureModel
    from models.residual_model import ResidualModel
    from models.calibration import IsotonicCalibrator
    from models.blend import ModelBlender
    from models.registry import save_artifact

    feature_cols = get_feature_columns()
    target_col = "actual_tmax"

    if target_col not in df.columns:
        logger.error("No target column '%s' in feature data", target_col)
        return

    avail_cols = [c for c in feature_cols if c in df.columns]
    X_all = df[avail_cols].fillna(0.0)
    y_all = df[target_col]

    blender = ModelBlender()
    station_groups = df.groupby("station") if "station" in df.columns else [("ALL", df)]

    for station, station_df in station_groups:
        logger.info("Training models for station: %s", station)
        X_st = station_df[avail_cols].fillna(0.0)
        y_st = station_df[target_col]

        if len(X_st) < 100:
            logger.warning("Insufficient data for %s (%d rows)", station, len(X_st))
            continue

        # NGBoost
        ngb = NGBoostTemperatureModel(n_estimators=500, learning_rate=0.01)
        ngb.fit(X_st, y_st)
        ngb_score = ngb.log_score(X_st, y_st)
        save_artifact(ngb, "ngboost", station, crps_val=abs(ngb_score))
        logger.info("NGBoost fitted for %s (log_score=%.4f)", station, ngb_score)

        # QRF
        qrf = QRFTemperatureModel(n_estimators=500, min_samples_leaf=20)
        qrf.fit(X_st, y_st)
        qrf_score = qrf.log_score(X_st, y_st)
        save_artifact(qrf, "qrf", station, crps_val=abs(qrf_score))
        logger.info("QRF fitted for %s (crps=%.4f)", station, qrf_score)

        blender.compute_weights_from_log_scores(ngb_score, qrf_score)

        # Residual model
        residual_features = [c for c in [
            "obs_minus_model_lag1", "obs_minus_model_lag2", "obs_minus_model_lag3",
            "lead_time_hours", "month_sin", "month_cos",
        ] + [f"regime_cluster_{i}" for i in range(12)] if c in X_st.columns]

        mu_pred, _ = ngb.predict_distribution(X_st)
        residuals = y_st.values - mu_pred
        if residual_features:
            res_model = ResidualModel(station=station)
            res_model.fit(X_st[residual_features], pd.Series(residuals, index=X_st.index))
            save_artifact(res_model, "residual", station)

        # Calibration per lead bucket
        for lead_bucket in ["D1-2", "D3-4", "D5-7"]:
            if "lead_time_hours" in station_df.columns:
                if lead_bucket == "D1-2":
                    mask = station_df["lead_time_hours"] <= 48
                elif lead_bucket == "D3-4":
                    mask = (station_df["lead_time_hours"] > 48) & (station_df["lead_time_hours"] <= 96)
                else:
                    mask = station_df["lead_time_hours"] > 96

                sub_df = station_df[mask]
                if len(sub_df) < 50:
                    continue

                X_sub = sub_df[avail_cols].fillna(0.0)
                y_sub = sub_df[target_col]
                threshold = float(y_sub.mean())
                raw_probs = ngb.predict_prob_above(X_sub, threshold)
                outcomes = (y_sub > threshold).astype(float).values

                cal = IsotonicCalibrator()
                cal.fit(raw_probs, outcomes)
                cal_path = f"data/calibrators/{station}_{lead_bucket}.pkl"
                cal.save(cal_path)
                logger.info("Calibrator saved for %s/%s", station, lead_bucket)

    logger.info("Blend weights: %s", blender.weights)
    logger.info("=== Initial training complete! ===")
    logger.info("Backtest report: data/backtest_report.html")


if __name__ == "__main__":
    main()
