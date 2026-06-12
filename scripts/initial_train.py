"""
One-time script: train all models from scratch.

Run order:
  1. python scripts/bootstrap_history.py
  2. python scripts/build_feature_matrix.py    ← assembles features.parquet
  3. python scripts/initial_train.py           ← this script

Usage: PYTHONPATH=. python scripts/initial_train.py
"""
import logging
import sys
from datetime import date
from dateutil.relativedelta import relativedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HIST_DIR = Path("data/historical")
CALIBRATOR_DIR = Path("data/calibrators")


def load_feature_data() -> pd.DataFrame:
    feat_path = HIST_DIR / "features.parquet"
    if not feat_path.exists():
        logger.error(
            "features.parquet not found.\n"
            "Run: PYTHONPATH=. python scripts/build_feature_matrix.py"
        )
        return pd.DataFrame()
    df = pd.read_parquet(feat_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded features.parquet: %d rows, %d stations", len(df), df["station"].nunique())
    return df


def train_final_models(df: pd.DataFrame) -> None:
    """Step 3: train final per-station models on ALL data and persist a global
    blend-weight artifact.

    Split out from main() so it can be re-run on its own (e.g. to recompute
    blend weights) without repeating the ~1.5hr walk-forward backtest.
    """
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
        logger.error("No 'actual_tmax' column in feature data")
        return

    logger.info("=== Training final models ===")
    CALIBRATOR_DIR.mkdir(parents=True, exist_ok=True)

    blender = ModelBlender()
    ngb_scores: dict[str, float] = {}
    qrf_scores: dict[str, float] = {}
    ngb_crps_oos: dict[str, float] = {}
    qrf_crps_oos: dict[str, float] = {}

    stations_in_data = df["station"].unique().tolist()
    logger.info("Stations with data: %s", stations_in_data)

    for station in stations_in_data:
        station_df = df[df["station"] == station].copy()
        avail_cols = [c for c in feature_cols if c in station_df.columns]
        X_st = station_df[avail_cols].fillna(0.0)
        y_st = station_df[target_col]

        if len(X_st) < 200:
            logger.warning("Skipping %s — only %d rows", station, len(X_st))
            continue

        logger.info("--- %s: %d rows ---", station, len(X_st))

        # NGBoost
        ngb = NGBoostTemperatureModel(n_estimators=500, learning_rate=0.01)
        ngb.fit(X_st, y_st)
        ngb_score = ngb.log_score(X_st, y_st)
        ngb_scores[station] = ngb_score
        save_artifact(ngb, "ngboost", station, crps_val=abs(ngb_score))
        logger.info("NGBoost %s log_score=%.4f", station, ngb_score)

        # QRF
        qrf = QRFTemperatureModel(n_estimators=500, min_samples_leaf=20)
        qrf.fit(X_st, y_st)
        qrf_score = qrf.log_score(X_st, y_st)
        qrf_scores[station] = qrf_score
        save_artifact(qrf, "qrf", station, crps_val=abs(qrf_score))
        logger.info("QRF %s crps=%.4f", station, qrf_score)

        # Held-out CRPS for both models, used to set the production blend weights.
        # NGBoost's log_score (NLL) and QRF's log_score (CRPS) are different units
        # and computed in-sample above, so they aren't comparable for blending —
        # use a dedicated 85/15 split with the same CRPS metric for both.
        val_n = max(30, int(len(X_st) * 0.15))
        X_tr_bw, X_val_bw = X_st.iloc[:-val_n], X_st.iloc[-val_n:]
        y_tr_bw, y_val_bw = y_st.iloc[:-val_n], y_st.iloc[-val_n:]

        ngb_bw = NGBoostTemperatureModel(n_estimators=500, learning_rate=0.01)
        ngb_bw.fit(X_tr_bw, y_tr_bw)
        ngb_crps_oos[station] = ngb_bw.crps(X_val_bw, y_val_bw)

        qrf_bw = QRFTemperatureModel(n_estimators=500, min_samples_leaf=20)
        qrf_bw.fit(X_tr_bw, y_tr_bw)
        qrf_crps_oos[station] = qrf_bw.log_score(X_val_bw, y_val_bw)

        logger.info("Held-out CRPS %s: NGBoost=%.4f QRF=%.4f",
                     station, ngb_crps_oos[station], qrf_crps_oos[station])

        # Residual model (LightGBM)
        residual_feature_cols = [c for c in [
            "obs_minus_model_lag1", "obs_minus_model_lag2", "obs_minus_model_lag3",
            "obs_minus_model_roll_mean", "obs_minus_model_roll_std",
            "lead_time_hours", "day_of_year_sin", "day_of_year_cos",
        ] if c in X_st.columns]

        mu_pred, _ = ngb.predict_distribution(X_st)
        residuals = pd.Series(y_st.values - mu_pred, index=X_st.index)
        if residual_feature_cols and residuals.abs().mean() > 0.01:
            res_model = ResidualModel(station=station)
            res_model.fit(X_st[residual_feature_cols], residuals)
            save_artifact(res_model, "residual", station)
            logger.info("Residual model saved for %s (mean_residual=%.2f°F)", station, residuals.mean())

        # Isotonic calibration per lead bucket
        for lead_bucket, lh_min, lh_max in [("D1-2", 0, 48), ("D3-4", 49, 96), ("D5-7", 97, 999)]:
            mask = (station_df["lead_hour"] >= lh_min) & (station_df["lead_hour"] <= lh_max)
            sub = station_df[mask]
            if len(sub) < 100:
                logger.debug("Skipping calibrator %s/%s — only %d rows", station, lead_bucket, len(sub))
                continue

            X_sub = sub[avail_cols].fillna(0.0)
            y_sub = sub[target_col]
            threshold = float(y_sub.median())
            raw_probs = ngb.predict_prob_above(X_sub, threshold)
            outcomes = (y_sub > threshold).astype(float).values

            cal = IsotonicCalibrator()
            cal.fit(raw_probs, outcomes)
            cal_path = CALIBRATOR_DIR / f"{station}_{lead_bucket}.pkl"
            cal.save(str(cal_path))
            logger.info("Calibrator saved: %s/%s (n=%d, slope=%.3f)",
                        station, lead_bucket, len(sub), cal.reliability_slope())

    # Compute blend weights from held-out CRPS (lower = better; negate since
    # compute_weights_from_log_scores treats higher = better)
    if ngb_crps_oos and qrf_crps_oos:
        mean_ngb_crps = float(np.mean(list(ngb_crps_oos.values())))
        mean_qrf_crps = float(np.mean(list(qrf_crps_oos.values())))
        blender.compute_weights_from_log_scores(-mean_ngb_crps, -mean_qrf_crps)
        logger.info("Held-out mean CRPS: NGBoost=%.4f QRF=%.4f", mean_ngb_crps, mean_qrf_crps)
        logger.info("Final blend weights: %s", blender.weights)
        blender_path = save_artifact(blender, "blender", "global")
        logger.info("Blend weights saved to %s", blender_path)

    logger.info("=== Training complete ===")


def main() -> None:
    from config.settings import get_settings
    from db.session import init_db
    from processing.bias_correction import BiasCorrectionRegistry
    from backtest.runner import BacktestRunner

    settings = get_settings()
    init_db(settings.DB_URL)

    df = load_feature_data()
    if df.empty:
        return

    if "actual_tmax" not in df.columns:
        logger.error("No 'actual_tmax' column in feature data")
        return

    # -------------------------------------------------------------------------
    # Step 1: Seed Kalman bias correctors from history (last 60 days)
    # -------------------------------------------------------------------------
    logger.info("=== Seeding bias correctors from history ===")
    bias_registry = BiasCorrectionRegistry()
    bias_registry.initialize_from_history(df, window_days=60)
    bias_registry.persist()
    logger.info("Bias correctors saved to data/bias_correctors/")

    # -------------------------------------------------------------------------
    # Step 2: Walk-forward backtest (produces out-of-sample metrics + calibration data)
    # -------------------------------------------------------------------------
    end_date = date.today()
    start_date = end_date - relativedelta(years=4)
    logger.info("=== Walk-forward backtest %s → %s ===", start_date, end_date)

    runner = BacktestRunner(settings=settings, start_date=start_date, end_date=end_date)
    report = runner.run()
    summary = report.summary()

    if summary:
        logger.info("Backtest summary:")
        for k, v in summary.items():
            logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    report.to_csv("data/backtest_results.csv")
    report.to_html("data/backtest_report.html")
    logger.info("Backtest report: data/backtest_report.html")

    # -------------------------------------------------------------------------
    # Step 3: Train final models on ALL data, per station × lead bucket
    # -------------------------------------------------------------------------
    train_final_models(df)
    logger.info("Next step: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
