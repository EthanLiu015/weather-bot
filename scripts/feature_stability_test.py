"""
Parameter stability test for the feature matrix.

For each feature in processing.features.get_feature_columns(), computes the
Pearson correlation with actual_tmax separately for each calendar year, plus
the per-year mean/std of the feature itself. A feature whose correlation with
the target swings widely (or flips sign) across years indicates an unstable
feature-target relationship — evidence that an expanding training window may
dilute a regime-dependent signal, or that the feature is mostly noise.

Usage: PYTHONPATH=. python scripts/feature_stability_test.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HIST_DIR = Path("data/historical")
REPORT_PATH = Path("data/feature_stability_report.csv")


def compute_stability_report(df: pd.DataFrame, feature_cols: list[str], target_col: str = "actual_tmax") -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    years = sorted(df["year"].unique())

    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue

        corr_by_year = {}
        mean_by_year = {}
        std_by_year = {}
        for year in years:
            sub = df.loc[df["year"] == year, [col, target_col]].dropna()
            if len(sub) < 30 or sub[col].std() == 0:
                corr_by_year[year] = float("nan")
            else:
                corr_by_year[year] = float(sub[col].corr(sub[target_col]))
            mean_by_year[year] = float(sub[col].mean()) if len(sub) else float("nan")
            std_by_year[year] = float(sub[col].std()) if len(sub) else float("nan")

        corr_values = np.array([v for v in corr_by_year.values() if not np.isnan(v)])
        mean_values = np.array([v for v in mean_by_year.values() if not np.isnan(v)])

        overall_std = df[col].std()
        row = {
            "feature": col,
            "corr_mean": float(np.mean(corr_values)) if len(corr_values) else float("nan"),
            "corr_std": float(np.std(corr_values)) if len(corr_values) else float("nan"),
            "corr_range": float(np.max(corr_values) - np.min(corr_values)) if len(corr_values) else float("nan"),
            "sign_flip": bool(len(corr_values) and (corr_values.max() > 0) and (corr_values.min() < 0)),
            "mean_drift_pct_of_std": (
                float((np.max(mean_values) - np.min(mean_values)) / overall_std)
                if len(mean_values) and overall_std > 0 else float("nan")
            ),
        }
        for year in years:
            row[f"corr_{year}"] = corr_by_year[year]
        rows.append(row)

    report = pd.DataFrame(rows).sort_values("corr_std", ascending=False, na_position="last")
    return report.reset_index(drop=True)


def main() -> None:
    from processing.features import get_feature_columns

    feat_path = HIST_DIR / "features.parquet"
    df = pd.read_parquet(feat_path)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded %s: %d rows, %d stations", feat_path, len(df), df["station"].nunique())

    feature_cols = get_feature_columns()
    report = compute_stability_report(df, feature_cols)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    logger.info("Stability report written to %s", REPORT_PATH)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    constant = report[report["corr_std"].isna()]
    scored = report[report["corr_std"].notna()].sort_values("corr_std", ascending=False)

    logger.info("=== Zero-variance features (no signal in any year, %d) ===", len(constant))
    print(constant["feature"].to_string(index=False))

    logger.info("=== Most unstable features (corr with actual_tmax varies most across years) ===")
    print(scored.head(10)[["feature", "corr_mean", "corr_std", "corr_range", "sign_flip"]].to_string(index=False))

    logger.info("=== Most stable features ===")
    print(scored.tail(10)[["feature", "corr_mean", "corr_std", "corr_range", "sign_flip"]].to_string(index=False))

    # Sign flips are only meaningful where the correlation is non-trivial in
    # at least one year — near-zero correlations flip sign on pure noise.
    sign_flips = scored[scored["sign_flip"] & (scored["corr_range"] > 0.1)]
    logger.info("=== Meaningful sign flips across years (|corr_range| > 0.1, %d) ===", len(sign_flips))
    print(sign_flips[["feature", "corr_mean", "corr_std", "corr_range"]].to_string(index=False))

    high_drift = report[report["mean_drift_pct_of_std"] > 0.5].sort_values("mean_drift_pct_of_std", ascending=False)
    logger.info("=== Features with large year-to-year mean drift (> 0.5 std, %d) ===", len(high_drift))
    print(high_drift[["feature", "mean_drift_pct_of_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
