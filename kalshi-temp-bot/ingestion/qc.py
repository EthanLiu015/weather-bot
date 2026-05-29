import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TEMP_RANGE_F = (-60.0, 130.0)
WIND_SPEED_RANGE_KT = (0.0, 200.0)
DEWPOINT_SPREAD_MAX_F = 60.0


def validate_metar_obs(obs: dict) -> tuple[bool, list[str]]:
    issues: list[str] = []

    temp_f = obs.get("temp_f")
    if temp_f is not None:
        if not (TEMP_RANGE_F[0] <= temp_f <= TEMP_RANGE_F[1]):
            issues.append(f"temp_f {temp_f} out of range {TEMP_RANGE_F}")

    dewpoint_f = obs.get("dewpoint_f")
    if temp_f is not None and dewpoint_f is not None:
        if dewpoint_f > temp_f:
            issues.append(f"dewpoint {dewpoint_f} > temp {temp_f}")
        if temp_f - dewpoint_f > DEWPOINT_SPREAD_MAX_F:
            issues.append(f"dewpoint spread {temp_f - dewpoint_f:.1f}°F exceeds {DEWPOINT_SPREAD_MAX_F}")

    wind_kt = obs.get("wind_speed_kt")
    if wind_kt is not None:
        if not (WIND_SPEED_RANGE_KT[0] <= wind_kt <= WIND_SPEED_RANGE_KT[1]):
            issues.append(f"wind_speed_kt {wind_kt} out of range")

    return len(issues) == 0, issues


def qc_metar_list(obs_list: list[dict]) -> list[dict]:
    clean = []
    for obs in obs_list:
        valid, issues = validate_metar_obs(obs)
        if valid:
            clean.append(obs)
        else:
            logger.warning("QC rejected METAR obs %s: %s", obs.get("observation_time"), issues)
    return clean


def validate_gefs_member(df: pd.DataFrame) -> tuple[bool, list[str]]:
    issues: list[str] = []

    if df.empty:
        issues.append("empty dataframe")
        return False, issues

    temp_cols = [c for c in df.columns if "t2m" in c.lower() or "tmp" in c.lower() or "temperature" in c.lower()]
    for col in temp_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            issues.append(f"all NaN in column {col}")
            continue
        if vals.min() < 200 or vals.max() > 330:
            issues.append(f"{col} values out of Kelvin range: min={vals.min():.1f}, max={vals.max():.1f}")

    nan_frac = df.isna().mean().mean()
    if nan_frac > 0.5:
        issues.append(f"NaN fraction {nan_frac:.2f} > 0.5")

    return len(issues) == 0, issues


def qc_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    initial_len = len(df)
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            logger.warning("Expected feature column %s missing; filling with 0", col)
            df[col] = 0.0

    high_nan_cols = df[feature_cols].columns[df[feature_cols].isna().mean() > 0.8].tolist()
    if high_nan_cols:
        logger.warning("Columns with >80%% NaN: %s", high_nan_cols)

    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)

    logger.info("QC: kept %d/%d rows", len(df), initial_len)
    return df
