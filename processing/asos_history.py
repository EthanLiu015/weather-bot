import datetime as dt

import pandas as pd


def compute_daily_residuals(
    actual_tmax: dict[dt.date, float],
    forecast_tmax: dict[dt.date, float],
) -> dict[dt.date, float]:
    """obs_minus_model residual (actual - forecast) for dates present in both."""
    return {
        date: actual_tmax[date] - forecast_tmax[date]
        for date in actual_tmax
        if date in forecast_tmax
    }


def build_asos_history_df(residuals_by_station: dict[str, dict[dt.date, float]]) -> pd.DataFrame:
    """Column-per-station, DatetimeIndex-by-date DataFrame of obs_minus_model
    residuals, in the format build_feature_matrix expects for asos_history."""
    if not residuals_by_station:
        return pd.DataFrame()

    df = pd.DataFrame({
        station: pd.Series(residuals, dtype=float)
        for station, residuals in residuals_by_station.items()
    })
    df.index = pd.DatetimeIndex(df.index)
    return df
