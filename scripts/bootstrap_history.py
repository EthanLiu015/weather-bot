"""
One-time script: pull 5 years of ASOS + model history and build diurnal climatology.

Usage: python scripts/bootstrap_history.py
"""
import asyncio
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATIONS = ["KORD", "KJFK", "KLAX"]
HIST_DIR = Path("data/historical")
DIURNAL_DIR = Path("data/diurnal_climatology")
REGIME_DIR = Path("data/regime_clusters")
NOAA_CDO_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

STATION_IDS = {
    "KORD": "GHCND:USW00094846",
    "KJFK": "GHCND:USW00094789",
    "KLAX": "GHCND:USW00023174",
}


async def fetch_ghcnd_data(station_id: str, start: date, end: date, token: str) -> pd.DataFrame:
    all_rows = []
    current = start
    async with httpx.AsyncClient() as client:
        while current < end:
            chunk_end = min(current + timedelta(days=364), end)
            params = {
                "datasetid": "GHCND",
                "stationid": station_id,
                "startdate": current.isoformat(),
                "enddate": chunk_end.isoformat(),
                "datatypeid": "TMAX,TMIN,PRCP,AWND",
                "limit": 1000,
                "units": "standard",
            }
            try:
                resp = await client.get(
                    f"{NOAA_CDO_BASE}/data",
                    params=params,
                    headers={"token": token},
                    timeout=30.0,
                )
                resp.raise_for_status()
                data = resp.json().get("results", [])
                all_rows.extend(data)
                logger.info("Fetched %d rows for %s %s–%s", len(data), station_id, current, chunk_end)
            except Exception as exc:
                logger.warning("GHCND fetch failed: %s", exc)
            current = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.5)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


async def fetch_hourly_asos(station: str, start: date, end: date) -> pd.DataFrame:
    """Fetch hourly ASOS from Iowa State's mesonet archive."""
    rows = []
    current = start
    async with httpx.AsyncClient() as client:
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)
            url = (
                f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
                f"?station={station}&data=tmpf&year1={current.year}&month1={current.month}"
                f"&day1={current.day}&year2={chunk_end.year}&month2={chunk_end.month}"
                f"&day2={chunk_end.day}&tz=UTC&format=comma&latlon=no&direct=no"
            )
            try:
                resp = await client.get(url, timeout=30.0)
                resp.raise_for_status()
                lines = resp.text.strip().splitlines()
                for line in lines[2:]:
                    parts = line.split(",")
                    if len(parts) >= 3:
                        try:
                            dt = pd.to_datetime(parts[1].strip())
                            tmp = float(parts[2].strip())
                            rows.append({"datetime": dt, "tmpf": tmp, "station": station})
                        except (ValueError, IndexError):
                            pass
                logger.info("Fetched %d hourly obs for %s %s–%s", len(lines) - 2, station, current, chunk_end)
            except Exception as exc:
                logger.warning("Hourly ASOS fetch failed for %s %s: %s", station, current, exc)
            current = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.3)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["datetime", "tmpf", "station"])


def build_diurnal_climatology(hourly_df: pd.DataFrame, station: str) -> None:
    DIURNAL_DIR.mkdir(parents=True, exist_ok=True)
    if hourly_df.empty:
        logger.warning("No hourly data for %s — skipping diurnal climatology", station)
        return

    hourly_df = hourly_df.copy()
    hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"], utc=True)
    hourly_df["date"] = hourly_df["datetime"].dt.date
    hourly_df["hour"] = hourly_df["datetime"].dt.hour
    hourly_df["month"] = hourly_df["datetime"].dt.month

    daily_max = hourly_df.groupby("date")["tmpf"].max().reset_index().rename(columns={"tmpf": "daily_tmax"})
    hourly_df = hourly_df.merge(daily_max, on="date")
    hourly_df["tmax_minus_obs"] = hourly_df["daily_tmax"] - hourly_df["tmpf"]

    for month in range(1, 13):
        month_data = hourly_df[hourly_df["month"] == month]
        clim: dict[str, dict] = {}
        for hour in range(24):
            hour_data = month_data[month_data["hour"] == hour]["tmax_minus_obs"].dropna()
            if len(hour_data) > 10:
                clim[str(hour)] = {
                    "mean": float(hour_data.mean()),
                    "std": float(hour_data.std()),
                    "p10": float(hour_data.quantile(0.1)),
                    "p90": float(hour_data.quantile(0.9)),
                    "n": len(hour_data),
                }
            else:
                clim[str(hour)] = {"mean": 0.0, "std": 3.0, "p10": -5.0, "p90": 8.0, "n": 0}

        out_path = DIURNAL_DIR / f"{station}_{month}.json"
        with open(out_path, "w") as f:
            json.dump(clim, f, indent=2)
        logger.info("Saved diurnal climatology: %s", out_path)


def build_regime_clusters(n_clusters: int = 12) -> None:
    REGIME_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Building synthetic regime clusters (ERA5 not downloaded — using random seed data)")
    rng = np.random.default_rng(42)
    n_samples = 1825
    features = rng.standard_normal((n_samples, 50))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    np.save(REGIME_DIR / "cluster_centroids.npy", kmeans.cluster_centers_)
    dates = pd.date_range(start="2019-01-01", periods=n_samples, freq="D")
    pd.DataFrame({"date": dates, "cluster": labels}).to_parquet(
        REGIME_DIR / "historical_labels.parquet", index=False
    )
    logger.info("Regime clusters saved to %s", REGIME_DIR)


async def main() -> None:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    end = date.today()
    start = end.replace(year=end.year - 5)

    logger.info("Bootstrapping %d years of history for stations: %s", 5, STATIONS)

    import os
    noaa_token = os.environ.get("NOAA_CDO_TOKEN", "")

    for station in STATIONS:
        logger.info("=== %s ===", station)

        hourly_df = await fetch_hourly_asos(station, start, end)
        if not hourly_df.empty:
            hourly_path = HIST_DIR / f"{station}_hourly.parquet"
            hourly_df.to_parquet(hourly_path, index=False)
            logger.info("Saved %d hourly obs to %s", len(hourly_df), hourly_path)

        build_diurnal_climatology(hourly_df, station)

        if noaa_token:
            station_id = STATION_IDS.get(station)
            if station_id:
                daily_df = await fetch_ghcnd_data(station_id, start, end, noaa_token)
                if not daily_df.empty:
                    daily_path = HIST_DIR / f"{station}_daily_ghcnd.parquet"
                    daily_df.to_parquet(daily_path, index=False)
                    logger.info("Saved %d daily GHCND rows to %s", len(daily_df), daily_path)
        else:
            logger.warning("NOAA_CDO_TOKEN not set — skipping GHCND download for %s", station)

    build_regime_clusters()
    logger.info("Bootstrap complete! Check data/ directory for output.")


if __name__ == "__main__":
    asyncio.run(main())
