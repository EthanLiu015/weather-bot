"""
One-time script: pull 5 years of ASOS + ERA5 history and build diurnal climatology.

Usage: python scripts/bootstrap_history.py
"""
import asyncio
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import cdsapi
import httpx
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans

from config.stations import STATION_REGISTRY, ALL_ICAO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HIST_DIR = Path("data/historical")
DIURNAL_DIR = Path("data/diurnal_climatology")
REGIME_DIR = Path("data/regime_clusters")
ERA5_DIR = Path("data/era5")
NOAA_CDO_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# GHCND station IDs for each ICAO
GHCND_IDS = {
    "KLGA":  "GHCND:USW00094789",
    "KORD":  "GHCND:USW00094846",
    "KLAX":  "GHCND:USW00023174",
    "KMIA":  "GHCND:USW00012839",
    "KIAH":  "GHCND:USW00012960",
    "KPHL":  "GHCND:USW00013739",
    "KATL":  "GHCND:USW00013874",
    "KAUS":  "GHCND:USW00013904",
    "KDEN":  "GHCND:USW00003017",
    "KMSY":  "GHCND:USW00012916",
    "KPHX":  "GHCND:USW00023183",
    "KSFO":  "GHCND:USW00023234",
    "KSEA":  "GHCND:USW00024233",
    "KBOS":  "GHCND:USW00014739",
    "KDFW":  "GHCND:USW00003927",
    "KDCA":  "GHCND:USW00013743",
    "KLAS":  "GHCND:USW00023169",
    "KMSP":  "GHCND:USW00014922",
    "KOKC":  "GHCND:USW00013967",
    "KSAT":  "GHCND:USW00012921",
}

# ERA5 variables to download
ERA5_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
             "total_cloud_cover", "total_precipitation", "surface_pressure",
             "2m_dewpoint_temperature"]


# ---------------------------------------------------------------------------
# Iowa State ASOS (hourly obs)
# ---------------------------------------------------------------------------

async def fetch_hourly_asos(station: str, start: date, end: date) -> pd.DataFrame:
    rows = []
    current = start
    async with httpx.AsyncClient() as client:
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)
            url = (
                f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
                f"?station={station}&data=tmpf,dwpf,sknt,vsby&year1={current.year}"
                f"&month1={current.month}&day1={current.day}"
                f"&year2={chunk_end.year}&month2={chunk_end.month}&day2={chunk_end.day}"
                f"&tz=UTC&format=comma&latlon=no&direct=no"
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
                            tmpf = float(parts[2].strip()) if parts[2].strip() not in ("M", "") else float("nan")
                            rows.append({"datetime": dt, "tmpf": tmpf, "station": station})
                        except (ValueError, IndexError):
                            pass
                logger.info("ASOS %s %s–%s: %d rows", station, current, chunk_end, len(lines) - 2)
            except Exception as exc:
                logger.warning("ASOS fetch failed for %s %s: %s", station, current, exc)
            current = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.3)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["datetime", "tmpf", "station"])


# ---------------------------------------------------------------------------
# NOAA GHCND (daily summaries — optional)
# ---------------------------------------------------------------------------

async def fetch_ghcnd_data(station_id: str, start: date, end: date, token: str) -> pd.DataFrame:
    all_rows: list[dict] = []
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
                logger.info("GHCND %s %s–%s: %d rows", station_id, current, chunk_end, len(data))
            except Exception as exc:
                logger.warning("GHCND fetch failed: %s", exc)
            current = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.5)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# ERA5 via CDS API
# ---------------------------------------------------------------------------

def _era5_area_for_stations() -> list[float]:
    lats = [s.lat for s in STATION_REGISTRY.values()]
    lons = [s.lon for s in STATION_REGISTRY.values()]
    north = max(lats) + 1.0
    south = min(lats) - 1.0
    west = min(lons) - 1.0
    east = max(lons) + 1.0
    return [round(north, 1), round(west, 1), round(south, 1), round(east, 1)]


def download_era5_month(year: int, month: int, out_dir: Path) -> Path | None:
    """Download one month of ERA5 data. CDS rejects full-year requests as too large."""
    out_path = out_dir / f"era5_{year}_{month:02d}.nc"
    if out_path.exists() and out_path.stat().st_size > 10_000:
        logger.info("ERA5 %d-%02d cached (%s)", year, month, out_path.name)
        return out_path

    import calendar
    last_day = calendar.monthrange(year, month)[1]
    area = _era5_area_for_stations()
    logger.info("Downloading ERA5 %d-%02d ...", year, month)
    try:
        client = cdsapi.Client(quiet=True)
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ERA5_VARS,
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, last_day + 1)],
                "time": ["00:00", "06:00", "12:00", "18:00"],  # 6-hourly keeps size small
                "area": area,
                "format": "netcdf",
            },
            str(out_path),
        )
        logger.info("ERA5 %d-%02d: %.1f MB", year, month, out_path.stat().st_size / 1e6)
        return out_path
    except Exception as exc:
        logger.error("ERA5 %d-%02d failed: %s", year, month, exc)
        if out_path.exists():
            out_path.unlink()
        return None


def download_era5_year(year: int, out_dir: Path) -> list[Path]:
    """Download all 12 months for a year, return list of successful paths."""
    paths = []
    for month in range(1, 13):
        path = download_era5_month(year, month, out_dir)
        if path:
            paths.append(path)
    return paths


def extract_era5_station_series(nc_path: Path, station: str) -> pd.DataFrame:
    meta = STATION_REGISTRY[station]
    lat, lon = meta.lat, meta.lon
    lon_360 = lon % 360

    try:
        ds = xr.open_dataset(nc_path)
        station_ds = ds.sel(latitude=lat, longitude=lon_360, method="nearest")

        rows = []
        time_vals = station_ds.time.values
        for var in ds.data_vars:
            pass

        df = station_ds.to_dataframe().reset_index()
        df["station"] = station

        # Convert temperature from Kelvin to Fahrenheit
        for col in df.columns:
            if "t2m" in col or "d2m" in col or "temperature" in col.lower():
                df[col] = (df[col] - 273.15) * 9 / 5 + 32

        logger.info("ERA5 extracted for %s: %d rows", station, len(df))
        return df
    except Exception as exc:
        logger.warning("ERA5 extraction failed for %s from %s: %s", station, nc_path, exc)
        return pd.DataFrame()


def build_era5_feature_history(nc_paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for station in ALL_ICAO:
        station_frames = []
        for path in nc_paths:
            if path is None:
                continue
            df = extract_era5_station_series(path, station)
            if not df.empty:
                station_frames.append(df)

        if not station_frames:
            logger.warning("No ERA5 data extracted for %s", station)
            continue

        combined = pd.concat(station_frames, ignore_index=True)
        out_path = out_dir / f"{station}_era5.parquet"
        combined.to_parquet(out_path, index=False)
        logger.info("ERA5 history saved for %s: %d rows → %s", station, len(combined), out_path)


# ---------------------------------------------------------------------------
# Diurnal climatology
# ---------------------------------------------------------------------------

def build_diurnal_climatology(hourly_df: pd.DataFrame, station: str) -> None:
    DIURNAL_DIR.mkdir(parents=True, exist_ok=True)
    if hourly_df.empty:
        logger.warning("No hourly data for %s — skipping diurnal climatology", station)
        return

    df = hourly_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month

    daily_max = df.groupby("date")["tmpf"].max().reset_index().rename(columns={"tmpf": "daily_tmax"})
    df = df.merge(daily_max, on="date")
    df["tmax_minus_obs"] = df["daily_tmax"] - df["tmpf"]

    for month in range(1, 13):
        month_data = df[df["month"] == month]
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
    logger.info("Diurnal climatology saved for %s (12 months)", station)


# ---------------------------------------------------------------------------
# Synoptic regime clusters (500hPa geopotential from ERA5)
# ---------------------------------------------------------------------------

def build_regime_clusters_from_era5(nc_paths: list[Path], n_clusters: int = 12) -> None:
    """Cluster synoptic regimes using 2m temperature spatial patterns across all stations."""
    REGIME_DIR.mkdir(parents=True, exist_ok=True)

    t2m_frames = []
    time_index = []
    for path in nc_paths:
        if path is None:
            continue
        try:
            ds = xr.open_dataset(path)
            if "t2m" not in ds.data_vars:
                continue
            # Daily mean across all grid points — shape (time, lat*lon)
            daily = ds["t2m"].resample(time="1D").mean()
            flat = daily.values.reshape(len(daily.time), -1)
            t2m_frames.append(flat)
            time_index.extend(pd.to_datetime(daily.time.values).date.tolist())
        except Exception as exc:
            logger.warning("ERA5 t2m load failed for %s: %s", path, exc)

    if t2m_frames:
        features = np.vstack(t2m_frames)
        # Normalise so clustering captures spatial pattern not absolute temperature
        features = (features - features.mean(axis=1, keepdims=True)) / (features.std(axis=1, keepdims=True) + 1e-6)
        logger.info("Clustering %d daily t2m patterns into %d regimes", len(features), n_clusters)
    else:
        logger.warning("No ERA5 t2m data — building synthetic regime clusters")
        rng = np.random.default_rng(42)
        features = rng.standard_normal((1825, 50))
        time_index = pd.date_range(end=date.today(), periods=len(features), freq="D").date.tolist()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    np.save(REGIME_DIR / "cluster_centroids.npy", kmeans.cluster_centers_)

    pd.DataFrame({"date": time_index[:len(labels)], "cluster": labels}).to_parquet(
        REGIME_DIR / "historical_labels.parquet", index=False
    )
    logger.info("Regime clusters saved: %d days, %d clusters → %s", len(labels), n_clusters, REGIME_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    import os

    HIST_DIR.mkdir(parents=True, exist_ok=True)
    ERA5_DIR.mkdir(parents=True, exist_ok=True)

    end = date.today() - timedelta(days=1)
    start = end.replace(year=end.year - 5)
    years = list(range(start.year, end.year + 1))

    logger.info("Bootstrap: %s → %s | %d stations | years %s",
                start, end, len(ALL_ICAO), years)

    # --- Step 1: ERA5 download (one file per month to stay within CDS size limits) ---
    logger.info("=== ERA5 Download (%d years × 12 months) ===", len(years))
    era5_paths: list[Path] = []
    for year in years:
        month_paths = download_era5_year(year, ERA5_DIR)
        era5_paths.extend(month_paths)
    logger.info("ERA5: %d monthly files downloaded", len(era5_paths))

    # --- Step 2: Extract ERA5 time series per station ---
    logger.info("=== ERA5 Station Extraction ===")
    build_era5_feature_history(era5_paths, HIST_DIR)

    # --- Step 3: ASOS hourly obs per station ---
    logger.info("=== ASOS Hourly Observations ===")
    noaa_token = os.environ.get("NOAA_CDO_TOKEN", "")

    for station in ALL_ICAO:
        logger.info("--- %s (%s) ---", station, STATION_REGISTRY[station].city)

        hourly_df = await fetch_hourly_asos(station, start, end)
        if not hourly_df.empty:
            hourly_path = HIST_DIR / f"{station}_hourly.parquet"
            hourly_df.to_parquet(hourly_path, index=False)
            logger.info("Saved %d hourly rows → %s", len(hourly_df), hourly_path)

        build_diurnal_climatology(hourly_df, station)

        if noaa_token:
            ghcnd_id = GHCND_IDS.get(station)
            if ghcnd_id:
                daily_df = await fetch_ghcnd_data(ghcnd_id, start, end, noaa_token)
                if not daily_df.empty:
                    daily_path = HIST_DIR / f"{station}_daily_ghcnd.parquet"
                    daily_df.to_parquet(daily_path, index=False)
                    logger.info("Saved %d GHCND rows → %s", len(daily_df), daily_path)
        else:
            logger.info("NOAA_CDO_TOKEN not set — skipping GHCND for %s", station)

    # --- Step 4: Synoptic regime clusters ---
    logger.info("=== Regime Clusters ===")
    build_regime_clusters_from_era5([p for p in era5_paths if p])

    logger.info("Bootstrap complete!")
    logger.info("  ERA5 data:        %s", ERA5_DIR)
    logger.info("  Station history:  %s", HIST_DIR)
    logger.info("  Diurnal clim:     %s", DIURNAL_DIR)
    logger.info("  Regime clusters:  %s", REGIME_DIR)
    logger.info("Next step: python scripts/initial_train.py")


if __name__ == "__main__":
    asyncio.run(main())
