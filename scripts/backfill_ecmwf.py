"""
ECMWF historical backfill for data/historical/features.parquet.

Populates ecmwf_tmax/ecmwf_tmin (and recomputes ecmwf_gefs_tmax_delta) from two
complementary archives; run each phase separately to stay within disk limits:

  --source wb2   WeatherBench2 public GCS zarr  (2016-2022, free, no auth)
  --source s3    ECMWF open-data S3 archive      (2023-01-18 onward)
  --apply        Merge --output CSV into --features and write features.parquet

Disk strategy: each downloaded chunk/range is decoded in-memory and discarded
immediately so at most one GRIB message or zarr chunk lives on disk at a time.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import httpx
import numcodecs
import numpy as np
import pandas as pd
from gribapi.errors import GribInternalError

from config.stations import station_coords
from ingestion.nbm import _grib_values_at_indices, _grid_nearest_indices

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

WB2_BASE = "https://storage.googleapis.com/weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
WB2_VAR = "2m_temperature"
WB2_EPOCH = datetime(2016, 1, 1)
WB2_NLAT = 721
WB2_NLON = 1440
WB2_RES = 0.25

ECMWF_S3_BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
S3_FORMAT_CUTOVER = "20240301"

LEAD_HOURS = [24, 48, 72, 96, 120, 168]
LEAD_DAYS = [lh // 24 for lh in LEAD_HOURS]  # [1, 2, 3, 4, 5, 7]

_3H_STEPS = list(range(0, 145, 3))
_6H_STEPS = list(range(150, 241, 6))
ALL_S3_STEPS = _3H_STEPS + _6H_STEPS

_WB2_CODEC = numcodecs.Blosc()


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _wb2_lat_idx(lat: float) -> int:
    return round((90.0 - lat) / WB2_RES)


def _wb2_lon_idx(lon: float) -> int:
    return round((lon % 360) / WB2_RES)


def _wb2_flat_idx(lat: float, lon: float) -> int:
    return _wb2_lat_idx(lat) * WB2_NLON + _wb2_lon_idx(lon)


def _wb2_lead_indices_for_day(lead_day: int) -> list[int]:
    base = 4 * lead_day
    return [base, base + 1, base + 2, base + 3]


def _wb2_time_idx(init_date: str) -> int:
    dt = datetime.strptime(init_date, "%Y%m%d")
    total_hours = (dt - WB2_EPOCH).total_seconds() / 3600
    return int(total_hours / 12)


def _steps_for_lead_day(lead_day: int) -> list[int]:
    lo = 24 * lead_day
    hi = 24 * (lead_day + 1)
    return [s for s in ALL_S3_STEPS if lo <= s < hi]


def _ecmwf_s3_url(init_date: str, step: int) -> str:
    if init_date >= S3_FORMAT_CUTOVER:
        return (
            f"{ECMWF_S3_BASE}/{init_date}/00z/ifs/0p25/oper/"
            f"{init_date}000000-{step}h-oper-fc.grib2"
        )
    return (
        f"{ECMWF_S3_BASE}/{init_date}/00z/0p4-beta/oper/"
        f"{init_date}000000-{step}h-oper-fc.grib2"
    )


def _parse_ecmwf_s3_index(index_text: str) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for line in index_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        param = obj.get("param", "")
        if param and "_offset" in obj and "_length" in obj and param not in result:
            result[param] = (int(obj["_offset"]), int(obj["_length"]))
    return result


def _required_init_dates(features_df: pd.DataFrame, lead_hours: list[int] = LEAD_HOURS) -> list[str]:
    dates = pd.to_datetime(features_df["date"]).unique()
    init_dates = {
        (pd.Timestamp(d) - pd.Timedelta(hours=lh)).strftime("%Y%m%d")
        for d in dates
        for lh in lead_hours
    }
    return sorted(init_dates)


def _apply_ecmwf_backfill(features_df: pd.DataFrame, backfill_df: pd.DataFrame) -> pd.DataFrame:
    if backfill_df.empty:
        return features_df.copy()

    backfill = backfill_df.copy()
    backfill["lead_hour"] = backfill["lead_hour"].astype(features_df["lead_hour"].dtype)
    backfill["station"] = backfill["station"].astype(str)

    features = features_df.copy()
    features["_merge_date"] = pd.to_datetime(features["date"])
    backfill["_merge_date"] = pd.to_datetime(backfill["date"])
    backfill = backfill.drop(columns=["date"], errors="ignore")

    merged = features.merge(
        backfill[["_merge_date", "station", "lead_hour", "ecmwf_tmax", "ecmwf_tmin"]].rename(
            columns={"ecmwf_tmax": "_bf_tmax", "ecmwf_tmin": "_bf_tmin"}
        ),
        on=["_merge_date", "station", "lead_hour"],
        how="left",
    )
    merged = merged.drop(columns=["_merge_date"])

    merged["ecmwf_tmax"] = merged["_bf_tmax"].fillna(merged["ecmwf_tmax"])
    merged["ecmwf_tmin"] = merged["_bf_tmin"].fillna(merged["ecmwf_tmin"])
    merged = merged.drop(columns=["_bf_tmax", "_bf_tmin"])

    has_real_tmax = merged["ecmwf_tmax"] != 0.0
    merged["ecmwf_gefs_tmax_delta"] = 0.0
    merged.loc[has_real_tmax, "ecmwf_gefs_tmax_delta"] = (
        merged.loc[has_real_tmax, "ecmwf_tmax"] - merged.loc[has_real_tmax, "gefs_tmax_mean"]
    ).abs()

    return merged


# ── WB2 async ─────────────────────────────────────────────────────────────────

def _decode_wb2_chunk(raw: bytes) -> np.ndarray:
    return np.frombuffer(_WB2_CODEC.decode(raw), dtype="<f4").reshape(WB2_NLAT, WB2_NLON)


async def _fetch_wb2_chunk(time_idx: int, lead_idx: int, client: httpx.AsyncClient) -> bytes | None:
    url = f"{WB2_BASE}/{WB2_VAR}/{time_idx}.{lead_idx}.0.0"
    for attempt in range(3):
        try:
            resp = await client.get(url, timeout=60.0)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError as exc:
            logger.warning("WB2 chunk fetch failed %s (attempt %d): %s", url, attempt + 1, exc)
            await asyncio.sleep(2 ** attempt)
    return None


async def _backfill_init_date_wb2(
    init_date: str,
    station_flat_indices: dict[str, int],
    client: httpx.AsyncClient,
) -> list[dict]:
    time_idx = _wb2_time_idx(init_date)
    rows: list[dict] = []

    for lead_day in LEAD_DAYS:
        lead_hour = lead_day * 24
        vdate = pd.Timestamp(init_date) + pd.Timedelta(days=lead_day)
        station_temps: dict[str, list[float]] = {s: [] for s in station_flat_indices}

        chunks = await asyncio.gather(
            *(_fetch_wb2_chunk(time_idx, li, client) for li in _wb2_lead_indices_for_day(lead_day))
        )
        for raw in chunks:
            if raw is None:
                continue
            try:
                grid = _decode_wb2_chunk(raw)
            except Exception as exc:
                logger.warning("WB2 decode failed time_idx=%d: %s", time_idx, exc)
                continue
            for station, flat_idx in station_flat_indices.items():
                val_k = float(grid.flat[flat_idx])
                if not np.isnan(val_k):
                    station_temps[station].append((val_k - 273.15) * 9 / 5 + 32)

        for station, temps in station_temps.items():
            if temps:
                rows.append({
                    "date": vdate,
                    "station": station,
                    "lead_hour": lead_hour,
                    "ecmwf_tmax": max(temps),
                    "ecmwf_tmin": min(temps),
                })

    return rows


async def run_wb2_backfill(init_dates: list[str], output_csv: Path, concurrency: int = 12) -> None:
    coords = station_coords()
    station_flat_indices = {s: _wb2_flat_idx(lat, lon) for s, (lat, lon) in coords.items()}

    done: set[str] = set()
    if output_csv.exists():
        done = set(pd.read_csv(output_csv, usecols=["init_date"], dtype=str)["init_date"].unique())

    pending = [d for d in init_dates if d not in done]
    logger.info("WB2: %d/%d init_dates to process (%d done)", len(pending), len(init_dates), len(done))
    if not pending:
        return

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async with httpx.AsyncClient() as client:
        async def process(init_date: str) -> None:
            async with semaphore:
                rows = await _backfill_init_date_wb2(init_date, station_flat_indices, client)
            if rows:
                df = pd.DataFrame(rows)
                df.insert(0, "init_date", init_date)
                async with write_lock:
                    df.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)
            logger.info("WB2 %s: %d rows", init_date, len(rows))

        await asyncio.gather(*(process(d) for d in pending))


# ── S3 async ──────────────────────────────────────────────────────────────────

async def _fetch_s3_index(url: str, client: httpx.AsyncClient) -> str | None:
    index_url = url.replace(".grib2", ".index")
    for attempt in range(3):
        try:
            resp = await client.get(index_url, timeout=30.0)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as exc:
            logger.warning("S3 index fetch failed %s (attempt %d): %s", index_url, attempt + 1, exc)
            await asyncio.sleep(2 ** attempt)
    return None


async def _fetch_s3_range(
    url: str, offset: int, length: int, client: httpx.AsyncClient
) -> bytes | None:
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    for attempt in range(3):
        try:
            resp = await client.get(url, headers=headers, timeout=60.0)
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError as exc:
            logger.warning("S3 range fetch failed %s (attempt %d): %s", url, attempt + 1, exc)
            await asyncio.sleep(2 ** attempt)
    return None


async def _resolve_s3_grid(
    init_dates: list[str], client: httpx.AsyncClient, coords: dict
) -> tuple[dict[str, int], tuple] | None:
    first_step = _steps_for_lead_day(LEAD_DAYS[0])[0]
    for init_date in init_dates[:10]:
        url = _ecmwf_s3_url(init_date, first_step)
        index_text = await _fetch_s3_index(url, client)
        if index_text is None:
            continue
        index = _parse_ecmwf_s3_index(index_text)
        if "2t" not in index:
            continue
        offset, length = index["2t"]
        raw = await _fetch_s3_range(url, offset, length, client)
        if not raw:
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "2t.grib2"
            p.write_bytes(raw)
            try:
                return _grid_nearest_indices(str(p), coords)
            except (GribInternalError, Exception) as exc:
                logger.warning("S3 grid resolve failed %s: %s", init_date, exc)
    return None


async def _backfill_init_date_s3(
    init_date: str,
    beta_grid: tuple[dict[str, int], tuple] | None,
    ifs_grid: tuple[dict[str, int], tuple] | None,
    client: httpx.AsyncClient,
) -> list[dict]:
    grid = ifs_grid if init_date >= S3_FORMAT_CUTOVER else beta_grid
    if grid is None:
        return []

    indices, signature = grid
    lead_day_temps: dict[str, dict[int, list[float]]] = {
        station: {ld: [] for ld in LEAD_DAYS} for station in indices
    }

    all_steps = [s for ld in LEAD_DAYS for s in _steps_for_lead_day(ld)]

    async def fetch_step(step: int) -> dict[str, float] | None:
        url = _ecmwf_s3_url(init_date, step)
        index_text = await _fetch_s3_index(url, client)
        if index_text is None:
            return None
        idx = _parse_ecmwf_s3_index(index_text)
        if "2t" not in idx:
            return None
        offset, length = idx["2t"]
        raw = await _fetch_s3_range(url, offset, length, client)
        if not raw:
            return None
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "2t.grib2"
            p.write_bytes(raw)
            try:
                return _grib_values_at_indices(str(p), {}, indices, signature, derived_forecast=None)
            except (GribInternalError, Exception) as exc:
                logger.warning("S3 GRIB parse failed %s step=%d: %s", init_date, step, exc)
                return None

    step_results = await asyncio.gather(*(fetch_step(s) for s in all_steps))

    for step, vals in zip(all_steps, step_results):
        if vals is None:
            continue
        lead_day = step // 24
        if lead_day not in lead_day_temps[next(iter(indices))]:
            continue
        for station, val_k in vals.items():
            if not np.isnan(val_k):
                val_f = (val_k - 273.15) * 9 / 5 + 32
                lead_day_temps[station][lead_day].append(val_f)

    rows: list[dict] = []
    for station in indices:
        for lead_day in LEAD_DAYS:
            temps = lead_day_temps[station][lead_day]
            if not temps:
                continue
            rows.append({
                "date": pd.Timestamp(init_date) + pd.Timedelta(days=lead_day),
                "station": station,
                "lead_hour": lead_day * 24,
                "ecmwf_tmax": max(temps),
                "ecmwf_tmin": min(temps),
            })

    return rows


async def run_s3_backfill(init_dates: list[str], output_csv: Path, concurrency: int = 2) -> None:
    coords = station_coords()

    done: set[str] = set()
    if output_csv.exists():
        done = set(pd.read_csv(output_csv, usecols=["init_date"], dtype=str)["init_date"].unique())

    pending = [d for d in init_dates if d not in done]
    logger.info("S3: %d/%d init_dates to process (%d done)", len(pending), len(init_dates), len(done))
    if not pending:
        return

    beta_dates = [d for d in pending if d < S3_FORMAT_CUTOVER]
    ifs_dates = [d for d in pending if d >= S3_FORMAT_CUTOVER]

    async with httpx.AsyncClient() as client:
        beta_grid = await _resolve_s3_grid(beta_dates, client, coords) if beta_dates else None
        ifs_grid = await _resolve_s3_grid(ifs_dates, client, coords) if ifs_dates else None

        if beta_grid is None and beta_dates:
            logger.warning("Could not resolve 0p4-beta grid; %d dates skipped", len(beta_dates))
        if ifs_grid is None and ifs_dates:
            logger.warning("Could not resolve ifs/0p25 grid; %d dates skipped", len(ifs_dates))

        semaphore = asyncio.Semaphore(concurrency)
        write_lock = asyncio.Lock()

        async def process(init_date: str) -> None:
            async with semaphore:
                rows = await _backfill_init_date_s3(init_date, beta_grid, ifs_grid, client)
            if rows:
                df = pd.DataFrame(rows)
                df.insert(0, "init_date", init_date)
                async with write_lock:
                    df.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)
            logger.info("S3 %s: %d rows", init_date, len(rows))

        await asyncio.gather(*(process(d) for d in pending))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/historical/features.parquet")
    parser.add_argument("--output", default=None)
    parser.add_argument("--source", choices=["wb2", "s3"], help="Data source for backfill run")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.apply:
        output = args.output or "data/historical/ecmwf_backfill.csv"
        features_df = pd.read_parquet(args.features)
        backfill_df = pd.read_csv(output, parse_dates=["date"])
        merged = _apply_ecmwf_backfill(features_df, backfill_df)
        merged.to_parquet(args.features, index=False)
        logger.info("Merged %d backfill rows into %s", len(backfill_df), args.features)
        return

    if args.source is None:
        parser.error("--source is required when not using --apply")

    features_df = pd.read_parquet(args.features)
    all_init_dates = _required_init_dates(features_df)

    if args.source == "wb2":
        # WB2 zarr covers 2016-01-01 through 2023-01-10 (00z)
        init_dates = [d for d in all_init_dates if d <= "20230110"]
        if args.limit is not None:
            init_dates = init_dates[: args.limit]
        output = Path(args.output or "data/historical/ecmwf_backfill_wb2.csv")
        concurrency = args.concurrency or 12
        asyncio.run(run_wb2_backfill(init_dates, output, concurrency))
    else:
        # ECMWF S3 archive starts at 2023-01-18
        init_dates = [d for d in all_init_dates if d >= "20230118"]
        if args.limit is not None:
            init_dates = init_dates[: args.limit]
        output = Path(args.output or "data/historical/ecmwf_backfill_s3.csv")
        concurrency = args.concurrency or 2
        asyncio.run(run_s3_backfill(init_dates, output, concurrency))


if __name__ == "__main__":
    main()
