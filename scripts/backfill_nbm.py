"""
Historical NBM backfill for data/historical/features.parquet.

Populates nbm_t10/t25/t50/t75/t90/tmax/tmin/pop12/spread (and recomputes
nbm_gefs_delta) by byte-range GRIB2 reads against the noaa-nbm-grib2-pds S3
archive, processing and discarding each downloaded block immediately so disk
usage stays small regardless of date range.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import tempfile
from pathlib import Path

import httpx
import pandas as pd
from gribapi.errors import GribInternalError

from config.stations import station_coords
from ingestion.nbm import _grid_nearest_indices, _parse_nbm_files_for_stations

logger = logging.getLogger(__name__)

NBM_S3_BASE = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"

# features.parquet's lead_hour values. init_utc is always the 00z cycle
# (vdate is midnight and lead_hour is a multiple of 24), so only the 00z
# cycle's forecast hours are needed: tmax_fhour == lead_hour,
# tmin_fhour == lead_hour - 12.
LEAD_HOURS = [24, 48, 72, 96, 120, 168]

# The 9 NBM fields produced by ingestion.nbm._parse_nbm_files_for_stations,
# mapped onto features.parquet's nbm_<field> columns by _apply_nbm_backfill.
NBM_BACKFILL_COLS = ["t10", "t25", "t50", "t75", "t90", "tmax", "tmin", "pop12", "spread"]

# Fallback span (bytes) for the final message in an .idx file, where no
# subsequent message offset is available to bound the range.
_DEFAULT_TAIL_BYTES = 5_000_000


def _find_message_index(idx_lines: list[str], predicate) -> int | None:
    """Return the 1-based message index of the first idx line matching
    `predicate`, or None if no line matches."""
    for line in idx_lines:
        if predicate(line):
            return int(line.split(":", 1)[0])
    return None


# NBM core files carry separate APCP "prob >0.254" messages for the 1h, 6h,
# and 12h accumulation windows ending at the file's forecast hour (e.g. for
# f024: "23-24", "18-24", "12-24 hour acc fcst"). Only the 12h-wide window is
# POP12 — match on the window width rather than the absolute hours, since
# those shift with the forecast hour.
_APCP_PROB_RE = re.compile(r"APCP:surface:(\d+)-(\d+) hour acc fcst:prob >0\.254")


def _is_pop12_line(line: str) -> bool:
    """Match the NBM POP12 idx line (12h-wide APCP probability->0.254mm field)."""
    match = _APCP_PROB_RE.search(line)
    return bool(match) and int(match.group(2)) - int(match.group(1)) == 12


def _is_tmax_mean_line(line: str) -> bool:
    return "TMAX:2 m above ground" in line and "ens std dev" not in line


def _is_tmin_mean_line(line: str) -> bool:
    return "TMIN:2 m above ground" in line and "ens std dev" not in line


# (predicate, span) for each NBM field extracted by the backfill. tmax spans
# 2 messages (mean + std dev, both needed for the percentile derivation);
# tmin and pop12 are single messages.
_FIELD_SPECS = {
    "tmax": (_is_tmax_mean_line, 2),
    "tmin": (_is_tmin_mean_line, 1),
    "pop12": (_is_pop12_line, 1),
}


def _byte_range(idx_lines: list[str], start_index: int, span: int) -> tuple[int, int]:
    """Return the inclusive (start, end) byte range covering `span`
    consecutive messages starting at 1-based `start_index`."""
    offsets = {int(line.split(":", 1)[0]): int(line.split(":")[1]) for line in idx_lines}
    start = offsets[start_index]
    end_index = start_index + span
    if end_index in offsets:
        end = offsets[end_index] - 1
    else:
        end = start + _DEFAULT_TAIL_BYTES - 1
    return start, end


def _field_byte_range(idx_lines: list[str], field: str) -> tuple[int, int] | None:
    """Return the byte range covering the GRIB message(s) for `field`
    ("tmax", "tmin", or "pop12"), or None if the field isn't in this file."""
    predicate, span = _FIELD_SPECS[field]
    index = _find_message_index(idx_lines, predicate)
    if index is None:
        return None
    return _byte_range(idx_lines, index, span)


_NBM_PARTIAL_FILL_COLS = ["nbm_t10", "nbm_t25", "nbm_t75", "nbm_t90", "nbm_tmin", "nbm_pop12", "nbm_spread"]


def null_partial_nbm_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Null out columns in rows where nbm_t50 is valid (non-zero) but the
    remaining percentile/aux fields are all zero — indicating the backfill
    retrieved the median but the surrounding percentile messages were absent.

    Rows where nbm_t50 is itself zero (entire NBM slot missing) are left
    unchanged, as are fully-populated rows where t10 is non-zero.
    """
    partial_mask = (df["nbm_t50"].notna()) & (df["nbm_t50"] != 0.0) & (df["nbm_t10"] == 0.0) & (df["nbm_spread"] == 0.0)
    result = df.copy()
    for col in _NBM_PARTIAL_FILL_COLS:
        if col in result.columns:
            result.loc[partial_mask, col] = float("nan")
    return result


def _apply_nbm_backfill(features_df: pd.DataFrame, backfill_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join `backfill_df` (columns: date, station, lead_hour, plus
    NBM_BACKFILL_COLS) onto `features_df`, overwriting the corresponding
    nbm_* columns where backfill data is present, and recompute
    nbm_gefs_delta = |nbm_t50 - gefs_tmax_mean| for backfilled rows."""
    if backfill_df.empty:
        return features_df.copy()

    backfill = backfill_df.copy()
    backfill["lead_hour"] = backfill["lead_hour"].astype(features_df["lead_hour"].dtype)
    backfill["station"] = backfill["station"].astype(str)
    backfill = backfill.rename(columns={col: f"nbm_{col}" for col in NBM_BACKFILL_COLS})

    # features.parquet's `date` column is object dtype (datetime.date), while
    # the backfill CSV's `date` parses to datetime64 — merge on a normalized
    # temp key instead of relying on the dtypes matching directly.
    features = features_df.copy()
    features["_merge_date"] = pd.to_datetime(features["date"])
    backfill["_merge_date"] = pd.to_datetime(backfill["date"])
    backfill = backfill.drop(columns=["date"])

    merged = features.merge(backfill, on=["_merge_date", "station", "lead_hour"], how="left", suffixes=("", "_bf"))
    merged = merged.drop(columns=["_merge_date"])

    for col in NBM_BACKFILL_COLS:
        nbm_col = f"nbm_{col}"
        bf_col = f"{nbm_col}_bf"
        merged[nbm_col] = merged[bf_col].fillna(merged[nbm_col])
        merged = merged.drop(columns=[bf_col])

    has_real_t50 = merged["nbm_t50"] != 0.0
    merged["nbm_gefs_delta"] = 0.0
    merged.loc[has_real_t50, "nbm_gefs_delta"] = (
        merged.loc[has_real_t50, "nbm_t50"] - merged.loc[has_real_t50, "gefs_tmax_mean"]
    ).abs()

    return merged


def _nbm_archive_url(init_date: str, fhour: int) -> str:
    return f"{NBM_S3_BASE}/blend.{init_date}/00/core/blend.t00z.core.f{fhour:03d}.co.grib2"


async def _fetch_idx(url: str, client: httpx.AsyncClient) -> list[str] | None:
    for attempt in range(3):
        try:
            resp = await client.get(f"{url}.idx", timeout=30.0)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.text.strip().splitlines()
        except httpx.HTTPError as exc:
            logger.warning("idx fetch failed for %s (attempt %d): %s", url, attempt + 1, exc)
            await asyncio.sleep(2**attempt)
    return None


async def _fetch_range(url: str, byte_range: tuple[int, int], client: httpx.AsyncClient) -> bytes:
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"}
    for attempt in range(3):
        try:
            resp = await client.get(url, headers=headers, timeout=60.0)
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError as exc:
            logger.warning("range fetch failed for %s (attempt %d): %s", url, attempt + 1, exc)
            await asyncio.sleep(2**attempt)
    return b""


async def _fetch_fhour_block(init_date: str, fhour: int, fields: list[str], client: httpx.AsyncClient) -> bytes | None:
    """Fetch and concatenate the byte ranges for `fields` from the given
    forecast-hour file's .idx, or None if the run/file doesn't exist."""
    url = _nbm_archive_url(init_date, fhour)
    idx_lines = await _fetch_idx(url, client)
    if idx_lines is None:
        return None
    chunks = []
    for field in fields:
        byte_range = _field_byte_range(idx_lines, field)
        if byte_range is None:
            continue
        chunks.append(await _fetch_range(url, byte_range, client))
    return b"".join(chunks) if chunks else None


async def _backfill_init_date(
    init_date: str,
    client: httpx.AsyncClient,
    grid_indices: dict[str, int],
    grid_signature: tuple,
) -> list[dict]:
    """Fetch and parse NBM data for every (lead_hour, station) combination
    needed for a single 00z init_date, returning rows ready to feed into
    _apply_nbm_backfill. `grid_indices`/`grid_signature` come from
    _grid_nearest_indices, computed once for the whole run."""
    vdate = pd.Timestamp(init_date)
    rows: list[dict] = []
    for lead_hour in LEAD_HOURS:
        tmax_bytes, tmin_bytes = await asyncio.gather(
            _fetch_fhour_block(init_date, lead_hour, ["tmax", "pop12"], client),
            _fetch_fhour_block(init_date, lead_hour - 12, ["tmin"], client),
        )
        if tmax_bytes is None:
            logger.warning("NBM run %s f%03d unavailable, skipping lead_hour=%d", init_date, lead_hour, lead_hour)
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmax_path = Path(tmpdir) / "tmax.grib2"
            tmax_path.write_bytes(tmax_bytes)
            tmin_path = None
            if tmin_bytes:
                tmin_path = Path(tmpdir) / "tmin.grib2"
                tmin_path.write_bytes(tmin_bytes)

            vdate_lh = vdate + pd.Timedelta(hours=lead_hour)
            parsed = _parse_nbm_files_for_stations(
                str(tmax_path), str(tmin_path) if tmin_path else None, grid_indices, grid_signature,
            )
            for station, fields in parsed.items():
                rows.append({"date": vdate_lh, "station": station, "lead_hour": lead_hour, **fields})

    return rows


async def _resolve_grid(
    init_dates: list[str], client: httpx.AsyncClient, coords: dict[str, tuple[float, float]]
) -> tuple[dict[str, int], tuple]:
    """Compute grid_indices/grid_signature once, from the first init_date
    with an available f024 tmax file. Reused across the whole backfill run
    (see _grid_nearest_indices)."""
    for init_date in init_dates:
        tmax_bytes = await _fetch_fhour_block(init_date, LEAD_HOURS[0], ["tmax"], client)
        if tmax_bytes is None:
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            tmax_path = Path(tmpdir) / "tmax.grib2"
            tmax_path.write_bytes(tmax_bytes)
            try:
                return _grid_nearest_indices(str(tmax_path), coords)
            except GribInternalError as exc:
                logger.warning("Failed to resolve grid from %s f%03d: %s", init_date, LEAD_HOURS[0], exc)
    raise RuntimeError("Could not resolve NBM grid: no init_date has a usable f024 tmax file")


def _required_init_dates(features_df: pd.DataFrame, lead_hours: list[int] = LEAD_HOURS) -> list[str]:
    """Return the sorted, deduplicated list of 00z init_dates (YYYYMMDD)
    needed to backfill every (date, lead_hour) combination in `features_df`."""
    dates = pd.to_datetime(features_df["date"]).unique()
    init_dates = {
        (pd.Timestamp(d) - pd.Timedelta(hours=lh)).strftime("%Y%m%d")
        for d in dates
        for lh in lead_hours
    }
    return sorted(init_dates)


async def run_backfill(init_dates: list[str], output_csv: Path, concurrency: int = 8) -> None:
    """Backfill each init_date in `init_dates`, appending results to
    `output_csv` incrementally so the run is resumable. Already-completed
    init_dates (present in an existing output_csv) are skipped."""
    coords = station_coords()

    done: set[str] = set()
    if output_csv.exists():
        done = set(pd.read_csv(output_csv, usecols=["init_date"], dtype=str)["init_date"].unique())

    pending = [d for d in init_dates if d not in done]
    logger.info("Backfilling %d/%d init_dates (%d already done)", len(pending), len(init_dates), len(done))

    if not pending:
        return

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async with httpx.AsyncClient() as client:
        grid_indices, grid_signature = await _resolve_grid(pending, client, coords)

        async def process(init_date: str) -> None:
            async with semaphore:
                rows = await _backfill_init_date(init_date, client, grid_indices, grid_signature)
            if rows:
                df = pd.DataFrame(rows)
                df.insert(0, "init_date", init_date)
                async with write_lock:
                    df.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)
            logger.info("init_date %s: %d rows", init_date, len(rows))

        await asyncio.gather(*(process(d) for d in pending))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/historical/features.parquet")
    parser.add_argument("--output", default="data/historical/nbm_backfill.csv")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N init_dates (for smoke testing)")
    parser.add_argument("--apply", action="store_true", help="Merge --output into --features and write features.parquet")
    args = parser.parse_args()

    if args.apply:
        features_df = pd.read_parquet(args.features)
        backfill_df = pd.read_csv(args.output, parse_dates=["date"])
        merged = _apply_nbm_backfill(features_df, backfill_df)
        merged.to_parquet(args.features, index=False)
        logger.info("Merged %d backfill rows into %s (%d rows total)", len(backfill_df), args.features, len(merged))
        return

    features_df = pd.read_parquet(args.features)
    init_dates = _required_init_dates(features_df)
    if args.limit is not None:
        init_dates = init_dates[: args.limit]
    asyncio.run(run_backfill(init_dates, Path(args.output), args.concurrency))


if __name__ == "__main__":
    main()
