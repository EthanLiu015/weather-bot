"""Compact tennis tick part files: one parquet per finished day.

The recorder flushes minute-level part files all day; this cron job merges
every day-directory except today's into a single sorted, snappy-compressed
file and removes the parts. Keeps file counts and disk usage flat.

    PYTHONPATH=. python scripts/tennis_compact.py
"""
from __future__ import annotations

import glob
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

TICKS_DIR = "data/capture/tennis_ticks"
NUMERIC_COLUMNS = ["yes_bid", "yes_ask", "last_price", "volume_fp",
                    "open_interest_fp", "price", "delta_fp"]


def compact_day(day_dir: str) -> None:
    parts = sorted(glob.glob(f"{day_dir}/part-*.parquet"))
    if not parts:
        return
    merged_path = f"{day_dir}/ticks.parquet"
    frames = [pd.read_parquet(p) for p in parts]
    if os.path.exists(merged_path):
        frames.insert(0, pd.read_parquet(merged_path))
    for frame in frames:
        for col in NUMERIC_COLUMNS:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
    merged = pd.concat(frames, ignore_index=True, sort=False).sort_values(["ts", "ticker"])
    merged.to_parquet(merged_path, compression="snappy")
    for p in parts:
        os.remove(p)
    logger.info("%s: %d parts -> %s (%d rows, %.1f MB)", day_dir, len(parts),
                merged_path, len(merged), os.path.getsize(merged_path) / 1e6)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    for day_dir in sorted(glob.glob(f"{TICKS_DIR}/date=*")):
        if day_dir.endswith(today):
            continue
        compact_day(day_dir)


if __name__ == "__main__":
    main()
