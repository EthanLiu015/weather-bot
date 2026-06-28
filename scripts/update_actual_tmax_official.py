"""One-time migration: replace `actual_tmax` in features.parquet with the OFFICIAL
NWS daily max (IEM ASOS daily summary) at the station Kalshi settles on.

Why: the training target used to be the max of hourly METAR temps, which
disagreed with Kalshi settlements ~12% of the time (fatal on 2° brackets). Using
the official max — verified 100% agreement with Kalshi settlements — aligns what
the model learns with what the market resolves on. See
plans/nws-settlement-source.md and [[kalshi-bracket-markets]].

Surgical (target-only): preserves all forecast features (real ERA5/ECMWF/NBM
columns from the separate backfills). The obs_minus_model_* lag features and the
climatology normals remain on the old target — a known minor approximation
(≤2°F shift on ~12% of days); revisit via a full rebuild if results warrant.

Usage: PYTHONPATH=. python scripts/update_actual_tmax_official.py
"""
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = Path("data/historical/features.parquet")
IEM_START = date(2021, 1, 1)


def main() -> None:
    from ingestion.nws_daily import official_daily_tmax_series, SETTLEMENT_STATION

    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["_ds"] = df["date"].dt.strftime("%Y-%m-%d")
    end = date.today()

    backup = FEATURES_PATH.with_suffix(
        f".bak_pre_official_tmax_{date.today():%Y%m%d}.parquet"
    )
    if not backup.exists():
        df.drop(columns=["_ds"]).to_parquet(backup, index=False)
        logger.info("Backed up current features to %s", backup)

    old = df["actual_tmax"].copy()
    new_col = old.copy()
    total_changed = 0

    for station in sorted(df["station"].unique()):
        if station not in SETTLEMENT_STATION:
            logger.warning("No settlement station for %s — keeping hourly target", station)
            continue
        try:
            official = official_daily_tmax_series(station, IEM_START, end)
        except Exception as exc:
            logger.warning("IEM fetch failed for %s (%s) — keeping hourly target", station, exc)
            continue
        if official.empty:
            logger.warning("IEM empty for %s — keeping hourly target", station)
            continue

        day_to_max = {d.strftime("%Y-%m-%d"): v for d, v in official.items()}
        mask = df["station"] == station
        mapped = df.loc[mask, "_ds"].map(day_to_max)
        # Only overwrite where IEM has a value; keep hourly fallback otherwise.
        filled = mapped.fillna(old[mask])
        new_col.loc[mask] = filled.values

        matched = int(mapped.notna().sum())
        changed = int((mapped.notna() & (mapped.values != old[mask].values)).sum())
        total_changed += changed
        logger.info("%s: %d/%d rows matched IEM, %d changed (net%+.0f°F mean)",
                    station, matched, int(mask.sum()), changed,
                    float((filled - old[mask]).mean()))

    df["actual_tmax"] = new_col
    df = df.drop(columns=["_ds"])
    df.to_parquet(FEATURES_PATH, index=False)
    logger.info("Updated actual_tmax in %s (%d rows changed of %d)",
                FEATURES_PATH, total_changed, len(df))


if __name__ == "__main__":
    main()
