"""Obs-conditioned intraday edge test.

At local time T with settlement-aligned running max `run_max`, the final daily max is
  final = run_max + R,   R = residual rise ≥ 0  (the max can only climb).
So P(final > x) = P(R > x - run_max). We estimate R's distribution empirically per
snapshot time T (4/6/8/10pm local) from the training days, turn it into a
`prob_above(x)` function, price each Kalshi bracket with the existing bracket logic,
and edge-gate the model's fair value against the actual afternoon price.

The question: at these afternoon snapshots, does obs-conditioned fair value beat the
book's price (model Brier < market Brier; positive gated P&L)? If not here — with the
freshest possible signal — the intraday thesis is dead.

No look-ahead: R's distribution is fit on a disjoint TRAIN set of dates; the evaluated
day never contributes to its own residual distribution. Outcomes use the real Kalshi
settlement, not the reconstructed max.

    PYTHONPATH=. python -m research.intraday_edge
Data: data/historical/intraday_afternoon.parquet
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strategies.bracket_pricing import bracket_yes_prob
from backtest.real_market_eval import brier_score, per_trade_pnl

logger = logging.getLogger(__name__)

AFTERNOON_PATH = "data/historical/intraday_afternoon.parquet"
FEE_HINT = 0.07  # informational; per_trade_pnl books the real fee


def prob_final_above(residuals: np.ndarray, run_max: float, x: float) -> float:
    """P(final max > x) = P(R > x - run_max) from the empirical residual sample.

    x <= run_max ⇒ threshold already met (R ≥ 0) ⇒ 1.0. Large x ⇒ →0.
    """
    if residuals.size == 0:
        return float("nan")
    return float(np.mean(residuals > (x - run_max)))


def residuals_by_offset(day_df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Empirical residual-rise samples R = final_max − run_max(T), per offset.

    `day_df` is one row per (station, date, offset) with run_max; final_max is that
    station-day's largest run_max (the latest snapshot, ≈ settled max).
    """
    finals = day_df.groupby(["station", "date"])["run_max"].transform("max")
    r = day_df.assign(R=finals - day_df["run_max"])
    return {int(h): g["R"].dropna().to_numpy() for h, g in r.groupby("offset_h")}


def residuals_by_station_offset(day_df: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    """Empirical residual-rise samples R keyed by (station, offset).

    Same residual as `residuals_by_offset` but not pooled across stations — a
    coastal station's afternoon rise differs from a desert station's. Thin per-key
    samples are backstopped by the pooled model in `evaluate`.
    """
    finals = day_df.groupby(["station", "date"])["run_max"].transform("max")
    r = day_df.assign(R=finals - day_df["run_max"])
    return {(str(s), int(h)): g["R"].dropna().to_numpy()
            for (s, h), g in r.groupby(["station", "offset_h"])}


def _station_day_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Unique (station, date, offset) run_max rows — residuals are per station-day,
    not per market (all markets of a station-day share the same obs)."""
    return df.dropna(subset=["run_max"]).drop_duplicates(["station", "date", "offset_h"])


def evaluate(
    df: pd.DataFrame,
    min_edge: float = 0.04,
    train_frac: float = 0.5,
    min_station_samples: int = 8,
) -> dict:
    """Per-offset obs-conditioned edge vs the afternoon price, temporal split.

    Residuals are fit per (station, offset); a key with fewer than
    `min_station_samples` train rows falls back to the pooled per-offset sample so
    thin stations aren't priced off noise.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = np.sort(df["date"].unique())
    split = dates[int(len(dates) * train_frac)]
    train, test = df[df["date"] < split], df[df["date"] >= split]

    train_sd = _station_day_frame(train)
    R_by_off = residuals_by_offset(train_sd)
    R_by_so = residuals_by_station_offset(train_sd)

    results: dict[int, dict] = {}
    for h, grp in test.groupby("offset_h"):
        pooled = R_by_off.get(int(h))
        if pooled is None or pooled.size == 0:
            continue
        fair, mids, outs = [], [], []
        for row in grp.itertuples(index=False):
            if not np.isfinite(row.run_max):
                continue
            R = R_by_so.get((str(row.station), int(h)))
            if R is None or R.size < min_station_samples:
                R = pooled
            yes = bracket_yes_prob(
                lambda x, _R=R, _rm=row.run_max: prob_final_above(_R, _rm, x),
                row.strike_type, row.floor_strike, row.cap_strike,
            )
            if yes is None:
                continue
            fair.append(float(yes)); mids.append(float(row.price)); outs.append(float(row.settlement))
        if not fair:
            continue
        fair_a, mid_a, out_a = np.array(fair), np.array(mids), np.array(outs)
        pnl, traded = per_trade_pnl(fair_a, mid_a, out_a, min_edge=min_edge)
        n_tr = int(traded.sum())
        results[int(h)] = {
            "n": len(fair_a),
            "model_brier": brier_score(fair_a, out_a),
            "market_brier": brier_score(mid_a, out_a),
            "n_trades": n_tr,
            "pnl": float(pnl[traded].sum()),
            "win_rate": float((pnl[traded] > 0).mean()) if n_tr else 0.0,
            "mean_edge": float(np.abs(fair_a - mid_a)[traded].mean()) if n_tr else 0.0,
        }
    return {"by_offset": results, "train_days": int((dates < split).sum()),
            "test_days": int((dates >= split).sum()), "min_edge": min_edge}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = pd.read_parquet(AFTERNOON_PATH)
    res = evaluate(df)
    print("\n" + "=" * 72)
    print("OBS-CONDITIONED INTRADAY EDGE  (afternoon price vs run_max-conditioned fair)")
    print("=" * 72)
    print(f"  train days: {res['train_days']}  test days: {res['test_days']}  min_edge: {res['min_edge']}")
    print(f"  {'offset':>6} {'n':>5} {'modelB':>8} {'mktB':>8} {'edge?':>6} {'trades':>7} {'P&L$':>9} {'win%':>6}")
    for h, d in sorted(res["by_offset"].items()):
        edge = "YES" if d["model_brier"] < d["market_brier"] else "no"
        print(f"  {h:>4}h {d['n']:>5} {d['model_brier']:>8.4f} {d['market_brier']:>8.4f} "
              f"{edge:>6} {d['n_trades']:>7} {d['pnl']:>9.2f} {d['win_rate']:>6.0%}")
    print("-" * 72)
    print("  offset = local hour (16=4pm … 22=10pm). modelB<mktB and P&L>0 = obs edge.")


if __name__ == "__main__":
    main()
