"""Honesty audit of the between-bracket NO-fade signal (roadmap gate).

The fee/segment scan showed the only positive P&L is NO bets on 2°F "between"
brackets. Before believing it, this audit attacks it four ways:

  1. Integrity / leakage — rows are in the eval window, prices tradeable, no dup.
  2. De-correlate — the scan counts each station-day up to 6× (one row per lead)
     and once per between bracket. A real strategy trades a station-day ONCE at
     one decision lead. Re-score at a SINGLE lead and aggregate P&L per settlement
     DATE (the true independent unit), then block-bootstrap over dates for an
     honest CI / P(profit) / Sharpe.
  3. Does the model matter? — compare the model-gated NO trades against a NULL
     structural fade that sells EVERY between bracket above the price floor,
     ignoring the forecast. If the null earns the same, the model adds nothing
     and it is pure base-rate/anchoring structure.
  4. Fill + fee realism — re-price under the taker fee (you cross the spread) and
     under a half-cent adverse-fill haircut, since maker NO fills are not
     guaranteed.

    PYTHONPATH=. python -m research.audit_between            # uses --lead 24
    PYTHONPATH=. python -m research.audit_between --lead 48
Reads: data/historical/multilead_scored.parquet (from research.fee_segment_scan)
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from backtest.real_market_eval import per_trade_pnl, annualized_sharpe, brier_score
from backtest.track_b import MAKER_FEE_COEF, TAKER_FEE_COEF, kalshi_fee

logger = logging.getLogger(__name__)
SCORED_PATH = "data/historical/multilead_scored.parquet"
FLOOR = 0.15
MIN_EDGE = 0.04


def null_fade_pnl(mid: np.ndarray, outcome: np.ndarray, fee_coef: float,
                  min_price: float, haircut: float = 0.0) -> np.ndarray:
    """Structural NO-fade P&L, ignoring the model: SELL every bracket (buy NO)
    whose NO entry price (1-mid) clears the floor. `haircut` cents of adverse
    fill are added to the entry cost to stress maker-fill realism."""
    entry = 1.0 - mid                      # cost of a NO contract
    take = entry >= min_price
    pnl = np.zeros(len(mid))
    win = (1.0 - outcome) - entry          # NO pays 1 if YES didn't resolve
    fee = np.array([kalshi_fee(1.0, m, fee_coef=fee_coef) for m in mid])
    pnl[take] = win[take] - fee[take] - haircut
    return np.where(take, pnl, 0.0)


def block_bootstrap(date_pnl: pd.Series, n: int = 10000, seed: int = 0) -> dict:
    """Resample settlement DATES with replacement; each draw sums that date's
    total P&L. Dates are the independent unit (within-date trades are correlated
    by the shared weather regime)."""
    rng = np.random.default_rng(seed)
    vals = date_pnl.to_numpy()
    k = len(vals)
    totals = np.array([vals[rng.integers(0, k, k)].sum() for _ in range(n)])
    return {
        "mean_total": float(totals.mean()),
        "p05": float(np.percentile(totals, 5)),
        "p50": float(np.percentile(totals, 50)),
        "p95": float(np.percentile(totals, 95)),
        "prob_profit": float((totals > 0).mean()),
    }


def audit(df: pd.DataFrame, lead: int) -> None:
    b = df[df.strike_type == "between"].copy()

    # ── 1. integrity ────────────────────────────────────────────────────────
    print("=" * 74)
    print("1. INTEGRITY / LEAKAGE")
    print("=" * 74)
    bad_mid = ((b.mid <= 0) | (b.mid >= 1)).sum()
    bad_out = (~b.outcome.isin([0.0, 1.0])).sum()
    dups = b.duplicated(["lead", "station", "date", "mid"]).sum()
    print(f"  between rows: {len(b)}  bad_mid(≤0|≥1): {bad_mid}  "
          f"non-binary outcome: {bad_out}  dup(lead,stn,date,mid): {dups}")
    print(f"  leads present: {sorted(b.lead.unique().tolist())}  "
          f"dates: {b.date.nunique()}  stations: {b.station.nunique()}")

    # ── 2. de-correlate: single lead, P&L per settlement date ───────────────
    print("\n" + "=" * 74)
    print(f"2. DE-CORRELATED (single lead = {lead}h, model-gated NO), P&L per DATE")
    print("=" * 74)
    bl = b[b.lead == lead]
    fair, mid, out = bl.fair.to_numpy(), bl.mid.to_numpy(), bl.outcome.to_numpy()
    pnl, traded = per_trade_pnl(fair, mid, out, min_edge=MIN_EDGE,
                                fee_coef=MAKER_FEE_COEF, min_price=FLOOR)
    tr = bl.assign(pnl=pnl, traded=traded)
    t = tr[tr.traded]
    side = np.where(t.fair > t.mid, "YES", "NO")
    date_pnl = t.groupby("date")["pnl"].sum()
    sd_days = t.drop_duplicates(["station", "date"]).shape[0]
    print(f"  brackets traded: {len(t)}  (NO-share {np.mean(side=='NO'):.2f})  "
          f"station-days: {sd_days}  settlement-dates: {len(date_pnl)}")
    print(f"  total P&L: ${t.pnl.sum():.2f}   mean/date: ${date_pnl.mean():.2f}   "
          f"sd/date: ${date_pnl.std():.2f}   date-Sharpe: {annualized_sharpe(date_pnl.values):.2f}")
    bs = block_bootstrap(date_pnl)
    print(f"  block-bootstrap total P&L: mean ${bs['mean_total']:.1f}  "
          f"90% CI [${bs['p05']:.1f}, ${bs['p95']:.1f}]  P(profit)={bs['prob_profit']:.2f}")

    # concentration
    top = date_pnl.sort_values()
    tot = date_pnl.sum()
    print(f"  concentration: top-3 dates = {top.tail(3).sum()/tot:+.0%} of P&L | "
          f"worst-3 dates = ${top.head(3).sum():.2f}")

    # ── 3. does the model matter? null structural fade at the same lead ─────
    print("\n" + "=" * 74)
    print("3. NULL STRUCTURAL FADE (sell every between bracket, ignore model)")
    print("=" * 74)
    npnl = null_fade_pnl(mid, out, MAKER_FEE_COEF, FLOOR)
    n_taken = int((npnl != 0).sum())
    null_date = pd.Series(npnl, index=bl.date.values).groupby(level=0).sum()
    print(f"  null trades: {n_taken}   total P&L: ${npnl.sum():.2f}   "
          f"date-Sharpe: {annualized_sharpe(null_date.values):.2f}")
    print(f"  model-gated total ${t.pnl.sum():.2f} vs null ${npnl.sum():.2f}  → "
          f"{'MODEL ADDS LITTLE (structural)' if npnl.sum() >= 0.7*t.pnl.sum() else 'model contributes'}")

    # ── 4. fill / fee realism ───────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("4. FILL / FEE REALISM (single lead, model-gated NO)")
    print("=" * 74)
    for label, coef, hc in [("maker (base)", MAKER_FEE_COEF, 0.0),
                            ("taker (cross spread)", TAKER_FEE_COEF, 0.0),
                            ("maker + 1c adverse fill", MAKER_FEE_COEF, 0.01),
                            ("maker + 2c adverse fill", MAKER_FEE_COEF, 0.02)]:
        p2, tr2 = per_trade_pnl(fair, mid, out, min_edge=MIN_EDGE,
                                 fee_coef=coef, min_price=FLOOR)
        # apply haircut on taken rows
        p2 = np.where(tr2, p2 - hc, 0.0)
        print(f"  {label:<26} total P&L: ${p2[tr2].sum():+7.2f}   "
              f"date-Sharpe: {annualized_sharpe(pd.Series(p2, index=bl.date.values).groupby(level=0).sum().values):.2f}")
    print("=" * 74)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=24, help="decision lead to trade at")
    args = ap.parse_args()
    df = pd.read_parquet(SCORED_PATH)
    audit(df, lead=args.lead)


if __name__ == "__main__":
    main()
