"""Honesty audit of the mid-spread (contested) fade signal (roadmap gate).

The multi-lead scan shows positive gated P&L concentrated in the tercile of
markets priced NEAREST 0.5 (|d1_mid - 0.5| smallest) — the "contested" band.
Two facts make it suspicious before a dollar is risked:

  * In that band the model's Brier (~0.31) is WORSE than the book's (~0.22):
    the forecast has negative skill there, yet P&L is positive.
  * The P&L GROWS as the forecast lead lengthens ($0.6 @24h → $17 @72h) — i.e.
    it earns MORE as the forecast gets worse. A skill signal does the opposite.

Both point to a structural fade (uncertainty/anchoring premium on near-50/50
brackets), not forecast edge. This audit attacks it the same four ways as the
between-NO audit:

  1. Integrity / leakage — rows in the eval window, prices tradeable, no dup.
  2. De-correlate — the scan counts each station-day once per lead. A real
     strategy trades a station-day ONCE. Re-score at a SINGLE lead, aggregate
     P&L per settlement DATE (the independent unit), block-bootstrap over dates
     for an honest CI / P(profit) / date-Sharpe.
  3. Does the model matter? — compare the model-gated trades against a NULL
     structural fade that, ignoring the forecast, sells the uncertainty on every
     contested bracket (fade toward the nearer extreme). If the null earns the
     same, the "forecast" is decorative and this is pure structure.
  4. Fill + fee realism — re-price under the taker fee (you cross the spread)
     and under a 1-2c adverse-fill haircut, since maker fills near 50/50 (the
     most-contested, best-defended prices) are the least likely to rest.

    PYTHONPATH=. python -m research.audit_midspread            # --lead 24
    PYTHONPATH=. python -m research.audit_midspread --lead 72
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


def contested_mask(mid: np.ndarray, q: float = 1.0 / 3.0) -> np.ndarray:
    """The most-contested band: rows whose mid is in the nearest-to-0.5 tercile
    by |mid - 0.5|. `q` is the quantile of that distance kept (default lowest
    third)."""
    dist = np.abs(mid - 0.5)
    cut = np.quantile(dist, q)
    return dist <= cut


def null_fade_pnl(mid: np.ndarray, outcome: np.ndarray, fee_coef: float,
                  min_price: float, haircut: float = 0.0) -> np.ndarray:
    """Model-free structural fade of contested brackets: fade toward the nearer
    extreme (mid < 0.5 → buy NO, mid > 0.5 → buy YES), sizing 1 contract on every
    row whose entry price clears the floor. Isolates the anchoring/uncertainty
    premium with the forecast removed. `haircut` cents of adverse fill stress
    maker-fill realism."""
    buy_yes = mid > 0.5
    entry = np.where(buy_yes, mid, 1.0 - mid)      # cost of the faded side
    take = entry >= min_price
    win = np.where(buy_yes, outcome - mid, (1.0 - outcome) - (1.0 - mid))
    fee = np.array([kalshi_fee(1.0, m, fee_coef=fee_coef) for m in mid])
    pnl = np.where(take, win - fee - haircut, 0.0)
    return pnl


def block_bootstrap(date_pnl: pd.Series, n: int = 10000, seed: int = 0) -> dict:
    """Resample settlement DATES with replacement; each draw sums that date's
    total P&L. Dates are the independent unit (within-date trades share a weather
    regime)."""
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
    # contested band is defined per-lead on that lead's price distribution.
    bl = df[df.lead == lead].copy()
    con = contested_mask(bl.mid.to_numpy())
    b = bl[con].copy()

    # ── 1. integrity ────────────────────────────────────────────────────────
    print("=" * 74)
    print("1. INTEGRITY / LEAKAGE")
    print("=" * 74)
    bad_mid = ((b.mid <= 0) | (b.mid >= 1)).sum()
    bad_out = (~b.outcome.isin([0.0, 1.0])).sum()
    dups = b.duplicated(["lead", "station", "date", "strike_type", "mid"]).sum()
    print(f"  contested rows: {len(b)}  bad_mid(≤0|≥1): {bad_mid}  "
          f"non-binary outcome: {bad_out}  dup: {dups}")
    print(f"  mid range: [{b.mid.min():.3f}, {b.mid.max():.3f}]  "
          f"dates: {b.date.nunique()}  stations: {b.station.nunique()}  "
          f"strike_types: {sorted(b.strike_type.unique().tolist())}")

    # ── 2. de-correlate: single lead, P&L per settlement date ───────────────
    print("\n" + "=" * 74)
    print(f"2. DE-CORRELATED (single lead = {lead}h, model-gated), P&L per DATE")
    print("=" * 74)
    fair, mid, out = b.fair.to_numpy(), b.mid.to_numpy(), b.outcome.to_numpy()
    pnl, traded = per_trade_pnl(fair, mid, out, min_edge=MIN_EDGE,
                                fee_coef=MAKER_FEE_COEF, min_price=FLOOR)
    tr = b.assign(pnl=pnl, traded=traded)
    t = tr[tr.traded]
    side = np.where(t.fair > t.mid, "YES", "NO")
    date_pnl = t.groupby("date")["pnl"].sum()
    sd_days = t.drop_duplicates(["station", "date"]).shape[0]
    print(f"  brackets traded: {len(t)}  (NO-share {np.mean(side=='NO'):.2f})  "
          f"station-days: {sd_days}  settlement-dates: {len(date_pnl)}")
    print(f"  model Brier {brier_score(fair, out):.4f}  vs  market Brier "
          f"{brier_score(mid, out):.4f}   (model worse = negative skill here)")
    print(f"  total P&L: ${t.pnl.sum():.2f}   mean/date: ${date_pnl.mean():.2f}   "
          f"sd/date: ${date_pnl.std():.2f}   date-Sharpe: {annualized_sharpe(date_pnl.values):.2f}")
    bs = block_bootstrap(date_pnl)
    print(f"  block-bootstrap total P&L: mean ${bs['mean_total']:.1f}  "
          f"90% CI [${bs['p05']:.1f}, ${bs['p95']:.1f}]  P(profit)={bs['prob_profit']:.2f}")
    top = date_pnl.sort_values()
    tot = date_pnl.sum()
    print(f"  concentration: top-3 dates = {top.tail(3).sum()/tot:+.0%} of P&L | "
          f"worst-3 dates = ${top.head(3).sum():.2f}")

    # ── 3. does the model matter? null structural fade ──────────────────────
    print("\n" + "=" * 74)
    print("3. NULL STRUCTURAL FADE (fade every contested bracket to nearer")
    print("   extreme, ignore the model)")
    print("=" * 74)
    npnl = null_fade_pnl(mid, out, MAKER_FEE_COEF, FLOOR)
    n_taken = int((npnl != 0).sum())
    null_date = pd.Series(npnl, index=b.date.values).groupby(level=0).sum()
    print(f"  null trades: {n_taken}   total P&L: ${npnl.sum():.2f}   "
          f"date-Sharpe: {annualized_sharpe(null_date.values):.2f}")
    ratio = npnl.sum() / t.pnl.sum() if t.pnl.sum() else float("nan")
    print(f"  model-gated ${t.pnl.sum():.2f} vs null ${npnl.sum():.2f}  "
          f"(null/model {ratio:+.0%}) → "
          f"{'MODEL ADDS LITTLE (structural)' if npnl.sum() >= 0.7*t.pnl.sum() else 'model contributes'}")

    # ── 4. fill / fee realism ───────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("4. FILL / FEE REALISM (single lead, model-gated)")
    print("=" * 74)
    for label, coef, hc in [("maker (base)", MAKER_FEE_COEF, 0.0),
                            ("taker (cross spread)", TAKER_FEE_COEF, 0.0),
                            ("maker + 1c adverse fill", MAKER_FEE_COEF, 0.01),
                            ("maker + 2c adverse fill", MAKER_FEE_COEF, 0.02)]:
        p2, tr2 = per_trade_pnl(fair, mid, out, min_edge=MIN_EDGE,
                                 fee_coef=coef, min_price=FLOOR)
        p2 = np.where(tr2, p2 - hc, 0.0)
        print(f"  {label:<26} total P&L: ${p2[tr2].sum():+7.2f}   "
              f"date-Sharpe: {annualized_sharpe(pd.Series(p2, index=b.date.values).groupby(level=0).sum().values):.2f}")
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
