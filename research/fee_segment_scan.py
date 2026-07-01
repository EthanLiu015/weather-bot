"""Single-train fee + segment scan (roadmap steps 1 & 2).

Training the look-ahead-free bundle is the only expensive part, and the model's
fair value is INDEPENDENT of the fee/price-floor. So train once, score every
market at every lead, dump the per-market rows to parquet, and then re-score any
(fee, price-floor) config in milliseconds — no retrain.

Two questions this answers:
  1. (execution) Does the corrected Kalshi fee — and trading as a maker, and a
     $0.15 price floor — change the multi-lead P&L verdict?
  2. (segment mining) Is there ANY (station, strike_type, lead) cell where model
     Brier < market Brier with positive fee-aware P&L? If none survives, the
     cheap edge tests are exhausted.

    PYTHONPATH=. python -m research.fee_segment_scan            # train + score + save
    PYTHONPATH=. python -m research.fee_segment_scan --from-cache  # re-score only
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from backtest.real_market_eval import (
    MULTILEAD_HOURS,
    HIST_DIR,
    EVAL_START,
    EVAL_END,
    _load_eval_markets,
    _build_distribution_cache,
    build_fair_value_fn,
    brier_score,
    per_trade_pnl,
    _segment_breakdown,
)
from backtest.track_b import TAKER_FEE_COEF, MAKER_FEE_COEF
from strategies.bracket_pricing import bracket_yes_prob

logger = logging.getLogger(__name__)

SCORED_PATH = f"{HIST_DIR}/multilead_scored.parquet"


def score_all_leads(
    features_path: str = f"{HIST_DIR}/features.parquet",
    prices_path: str = f"{HIST_DIR}/kalshi_prices.parquet",
    eval_start: str = EVAL_START,
    eval_end: str = EVAL_END,
    n_estimators: int = 500,
    leads: list[int] | None = None,
) -> pd.DataFrame:
    """Train once, price every market at every lead. Returns one row per
    (market, lead) with fair/mid/outcome + segment labels."""
    from scripts.initial_train import train_models

    leads = leads or MULTILEAD_HOURS
    feats = pd.read_parquet(features_path)
    feats["date"] = pd.to_datetime(feats["date"])
    train_df = feats[feats["date"] < eval_start].copy()
    eval_feats = feats[(feats["date"] >= eval_start) & (feats["date"] <= eval_end)].copy()

    markets = _load_eval_markets(prices_path, eval_start, eval_end)
    logger.info("Training once (n_estimators=%d) on %d rows; %d markets to score",
                n_estimators, len(train_df), len(markets))
    bundle = train_models(train_df, n_estimators=n_estimators)

    rows: list[dict] = []
    for lead in leads:
        cache = _build_distribution_cache(bundle, eval_feats, lead_hour=lead)
        if not cache:
            logger.warning("Lead %dh: no eval rows, skip", lead)
            continue
        fn = build_fair_value_fn(bundle, eval_feats, cache=cache)
        n_priced = 0
        for m in markets.itertuples(index=False):
            ts = pd.Timestamp(m.date)
            yes = bracket_yes_prob(
                lambda x: fn(m.station, str(ts.date()), x),
                m.strike_type, m.floor_strike, m.cap_strike,
            )
            if yes is None:
                continue
            rows.append({
                "lead": lead,
                "station": str(m.station),
                "strike_type": m.strike_type,
                "month": ts.strftime("%Y-%m"),
                "date": str(ts.date()),
                "volume": pd.to_numeric(getattr(m, "volume", np.nan), errors="coerce"),
                "fair": float(yes),
                "mid": float(m.d1_mid),
                "outcome": float(m.settlement),
            })
            n_priced += 1
        logger.info("Lead %dh: priced %d markets", lead, n_priced)

    return pd.DataFrame(rows)


def _lead_pnl(df: pd.DataFrame, fee_coef: float, min_edge: float, min_price: float) -> dict:
    """Per-lead model/market Brier + gated P&L for one fee/floor config."""
    out = {}
    for lead, g in df.groupby("lead"):
        fair = g["fair"].to_numpy()
        mid = g["mid"].to_numpy()
        outc = g["outcome"].to_numpy()
        pnl, traded = per_trade_pnl(
            fair, mid, outc, min_edge=min_edge, fee_coef=fee_coef, min_price=min_price
        )
        n_tr = int(traded.sum())
        out[int(lead)] = {
            "n": len(g),
            "model_brier": brier_score(fair, outc),
            "market_brier": brier_score(mid, outc),
            "n_trades": n_tr,
            "pnl": float(pnl[traded].sum()),
            "win_rate": float((pnl[traded] > 0).mean()) if n_tr else 0.0,
        }
    return out


def _print_lead_table(title: str, res: dict) -> None:
    print(f"\n{title}")
    print(f"  {'lead':>5} {'n':>5} {'modelB':>8} {'mktB':>8} {'edge?':>6} {'trades':>7} {'P&L$':>9} {'win%':>6}")
    for lead, d in sorted(res.items()):
        edge = "YES" if d["model_brier"] < d["market_brier"] else "no"
        print(f"  {lead:>4}h {d['n']:>5} {d['model_brier']:>8.4f} {d['market_brier']:>8.4f} "
              f"{edge:>6} {d['n_trades']:>7} {d['pnl']:>9.2f} {d['win_rate']:>6.0%}")


def _print_segment(title: str, seg: dict) -> None:
    if not seg:
        return
    print(f"\n  By {title}:  n   modelB   mktB   edge?   trades   P&L$   win%")
    for key, d in seg.items():
        edge = "YES" if d["model_brier"] < d["market_brier"] else "no"
        print(f"    {key:<12} {d['n']:>5} {d['model_brier']:>7.4f} {d['market_brier']:>7.4f} "
              f"{edge:>5} {d['n_trades']:>7} {d.get('pnl', 0.0):>7.2f} {d.get('win_rate', 0.0):>5.0%}")


def report(df: pd.DataFrame, min_edge: float = 0.04) -> None:
    configs = [
        ("CORRECTED TAKER (7%·p·(1-p), no floor)", TAKER_FEE_COEF, 0.0),
        ("MAKER (1.75%·p·(1-p), no floor)", MAKER_FEE_COEF, 0.0),
        ("MAKER + $0.15 PRICE FLOOR", MAKER_FEE_COEF, 0.15),
        ("TAKER + $0.15 PRICE FLOOR", TAKER_FEE_COEF, 0.15),
    ]
    print("=" * 78)
    print("MULTI-LEAD FEE SCAN  (same markets/model, fee & floor varied — roadmap 1)")
    print("=" * 78)
    for title, coef, floor in configs:
        _print_lead_table(title, _lead_pnl(df, coef, min_edge, floor))

    # Segment mining (roadmap 2) under the friendliest execution config.
    print("\n" + "=" * 78)
    print("SEGMENT MINING (maker + $0.15 floor) — any cell with model<market Brier & +P&L?")
    print("=" * 78)
    fair = df["fair"].to_numpy(); mid = df["mid"].to_numpy(); outc = df["outcome"].to_numpy()
    pnl, traded = per_trade_pnl(fair, mid, outc, min_edge=min_edge,
                                fee_coef=MAKER_FEE_COEF, min_price=0.15)
    seg_args = (fair, mid, outc, pnl, traded)
    _print_segment("station", _segment_breakdown(df["station"].to_numpy(), *seg_args))
    _print_segment("strike_type", _segment_breakdown(
        df["strike_type"].to_numpy(), *seg_args, order=["greater", "less", "between"]))
    _print_segment("lead", _segment_breakdown(df["lead"].astype(str).to_numpy(), *seg_args))
    print("\n" + "=" * 78)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true", help="skip training, re-score saved parquet")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--min-edge", type=float, default=0.04)
    args = ap.parse_args()

    if args.from_cache:
        df = pd.read_parquet(SCORED_PATH)
        logger.info("Loaded %d scored rows from %s", len(df), SCORED_PATH)
    else:
        df = score_all_leads(n_estimators=args.n_estimators)
        df.to_parquet(SCORED_PATH, index=False)
        logger.info("Saved %d scored rows to %s", len(df), SCORED_PATH)

    report(df, min_edge=args.min_edge)


if __name__ == "__main__":
    main()
