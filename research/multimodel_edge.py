"""Multi-model disagreement edge test (roadmap 3).

New information the current model never saw: per-station daily-high forecasts from
AIFS (AI), ECMWF IFS, ICON, GFS, GraphCast, at the 24h decision lead. Two questions:

  A. SKILL — does the cross-model ensemble MEAN, priced into brackets with an
     empirically-fit error σ, beat the Kalshi book's Brier? (v1/v2 said our own
     model can't.)
  B. DISAGREEMENT FILTER — is the ensemble's edge concentrated where the models
     AGREE (low cross-model spread = high confidence)? Bucket by spread and read
     per-bucket Brier + fee-aware gated P&L.

No look-ahead: the error σ is fit on a temporal TRAIN split of eval dates; the
AIFS/etc forecasts are the Previous Runs archive (as-issued 24h out); outcomes are
real Kalshi settlements.

    PYTHONPATH=. python -m research.multimodel_edge
Data: data/historical/openmeteo_multimodel.parquet, kalshi_prices.parquet, features.parquet
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strategies.bracket_pricing import bracket_yes_prob
from backtest.real_market_eval import brier_score, per_trade_pnl, _load_eval_markets, EVAL_START, EVAL_END
from backtest.track_b import MAKER_FEE_COEF

logger = logging.getLogger(__name__)

MM_PATH = "data/historical/openmeteo_multimodel.parquet"
FEAT_PATH = "data/historical/features.parquet"
DECISION_LEAD = 24
PHYSICS = ["ecmwf", "icon", "gfs"]


def ensemble_stats(mm: pd.DataFrame, lead_hour: int = DECISION_LEAD) -> pd.DataFrame:
    """Per (station, date) cross-model ensemble features at one lead (°F).

    Returns mean/std/min/max over ALL models, the AIFS point forecast, and the
    signed AIFS-minus-physics gap (the AI-vs-physics disagreement)."""
    d = mm[mm.lead_hour == lead_hour]
    wide = d.pivot_table(index=["station", "date"], columns="model", values="tmax_f")
    models = [c for c in wide.columns]
    out = pd.DataFrame({
        "ens_mean": wide[models].mean(axis=1),
        "ens_std": wide[models].std(axis=1),
        "ens_min": wide[models].min(axis=1),
        "ens_max": wide[models].max(axis=1),
        "n_models": wide[models].notna().sum(axis=1),
        "aifs": wide["aifs"] if "aifs" in wide else np.nan,
    })
    phys = [m for m in PHYSICS if m in wide.columns]
    out["aifs_minus_phys"] = wide["aifs"] - wide[phys].mean(axis=1) if ("aifs" in wide and phys) else np.nan
    return out.reset_index()


def _price(mean: float, sigma: float, row) -> float | None:
    from scipy.stats import norm
    s = max(sigma, 1e-6)
    return bracket_yes_prob(
        lambda x: float(1.0 - norm.cdf(x, loc=mean, scale=s)),
        row.strike_type, row.floor_strike, row.cap_strike,
    )


def evaluate(train_frac: float = 0.5, min_edge: float = 0.04) -> None:
    mm = pd.read_parquet(MM_PATH)
    ens = ensemble_stats(mm)
    ens["date"] = pd.to_datetime(ens["date"])

    markets = _load_eval_markets(f"data/historical/kalshi_prices.parquet", EVAL_START, EVAL_END)
    markets["date"] = pd.to_datetime(markets["date"])

    feats = pd.read_parquet(FEAT_PATH)[["station", "date", "actual_tmax"]].dropna()
    feats["date"] = pd.to_datetime(feats["date"])
    actual = feats.drop_duplicates(["station", "date"])

    df = markets.merge(ens, on=["station", "date"], how="inner").merge(
        actual, on=["station", "date"], how="left")
    df = df[df["ens_mean"].notna()]
    logger.info("Matched %d markets to multi-model forecasts", len(df))

    # temporal split; fit a single error σ on train (actual − ens_mean).
    dates = np.sort(df["date"].unique())
    split = dates[int(len(dates) * train_frac)]
    train, test = df[df["date"] < split], df[df["date"] >= split]
    err = (train["actual_tmax"] - train["ens_mean"]).dropna()
    sigma = float(err.std())
    bias = float(err.mean())
    mae = float(err.abs().mean())
    logger.info("Ensemble train: bias=%.2f°F  MAE=%.2f°F  σ=%.2f°F (n=%d)", bias, mae, sigma, len(err))

    fair, mids, outs, spreads, aifs_gap = [], [], [], [], []
    for row in test.itertuples(index=False):
        yes = _price(row.ens_mean + bias, sigma, row)
        if yes is None:
            continue
        fair.append(float(yes)); mids.append(float(row.d1_mid)); outs.append(float(row.settlement))
        spreads.append(float(row.ens_std)); aifs_gap.append(float(row.aifs_minus_phys) if np.isfinite(row.aifs_minus_phys) else np.nan)

    fair, mids, outs = np.array(fair), np.array(mids), np.array(outs)
    spreads = np.array(spreads)
    pnl, traded = per_trade_pnl(fair, mids, outs, min_edge=min_edge, fee_coef=MAKER_FEE_COEF, min_price=0.15)

    print("\n" + "=" * 72)
    print("MULTI-MODEL ENSEMBLE EDGE @24h  (bias-corrected mean ± fitted σ, maker+floor)")
    print("=" * 72)
    print(f"  test markets: {len(fair)}   ensemble MAE(train): {mae:.2f}°F   σ: {sigma:.2f}°F")
    print(f"  A. SKILL:  model Brier {brier_score(fair, outs):.4f}   "
          f"market Brier {brier_score(mids, outs):.4f}   "
          f"{'BEATS BOOK' if brier_score(fair,outs) < brier_score(mids,outs) else 'no edge'}")
    print(f"     gated P&L ${pnl[traded].sum():.2f} on {int(traded.sum())} trades "
          f"(win {100*(pnl[traded]>0).mean():.0f}%)")

    print("\n  B. DISAGREEMENT FILTER — by cross-model spread tercile:")
    print(f"  {'spread':>10} {'n':>5} {'modelB':>8} {'mktB':>8} {'edge?':>6} {'trades':>7} {'P&L$':>8}")
    terc = np.asarray(pd.qcut(spreads, 3, labels=["low(agree)", "mid", "high(disagree)"]))
    for lab in ["low(agree)", "mid", "high(disagree)"]:
        m = terc == lab
        if not m.any():
            continue
        mb, kb = brier_score(fair[m], outs[m]), brier_score(mids[m], outs[m])
        edge = "YES" if mb < kb else "no"
        print(f"  {lab:>10} {m.sum():>5} {mb:>8.4f} {kb:>8.4f} {edge:>6} "
              f"{int(traded[m].sum()):>7} {pnl[m & traded].sum():>8.2f}")
    print("=" * 72)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate()


if __name__ == "__main__":
    main()
