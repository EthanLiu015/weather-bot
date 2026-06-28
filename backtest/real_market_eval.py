"""Real-markets evaluation harness.

The first trustworthy real-price evaluation: score the strategy against the
ACTUAL Kalshi markets at their real thresholds and binary settlements, rather
than the synthetic station×month-median markets the climatology backtest uses.

Design (see handoff.md):
  * Models are trained ONLY on data dated before the eval window, so they never
    see an eval-window outcome (no look-ahead).
  * For each real market we compute the model's P(temp > threshold) at that
    market's EXACT threshold, convert to the market's YES probability via its
    above/below type, and compare to the real decision-time price (d1_mid).
  * P&L is settled against the real binary settlement with the 5% Kalshi fee.
  * Verdict: model Brier vs the market's own Brier (~0.068). If the model's
    Brier is not lower, there is no edge.

The pricing/edge math reuses backtest.track_b.simulate_pnl so the dollar terms
match the climatology backtest exactly.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest.track_b import FEE_RATE

logger = logging.getLogger(__name__)

# The market's own forecasting skill, measured on the genuine candlestick prices
# (corr(price, settlement)=+0.71). A model only has edge if it beats this.
MARKET_BRIER_BENCHMARK = 0.068

# (station, date, threshold) -> calibrated P(temp > threshold), or None when the
# model cannot price that (station,date) (no feature row / untrained bucket).
ProbAboveFn = Callable[[str, str, float], Optional[float]]

# NWS daily highs are reported in whole °F, so a market boundary "high > 84"
# means "high ≥ 85"; the decision boundary on the continuous forecast sits at
# 84.5. These ±0.5 continuity corrections convert integer strike rules into
# thresholds for the continuous predictive distribution.
_HALF = 0.5


def bracket_yes_prob(
    prob_above: Callable[[float], Optional[float]],
    strike_type: str,
    floor_strike: Optional[float],
    cap_strike: Optional[float],
) -> Optional[float]:
    """Model's YES probability for a real Kalshi temperature bracket.

    Kalshi temperature markets are mutually-exclusive brackets:
      * greater (>F):       YES = P(high > F)        = prob_above(F + 0.5)
      * less   (<C):        YES = P(high < C)        = 1 - prob_above(C - 0.5)
      * between [F, C]:      YES = P(F ≤ high ≤ C)    = prob_above(F-0.5) - prob_above(C+0.5)

    `prob_above(x)` is the model's calibrated P(high > x); returns None when
    unpriceable (propagated as None so the market is skipped).
    """
    if strike_type == "greater":
        return prob_above(float(floor_strike) + _HALF)
    if strike_type == "less":
        p = prob_above(float(cap_strike) - _HALF)
        return None if p is None else 1.0 - p
    if strike_type == "between":
        lo = prob_above(float(floor_strike) - _HALF)
        hi = prob_above(float(cap_strike) + _HALF)
        if lo is None or hi is None:
            return None
        return max(0.0, lo - hi)
    raise ValueError(f"Unknown strike_type: {strike_type!r}")


def brier_score(probs, outcomes) -> float:
    """Mean squared error between forecast probabilities and binary outcomes."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if probs.size == 0:
        return float("nan")
    return float(np.mean((probs - outcomes) ** 2))


def per_trade_pnl(
    model_probs,
    market_mids,
    outcomes,
    min_edge: float = 0.04,
    contract_usd: float = 1.0,
    contract_sizes: Optional[np.ndarray] = None,
):
    """Per-market P&L, mirroring backtest.track_b.simulate_pnl row-by-row.

    Returns (pnl, traded): pnl[i] is the dollar P&L for market i (0 when the edge
    is below min_edge and no trade is taken); traded[i] flags whether a trade was
    taken. Summing pnl over traded rows reproduces simulate_pnl's aggregate — a
    parity test pins this. Exposed separately so daily P&L (hence Sharpe) can be
    grouped by resolution date.
    """
    probs = np.asarray(model_probs, dtype=float)
    mids = np.asarray(market_mids, dtype=float)
    outs = np.asarray(outcomes, dtype=float)
    n = probs.size
    pnl = np.zeros(n)
    traded = np.zeros(n, dtype=bool)

    for i in range(n):
        prob, mid, outcome = probs[i], mids[i], outs[i]
        if abs(prob - mid) < min_edge:
            continue
        size = float(contract_sizes[i]) if contract_sizes is not None else contract_usd
        if prob > mid:
            p = size * (outcome - mid)
        else:
            no_mid = 1.0 - mid
            p = size * ((1.0 - outcome) - no_mid)
        p -= FEE_RATE * size * mid
        pnl[i] = p
        traded[i] = True

    return pnl, traded


def annualized_sharpe(daily_pnl, periods_per_year: int = 252) -> float:
    """Annualised Sharpe of a daily P&L series. NaN if <2 days or zero variance."""
    arr = np.asarray(daily_pnl, dtype=float)
    if arr.size < 2:
        return float("nan")
    sd = arr.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(arr.mean() / sd * np.sqrt(periods_per_year))


def evaluate_real_markets(
    markets: pd.DataFrame,
    prob_above_fn: ProbAboveFn,
    min_edge: float = 0.04,
    contract_usd: float = 1.0,
    contract_sizes: Optional[np.ndarray] = None,
) -> dict:
    """Score the model's fair values against real Kalshi bracket markets.

    Args:
        markets: rows with columns station, date, strike_type, floor_strike,
            cap_strike, d1_mid (real decision-time YES price), settlement.
        prob_above_fn: (station, date, x) -> calibrated P(high > x), or None when
            that (station,date) cannot be priced (the market is then skipped).
        min_edge: minimum |fair_yes - d1_mid| to take a trade.
        contract_usd / contract_sizes: flat or Kelly-sized P&L.

    Returns aggregate metrics including model_brier vs market_brier and has_edge.
    """
    fair_yes: list[float] = []
    mids: list[float] = []
    outcomes: list[float] = []
    dates: list[str] = []
    strike_types: list[str] = []

    for row in markets.itertuples(index=False):
        date_str = str(pd.Timestamp(row.date).date())
        yes = bracket_yes_prob(
            lambda x: prob_above_fn(row.station, date_str, x),
            row.strike_type,
            row.floor_strike,
            row.cap_strike,
        )
        if yes is None:
            continue
        fair_yes.append(float(yes))
        mids.append(float(row.d1_mid))
        outcomes.append(float(row.settlement))
        dates.append(date_str)
        strike_types.append(row.strike_type)

    n = len(fair_yes)
    if n == 0:
        return {
            "num_scored_markets": 0,
            "num_simulated_trades": 0,
            "simulated_pnl_usd": 0.0,
            "win_rate": 0.0,
            "mean_edge": 0.0,
            "model_brier": float("nan"),
            "market_brier": float("nan"),
            "market_brier_benchmark": MARKET_BRIER_BENCHMARK,
            "has_edge": False,
            "daily_sharpe": float("nan"),
        }

    fair_arr = np.array(fair_yes)
    mid_arr = np.array(mids)
    out_arr = np.array(outcomes)

    pnl, traded = per_trade_pnl(
        fair_arr, mid_arr, out_arr,
        min_edge=min_edge, contract_usd=contract_usd, contract_sizes=contract_sizes,
    )

    num_trades = int(traded.sum())
    traded_pnl = pnl[traded]
    edges = np.abs(fair_arr - mid_arr)[traded]

    # Daily P&L → Sharpe. Markets settle on their resolution date, so summing
    # per-market P&L by date gives the strategy's daily return stream.
    daily = (
        pd.DataFrame({"date": dates, "pnl": pnl})
        .groupby("date")["pnl"].sum().values
    )

    model_brier = brier_score(fair_arr, out_arr)
    market_brier = brier_score(mid_arr, out_arr)

    # Breakdown by strike_type — the 2°-wide "between" brackets are far more
    # sensitive to the station/source mismatch than the "greater"/"less" tails,
    # so any genuine edge is most likely to surface in the tails.
    st_arr = np.array(strike_types)
    by_strike_type: dict[str, dict] = {}
    for st in ("greater", "less", "between"):
        mask = st_arr == st
        if not mask.any():
            continue
        by_strike_type[st] = {
            "n": int(mask.sum()),
            "model_brier": brier_score(fair_arr[mask], out_arr[mask]),
            "market_brier": brier_score(mid_arr[mask], out_arr[mask]),
        }

    return {
        "num_scored_markets": n,
        "by_strike_type": by_strike_type,
        "num_simulated_trades": num_trades,
        "simulated_pnl_usd": float(traded_pnl.sum()),
        "win_rate": float((traded_pnl > 0).sum() / num_trades) if num_trades else 0.0,
        "mean_edge": float(edges.mean()) if num_trades else 0.0,
        "model_brier": model_brier,
        "market_brier": market_brier,
        "market_brier_benchmark": MARKET_BRIER_BENCHMARK,
        "has_edge": model_brier < market_brier,
        "daily_sharpe": annualized_sharpe(daily),
    }


# ---------------------------------------------------------------------------
# Model fair-value closure (wires a trained bundle to real-market lookups)
# ---------------------------------------------------------------------------

def build_fair_value_fn(bundle: dict, eval_features: pd.DataFrame) -> ProbAboveFn:
    """Return prob_above(station, date, x) = calibrated model P(high > x), using
    the in-memory model bundle and the eval-window feature rows.

    For each (station, date) we use the SHORTEST available lead-hour feature row
    (the bot's decision is closest to resolution, ~24h out, where forecasts are
    sharpest).

    Performance: the NGBoost μ/σ, residual correction, spread inflation and QRF
    quantiles are all threshold-INDEPENDENT, so they are computed ONCE per
    (station, date) — batched per station-bucket (≈60 forest evaluations total)
    rather than once per market (~8k). Only the cheap threshold-dependent tail
    (normal CDF + QRF interpolation + blend + calibration + clamp) runs per
    market. A parity test pins this against EnsembleStrategy._compute_fair_value.
    """
    from scipy.interpolate import interp1d
    from processing.features import get_feature_columns
    from processing.bias_correction import get_lead_bucket
    from models.spread_inflation import apply_spread_inflation_from_stats
    from models.qrf_model import DEFAULT_QUANTILES
    from strategies.ensemble_strategy import (
        RESIDUAL_SIGMA_FLOOR,
        FAIR_VALUE_FLOOR,
        FAIR_VALUE_CEIL,
        _scipy_norm,
    )

    avail_cols = [c for c in get_feature_columns() if c in eval_features.columns]
    weights = bundle["blender"].weights

    ef = eval_features.copy()
    ef["date"] = pd.to_datetime(ef["date"])
    # Shortest lead per (station, date) — the freshest forecast the bot would use.
    ef = ef.sort_values("lead_hour").drop_duplicates(["station", "date"], keep="first")
    ef["_date_str"] = ef["date"].dt.strftime("%Y-%m-%d")
    ef["_bucket"] = ef["lead_hour"].astype(int).map(get_lead_bucket)
    ef["_key"] = ef["station"].astype(str) + "_" + ef["_bucket"]

    # Per (station, date) cache of the threshold-independent distribution.
    cache: dict[tuple[str, str], dict] = {}

    for (station, model_key), grp in ef.groupby(["station", "_key"]):
        ngb = bundle["ngboost"].get(model_key)
        if ngb is None:
            continue  # station-bucket not trained (too few pre-window rows)
        qrf = bundle["qrf"].get(model_key)
        residual = bundle["residual"].get(station)
        calibrator = bundle["calibrator"].get(model_key)

        X = grp[avail_cols].fillna(0.0)
        ecmwf_offset = grp["ecmwf_tmax"].fillna(0.0).to_numpy(dtype=float)

        mu, sigma = ngb.predict_distribution(X)
        if residual is not None:
            try:
                mu = mu + residual.predict(X)
            except Exception:
                pass
        mu = mu + ecmwf_offset

        if "gefs_tmax_std" in X.columns and "gefs_tmax_range" in X.columns:
            _, sigma = apply_spread_inflation_from_stats(
                mu, sigma,
                X["gefs_tmax_std"].fillna(0).values,
                X["gefs_tmax_range"].fillna(0).values,
            )
        sigma = np.maximum(sigma, RESIDUAL_SIGMA_FLOOR)

        q_vals = qrf.predict_quantiles(X).to_numpy() if qrf is not None else None

        for i, date_str in enumerate(grp["_date_str"].tolist()):
            cache[(station, date_str)] = {
                "mu": float(mu[i]),
                "sigma": float(sigma[i]),
                "ecmwf_offset": float(ecmwf_offset[i]),
                "q_vals": q_vals[i] if q_vals is not None else None,
                "calibrator": calibrator,
            }

    def prob_above(station: str, date_str: str, threshold: float):
        c = cache.get((station, date_str))
        if c is None:
            return None
        raw_prob = float(
            1.0 - _scipy_norm.cdf(threshold, loc=c["mu"], scale=max(c["sigma"], 0.01))
        )
        if c["q_vals"] is not None:
            try:
                interp = interp1d(
                    c["q_vals"], DEFAULT_QUANTILES,
                    kind="linear", bounds_error=False, fill_value=(0.0, 1.0),
                )
                qrf_prob = float(1.0 - interp(threshold - c["ecmwf_offset"]))
                raw_prob = weights["ngboost"] * raw_prob + weights["qrf"] * qrf_prob
            except Exception:
                pass
        cal = c["calibrator"]
        if cal is not None and cal._iso is not None:
            # Use the isotonic prediction directly. This is exactly the cal_prob
            # production's calibrate() returns; we skip its bootstrap_ci (1000
            # isotonic refits per call) because the eval never uses ci_width and
            # the harness prices ~8k bracket boundaries.
            cal_prob = float(cal._iso.predict([raw_prob])[0])
        else:
            cal_prob = raw_prob
        return min(FAIR_VALUE_CEIL, max(FAIR_VALUE_FLOOR, float(cal_prob)))

    return prob_above


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

HIST_DIR = "data/historical"
EVAL_START = "2026-04-11"
EVAL_END = "2026-05-27"


def run_evaluation(
    features_path: str = f"{HIST_DIR}/features.parquet",
    prices_path: str = f"{HIST_DIR}/kalshi_prices.parquet",
    eval_start: str = EVAL_START,
    eval_end: str = EVAL_END,
    min_edge: float = 0.04,
    n_estimators: int = 500,
) -> dict:
    """Train look-ahead-free models on data before `eval_start`, then score them
    against the real Kalshi markets settling in [eval_start, eval_end]."""
    from scripts.initial_train import train_models

    feats = pd.read_parquet(features_path)
    feats["date"] = pd.to_datetime(feats["date"])
    train_df = feats[feats["date"] < eval_start].copy()
    eval_feats = feats[(feats["date"] >= eval_start) & (feats["date"] <= eval_end)].copy()
    logger.info("Train rows (< %s): %d | eval-window feature rows: %d",
                eval_start, len(train_df), len(eval_feats))

    from config.series import is_low_temp_series

    prices = pd.read_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    # The bot trades HIGH-temp (tmax) markets only — production skips low-temp
    # (overnight-minimum) series entirely (EnsembleStrategy._get_active_tickers),
    # and our models predict actual_tmax. Scoring low-temp markets with a tmax
    # model is meaningless, so exclude them to match the live trading scope.
    not_low = ~prices["series"].map(is_low_temp_series)
    markets = prices[
        not_low
        & (prices["date"] >= eval_start)
        & (prices["date"] <= eval_end)
        & prices["strike_type"].notna()
        & prices["settlement"].notna()
        & prices["d1_mid"].notna()
        & (prices["d1_mid"] > 0.0)
        & (prices["d1_mid"] < 1.0)
    ].copy()
    logger.info("Real HIGH-temp bracket markets in window with settlement+price: %d", len(markets))

    logger.info("Training look-ahead-free models (n_estimators=%d)...", n_estimators)
    bundle = train_models(train_df, n_estimators=n_estimators)

    prob_above_fn = build_fair_value_fn(bundle, eval_feats)
    result = evaluate_real_markets(markets, prob_above_fn, min_edge=min_edge)

    result["eval_start"] = eval_start
    result["eval_end"] = eval_end
    result["min_edge"] = min_edge
    return result


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Real-markets evaluation harness")
    parser.add_argument("--eval-start", default=EVAL_START)
    parser.add_argument("--eval-end", default=EVAL_END)
    parser.add_argument("--min-edge", type=float, default=0.04)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()

    result = run_evaluation(
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        min_edge=args.min_edge,
        n_estimators=args.n_estimators,
    )

    print("\n" + "=" * 60)
    print("REAL-MARKETS EVALUATION  (no look-ahead)")
    print("=" * 60)
    print(f"  Window:            {result['eval_start']} → {result['eval_end']}")
    print(f"  Scored markets:    {result['num_scored_markets']}")
    print(f"  Simulated trades:  {result['num_simulated_trades']}")
    print(f"  P&L (flat $1):     ${result['simulated_pnl_usd']:.2f}")
    print(f"  Win rate:          {result['win_rate']:.1%}")
    print(f"  Mean edge:         {result['mean_edge']:.3f}")
    print(f"  Daily Sharpe:      {result['daily_sharpe']:.2f}")
    print("-" * 60)
    print(f"  Model Brier:       {result['model_brier']:.4f}")
    print(f"  Market Brier:      {result['market_brier']:.4f}  (benchmark {result['market_brier_benchmark']})")
    verdict = "EDGE: model beats market" if result["has_edge"] else "NO EDGE: model does not beat market"
    print(f"  VERDICT:           {verdict}")
    print("-" * 60)
    print("  By strike_type     n     model Brier   market Brier   edge?")
    for st, d in result.get("by_strike_type", {}).items():
        edge = "YES" if d["model_brier"] < d["market_brier"] else "no"
        print(f"    {st:<8}     {d['n']:>5}      {d['model_brier']:.4f}        {d['market_brier']:.4f}       {edge}")
    print("=" * 60)


if __name__ == "__main__":
    main()
