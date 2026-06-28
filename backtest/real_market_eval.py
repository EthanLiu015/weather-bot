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


def _segment_breakdown(
    labels: np.ndarray,
    fair: np.ndarray,
    mids: np.ndarray,
    outs: np.ndarray,
    pnl: np.ndarray,
    traded: np.ndarray,
    order: Optional[list] = None,
) -> dict:
    """Per-segment model vs market Brier and P&L, keyed by `labels[i]`.

    A genuine edge, if any, hides in a segment (a specific station, a thin
    volume bucket, a particular month) rather than the aggregate — so each
    segment reports the same model-vs-market comparison the top line does.
    `order` fixes key iteration order; otherwise segments come out sorted.
    """
    out: dict[str, dict] = {}
    keys = order if order is not None else sorted(set(labels.tolist()))
    for key in keys:
        mask = labels == key
        if not mask.any():
            continue
        n_tr = int(traded[mask].sum())
        tr_pnl = pnl[mask][traded[mask]]
        out[str(key)] = {
            "n": int(mask.sum()),
            "model_brier": brier_score(fair[mask], outs[mask]),
            "market_brier": brier_score(mids[mask], outs[mask]),
            "n_trades": n_tr,
            "pnl": float(tr_pnl.sum()),
            "win_rate": float((tr_pnl > 0).sum() / n_tr) if n_tr else 0.0,
        }
    return out


def _volume_decile_labels(volumes: np.ndarray) -> np.ndarray:
    """Map each volume to a decile label D0 (thinnest) … D9 (most liquid).

    Markets are likely priced sharpest where they're liquid; the thin tail is
    where mispricings (our only realistic edge) would live, so bucketing by
    liquidity is a direct test of that hypothesis.
    """
    ranks = pd.Series(volumes).rank(method="first")
    deciles = np.clip(((ranks - 1) / len(volumes) * 10).astype(int), 0, 9)
    return np.array([f"D{d}" for d in deciles])


def _calib_stats(err: np.ndarray, sigma: np.ndarray) -> dict:
    """Calibration stats for residuals `err` (= actual - μ) under predicted
    std `sigma`. If σ is well-calibrated, z = err/σ ~ N(0,1): z_std ≈ 1,
    coverage_1sigma ≈ 0.683, coverage_2sigma ≈ 0.954. z_std > 1 (or coverage
    below those) means σ is too small (overconfident); < 1 underconfident."""
    err = np.asarray(err, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if err.size == 0:
        nan = float("nan")
        return {
            "n": 0, "z_mean": nan, "z_std": nan, "coverage_1sigma": nan,
            "coverage_2sigma": nan, "rmse": nan, "mae": nan, "mean_sigma": nan,
        }
    z = err / sigma
    return {
        "n": int(err.size),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "coverage_1sigma": float(np.mean(np.abs(z) <= 1.0)),
        "coverage_2sigma": float(np.mean(np.abs(z) <= 2.0)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mean_sigma": float(sigma.mean()),
    }


def sigma_calibration_report(
    err, sigma, group_labels: Optional[dict] = None
) -> dict:
    """σ-calibration of the predictive distribution (diagnostic #3).

    Compares the model's predicted σ to realized forecast error so we can tell
    whether the sigma-floor + spread-inflation (tuned against climatology) make
    the bracket probabilities over- or under-confident on real markets.

    Returns {"overall": stats, "by_<group>": {label: stats, ...}, ...}; stats
    are the ideal-1.0 z_std and ~0.683/0.954 coverages from `_calib_stats`.
    """
    err = np.asarray(err, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    out = {"overall": _calib_stats(err, sigma)}
    for gname, labels in (group_labels or {}).items():
        labels = np.asarray(labels)
        out[f"by_{gname}"] = {
            str(k): _calib_stats(err[labels == k], sigma[labels == k])
            for k in sorted(set(labels.tolist()))
        }
    return out


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
    stations: list[str] = []
    months: list[str] = []
    volumes: list[float] = []

    has_volume = "volume" in markets.columns

    for row in markets.itertuples(index=False):
        ts = pd.Timestamp(row.date)
        date_str = str(ts.date())
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
        stations.append(str(row.station))
        months.append(ts.strftime("%Y-%m"))
        if has_volume:
            volumes.append(pd.to_numeric(row.volume, errors="coerce"))

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

    # Segment breakdowns — any genuine edge hides in a segment, not the
    # aggregate. by_strike_type: the 2°-wide "between" brackets are far more
    # sensitive to the station/source mismatch than the "greater"/"less" tails;
    # by_station/by_volume_decile/by_month hunt for thin-market, station-specific
    # or seasonal mispricings.
    seg_args = (fair_arr, mid_arr, out_arr, pnl, traded)
    by_strike_type = _segment_breakdown(
        np.array(strike_types), *seg_args, order=["greater", "less", "between"]
    )
    by_station = _segment_breakdown(np.array(stations), *seg_args)
    by_month = _segment_breakdown(np.array(months), *seg_args)

    by_volume_decile: dict[str, dict] = {}
    if has_volume:
        vol_arr = np.array(volumes, dtype=float)
        if np.isfinite(vol_arr).any():
            decile_labels = _volume_decile_labels(vol_arr)
            by_volume_decile = _segment_breakdown(
                decile_labels, *seg_args, order=[f"D{i}" for i in range(10)]
            )

    return {
        "num_scored_markets": n,
        "by_strike_type": by_strike_type,
        "by_station": by_station,
        "by_month": by_month,
        "by_volume_decile": by_volume_decile,
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

def _build_distribution_cache(bundle: dict, eval_features: pd.DataFrame) -> dict:
    """Per (station, date) cache of the threshold-INDEPENDENT predictive
    distribution (μ, σ, ECMWF offset, QRF quantiles, calibrator), using the
    SHORTEST available lead per (station, date) — the freshest forecast the bot
    would use (~24h out). Computed ONCE per station-bucket (≈60 forest evals,
    not ~8k), since μ/σ/quantiles don't depend on the bracket threshold.

    Each entry also carries the `lead_hour`/`lead_bucket` used, so the σ being
    cached can be checked against realized error by lead (diagnostic #3).
    """
    from processing.features import get_feature_columns
    from processing.bias_correction import get_lead_bucket
    from models.spread_inflation import apply_spread_inflation_from_stats
    from strategies.ensemble_strategy import RESIDUAL_SIGMA_FLOOR

    avail_cols = [c for c in get_feature_columns() if c in eval_features.columns]

    ef = eval_features.copy()
    ef["date"] = pd.to_datetime(ef["date"])
    ef = ef.sort_values("lead_hour").drop_duplicates(["station", "date"], keep="first")
    ef["_date_str"] = ef["date"].dt.strftime("%Y-%m-%d")
    ef["_bucket"] = ef["lead_hour"].astype(int).map(get_lead_bucket)
    ef["_key"] = ef["station"].astype(str) + "_" + ef["_bucket"]

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
        lead_hours = grp["lead_hour"].astype(int).tolist()
        buckets = grp["_bucket"].tolist()

        for i, date_str in enumerate(grp["_date_str"].tolist()):
            cache[(station, date_str)] = {
                "mu": float(mu[i]),
                "sigma": float(sigma[i]),
                "ecmwf_offset": float(ecmwf_offset[i]),
                "q_vals": q_vals[i] if q_vals is not None else None,
                "calibrator": calibrator,
                "lead_hour": lead_hours[i],
                "lead_bucket": buckets[i],
            }

    return cache


def build_fair_value_fn(
    bundle: dict, eval_features: pd.DataFrame, cache: Optional[dict] = None
) -> ProbAboveFn:
    """Return prob_above(station, date, x) = calibrated model P(high > x), using
    the in-memory model bundle and the eval-window feature rows.

    Only the cheap threshold-dependent tail (normal CDF + QRF interpolation +
    blend + calibration + clamp) runs per market; the threshold-independent
    distribution is cached once per (station, date) by _build_distribution_cache.
    Pass a precomputed `cache` to avoid rebuilding it (e.g. when the caller also
    needs μ/σ for the σ-calibration check). A parity test pins this against
    EnsembleStrategy._compute_fair_value.
    """
    from scipy.interpolate import interp1d
    from models.qrf_model import DEFAULT_QUANTILES
    from strategies.ensemble_strategy import (
        FAIR_VALUE_FLOOR,
        FAIR_VALUE_CEIL,
        _scipy_norm,
    )

    weights = bundle["blender"].weights
    if cache is None:
        cache = _build_distribution_cache(bundle, eval_features)

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

    cache = _build_distribution_cache(bundle, eval_feats)
    prob_above_fn = build_fair_value_fn(bundle, eval_feats, cache=cache)
    result = evaluate_real_markets(markets, prob_above_fn, min_edge=min_edge)

    result["sigma_calibration"] = compute_sigma_calibration(cache, eval_feats)

    result["eval_start"] = eval_start
    result["eval_end"] = eval_end
    result["min_edge"] = min_edge
    return result


def compute_sigma_calibration(cache: dict, eval_features: pd.DataFrame) -> dict:
    """σ-calibration on the eval window (diagnostic #3): for each cached
    (station, date) predictive distribution, compare predicted σ to the realized
    error (actual_tmax − μ), broken down by station and lead bucket.

    Uses the SHORTEST lead per (station, date) — the same row the cache and the
    bot's decision use — so predicted σ and realized error are aligned."""
    ef = eval_features.copy()
    ef["date"] = pd.to_datetime(ef["date"])
    ef = ef.sort_values("lead_hour").drop_duplicates(["station", "date"], keep="first")
    date_strs = ef["date"].dt.strftime("%Y-%m-%d")
    actual_by_key = {
        (str(st), ds): float(act)
        for st, ds, act in zip(ef["station"], date_strs, ef["actual_tmax"])
        if pd.notna(act)
    }

    errs, sigmas, stations, buckets = [], [], [], []
    for (station, date_str), c in cache.items():
        actual = actual_by_key.get((station, date_str))
        if actual is None:
            continue
        errs.append(float(actual) - c["mu"])
        sigmas.append(c["sigma"])
        stations.append(station)
        buckets.append(c["lead_bucket"])

    return sigma_calibration_report(
        np.array(errs), np.array(sigmas),
        group_labels={"station": np.array(stations), "lead_bucket": np.array(buckets)},
    )


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
    def _print_segments(title: str, seg: dict, width: int = 10) -> None:
        if not seg:
            return
        print(f"  By {title:<14}  n   model B   market B  edge?    P&L   win%")
        for key, d in seg.items():
            edge = "YES" if d["model_brier"] < d["market_brier"] else "no"
            print(
                f"    {key:<{width}} {d['n']:>5}   {d['model_brier']:.4f}   {d['market_brier']:.4f}   "
                f"{edge:<4}  {d.get('pnl', 0.0):>7.2f}  {d.get('win_rate', 0.0):>4.0%}"
            )
        print("-" * 60)

    _print_segments("strike_type", result.get("by_strike_type", {}))
    _print_segments("station", result.get("by_station", {}))
    _print_segments("volume decile", result.get("by_volume_decile", {}))
    _print_segments("month", result.get("by_month", {}))

    sig = result.get("sigma_calibration")
    if sig:
        o = sig["overall"]
        print("  σ-CALIBRATION (ideal: z_std≈1.0, cov1σ≈0.683, cov2σ≈0.954)")
        print(f"    overall    n={o['n']}  z_mean={o['z_mean']:+.2f}  z_std={o['z_std']:.2f}  "
              f"cov1σ={o['coverage_1sigma']:.2f}  cov2σ={o['coverage_2sigma']:.2f}  "
              f"mae={o['mae']:.2f}  mean_σ={o['mean_sigma']:.2f}")
        verdict = (
            "OVERCONFIDENT (σ too small)" if o["z_std"] > 1.1
            else "UNDERCONFIDENT (σ too large)" if o["z_std"] < 0.9
            else "well-calibrated"
        )
        print(f"    verdict:   {verdict}")
        for grp in ("by_lead_bucket", "by_station"):
            print(f"    {grp}:")
            for key, s in sig.get(grp, {}).items():
                print(f"      {key:<10} n={s['n']:>4}  z_std={s['z_std']:.2f}  "
                      f"cov1σ={s['coverage_1sigma']:.2f}  mae={s['mae']:.2f}  mean_σ={s['mean_sigma']:.2f}")
        print("-" * 60)
    print("=" * 60)


if __name__ == "__main__":
    main()
