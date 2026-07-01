"""Tests for the real-markets evaluation harness.

Scores the strategy against ACTUAL Kalshi temperature markets, which are
mutually-exclusive °F BRACKETS (strike_type greater/less/between), not above/
below contracts. No synthetic thresholds, no look-ahead. See handoff.md.
"""

import math

import pandas as pd
import pytest

import numpy as np

from backtest.real_market_eval import (
    bracket_yes_prob,
    brier_score,
    per_trade_pnl,
    annualized_sharpe,
    evaluate_real_markets,
    sigma_calibration_report,
)
from backtest.track_b import simulate_pnl


# ── Bracket pricing: strike_type → YES probability ───────────────────────────
# prob_above(x) is the model's P(high > x). Highs are integer °F, so a ">F" rule
# decides at F+0.5 (continuity correction).

def test_greater_bracket_uses_floor_plus_half():
    seen = {}
    def pa(x):
        seen["x"] = x
        return 0.7
    assert bracket_yes_prob(pa, "greater", floor_strike=84, cap_strike=None) == pytest.approx(0.7)
    assert seen["x"] == pytest.approx(84.5)


def test_less_bracket_is_complement_at_cap_minus_half():
    seen = {}
    def pa(x):
        seen["x"] = x
        return 0.7  # P(high > 76.5)
    # "76° or below" (cap 77): YES = 1 - P(high > 76.5) = 0.3
    assert bracket_yes_prob(pa, "less", floor_strike=None, cap_strike=77) == pytest.approx(0.3)
    assert seen["x"] == pytest.approx(76.5)


def test_between_bracket_is_cdf_difference():
    # between [83, 84]: YES = P(high>82.5) - P(high>84.5).
    probs = {82.5: 0.8, 84.5: 0.3}
    assert bracket_yes_prob(lambda x: probs[x], "between", 83, 84) == pytest.approx(0.5)


def test_between_bracket_clamped_non_negative():
    # Numerical noise could make hi > lo slightly; result must not go negative.
    assert bracket_yes_prob(lambda x: 0.5 if x < 0 else 0.50001, "between", 0, 1) == 0.0


def test_bracket_returns_none_when_prob_unavailable():
    assert bracket_yes_prob(lambda x: None, "greater", 84, None) is None


def test_unknown_strike_type_raises():
    with pytest.raises(ValueError):
        bracket_yes_prob(lambda x: 0.5, "sideways", 1, 2)


# ── Brier score ──────────────────────────────────────────────────────────────

def test_brier_score_perfect_forecast_is_zero():
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_brier_score_constant_half_against_balanced_outcomes():
    assert brier_score([0.5, 0.5], [0.0, 1.0]) == pytest.approx(0.25)


def test_brier_score_empty_is_nan():
    assert math.isnan(brier_score([], []))


# ── per_trade_pnl parity with the canonical simulate_pnl ─────────────────────

def test_per_trade_pnl_aggregate_matches_simulate_pnl():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.05, 0.95, 50)
    mids = rng.uniform(0.05, 0.95, 50)
    outcomes = rng.integers(0, 2, 50).astype(float)

    pnl, traded = per_trade_pnl(probs, mids, outcomes, min_edge=0.04)
    ref = simulate_pnl(probs, mids, outcomes, min_edge=0.04)

    assert int(traded.sum()) == ref["num_simulated_trades"]
    assert pnl[traded].sum() == pytest.approx(ref["simulated_pnl_usd"])


def test_per_trade_pnl_zeroes_untraded_rows():
    pnl, traded = per_trade_pnl([0.51], [0.50], [1.0], min_edge=0.04)
    assert traded.tolist() == [False]
    assert pnl.tolist() == [0.0]


# ── Sharpe ────────────────────────────────────────────────────────────────

def test_annualized_sharpe_positive_for_net_positive_pnl():
    assert annualized_sharpe([2.0, 1.0, 3.0, 2.0]) > 0


def test_annualized_sharpe_nan_for_zero_variance():
    assert math.isnan(annualized_sharpe([1.0, 1.0, 1.0]))


def test_annualized_sharpe_nan_for_single_day():
    assert math.isnan(annualized_sharpe([1.0]))


# ── evaluate_real_markets: end-to-end on synthetic brackets + fake model ──────

def _markets(rows):
    return pd.DataFrame(
        rows,
        columns=["station", "date", "strike_type", "floor_strike",
                 "cap_strike", "d1_mid", "settlement"],
    )


def test_evaluate_skips_markets_with_no_model_fair_value():
    markets = _markets([
        ("KORD", "2026-04-12", "greater", 80.0, None, 0.40, 1.0),
    ])
    result = evaluate_real_markets(markets, lambda s, d, x: None, min_edge=0.04)
    assert result["num_scored_markets"] == 0
    assert result["num_simulated_trades"] == 0


def test_evaluate_reports_model_and_market_brier():
    # Two greater-brackets. prob_above high/low → YES 0.9 / 0.1; outcomes 1 / 0
    # → model Brier 0.01. Market mids 0.5 → Brier 0.25.
    markets = _markets([
        ("KORD", "2026-04-12", "greater", 80.0, None, 0.50, 1.0),
        ("KORD", "2026-04-13", "greater", 80.0, None, 0.50, 0.0),
    ])
    pa = {("KORD", "2026-04-12"): 0.9, ("KORD", "2026-04-13"): 0.1}
    result = evaluate_real_markets(markets, lambda s, d, x: pa[(s, d)], min_edge=0.04)
    assert result["num_scored_markets"] == 2
    assert result["model_brier"] == pytest.approx(0.01)
    assert result["market_brier"] == pytest.approx(0.25)
    assert result["has_edge"] is True


def test_evaluate_less_bracket_direction_flows_into_pnl():
    # "below" (less) bracket priced 0.50. prob_above=0.1 → YES = 0.9.
    # edge 0.4 ≥ min_edge → buy YES. settlement 1.0 → positive P&L.
    markets = _markets([
        ("KORD", "2026-04-12", "less", None, 80.0, 0.50, 1.0),
    ])
    result = evaluate_real_markets(markets, lambda s, d, x: 0.1, min_edge=0.04)
    assert result["num_simulated_trades"] == 1
    assert result["simulated_pnl_usd"] > 0


def test_evaluate_reports_breakdown_by_strike_type():
    markets = _markets([
        ("KORD", "2026-04-12", "greater", 80.0, None, 0.50, 1.0),
        ("KORD", "2026-04-13", "between", 80.0, 81.0, 0.50, 0.0),
    ])
    pa = {("KORD", "2026-04-12"): 0.9, ("KORD", "2026-04-13"): 0.5}
    result = evaluate_real_markets(markets, lambda s, d, x: pa[(s, d)], min_edge=0.04)
    bt = result["by_strike_type"]
    assert bt["greater"]["n"] == 1
    assert bt["between"]["n"] == 1
    assert "less" not in bt  # none present


def test_evaluate_respects_min_edge():
    markets = _markets([
        ("KORD", "2026-04-12", "greater", 80.0, None, 0.50, 1.0),
    ])
    result = evaluate_real_markets(markets, lambda s, d, x: 0.52, min_edge=0.04)
    assert result["num_scored_markets"] == 1
    assert result["num_simulated_trades"] == 0


# ── Segment breakdowns (station / month / volume) ────────────────────────────
# The handoff's highest-value diagnostic: any thin edge hides in segments
# (specific stations, thin/illiquid markets, particular months), not in the
# aggregate. Each segment reports n plus model vs market Brier and P&L.

def test_evaluate_breaks_down_by_station():
    markets = _markets([
        ("KORD", "2026-04-12", "greater", 80.0, None, 0.50, 1.0),
        ("KLAX", "2026-04-12", "greater", 80.0, None, 0.50, 0.0),
    ])
    pa = {"KORD": 0.9, "KLAX": 0.1}
    result = evaluate_real_markets(markets, lambda s, d, x: pa[s], min_edge=0.04)
    bs = result["by_station"]
    assert bs["KORD"]["n"] == 1 and bs["KLAX"]["n"] == 1
    assert bs["KORD"]["model_brier"] == pytest.approx(0.01)
    assert "market_brier" in bs["KORD"] and "pnl" in bs["KORD"]


def test_evaluate_breaks_down_by_month():
    markets = _markets([
        ("KORD", "2026-04-12", "greater", 80.0, None, 0.50, 1.0),
        ("KORD", "2026-05-20", "greater", 80.0, None, 0.50, 0.0),
    ])
    result = evaluate_real_markets(markets, lambda s, d, x: 0.9, min_edge=0.04)
    bm = result["by_month"]
    assert bm["2026-04"]["n"] == 1 and bm["2026-05"]["n"] == 1


def test_evaluate_breaks_down_by_volume_decile_when_volume_present():
    rows = []
    for i in range(20):
        rows.append({
            "station": "KORD", "date": "2026-04-12", "strike_type": "greater",
            "floor_strike": 80.0, "cap_strike": None, "d1_mid": 0.50,
            "settlement": float(i % 2), "volume": float(i + 1) * 100,
        })
    markets = pd.DataFrame(rows)
    result = evaluate_real_markets(markets, lambda s, d, x: 0.9, min_edge=0.04)
    bv = result["by_volume_decile"]
    # Deciles ordered low→high volume; total n across deciles == scored markets.
    assert sum(v["n"] for v in bv.values()) == result["num_scored_markets"]
    assert all("model_brier" in v and "market_brier" in v for v in bv.values())


def test_evaluate_omits_volume_decile_when_absent():
    markets = _markets([
        ("KORD", "2026-04-12", "greater", 80.0, None, 0.50, 1.0),
    ])
    result = evaluate_real_markets(markets, lambda s, d, x: 0.9, min_edge=0.04)
    assert result.get("by_volume_decile") in (None, {})


# ── σ-calibration report (diagnostic #3) ─────────────────────────────────────
# err = actual - μ, sigma = predicted std. If σ is well-calibrated, the
# standardized residual z = err/σ has std ≈ 1 and ~68% of |z| ≤ 1. z_std > 1
# (or coverage < 0.68) ⇒ overconfident (σ too small); < 1 ⇒ underconfident.

def test_sigma_calibration_perfectly_calibrated_unit_residuals():
    # err = ±1 with σ = 1 → z = ±1, std exactly 1.0, all |z| ≤ 1.
    err = np.array([1.0, -1.0, 1.0, -1.0])
    sigma = np.ones(4)
    rep = sigma_calibration_report(err, sigma)["overall"]
    assert rep["n"] == 4
    assert rep["z_std"] == pytest.approx(1.0)
    assert rep["z_mean"] == pytest.approx(0.0)
    assert rep["coverage_1sigma"] == pytest.approx(1.0)
    assert rep["mean_sigma"] == pytest.approx(1.0)


def test_sigma_calibration_detects_overconfidence():
    # err twice as large as σ → z = ±2, std 2.0 (σ too small), no |z| ≤ 1.
    err = np.array([2.0, -2.0, 2.0, -2.0])
    sigma = np.ones(4)
    rep = sigma_calibration_report(err, sigma)["overall"]
    assert rep["z_std"] == pytest.approx(2.0)
    assert rep["coverage_1sigma"] == pytest.approx(0.0)
    assert rep["coverage_2sigma"] == pytest.approx(1.0)


def test_sigma_calibration_groups_by_label():
    err = np.array([1.0, -1.0, 3.0, -3.0])
    sigma = np.ones(4)
    labels = {"station": np.array(["KORD", "KORD", "KLAX", "KLAX"])}
    rep = sigma_calibration_report(err, sigma, group_labels=labels)
    assert rep["by_station"]["KORD"]["z_std"] == pytest.approx(1.0)
    assert rep["by_station"]["KLAX"]["z_std"] == pytest.approx(3.0)
    assert rep["by_station"]["KORD"]["n"] == 2


def test_sigma_calibration_empty_is_nan():
    rep = sigma_calibration_report(np.array([]), np.array([]))["overall"]
    assert rep["n"] == 0
    assert math.isnan(rep["z_std"])


def test_build_distribution_cache_exposes_mu_sigma():
    from scripts.initial_train import train_models
    from backtest.real_market_eval import _build_distribution_cache

    df = _synthetic_station_features()
    bundle = train_models(df, n_estimators=20, learning_rate=0.05, min_rows=100)
    cache = _build_distribution_cache(bundle, df)
    d = str(df["date"].iloc[0].date())
    entry = cache[("KORD", d)]
    assert "mu" in entry and "sigma" in entry
    assert entry["sigma"] > 0
    assert "lead_bucket" in entry


def test_compute_sigma_calibration_end_to_end_against_actuals():
    from scripts.initial_train import train_models
    from backtest.real_market_eval import (
        _build_distribution_cache,
        compute_sigma_calibration,
    )

    df = _synthetic_station_features()
    bundle = train_models(df, n_estimators=20, learning_rate=0.05, min_rows=100)
    cache = _build_distribution_cache(bundle, df)
    rep = compute_sigma_calibration(cache, df)
    assert rep["overall"]["n"] > 0
    assert rep["overall"]["mean_sigma"] > 0
    assert "by_station" in rep and "KORD" in rep["by_station"]
    assert "by_lead_bucket" in rep


# ── build_fair_value_fn wiring (trains a tiny real bundle) ───────────────────

def _synthetic_station_features(seed=0, nrows=150):
    from processing.features import get_feature_columns
    rng = np.random.default_rng(seed)
    cols = get_feature_columns()
    df = pd.DataFrame({c: rng.normal(0, 1, nrows) for c in cols})
    df["station"] = "KORD"
    df["lead_hour"] = 24
    df["date"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(nrows), unit="D")
    df["ecmwf_tmax"] = rng.normal(75, 5, nrows)
    df["actual_tmax"] = df["ecmwf_tmax"] + rng.normal(0, 3, nrows)
    return df


def test_prob_above_fn_prices_known_and_none_for_unknown():
    from scripts.initial_train import train_models
    from backtest.real_market_eval import build_fair_value_fn

    df = _synthetic_station_features()
    bundle = train_models(df, n_estimators=20, learning_rate=0.05, min_rows=100)
    fn = build_fair_value_fn(bundle, df)

    d = str(df["date"].iloc[0].date())
    p = fn("KORD", d, 75.0)
    assert p is not None and 0.0 <= p <= 1.0
    assert fn("KORD", "2030-12-31", 75.0) is None
    assert fn("KZZZ", d, 75.0) is None


def test_prob_above_fn_matches_compute_fair_value_for_parity():
    """The batched closure must reproduce production _compute_fair_value exactly."""
    from processing.features import get_feature_columns
    from scripts.initial_train import train_models
    from backtest.real_market_eval import build_fair_value_fn
    from strategies.ensemble_strategy import EnsembleStrategy

    df = _synthetic_station_features(seed=3)
    bundle = train_models(df, n_estimators=20, learning_rate=0.05, min_rows=100)
    fn = build_fair_value_fn(bundle, df)

    cols = get_feature_columns()
    avail = [c for c in cols if c in df.columns]
    row = df.iloc[0]
    d = str(row["date"].date())
    X = pd.DataFrame([{c: row[c] for c in avail}]).fillna(0.0)

    for threshold in (65.0, 75.0, 85.0):
        ref = EnsembleStrategy._compute_fair_value(
            X=X,
            ngboost_model=bundle["ngboost"]["KORD_D1-2"],
            qrf_model=bundle["qrf"]["KORD_D1-2"],
            residual_model=bundle["residual"].get("KORD"),
            blender=bundle["blender"],
            calibrator=bundle["calibrator"]["KORD_D1-2"],
            threshold=threshold,
            ecmwf_offset=float(row["ecmwf_tmax"]),
        )["cal_prob"]
        assert fn("KORD", d, threshold) == pytest.approx(ref, abs=1e-9)


# ── multi-lead cache selection (task 1: multi-lead edge map) ──────────────────
# The default cache takes the shortest lead per (station, date); the multi-lead
# map must instead pin a chosen lead so a fair value can be produced at each lead
# {24,48,72,96,120,168} for the SAME target market.

def _synthetic_multilead_features(seed=1, ndates=150):
    from processing.features import get_feature_columns
    rng = np.random.default_rng(seed)
    cols = get_feature_columns()
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(ndates), unit="D")
    frames = []
    for lead, offset in ((24, 0.0), (120, 10.0)):
        d = pd.DataFrame({c: rng.normal(0, 1, ndates) for c in cols})
        d["station"] = "KORD"
        d["lead_hour"] = lead
        d["date"] = dates
        d["ecmwf_tmax"] = rng.normal(75, 5, ndates) + offset
        d["actual_tmax"] = d["ecmwf_tmax"] - offset + rng.normal(0, 3, ndates)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def test_build_distribution_cache_pins_requested_lead():
    from scripts.initial_train import train_models
    from backtest.real_market_eval import _build_distribution_cache

    df = _synthetic_multilead_features()
    bundle = train_models(df, n_estimators=20, learning_rate=0.05, min_rows=100)

    c24 = _build_distribution_cache(bundle, df, lead_hour=24)
    c120 = _build_distribution_cache(bundle, df, lead_hour=120)
    d = str(df["date"].iloc[0].date())

    assert c24[("KORD", d)]["lead_hour"] == 24
    assert c24[("KORD", d)]["lead_bucket"] == "D1-2"
    assert c120[("KORD", d)]["lead_hour"] == 120
    assert c120[("KORD", d)]["lead_bucket"] == "D5-7"
    # Different lead ⇒ different feature row + different bucket model ⇒ a
    # distinct forecast (the whole point of pinning the lead).
    assert abs(c120[("KORD", d)]["mu"] - c24[("KORD", d)]["mu"]) > 1.0


def test_build_distribution_cache_default_takes_shortest_lead():
    from scripts.initial_train import train_models
    from backtest.real_market_eval import _build_distribution_cache

    df = _synthetic_multilead_features()
    bundle = train_models(df, n_estimators=20, learning_rate=0.05, min_rows=100)

    cache = _build_distribution_cache(bundle, df)  # no lead_hour ⇒ shortest
    d = str(df["date"].iloc[0].date())
    assert cache[("KORD", d)]["lead_hour"] == 24
