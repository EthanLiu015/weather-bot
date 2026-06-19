"""
Monte Carlo simulation of the trading strategy.

Three simulation modes:
  1. Bootstrap resampling: draw N months (with replacement) from the observed
     fold P&L distribution to build a synthetic multi-year equity curve.
     Reveals outcome uncertainty given the observed monthly return distribution.

  2. Trade-level bootstrap: resample individual trades with replacement to
     build equity curves from the trade level up. This captures within-month
     variance that fold-level bootstrapping averages out.

  3. Outcome perturbation: flip each trade outcome with probability p (the
     calibration uncertainty), simulating the impact of model miscalibration.

Outputs
-------
  - Summary table: P&L percentiles, Sharpe, max drawdown, ruin probability
  - data/montecarlo_equity_curves.parquet: all simulated curves for plotting
  - data/montecarlo_results.csv: percentile summary table

Requires data/backtest_trades.parquet. Run scripts/run_backtest.py first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.track_b import simulate_pnl

TRADES_PATH = Path("data/backtest_trades.parquet")
N_SIMS = 10_000
RUIN_THRESHOLD_USD = -500.0  # equity curve reaching this = "ruin"
RANDOM_SEED = 42

# Production trading parameters (must match config/settings.py)
DEFAULT_MIN_EDGE = 0.04


# ── helpers ──────────────────────────────────────────────────────────────────

def _sharpe(monthly_pnls: np.ndarray) -> float:
    if monthly_pnls.std() < 1e-9:
        return float("nan")
    return float(monthly_pnls.mean() / monthly_pnls.std() * np.sqrt(12))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    return float((peak - equity_curve).max())


def _pnl_from_trades(
    probs: np.ndarray,
    mids: np.ndarray,
    outcomes: np.ndarray,
    sizes: np.ndarray,
    min_edge: float,
) -> float:
    res = simulate_pnl(
        model_probs=probs,
        market_mids=mids,
        outcomes=outcomes,
        min_edge=min_edge,
        contract_sizes=sizes,
    )
    return res["simulated_pnl_usd"]


def _percentiles(arr: np.ndarray) -> dict:
    return {
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


# ── Mode 1: fold-level bootstrap ─────────────────────────────────────────────

def fold_level_bootstrap(monthly_pnls: np.ndarray, n_months: int) -> dict:
    """
    Resample observed monthly P&L values with replacement.
    Simulates N_SIMS trajectories of n_months trading months.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_obs = len(monthly_pnls)
    # shape: (N_SIMS, n_months)
    draws = rng.choice(monthly_pnls, size=(N_SIMS, n_months), replace=True)
    equity = np.cumsum(draws, axis=1)

    total_pnl = equity[:, -1]
    sharpes = np.array([_sharpe(draws[i]) for i in range(N_SIMS)])
    max_dds = np.array([_max_drawdown(equity[i]) for i in range(N_SIMS)])
    ruin_frac = float((total_pnl < RUIN_THRESHOLD_USD).mean())

    return {
        "mode": "fold_bootstrap",
        "n_months_simulated": n_months,
        "total_pnl": _percentiles(total_pnl),
        "sharpe": _percentiles(sharpes[np.isfinite(sharpes)]),
        "max_drawdown": _percentiles(max_dds),
        "ruin_probability": ruin_frac,
        "equity_curves": equity,  # (N_SIMS, n_months)
    }


# ── Mode 2: trade-level bootstrap ────────────────────────────────────────────

def trade_level_bootstrap(df: pd.DataFrame, n_months: int) -> dict:
    """
    Resample individual trades with replacement to build monthly P&L.
    Uses the observed trades-per-month distribution for the resample size.
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    monthly_counts = df.groupby("fold_month").size().values
    avg_trades_per_month = int(monthly_counts.mean())

    probs_all = df["trade_prob"].values
    mids_all = df["market_mid"].values
    outcomes_all = df["outcome"].values
    sizes_all = df["contract_size"].values
    n_all = len(df)

    monthly_pnls = np.zeros((N_SIMS, n_months))
    for month_idx in range(n_months):
        # Draw a random month worth of trades
        n_draw = rng.integers(
            int(monthly_counts.min()), int(monthly_counts.max()) + 1,
            size=N_SIMS
        )
        for sim_idx in range(N_SIMS):
            idx = rng.integers(0, n_all, size=int(n_draw[sim_idx]))
            monthly_pnls[sim_idx, month_idx] = _pnl_from_trades(
                probs_all[idx], mids_all[idx],
                outcomes_all[idx], sizes_all[idx],
                DEFAULT_MIN_EDGE,
            )

    equity = np.cumsum(monthly_pnls, axis=1)
    total_pnl = equity[:, -1]
    sharpes = np.array([_sharpe(monthly_pnls[i]) for i in range(N_SIMS)])
    max_dds = np.array([_max_drawdown(equity[i]) for i in range(N_SIMS)])
    ruin_frac = float((total_pnl < RUIN_THRESHOLD_USD).mean())

    return {
        "mode": "trade_bootstrap",
        "n_months_simulated": n_months,
        "total_pnl": _percentiles(total_pnl),
        "sharpe": _percentiles(sharpes[np.isfinite(sharpes)]),
        "max_drawdown": _percentiles(max_dds),
        "ruin_probability": ruin_frac,
        "equity_curves": equity,
    }


# ── Mode 3: outcome perturbation ─────────────────────────────────────────────

def outcome_perturbation(df: pd.DataFrame, flip_probs: list[float]) -> pd.DataFrame:
    """
    For each flip probability p, randomly flip each trade outcome with
    probability p to simulate the effect of model miscalibration.

    A perfectly calibrated model with mean edge of 10% still has calibration
    uncertainty. Flipping outcomes probes what happens if the true win rate
    is lower than the model believes.
    """
    rng = np.random.default_rng(RANDOM_SEED + 2)
    probs = df["trade_prob"].values
    mids = df["market_mid"].values
    outcomes_true = df["outcome"].values
    sizes = df["contract_size"].values

    rows = []
    for p_flip in flip_probs:
        total_pnls = []
        for _ in range(1000):  # 1000 perturbation samples
            flips = rng.random(len(outcomes_true)) < p_flip
            outcomes_perturbed = np.where(flips, 1.0 - outcomes_true, outcomes_true)
            res = simulate_pnl(
                model_probs=probs,
                market_mids=mids,
                outcomes=outcomes_perturbed,
                min_edge=DEFAULT_MIN_EDGE,
                contract_sizes=sizes,
            )
            total_pnls.append(res["simulated_pnl_usd"])

        arr = np.array(total_pnls)
        rows.append({
            "flip_probability": p_flip,
            "mean_pnl": arr.mean(),
            "p5_pnl": np.percentile(arr, 5),
            "p50_pnl": np.percentile(arr, 50),
            "p95_pnl": np.percentile(arr, 95),
            "prob_negative": float((arr < 0).mean()),
        })

    return pd.DataFrame(rows)


# ── reporting ─────────────────────────────────────────────────────────────────

def _print_sim_result(label: str, result: dict) -> None:
    print(f"\n── {label} ({result['n_months_simulated']} months simulated, {N_SIMS:,} runs) ──")
    print(f"  Total P&L  p5=${result['total_pnl']['p5']:>9.2f}  "
          f"p25=${result['total_pnl']['p25']:>9.2f}  "
          f"median=${result['total_pnl']['p50']:>9.2f}  "
          f"p75=${result['total_pnl']['p75']:>9.2f}  "
          f"p95=${result['total_pnl']['p95']:>9.2f}")
    print(f"  Sharpe     p5={result['sharpe']['p5']:>6.3f}  "
          f"p25={result['sharpe']['p25']:>6.3f}  "
          f"median={result['sharpe']['p50']:>6.3f}  "
          f"p75={result['sharpe']['p75']:>6.3f}  "
          f"p95={result['sharpe']['p95']:>6.3f}")
    print(f"  Max DD     p5=${result['max_drawdown']['p5']:>8.2f}  "
          f"p25=${result['max_drawdown']['p25']:>8.2f}  "
          f"median=${result['max_drawdown']['p50']:>8.2f}  "
          f"p75=${result['max_drawdown']['p75']:>8.2f}  "
          f"p95=${result['max_drawdown']['p95']:>8.2f}")
    print(f"  Ruin probability (< ${abs(RUIN_THRESHOLD_USD):.0f}): "
          f"{result['ruin_probability']:.1%}")


def main() -> None:
    if not TRADES_PATH.exists():
        print(f"ERROR: {TRADES_PATH} not found.")
        print("Run: PYTHONPATH=. python scripts/run_backtest.py --start 2021-05-27 --end 2026-05-27 --train-years 3")
        sys.exit(1)

    df = pd.read_parquet(TRADES_PATH)
    n_folds = df["fold_month"].nunique()
    n_trades_total = len(df)

    # Compute observed monthly P&L
    monthly_pnls = np.array([
        simulate_pnl(
            model_probs=df.loc[df["fold_month"] == m, "trade_prob"].values,
            market_mids=df.loc[df["fold_month"] == m, "market_mid"].values,
            outcomes=df.loc[df["fold_month"] == m, "outcome"].values,
            min_edge=DEFAULT_MIN_EDGE,
            contract_sizes=df.loc[df["fold_month"] == m, "contract_size"].values,
        )["simulated_pnl_usd"]
        for m in sorted(df["fold_month"].unique())
    ])

    print("=" * 70)
    print("MONTE CARLO SIMULATION")
    print("=" * 70)
    print(f"  Observed folds : {n_folds}")
    print(f"  Total trades   : {n_trades_total:,}")
    print(f"  Mean monthly P&L  : ${monthly_pnls.mean():.2f}")
    print(f"  StdDev monthly P&L: ${monthly_pnls.std():.2f}")
    print(f"  Min/Max monthly   : ${monthly_pnls.min():.2f} / ${monthly_pnls.max():.2f}")
    print(f"  Observed Sharpe   : {_sharpe(monthly_pnls):.3f}")
    print(f"  Observed max DD   : ${_max_drawdown(np.cumsum(monthly_pnls)):.2f}")

    # ── Mode 1 ────────────────────────────────────────────────────────────────
    print("\n[1/3] Running fold-level bootstrap...")
    mode1_12 = fold_level_bootstrap(monthly_pnls, n_months=12)
    mode1_24 = fold_level_bootstrap(monthly_pnls, n_months=24)
    _print_sim_result("Fold bootstrap  – 12 months", mode1_12)
    _print_sim_result("Fold bootstrap  – 24 months", mode1_24)

    # ── Mode 2 ────────────────────────────────────────────────────────────────
    print("\n[2/3] Running trade-level bootstrap...")
    mode2_12 = trade_level_bootstrap(df, n_months=12)
    _print_sim_result("Trade bootstrap – 12 months", mode2_12)

    # ── Mode 3 ────────────────────────────────────────────────────────────────
    print("\n[3/3] Running outcome perturbation (calibration stress test)...")
    flip_probs = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    perturb_df = outcome_perturbation(df, flip_probs)
    print("\n  flip_prob  mean_pnl   p5_pnl   p50_pnl   p95_pnl   prob_neg")
    for _, row in perturb_df.iterrows():
        marker = " ← unperturbed" if row["flip_probability"] == 0.0 else ""
        print(f"  {row['flip_probability']:.2f}       "
              f"${row['mean_pnl']:>8.2f}  "
              f"${row['p5_pnl']:>8.2f}  "
              f"${row['p50_pnl']:>8.2f}  "
              f"${row['p95_pnl']:>8.2f}  "
              f"{row['prob_negative']:.1%}{marker}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path("data/montecarlo")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary to CSV
    summary_rows = []
    for label, res in [
        ("fold_bootstrap_12m", mode1_12),
        ("fold_bootstrap_24m", mode1_24),
        ("trade_bootstrap_12m", mode2_12),
    ]:
        summary_rows.append({
            "simulation": label,
            "n_months": res["n_months_simulated"],
            **{f"pnl_{k}": v for k, v in res["total_pnl"].items()},
            **{f"sharpe_{k}": v for k, v in res["sharpe"].items()},
            **{f"maxdd_{k}": v for k, v in res["max_drawdown"].items()},
            "ruin_prob": res["ruin_probability"],
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "montecarlo_summary.csv", index=False)
    perturb_df.to_csv(out_dir / "perturbation_results.csv", index=False)

    # Save equity curves (fold bootstrap, 12m) for plotting
    curves_df = pd.DataFrame(
        mode1_12["equity_curves"],
        columns=[f"month_{i+1}" for i in range(12)],
    )
    curves_df.to_parquet(out_dir / "equity_curves_12m.parquet", index=False)

    print(f"\nResults saved to {out_dir}/")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("  Fold bootstrap: what happens if the next N months look like observed folds")
    print("  Trade bootstrap: more granular; captures within-month variance")
    print("  Outcome perturbation: what if the model wins less often than it thinks?")
    print("    flip_prob=0.05 means 5% of predicted wins become losses (and vice versa)")
    print("    If strategy is still profitable at flip_prob=0.10, it has a cushion against")
    print("    mild miscalibration without significant capital risk.")
    print("  Ruin probability: fraction of simulated runs ending below $-500 total")


if __name__ == "__main__":
    main()
