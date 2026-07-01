"""
Parameter stability analysis for the trading strategy.

Tests how sensitive key trading metrics are to variations in four parameters:
  - MIN_EDGE_CENTS      : minimum probability edge required to place a trade
  - KELLY_FRACTION      : fraction of full-Kelly bet to stake
  - MAX_EXPOSURE_USD    : cap on dollar exposure per ticker
  - RESIDUAL_SIGMA_FLOOR: minimum sigma applied to residual model outputs

For each parameter, all other parameters are held at their production defaults.
Metric reported per configuration:
  total_pnl, trades, win_rate, mean_edge, pnl_per_trade, sharpe (across folds),
  max_drawdown

Requires data/backtest_trades.parquet (written by backtest/runner.py during a
full walk-forward run). Run scripts/run_backtest.py first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from backtest.track_b import simulate_pnl
from trading.kelly import compute_size

TRADES_PATH = Path("data/backtest_trades.parquet")
HORIZON_MULTIPLIERS = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.3, 5: 0.2}

# Production defaults
DEFAULT_MIN_EDGE = 0.04
DEFAULT_KELLY_FRACTION = 0.25
DEFAULT_MAX_EXPOSURE = 200.0
DEFAULT_SIGMA_FLOOR = 2.0

# Parameter grids
GRIDS = {
    "min_edge_cents": [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20],
    "kelly_fraction": [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.00],
    "max_exposure_usd": [25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0],
    "sigma_floor": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
}


def _recompute_sizes(
    df: pd.DataFrame,
    kelly_fraction: float,
    max_exposure_usd: float,
    sigma_floor: float,
) -> np.ndarray:
    """Re-compute Kelly contract sizes with new parameters."""
    sizes = []
    for _, row in df.iterrows():
        sigma_eff = max(float(row["sigma"]), sigma_floor)
        mu_abs = float(row["mu"])
        ci_width = float(min(2.0 * sigma_eff / max(abs(mu_abs), 1.0), 1.0))
        horizon_days = max(1, int(row["lead_hour"]) // 24)
        s = compute_size(
            fair_value=float(row["trade_prob"]),
            market_price=float(row["market_mid"]),
            ci_width=ci_width,
            horizon_days=horizon_days,
            kelly_fraction=kelly_fraction,
            max_exposure_usd=max_exposure_usd,
            horizon_multipliers=HORIZON_MULTIPLIERS,
            strategy_lock=False,
        )
        sizes.append(float(s))
    return np.array(sizes)


def _fold_pnl(
    df: pd.DataFrame,
    min_edge: float,
    contract_sizes: np.ndarray,
) -> list[float]:
    """Compute per-fold P&L list for Sharpe / drawdown calculations."""
    fold_pnls = []
    for month in sorted(df["fold_month"].unique()):
        mask = df["fold_month"] == month
        res = simulate_pnl(
            model_probs=df.loc[mask, "trade_prob"].values,
            market_mids=df.loc[mask, "market_mid"].values,
            outcomes=df.loc[mask, "outcome"].values,
            min_edge=min_edge,
            contract_sizes=contract_sizes[mask.values],
        )
        fold_pnls.append(res["simulated_pnl_usd"])
    return fold_pnls


def _sharpe(monthly_pnls: list[float]) -> float:
    arr = np.array(monthly_pnls)
    if arr.std() < 1e-9:
        return float("nan")
    return float(arr.mean() / arr.std() * np.sqrt(12))


def _max_drawdown(monthly_pnls: list[float]) -> float:
    equity = np.cumsum(monthly_pnls)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(dd.max()) if len(dd) else 0.0


def _run_config(
    df: pd.DataFrame,
    min_edge: float,
    kelly_fraction: float,
    max_exposure_usd: float,
    sigma_floor: float,
) -> dict:
    sizes = _recompute_sizes(df, kelly_fraction, max_exposure_usd, sigma_floor)
    res = simulate_pnl(
        model_probs=df["trade_prob"].values,
        market_mids=df["market_mid"].values,
        outcomes=df["outcome"].values,
        min_edge=min_edge,
        contract_sizes=sizes,
    )
    fold_pnls = _fold_pnl(df, min_edge, sizes)
    return {
        "total_pnl": res["simulated_pnl_usd"],
        "trades": res["num_simulated_trades"],
        "win_rate": res["win_rate"],
        "mean_edge": res["mean_edge"],
        "pnl_per_trade": (
            res["simulated_pnl_usd"] / res["num_simulated_trades"]
            if res["num_simulated_trades"] > 0 else 0.0
        ),
        "sharpe_annualised": _sharpe(fold_pnls),
        "max_monthly_drawdown": _max_drawdown(fold_pnls),
        "edge_pct": res["edge_above_threshold_pct"],
    }


def run_stability_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}

    print("\n" + "=" * 70)
    print("PARAMETER STABILITY ANALYSIS")
    print(f"  {len(df):,} trades  |  {df['fold_month'].nunique()} folds  |  "
          f"{df['station'].nunique() if 'station' in df.columns else '?'} stations")
    print("=" * 70)

    for param_name, grid in GRIDS.items():
        rows = []
        for val in grid:
            cfg = dict(
                min_edge=DEFAULT_MIN_EDGE,
                kelly_fraction=DEFAULT_KELLY_FRACTION,
                max_exposure_usd=DEFAULT_MAX_EXPOSURE,
                sigma_floor=DEFAULT_SIGMA_FLOOR,
            )
            cfg[{
                "min_edge_cents": "min_edge",
                "kelly_fraction": "kelly_fraction",
                "max_exposure_usd": "max_exposure_usd",
                "sigma_floor": "sigma_floor",
            }[param_name]] = val

            r = _run_config(df, **cfg)
            r[param_name] = val
            rows.append(r)

        table = pd.DataFrame(rows).set_index(param_name)
        results[param_name] = table

        print(f"\n── {param_name} (default = {DEFAULT_MIN_EDGE if param_name=='min_edge_cents' else DEFAULT_KELLY_FRACTION if param_name=='kelly_fraction' else DEFAULT_MAX_EXPOSURE if param_name=='max_exposure_usd' else DEFAULT_SIGMA_FLOOR}) ──")
        print(table[["total_pnl", "trades", "win_rate", "pnl_per_trade",
                     "sharpe_annualised", "max_monthly_drawdown"]].to_string(
            float_format=lambda x: f"{x:.3f}"
        ))

    return results


def _save_results(results: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for param_name, table in results.items():
        table.to_csv(out_dir / f"stability_{param_name}.csv")

    # Print summary: find optimal per parameter
    print("\n" + "=" * 70)
    print("OPTIMAL VALUES (max Sharpe ratio)")
    print("=" * 70)
    for param_name, table in results.items():
        best_idx = table["sharpe_annualised"].idxmax()
        best_row = table.loc[best_idx]
        current_val = {
            "min_edge_cents": DEFAULT_MIN_EDGE,
            "kelly_fraction": DEFAULT_KELLY_FRACTION,
            "max_exposure_usd": DEFAULT_MAX_EXPOSURE,
            "sigma_floor": DEFAULT_SIGMA_FLOOR,
        }[param_name]
        marker = "  ← CURRENT" if abs(float(best_idx) - current_val) < 1e-9 else f"  ← CURRENT is {current_val}"
        print(f"  {param_name:25s}  best={best_idx:<8}  Sharpe={best_row['sharpe_annualised']:.3f}{marker}")


def main() -> None:
    if not TRADES_PATH.exists():
        print(f"ERROR: {TRADES_PATH} not found.")
        print("Run: PYTHONPATH=. python scripts/run_backtest.py --start 2021-05-27 --end 2026-05-27 --train-years 3")
        sys.exit(1)

    df = pd.read_parquet(TRADES_PATH)
    print(f"Loaded {len(df):,} trade records from {df['fold_month'].nunique()} folds")

    results = run_stability_analysis(df)
    _save_results(results, Path("data/stability"))
    print(f"\nCSV tables saved to data/stability/")


if __name__ == "__main__":
    main()
