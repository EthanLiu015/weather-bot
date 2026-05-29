import numpy as np
import pandas as pd

FEE_RATE = 0.05  # 5% of contract value per contract


def simulate_pnl(
    model_probs: np.ndarray,
    market_mids: np.ndarray,
    outcomes: np.ndarray,
    min_edge: float = 0.04,
    contract_usd: float = 1.0,
) -> dict:
    total_pnl = 0.0
    num_trades = 0
    num_wins = 0
    edges = []

    for prob, mid, outcome in zip(model_probs, market_mids, outcomes):
        edge = abs(prob - mid)
        if edge < min_edge:
            continue

        num_trades += 1
        edges.append(edge)

        if prob > mid:
            # Buy Yes at mid
            pnl = contract_usd * (outcome - mid)
        else:
            # Buy No: outcome=0 means Yes didn't resolve, so No pays out
            no_mid = 1.0 - mid
            pnl = contract_usd * ((1.0 - outcome) - no_mid)

        fee = FEE_RATE * contract_usd * mid
        pnl -= fee
        total_pnl += pnl
        if pnl > 0:
            num_wins += 1

    return {
        "simulated_pnl_usd": total_pnl,
        "num_simulated_trades": num_trades,
        "win_rate": num_wins / num_trades if num_trades > 0 else 0.0,
        "mean_edge": float(np.mean(edges)) if edges else 0.0,
        "edge_above_threshold_pct": float(num_trades / len(model_probs)) if len(model_probs) > 0 else 0.0,
    }


def compute_edge_decay(
    edges: np.ndarray,
    horizons: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame({"edge": edges, "horizon": horizons})
    return df.groupby("horizon")["edge"].mean().reset_index().rename(columns={"edge": "mean_edge"})
