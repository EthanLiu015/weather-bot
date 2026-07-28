import numpy as np
import pandas as pd

# Kalshi trading fee: `coef * C * p * (1-p)` per order, capped at $0.035/contract
# (see kalshi.com/docs/kalshi-fee-schedule.pdf). Symmetric in p — a YES at p and a
# NO at 1-p pay the same. The old flat `0.05 * size * p` model was asymmetric and
# over-charged the tails / under-charged the middle.
TAKER_FEE_COEF = 0.07
MAKER_FEE_COEF = 0.0175
FEE_PER_CONTRACT_CAP = 0.035


def kalshi_fee(size: float, price: float, fee_coef: float = TAKER_FEE_COEF) -> float:
    """Kalshi fee for `size` contracts at `price` (dollars, 0<p<1).

    fee = size * min(fee_coef * p * (1-p), 0.035). `size` is the contract count
    (max-payoff USD units). Pass MAKER_FEE_COEF for resting limit orders.
    """
    per_contract = fee_coef * price * (1.0 - price)
    return size * min(per_contract, FEE_PER_CONTRACT_CAP)


def simulate_pnl(
    model_probs: np.ndarray,
    market_mids: np.ndarray,
    outcomes: np.ndarray,
    min_edge: float = 0.04,
    contract_usd: float = 1.0,
    contract_sizes: np.ndarray | None = None,
    fee_coef: float = TAKER_FEE_COEF,
    min_price: float = 0.0,
) -> dict:
    """Simulate P&L for a set of forecasts against market prices.

    Args:
        contract_sizes: Optional per-row contract sizes in USD. When provided,
            each trade uses `contract_sizes[i]` instead of the flat `contract_usd`.
            Pass Kelly-computed sizes here to simulate realistic production sizing.
        fee_coef: Kalshi fee coefficient (TAKER_FEE_COEF default, MAKER_FEE_COEF
            for resting limit orders).
        min_price: skip a trade whose entry price (mid for YES, 1-mid for NO) is
            below this floor — below ~$0.15 fee drag makes a win near-impossible.
    """
    total_pnl = 0.0
    num_trades = 0
    num_wins = 0
    edges = []

    for i, (prob, mid, outcome) in enumerate(zip(model_probs, market_mids, outcomes)):
        edge = abs(prob - mid)
        if edge < min_edge:
            continue

        entry_price = mid if prob > mid else (1.0 - mid)
        if entry_price < min_price:
            continue

        num_trades += 1
        edges.append(edge)

        size = float(contract_sizes[i]) if contract_sizes is not None else contract_usd

        if prob > mid:
            # Buy Yes at mid
            pnl = size * (outcome - mid)
        else:
            # Buy No: outcome=0 means Yes didn't resolve, so No pays out
            no_mid = 1.0 - mid
            pnl = size * ((1.0 - outcome) - no_mid)

        pnl -= kalshi_fee(size, mid, fee_coef=fee_coef)
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
