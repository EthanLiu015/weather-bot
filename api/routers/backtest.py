from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path
import csv
import math

router = APIRouter(prefix="/backtest", tags=["backtest"])

RESULTS_CSV = Path("data/backtest_results.csv")


def _float(val: str) -> float | None:
    if not val:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except ValueError:
        return None


@router.get("/results")
async def get_backtest_results() -> JSONResponse:
    if not RESULTS_CSV.exists():
        return JSONResponse({"folds": [], "summary": {}})

    folds = []
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            folds.append({
                "fold_month": row["fold_month"],
                "crps": _float(row["crps"]),
                "mae": _float(row["mae"]),
                "brier_score": _float(row["brier_score"]),
                "reliability_slope": _float(row["reliability_slope"]),
                "simulated_pnl_usd": _float(row["simulated_pnl_usd"]),
                "num_simulated_trades": int(row["num_simulated_trades"]) if row["num_simulated_trades"] else 0,
                "edge_above_threshold_pct": _float(row["edge_above_threshold_pct"]),
            })

    valid = [f for f in folds if f["crps"] is not None]
    summary = {}
    if valid:
        import numpy as np
        summary = {
            "num_folds": len(valid),
            "mean_crps": float(np.mean([f["crps"] for f in valid])),
            "mean_mae": float(np.mean([f["mae"] for f in valid])),
            "mean_brier": float(np.mean([f["brier_score"] for f in valid])),
            "mean_reliability_slope": float(np.mean([f["reliability_slope"] for f in valid if f["reliability_slope"]])),
            "total_simulated_pnl": float(sum(f["simulated_pnl_usd"] for f in valid)),
            "total_trades": sum(f["num_simulated_trades"] for f in valid),
        }

    return JSONResponse({"folds": folds, "summary": summary})
