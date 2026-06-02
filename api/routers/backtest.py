from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path
import csv
import math
import numpy as np

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


def _int(val: str) -> int:
    try:
        return int(val) if val else 0
    except ValueError:
        return 0


@router.get("/results")
async def get_backtest_results() -> JSONResponse:
    if not RESULTS_CSV.exists():
        return JSONResponse({"folds": [], "summary": {}})

    folds = []
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            folds.append({
                "fold_month":            row["fold_month"],
                "crps":                  _float(row.get("crps", "")),
                "mae":                   _float(row.get("mae", "")),
                "brier_score":           _float(row.get("brier_score", "")),
                "reliability_slope":     _float(row.get("reliability_slope", "")),
                "simulated_pnl_usd":     _float(row.get("simulated_pnl_usd", "")),
                "num_simulated_trades":  _int(row.get("num_simulated_trades", "")),
                "edge_above_threshold_pct": _float(row.get("edge_above_threshold_pct", "")),
                "real_price_pnl":        _float(row.get("real_price_pnl", "")) or 0.0,
                "real_price_trades":     _int(row.get("real_price_trades", "")),
                "clim_price_pnl":        _float(row.get("clim_price_pnl", "")) or 0.0,
                "clim_price_trades":     _int(row.get("clim_price_trades", "")),
            })

    valid = [f for f in folds if f["crps"] is not None]
    all_folds = folds  # include folds with null CRPS for PnL charts

    summary: dict = {}
    if valid:
        summary = {
            "num_folds":              len(valid),
            "mean_crps":              float(np.mean([f["crps"] for f in valid])),
            "mean_mae":               float(np.mean([f["mae"] for f in valid])),
            "mean_brier":             float(np.mean([f["brier_score"] for f in valid])),
            "mean_reliability_slope": float(np.mean([f["reliability_slope"] for f in valid if f["reliability_slope"]])),
            "total_simulated_pnl":    float(sum(f["simulated_pnl_usd"] or 0 for f in all_folds)),
            "total_trades":           sum(f["num_simulated_trades"] for f in all_folds),
            "total_real_price_pnl":   float(sum(f["real_price_pnl"] for f in all_folds)),
            "total_real_price_trades": sum(f["real_price_trades"] for f in all_folds),
            "total_clim_price_pnl":   float(sum(f["clim_price_pnl"] for f in all_folds)),
            "total_clim_price_trades": sum(f["clim_price_trades"] for f in all_folds),
        }

    return JSONResponse({"folds": all_folds, "summary": summary})
