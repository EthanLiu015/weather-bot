from fastapi import APIRouter, Request

router = APIRouter(prefix="/positions", tags=["positions"])


@router.get("")
async def list_positions(request: Request) -> list[dict]:
    tracker = request.app.state.position_tracker
    return tracker.get_all_positions()


@router.get("/pnl")
async def get_pnl(request: Request) -> dict:
    tracker = request.app.state.position_tracker
    series = tracker.total_pnl_series()
    total = sum(d["daily_pnl"] for d in series)
    return {
        "total_realized_pnl": total,
        "series": series,
    }
