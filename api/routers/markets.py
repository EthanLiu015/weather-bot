from fastapi import APIRouter, Request
from db.models import MarketSnapshot
from db.session import get_session

router = APIRouter(prefix="/markets", tags=["markets"])


@router.get("")
async def list_markets(request: Request) -> list[dict]:
    shared_state = request.app.state.shared_state
    snap = shared_state.snapshot()
    result = []
    for ticker, state in snap.items():
        result.append({
            "ticker": ticker,
            "market_mid": state.get("market_mid"),
            "fair_value_a": state.get("fair_a"),
            "fair_value_b": state.get("fair_b"),
            "blended_fair": state.get("blended_fair"),
            "ci_width": state.get("ci_width"),
            "horizon_days": state.get("horizon_days"),
            "net_contracts": state.get("net_contracts"),
            "strategy_lock": state.get("strategy_lock"),
            "last_updated": state.get("last_updated"),
        })
    return result


@router.get("/{ticker}")
async def get_market(ticker: str, request: Request) -> dict:
    shared_state = request.app.state.shared_state
    snap = shared_state.snapshot()
    state = snap.get(ticker, {})

    history = []
    with get_session() as db:
        rows = (
            db.query(MarketSnapshot)
            .filter(MarketSnapshot.ticker == ticker)
            .order_by(MarketSnapshot.timestamp.desc())
            .limit(20)
            .all()
        )
        history = [
            {
                "timestamp": r.timestamp.isoformat(),
                "yes_mid": r.yes_mid,
                "blended_fair": r.blended_fair,
            }
            for r in rows
        ]

    return {"ticker": ticker, "state": state, "history": history}
