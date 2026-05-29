import logging
from fastapi import APIRouter, Request, HTTPException

router = APIRouter(prefix="/controls", tags=["controls"])
logger = logging.getLogger(__name__)


@router.post("/kill")
async def kill_switch(request: Request) -> dict:
    risk = request.app.state.risk_controls
    risk.trigger_kill_switch()
    logger.critical("Kill switch triggered via API")
    return {"status": "killed", "message": "All trading halted and orders cancelled"}


@router.post("/resume")
async def resume(request: Request) -> dict:
    risk = request.app.state.risk_controls
    risk.resume()
    logger.info("Bot resumed via API")
    return {"status": "resumed", "message": "Trading re-enabled"}


@router.post("/retrain")
async def retrain(request: Request) -> dict:
    ensemble = request.app.state.ensemble_strategy
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Ensemble strategy not available")
    try:
        import asyncio
        asyncio.create_task(ensemble.run_cycle())
        return {"status": "triggered", "message": "Retraining cycle started in background"}
    except Exception as exc:
        logger.error("Retrain trigger failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
async def get_status(request: Request) -> dict:
    settings = request.app.state.settings
    risk = request.app.state.risk_controls
    return {
        "bot_active": settings.BOT_ACTIVE,
        "daily_pnl": risk.daily_pnl(),
        "max_daily_loss": settings.MAX_DAILY_LOSS_USD,
        "max_exposure_per_ticker": settings.MAX_EXPOSURE_PER_TICKER_USD,
    }
