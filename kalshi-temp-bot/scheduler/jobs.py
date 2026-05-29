import logging
from datetime import datetime
import zoneinfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor

logger = logging.getLogger(__name__)

STATION_TZ = {
    "KORD": "America/Chicago",
    "KJFK": "America/New_York",
    "KLAX": "America/Los_Angeles",
}


def build_scheduler(
    ensemble_strategy,
    d0_strategy,
    order_manager,
    shared_state,
    kalshi_client,
    position_tracker,
    ws_broadcaster,
    settings,
) -> AsyncIOScheduler:
    executors = {"default": AsyncIOExecutor()}
    scheduler = AsyncIOScheduler(executors=executors)
    last_run_holder = {"ts": datetime.min}

    async def detect_model_run_job():
        is_new = ensemble_strategy.detect_new_model_run(last_run_holder["ts"])
        if is_new:
            logger.info("New GEFS run detected — triggering ensemble cycle")
            await run_ensemble_cycle_job()

    async def run_ensemble_cycle_job():
        try:
            await ensemble_strategy.run_cycle()
            last_run_holder["ts"] = datetime.utcnow()
            await order_manager.process_tick()
        except Exception as exc:
            logger.error("Ensemble cycle failed: %s", exc)

    async def d0_loop_job():
        for station in settings.STATIONS:
            tz = STATION_TZ.get(station, "UTC")
            local_hour = datetime.now(zoneinfo.ZoneInfo(tz)).hour
            if not (6 <= local_hour <= 23):
                continue
            tickers = [t for t in shared_state.get_all_tickers() if station[1:] in t]
            for ticker in tickers:
                try:
                    import re
                    m = re.search(r"(\d+)", ticker)
                    threshold = float(m.group(1)) if m else 70.0
                    await d0_strategy.run_cycle(station, ticker, threshold)
                    await order_manager.process_tick()
                except Exception as exc:
                    logger.error("D0 cycle failed for %s/%s: %s", station, ticker, exc)

    async def sync_fills_job():
        try:
            await order_manager.sync_fills()
        except Exception as exc:
            logger.error("sync_fills failed: %s", exc)

    async def market_snapshot_job():
        tickers = shared_state.get_all_tickers()
        for ticker in tickers:
            try:
                market = await kalshi_client.get_market(ticker)
                yes_bid = market.get("yes_bid", 50) / 100.0
                yes_ask = market.get("yes_ask", 50) / 100.0
                shared_state.update_market(ticker, yes_bid, yes_ask)

                from db.models import MarketSnapshot
                from db.session import get_session
                snap = shared_state.snapshot().get(ticker, {})
                with get_session() as db:
                    db.add(MarketSnapshot(
                        ticker=ticker,
                        timestamp=datetime.utcnow(),
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        yes_mid=(yes_bid + yes_ask) / 2.0,
                        fair_value_a=snap.get("fair_a"),
                        fair_value_b=snap.get("fair_b"),
                        blended_fair=snap.get("blended_fair"),
                        ci_width=snap.get("ci_width"),
                    ))
            except Exception as exc:
                logger.debug("Snapshot failed for %s: %s", ticker, exc)

        if ws_broadcaster is not None:
            try:
                await ws_broadcaster.broadcast(shared_state.snapshot())
            except Exception as exc:
                logger.warning("WS broadcast failed: %s", exc)

    async def daily_close_job():
        try:
            summary = position_tracker.close_daily()
            logger.info("Daily close: %s", summary)

            from db.session import get_session
            from db.models import CalibrationSnapshot
            with get_session() as db:
                for station in settings.STATIONS:
                    db.add(CalibrationSnapshot(
                        station=station,
                        lead_bucket="D1-2",
                        brier_score=None,
                        reliability_slope=None,
                        sharpness=None,
                        recorded_at=datetime.utcnow(),
                    ))
        except Exception as exc:
            logger.error("Daily close failed: %s", exc)

    # 1. Detect new model run every 5 minutes
    scheduler.add_job(detect_model_run_job, "interval", minutes=5, id="detect_model_run")

    # 2. Fallback ensemble cycle 4× daily
    scheduler.add_job(run_ensemble_cycle_job, "cron", hour="0,6,12,18", minute=30, id="ensemble_fallback")

    # 3. D-0 loop every 20 minutes
    scheduler.add_job(d0_loop_job, "interval", minutes=20, id="d0_loop")

    # 4. Sync fills every 60 seconds
    scheduler.add_job(sync_fills_job, "interval", seconds=60, id="sync_fills")

    # 5. Market snapshot every 60 seconds
    scheduler.add_job(market_snapshot_job, "interval", seconds=60, id="market_snapshot")

    # 6. Daily close at 23:59 UTC
    scheduler.add_job(daily_close_job, "cron", hour=23, minute=59, id="daily_close")

    return scheduler
