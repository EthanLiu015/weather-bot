import logging
import re
from datetime import datetime
import zoneinfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.executors.asyncio import AsyncIOExecutor
from config.stations import station_timezones
from strategies.ensemble_strategy import EnsembleStrategy

logger = logging.getLogger(__name__)

# Use registry from stations config — covers all 20 stations
STATION_TZ = station_timezones()

# Series-prefix → ICAO mapping (mirrors ensemble_strategy._SERIES_TO_STATION)
_SERIES_TO_STATION: dict[str, str] = {
    # Chicago
    "KXHIGHCHI":  "KORD", "KXLOWTCHI":  "KORD",
    # New York
    "KXHIGHNY":   "KLGA", "KXHIGHNY0":  "KLGA", "KXLOWTNYC":  "KLGA",
    # Los Angeles
    "KXHIGHLAX":  "KLAX", "KXLOWTLAX":  "KLAX",
    # Miami
    "KXHIGHMIA":  "KMIA", "KXLOWTMIA":  "KMIA",
    # Houston
    "KXHIGHOU":   "KIAH", "KXHIGHHOU":  "KIAH", "KXHOUHIGH":  "KIAH",
    "KXLOWTHOU":  "KIAH", "KXHIGHTHOU": "KIAH",
    # Philadelphia
    "KXHIGHPHIL": "KPHL", "KXLOWTPHIL": "KPHL",
    # Atlanta
    "KXHIGHATL":  "KATL", "KXHIGHTATL": "KATL", "KXLOWTATL":  "KATL",
    # Austin
    "KXHIGHAUS":  "KAUS", "KXLOWTAUS":  "KAUS",
    # Denver
    "KXDENHIGH":  "KDEN", "KXHIGHDEN":  "KDEN", "KXLOWTDEN":  "KDEN",
    # New Orleans
    "KXHIGHTNOLA":"KMSY", "KXLOWTNOLA": "KMSY",
    # Phoenix
    "KXHIGHTPHX": "KPHX", "KXLOWTPHX":  "KPHX",
    # San Francisco
    "KXHIGHTSFO": "KSFO", "KXLOWTSFO":  "KSFO",
    # Seattle
    "KXHIGHTSEA": "KSEA", "KXLOWTSEA":  "KSEA",
    # Boston
    "KXHIGHTBOS": "KBOS", "KXLOWTBOS":  "KBOS",
    # Dallas
    "KXHIGHTDAL": "KDFW", "KXLOWTDAL":  "KDFW",
    # Washington DC
    "KXHIGHTDC":  "KDCA", "KXLOWTDC":   "KDCA",
    # Las Vegas
    "KXHIGHTLV":  "KLAS", "KXLOWTLV":   "KLAS",
    # Minneapolis
    "KXHIGHTMIN": "KMSP", "KXLOWTMIN":  "KMSP",
    # Oklahoma City
    "KXHIGHTOKC": "KOKC", "KXLOWTOKC":  "KOKC",
    # San Antonio
    "KXHIGHTSATX":"KSAT", "KXLOWTSATX": "KSAT",
}


def _ticker_station(ticker: str) -> str | None:
    series = ticker.split("-")[0]
    return _SERIES_TO_STATION.get(series)


def _ticker_threshold(ticker: str) -> float | None:
    m = re.search(r"-[TB]([\d.]+)$", ticker)
    return float(m.group(1)) if m else None


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

    # ── Fix 4: seed shared_state with live markets on startup ────────────────
    async def seed_active_markets():
        """Fetch all open temperature markets sequentially to stay within rate limits."""
        try:
            import asyncio as _aio
            import httpx as _httpx

            all_markets: list[dict] = []
            for series in _SERIES_TO_STATION:
                for attempt in range(4):
                    try:
                        data = await kalshi_client._request(
                            "GET", "/markets",
                            params={"series_ticker": series, "status": "open", "limit": 50},
                            retries=1,  # don't let _request retry — we handle it here
                        )
                        all_markets.extend(data.get("markets", []))
                        await _aio.sleep(0.15)  # 150ms between requests ≈ 6 req/sec
                        break
                    except Exception as exc:
                        if "429" in str(exc) and attempt < 3:
                            wait = 2 ** attempt
                            logger.debug("Rate limited on %s, waiting %ds", series, wait)
                            await _aio.sleep(wait)
                        else:
                            break

            fresh_tickers: set[str] = set()
            for m in all_markets:
                ticker = m.get("ticker", "")
                if _ticker_station(ticker) is None:
                    continue
                yes_bid_raw = m.get("yes_bid") or m.get("yes_bid_dollars") or 50
                yes_ask_raw = m.get("yes_ask") or m.get("yes_ask_dollars") or 50
                scale = 100.0 if isinstance(yes_bid_raw, int) and yes_bid_raw > 1 else 1.0
                shared_state.update_market(
                    ticker,
                    float(yes_bid_raw) / scale,
                    float(yes_ask_raw) / scale,
                )
                fresh_tickers.add(ticker)

            # Remove any tickers that are no longer open on Kalshi
            if fresh_tickers:
                shared_state.replace_active_tickers(fresh_tickers)

            logger.info("Seeded shared_state with %d live temperature markets (%d series)",
                        len(fresh_tickers), len(_SERIES_TO_STATION))
        except Exception as exc:
            logger.warning("Market seed failed: %s", exc)

    # ── Job definitions ───────────────────────────────────────────────────────

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
            shared_state.post_alert("ensemble_error", f"Ensemble cycle failed: {exc}")

    async def d0_loop_job():
        # Fix 1: match tickers using the series→station map, not station shortname
        # Fix 1b: extract threshold from -T81 suffix, not first digit in string
        active_tickers = shared_state.get_all_tickers()
        if not active_tickers:
            logger.debug("D0 loop: shared_state empty, skipping (ensemble hasn't run yet)")
            return

        for ticker in active_tickers:
            station = _ticker_station(ticker)
            if station is None or station not in settings.STATIONS:
                continue

            tz = STATION_TZ.get(station, "UTC")
            local_hour = datetime.now(zoneinfo.ZoneInfo(tz)).hour
            if not (6 <= local_hour <= 23):
                continue

            threshold = _ticker_threshold(ticker)
            if threshold is None:
                continue

            try:
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

        # Fix 3: use correct market endpoint (not broken candlestick path)
        for ticker in tickers:
            try:
                market = await kalshi_client.get_market(ticker)
                # Kalshi returns prices in cents (0-100); normalize to 0-1
                yes_bid_raw = market.get("yes_bid") or market.get("yes_bid_dollars")
                yes_ask_raw = market.get("yes_ask") or market.get("yes_ask_dollars")
                if yes_bid_raw is None:
                    continue
                # Handle both cents (int) and dollar (float) formats
                scale = 100.0 if isinstance(yes_bid_raw, int) and yes_bid_raw > 1 else 1.0
                yes_bid = float(yes_bid_raw) / scale
                yes_ask = float(yes_ask_raw) / scale
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
                await ws_broadcaster.broadcast(shared_state.full_snapshot())
            except Exception as exc:
                logger.warning("WS broadcast failed: %s", exc)

        # Evict tickers not seen in a price update for 5 minutes
        shared_state.evict_stale_tickers(max_age_seconds=300)

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

    # ── Register jobs ─────────────────────────────────────────────────────────

    # Seed markets on startup and then every 5 minutes
    scheduler.add_job(seed_active_markets, "date", id="seed_markets_startup")
    scheduler.add_job(seed_active_markets, "interval", minutes=5, id="seed_markets")

    # 1. Detect new GEFS run every 5 minutes
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
