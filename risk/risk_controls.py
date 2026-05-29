import logging
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from db.models import DailyPnL, Position

logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 300  # 5 minutes


class RiskControls:
    def __init__(self, settings, db_session_factory) -> None:
        self._settings = settings
        self._db_factory = db_session_factory
        self._cooldowns: dict[str, datetime] = {}
        self._order_manager = None

    def set_order_manager(self, order_manager) -> None:
        self._order_manager = order_manager

    def can_trade(self, ticker: str) -> tuple[bool, str]:
        if not self._settings.BOT_ACTIVE:
            return False, "Kill switch active"

        today_pnl = self.daily_pnl()
        if today_pnl < -self._settings.MAX_DAILY_LOSS_USD:
            return False, f"Daily loss limit hit: ${today_pnl:.2f}"

        cooldown_end = self._cooldowns.get(ticker)
        if cooldown_end is not None and datetime.utcnow() < cooldown_end:
            remaining = (cooldown_end - datetime.utcnow()).seconds
            return False, f"Cooldown active for {remaining}s"

        exposure = self._ticker_exposure(ticker)
        if exposure >= self._settings.MAX_EXPOSURE_PER_TICKER_USD:
            return False, f"Max exposure reached: ${exposure:.2f}"

        return True, ""

    def trigger_kill_switch(self) -> None:
        self._settings.BOT_ACTIVE = False
        logger.critical("KILL SWITCH TRIGGERED — all trading halted")
        if self._order_manager is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._order_manager.cancel_all())
                else:
                    loop.run_until_complete(self._order_manager.cancel_all())
            except Exception as exc:
                logger.error("Error cancelling orders on kill: %s", exc)

    def resume(self) -> None:
        self._settings.BOT_ACTIVE = True
        logger.info("Bot resumed — trading re-enabled")

    def record_rejected_order(self, ticker: str) -> None:
        self._cooldowns[ticker] = datetime.utcnow() + timedelta(seconds=COOLDOWN_SECONDS)
        logger.warning("Cooldown set for ticker %s for %ds", ticker, COOLDOWN_SECONDS)

    def daily_pnl(self) -> float:
        try:
            with self._db_factory() as db:
                row = db.query(DailyPnL).filter(DailyPnL.date == date.today()).first()
                return float(row.realized_pnl) if row else 0.0
        except Exception as exc:
            logger.error("Failed to query daily PnL: %s", exc)
            return 0.0

    def _ticker_exposure(self, ticker: str) -> float:
        try:
            with self._db_factory() as db:
                pos = db.query(Position).filter(Position.ticker == ticker).first()
                if pos is None:
                    return 0.0
                return abs(pos.net_contracts) * pos.avg_entry_price
        except Exception as exc:
            logger.error("Failed to query position for %s: %s", ticker, exc)
            return 0.0
