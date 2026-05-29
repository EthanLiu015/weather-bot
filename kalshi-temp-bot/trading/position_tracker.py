import logging
from datetime import date
from db.models import Position, DailyPnL
from db.session import get_session

logger = logging.getLogger(__name__)


class PositionTracker:
    def __init__(self, db_session_factory) -> None:
        self._db_factory = db_session_factory

    def get_all_positions(self) -> list[dict]:
        with get_session() as db:
            positions = db.query(Position).all()
            return [
                {
                    "ticker": p.ticker,
                    "net_contracts": p.net_contracts,
                    "avg_entry_price": p.avg_entry_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "realized_pnl": p.realized_pnl,
                    "last_updated": p.last_updated.isoformat() if p.last_updated else None,
                }
                for p in positions
            ]

    def update_unrealized_pnl(self, ticker: str, current_market_price: float) -> float:
        with get_session() as db:
            pos = db.query(Position).filter(Position.ticker == ticker).first()
            if pos is None or pos.net_contracts == 0:
                return 0.0
            unrealized = pos.net_contracts * (current_market_price - pos.avg_entry_price)
            pos.unrealized_pnl = unrealized
            return unrealized

    def record_realized_pnl(self, ticker: str, pnl_delta: float, fee: float = 0.0) -> None:
        with get_session() as db:
            pos = db.query(Position).filter(Position.ticker == ticker).first()
            if pos:
                pos.realized_pnl += pnl_delta

            today = date.today()
            daily = db.query(DailyPnL).filter(DailyPnL.date == today).first()
            if daily is None:
                daily = DailyPnL(date=today, realized_pnl=0.0, fees_paid=0.0, num_trades=0, num_fills=0)
                db.add(daily)
            daily.realized_pnl += pnl_delta
            daily.fees_paid += fee
            daily.num_fills += 1

    def close_daily(self) -> dict:
        today = date.today()
        with get_session() as db:
            daily = db.query(DailyPnL).filter(DailyPnL.date == today).first()
            if daily is None:
                return {"date": today.isoformat(), "realized_pnl": 0.0, "fees_paid": 0.0}
            return {
                "date": daily.date.isoformat(),
                "realized_pnl": daily.realized_pnl,
                "fees_paid": daily.fees_paid,
                "num_trades": daily.num_trades,
                "num_fills": daily.num_fills,
            }

    def total_pnl_series(self) -> list[dict]:
        with get_session() as db:
            rows = db.query(DailyPnL).order_by(DailyPnL.date).all()
            cumulative = 0.0
            result = []
            for row in rows:
                cumulative += row.realized_pnl
                result.append({
                    "date": row.date.isoformat(),
                    "daily_pnl": row.realized_pnl,
                    "cumulative_pnl": cumulative,
                    "fees_paid": row.fees_paid,
                })
            return result
