from sqlalchemy import Column, Integer, String, Float, DateTime, Date
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(50), nullable=False, index=True)
    kalshi_order_id = Column(String(100), unique=True, index=True)
    side = Column(String(4), nullable=False)
    price = Column(Integer, nullable=False)
    size = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    submitted_at = Column(DateTime, default=datetime.utcnow)
    filled_at = Column(DateTime)
    fill_price = Column(Float)


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(50), nullable=False, unique=True, index=True)
    net_contracts = Column(Integer, nullable=False, default=0)
    avg_entry_price = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DailyPnL(Base):
    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    fees_paid = Column(Float, nullable=False, default=0.0)
    num_trades = Column(Integer, nullable=False, default=0)
    num_fills = Column(Integer, nullable=False, default=0)
