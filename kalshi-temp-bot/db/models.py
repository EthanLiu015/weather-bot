from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, Text
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    pass


class ForecastRun(Base):
    __tablename__ = "forecast_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    station = Column(String(10), nullable=False, index=True)
    model_source = Column(String(20), nullable=False)
    run_time = Column(DateTime, nullable=False)
    lead_time_hours = Column(Integer, nullable=False)
    mu = Column(Float, nullable=False)
    sigma = Column(Float, nullable=False)
    calibrated_prob = Column(Float)
    ci_lower = Column(Float)
    ci_upper = Column(Float)
    threshold = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    yes_bid = Column(Float)
    yes_ask = Column(Float)
    yes_mid = Column(Float)
    volume = Column(Integer)
    open_interest = Column(Integer)
    fair_value_a = Column(Float)
    fair_value_b = Column(Float)
    blended_fair = Column(Float)
    ci_width = Column(Float)


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(50), nullable=False, index=True)
    kalshi_order_id = Column(String(100), unique=True, index=True)
    side = Column(String(4), nullable=False)
    price = Column(Integer, nullable=False)
    size = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    strategy = Column(String(2), nullable=False)
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


class CalibrationSnapshot(Base):
    __tablename__ = "calibration_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    station = Column(String(10), nullable=False, index=True)
    lead_bucket = Column(String(10), nullable=False)
    brier_score = Column(Float)
    reliability_slope = Column(Float)
    sharpness = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(30), nullable=False)
    station = Column(String(10), nullable=False)
    version = Column(String(20), nullable=False)
    path = Column(Text, nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)
    crps_val = Column(Float)
    mae_val = Column(Float)
