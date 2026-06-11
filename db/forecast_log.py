import datetime as dt

from db.models import DailyForecastLog
from db.session import get_session


def record_forecast_tmax(station: str, target_date: dt.date, forecast_tmax_f: float) -> None:
    """Persist the GEFS D+1 Tmax forecast for `target_date`, once. Later cycles
    diff this against the observed Tmax to compute obs_minus_model_lag*."""
    with get_session() as db:
        existing = (
            db.query(DailyForecastLog)
            .filter_by(station=station, target_date=target_date)
            .first()
        )
        if existing is None:
            db.add(DailyForecastLog(
                station=station,
                target_date=target_date,
                forecast_tmax_f=forecast_tmax_f,
            ))


def get_forecast_tmax(station: str, dates: list[dt.date]) -> dict[dt.date, float]:
    if not dates:
        return {}
    with get_session() as db:
        rows = (
            db.query(DailyForecastLog)
            .filter(
                DailyForecastLog.station == station,
                DailyForecastLog.target_date.in_(dates),
            )
            .all()
        )
        return {row.target_date: row.forecast_tmax_f for row in rows}
