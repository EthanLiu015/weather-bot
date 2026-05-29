from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Any
import json


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    KALSHI_API_KEY: str
    KALSHI_PRIVATE_KEY_PATH: str
    KALSHI_BASE_URL: str = "https://external-api.kalshi.com/trade-api/v2"
    ECMWF_API_KEY: str = ""
    STATIONS: list[str] = ["KORD", "KJFK", "KLAX"]
    DB_URL: str = "sqlite:///./kalshi_bot.db"
    BOT_ACTIVE: bool = True
    MAX_DAILY_LOSS_USD: float = 500.0
    MAX_EXPOSURE_PER_TICKER_USD: float = 200.0
    KELLY_FRACTION: float = 0.25
    MIN_EDGE_CENTS: float = 4.0
    MAX_CI_WIDTH: float = 0.12
    HORIZON_MULTIPLIERS: dict[int, float] = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.3, 5: 0.2}
    LOG_LEVEL: str = "INFO"

    @field_validator("STATIONS", mode="before")
    @classmethod
    def parse_stations(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v

    @field_validator("HORIZON_MULTIPLIERS", mode="before")
    @classmethod
    def parse_horizon_multipliers(cls, v: Any) -> dict[int, float]:
        if isinstance(v, str):
            raw = json.loads(v)
            return {int(k): float(val) for k, val in raw.items()}
        if isinstance(v, dict):
            return {int(k): float(val) for k, val in v.items()}
        return v


def get_settings() -> Settings:
    return Settings()
