from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Any
import json


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # silently ignore unknown env vars (e.g. NOAA_CDO_TOKEN)
    )

    KALSHI_API_KEY: str
    KALSHI_PRIVATE_KEY_PATH: str
    KALSHI_BASE_URL: str = "https://external-api.kalshi.com/trade-api/v2"
    DB_URL: str = "sqlite:///./kalshi_bot.db"
    BOT_ACTIVE: bool = True
    PAPER_TRADING: bool = True  # when True: real market data, simulated fills, no real orders
    MAX_DAILY_LOSS_USD: float = 500.0
    MAX_EXPOSURE_PER_TICKER_USD: float = 200.0
    KELLY_FRACTION: float = 0.25
    MIN_EDGE_CENTS: float = 4.0
    MAX_CI_WIDTH: float = 0.12
    MIN_ENTRY_PRICE: float = 0.15
    MIN_VOLUME: float = 10.0
    MAX_EDGE_CENTS: float = 60.0
    MAX_DISAGREEMENT_CENTS: float = 40.0
    MAX_SPREAD_CENTS: float = 30.0
    MODEL_BLEND_WEIGHT: float = 0.30
    EXECUTION_MODE: str = "model"
    HORIZON_MULTIPLIERS: dict[int, float] = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.3, 5: 0.2}
    LOG_LEVEL: str = "INFO"

    # Tennis order-flow momentum taker (plans/tennis-mm-next-steps.md) — separate
    # kill switch from BOT_ACTIVE since it's an independent strategy/process.
    TENNIS_ENABLED: bool = False
    TENNIS_CONTRACT_SIZE: int = 1  # fixed — sizing beyond 1 contract is unvalidated
    TENNIS_HOLD_SECONDS: int = 30  # strongest validated horizon (cluster t-stat 7.2-8.0)
    TENNIS_MAX_CONCURRENT_POSITIONS: int = 5

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
