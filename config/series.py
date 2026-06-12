def is_low_temp_series(series: str) -> bool:
    """True if `series` is a Kalshi low-temperature ("LOWT") series ticker prefix."""
    return series.startswith("KXLOWT")
