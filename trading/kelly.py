def compute_size(
    fair_value: float,
    market_price: float,
    ci_width: float,
    horizon_days: int,
    kelly_fraction: float,
    max_exposure_usd: float,
    horizon_multipliers: dict[int, float],
    strategy_lock: bool,
) -> int:
    if market_price <= 0.0 or market_price >= 1.0:
        return 0

    # Full Kelly formula
    b = (1.0 - market_price) / market_price
    p = fair_value
    q = 1.0 - p
    if b <= 0:
        return 0

    kelly_pct = (b * p - q) / b

    if kelly_pct <= 0:
        return 0

    # Adjustments
    kelly_pct *= kelly_fraction
    kelly_pct *= max(0.0, 1.0 - ci_width)
    kelly_pct *= horizon_multipliers.get(horizon_days, 0.2)
    if strategy_lock:
        kelly_pct *= 0.5

    if kelly_pct <= 0:
        return 0

    usd_size = kelly_pct * max_exposure_usd
    contracts = int(usd_size / market_price)
    max_contracts = int(max_exposure_usd / market_price)
    return max(0, min(contracts, max_contracts))
