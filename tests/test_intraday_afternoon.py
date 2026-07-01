"""Tests for the afternoon-price extractor — the piece that turns Kalshi hourly
candles into 'the traded price as of time T' without leaking the outcome."""
from research.intraday_afternoon import price_at


def _candle(ts, close):
    return {"end_period_ts": ts, "price": {"close_dollars": close}}


def test_returns_latest_price_at_or_before_cutoff():
    candles = [_candle(100, "0.30"), _candle(200, "0.40"), _candle(300, "0.55")]
    # Cutoff 250 → the 200 candle is the most recent at-or-before.
    assert price_at(candles, 250) == 0.40


def test_ignores_candles_after_cutoff():
    candles = [_candle(100, "0.30"), _candle(400, "0.90")]
    assert price_at(candles, 250) == 0.30


def test_skips_degenerate_zero_and_one_closes():
    # 0 and 1 are settled/degenerate; fall back to the last genuine in-(0,1) price.
    candles = [_candle(100, "0.30"), _candle(200, "1.0"), _candle(300, "0.0")]
    assert price_at(candles, 350) == 0.30


def test_falls_back_to_mean_when_close_missing():
    candles = [{"end_period_ts": 100, "price": {"close_dollars": None, "mean_dollars": "0.42"}}]
    assert price_at(candles, 150) == 0.42


def test_none_when_no_usable_candle():
    assert price_at([], 100) is None
    assert price_at([_candle(500, "0.5")], 100) is None  # all after cutoff
