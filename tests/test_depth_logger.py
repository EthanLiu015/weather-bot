"""Unit tests for the depth logger's buffering + sequencing logic."""
from bot.marketdata.depth_logger import _Buffers
from bot.marketdata.orderbook import OrderBook


def test_record_trade_coerces_price_and_count_to_float():
    buf = _Buffers()
    buf.record_trade(
        {"market_ticker": "X", "yes_price_dollars": "0.2800", "count_fp": "187.03",
         "taker_side": "yes"}, ts=1.0)
    row = buf.trades[0]
    assert row["yes_price"] == 0.28 and isinstance(row["yes_price"], float)
    assert row["count"] == 187.03 and isinstance(row["count"], float)
    assert row["taker_side"] == "yes"


def test_note_seq_flags_global_gap():
    # Kalshi's orderbook seq is global per channel; a break means a dropped message.
    buf = _Buffers()
    for s in (1, 2, 3):
        buf.note_seq(s)
    assert buf.resync_needed is False and buf.gaps == 0
    buf.note_seq(5)  # skipped 4
    assert buf.resync_needed is True and buf.gaps == 1


def test_note_seq_contiguous_across_markets_is_not_a_gap():
    # Interleaved markets still share one increasing sequence — no false gaps.
    buf = _Buffers()
    for s in range(1, 50):
        buf.note_seq(s)
    assert buf.gaps == 0 and buf.resync_needed is False


def test_reset_clears_seq_so_resubscribe_restarts_clean():
    buf = _Buffers()
    buf.note_seq(10)
    buf.reset()
    assert buf.last_seq is None
    buf.note_seq(1)  # server restarts at 1 after resubscribe
    assert buf.gaps == 0 and buf.resync_needed is False


def test_record_top_dedups_unchanged_and_logs_changes():
    buf = _Buffers()
    ob = OrderBook("X")
    ob.apply_snapshot([("0.40", "100")], [("0.55", "20")])
    buf.record_top("X", ob, ts=1.0)
    buf.record_top("X", ob, ts=2.0)            # unchanged -> not logged again
    assert len(buf.book) == 1
    ob.apply_delta("0.41", "30", "yes")        # top changes
    buf.record_top("X", ob, ts=3.0)
    assert len(buf.book) == 2 and buf.book[-1]["yes_bid"] == 41
