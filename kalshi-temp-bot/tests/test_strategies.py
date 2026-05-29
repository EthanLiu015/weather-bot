import pytest
from strategies.shared_state import SharedState


def test_shared_state_blended_fair_both_present():
    state = SharedState()
    state.update_fair_a("TICKER-A", 0.70, 0.05, 2)
    state.update_fair_b("TICKER-A", 0.68)
    blended = state.blended_fair("TICKER-A")
    assert blended == pytest.approx(0.69)


def test_shared_state_blended_fair_only_a():
    state = SharedState()
    state.update_fair_a("TICKER-B", 0.65, 0.05, 3)
    blended = state.blended_fair("TICKER-B")
    assert blended == pytest.approx(0.65)


def test_shared_state_strategy_lock_when_disagree():
    state = SharedState()
    state.update_fair_a("LOCK-TICKER", 0.80, 0.05, 1)
    state.update_fair_b("LOCK-TICKER", 0.70)  # 10¢ difference > 5¢ threshold
    snap = state.snapshot()
    assert snap["LOCK-TICKER"]["strategy_lock"] is True


def test_shared_state_no_lock_when_agree():
    state = SharedState()
    state.update_fair_a("AGREE-TICKER", 0.70, 0.05, 1)
    state.update_fair_b("AGREE-TICKER", 0.72)  # 2¢ difference < 5¢
    snap = state.snapshot()
    assert snap["AGREE-TICKER"]["strategy_lock"] is False


def test_should_trade_false_when_ci_too_wide():
    state = SharedState(min_edge=0.04, max_ci_width=0.12)
    state.update_fair_a("WIDE-CI", 0.80, 0.20, 1)  # ci_width=0.20 > 0.12
    state.update_market("WIDE-CI", 0.55, 0.57)
    assert not state.should_trade("WIDE-CI")


def test_should_trade_false_when_no_fair():
    state = SharedState()
    state.update_market("NO-FAIR", 0.50, 0.52)
    assert not state.should_trade("NO-FAIR")


def test_should_trade_true_with_sufficient_edge():
    state = SharedState(min_edge=0.04, max_ci_width=0.12)
    state.update_fair_a("GOOD-EDGE", 0.75, 0.05, 1)
    state.update_market("GOOD-EDGE", 0.55, 0.57)
    assert state.should_trade("GOOD-EDGE")


def test_update_fill_changes_net_contracts():
    state = SharedState()
    state.update_fair_a("FILL-TEST", 0.70, 0.05, 1)
    state.update_fill("FILL-TEST", 5, 0.68)
    snap = state.snapshot()
    assert snap["FILL-TEST"]["net_contracts"] == 5


def test_get_all_tickers():
    state = SharedState()
    state.update_fair_a("T1", 0.60, 0.05, 1)
    state.update_fair_a("T2", 0.70, 0.05, 1)
    tickers = state.get_all_tickers()
    assert "T1" in tickers
    assert "T2" in tickers
