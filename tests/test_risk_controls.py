import pytest
from unittest.mock import MagicMock, patch
from risk.risk_controls import RiskControls


def _make_settings(active: bool = True, max_loss: float = 500.0, max_exposure: float = 200.0):
    s = MagicMock()
    s.BOT_ACTIVE = active
    s.MAX_DAILY_LOSS_USD = max_loss
    s.MAX_EXPOSURE_PER_TICKER_USD = max_exposure
    return s


def _make_controls(active: bool = True):
    settings = _make_settings(active=active)
    db_factory = MagicMock()
    controls = RiskControls(settings=settings, db_session_factory=db_factory)
    return controls, settings


def test_kill_switch_blocks_all_trading():
    controls, settings = _make_controls(active=True)
    with patch.object(controls, 'daily_pnl', return_value=0.0), \
         patch.object(controls, '_ticker_exposure', return_value=0.0):
        ok, _ = controls.can_trade("TICKER1")
        assert ok
    settings.BOT_ACTIVE = False
    ok, reason = controls.can_trade("TICKER1")
    assert not ok
    assert "Kill switch" in reason


def test_daily_loss_limit_enforced():
    controls, _ = _make_controls()
    with patch.object(controls, 'daily_pnl', return_value=-600.0), \
         patch.object(controls, '_ticker_exposure', return_value=0.0):
        ok, reason = controls.can_trade("TICKER1")
        assert not ok
        assert "Daily loss" in reason


def test_can_trade_when_conditions_met():
    controls, _ = _make_controls()
    with patch.object(controls, 'daily_pnl', return_value=10.0), \
         patch.object(controls, '_ticker_exposure', return_value=50.0):
        ok, reason = controls.can_trade("TICKER1")
        assert ok
        assert reason == ""


def test_exposure_limit_blocks_trading():
    controls, _ = _make_controls()
    with patch.object(controls, 'daily_pnl', return_value=0.0), \
         patch.object(controls, '_ticker_exposure', return_value=200.0):
        ok, reason = controls.can_trade("TICKER1")
        assert not ok
        assert "exposure" in reason.lower()


def test_resume_re_enables_trading():
    controls, settings = _make_controls(active=False)
    assert not settings.BOT_ACTIVE
    controls.resume()
    assert settings.BOT_ACTIVE


def test_cooldown_blocks_ticker():
    controls, _ = _make_controls()
    controls.record_rejected_order("HOT_TICKER")
    with patch.object(controls, 'daily_pnl', return_value=0.0), \
         patch.object(controls, '_ticker_exposure', return_value=0.0):
        ok, reason = controls.can_trade("HOT_TICKER")
        assert not ok
        assert "Cooldown" in reason
