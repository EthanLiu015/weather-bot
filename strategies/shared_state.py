import threading
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

STRATEGY_LOCK_THRESHOLD = 0.05


@dataclass
class TickerState:
    fair_a: float | None = None
    fair_b: float | None = None
    ci_width: float = 0.0
    market_mid: float = 0.5
    net_contracts: int = 0
    last_order_id: str | None = None
    horizon_days: int = 1
    strategy_lock: bool = False
    last_updated: datetime = field(default_factory=datetime.utcnow)


class SharedState:
    def __init__(self, min_edge: float = 0.04, max_ci_width: float = 0.12) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, TickerState] = {}
        self._min_edge = min_edge
        self._max_ci_width = max_ci_width
        self._alerts: list[dict] = []  # [{type, message, timestamp}]

    def post_alert(self, alert_type: str, message: str) -> None:
        with self._lock:
            self._alerts.append({
                "type": alert_type,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            })
            # Keep only the last 20 alerts
            self._alerts = self._alerts[-20:]
        logger.warning("ALERT [%s]: %s", alert_type, message)

    def clear_alerts(self, alert_type: str | None = None) -> None:
        with self._lock:
            if alert_type:
                self._alerts = [a for a in self._alerts if a["type"] != alert_type]
            else:
                self._alerts = []

    def get_alerts(self) -> list[dict]:
        with self._lock:
            return list(self._alerts)

    def _get_or_create(self, ticker: str) -> TickerState:
        if ticker not in self._state:
            self._state[ticker] = TickerState()
        return self._state[ticker]

    def update_fair_a(
        self,
        ticker: str,
        fair_value: float,
        ci_width: float,
        horizon_days: int,
    ) -> None:
        with self._lock:
            ts = self._get_or_create(ticker)
            ts.fair_a = fair_value
            ts.ci_width = ci_width
            ts.horizon_days = horizon_days
            ts.last_updated = datetime.utcnow()
            self._update_lock(ts)
        logger.debug("fair_a updated: %s → %.3f (ci=%.3f)", ticker, fair_value, ci_width)

    def update_fair_b(self, ticker: str, fair_value: float) -> None:
        with self._lock:
            ts = self._get_or_create(ticker)
            ts.fair_b = fair_value
            ts.last_updated = datetime.utcnow()
            self._update_lock(ts)
        logger.debug("fair_b updated: %s → %.3f", ticker, fair_value)

    def update_market(self, ticker: str, yes_bid: float, yes_ask: float) -> None:
        with self._lock:
            ts = self._get_or_create(ticker)
            ts.market_mid = (yes_bid + yes_ask) / 2.0
            ts.last_updated = datetime.utcnow()

    def update_fill(self, ticker: str, contracts_delta: int, fill_price: float) -> None:
        with self._lock:
            ts = self._get_or_create(ticker)
            ts.net_contracts += contracts_delta
            ts.last_updated = datetime.utcnow()

    def _update_lock(self, ts: TickerState) -> None:
        if ts.fair_a is not None and ts.fair_b is not None:
            ts.strategy_lock = abs(ts.fair_a - ts.fair_b) > STRATEGY_LOCK_THRESHOLD

    @staticmethod
    def _compute_blended(ts: TickerState) -> float | None:
        if ts.fair_a is not None and ts.fair_b is not None:
            return (ts.fair_a + ts.fair_b) / 2.0
        return ts.fair_a if ts.fair_a is not None else ts.fair_b

    def replace_active_tickers(self, new_tickers: set[str]) -> None:
        """Replace the active ticker set with a fresh list from Kalshi.
        Only called when we have high confidence in the result (enough tickers returned).
        Removes any tickers not in new_tickers (settled/expired markets)."""
        if not new_tickers:
            return
        with self._lock:
            to_remove = [t for t in self._state if t not in new_tickers]
            for t in to_remove:
                del self._state[t]
            if to_remove:
                logger.info("Removed %d stale tickers from shared_state", len(to_remove))

    def evict_stale_tickers(self, max_age_seconds: int = 300) -> None:
        """Remove tickers whose market price hasn't been updated in max_age_seconds.
        Handles the case where Kalshi closes/settles markets without us explicitly
        knowing — if we stop receiving prices for a ticker, it's gone."""
        cutoff = datetime.utcnow()
        from datetime import timedelta
        threshold = cutoff - timedelta(seconds=max_age_seconds)
        with self._lock:
            to_remove = [
                t for t, ts in self._state.items()
                if ts.last_updated < threshold
            ]
            for t in to_remove:
                del self._state[t]
        if to_remove:
            logger.info("Evicted %d stale tickers (not updated in %ds)", len(to_remove), max_age_seconds)

    def blended_fair(self, ticker: str) -> float | None:
        with self._lock:
            ts = self._state.get(ticker)
            if ts is None:
                return None
            return self._compute_blended(ts)

    def should_trade(self, ticker: str) -> bool:
        with self._lock:
            ts = self._state.get(ticker)
            if ts is None:
                return False
            if ts.ci_width > self._max_ci_width:
                return False
            blended = self._compute_blended(ts)
            if blended is None:
                return False
            edge = abs(blended - ts.market_mid)
            return edge >= self._min_edge

    def get_all_tickers(self) -> list[str]:
        with self._lock:
            return list(self._state.keys())

    def snapshot(self) -> dict:
        with self._lock:
            result = {}
            for ticker, ts in self._state.items():
                result[ticker] = {
                    "fair_a": ts.fair_a,
                    "fair_b": ts.fair_b,
                    "ci_width": ts.ci_width,
                    "market_mid": ts.market_mid,
                    "net_contracts": ts.net_contracts,
                    "last_order_id": ts.last_order_id,
                    "horizon_days": ts.horizon_days,
                    "strategy_lock": ts.strategy_lock,
                    "last_updated": ts.last_updated.isoformat(),
                    "blended_fair": self._compute_blended(ts),
                }
            return result

    def full_snapshot(self) -> dict:
        """Full snapshot including alerts — used for WebSocket broadcasts.
        Markets is a dict keyed by ticker; broadcaster normalises to array."""
        return {
            "markets": self.snapshot(),  # dict — broadcaster converts to array
            "alerts": self.get_alerts(),
        }
