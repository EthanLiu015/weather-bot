"""Local replica of a Kalshi binary-market order book.

Kalshi posts two resting-bid sides for each market: `yes` (bids to BUY yes) and
`no` (bids to BUY no). A NO bid at price n cents is equivalent to a YES OFFER at
(100 - n) cents, so:

    best YES bid  = max(yes prices)
    best YES ask  = 100 - max(no prices)

The book is seeded by an `orderbook_snapshot` and maintained by incremental
`orderbook_delta` messages. Each delta carries a monotonically increasing `seq`;
a gap means we missed a message and the local book is stale until re-snapshotted.

Prices are tracked in integer cents; quantities in contracts.
"""
from __future__ import annotations

from typing import Iterable, Optional


def to_cents(price) -> int:
    """Kalshi sends dollar strings ("0.0800"); normalise to integer cents."""
    return int(round(float(price) * 100))


class OrderBook:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.yes: dict[int, float] = {}
        self.no: dict[int, float] = {}
        self.seq: Optional[int] = None
        self.stale: bool = False

    def apply_snapshot(
        self, yes_levels: Iterable, no_levels: Iterable, seq: Optional[int] = None
    ) -> None:
        self.yes = {to_cents(p): float(q) for p, q in yes_levels if float(q) > 0}
        self.no = {to_cents(p): float(q) for p, q in no_levels if float(q) > 0}
        self.seq = seq
        self.stale = False

    def apply_delta(
        self, price, delta, side: str, seq: Optional[int] = None
    ) -> None:
        if seq is not None and self.seq is not None and seq != self.seq + 1:
            # Missed a message — the local book can no longer be trusted until a
            # fresh snapshot is requested.
            self.stale = True
        book = self.yes if side == "yes" else self.no
        c = to_cents(price)
        new_q = book.get(c, 0.0) + float(delta)
        if new_q <= 0:
            book.pop(c, None)
        else:
            book[c] = new_q
        if seq is not None:
            self.seq = seq

    @property
    def yes_bid(self) -> Optional[int]:
        return max(self.yes) if self.yes else None

    @property
    def yes_ask(self) -> Optional[int]:
        return (100 - max(self.no)) if self.no else None

    @property
    def spread(self) -> Optional[int]:
        b, a = self.yes_bid, self.yes_ask
        return (a - b) if (b is not None and a is not None) else None

    def top(self) -> dict:
        """Best-quote snapshot row (cents / contracts), for logging."""
        b, a = self.yes_bid, self.yes_ask
        return {
            "yes_bid": b,
            "yes_ask": a,
            "yes_bid_sz": self.yes.get(b) if b is not None else None,
            "yes_ask_sz": self.no.get(100 - a) if a is not None else None,
            "spread": (a - b) if (b is not None and a is not None) else None,
            "stale": self.stale,
        }
