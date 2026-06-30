# Kalshi Market Maker

A market-making bot and research toolkit for Kalshi temperature markets.

## Why market-making

This project began as a forecasting bot (ensemble weather models → calibrated
probabilities → directional bets). Extensive evaluation showed **no trustworthy
edge**: the model's Brier (~0.14) was worse than the Kalshi market's (~0.10) in
*every* segment — station, volume decile, month, strike type — and at every point
in the trading window. The market aggregates excellent NWS/NBM guidance and is a
better forecaster than the model, end to end. Beating it on forecast skill is not
viable here.

So the project pivoted to **market-making**, which doesn't require out-forecasting
the market: quote a spread around the market's own mid, earn the spread from
flow, and manage inventory. The forecasting pipeline was removed (recoverable in
git history) and the repo slimmed to the execution + research core below.

## What we know so far (research findings)

| factor | finding | source |
|---|---|---|
| Maker fees | 25% of taker: ~0.16¢/contract in the tails, ~0.44¢ at mid | Kalshi fee schedule |
| Flow | median 8,534 contracts/market; 90% ≥ 2,000 | `kalshi_prices.parquet` |
| Adverse selection | **maker markout ≈ +0.07¢ at 1–5min** (benign, not toxic) | `trade_tape_mm.py` |
| Structural premium | ~4% overround, concentrated in longshot brackets | bracket-sum analysis |

The gating risk — toxic/informed flow that runs makers over — is **not present**
(markout is benign to slightly favorable at MM horizons). What's still unknown is
how much *spread* we can actually capture, which requires live order-book/depth
data (the next step).

## Structure

```
bot/
  config/      settings, station + series registries
  trading/     kalshi_client (REST, signed), position_tracker
  db/          SQLAlchemy models + session (orders, positions, daily PnL)
  risk/        risk controls (drawdown, exposure, cooldowns, kill switch)
  research/    market-data + viability research tools (below)
tests/         test suite
data/          market data (trades, intraday prices, market snapshots)
keys/          Kalshi RSA private key (gitignored)
```

## Research tools (`bot/research/`)

```bash
# Pull historical market data (settled markets + candlestick decision prices)
PYTHONPATH=. python -m bot.research.fetch_kalshi_history

# Market efficiency vs time-to-resolution (intraday)
PYTHONPATH=. python -m bot.research.intraday_efficiency --limit 2500

# Trades-only market-making viability probe (markout / adverse selection)
PYTHONPATH=. python -m bot.research.trade_tape_mm --limit 600
PYTHONPATH=. python -m bot.research.trade_tape_mm --no-fetch   # reuse saved tape
```

## Next: the depth logger

The trade tape cleared the gating risk; the remaining unknown (capturable spread,
fill rates, queue dynamics) needs live quote/depth data, which Kalshi only exposes
live (REST snapshot + `orderbook_delta` websocket). The next build is a
websocket order-book + trade logger to accumulate that dataset, then a realized-
spread / fill-rate analysis, then an Avellaneda-Stoikov-style quoting strategy
paper-traded against the live API.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env          # set KALSHI_API_KEY, KALSHI_PRIVATE_KEY_PATH
mkdir -p keys                 # place RSA private key at ./keys/kalshi_private.pem
PYTHONPATH=. pytest tests/ -q
```
