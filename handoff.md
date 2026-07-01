# Handoff — Kalshi trading research

_Last updated: 2026-07-01. This repo began as a weather-forecasting trading bot,
pivoted to market-making research, and is now exploring a sports/sportsbook angle.
The old forecasting handoff was deleted in the pivot; this is a fresh one covering
the current state and how we got here._

---

## 🧭 START HERE — current state & the one live decision

**Where we are:** two candidate edges have been rigorously tested and **both came up
empty**, and we've turned to a third:

1. **Weather forecasting** → NO edge (the market out-forecasts us in every segment).
2. **Market-making on Kalshi weather** → a naive/moderate strategy **loses** once
   inventory + exit costs are integrated (full round-trip backtest is negative).
3. **Sports: sportsbook-line vs Kalshi +EV** → the current lead. Not yet built/tested.

**The collector is still running** (72h websocket depth logger, started 2026-06-30,
~600 book shards in). Keep it running — the MM analyses below are on ~2–4h of
partial data and want the full multi-day set to firm up.

**The one live decision (for the user):** the sports strategy needs an **odds API
key** (the sharp reference). The Odds API has a free tier (~500 req/mo). With it we
can build + measure the actual edge; without it we can only build the Kalshi side.
→ See "Next steps" §7.

---

## 1. The big-picture conclusions (why we are where we are)

### Weather forecasting is efficient — no edge (settled, code deleted)
The forecasting model (NGBoost+QRF+ECMWF anchor, blended MAE ~3.1°F) was rigorously
evaluated against real Kalshi bracket prices: **model Brier 0.14 vs market 0.096, no
edge in ANY segment** (station / volume decile / month / strike type) and at no point
in the trading window. The deficit is structural, not model quality:
- **Information horizon:** we forecast at 24h lead; the market prices off ~6h-lead
  guidance + live obs, and aggregates the same public NWS/NBM data.
- **Bracket precision wall:** 2°F-wide brackets need sub-1°F sharpness.
- σ-calibration is fine (z_std 1.01); the ML is fine; the market is just a better
  short-horizon predictor with fresher data.
Improving the model can at best *match* the market, not beat it. The whole
forecasting/weather pipeline was deleted (recoverable in git before commit
`b27713e`).

### Market-making on Kalshi weather — naive strategy loses (the decisive result)
The full round-trip backtest (`bot/research/backtest.py`) — two-sided quoting,
queue-aware fills, inventory, passive + forced-cross exits, fees — is **negative at
every queue-priority level**: −1.49¢/contract (passive), −0.58¢ (25% front-of-queue),
−0.47¢ (50%). The piecewise analyses looked positive in isolation, but **inventory /
adverse-selection cost** (you accumulate losing positions in one-sided/informed flow
and pay to flatten) outweighs the spread capture. See §5 for the full chain and the
nuances (a *sophisticated* inventory-skewing strategy might claw some back, but the
baseline is clearly negative — the bar is high).

### Sports (current lead — untested)
Kalshi has a large **Sports** category (2,280 series). The realistic retail edge is
NOT a better model — it's **using sharp sportsbook lines (Pinnacle / no-vig
consensus) as truth and betting Kalshi when it's mispriced past fees.** This inverts
the weather problem: the sharp reference lives *outside* Kalshi, so you have a better
"model" for free, and Kalshi (retail, laggy) misprices against it. Fits the user's
profile (sports knowledge, small capital, automated: the +EV single-venue version
needs sportsbook *data*, not a sportsbook *account*). Known but competed; edges are
thin and appear around line moves. **Not yet built.**

---

## 2. Repo structure (post-pivot)

```
bot/                     # the package (import as `from bot.x import ...`)
  config/                # settings, station + series registries
  trading/               # kalshi_client (REST, signed), position_tracker
  db/                    # SQLAlchemy models + session (orders/positions/pnl; create_all)
  risk/                  # risk controls (drawdown, exposure, cooldowns, kill switch)
  marketdata/            # LIVE feed: orderbook.py (book replica), depth_logger.py, audit.py
  research/              # all analysis tools (below)
tests/                   # 49 passing
data/
  marketdata/            # depth logger output (book/ + trades/ parquet shards; GITIGNORED)
  historical/            # kalshi_prices.parquet, trades.parquet, intraday_prices.parquet
keys/                    # Kalshi RSA private key (gitignored)
```
Run everything with `PYTHONPATH=.`. Deps trimmed to numpy/pandas/pyarrow/httpx/
websockets/cryptography/sqlalchemy/pydantic. Project renamed `kalshi-market-maker`.

---

## 3. The data collection (depth logger)

Kalshi exposes order-book depth only LIVE. `bot/marketdata/depth_logger.py` connects
to the websocket, subscribes to `orderbook_delta` + `trade` for all active temperature
markets, maintains a local `OrderBook` per market, and flushes top-of-book CHANGES +
trades to parquet shards under `data/marketdata/`.

- **WS gotchas (fixed):** host is `external-api-ws.kalshi.com` (NOT the REST host,
  which 404s); auth requires **RSA-PSS** (REST's PKCS1v15 gets 401). `seq` is
  **global per channel**, not per-market (gap detection is connection-level).
- **Commands:**
  - `PYTHONPATH=. python -m bot.marketdata.depth_logger --hours 72` — collect
  - `... --smoke` — print raw frames (verify feed)
  - `... --validate` — cross-check the live WS book vs REST + invariants (PASS gate)
- **Monitor / stop:** `ls data/marketdata/book | wc -l`; `pkill -f "depth_logger --hours"`.
- Schema: book = ts, ticker, yes_bid, yes_ask, yes_bid_sz, yes_ask_sz, spread (cents/
  contracts); trades = ts, ticker, yes_price (float), count, taker_side.

---

## 4. ⚠️ Data-quality bugs found & fixed (the hard-won lessons)

Data correctness was a repeated theme (the forecasting project was poisoned by silent
data bugs). The **validate gate** (WS vs REST) and the **audit tool** (on persisted
data) caught real bugs — always run both before trusting the data:
- `PYTHONPATH=. python -m bot.marketdata.audit` — audits shards on disk (crossed
  books, price/size ranges, feed holes, bad trades). Re-runnable anytime.

Bugs caught (all fixed, tested):
1. **Per-market seq-gap detection** → 11k false "gaps" (seq is global). Fixed: connection-level.
2. **Floating-point dust phantom levels** → 168 crossed books. Summing deltas lands on
   ~1e-15 not 0, leaving a phantom best level. Fixed: drop levels ≤ 1e-6.
3. **Flush filename collision** (second-resolution) → overwrote shards on reconnect. Fixed: counter.
4. **Trades stored as strings** → coerce to float.
5. **Stale ticker universe** → a multi-day run never picked up new days' markets. Fixed:
   periodic resubscribe (30 min).
6. **Unguarded dispatch** → one bad frame forced a reconnect. Fixed: try/except.

Also two NBM data bugs (in the now-deleted forecasting code, but the lesson stands):
boustrophedon value-scrambling and a date/window off-by-one — both silent, both caught
by auditing values against reality. **Audit everything against an independent source.**

---

## 5. The market-making research chain (`bot/research/`, on partial data)

Run order = the logical build-up. Each is unit-tested; all read `data/marketdata/`.

| tool | what it measures | finding (partial ~2–4h data) |
|---|---|---|
| `trade_tape_mm.py` | markout / adverse selection from the trade tape alone | flow is benign (+0.07¢ at 1–5min) — not toxic. GREEN gate to build the logger. |
| `mm_edge.py` | net maker edge vs the real book MID, per unwind horizon; segmented | +2.5¢ gross at 0s → ~0 by 5s → negative by 30s. Edge lives in tail/near-certain brackets, 2–6¢ spreads, specific markets. |
| `fill_model.py` | queue-aware capacity + fill selection; front-of-queue φ sweep | passive back-of-queue LOSES (−0.13¢@5s, captures ~5% of volume); break-even needs **~18% front-of-queue** priority. |
| `contestedness.py` | competition proxies (touch depth, quote churn) × edge, by market | quiet inland markets (LV/MIN/PHX/OKC/SEA) = thin+slow = front-of-queue by presence (no speed race). Liquid ones (NY/LAX/MIA) = deep+fast = contested. |
| `profit_estimate.py` | $ estimate: daily_vol × capture × net/contract | ~$4–9k/mo GROSS (model), but noisy (one market dominates) & optimistic; not scalable. |
| `exit_cost.py` | realistic exits (passive when favorable, cross when adverse) | edge SURVIVES exit costs at ≤5s (+0.11¢ overall; positive in most niche markets). |
| `backtest.py` | **full round-trip** (queue + inventory + exits + fees) | **DECISIVE: NEGATIVE** −0.47 to −1.49¢/contract at all φ. Inventory cost kills it. |

**Bottom line of the chain:** the spread is capturable in isolation, but a realistic
two-sided MM loses on this market once inventory + adverse selection are integrated —
even in the quiet niche, even with priority. The latency/priority race in the liquid
markets is one we (retail hardware) can't win; the presence-based niche is thin and
the full backtest still says negative.

**Key mechanic learned:** MM viability ≈ fast unwind (seconds) + queue priority +
inventory control. All three are needed and the integration is worse than any single
piece suggests.

---

## 6. Other angles already checked (dead or thin)
- **Bracket internal-consistency arbitrage:** prices sum to ~1.0 (efficient); the ~4%
  overround (favorite-longshot) is maker-only ≈ the spread — not a taker edge.
- **Time-of-day / intraday:** market doesn't trade before ~D-1 evening; sharp from the
  moment it's liquid; our model is worse than the market at every time. No timing edge.
- **Kalshi market scan:** ~45k "open" markets are mostly **dead auto-generated parlays**
  (KXMVESPORTSMULTIGAMEEXTENDED, KXMVECROSSCATEGORY) with zero volume. The list
  endpoint is sparse (ticker-only); real liquidity is in individual game/event markets.
  Categories via `/series`: Sports 2280, Entertainment 2460, Politics 2020, Elections
  1420, Financials 600, Economics 584, Crypto 253, Weather 285, …

---

## 7. Next steps (prioritized)

### A. Sports: sportsbook-line vs Kalshi +EV (the current lead)
1. **[needs odds key]** Get an odds feed (The Odds API free tier, or Pinnacle). This is
   the gating input — the sharp "true probability."
2. **Kalshi sports fetcher** (buildable now): pull the *liquid* individual sports-game
   markets + live prices/orderbooks via the `/events` + `/series` endpoints (category
   Sports), skipping the dead parlays. Assess how much liquidity/volume actually exists.
3. **Edge engine:** map Kalshi market ↔ sportsbook event; compute no-vig fair prob from
   sharp lines; flag Kalshi markets mispriced past fees (Kalshi fee = 0.07·p·(1−p)
   taker); size with fractional Kelly.
4. **Backtest / paper-trade** the +EV bets before any real money; measure realized edge
   vs the sharp line, and how fast edges decay (line-move latency).
   - Honest caveat: this edge is competed and thin; +EV (unhedged single-venue) has
     variance — needs bankroll discipline. Validate the magnitude before committing.

### B. Market-making (only if revisiting)
- Re-run the whole `bot/research/` chain on the **full multi-day** collection to regress
  the 2–4h noise (esp. per-market outliers like Minneapolis in profit_estimate).
- A *sophisticated* inventory-skewing / quote-inside / Avellaneda-Stoikov strategy is
  the only version with a chance, and it must recover >0.5¢/contract against informed
  flow — high bar. Probably not worth it given the negative baseline.

### C. Housekeeping
- Collector is running; let it finish (or extend). Data in `data/marketdata/` (gitignored).
- `PYTHONPATH=. pytest tests/ -q` → 49 passing.

---

## 8. Useful commands
```bash
# collection + data integrity
PYTHONPATH=. python -m bot.marketdata.depth_logger --hours 72     # collect
PYTHONPATH=. python -m bot.marketdata.depth_logger --validate     # live WS vs REST gate
PYTHONPATH=. python -m bot.marketdata.audit                       # audit persisted shards

# MM research chain (run on collected data)
PYTHONPATH=. python -m bot.research.mm_edge
PYTHONPATH=. python -m bot.research.fill_model
PYTHONPATH=. python -m bot.research.contestedness
PYTHONPATH=. python -m bot.research.exit_cost
PYTHONPATH=. python -m bot.research.backtest        # the decisive round-trip number
PYTHONPATH=. python -m bot.research.profit_estimate

# tests
PYTHONPATH=. pytest tests/ -q
```

Memory (Claude, cross-session): see the `market-making-pivot` note — the older weather
memories describe now-deleted code and are historical only.
