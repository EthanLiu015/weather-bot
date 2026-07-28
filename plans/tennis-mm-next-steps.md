# Tennis market-making — next steps

Status as of 2026-07-27 (updated; originally 2026-07-16). Context: pivoted
from weather forecasting (concluded efficient vs Kalshi, see `handoff.md`)
to a tennis market-making feasibility study. This plan covers what's left to
build before we can honestly answer "can we market-make these matches."

## RESULT: (a) spread-capture MM — not viable (2026-07-27)

Built `scripts/mm_feasibility.py`: queue-aware fill simulator (join FIFO at
best price the instant it's best, queue position = resting size ahead, walk
the trade tape to detect fills) + markout analysis, no external odds needed.
Ran on all 3 days of WS-depth data so far (07-16, 07-17, 07-18 — recorder
died 07-18 to 07-27, see [[forward_capture_programs]] for the crontab/EPERM
fix; Phase 1's 07-30 target is pushed back accordingly since 9 days of depth
data were lost to that outage).

**19,654 simulated fills, 245 market-days, all markets:**

| horizon | mean markout (net of 0.0175 maker fee) | median | win rate |
|---|---|---|---|
| 5s | -0.0324 | -0.0090 | 27.9% |
| 30s | -0.0317 | -0.0113 | 29.2% |
| 120s | -0.0300 | -0.0135 | 32.4% |

Median fill wait 4s (p90 92s) — liquidity/queue-priority isn't the
bottleneck, so this isn't a "not enough depth yet" result worth waiting on.
Fills cluster at exactly the moments the book is about to reprice against
you (adverse selection / winner's curse), and it doesn't recover by holding
longer. **Naive top-of-book queue-joining MM loses money net of fees.**
Detail: `data/capture/mm_feasibility_fills.parquet`.

This resolves the "Decision needed" section below in practice: (a) is
tested and dead. Phases 2/3 (paper-simulator, spread/depth analysis) are
superseded by this result — no need to build them for the (a) path. Only
(b) remains open, if pursued at all.

## Done

- WS tick recorder (`live/tennis_recorder.py`) rewritten: was 1s REST poll,
  top-of-book only → now push-driven via Kalshi's `orderbook_delta` +
  `ticker` channels, full price-level depth, reconnect + seq-gap resync.
  Deployed, replacing the old process.
- Fixed `trading/kalshi_client.py` request signing (was PKCS1v15 + wrong
  path, should be RSA-PSS + full `/trade-api/v2/...` path — every
  authenticated REST call was silently 401ing). Verified against live API.
- Reclaimed disk (deleted stale `data/gefs`, `data/nbm`, `data/era5` —
  re-fetchable, nothing live reads them).

## Decision needed before building further

**Directional edge vs. pure spread-capture MM.** No live point-by-point score
feed exists (`tennis_events.parquet` only has title/subtitle/close_time,
refreshed hourly). The Odds API sharp-book compare is daily-cron and
quota-capped (500 req/mo free tier) — too slow to serve as a live fair-value
oracle mid-match. Two paths:

- **(a) Spread-capture only** — quote both sides, profit from the spread net
  of fees, no claim about true win probability. Lower ceiling, doesn't need a
  score feed, buildable now.
- **(b) Directional** — needs a live score/point feed + a point-based win
  probability model (server, game/set state → prob). Higher ceiling, real
  build cost, unblocks using order-book jumps as a "the market knows
  something" signal too.

Pick one before Phase 2 — the simulator design (Phase 2) differs depending on
whether it needs a fair-value input or not.

## Phase 1 — let depth data accumulate

The WS recorder went live 2026-07-16; all prior capture was top-of-book only.
The "is there size to trade against" question restarts from zero on this
date. Target: **2026-07-30** (2 weeks) before trusting any fill-rate/depth
conclusion, same reasoning as the original settlement-calibration gate.

## Phase 2 — MM paper-simulator

Doesn't exist yet. `scripts/sniper_paper.py` simulates a *taker* (cross the
spread on a signal edge) — nothing simulates posting resting two-sided quotes
and replaying the captured ladder to estimate fills.

Needs:
- Ladder replay from `data/capture/tennis_ticks` (`snapshot` + `delta` rows
  reconstruct the book at any point in time).
- Fill logic: given a resting quote at price P, did a later delta/trade cross
  it? (Conservative: assume filled only if book traded through P, not just
  touched it.)
- Inventory tracking across a match (can't just look at isolated fills —
  need running position and exposure).
- Fee model: reuse `backtest/track_b.py` (`MAKER_FEE_COEF = 0.0175`,
  `TAKER_FEE_COEF = 0.07`, `kalshi_fee()`) — confirm Kalshi's fee schedule
  doesn't special-case sports/tennis before assuming this transfers as-is.
- If (b) was chosen above: adverse-selection modeling needs the win-prob
  model as fair value; if (a), skip that and just measure realized spread
  capture net of fees.

## Phase 3 — MM-specific analysis

`scripts/analyze_capture.py` only does Brier scoring (Kalshi vs sharp book)
and taker-signal P&L — no market-making metrics. Add:
- Realized bid-ask spread over time, by series/tour level.
- Book-move frequency/volatility as a baseline microstructure read (how often
  does top-of-book change absent any external trigger — useful even before
  Phase 0(b) is resolved).
- Depth-at-top and depth-within-N-cents distributions, to size how much could
  actually be quoted without moving the market.

## Phase 4 — directional forecasting (path (b), since (a) is dead)

Two tiers, cheap-to-expensive, both use existing tick data or things already
in the repo — neither needs `ODDS_API_KEY` (that's the daily sharp-book
Brier check in `analyze_capture.py`, a different signal):

**4a. Order-flow momentum — RESULT: strong hit (2026-07-27).**
Built `scripts/tennis_momentum_signal.py`: at every trade print, records
book imbalance and a 5-trade rolling sign-momentum, then correlates against
mid-price move 5/30/120s later. Ran full-scale on all 3 v2 days, all
markets: **108,346 observations, 204 market-days.**

| horizon | corr(trade_sign, fwd_ret) | mean fwd_ret \| sign=+1 | mean fwd_ret \| sign=-1 |
|---|---|---|---|
| 5s | +0.49 | +0.030 | -0.040 |
| 30s | +0.46 | +0.028 | -0.039 |
| 120s | +0.39 | +0.025 | -0.036 |

Passed integrity checks before trusting: (1) placebo — shuffling trade_sign
within each ticker collapses corr 0.48→0.07, rules out an alignment bug;
(2) per-market — all 20 markets in the validation sample independently show
positive correlation (range 0.26–0.67), not a few outliers; (3) excluding
the top 1% biggest moves barely changes it (0.485→0.485), not tail-driven.
This is the mirror image of the MM Result above — the same flow that makes
resting orders lose money is directionally predictable from the public tape
alone, no live score feed needed. Detail: `data/capture/tennis_momentum_obs.parquet`.

**Taker P&L — RESULT: real, but asymmetric (2026-07-27).**
Built `scripts/tennis_taker_pnl.py`: enter by crossing the spread (buy ask,
not mid) the instant a trade prints with a sign, exit by crossing again `H`
seconds later, net of `TAKER_FEE_COEF = 0.07` on both legs (4x the maker
coef that ate the MM result). Full-scale, 108,346 signals, 204 market-days.

Per-observation numbers look strong but are inflated by within-market
autocorrelation (naive t-stat ~+83, not trustworthy — same pitfall as the
correlation check). Cluster-robust: grouped by market, one mean per market,
t-stat on *that* distribution:

| side | markets (n≥10 obs) | frac. positive | mean of per-market means | cluster t-stat |
|---|---|---|---|---|
| "no" (fade upward trades / follow downward pressure) | 159 | 67% | +0.034 | 7.20 |
| "yes" (follow upward trades) | 173 | 53% | +0.009 | 2.29 |

**The edge is real but concentrated on the "no" side** — buying no when a
trade prints at the bid (yes sold) is ~3-4x stronger and more reliable than
buying yes when a trade prints at the ask. Matches the earlier momentum
asymmetry (bearish-signed trades had bigger average forward move than
bullish ones). "Yes"-side following is only borderline significant and
close to a coin flip market-by-market — don't trade it as-is. Payoff shape
is trend-following (right-skewed: median pnl negative, mean positive, most
of the return from a smaller number of large moves) — expect a real
strategy on this to look like losing most individual trades while being
profitable in aggregate, which has real psychological/drawdown implications
even if the math holds. Detail: `data/capture/tennis_taker_pnl.parquet`.

**Latency robustness — RESULT: edge does not erode (2026-07-27).**
Added `--latency_ms` to `scripts/tennis_taker_pnl.py`: entry price is now
looked up at `signal_ts + latency` instead of instantly (same real captured
book states, just a later one). Tested 250/1000/3000/5000ms on the 20-market
sample, then confirmed 1000ms at full scale (108,332 signals, 203
market-days):

| latency | "no" side cluster t-stat | "yes" side cluster t-stat |
|---|---|---|
| 0ms (baseline) | 7.20 | 2.29 |
| 1000ms | 7.97 | 2.42 |

Statistically indistinguishable from the instant-reaction baseline — if
anything marginally stronger. 250ms on the sample showed an even bigger
edge than 0ms (likely because entering at the exact signal instant catches
the book mid-walk from the triggering trade itself, a worse price than
waiting a quarter second for it to settle). **This means the edge is not a
"who reacts fastest" race** — a bot with ~1s of real-world API/network
latency is not meaningfully disadvantaged vs. instant execution, at least
up to 5s tested. Meaningfully de-risks the biggest open question from
`plans/tennis-momentum-edge-explained.md` §7.

**Still not accounted for:** 1-contract size only (no market-impact/depth
cost for larger size, unlike the queue-aware MM sim), and only 3 calendar
days of underlying data (no out-of-sample period yet — recorder is running
continuously now, so this should be re-checked once more days accumulate).

**Size/depth cost — BLOCKED on a capture-side data-fidelity gap
(2026-07-27).** Built `scripts/tennis_size_cost.py` to walk the actual
resting-order ladder (VWAP execution price) instead of assuming infinite
depth at the best price. Result was ~0% win rate at every size, including
1 contract — too extreme to be real, so it was checked rather than
reported. Root cause: our offline replay of `yes_book`/`no_book` from
`snapshot`+`delta` rows drifts significantly from the true book — checked
against the recorder's own authoritative cached `yes_bid`/`yes_ask` on one
market's full day: **79% of ticker updates mismatched our reconstructed
book, by 14.6¢ on average (up to 63¢).** The parquet never stored the
WebSocket seq/sid numbers the *live* recorder uses internally for gap
detection (`live/tennis_recorder.py` `last_seq`/`needs_resync`), so an
offline replay has no way to detect a dropped/misordered delta and silently
drifts until the next lucky snapshot.

This does **not** invalidate the momentum/taker-P&L results above — those
only ever use book reconstruction for the initial trade-sign match (tight
tolerance, self-validating: if it matched, the book was in sync at that
instant) and price everything off the recorder's cached fields directly,
never off our own reconstructed ladder. It does mean:
- `mm_feasibility.py`'s queue-position/fill-timing sim (Result at the top
  of this file) also leans on reconstructed book *size*, not just price,
  over the multi-second life of a resting order — same exposure, not
  separately audited. Its negative result (MM loses) is the pessimistic
  direction already, so this is a lower-priority recheck than a positive
  result would be, but worth knowing about.
- `tennis_size_cost.py` cannot answer the size question with the currently
  captured 07-16/17/18 data — this needs a capture-side fix, not a
  script-side one: have `live/tennis_recorder.py` persist `seq`/`sid` per
  row (enabling real gap detection in offline replay, mirroring what the
  live process already does) and/or snapshot on a timer, not only at
  subscribe/resync, so drift self-heals faster. Full explanation of what
  was checked and why: `plans/tennis-momentum-edge-explained.md`.

**Recorder fix — DEPLOYED (2026-07-28).** `live/tennis_recorder.py`:
added `seq`/`sid` to `ROW_COLUMNS` (populated from every WS message,
confirmed live: 14,840/15,262 rows in the first post-restart part carry
them — the small remainder is expected, not every message type does), and
`periodic_resync_loop` forces a full reconnect+resubscribe every
`PERIODIC_RESYNC_S = 900` (15 min) via the existing `needs_resync` flag —
reused the already-proven seq-gap resync path rather than inventing a new
one, so this piggybacks on tested behavior. Restarted the live process to
pick it up. This bounds future drift to at most 15 minutes between
resnapshots and gives offline replay the seq numbers to actually detect a
gap, but **does not fix the already-captured 07-16/17/18 data** — the size
question stays open until enough post-2026-07-28 data accumulates under
the new schema. Old ticks files (pre-07-28) lack `seq`/`sid` — that's
expected schema evolution, not a bug, don't treat their absence there as
another mismatch to chase.

**4b. Live score-based win probability (the original Phase 4 scope).**
Needs a live point-by-point score source (provider TBD — not yet scoped)
and a point-based win probability model (server, game/set state → prob),
scored against Kalshi's live price for divergence. Higher ceiling, real
build cost (comparable in scope to the weather NBM/ensemble forecasting
work already done in this repo — see `handoff.md`). Don't start this until
4a's result is in: if order-flow momentum alone shows nothing, that's weak
evidence the market already reacts to score changes near-instantly, which
lowers the expected payoff of building 4b before deciding it's worth the
build cost.

## Phase 5 — tennis-specific execution wiring — BUILT, paper-only (2026-07-28)

Implemented per `/Users/ethan/.claude/plans/reactive-greeting-globe.md`
(full design rationale there): `trading/tennis_order_manager.py` (one-shot
"no"-side taker, fixed size, fixed 30s hold, settlement-aware exit) +
`live/tennis_signal_bot.py` (separate WS process, reuses `Book` from
`live/tennis_recorder.py`, sign-detection ported from
`scripts/tennis_momentum_signal.py`) + `config/settings.py` tennis knobs +
`tests/test_tennis_order_manager.py` (9 new tests, TDD, all passing;
32/32 full suite passes, no regressions). `TENNIS_ENABLED=False` by
default — confirmed the bot exits cleanly and immediately when disabled,
writes no pidfile, opens no WS connection. Not cronned yet — that's a
deliberate next step after a manual paper-trading review run (see
verification steps in the plan file).

Original scope notes below, superseded by the above:

`trading/order_manager.py` / `strategies/*` are weather-specific
(`blended_fair` from NBM/ensemble blend, filter-cascade tuned for daily-high
brackets) — not reusable as-is. Need a parallel module for tennis:
- Quote-both-sides-and-manage-inventory loop (current `OrderManager` is a
  one-shot directional edge-taker shape, structurally different from an MM
  loop).
- Tennis-specific risk config (`config/settings.py` has zero tennis knobs
  today — no position limits, no MM-specific settings).
- Settlement edge cases: confirm walkover/retirement handling before holding
  any position into a match — a retirement mid-match settles differently
  than a clean completion and could catch a resting MM position off guard.

## Phase 6 — stress test at scale

WS subscription verified at ~82 concurrent markets. Slam/peak tournament
days can run 300+ concurrent. Confirm the recorder holds up (reconnect
behavior, subscription-chunking logic in `Recorder.subscribe_new`) before
trusting depth data on a big day.

## Phase 7 — graduate to paper trading, then live

Once Phases 2–6 are in place and Phase 1's 2-week data confirms the
liquidity/spread picture is viable: run the Phase 2 simulator live in
shadow/paper mode for a real sample before risking size. `PAPER_TRADING=true`
in `.env` already gates real order submission — keep it there until paper
P&L across a real sample (not a handful of matches) supports it.
