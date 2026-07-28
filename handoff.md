# Handoff — Weather forecasting v2, autoresearch session (2026-07-01 night)

_Branch: `forecasting-v2`. This file replaces the previous handoff (recover via git
history; its durable conclusions are carried in "Frozen Decisions" below)._

---

## 1. Project Objective — CONCLUDED: PLATEAU / BLOCKED (two loops)

Goal was: beat the Kalshi KXHIGH market on calibration — **model Brier < market
Brier (0.0951)** on the real-market eval window (Apr 11 → May 27 2026 test split),
using only pre-14:00-UTC information. Two autoresearch loops ran:
`orchestrator-260701-1933` (cycles 0–7, pure models + blend, **PLATEAU**) and
`orchestrator-260701-2330` (cycles 8–9, market-input models, **BLOCKED**).

**Final verdict — the predicate is unreachable, with a four-part proof chain:**
1. Pure fair models exhausted at 0.1193 (NBM/ensembles/members/obs/truth).
2. Shrinkage blend w = 0.000 — the model has zero orthogonal information.
3. `d1_mid` itself is well-calibrated: renorm/isotonic/logistic/stacking all
   lose on test (in-sample recalibration gain ≈ 0.0001 — nothing to extract).
4. The book's one real flaw (tick floor: 0.01-brackets settle 0.44 %) is worth
   ≤ 0.00005 Brier — an order of magnitude below significance. Intraday
   denoising data no longer exists (Kalshi rolling purge).
0.0951 is the book's irreducible level on this window.

## 2. Cycle Results (all honest: train-only fitting, test scored once, block bootstrap)

| Cycle | What | Test Brier | Market | Verdict |
|---|---|---|---|---|
| 0 | Baseline fresh ensemble + per-station σ + isotonic | 0.1211 | 0.0951 | loses |
| 1 | Inverse-MSE weights + EMOS (`research/ensemble_upgrade.py`) | 0.0984 | 0.0951 | loses (and data later shown look-ahead) |
| 2 | Walk-forward refit on fresh data (`research/ensemble_walkforward.py`) | 0.0914 | 0.0951 | INVALID — post-cutoff look-ahead; stale24 variant 0.1213 |
| 3 | NBM 07Z/12Z station bulletins (`research/nbm_edge.py`) | 0.1195 | 0.0951 | loses, P(model better)=0.00 |
| 4 | Settlement-truth audit (`research/settlement_truth.py`) | — | — | alarm was artifact; truth confirmed clean (below) |
| 5a | Ensemble member PDFs (`research/ensemble_pdf.py`) | — | — | BLOCKED: no fair member data on Open-Meteo |
| 5b | Morning-obs conditioning (`research/obs_conditioning.py`) | 0.1202–0.1219 | 0.0951 | loses, P=0.00; runmax truncation worth only 0.0017 |
| 6 | Market-shrinkage blend (`research/blend_fallback.py`) | 0.0951 (=mkt) | 0.0951 | w*=0.000 — zero orthogonal info |
| 7 | TRUE member PDFs, fair (GEFS 31 via dynamical.org Zarr + ECMWF IFS ENS 51 via Icechunk, 00Z inits) | 0.1193 | 0.0951 | loses, P=0.00; = cycle-3 NBM-only; shape worthless |
| 8 | Market self-recalibration C0–C4 (`research/market_recalibration.py`): simplex renorm, isotonic, logistic, stacking | 0.0952–0.0960 | 0.0951 | all lose; d1_mid is well-calibrated; renorm of the 1.04 vig-oversum HURTS |
| 9a | Microstructure denoising (VWAP / quote-mid of pre-14:00 trades) | — | — | BLOCKED: Kalshi purged Apr–May trades + candles; captured bid/ask 100 % degenerate |
| 9b | Tick-floor tail sharpening (0.01 settles 0.44 % — real 2.3× overpricing) | — | — | analytic kill: perfect tail pricing ≤ 0.00005 Brier, below any significance. **Loop closed: BLOCKED** |

## 3. Durable Findings (this session)

- **The −18.75 °F "ERA5 corruption" alarm was an artifact.** The throwaway
  implied-tmax script mixed **KXLOWT (daily-low)** brackets into the recovery.
  KATL/KIAH have NO KXHIGH series at all (only KXLOW). With HIGH-only recovery,
  `features.parquet::actual_tmax` falls inside the settled bracket on **100 % of
  1206 station-days** — training truth already equals settlement truth. No
  recalibration was needed; cycles 1–3 numbers stand.
- **Kalshi settles NY on KNYC (Central Park) and Chicago on KMDW (Midway)**, not
  KLGA/KORD (proven: official CLI highs for KNYC/KMDW sit inside the settled
  bracket 100 % vs 50/59 % for KLGA/KORD). Yet re-pointing NBM at the true
  stations does NOT help (KNYC MAE 1.96 vs KLGA 1.77; KMDW 2.04 vs KORD 2.13) —
  NBM's station calibration already absorbs the mismatch.
- **No usable ensemble-member data exists on Open-Meteo at all** for our window:
  `previous_dayN` vars return all-null for every ensemble slug, and the plain
  assembled archive is a rolling ~2-week window — April/May 2026 is gone, fair
  or unfair. **The working member source is dynamical.org**: NOAA GEFS 35-day
  Zarr (`https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr`,
  31 members, inits 2020-10→present, `maximum_temperature_2m`) and ECMWF IFS ENS
  Icechunk (`dynamical_catalog.open("ecmwf-ifs-ens-forecast-15-day-0-25-degree")`,
  51 members, 00Z-only inits since 2024-04). Both fair (00Z posts pre-cutoff);
  point reads are KB-scale.
- **Member PDFs don't help (cycle 7).** 82 pooled fair members: raw member-mean
  MAE ~3.3 °F (grid representativeness; KLAS/KSFO/KPHX worst), member spread adds
  nothing over NBM's own `xnd` (P3 0.1193 ≈ cycle-3 0.1195), and the empirical
  member CDF is WORSE than a calibrated Gaussian. Distribution shape was the last
  pure lever.
- **Morning obs don't close the gap.** Hourly METARs from the true settlement
  stations (obs ≤ 14:00 UTC; runmax coverage 1.00): physical truncation at the
  morning running max improves Brier by only 0.0017; trajectory/warming-rate
  regressors add nothing. The book's sharpness does not come from morning obs.
- **The market's edge is not run freshness, not NBM, not obs, not truth quality.**
  Everything legal we stack reaches ~0.12; the book sits at 0.095. Cycle 2's
  fresh-data 0.0914 shows post-cutoff model runs DO explain the book's level —
  i.e. the 14:00 UTC price already impounds information equivalent to runs that
  only get published later. Consistent with w*=0: the book is simply efficient.

## 4. Files Created This Session

- `research/settlement_truth.py` — implied-tmax recovery (KXHIGH-only) + IEM CLI
  cross-check; wrote `data/historical/cli_truth.parquet`
- `scripts/backfill_ensemble_members.py` — ensemble-member backfill (returns 0
  rows — Open-Meteo has no fair member data; kept for AWS-GRIB future)
- `research/ensemble_pdf.py` — member-PDF eval P0–P3 (blocked on data)
- `scripts/backfill_obs_hourly.py` — IEM ASOS hourly METARs, settlement-station
  mapped (KLGA→NYC, KORD→MDW); wrote `data/historical/obs_hourly.parquet`
- `research/obs_conditioning.py` — cycle 5 M0–M3 (obs features, runmax truncation)
- `research/blend_fallback.py` — cycle 6 shrinkage blend, closed-form wf weight

## 5. Frozen Decisions (carried forward; do NOT re-litigate)

- **Fairness rule:** only information issued ≤ 14:00 UTC. `openmeteo_fresh.parquet`
  banned for claims. NBM runs ≤ 12Z legal. Previous-runs 24 h data legal.
- **Honesty protocol:** walk-forward params from station-days < d; variant
  selection on train; test scored once; block-bootstrap over dates; positive
  claims need P ≥ 0.95 + leakage audit.
- **Dead ends (audited, killed):** 24 h/multi-lead edge; intraday afternoon
  obs-conditioning; between-NO fade; mid-spread fade; run-freshness (cycle 3);
  settlement-truth recalibration (cycle 4); morning-obs conditioning (cycle 5);
  ANY pure-model edge and ANY blend edge (cycle 6, w=0).
- **Truth:** `actual_tmax` = settlement truth (verified). Kalshi settlement
  stations: NY=KNYC, CHI=KMDW, rest match ICAO. KATL/KIAH have no KXHIGH series.
- **Fee model:** `kalshi_fee = size · min(coef·p·(1−p), 0.035)`, 0.07 taker /
  0.0175 maker.
- **Eval spine:** `backtest/real_market_eval.py` (`_load_eval_markets` excludes
  KXLOWT, `brier_score`), window EVAL_START..EVAL_END, split via
  `research/ensemble_upgrade.temporal_split`.

### Reproduce-this-session commands
```bash
PYTHONPATH=. python -m research.settlement_truth      # cycle 4 audit
PYTHONPATH=. python scripts/backfill_obs_hourly.py    # rebuild obs_hourly.parquet
PYTHONPATH=. python -m research.obs_conditioning      # cycle 5
PYTHONPATH=. python -m research.blend_fallback        # cycle 6 (w=0 result)
```

---

## Session 2026-07-06 — edge autopsy + cross-market scan (orchestrator-260706-goalAB)

**Why others "win" KXHIGH (Goal A):** public bots are paper-trading or n=16 (one
is methodologically our cycle 7, archived April 2026). The credible postmortem
(northlakelabs) says winners are **latency arbs executing within seconds of each
NWS model cycle**; fees kill sub-15¢ contracts. This reconciles every negative we
have: cycle-2's post-cutoff "win" (0.0914) is exactly the information those bots
harvest intra-day; every static snapshot we can backtest is post-race (hence
blend w=0). A "similar strategy" = live model-cycle sniper — unbacktestable
(ticks purged), forward-paper-trade only; we own pricing/calibration/fees,
missing websocket book + release trigger + execution loop.

**Cross-category scan (Goal B):** `scripts/kalshi_market_scan.py` →
`autoresearch/orchestrator-260706-goalAB/{report.md, market-scan-results.tsv}`.
Highlights: BTC-bracket longshot "anomaly" was pseudo-replication (one event
settles a whole strike ladder — killed, but crypto-book staleness vs 24/7 spot is
the forward-testable cousin of weather sniping); WTA shows favorite-longshot-bias
direction (n=60 events, needs sharp-book cross-check); new HOURLY temp markets
(TWC-settled, KNYC) are calibrated but have ZERO liquidity; CPI/Fed books
untradable. **Structural: Kalshi's ~10-week purge caps every backtest — any
future program must forward-capture daily (settled + candles + books).**

## Forward-capture programs (BUILT + CRON-INSTALLED 2026-07-06)

User crontab now runs four jobs (was empty before; `crontab -l` to inspect,
`crontab -r` to remove; macOS cron skips runs while the machine sleeps):
- `scripts/capture_books.py` hourly :05 — top-of-book snapshots (BTC/ETH/WTA/
  NYC+CHI highs/TSA) → `data/capture/books.parquet`. Feeds crypto-staleness study.
- `scripts/capture_daily.py` daily 10:00 — newly settled markets + probe candle
  (close−24h/mid-life) for 24 series → `data/capture/settled_probe.parquet`.
  Beats the ~10-week API purge; calibration sample grows forever.
- `scripts/sniper_paper.py` every 30 min — PAPER model-cycle sniper: latest NBM
  NBS bulletin per station (IEM) → bracket fairs (per-station bias + EMOS from
  local history, per-target-date txn/xnd, integer-vs-x.99 strike semantics) vs
  live books, net-of-fee edges logged → `data/capture/sniper_signals.parquet`.
  NO orders. Mid-afternoon "edges" are stale-model illusions BY DESIGN — the
  analysis buckets by signal age; only fresh-after-release buckets test the
  latency hypothesis.
- `scripts/tennis_odds_compare.py` daily 09:00 — Kalshi ATP+WTA+challenger books vs de-vigged
  sharp lines (The Odds API; set `ODDS_API_KEY` env in crontab to activate
  matching, else logs Kalshi-only) → `data/capture/tennis_compare.parquet`.

### Live tennis tick infrastructure (added 2026-07-06, rewritten to v2 on 2026-07-16)
- `live/tennis_recorder.py` — **v2 (2026-07-16): WS push, not REST polling.**
  Single WebSocket (`orderbook_delta` + `ticker` channels) across all open
  tennis markets (ATP/WTA/challengers, ~74–244 concurrent) — full
  price-level depth (snapshot + signed deltas), not just top-of-book; real
  push latency, not a 1s poll floor. Replaced the v1 REST-poll recorder
  described in the original 2026-07-06 note below (kept for the CloudFront
  cache-bust discovery, which no longer applies — WS has no such caching
  layer). Rows: `type` ∈ {snapshot, delta, ticker} →
  `data/capture/tennis_ticks/date=YYYY-MM-DD/part-*.parquet`; "who is
  playing" metadata → `data/capture/tennis_events.parquet`. Pauses itself if
  free disk < 2 GB. Stop: `kill $(cat data/capture/tennis_recorder.pid)`.
- **2026-07-28 fixes** (see `plans/tennis-mm-next-steps.md` "Recorder fix"
  for full detail): (1) the `*/5 * * * *` watchdog cron line was silently
  `EPERM`-failing every run for 10 days straight (macOS blocks cron from
  directly exec'ing a script under `~/Documents`; errors went to local
  mail, not a log) — fixed by wrapping the cron line through `/bin/bash`
  explicitly. (2) Added `seq`/`sid` columns (populated from every WS
  message) and a 15-min `periodic_resync_loop` (forces a full
  reconnect+resubscribe via the existing seq-gap `needs_resync` path) —
  offline replay of the 07-16/17/18 data (before this fix) was found to
  drift 79% of the time by 14.6¢ average vs the recorder's own
  authoritative top-of-book, because there was no way to detect a
  dropped/misordered delta after the fact. Data captured before 2026-07-28
  should not be trusted for anything beyond top-of-book (which the
  recorder's own cached `yes_bid`/`yes_ask` fields on each row are fine
  for) — full ladder-depth analysis on that window is unreliable.
- `scripts/tennis_watchdog.sh` (cron */5, fixed 2026-07-28) restarts the
  daemon; macOS sleep still gaps the data. `scripts/tennis_compact.py`
  (cron 03:30) merges finished days into one file per day — also had a
  dtype bug (recorder's price format changed string→float mid-day on
  07-16) fixed 2026-07-27, `NUMERIC_COLUMNS` now coerced via
  `pd.to_numeric` before merge.

**Scoring: after ≥2 weeks run `PYTHONPATH=. python scripts/analyze_capture.py`**
— sniper paper P&L by signal-age bucket, tennis Brier kalshi-vs-sharp, accumulated
per-series calibration (event-clustered). Everything under `data/capture/`
(gitignored data; logs in `data/capture/logs/`).

## Session 2026-07-27/28 — tennis MM dead, order-flow taker edge found + built (paper-only)

Full detail and exact numbers: `plans/tennis-mm-next-steps.md` (research log)
and `plans/tennis-momentum-edge-explained.md` (plain-language writeup of the
edge, its asymmetry, and every way it could still be wrong). Summary:

- **Passive market-making (spread capture) — DEAD.** Built
  `scripts/mm_feasibility.py`: queue-aware fill simulator (join FIFO at best
  price, walk the trade tape to detect fills) + markout. 19,654 fills,
  245 market-days: mean markout **negative at every horizon** (5s/30s/120s),
  net of the 0.0175 maker fee, win rate 28–32%. Fills cluster exactly when
  the book is about to reprice against the resting side — classic adverse
  selection. Don't reopen this path.
- **Order-flow momentum taker — REAL, but one-sided.** Mirror question:
  since resting orders reliably lose to the flow, is that flow *takeable*?
  `scripts/tennis_momentum_signal.py` (screening) →
  `scripts/tennis_taker_pnl.py` (fee-netted P&L, cluster-robust). 108,346
  signals, 204 market-days. Buying "no" when a trade prints at the bid
  (bearish pressure): cluster t-stat (one mean per market, not per-trade —
  naive per-trade stats were inflated ~10× by autocorrelation) **7.2–8.0**,
  67% of markets individually profitable. The mirror "yes"-side (following
  bullish prints) is weak/inconsistent (t≈2.3, 53% of markets) — **not
  traded**. Payoff is trend-following-shaped: median trade loses, mean
  trade wins.
- **Latency-robust up to 5s tested** (`tennis_taker_pnl.py --latency_ms`) —
  entry priced at signal_ts+latency instead of instantly; cluster t-stat
  at 1000ms (7.97) was *stronger* than instant (7.20), not weaker. This
  isn't a "who reacts fastest" race — real execution latency doesn't
  appear to erode it, at least out to 5s.
- **Size/depth cost — BLOCKED then partially unblocked.** Walking the full
  order-book ladder (not just top-of-book) for realistic order sizes
  needed our own book reconstruction, which was found to drift badly (see
  recorder-fix note above) — that test's numbers were discarded, not
  reported. Recorder fixed 2026-07-28 (seq/sid + periodic resync); the
  size question needs new data under the fixed schema before it's
  retestable — not answered yet.
- **Phase 5 built (paper-trading only):** `trading/tennis_order_manager.py`
  (fixed 1-contract size, fixed 30s hold — matches what's actually
  validated, not Kelly-sized) + `live/tennis_signal_bot.py` (separate WS
  process from the capture daemon, on purpose — reuses `Book` +
  sign-detection but no parquet writes) + `config/settings.py`
  (`TENNIS_ENABLED=False` by default, separate kill switch from
  `BOT_ACTIVE`) + `tests/test_tennis_order_manager.py` (9 tests, TDD).
  32/32 full suite passes. **Not cronned.** Full design rationale:
  `/Users/ethan/.claude/plans/reactive-greeting-globe.md`.

**Next step:** manual paper-trading review run — `TENNIS_ENABLED=true` in
`.env`, run `PYTHONPATH=. python live/tennis_signal_bot.py` for a few
minutes against live markets, confirm `[PAPER]` fills land in the DB
(`orders`/`positions`/`daily_pnl`) on real detected signals. Then decide on
cronning it for a real paper-trading sample before ever flipping
`PAPER_TRADING=false`.

## Where a future session could go (only remaining ideas)

1. ~~Ensemble members~~ — DONE (cycle 7, via dynamical.org; no edge). Every pure
   forecasting lever is now exhausted.
2. ~~Different market/series~~ — DONE, pivoted to tennis (above). Weather's
   KXHIGH machinery is dormant, not deleted from history (working tree has
   large uncommitted deletions from the pivot — see
   `plans/tennis-mm-next-steps.md` "What's reusable vs. what's new").
3. ~~Maker-side microstructure~~ — DONE, dead (above). Order-flow **taker**
   is the live thread now; see the 07-27/28 session above for where it
   stands and the next step.

### Session addenda (cycle 7 infrastructure)
- New scripts: `scripts/backfill_gefs_members_zarr.py`,
  `scripts/backfill_ecmwf_members_icechunk.py` (fair member backfills;
  per-thread dataset handles — xarray lazy indexes are not thread-safe shared),
  `scripts/backfill_ecmwf_members_openmeteo.py` (dead — archive window).
- `data/historical/ensemble_members.parquet`: 125,460 rows = 82 members × 1530
  station-days (gitignored).
- Env gotcha: `pip install zarr`/`icechunk` drags numpy→2.x and breaks the conda
  stack (scipy/numexpr/bottleneck compiled vs 1.x). Pin `numpy<2`; working set:
  numpy 1.26.4 + zarr 3.1.5 + icechunk 1.1.21→2.1.0 (via dynamical-catalog 0.5.0,
  numpy stays 1.26.4).
