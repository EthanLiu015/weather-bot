# Scope: align training target + settlement to the official NWS daily max

**Status:** scoping (no code yet). Prepared 2026-06-28.

## Why
The real-markets eval ([[real-markets-eval-harness]]) showed NO edge, but the
model is partly handicapped because it is trained on a DIFFERENT temperature
than Kalshi settles on:

- Our training target `actual_tmax` = **max of hourly METAR temps** per local
  day (`build_asos_daily_tmax` in `scripts/build_feature_matrix.py`, reading
  `data/historical/{station}_hourly.parquet`). Hourly sampling misses sub-hourly
  peaks → underestimates the true daily high by ~0–2°F.
- Kalshi settles on the **NWS "Climatological Report (Daily)"** official max at a
  specific station.
- Two stations are also mapped to the wrong airport entirely:
  - Chicago: we use **KORD** (O'Hare); Kalshi settles **Chicago Midway (KMDW)**.
  - New York: we use **KLGA** (LaGuardia); Kalshi settles **Central Park (KNYC)**.

Net effect: `(our actual_tmax == Kalshi settlement)` agrees only **80–92%** by
station, and a 1°F gap flips ~16% of the tight 2°-wide `between` brackets. The
model predicts our number but is scored against Kalshi's → unfair to the model
(the market is scored against its own settlement, so it is not penalized).

Goal: make the model train on, and the harness reason about, the SAME official
daily max Kalshi uses, per the EXACT settlement station.

## Data source (VALIDATED)
**IEM (Iowa Environmental Mesonet) ASOS daily summary** — free, no auth, deep
history, gives the official `max_temp_f` (true daily max, not hourly-sampled).

```
GET https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py
    ?network={STATE}_ASOS&stations={ID3}
    &year1=&month1=&day1=&year2=&month2=&day2=&format=comma
```
Returns CSV: `station,day,max_temp_f,min_temp_f,...,climo_high_f,...`.
Confirmed working for MDW: 2026-05-27 max_temp_f = 82.0, which matches the Kalshi
KXHIGHCHI winner that day (B81.5 = "81° to 82°"). `min_temp_f` also covers the
KXLOW* low-temp series if we ever want them.

Fallbacks if IEM has gaps: NOAA GHCN-Daily TMAX (NCEI) at the station's GHCND id;
or last-resort the existing hourly max.

## Series → settlement station map (from each market's `rules_primary`)
18/20 already use the right airport (LAX, MIA, PHL, AUS, DEN, PHX, SFO, SEA, BOS,
DFW, DCA, LAS, MSP, OKC, SAT, MSY, + IAH/ATL to confirm). The two to fix:

| series | current | correct station | IEM network/id | coords (for features) |
|---|---|---|---|---|
| KXHIGHCHI / KXLOWTCHI | KORD | Chicago Midway | IL_ASOS / MDW | 41.786, -87.752 |
| KXHIGHNY / KXLOWNYC   | KLGA | NYC Central Park | NY_ASOS / NYC | 40.779, -73.969 |

(Action item: for the other 18, confirm the IEM 3-letter id + state network — it
is just the ICAO minus the leading `K`, network = the US state's `{ST}_ASOS`.)

## Decided phasing (2026-06-28)
Fix the **training target for all 20 stations** first (cheap, no feature
changes) and measure the Brier lift BEFORE investing in the Chicago/NY feature
re-extraction. Key insight: the settlement-station mismatch only needs to change
the *target source* per station (use MDW/NYC's official max for Chicago/NY) — the
*features* (forecast proxies at the O'Hare/LaGuardia coords) are a second-order
residual we can defer. So one `SETTLEMENT_STATION` map fixes the target for all
20 at once.

## Work breakdown (PR-sized slices, TDD)

### Slice 0 — de-risk: resolve IEM ids + verify day boundary  ← DO FIRST
- Build `SETTLEMENT_STATION: {icao -> (iem_network, iem_id)}` for all 20
  (18 = own airport `ICAO[1:]` + `{STATE}_ASOS`; Chicago→IL_ASOS/MDW,
  NY→NY_ASOS/NYC). Verify each resolves on IEM (some networks are non-obvious,
  e.g. DCA).
- Pull IEM `max_temp_f` for the eval window and recompute
  `(official_max == Kalshi settlement)` agreement per station. **Gate:** if it is
  not ≥ ~97% for matched stations, the local-day boundary is misaligned — fix
  that before proceeding (do NOT build on a broken premise).

### Slice 1 — IEM daily ingestion (no behavior change yet)
- New `ingestion/nws_daily.py`: `fetch_official_daily_tmax(network, station_id,
  start, end) -> pd.Series[date -> max_temp_f]` + the `SETTLEMENT_STATION` map.
  Pure CSV parser tested on a captured sample (handle `None`, missing days,
  header, the trailing-comma rows).

### Slice 2 — switch the training target source (ALL 20 via SETTLEMENT_STATION)
- In `build_feature_matrix.py`, replace `build_asos_daily_tmax` (hourly max) with
  the IEM `max_temp_f` join, keyed by each station's SETTLEMENT station (so
  Chicago uses MDW, NY uses NYC; the rest use their own airport).
- Rebuild `features.parquet` so `actual_tmax` = official daily max. Forecast
  features unchanged. Back up the current `features.parquet` first.

### Slice 3 — re-run + verify (gate before any further work)
- Re-run `backtest/real_market_eval.py`.
- **Acceptance:** per-station `(actual_tmax == settlement)` agreement ≥ ~97%;
  model Brier drops, overall + per strike_type (biggest lift on `between`).
- Decision point: only if the lift is material AND Chicago/NY still lag do we do
  Slice 4.

### Slice 4 — (DEFERRED) Chicago/NY feature coords
- Add KMDW + KNYC coords; re-interpolate ERA5/forecast features at them; retrain.
  Only pursue if Slice 3 shows the feature-coord residual matters.

## Risks / decisions to confirm
- **Day boundary.** Kalshi resolves "for <calendar date>" in LOCAL time; IEM
  daily is local-calendar-day too — confirm they line up (esp. west-coast late
  highs). A 1-day or tz offset would silently wreck agreement.
- **Eval settlement is already correct.** The harness uses Kalshi's `settlement`
  field as truth, which already IS the NWS number — so Slices only change the
  TRAINING target (and the agreement diagnostic), not the harness's outcome.
- **Scope choice for the user:** do Slice 2 alone first (cheap, no feature
  changes, fixes 18 stations) and measure the Brier lift before committing to
  Slice 3's feature re-extraction for Chicago/NY?
- IEM rate limits on a full backfill (batch by station, 1 request per station for
  the whole date range).
- Even fully aligned, the eval may still show no edge (tails already show market
  0.016 vs model 0.064) — this makes the comparison HONEST, not necessarily
  profitable. Worth stating before investing in Slice 3.
