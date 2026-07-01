# Handoff — Weather forecasting, v2 (edge-gated, multi-lead)

_Branch: `forecasting-v2` (created 2026-07-01 from commit `95ca4e9`, the last good
forecasting state before the market-making pivot). `main` still holds all the
market-making research; this branch is a clean forecasting tree. See the bottom for
how we got here._

---

## 🧭 START HERE — the new thesis

We are **back on weather forecasting**, but with a sharper thesis than v1.

**v1's verdict (rigorous, but narrow):** the model has NO edge **at 24h lead** —
model Brier 0.145 vs market 0.096, no edge in any *station / volume / month / strike*
segment. The cause looked structural: we forecast at 24h; the market prices off
~6h-lead guidance + live observations.

**What v1 never tested — the opening:** the eval only ever scored **one lead (D+1,
24h)** against **one price (`d1_mid`)**. It never evaluated:
- **other leads** (the harness hardcoded `lead_hour == 24`; multi-lead was gap #10
  in `plans/model-gaps.md`, never done), or
- **intraday / short lead**, when the day's high is becoming *observable* (the running
  max is partly locked) but the retail book may lag.

**The v2 thesis (user's framing):** *forecast at all times of day, and TAKE A TRADE
whenever the modeled edge clears a fee-aware threshold — not just at 24h.* The edge, if
it exists, is concentrated where our information is freshest relative to the book
(short lead / intraday), but we don't restrict to that a priori — we let the edge gate
decide. This inverts v1: instead of "beat the 24h forecast" (proven hard), it's "find
the (lead, market, time) cells where the model's fair value diverges from the live
price enough to trade through fees."

The edge gate already exists in code: `backtest/real_market_eval.py::per_trade_pnl`
trades iff `|model_prob − market_mid| > min_edge` and books the 5% Kalshi fee. The v2
work is to run that gate across **many leads and intraday snapshots**, not just D+1.

---

## Current state (this session)

- ✅ Branch `forecasting-v2` cut from `95ca4e9`; leftover empty MM `bot/` dir removed.
- ✅ Full model stack imports; **test suite green (368 passed)**.
- ✅ **`features.parquet` rebuilt** from surviving ERA5 (see "Data" below). This was the
  one missing artifact — the raw ERA5/GEFS/NBM backfill itself survived on disk.
- ⏭️ Not yet done: the multi-lead / intraday edge evaluation (the core v2 work).

## Data inventory (what survived the pivot, all gitignored under `data/`)
- `era5/` — **1.7 GB** ERA5 reanalysis, 2021–2026 (ground-truth + forecast proxy for
  training). This is what `features.parquet` is rebuilt from.
- `nbm/` 369 MB, `gefs/` 87 MB — NWP forecast inputs (NBM historical is sparse; the
  feature build intentionally leaves historical NBM NaN).
- `calibrators/` 28 MB, `bias_correctors/`, `climatology/`, `diurnal_climatology/` —
  trained artifacts.
- `historical/kalshi_prices.parquet` — **13,670 real Kalshi markets** (Apr–Jun 2026)
  with `d1_mid`, `settlement`, `yes_bid/ask`, `volume`, `strike_type`, strikes. The
  real-price truth for the D+1 eval.
- `historical/intraday_prices.parquet` — **2,327 markets × price snapshots at
  p-12/p-6/p+0/p+6/p+12/p+14h** around each market, + `settlement`. THE dataset for the
  intraday/short-lead edge test.
- **Missing / gone:** `ecmwf/` (0 B — the ECMWF anchor backfill; not needed for the
  historical ERA5 feature build, but was a live-model input) and the old
  `features.parquet` (now rebuilt).

---

## The v2 plan (prioritized)

### 1. Multi-lead edge map (extends the existing harness)
Generalize `real_market_eval` beyond `lead_hour == 24`: for each lead in
`{24,48,72,96,120,168}` compute model fair prob at each real market's exact threshold,
run the edge gate vs the market price available at that lead, and report model-vs-market
Brier **and gated P&L per lead**. Deliverable: a lead × segment table showing whether
any lead has positive gated P&L. (This is the cheap, first test — reuses trained models
+ `features.parquet`, no new data.)

### 2. Intraday / short-lead edge (the main event)
Use `intraday_prices.parquet`. At each snapshot (esp. `p+0`, `p+6`) the real signal is
the **running observed max** from live ASOS obs — the model must ingest obs-to-date and
predict P(final daily max > threshold) given "max so far". Compare that to the snapshot
market price; gate + settle. The empirical question: is our obs-conditioned fair value
sharper/faster than the retail book at that same moment? If not here, nowhere.

### 3. Model refinement (only where it changes the verdict)
Work `plans/model-gaps.md` selectively — prioritize the gaps that matter for the above:
per-station-month calibration (#7), multi-lead eval (#10, folded into step 1),
inverse-CRPS blend weights + walk-forward QRF (#3). De-prioritize pure 24h model-quality
polish; v1 shows it can at best *match* the market there.

### 4. Trading layer
Once a positive-edge cell is found: fee-aware edge threshold sweep, fractional-Kelly
sizing, bracket/time selectivity, and honest transaction-cost + variance accounting
before any capital.

**Honesty rule (hard-won):** audit every data artifact against an independent source
before trusting a P&L number. v1 was repeatedly poisoned by silent data bugs (NBM
boustrophedon scramble, date off-by-ones). A positive result must survive a leakage
audit (`backtest/leakage_audit.py`) and a no-look-ahead check.

---

## Key files
- `backtest/real_market_eval.py` — real-price eval + edge gate (`per_trade_pnl`). The
  spine of v2; extend for multi-lead/intraday.
- `backtest/runner.py`, `backtest/track_b.py` — climatology backtest + P&L/fee math.
- `backtest/leakage_audit.py` — look-ahead guard. Run on any positive result.
- `models/` — `ngboost_model` (μ,σ), `qrf_model`, `blend`, `calibration`,
  `residual_model`, `spread_inflation`. Several components trained but unwired (gaps #1–3).
- `strategies/bracket_pricing.py` — `bracket_yes_prob`: model dist → Kalshi bracket
  (greater/less/between) prob. 2°F brackets need sub-1°F sharpness.
- `processing/features.py`, `scripts/build_feature_matrix.py` — feature matrix
  (rebuildable from ERA5; output `data/historical/features.parquet`).
- `plans/model-gaps.md` — 15 known model gaps in 6 PRs (mostly undone).

## Commands
```bash
PYTHONPATH=. pytest tests/ -q                              # 368 passing
PYTHONPATH=. python scripts/build_feature_matrix.py        # rebuild features.parquet
PYTHONPATH=. python scripts/initial_train.py               # (re)train models
PYTHONPATH=. python -m backtest.real_market_eval           # D+1 real-price eval (v1 verdict)
```

## How we got here (lineage)
weather forecasting (v1, no 24h edge) → market-making research on `main` (naive MM
loses once inventory+exit costs integrated; full round-trip negative) → brief sports/
sportsbook-+EV probe (Kalshi sports = mostly dead auto-parlays; real game markets are
thin/new, e.g. Wimbledon ATP/WTA matches with ~1–2 OI at open) → **back to forecasting
with the edge-gated multi-lead thesis (this branch).**

Cross-session memory: the `market-making-pivot` note + older weather notes describe
prior states; treat as historical. Re-read this file first.
