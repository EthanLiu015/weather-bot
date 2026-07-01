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

## Current state (updated 2026-07-01 afternoon — ASOS + intraday edge test)

- ✅ Branch `forecasting-v2` cut from `95ca4e9`; leftover empty MM `bot/` dir removed.
- ✅ Full model stack imports; **test suite green** (now includes ASOS/intraday tests).
- ✅ **`features.parquet` rebuilt** from surviving ERA5 (see "Data" below).
- ✅ **Intraday obs pipeline built** (commits `78a73f2`, `f7e28e4`, `c0883f3`) — see the
  ASOS / data-fetching section below.
- ✅ **Step 2 (intraday / short-lead edge) RAN — verdict is NEGATIVE.** The
  obs-conditioned afternoon model does **not** beat the retail book at any afternoon
  hour. Table below.
- ✅ **Refinement DONE — step 2 CLOSED.** Replaced the pooled residual model with a
  per-(station, offset) residual model (pooled fallback when a station has <8 train rows)
  in `research/intraday_edge.py`. Re-ran: **negative holds** — mktB < modelB at every
  afternoon hour; 18/20h got *slightly worse* (more trades → more negative P&L). No obs
  edge; market efficient intraday. Second table below.
- ✅ **Step 1 (multi-lead edge map) RAN — verdict NEGATIVE.** Extended
  `real_market_eval` to score the SAME markets at each lead {24,48,72,96,120,168}h
  (train once, re-cache per lead). No lead beats the book; model Brier ~0.18 vs constant
  market Brier 0.095, all leads negative P&L + Sharpe. Table below. Confirms + extends v1:
  no edge at 24h AND no edge at any longer lead.
- ✅ **Fee model FIXED (roadmap 1)** — commit `8915aea`. Backtest booked a flat
  `0.05*size*mid`; real Kalshi is `kalshi_fee = size*min(coef*p*(1-p), 0.035)`, symmetric
  in p, coef 0.07 taker / 0.0175 maker. Wired through `simulate_pnl`, `per_trade_pnl`, the
  eval harness (+ `--maker`/`--min-price` flags) and fee-adjusted Kelly. **Re-reads every
  prior P&L number.**
- ✅ **Single-train fee + segment scan (roadmap 1 & 2)** — `research/fee_segment_scan.py`.
  Trains ONCE (~35 min), scores every market at every lead, dumps
  `data/historical/multilead_scored.parquet`; then `--from-cache` re-scores any
  (fee, floor) in <1s. Results + reading below.
- ✅ **Between-NO signal AUDITED — verdict ARTIFACT, killed** (`research/audit_between.py`).
  Four attacks, all fatal (table below). The candidate is dead; do not build on it.
- ✅ **Roadmap 3 (multi-model NWP) BUILT + RAN.** New leakage-free data from Open-Meteo
  Previous Runs API — AIFS/ECMWF/ICON/GFS/GraphCast daily-high at 24/48/72h
  (`ingestion/openmeteo.py`, `scripts/backfill_openmeteo.py` →
  `data/historical/openmeteo_multimodel.parquet`, 15,578 rows). Study in
  `research/multimodel_edge.py`. **Verdict: big forecast gain, still no edge.**
  - Ensemble mean (bias-corrected) @24h: **MAE 2.3°F, Brier 0.130** — vs our ERA5 model's
    0.18. The multi-model forecast is FAR better than what we had.
  - **But market Brier 0.095 still wins** at every cross-model-spread tercile. Disagreement
    filter fails: when models agree both we and the market are sharp; when they disagree we
    degrade, market stays sharp. Gated P&L ~flat (+$8.84/725 trades, win 54%).
  - The residual gap (0.130 vs 0.095) is **calibration**, not point accuracy (σ is a crude
    global 2.4°F). → roadmap 4.
- ⏭️ **Next: roadmap 4 (calibration), sober.** Per-station-month σ + isotonic/conformal on
  the multi-model ensemble mean is the one remaining lever that could approach (not likely
  beat) the book. If a per-(station,month) σ + calibrated bracket probs don't cross 0.095,
  the forecast side is definitively exhausted and the market is efficient at the 24h decision.
  Everything points that way; keep expectations at "match, maybe," not "beat." Not pivoting.

### Roadmap 3 result — multi-model ensemble @24h (bias-corrected mean ± fitted σ)
```
A. SKILL:  model Brier 0.1298   market Brier 0.0951   → no edge (but 0.18→0.13 vs our model)
   gated P&L $8.84 on 725 trades (win 54%, maker + $0.15 floor)
B. DISAGREEMENT FILTER — by cross-model spread tercile:
   spread            n   modelB    mktB   edge?  trades   P&L$
   low(agree)      720   0.1130   0.0828    no     223   -3.97
   mid             720   0.1278   0.1083    no     236   +9.62
   high(disagree)  720   0.1487   0.0943    no     266   +3.19
```
Reproduce: `PYTHONPATH=. python -m research.multimodel_edge`
(re-backfill: `PYTHONPATH=. python scripts/backfill_openmeteo.py`)

### Roadmap 1 result — fee/floor turns blanket-negative into ~flat-to-small-positive
Same markets/model as multi-lead step 1; only the fee & price-floor vary (per-lead P&L $):
```
config                         24h    48h    72h    96h   120h   168h
old flat-5% (pre-fix)         -39    -32    -22    -21    -34    -29
corrected taker               -34    -25    -12    -13    -26    -25
maker                         -15     -5     +8     +7     -6     -5
maker + $0.15 price floor      -1     +5    +18    +16     +4     +2
taker + $0.15 price floor     -18    -13     +0     -1    -14    -16
```
Floor lifts win% 34%→62%. **But `edge?` = no at every lead** — model Brier ~0.18 still ≫
market ~0.095. Positive P&L is NOT forecast edge (see roadmap 2).

### Roadmap 2 result — segment mining (maker + $0.15 floor); NO forecast-edge cell
- **No station / lead / strike cell has model Brier < market Brier.** Confirms v1/v2.
- All positive P&L lives in **`between` (2°F) brackets: +$97.9** (`greater` -$29, `less`
  -$25). Diagnostic on the scored parquet: **97% of between trades are NO bets** (8763/8988),
  buy NO at mean price ~0.65, win 66%. This is **mechanically fading the modal bracket** — a
  single 2°F bracket rarely contains the settlement, so NO is structurally favored. It is a
  **price/anchoring fade, not weather skill** (calibration is worse than the book), it is
  **short-vol** (sells the favorite → fat-tail risk), and its **Sharpe 3.82 is a mirage**:
  8988 "trades" collapse to **683 unique station-days** (6 leads × several brackets, all
  correlated), so the effective independent sample is tiny.
- Winners lean Texas/coastal (KSAT +41, KDFW +16, KLAX +14, KSEA/KAUS +12); losers desert/NE
  (KPHX -19, KLAS -18, KLGA -12) — plausibly a regime/fat-tail miscalibration.

Reproduce: `PYTHONPATH=. python -m research.fee_segment_scan --from-cache`

### Audit of the between-NO signal — ARTIFACT (all four attacks fatal)
`PYTHONPATH=. python -m research.audit_between --lead 24`
1. **De-correlation kills the size.** The $97.9 counted each station-day up to 6× (one row
   per lead). At a SINGLE decision lead (24h) the real figure is **+$12.26** over 39 dates.
   Block-bootstrap over dates: 90% CI **[−$5.1, +$30.7]**, P(profit) **0.87** — not
   significant. **Top-3 dates = 102% of P&L** (all profit from 3 days; rest net negative).
2. **The model adds nothing.** A NULL structural fade (sell EVERY between bracket above the
   floor, ignore the forecast) earns **+$32.55 > $12.26** — the model *subtracts* value. The
   "signal" is 100% structural base-rate/anchoring fade, zero weather skill.
3. **Fill/fee realism destroys it.** maker perfect-fill +$12.26 → **taker −$3.24**, **maker
   +1¢ adverse −$2.36**, **maker +2¢ adverse −$16.98**. Survives only on flawless maker
   fills at mid with no adverse selection — which is exactly what NO-fills on soon-worthless
   brackets won't give you.
4. Integrity clean (mids in (0,1), binary outcomes, 40 dates/18 stations; 766 "dups" are a
   coarse (lead,station,date,mid) key colliding across the day's between brackets — benign).

**Conclusion: no tradeable edge in KXHIGH from this model, forecast or microstructure.**

---

## ASOS & data fetching — exactly what's being done (READ THIS)

This is the freshest work and where the session died. Goal of the pipeline: get the
**live observed running max** of the day's temperature at afternoon moments, so the model
can price `P(final daily max > threshold | max-so-far)` and edge-gate it against the
Kalshi afternoon traded price. Three pieces:

### 1. 1-minute ASOS fetcher — `ingestion/asos_1min.py` (commit `78a73f2`)
- Pulls **1-minute temperature** from the **IEM ASOS 1-minute service** — the only
  sub-hourly obs source we have (existing ingestion topped out at hourly METAR / daily
  max). `parse_1min_csv` + `running_max` are pure + unit-tested; the network call is a
  thin wrapper.
- **Critical settlement-alignment finding:** the RAW 1-min running max sits a consistent
  **+1 °F above the official daily max** (MSP 81/80, DEN 80/79, PHX 104/103) because
  **Kalshi settlement uses the ASOS 5-minute AVERAGE temperature, not the 1-min peak.** A
  **5-min-avg reconstruction** (`settlement_running_max` / `settlement_max_at`, no
  look-ahead) removes most of the bias but leaves ±1 °F residual → the intraday model must
  **learn an obs→settlement mapping**, not treat obs as truth. Matters hugely for 2 °F
  brackets.

### 2. Afternoon price + settlement-aligned backfill — `research/intraday_afternoon.py` (commit `f7e28e4`)
- The old `intraday_prices.parquet` only reached **+14h UTC (~9am, pre-high)** — useless
  for an obs-conditioned test. This pulls the **Kalshi traded price at afternoon LOCAL
  hours (4/6/8/10pm)** from **hourly candlesticks**
  (`trading/kalshi_client.get_candlesticks_range`) and pairs each with the
  **settlement-aligned running max known at that moment**.
- **Validated:** run_max at **10pm local matches settlement EXACTLY (8/8 station-days).**
  Fixed a **UTC-vs-local-day bug** that grabbed the prior day's peak for west-coast
  stations (Seattle 91→77 °F).
- **Output: `data/historical/intraday_afternoon.parquet`** (gitignored) — **5,277 markets
  × offsets {16,18,20,22} local**, 49 dates, 18 stations, **~92% with run_max** (rows with
  run_max: 4834/4840/4864/4876). Usable window = candlestick history (~10wk) ∩ IEM 1-min
  archive lag (~2wk).

### 3. Obs-conditioned edge test — `research/intraday_edge.py` (commit `c0883f3`)
- Models final daily max = `run_max + R`, `R ≥ 0` (residual rise). Estimates `R`'s
  distribution **empirically from TRAIN days** (currently **pooled** — the crude part),
  turns it into `P(final > x)`, prices each bracket via existing `bracket_yes_prob`, and
  **edge-gates** fair value vs the afternoon traded price. Temporal train/test split, real
  settlement outcomes, no look-ahead. `prob_final_above` unit-tested.

### Result (reproduced 2026-07-01, `min_edge=0.04`, train 24d / test 25d)
```
offset     n   modelB     mktB  edge?  trades      P&L$   win%
  16h  2531   0.0721   0.0349     no     799     22.66    34%
  18h  2537   0.0511   0.0030     no     193      0.52    31%
  20h  2549   0.0516   0.0002     no     144     -3.95     7%
  22h  2555   0.0532   0.0001     no     146     -4.26     7%
```
**Read:** market Brier < model Brier at **every** hour. By 8–10pm the book is near-perfect
(mktB ≈ 0.0001 — run_max ≈ settlement, market already knows) and gated P&L is negative. At
4pm the model has the most trades (799) and a small +$22 P&L, but it's still worse-calibrated
than the market → treat as variance, not edge. **No obs edge as currently modeled.**

### Result 2 — per-(station, offset) residuals (refinement, `min_edge=0.04`, min_station_samples=8)
```
offset     n   modelB     mktB  edge?  trades      P&L$   win%
  16h  2531   0.0722   0.0349     no     772     25.65    38%
  18h  2537   0.0514   0.0030     no     346     -3.63    23%
  20h  2549   0.0523   0.0002     no     247     -8.41     4%
  22h  2555   0.0532   0.0001     no     146     -4.26     7%
```
**Verdict UNCHANGED.** Fitting residuals per station (not pooled) did NOT create edge — mktB
still < modelB at every hour, and 18/20h actually got worse (more trades gated in → more
negative P&L). This answers the open question: the negative is **market efficiency**, not a
too-crude residual model. **Step 2 is closed: no obs edge; the retail book prices the
observable running max as well as or better than we can.**

### Commands (ASOS / intraday)
```bash
PYTHONPATH=. python research/intraday_afternoon.py   # rebuild intraday_afternoon.parquet (slow: Kalshi candlestick pulls, hard rate-limited)
PYTHONPATH=. python research/intraday_edge.py        # run the obs-conditioned edge test (table above)
```

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
  p-12/p-6/p+0/p+6/p+12/p+14h** around each market, + `settlement`. Superseded for the
  edge test: only reaches ~9am (pre-high), so useless for obs-conditioning.
- `historical/intraday_afternoon.parquet` — **NEW**, the real intraday dataset: 5,277
  markets × afternoon local hours {16,18,20,22}, settlement-aligned run_max, ~92% obs. See
  the ASOS section above. This is what step 2 actually ran on.
- **Missing / gone:** `ecmwf/` (0 B — the ECMWF anchor backfill; not needed for the
  historical ERA5 feature build, but was a live-model input) and the old
  `features.parquet` (now rebuilt).

---

## The v2 plan (prioritized)

### 1. Multi-lead edge map (extends the existing harness) — ✅ RAN, ❌ NEGATIVE
Generalized `real_market_eval` beyond `lead_hour == 24`. New:
`_build_distribution_cache(..., lead_hour=L)` pins the forecast to lead `L` (default
still = shortest lead), and `run_multilead_evaluation` trains ONCE then scores the SAME
markets at each lead. Run it with:
```bash
PYTHONPATH=. python -m backtest.real_market_eval --multilead
```
The market price is the fixed decision-time `d1_mid` (only price we stored), so market
Brier is constant across leads and only the model's forecast row varies. **Result (train
< 2026-04-11, eval 2026-04-11→05-27, min_edge=0.04, n_estimators=500):**
```
 lead     n   modelB     mktB  edge?  trades      P&L$   win%  sharpe
  24h  4108   0.1825   0.0945     no    3120    -38.98    34%   -6.58
  48h  4216   0.1786   0.0953     no    3193    -31.80    34%   -4.80
  72h  4216   0.1810   0.0953     no    3223    -21.83    34%   -3.33
  96h  4216   0.1821   0.0953     no    3230    -20.74    34%   -3.13
 120h  4216   0.1845   0.0953     no    3246    -34.33    34%   -5.64
 168h  4216   0.1832   0.0953     no    3197    -29.35    34%   -4.66
```
**Read:** no lead beats the book — model Brier ~0.18 vs constant market Brier ~0.095 at
EVERY lead, all gated P&L negative, all Sharpe < 0. The model degrades slightly with lead
(as expected) but was never competitive to begin with. This confirms and extends v1: no
edge at 24h, and none at any longer lead. Cheap edge tests (this + intraday) both
exhausted and negative.

### 2. Intraday / short-lead edge (the main event) — ✅ RAN, ❌ NEGATIVE, ✅ CLOSED
Built the full obs pipeline (1-min ASOS → 5-min-avg settlement-aligned run_max →
afternoon Kalshi price) and ran the edge test on `intraday_afternoon.parquet`. Model does
NOT beat the market at any afternoon hour; by 8–10pm the book is near-perfect. The pooled
residual model was then refined to **per-(station, offset) residuals** (pooled fallback
<8 samples) and re-run — **verdict unchanged, negative held.** Step 2 closed: no obs edge;
market efficient intraday. Next task is **step 1 (multi-lead edge map)**.

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
PYTHONPATH=. pytest tests/ -q                              # 393 passing
PYTHONPATH=. python scripts/build_feature_matrix.py        # rebuild features.parquet
PYTHONPATH=. python scripts/initial_train.py               # (re)train models
PYTHONPATH=. python -m backtest.real_market_eval           # D+1 real-price eval (v1 verdict)
PYTHONPATH=. python -m backtest.real_market_eval --multilead  # step 1 multi-lead edge map
```

## How we got here (lineage)
weather forecasting (v1, no 24h edge) → market-making research on `main` (naive MM
loses once inventory+exit costs integrated; full round-trip negative) → brief sports/
sportsbook-+EV probe (Kalshi sports = mostly dead auto-parlays; real game markets are
thin/new, e.g. Wimbledon ATP/WTA matches with ~1–2 OI at open) → **back to forecasting
with the edge-gated multi-lead thesis (this branch).**

Cross-session memory: the `market-making-pivot` note + older weather notes describe
prior states; treat as historical. Re-read this file first.
