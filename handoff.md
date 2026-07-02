# Handoff — Weather forecasting v2, autoresearch session (2026-07-01 night)

_Branch: `forecasting-v2`. This file replaces the previous handoff (recover via git
history; its durable conclusions are carried in "Frozen Decisions" below)._

---

## 1. Project Objective — CONCLUDED: PLATEAU

Goal was: beat the Kalshi KXHIGH market on calibration — **model Brier < market
Brier (0.0951)** on the real-market eval window (Apr 11 → May 27 2026 test split),
using only pre-14:00-UTC information. The autoresearch loop
(`autoresearch/orchestrator-260701-1933/orchestrator-state.json`, status
**PLATEAU**) ran 6 cycles and exhausted every planned lever.

**Final verdict: KXHIGH `d1_mid` at 14:00 UTC is efficient with respect to every
legal public information source we could obtain.** The killer result (cycle 6):
the walk-forward closed-form optimal shrinkage weight for our best fair model is
**w = 0.000** — even handed the market price for free, the model contributes zero
orthogonal information. No model in this information set can hit the predicate.

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
| 6 | Market-shrinkage blend (`research/blend_fallback.py`) | 0.0951 (=mkt) | 0.0951 | w*=0.000 — zero orthogonal info. **Loop stopped** |

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
- **No fair ensemble-member data exists on Open-Meteo**: `temperature_2m_previous_dayN`
  returns all-null for ensemble models (gfs025, ecmwf_ifs025) on both the
  ensemble API and previous-runs API, historical and forecast mode. Plain
  historical ensemble data is assembled from latest runs = post-cutoff look-ahead
  (banned). Only path is AWS GRIB byte-range backfill (noaa-gefs-pds).
  `scripts/backfill_ensemble_members.py` + `research/ensemble_pdf.py` are written
  and ready if that data is ever obtained — but cycle 6's w=0 makes the prior
  very low.
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

## Where a future session could go (only remaining ideas)

1. **AWS GRIB ensemble members** (GEFS `noaa-gefs-pds` byte-range via .idx; maybe
   ECMWF open-data) → feed `research/ensemble_pdf.py`. Low prior: w=0 says the
   book already impounds mean+σ; only distribution SHAPE could matter.
2. **Different market/series** — the machinery (eval spine, walk-forward, honesty
   protocol) transfers to KXLOW or non-weather series where the book may be
   softer. KXLOW data is already in `kalshi_prices.parquet`.
3. **Maker-side microstructure** rather than forecasting — but see dead ends
   before reopening anything.
