# Kalshi Weather Bot — Handoff

## Goal

Build and deploy an automated options trading bot that bets on Kalshi temperature markets (e.g., "Will NYC max temp exceed 85°F on July 4?"). The core idea is that a probabilistic ensemble model trained on GEFS/ECMWF/NBM weather forecast data can price these binary outcomes more accurately than the market, and that a fee-adjusted Kelly criterion can size positions to maximise risk-adjusted returns.

The full pipeline: ingest live weather forecasts → build feature matrix → run NGBoost+QRF ensemble → calibrate probabilities → compute Kelly-sized orders → submit via Kalshi API → monitor/alert.

---

## 🧭 START HERE (2026-06-28) — Why the model has no edge, and where to look next

We now have a **trustworthy, confound-free verdict: the model does NOT beat the Kalshi market** (model Brier ~0.146 vs market ~0.096 on real bracket markets; market wins every strike type). Two confounds have been eliminated this session — markets are now priced as the real 2°F brackets they are ([[kalshi-bracket-markets]]), and the training target is now the exact official NWS max Kalshi settles on (100% agreement). Aligning the target did NOT change the result, so the gap is **genuine forecast skill**, not a data artifact.

The next agent's job is to **diagnose the skill gap and look for any exploitable edge.** Below is the grounded "why" + a prioritised investigation plan. Detailed history is in the dated sections further down.

### Why the model underperforms the market (evidence)
Measured on the eval window (Apr 11–May 27, lead 24h, vs official max):

| input / model | corr w/ actual | MAE °F | note |
|---|---|---|---|
| `gefs_tmax_mean` | 0.56 | **15.0** | **BROKEN** — it's ERA5 *instantaneous* t2m at init time, NOT a daily-max forecast |
| `nbm_t50` | 0.68 | **8.1** | suspiciously weak (NBM MaxT should be ~2–3°F) — likely a backfill/units bug |
| `ecmwf_tmax` | 0.93 | 3.74 | the real workhorse feature (backfilled; differs from gefs proxy by ~14°F) |
| **blended model μ** | 0.94 | **3.1** | the model DOES add value over raw ECMWF — the ML isn't the problem |

Root causes, in rough priority:
1. **Information-horizon gap (biggest).** The market price we score against is the last trade by ~14:00 UTC on resolution day (≈9am local, ~6–9h before the afternoon high). Our shortest feature is a **24h-lead** forecast (issued ~30h before the high). The market simply has fresher data (same-morning NBM/HRRR + early obs). A 24h forecast (MAE ~3°F) cannot beat a market pricing off ~6h-lead guidance.
2. **Forecast precision vs bracket width.** Brackets are **2°F** wide; model error is ~3°F (σ≈4.3°F), so the predictive distribution smears across 2–3 brackets → best-case per-bracket Brier ~0.15. To win you need sub-1°F sharpness or an inefficiency.
3. **Fake ensemble uncertainty.** Historical GEFS spread features (`gefs_tmax_std/range/iqr/p10..p90`) are faked from ERA5 temporal windows, not a real 20-member ensemble → the model's σ (hence bracket probabilities) is miscalibrated by construction. Real GEFS ensemble spread would directly improve bracket pricing.
4. **A core feature is dead weight.** `gefs_tmax_mean` (15°F MAE) carries no daily-max signal; the trees lean on `ecmwf_tmax`. Fixing/removing it is low-effort.

### Next steps — diagnose issues & gaps

**0. FIRST: run the definitive 500-tree harness and record the numbers.** (The 80-tree run gave model Brier 0.146; a 500-tree run was attempted but killed before finishing — finish it for the official record.) ~20–25 min train + fast eval:
```bash
PYTHONPATH=. python -m backtest.real_market_eval --n-estimators 500 | tee /tmp/harness_official500.log
```
Expected ≈ the 80-tree result (NO edge); replace the "n_est=80; 500-tree confirm pending" note in the Follow-up section with the actual figures (overall + per strike_type). If it differs materially from 0.146, STOP and investigate — that would be surprising.

Then the diagnostics below (cheap, no new data):

1. **Per-segment edge breakdown in the harness.** Extend `evaluate_real_markets` (already does `by_strike_type`) to also break model−market Brier and P&L down **by station, by lead bucket, by volume decile, by month**. Edge, if any, hides in segments (thin/illiquid markets, specific stations, longer-dated D3–7 where forecasts diverge and the market may be less efficient). This is the single highest-value diagnostic.
2. **Feature-importance + ablation audit.** Which features the production models actually use; confirm `gefs_tmax_mean` is noise; quantify how much `ecmwf_tmax` alone explains. Drop/repair dead features.
3. **σ-calibration check.** Compare predicted σ to realized error by lead/station; the spread-inflation + sigma-floor (2.0°F) were tuned against climatology, not real brackets — verify they're not over/under-confident on the bracket task.
4. **Investigate `nbm_t50` MAE 8°F.** NBM is purpose-built MaxT guidance; 8°F error implies a backfill bug (wrong field/units/time-agg). Fixing it could add a strong feature.
5. **Liquidity/volume study.** Compute market Brier vs `volume` — the market is likely sharpest on liquid markets; thin markets are where mispricings (our only realistic edge) would live.

### Possible edges to explore (where more data / different angles could win)
- **Fresher data (highest leverage).** Ingest short-lead guidance to match the market's horizon: **HRRR (0–18h), RAP, the latest same-day NBM MaxT grid**, and live METAR nowcasts. Score at the market's actual decision time, not 24h out. This directly attacks root cause #1.
- **Real GEFS ensemble (not ERA5 proxy)** for genuine forecast uncertainty → calibrated bracket probabilities (root cause #3). Same for a real ECMWF ensemble (ENS) if obtainable.
- **Use NWS/NBM official MaxT forecast as a direct feature/anchor** — the market essentially prices off this; matching it is table stakes, beating it needs a bias model on top.
- **Microclimate / station-bias models.** Persistent local biases (marine layer at SFO/LAX, lake effect at KORD/Midway, UHI) where NBM has known systematic errors — a learned per-station correction on the official MaxT could create thin edge.
- **Inefficiency hunting, not forecast-beating.** Tails (`greater`/`less`) are where the market is most confident (Brier 0.016) — unlikely. Better: thin/early/long-dated markets, opening-auction mispricings, or stale prices right after a forecast update.
- **Longer-dated markets (D3–7).** At longer lead the market's information advantage shrinks and forecasts diverge; if any model edge exists it's more likely here than at D1. Check the per-lead breakdown (step 1).

### Reality check for the next agent
Day-ahead daily-max temperature is a *very* well-forecast quantity and the market aggregates excellent NWS/NBM guidance — beating it is genuinely hard. The model already matches/edges raw ECMWF; the deficit is **information freshness + ensemble calibration + bracket precision**, not a broken ML pipeline. Set expectations accordingly: the realistic win is a *thin* edge in a *segment* (thin markets, longer leads, specific stations), not a broad alpha. Decide early whether that's worth the data-engineering cost of short-lead ingestion.

---

## ⚠️ READ FIRST — Critical correction (2026-06-23)

**Every "real Kalshi price" performance number in the older sections below is FAKE. Do not cite the $95,983 P&L or any real-price Sharpe (3.05 / 4.0 / 5.6 / 41).** They were artifacts of three compounding bugs, found and (mostly) fixed this session:

1. **Fetch fabricated 0.5 prices** ✅ FIXED. `d1_mid = (prev_yes_bid + prev_yes_ask)/2` collapsed to **0.5** for 94% of settled markets (empty book = bid 0 / ask 1). `_get_market_mid` accepted them as real → the model "beat" a constant coin-flip. Fix: `scripts/fetch_kalshi_history.py::_compute_d1_mid` returns NaN for empty/degenerate/crossed books; never uses `last_price` (look-ahead).
2. **No empty-book guard** ✅ FIXED. `backtest/runner.py::_get_market_mid` now rejects `d1_mid == 0.5` (defense-in-depth).
3. **Backtest evaluates synthetic thresholds, prices them with real markets at *different* thresholds** ❌ NOT FIXED (architectural). The backtest scores synthetic station×month-median markets (~50/50) but `_get_market_mid` matches a real Kalshi price from a market at a nearby-but-different threshold (within 5°F). Price refers to threshold A, outcome to threshold B → no correspondence. Proof: real-price fold market Brier = **0.498** (vs 0.068 on the raw data); Sharpe a fake **41**.

**Real candlestick prices ARE now obtainable** (the candlestick endpoint needed `/series/{series_ticker}/markets/{ticker}/candlesticks`, not `/markets/{ticker}/candlesticks` → it had 404'd; the old D-1 48–24h window is also empty because these markets only trade in the final ~24h). Re-fetched → `data/historical/kalshi_prices.parquet` = **13,631 genuine prices** (Apr 11 – Jun 22 2026): 0.5% at 0.5, **corr(price, settlement) = +0.71, market Brier 0.068**. The market is a STRONG forecaster — any real edge will be thin at best.

**Honest current state: there is NO trustworthy real-price backtest.** The only defensible metrics are the calibration numbers (CRPS 3.56, Brier 0.058, reliability slope 0.984) and the robustness/Monte-Carlo *structure* — all independent of pricing. See [[kalshi-price-history-rolling-window]] in memory.

### Other things done this session (2026-06-22 → 06-23)
- **Param stability + Monte Carlo run.** Sharpe flat ~3.05 across 32 configs (CV 0.58%); kelly/exposure are pure leverage. 0% ruin prob; survives 20% outcome-perturbation. (All vs climatology — same caveat.) Outputs in `data/stability/`, `data/montecarlo/`, `data/fold_variance/` (P&L variance is seasonal, corr(pnl/trade, CRPS)=+0.37).
- **Production models retrained** per-lead-bucket (`KATL_D1-2` composite keys). 60/60 ngboost+qrf+calibrators; blender ngboost 0.529 / qrf 0.471. Run `train_final_models()` DIRECTLY (not `main()`, which re-runs the 17-hr backtest); needs `init_db()` first.
- **Fair-value clamp** (TDD): `FAIR_VALUE_FLOOR=0.02`/`CEIL=0.98` in `strategies/ensemble_strategy.py` — kills the cal_prob=1.0 overconfidence at coastal stations (KSFO/KLAX).
- **Test suite: 310 passing** (was 293).
- **Storage:** pruned ~26 GB of regenerable GEFS/NBM caches + model backup; `data/` 36 GB → 9.4 GB. `data/era5` (rebuild source) and `data/historical/` KEPT.

### Data file state (current)
- `data/historical/kalshi_prices.parquet` — **genuine** prices (13,631; Apr 11–Jun 22). Backups: `*.backup_20260622_*` (old fabricated) and `*.prefix_fix_backup_20260623_*`.
- `data/backtest_trades.parquet` — restored to the ORIGINAL 24-fold run (fake-price; used by montecarlo/param_stability). The 2-fold real-price experiment is saved aside as `data/backtest_trades.realprice_mismatch_experiment.parquet`.
- `data/backtest_results_realprice.csv` — the 2-fold (Apr/May) real-price run (mismatched; do not trust P&L).

---

## ✅ DONE (2026-06-27) — Real-markets evaluation harness BUILT + first trustworthy result

**Result (definitive, n_est=500 production-matching): NO EDGE.** Model Brier **0.1448** vs market Brier **0.0963** on 3,340 real high-temp bracket markets (Apr 11–May 27). Flat-$1 P&L −$8.91 / 2,474 trades, win rate 31.7%, daily Sharpe −1.4. 500 trees barely moved the result vs an 80-tree run (0.146 → 0.1448) → this is a **genuine no-edge result, not undertraining**. The market beats the model in EVERY strike type:

| strike_type | n | model Brier | market Brier | edge? |
|---|---|---|---|---|
| greater (tail) | 557 | 0.0641 | **0.0159** | no |
| less (tail) | 557 | 0.1643 | **0.0736** | no |
| between (2° bin) | 2226 | 0.1601 | **0.1222** | no |

The model's calibrated forecasts, mapped to Kalshi's real brackets, are **less accurate than the market's own prices** — even in the tails where the station/source mismatch barely bites. Kalshi temperature markets are a strong forecaster; this strategy as-is has no demonstrable edge.

### Two big discoveries this session
1. **Kalshi temp markets are mutually-exclusive ~2°F BRACKETS, not above/below.** Proven: 1,206/1,207 (station,date,series) groups have exactly one settled winner. The old fetch mislabeled `B…` tickers as "below" (they're `between` bins) and `T…` as "above" (they're `greater`/`less` tails). The API exposes the truth via `strike_type ∈ {greater,less,between}` + `floor_strike`/`cap_strike` + `subtitle`. **FIXED** in `scripts/fetch_kalshi_history.py` (`_strike_fields`); re-fetched `kalshi_prices.parquet` (13,670 rows, now has `strike_type/floor_strike/cap_strike/subtitle`; old above/below `threshold`/`market_type` columns GONE).
2. **PRODUCTION PRICING BUG (still open).** `strategies/ensemble_strategy.py` prices EVERY ticker as `P(tmax > threshold)` (`_ticker_to_threshold` + `_compute_fair_value`), ignoring T-vs-B. For bracket markets the correct YES is `P(floor ≤ tmax ≤ cap)`. **The deployed bot mis-prices every bracket market (most of them).** Needs a fix mirroring `bracket_yes_prob` (TDD + approval).

### How the harness prices brackets (correct semantics, integer-°F continuity ±0.5)
- `greater` (>F, "F+1° or above"): YES = `P(high > F+0.5)`
- `less` (<C, "C−1° or below"): YES = `1 − P(high > C−0.5)`
- `between [F,C]` ("F° to C°"): YES = `P(high > F−0.5) − P(high > C+0.5)`
`prob_above(x)` = the production calibrated `P(high>x)` (reuses `_compute_fair_value`'s exact ngboost+residual+qrf-blend+isotonic path; a parity test pins it).

### Caveat that caps achievable model Brier: station/source mismatch
Kalshi settles on the official NWS station (e.g. Chicago **Midway**); our features use the ASOS station (e.g. KORD = **O'Hare**). `(our actual_tmax vs settlement)` agree only **87.6%** overall (greater 98%, less 93%, **between 84%** — a 1°F source gap flips ~16% of tight 2° brackets). The model is structurally handicapped on `between` brackets. See per-strike-type breakdown in the harness output — any genuine edge would surface in the tails (greater/less).

### Files added/changed
- **NEW** `backtest/real_market_eval.py` — the harness. Run: `PYTHONPATH=. python -m backtest.real_market_eval [--n-estimators 500] [--eval-start … --eval-end …]`. Trains look-ahead-free models on data **before** the eval window (single split — chosen because the window is only ~6.5wk and there's 5yr of prior weather data; daily walk-forward was rejected as overkill). Batched per station-bucket (~60 forest evals, not ~8k). Reports model vs market Brier overall + per strike_type, flat-$1 P&L, win rate, daily Sharpe.
- **REFACTOR** `scripts/initial_train.py` — extracted pure in-memory `train_models(df) → bundle` (single training recipe, no registry writes); `train_final_models` now persists from it. Harness consumes the bundle directly so it never clobbers production models.
- `scripts/fetch_kalshi_history.py` — `_strike_fields`; rows now carry real strike structure.
- Tests: `tests/test_real_market_eval.py` (21), `tests/test_train_models_bundle.py` (2), `tests/test_fetch_kalshi_history.py` (+4 strike tests).
- Backups: `data/historical/kalshi_prices.preBracket_backup_*.parquet` (the old above/below-labelled data).

### ⚠️ PERF GOTCHA fixed
`IsotonicCalibrator.calibrate()` runs `bootstrap_ci` (1000 isotonic refits) per call — fine for production (a few dozen tickers/cycle) but the harness prices ~8k bracket boundaries → it hung. The harness calls `calibrator._iso.predict()` directly for `cal_prob` (identical value, no CI), since the eval doesn't use ci_width.

### Follow-up (2026-06-28) — aligned training target to the official NWS max; STILL no edge
Closed the source/station mismatch: the training target `actual_tmax` was the max of **hourly** METAR temps, which underestimates the true daily peak by ~1°F and disagreed with Kalshi settlements ~12% of the time. Switched it to the **official NWS daily max** (IEM ASOS daily summary) at the exact station Kalshi settles on (incl. Chicago→Midway, NY→Central Park). Verified: official-max vs Kalshi settlement agreement = **100%** (was 87.6%). Migrated `features.parquet` (131,852 rows changed, +1°F mean; `data/historical/features.parquet.bak_pre_official_tmax_*` backup).

**Result: model Brier ~0.146 → essentially UNCHANGED** (n_est=80; 500-tree confirm pending). So the measurement gap was NOT the cause — the model genuinely lacks edge. Even predicting exactly the settlement quantity, its day-ahead forecast (σ≈4°F spread over 2° brackets) is less sharp than the market's prices. This is the cleaner, confound-free "no edge" conclusion. Files: `ingestion/nws_daily.py` (+`SETTLEMENT_STATION`), `scripts/update_actual_tmax_official.py`, `build_feature_matrix.build_daily_tmax` (now defaults to official source w/ hourly fallback), `tests/test_nws_daily.py` (7). Plan: `plans/nws-settlement-source.md`. Known minor approximation: obs_minus_model lag features + climatology normals still on the old target (target-only migration; full rebuild would refresh them but risks wiping ECMWF/NBM backfills).

### Open / next
- **Fix the production bracket-pricing bug** (ensemble_strategy) — the bot is live-mispricing every bracket. But note: even correctly priced + target-aligned, the eval shows NO edge, so fixing it makes the bot *correct*, not *profitable*.
- The honest strategic question: with no edge vs Kalshi on this data, is the project worth continuing as-is? Possible angles: source-match to the exact NWS settlement station per series (lifts the `between` ceiling — see caveat); restrict to tails only; different markets/lead times; or accept the market is efficient here.
- Definitive numbers are in `/tmp/harness_final.log`; re-run with `PYTHONPATH=. python -m backtest.real_market_eval --n-estimators 500`.

---

## (historical) NEXT SESSION plan — Build the real-markets evaluation harness  ✅ done above

**Goal:** the first trustworthy real-price evaluation. Score the strategy against the ACTUAL Kalshi markets at their real thresholds/settlements, not synthetic median-threshold markets.

**Why a new harness (not a runner.py tweak):** `backtest/runner.py` is built around synthetic station×month-median thresholds for the climatology benchmark. Real markets have their own thresholds and binary settlements. The two can't be reconciled by nearest-threshold matching (that's bug #3).

**Inputs available:**
- `data/historical/kalshi_prices.parquet` — real markets with columns `ticker, series, station, date, threshold, market_type (above/below), d1_mid (real decision-time price), settlement (1.0/0.0), volume`.
- `data/historical/features.parquet` — model features per (station, date, lead_hour); covers through 2026-05-27 (so real-market eval overlaps ~Apr 11 – May 27).
- Trained models via `models.registry.load_latest_artifact` (per `{station}_{lead_bucket}`), `_compute_fair_value` in `strategies/ensemble_strategy.py`.

**Harness spec (proposed):**
1. For each real market row (station, date, threshold, market_type) with a non-null `settlement` AND a real `d1_mid`:
2. Get the model's fair value AT THAT EXACT threshold/type/date via the ensemble (`_compute_fair_value` with `threshold=market.threshold`, correct `market_type` — note "below" = `1 - P(above)`).
3. **No look-ahead:** do NOT use the all-data production models for the eval window. Either (a) train walk-forward models up to date−1, or (b) restrict eval to a window the production models did not train on. Decide first — this is the crux of validity.
4. edge = fair − d1_mid; trade if |edge| ≥ MIN_EDGE; size via `compute_size` (or flat $1 for a clean Sharpe); settle P&L against `settlement` with the 5% fee.
5. Aggregate: per-market P&L, win rate, **model Brier vs settlement compared to market Brier (0.068)** — if model Brier ≥ 0.068, there is NO edge. Report daily/monthly Sharpe with explicit sample-size caveats (~6–7 weeks of overlap only).

**Watch out for:** market_type direction (above vs below); the features×real-price overlap is short (Apr 11–May 27) → small sample, fragile Sharpe; the production models have look-ahead over this window (must address in step 3); volume-weight or filter illiquid markets.

---

## Current State (as of this handoff) — ⚠️ partly superseded by the correction above

### What works end-to-end
- Live ingestion: GEFS (20 ensemble members), ECMWF, NBM all ingest and build a feature matrix
- Model registry: NGBoost + QRF + ResidualModel + IsotonicCalibrator artifacts stored per `(station, lead_bucket)` key
- Strategy loop (`EnsembleStrategy.run_cycle`): fetches active tickers, builds features, selects the correct per-lead-bucket model, computes calibrated fair value, applies Kelly sizing, submits paper trades
- Risk controls: drawdown limits, per-ticker cooldowns, CI-width gate
- `PAPER_TRADING = True`, `BOT_ACTIVE = True` — no live money is at risk
- 310 passing tests (full suite, as of 2026-06-23)

### Backtest results (just completed — 24 folds, May 2024 → Apr 2026)
Models trained with per-lead-bucket specialisation (D1-2, D3-4, D5-7) on expanding window.

| Metric | Value |
|---|---|
| Mean CRPS | 3.562 |
| Mean Brier score | 0.0582 |
| Mean reliability slope | 0.984 (1.0 = perfect; std = 0.058) |
| Folds with slope < 0.90 | 2 / 24 |
| **vs. Climatological market prices** | |
| Total simulated P&L (21 clim folds) | $156,753 |
| Monthly mean / std | $7,464 / $2,538 |
| Annualised Sharpe (clim baseline) | **10.2** |
| **vs. Real Kalshi prices** | ❌ **FAKE — see correction banner above.** The "$95,983 / 639 trades" was the model beating a fabricated constant 0.5, NOT real prices. Do not cite. |

**Critical caveat (now superseded):** This section predates the 2026-06-23 discovery that the "real Kalshi price" pipeline was feeding a fabricated 0.5 to 94% of trades (bug #1) and that the backtest can't consume real prices anyway (bug #3). Treat ALL real-price numbers here as invalid. The climatology-vs numbers (Sharpe ~3.05/10.2) are real but only measure edge-vs-climatology, which is necessary-not-sufficient. Param-stability + Monte Carlo were since run (see correction banner).

### Saved outputs ready for analysis
- `data/backtest_results.csv` — 24-fold fold-level metrics
- `data/backtest_trades.parquet` — 13,208 individual trade records: `trade_prob`, `market_mid`, `outcome`, `contract_size`, `sigma`, `mu`, `lead_hour`, `station`, `date`, `threshold`, `is_real_price`

---

## Files Changed This Session

### New files
| File | Purpose |
|---|---|
| `scripts/param_stability.py` | Parameter stability analysis — varies MIN_EDGE, KELLY_FRACTION, MAX_EXPOSURE, SIGMA_FLOOR across grids; re-runs simulate_pnl on saved trade data without retraining |
| `scripts/montecarlo.py` | Three Monte Carlo modes: fold bootstrap (12/24m), trade bootstrap (12m), outcome perturbation (calibration stress test with 0–20% outcome flipping) |

### Modified files (this session's fixes)

**Fix 1 — Calibrator spectrum expansion** (`models/calibration.py`, `scripts/initial_train.py`, `backtest/runner.py`)
- Added `_CALIBRATION_PERCENTILES = [5,15,25,35,50,65,75,85,95]`
- Added `build_calibration_dataset()` — trains calibrator on 9 percentile thresholds instead of the single median, so isotonic regression covers the full [0.05, 0.95] probability range actually seen at inference

**Fix 2 — Fee-adjusted Kelly** (`trading/kelly.py`)
- Changed `b = (1 - market_price) / market_price` → `b = (1 - (1 + FEE_RATE) * market_price) / market_price`
- Imported `FEE_RATE = 0.05` from `backtest/track_b.py`; at `market_price > ~0.952` the bet is now correctly zeroed out

**Fix 3 — ecmwf_diurnal_range noise injection** (`backtest/runner.py`)
- Added `"ecmwf_diurnal"` to the substring list that selects columns for forecast noise perturbation; the column was previously silently skipped

**Fix 4 — Kelly-sized simulate_pnl** (`backtest/track_b.py`, `backtest/runner.py`)
- Added `contract_sizes: np.ndarray | None` parameter to `simulate_pnl`
- Backtest now computes per-row Kelly sizes using `compute_size()` and passes them to `simulate_pnl`, so backtest P&L is in realistic dollar terms rather than flat $1/trade

**Fix 5 — Per-station backtest training** (`backtest/runner.py`)
- Replaced single pooled NGBoost+QRF fold with a per-station loop using `pd.Series` index-alignment for accumulation
- Mirrors `initial_train.py` architecture so backtest metrics actually predict production performance
- Stations with < 100 training rows are skipped with a warning

**Fix 6 — Zero ecmwf_diurnal_range filtering** (`backtest/runner.py`)
- Added `train_df = train_df[train_df["ecmwf_diurnal_range"] != 0.0]` after loading training data
- Removes 1,644 ERA5 artefact rows (where tmax=tmin=t2m_f) that would teach a spurious "is this ERA5?" signal

**Fix 7 — Per-lead-bucket model registry** (`processing/bias_correction.py`, `strategies/ensemble_strategy.py`, `api/main.py`, `scripts/initial_train.py`, `backtest/runner.py`)
- Added `LEAD_BUCKET_HOUR_RANGES = [("D1-2", 0, 48), ("D3-4", 49, 96), ("D5-7", 97, 999)]`
- Models saved/loaded with composite key `f"{station}_{lead_bucket}"` (e.g., `"KORD_D3-4"`)
- `ensemble_strategy.run_cycle` computes `lead_bucket` before model lookup; falls back to bare `station` key for backward compatibility
- `api/main.py` loads a 3× larger registry (20 stations × 3 buckets)
- `initial_train.py` inner loop over `LEAD_BUCKET_HOUR_RANGES` trains separate NGBoost+QRF+calibrator per bucket
- `backtest/runner.py` adds an inner lead-bucket loop inside the per-station loop

**Trade-level data saving** (`backtest/runner.py`)
- `_run_fold` now appends a row-per-trade DataFrame to `data/backtest_trades.parquet` after each fold
- Enables parameter stability testing without retraining models

---

## Tests Added This Session

All 293 tests pass. New tests cover:

- `tests/test_backtest_runner.py`: 16 tests — per-station training, lead-bucket inner loop (3 fits per station), station skip logic, zero diurnal range filtering, noise injection on diurnal column, sigma floor constant
- `tests/test_kelly.py`: 5 tests — fee-adjusted sizing, Kelly zeros at market_price=0.95, Kelly-sized simulate_pnl linearity, contract_sizes array acceptance
- `tests/test_calibration_pr4.py`: 5 tests — `build_calibration_dataset` importability, returns 9× rows, spans full prob range
- `tests/test_ensemble_strategy_inference.py`: 4 tests — `LEAD_BUCKET_HOUR_RANGES` importable and covers all buckets, `run_cycle` selects D1-2 model for horizon=1, D3-4 model for horizon=3

---

## Failed Attempts / Gotchas

**Initial timing estimate was badly wrong.** Estimated 97s/fold from a test with a small training set; actual folds took 2–3 hours each for later folds because the expanding training window grows to 4+ years of data by fold 20+. The backtest took ~17 hours end-to-end. Future runs: reduce `n_estimators` in `_run_fold` from 200 to 100, or use `--train-years 2` for iteration.

**Coverage gate killed the `run_cycle` tests.** The `_compute_coverage_by_bucket` method returns 0.0 for all buckets when `gefs_raw={}` (empty mock). The 0.50 gate then skips every ticker. Fix: added `patch.object(strategy, "_compute_coverage_by_bucket", return_value=full_coverage)` to both inference tests.

**`test_run_fold_excludes_zero_diurnal_range_from_training` needed larger data.** First version used 60 non-zero rows, which fell below the 100-row per-bucket minimum after the per-lead-bucket loop was added, producing no predictions and an assertion error. Fixed by scaling to 300 total / 150 non-zero rows.

**`test_run_fold_skips_station_with_insufficient_train_data` had shape mismatch.** After per-station loop with KLAX skipped (50 rows), `train_df` was 170 rows but arrays were only 120 elements. Fixed by filtering `train_df = train_df.loc[valid_train]` before reassigning the flat arrays.

**Calibrator test premise was wrong.** First version asserted single-threshold calibrator probs "cluster near 0.5 (min > 0.2, max < 0.8)" but NGBoost naturally assigns varied probs even at a single threshold. Redesigned tests to focus on the production `build_calibration_dataset` function directly.

---

## Session Update (2026-06-22) — Steps 1–6 executed

All six next-step items below were run this session. Summary of outcomes:

- **Step 1 (param stability):** Sharpe flat at ~3.05 across every grid value. `kelly_fraction` and `max_exposure_usd` are pure leverage (P&L scales linearly, win_rate identical, Sharpe unchanged) — the script's "optimal=1.0/$500" is just "bet more," not a real edge. `min_edge_cents` mild peak ~0.08–0.10, `sigma_floor` inert. **Keep config as-is — flat plateau, not a fragile peak.** Outputs in `data/stability/`. Caveat: `max_monthly_drawdown=0.000` everywhere is a climatology-pricing artifact.
- **Step 2 (Monte Carlo):** Ruin prob 0.0% across fold/trade bootstraps. Outcome perturbation stays profitable through 20% flipping (mean $191k at 20%, prob_neg 0%) — comfortably clears the 10% cushion bar. Outputs in `data/montecarlo/`.
- **Step 6 (P&L variance):** **Systematic/seasonal, not noise.** Edge-per-trade tracks forecast difficulty: shoulder/winter months (Feb–Mar ~$23/trade, Sep–Oct ~$19) vs summer (Jun/Aug ~$9). corr(pnl_per_trade, CRPS)=+0.37 — climatology is a worse benchmark when weather variance is high, so relative edge grows. Plot + table in `data/fold_variance/`. NOTE: this is edge-vs-climatology seasonality; real markets likely price it.
- **Step 3 (retrain):** DONE. All 20 stations × 3 buckets retrained with composite keys (`KATL_D1-2`). 60/60 ngboost+qrf+calibrators present, blender weights ngboost 0.529 / qrf 0.471. Took ~21 min (NOT 2–4 hr — `train_final_models` fits each station once, no walk-forward). Old models backed up to `data/models_backup_20260622_111846/`. GOTCHA: call `train_final_models` directly (not `main()`, which re-runs the 17 hr backtest); it needs `init_db()` first because `save_artifact` writes DB metadata.
- **Step 5 (Kalshi data gap):** Re-fetched + merged. Coverage now **2026-03-26 → 2026-06-21 (88 days, +31%)**. KEY FINDING: Kalshi's settled-markets endpoint serves only a **rolling ~10-week window** — re-fetching rolled forward (gained June, dropped late March). **2024 backfill is impossible via the API**; the only path is forward-accumulation (schedule the fetch periodically + union). Candlestick endpoint 404s for expired markets (falls back to prev bid/ask proxy; that's why the fetch takes ~22 min of retries).
- **Step 4 (shadow verification):** Inference path verified OFFLINE on recent feature rows (not yet a live-API cycle). Checks PASS: all 20×3 buckets produce forecasts; composite keys resolve 60/60 (no bare-key fallback); Kelly non-zero on visible edge 60/60. **Check 2 FINDING:** 4/60 fair values out of [0.05,0.95], all coastal CA (KSFO/KLAX), raw_prob already 0.97–0.995 pre-calibration (low-variance marine climate + learned ECMWF cold-bias correction — not a calibrator bug). **KSFO_D1-2 = exactly 1.000** → recommend clamping cal_prob to e.g. [0.02, 0.98] to prevent overconfident Kelly sizing (needs failing test + approval first).

### Follow-up items DONE (same session, after the above)

- **Fair-value clamp (TDD):** Added `FAIR_VALUE_FLOOR=0.02` / `FAIR_VALUE_CEIL=0.98` in `strategies/ensemble_strategy.py`; `_compute_fair_value` now clamps the returned `cal_prob` to [0.02, 0.98]. 5 new tests in `tests/test_ensemble_strategy_inference.py` (ceiling, floor, in-range passthrough, uncalibrated-extreme, bounds-sane). Full suite **298 passed** (was 293). Verified: former offenders (KSFO_D1-2 etc.) now cap at exactly 0.980; no cal_prob = 1.0 anywhere. NOTE clamp is [0.02,0.98], intentionally wider than the [0.05,0.95] sanity band — 0.98 is legitimate high confidence at low-variance marine stations; the goal was killing the exactly-1.0 "zero-loss" value that breaks Kelly.

- **Live `run_cycle` (executed):** Ran one real inference-only cycle (registry load + live KalshiClient paper mode + live ingestion). CONFIRMED: production registry loads in the live path (60 ngboost / 60 qrf / 60 residual / 60 calibrators); Kalshi paper client connects ("NO real orders will be submitted"); ingestion degrades gracefully on missing data. **GEFS finding (timing, not a bug):** the latest run (today 18z) wasn't published yet at run time → NOMADS returns **403** for a not-yet-existing run dir. Verified: today's `18z`→403 but `06z`→200 (User-Agent irrelevant). So no live forecasts were produced this cycle — purely because the newest model run wasn't out. Live forecast-production behavior (fair values in range, all 3 buckets, non-zero Kelly) was already confirmed offline on recent features.

**Recommended next (optional robustness):** `fetch_latest_gefs_run` tries only the latest cycle and logs "not yet available" when it 403s; consider a fallback to the most recent *published* run (06z/12z) so a manually-triggered cycle isn't empty between model publications. Needs TDD + approval.

**Original pending items (now both done above):** (a) live `run_cycle`; (b) cal_prob clamp.

---

## Original Next Steps (now executed — see Session Update above)

### 1. Run parameter stability analysis (immediate — ~5 minutes)
```bash
PYTHONPATH=. python scripts/param_stability.py
```
Reads `data/backtest_trades.parquet`. Re-runs `simulate_pnl` with varied `MIN_EDGE_CENTS`, `KELLY_FRACTION`, `MAX_EXPOSURE_USD`, and `RESIDUAL_SIGMA_FLOOR` across grids of 8 values each. Outputs Sharpe-vs-parameter curves to `data/stability/`. Key question: is the current configuration near a Sharpe peak, or is there a clearly better setting?

### 2. Run Monte Carlo simulation (immediate — ~10 minutes)
```bash
PYTHONPATH=. python scripts/montecarlo.py
```
Three modes on the same trade data:
- Fold bootstrap: P&L distribution over 12/24 synthetic months
- Trade bootstrap: bottom-up resample
- **Outcome perturbation** (most important): flips outcomes with p=0–20% to stress-test calibration. If the strategy survives 10% outcome flipping (i.e., 10% of model wins become losses and vice versa), it has a real edge cushion. If it breaks at 5%, calibration is the critical risk.

### 3. Run initial_train.py to produce per-lead-bucket production models
```bash
PYTHONPATH=. python scripts/initial_train.py
```
The models in `data/models/` still use bare station keys (`KATL`, not `KATL_D1-2`). The live inference loop will fall back to the bare key if it exists, but the production models should be retrained with the new per-lead-bucket scheme. Expect ~2–4 hours for all 20 stations × 3 buckets.

### 4. Live shadow run (pre-deployment verification)
After retraining, run one strategy cycle against the live Kalshi API with `PAPER_TRADING=True`:
- Confirm composite model keys (`KORD_D3-4`) resolve to non-None models
- Confirm fair values are in range [0.05, 0.95]
- Confirm Kelly sizes are non-zero for markets with visible edge
- Confirm all three lead buckets produce forecasts

### 5. Fix the real-price Kalshi data gap
Only 3 of 24 backtest folds had real Kalshi market prices (Mar–Apr 2026). The backtest is essentially evaluating the model against itself (climatology), which inflates the Sharpe. Obtaining historical Kalshi order-book data for 2024 would give a much more honest performance estimate. The `data/kalshi_prices.parquet` file exists but is sparse.

### 6. Investigate large P&L variance across folds
The clim-baseline folds swing from $2,701 (Jun 2025) to $12,699 (Sep 2024). This may reflect seasonality (larger temperature variance in shoulder seasons → more edge vs. climatology). Plotting P&L vs. month of year across folds would confirm whether this is systematic or noise.

---

## Key Configuration (do not change before paper trading)
```
PAPER_TRADING = True      # in .env — must stay True
BOT_ACTIVE    = True      # in .env — controls run_cycle execution
KELLY_FRACTION           = 0.25
MIN_EDGE_CENTS           = 4.0
MAX_EXPOSURE_PER_TICKER_USD = 200.0
RESIDUAL_SIGMA_FLOOR     = 2.0   # in backtest/runner.py AND strategies/ensemble_strategy.py
```
