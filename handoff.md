# Kalshi Weather Bot — Handoff

## Goal

Build and deploy an automated options trading bot that bets on Kalshi temperature markets (e.g., "Will NYC max temp exceed 85°F on July 4?"). The core idea is that a probabilistic ensemble model trained on GEFS/ECMWF/NBM weather forecast data can price these binary outcomes more accurately than the market, and that a fee-adjusted Kelly criterion can size positions to maximise risk-adjusted returns.

The full pipeline: ingest live weather forecasts → build feature matrix → run NGBoost+QRF ensemble → calibrate probabilities → compute Kelly-sized orders → submit via Kalshi API → monitor/alert.

---

## Current State (as of this handoff)

### What works end-to-end
- Live ingestion: GEFS (20 ensemble members), ECMWF, NBM all ingest and build a feature matrix
- Model registry: NGBoost + QRF + ResidualModel + IsotonicCalibrator artifacts stored per `(station, lead_bucket)` key
- Strategy loop (`EnsembleStrategy.run_cycle`): fetches active tickers, builds features, selects the correct per-lead-bucket model, computes calibrated fair value, applies Kelly sizing, submits paper trades
- Risk controls: drawdown limits, per-ticker cooldowns, CI-width gate
- `PAPER_TRADING = True`, `BOT_ACTIVE = True` — no live money is at risk
- 293 passing tests (full suite)

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
| **vs. Real Kalshi prices (Mar–Apr 2026)** | |
| Total real-price P&L | $95,983 over 639 trades |

**Critical caveat on the Sharpe / P&L figures:** The majority of folds (21 of 24) use *climatological probabilities* as the synthetic market price, not real Kalshi order-book mid prices. Beating climatology is a necessary but not sufficient condition for profitability — real Kalshi markets already incorporate some weather signal. The only folds with real market prices are March and April 2026 (639 trades, $95,983 P&L). Those numbers look good but the sample is too small to draw conclusions. The parameter stability and Monte Carlo analyses still need to be run to properly evaluate the strategy.

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

## Next Steps (in priority order)

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
