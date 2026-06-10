# Model Gaps Remediation Plan

15 gaps grouped into 6 PRs, ordered from lowest-risk/highest-confidence to highest-risk/largest-scope.

---

## PR 1 — Backtest: Surface failures instead of masking them
**Files:** `backtest/runner.py`

| # | Gap | Fix |
|---|-----|-----|
| 13 | NaN predictions silently imputed with `nanmedian` before metrics | Raise `ValueError` (or skip fold) instead of imputing; log the fold and root feature that produced NaN |
| 14 | Empty fold returns `crps=nan, pnl=0` silently | Raise `ValueError` with reason (empty train vs empty test) so callers see the data gap |
| 15 | Threshold matching has no distance guard | Reject match and fall back to climatological mid if `abs(closest["threshold"] - threshold) > 5.0` |

---

## PR 2 — Features: Temporal correctness
**Files:** `processing/features.py`

| # | Gap | Fix |
|---|-----|-----|
| 5 | `obs_minus_model_lag1/2/3` uses positional `[-3:]` with no timestamp check | Sort by date before slicing; assert last entry is within 3 days of reference date or set lags to NaN |
| 6 | Regime cluster one-hot always uses `regime_labels.iloc[-1]` | Accept `reference_date` param in `build_feature_row`; look up label by date rather than always taking latest |

---

## PR 3 — Wire unused model components
**Files:** `backtest/runner.py`, `models/spread_inflation.py`, `models/residual_model.py`

| # | Gap | Fix |
|---|-----|-----|
| 1 | `apply_spread_inflation()` never called | After `ngb.predict_distribution()`, call `apply_spread_inflation(mu, sigma, member_temps)` using GEFS member temps from the feature row |
| 2 | `ResidualModel` trained but predictions never applied to mu | After NGBoost prediction, load or fit `ResidualModel`; add `residual_model.predict(X_test)` to `mu_test` |

Note: GEFS member temps need to be threaded from the features parquet into `_run_fold` — check if `gefs_member_*` columns exist before wiring.

---

## PR 4 — Calibration correctness
**Files:** `backtest/runner.py`, `models/calibration.py`, `processing/bias_correction.py`

| # | Gap | Fix |
|---|-----|-----|
| 7 | Calibrator trained on global threshold, applied to per-station trade probs | Train a second per-station-month calibrator (or fit the calibrator on training-set per-row probs); avoid using global-threshold ISO regression for per-station trades |
| 8 | `MIN_SAMPLES=100` hard-falls back to raw probs | Add weighted partial fit: blend isotonic (if ≥30 samples) with prior-smoothed estimate; document the 100-sample threshold and why |
| 9 | Kalman bias correction: hardcoded `process_noise=0.1`, `obs_noise=1.5`, 60-day window | Expose these as config params in `KalmanBiasCorrector.__init__`; derive window adaptively per station by inspecting seasonal bias autocorrelation |

---

## PR 5 — Backtest coverage: QRF, blend weights, noise calibration
**Files:** `backtest/runner.py`, `models/qrf_model.py`, `models/blend.py`

| # | Gap | Fix |
|---|-----|-----|
| 3 | QRF not evaluated walk-forward; blend weights always 0.5/0.5 | Add `QRFTemperatureModel` to `_run_fold`; compute out-of-fold CRPS for NGBoost and QRF; weight blend by inverse-CRPS |
| 11 | Climatological fallback adds fixed-seed Gaussian noise (`std=0.02`) | Remove noise; use raw clim prob as mid (it already ~0.50 by construction); note in log that it's unnoised |
| 12 | Forecast-error fallback stds `{D1-2: 3°F, D3-4: 5°F, D5-7: 7°F}` are placeholder | Compute empirical stds from `data/historical/features.parquet` residuals (`gefs_tmax_mean - actual_tmax`) grouped by lead bucket; write result to `data/historical/forecast_error_distributions.parquet` if not present |

---

## PR 6 — Market coverage: tmin and multi-day leads
**Files:** `backtest/runner.py`

| # | Gap | Fix |
|---|-----|-----|
| 4 | `market_type == "above"` hardcoded; no tmin/"below" support | Add `market_type` param to `_run_fold`; when `"below"`, swap target to `actual_tmin` and flip probability: `P(Tmin < threshold)` |
| 10 | Only `lead_hour == 24` (D+1) evaluated | Replace `d1_mask` with a `lead_hours` param (e.g. `[24, 48, 72, 96, 120, 144, 168]`); run simulation per lead bucket and report PnL decomposed by lead |

---

## Execution order

Work through PRs 1→6 in order. Each PR should be:
1. Tests written first (TDD — no production changes without a failing test)
2. Minimum code to make tests green
3. Mutation tested before moving on

PRs 1–2 are safe refactors with no model-quality risk.
PR 3 may change backtest PnL numbers — document the delta.
PR 4 may change calibration quality — compare Brier scores before/after.
PRs 5–6 expand scope — run full backtest and compare report before/after.
