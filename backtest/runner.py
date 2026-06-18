import logging
from datetime import date, timedelta
from pathlib import Path
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

from backtest.leakage_audit import audit_no_leakage
from backtest.track_a import track_a_metrics
from backtest.track_b import simulate_pnl
from backtest.report import BacktestReport, FoldResult
from models.ngboost_model import NGBoostTemperatureModel
from models.calibration import IsotonicCalibrator
from models.residual_model import ResidualModel
from models.qrf_model import QRFTemperatureModel
from models.blend import ModelBlender
from models.spread_inflation import apply_spread_inflation_from_stats
from processing.features import get_feature_columns

logger = logging.getLogger(__name__)

# Number of contiguous windows the held-out validation slice is split into
# when estimating CRPS for NGBoost/QRF blend weights (see windowed_mean_score).
BLEND_VALIDATION_WINDOWS = 3

# Minimum predicted sigma (°F) applied after residual reframing shrinks the
# target distribution from ~18°F to ~5°F std. Must match RESIDUAL_SIGMA_FLOOR
# in strategies/ensemble_strategy.py so backtest and live inference are consistent.
RESIDUAL_SIGMA_FLOOR = 2.0


class BacktestRunner:
    def __init__(
        self,
        settings,
        start_date: date,
        end_date: date,
        train_window_years: int = 3,
    ) -> None:
        """train_window_years is the minimum amount of history (from
        start_date) required before the first test month; every fold after
        that trains on all data accumulated since start_date (expanding
        window)."""
        self._settings = settings
        self._start = start_date
        self._end = end_date
        self._train_window_years = train_window_years
        self._kalshi_prices = self._load_kalshi_prices()

    def run(self) -> BacktestReport:
        report = BacktestReport()
        test_month = self._start + relativedelta(years=self._train_window_years)

        # Anchored/expanding window: every fold trains on all data from
        # start_date up to the day before the test month, so the training
        # set grows with each fold — matching production models, which are
        # trained on the full available history rather than a rolling slice.
        train_start = self._start

        while test_month <= self._end:
            train_end = test_month - timedelta(days=1)
            logger.info("Running fold: train %s → %s, test month %s", train_start, train_end, test_month)

            try:
                fold = self._run_fold(train_start, train_end, test_month)
                report.folds.append(fold)
            except Exception as exc:
                logger.error("Fold %s failed: %s", test_month, exc)

            test_month += relativedelta(months=1)

        logger.info("Backtest complete: %d folds", len(report.folds))
        return report

    def _run_fold(self, train_start: date, train_end: date, test_month: date) -> FoldResult:
        feature_cols = get_feature_columns()

        train_df = self._load_historical_features(train_start, train_end)
        test_df = self._load_historical_features(test_month, test_month + relativedelta(months=1) - timedelta(days=1))

        if train_df.empty:
            raise ValueError(f"Fold {test_month}: train data is empty for {train_start} → {train_end}")
        if test_df.empty:
            raise ValueError(f"Fold {test_month}: test data is empty for {test_month}")

        ok, issues = audit_no_leakage(train_df, test_df, date_col="date", train_end=train_end)
        if not ok:
            raise ValueError(f"Leakage detected in fold {test_month}: {issues}")

        avail_cols = [c for c in feature_cols if c in train_df.columns]
        target_col = "actual_tmax"

        if target_col not in train_df.columns:
            logger.warning("No target column in fold %s — skipping model training", test_month)
            return FoldResult(
                fold_month=test_month,
                crps=float("nan"), mae=float("nan"), brier_score=float("nan"),
                reliability_slope=float("nan"), simulated_pnl_usd=0.0,
                num_simulated_trades=0, edge_above_threshold_pct=0.0,
            )

        X_train = train_df[avail_cols].fillna(0.0)
        y_train = train_df[target_col]
        X_test = test_df[avail_cols].fillna(0.0)
        y_test = test_df[target_col]

        # Residual reframing: ecmwf_tmax is an inference-time offset, not a feature.
        # Models train on (actual_tmax - ecmwf_tmax); ecmwf_test converts back at eval.
        ecmwf_train = train_df["ecmwf_tmax"].fillna(train_df["ecmwf_tmax"].mean()).values
        ecmwf_test = test_df["ecmwf_tmax"].fillna(test_df["ecmwf_tmax"].mean()).values
        y_train_residual = pd.Series(y_train.values - ecmwf_train, index=y_train.index)

        # Add forecast-error noise to ERA5-derived temperature features during training.
        # ERA5 reanalysis is near-perfect; real GEFS/ECMWF forecasts have uncertainty.
        # Noise std by lead bucket (°F): D1-2≈3°F, D3-4≈5°F, D5-7≈7°F
        X_train = self._add_forecast_noise(X_train, train_df, avail_cols)

        ngb = NGBoostTemperatureModel(n_estimators=200, learning_rate=0.05)
        ngb.fit(X_train, y_train_residual)
        mu_corr_test, sigma_test = ngb.predict_distribution(X_test)

        # ResidualModel corrects NGBoost's own residual errors. Both y_train_residual
        # and mu_corr_train are in residual space, so residuals_train is the
        # second-order correction (added in absolute space via ecmwf offset below).
        mu_corr_train, sigma_corr_train = ngb.predict_distribution(X_train)
        residuals_train = y_train_residual.values - mu_corr_train
        res_model = ResidualModel(station="all")
        res_model.fit(X_train, pd.Series(residuals_train, index=y_train.index))
        mu_corr_test = mu_corr_test + res_model.predict(X_test)

        # Blend with QRF using out-of-fold log-score weighting on residual targets
        # so CRPS values are comparable between NGBoost and QRF (same units).
        val_n = max(30, int(len(X_train) * 0.15))
        X_val_ngb = X_train.iloc[-val_n:]
        y_val_ngb = y_train_residual.iloc[-val_n:]
        ngb_log_score = ngb.crps(X_val_ngb, y_val_ngb, n_windows=BLEND_VALIDATION_WINDOWS)
        mu_blended_residual, sigma_test = self._fit_qrf_and_blend(
            X_train=X_train,
            y_train=y_train_residual,
            X_test=X_test,
            ngb_mu=mu_corr_test,
            ngb_sigma=sigma_test,
            ngb_log_score=ngb_log_score,
        )
        # Shift blended residual predictions to absolute temperature scale
        mu_test = mu_blended_residual + ecmwf_test

        # Widen sigma when ensemble members cluster tightly (overconfidence signal)
        if "gefs_tmax_std" in X_test.columns and "gefs_tmax_range" in X_test.columns:
            _, sigma_test = apply_spread_inflation_from_stats(
                mu=mu_test,
                sigma=sigma_test,
                std_arr=X_test["gefs_tmax_std"].fillna(0).values,
                range_arr=X_test["gefs_tmax_range"].fillna(0).values,
            )

        # Apply minimum sigma floor: residual reframing shrinks target std from
        # ~18°F to ~5°F, which can produce overconfident distributions.
        sigma_test = np.maximum(sigma_test, RESIDUAL_SIGMA_FLOOR)

        # ── Forecast skill metrics use a single global threshold ──────────────
        # NGBoost now predicts residuals; compute calibration probs in absolute space
        # using the blended absolute mu rather than calling predict_prob_above directly.
        from scipy.stats import norm as _norm_fold
        global_threshold = float(y_train.mean())
        prob_forecasts = 1.0 - _norm_fold.cdf(
            global_threshold, loc=mu_test, scale=np.maximum(sigma_test, 0.01)
        )
        outcomes = (y_test > global_threshold).astype(float).values

        calibrator = IsotonicCalibrator()
        mu_train_abs_cal = mu_corr_train + ecmwf_train
        prob_train = 1.0 - _norm_fold.cdf(
            global_threshold, loc=mu_train_abs_cal, scale=np.maximum(sigma_corr_train, 0.01)
        )
        outcomes_train = (y_train > global_threshold).astype(float).values
        calibrator.fit(prob_train, outcomes_train)
        cal_probs = calibrator._iso.predict(prob_forecasts) if calibrator._iso is not None else prob_forecasts

        logger.info("Fold debug: len(mu)=%d len(obs)=%d mu_range=[%.1f,%.1f]",
                    len(mu_test), len(y_test),
                    float(np.nanmin(mu_test)), float(np.nanmax(mu_test)))
        self._validate_predictions(mu_test, sigma_test, y_test.values, test_month)

        metrics = track_a_metrics(
            prob_forecasts=cal_probs,
            mu_forecasts=mu_test,
            sigma_forecasts=sigma_test,
            observations=y_test.values,
            outcomes=outcomes,
        )

        # ── Trading simulation uses per-station-month thresholds ─────────────
        # Station×month median → clim prob ≈ 0.50 by construction, so edge is
        # real model conviction, not a systematic station-temperature artefact.
        train_df = train_df.copy()
        train_df["_month"] = pd.to_datetime(train_df["date"]).dt.month
        station_month_median = (
            train_df.groupby(["station", "_month"])[target_col].median()
        )

        # Evaluate all lead hours (D1-D7); filter to [24] for backward compat
        eval_lead_hours = getattr(self._settings, "EVAL_LEAD_HOURS", [24])
        d1_test, mu_d1, sigma_d1 = self._filter_by_lead_hours(
            test_df=test_df, mu=mu_test, sigma=sigma_test, lead_hours=eval_lead_hours
        )

        market_type = getattr(self._settings, "MARKET_TYPE", "above")

        # Per-row threshold and probability
        row_thresholds = np.array([
            float(station_month_median.get(
                (row["station"], pd.to_datetime(row["date"]).month),
                global_threshold,
            ))
            for _, row in d1_test.iterrows()
        ])
        trade_probs_raw = self._compute_trade_prob(mu_d1, sigma_d1, row_thresholds, market_type)

        # Calibrate with a trade-specific calibrator trained on per-row training probs
        train_thresholds = np.array([
            float(station_month_median.get(
                (row["station"], pd.to_datetime(row["date"]).month),
                global_threshold,
            ))
            for _, row in d1_test.iterrows()  # only D1 rows in test; use training analogue
        ])
        # mu_corr_train is in residual space; add ecmwf_train to get absolute predictions
        # for the trade calibrator (thresholds are absolute temperatures).
        mu_train_pred = mu_corr_train + ecmwf_train
        sigma_train_pred = sigma_corr_train
        train_row_thresholds = np.array([
            float(station_month_median.get(
                (row["station"], pd.to_datetime(row["date"]).month),
                global_threshold,
            ))
            for _, row in train_df.iterrows()
        ])
        trade_calibrator = self._build_trade_calibrator(
            mu_train=mu_train_pred,
            sigma_train=sigma_train_pred,
            y_train=y_train.values,
            row_thresholds_train=train_row_thresholds,
        )
        trade_probs = trade_calibrator._iso.predict(trade_probs_raw) if trade_calibrator._iso is not None else trade_probs_raw
        if market_type == "below":
            trade_outcomes = (d1_test[target_col].values < row_thresholds).astype(float)
        else:
            trade_outcomes = (d1_test[target_col].values > row_thresholds).astype(float)

        # Build market mids and track which rows used real Kalshi prices
        market_mids, is_real = self._climatological_mids(
            train_df=train_df,
            test_df=d1_test,
            thresholds=row_thresholds,
            target_col=target_col,
            market_type=market_type,
        )

        sim = simulate_pnl(
            model_probs=trade_probs,
            market_mids=market_mids,
            outcomes=trade_outcomes,
            min_edge=self._settings.MIN_EDGE_CENTS / 100.0,
        )

        # Split PnL by price source: real Kalshi vs climatological
        real_sim = simulate_pnl(
            model_probs=trade_probs[is_real],
            market_mids=market_mids[is_real],
            outcomes=trade_outcomes[is_real],
            min_edge=self._settings.MIN_EDGE_CENTS / 100.0,
        ) if is_real.any() else {"simulated_pnl_usd": 0.0, "num_simulated_trades": 0}

        clim_sim = simulate_pnl(
            model_probs=trade_probs[~is_real],
            market_mids=market_mids[~is_real],
            outcomes=trade_outcomes[~is_real],
            min_edge=self._settings.MIN_EDGE_CENTS / 100.0,
        ) if (~is_real).any() else {"simulated_pnl_usd": 0.0, "num_simulated_trades": 0}

        return FoldResult(
            fold_month=test_month,
            crps=metrics["crps"],
            mae=metrics["mae"],
            brier_score=metrics["brier_score"],
            reliability_slope=metrics["reliability_slope"],
            simulated_pnl_usd=sim["simulated_pnl_usd"],
            num_simulated_trades=sim["num_simulated_trades"],
            edge_above_threshold_pct=sim["edge_above_threshold_pct"],
            real_price_pnl=real_sim["simulated_pnl_usd"],
            real_price_trades=real_sim["num_simulated_trades"],
            clim_price_pnl=clim_sim["simulated_pnl_usd"],
            clim_price_trades=clim_sim["num_simulated_trades"],
        )

    @staticmethod
    def _compute_trade_prob(
        mu: np.ndarray,
        sigma: np.ndarray,
        threshold: np.ndarray,
        market_type: str = "above",
    ) -> np.ndarray:
        from scipy.stats import norm as _norm
        cdf = _norm.cdf(threshold, loc=mu, scale=np.maximum(sigma, 0.01))
        if market_type == "below":
            return cdf
        return 1.0 - cdf

    @staticmethod
    def _filter_by_lead_hours(
        test_df: pd.DataFrame,
        mu: np.ndarray,
        sigma: np.ndarray,
        lead_hours: list[int] | None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if "lead_hour" not in test_df.columns or lead_hours is None:
            return test_df, mu, sigma
        mask = test_df["lead_hour"].isin(lead_hours)
        return test_df[mask].copy(), mu[mask], sigma[mask]

    @staticmethod
    def _fit_qrf_and_blend(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        ngb_mu: np.ndarray,
        ngb_sigma: np.ndarray,
        ngb_log_score: float,
        n_estimators: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        val_n = max(30, int(len(X_train) * 0.15))
        X_val, y_val = X_train.iloc[-val_n:], y_train.iloc[-val_n:]
        X_tr, y_tr = X_train.iloc[:-val_n], y_train.iloc[:-val_n]

        qrf = QRFTemperatureModel(n_estimators=n_estimators, min_samples_leaf=10)
        qrf.fit(X_tr, y_tr)

        qrf_log_score = qrf.log_score(X_val, y_val, n_windows=BLEND_VALIDATION_WINDOWS)
        blender = ModelBlender()
        # ngb_log_score / qrf_log_score are CRPS values (lower = better), but
        # compute_weights_from_log_scores treats higher = better, so negate.
        blender.compute_weights_from_log_scores(-ngb_log_score, -qrf_log_score)

        q_df = qrf.predict_quantiles(X_test)
        qrf_mu = q_df["q50"].values
        qrf_sigma = np.maximum((q_df["q75"].values - q_df["q25"].values) / 1.35, 0.01)

        return blender.blend_mu_sigma(ngb_mu, ngb_sigma, qrf_mu, qrf_sigma)

    @staticmethod
    def _compute_error_distributions(features_df: pd.DataFrame) -> pd.DataFrame:
        from processing.bias_correction import get_lead_bucket
        required = {"station", "lead_hour", "gefs_tmax_mean", "actual_tmax"}
        if not required.issubset(features_df.columns):
            return pd.DataFrame()
        df = features_df.dropna(subset=["gefs_tmax_mean", "actual_tmax"]).copy()
        df["residual"] = df["gefs_tmax_mean"] - df["actual_tmax"]
        df["lead_hours"] = df["lead_hour"].astype(int)
        df["month"] = pd.to_datetime(df["date"]).dt.month

        rows = []
        for (station, month, lead), grp in df.groupby(["station", "month", "lead_hours"]):
            if len(grp) >= 10:
                rows.append({
                    "station": station,
                    "month": month,
                    "lead_hours": lead,
                    "std_error_f": float(grp["residual"].std()),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _build_trade_calibrator(
        mu_train: np.ndarray,
        sigma_train: np.ndarray,
        y_train: np.ndarray,
        row_thresholds_train: np.ndarray,
    ) -> IsotonicCalibrator:
        from scipy.stats import norm as _norm
        probs = 1.0 - _norm.cdf(
            row_thresholds_train,
            loc=mu_train,
            scale=np.maximum(sigma_train, 0.01),
        )
        outcomes = (y_train > row_thresholds_train).astype(float)
        cal = IsotonicCalibrator()
        cal.fit(probs, outcomes)
        return cal

    @staticmethod
    def _validate_predictions(
        mu: np.ndarray,
        sigma: np.ndarray,
        observations: np.ndarray,
        fold_month,
    ) -> None:
        mu_nan = int(np.isnan(mu).sum())
        obs_nan = int(np.isnan(observations).sum())
        if mu_nan > 0 or obs_nan > 0:
            raise ValueError(
                f"Fold {fold_month}: {mu_nan} NaN mu predictions, {obs_nan} NaN observations — "
                "fix the upstream feature pipeline rather than imputing"
            )

    @staticmethod
    def _load_error_distributions() -> pd.DataFrame:
        err_path = Path("data/historical/forecast_error_distributions.parquet")
        if err_path.exists():
            df = pd.read_parquet(err_path)
            logger.info("Loaded empirical forecast error distributions: %d buckets", len(df))
            return df.set_index(["station", "month", "lead_hours"])
        return pd.DataFrame()

    def _add_forecast_noise(
        self,
        X: pd.DataFrame,
        meta_df: pd.DataFrame,
        avail_cols: list[str],
    ) -> pd.DataFrame:
        """
        Vectorised: add empirical forecast-error noise to ERA5 temperature features.
        Uses real ERA5-vs-ASOS error std per (station, month, lead_hours) bucket.
        Falls back to lead-time-scaled Gaussian when distributions unavailable.
        """
        TEMP_COLS = [c for c in avail_cols if any(
            k in c for k in ("gefs_tmax", "gefs_tmin", "ecmwf_tmax", "ecmwf_tmin", "ecmwf_diurnal", "nbm_t")
        )]
        if not TEMP_COLS or "lead_hour" not in meta_df.columns:
            return X

        err_lookup = self._load_error_distributions()
        FALLBACK_STD = {(0, 48): 3.0, (49, 96): 5.0, (97, 999): 7.0}
        avail_leads = sorted(err_lookup.index.get_level_values("lead_hours").unique()) if not err_lookup.empty else []

        # Build per-row std array — vectorised lookup
        meta = meta_df.reset_index(drop=True).copy()
        meta["_month"] = pd.to_datetime(meta["date"]).dt.month
        meta["_lead"]  = meta["lead_hour"].astype(int)

        def get_std(row) -> float:
            if not err_lookup.empty and avail_leads:
                closest = min(avail_leads, key=lambda l: abs(l - row["_lead"]))
                key = (row["station"], row["_month"], closest)
                try:
                    return float(err_lookup.loc[key, "std_error_f"])
                except KeyError:
                    pass
            for (lo, hi), fb in FALLBACK_STD.items():
                if lo <= row["_lead"] <= hi:
                    return fb
            return 5.0

        std_arr = meta.apply(get_std, axis=1).values  # shape (n,)

        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1, size=(len(X), len(TEMP_COLS))) * std_arr[:, None]

        X = X.copy()
        tc_idx = [X.columns.get_loc(c) for c in TEMP_COLS if c in X.columns]
        if tc_idx:
            X.iloc[:, tc_idx] = X.iloc[:, tc_idx].values + noise[:, :len(tc_idx)]
        return X

    @staticmethod
    def _load_kalshi_prices() -> pd.DataFrame:
        path = "data/historical/kalshi_prices.parquet"
        try:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            logger.info("Loaded %d real Kalshi market prices (%s → %s)",
                        len(df), df["date"].min(), df["date"].max())
            return df
        except FileNotFoundError:
            logger.info("No kalshi_prices.parquet found — using climatological mids")
            return pd.DataFrame()

    def _get_market_mid(
        self,
        station: str,
        date_: date,
        threshold: float,
        market_type: str = "above",
    ) -> float | None:
        """Return real Kalshi D+1 mid for (station, date, threshold, market_type) if available."""
        if self._kalshi_prices.empty:
            return None
        mask = (
            (self._kalshi_prices["station"] == station) &
            (self._kalshi_prices["date"] == date_) &
            (self._kalshi_prices["market_type"] == market_type)
        )
        candidates = self._kalshi_prices[mask]
        if candidates.empty:
            return None
        # Pick the contract whose threshold is closest to ours
        closest = candidates.iloc[(candidates["threshold"] - threshold).abs().argmin()]
        if abs(closest["threshold"] - threshold) > 5.0:
            return None
        mid = closest["d1_mid"]
        if pd.isna(mid) or mid <= 0 or mid >= 1:
            return None
        return float(mid)

    def _climatological_mids(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        thresholds: np.ndarray,
        target_col: str,
        market_type: str = "above",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute market mid for each test row using its per-row threshold.
        Thresholds are station×month medians, so clim_prob ≈ 0.50 — edge only
        appears when the model genuinely diverges from the historical base rate.

        Prefers real Kalshi prices when available, falls back to climatology.
        """
        mids = np.full(len(test_df), 0.5)
        is_real = np.zeros(len(test_df), dtype=bool)
        real_price_count = 0

        train_df = train_df.copy()
        train_df["_month"] = pd.to_datetime(train_df["date"]).dt.month

        for i, (_, row) in enumerate(test_df.iterrows()):
            threshold_i = float(thresholds[i])
            station = row.get("station")
            row_date = row["date"] if isinstance(row["date"], date) else pd.to_datetime(row["date"]).date()
            month = pd.to_datetime(row["date"]).month

            # Prefer real Kalshi market price at the closest matching threshold
            real_mid = self._get_market_mid(station, row_date, threshold_i, market_type)
            if real_mid is not None:
                mids[i] = real_mid
                is_real[i] = True
                real_price_count += 1
                continue

            # Fall back: P(Tmax > threshold_i | station, month) from training history
            # Using the station×month median as threshold keeps this near 0.5
            hist = train_df[
                (train_df["station"] == station) &
                (train_df["_month"] == month)
            ][target_col].dropna()

            if len(hist) >= 10:
                clim_prob = float((hist > threshold_i).mean())
                clim_prob = float(np.clip(clim_prob, 0.05, 0.95))
            else:
                clim_prob = 0.5

            mids[i] = float(np.clip(clim_prob, 0.02, 0.98))

        if real_price_count > 0:
            logger.info("Market mids: %d real Kalshi prices, %d climatological",
                        real_price_count, len(test_df) - real_price_count)
        return mids, is_real

    def _load_historical_features(self, start: date, end: date) -> pd.DataFrame:
        hist_path = "data/historical/features.parquet"
        try:
            df = pd.read_parquet(hist_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df[(df["date"] >= start) & (df["date"] <= end)].copy()
            return df
        except FileNotFoundError:
            logger.warning("Historical features not found at %s — returning empty", hist_path)
            return pd.DataFrame()
        except Exception as exc:
            logger.error("Error loading historical features: %s", exc)
            return pd.DataFrame()
