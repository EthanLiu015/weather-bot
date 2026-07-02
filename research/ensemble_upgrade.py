"""Ensemble distribution upgrade sweep (autoresearch cycle 1).

The fresh same-day multi-model ensemble loses to the book on calibration
(model Brier 0.121 vs market 0.095) with a Gaussian around the bias-corrected
mean. This sweep asks whether the residual DISTRIBUTION SHAPE, a spread-skill
EMOS sigma, heavier tails, or inverse-MSE model weights close that gap.

All parameters are fit on the temporal TRAIN split; variants are ranked by
TRAIN Brier and every variant's TEST Brier is reported alongside the market's.

    PYTHONPATH=. python -m research.ensemble_upgrade
Data: data/historical/openmeteo_fresh.parquet, kalshi_prices.parquet, features.parquet
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize

from strategies.bracket_pricing import bracket_yes_prob
from backtest.real_market_eval import brier_score, _load_eval_markets, EVAL_START, EVAL_END
from research.multimodel_edge import ensemble_stats

logger = logging.getLogger(__name__)

FRESH_PATH = "data/historical/openmeteo_fresh.parquet"
FEAT_PATH = "data/historical/features.parquet"
FRESH_LEAD = 6
MIN_STATION_DAYS = 15


def load_joined(mm_path: str = FRESH_PATH, lead_hour: int = FRESH_LEAD) -> pd.DataFrame:
    mm = pd.read_parquet(mm_path)
    ens = ensemble_stats(mm, lead_hour=lead_hour)
    ens["date"] = pd.to_datetime(ens["date"])
    markets = _load_eval_markets("data/historical/kalshi_prices.parquet", EVAL_START, EVAL_END)
    markets["date"] = pd.to_datetime(markets["date"])
    feats = pd.read_parquet(FEAT_PATH)[["station", "date", "actual_tmax"]].dropna()
    feats["date"] = pd.to_datetime(feats["date"])
    actual = feats.drop_duplicates(["station", "date"])
    df = markets.merge(ens, on=["station", "date"], how="inner").merge(
        actual, on=["station", "date"], how="left")
    return df[df["ens_mean"].notna() & df["actual_tmax"].notna()].reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train_frac: float = 0.5):
    dates = np.sort(df["date"].unique())
    split = dates[int(len(dates) * train_frac)]
    return df[df["date"] < split], df[df["date"] >= split]


def station_days(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (station, date): residual vs ensemble mean + spread."""
    g = df.drop_duplicates(["station", "date"])[
        ["station", "date", "ens_mean", "ens_std", "actual_tmax"]].copy()
    g["resid"] = g["actual_tmax"] - g["ens_mean"]
    return g


def fit_station_bias(sd: pd.DataFrame) -> tuple[dict, float]:
    global_bias = float(sd["resid"].mean())
    per = {st: float(g["resid"].mean()) for st, g in sd.groupby("station")
           if len(g) >= MIN_STATION_DAYS}
    return per, global_bias


def fit_emos(sd: pd.DataFrame, biases: dict, global_bias: float):
    """sigma_day^2 = a + b * ens_std^2, Gaussian max-likelihood on train."""
    centered = sd["resid"].values - np.array(
        [biases.get(s, global_bias) for s in sd["station"]])
    spread2 = sd["ens_std"].values ** 2

    def nll(params):
        a, b = params
        var = np.maximum(a + b * spread2, 0.05)
        return float(np.sum(0.5 * np.log(2 * np.pi * var) + centered ** 2 / (2 * var)))

    res = minimize(nll, x0=[1.0, 0.5], method="Nelder-Mead")
    a, b = res.x
    return float(a), float(b)


def fit_student_t(sd: pd.DataFrame, biases: dict, global_bias: float):
    centered = sd["resid"].values - np.array(
        [biases.get(s, global_bias) for s in sd["station"]])
    df_, loc, scale = student_t.fit(centered, floc=0.0)
    return float(df_), float(scale)


def fit_model_weights(train: pd.DataFrame, mm_path: str = FRESH_PATH,
                      lead_hour: int = FRESH_LEAD) -> dict:
    """Inverse-MSE weights per model, fit on train station-days."""
    mm = pd.read_parquet(mm_path)
    d = mm[mm.lead_hour == lead_hour]
    wide = d.pivot_table(index=["station", "date"], columns="model", values="tmax_f").reset_index()
    wide["date"] = pd.to_datetime(wide["date"])
    sd = train.drop_duplicates(["station", "date"])[["station", "date", "actual_tmax"]]
    j = sd.merge(wide, on=["station", "date"], how="inner")
    weights = {}
    for m in ["aifs", "graphcast", "gfs", "icon", "ecmwf"]:
        if m in j:
            err = (j["actual_tmax"] - j[m]).dropna()
            if len(err) >= 30:
                weights[m] = 1.0 / max(float((err ** 2).mean()), 1e-6)
    tot = sum(weights.values())
    return {m: w / tot for m, w in weights.items()}


def weighted_mean_frame(df: pd.DataFrame, weights: dict, mm_path: str = FRESH_PATH,
                        lead_hour: int = FRESH_LEAD) -> pd.DataFrame:
    mm = pd.read_parquet(mm_path)
    d = mm[mm.lead_hour == lead_hour]
    wide = d.pivot_table(index=["station", "date"], columns="model", values="tmax_f").reset_index()
    wide["date"] = pd.to_datetime(wide["date"])
    cols = [m for m in weights if m in wide]
    w = np.array([weights[m] for m in cols])
    vals = wide[cols].values
    mask = ~np.isnan(vals)
    wsum = (np.where(mask, vals, 0) * w).sum(axis=1)
    wtot = (mask * w).sum(axis=1)
    wide["wmean"] = np.where(wtot > 0, wsum / wtot, np.nan)
    out = df.merge(wide[["station", "date", "wmean"]], on=["station", "date"], how="left")
    out["ens_mean"] = out["wmean"].fillna(out["ens_mean"])
    return out.drop(columns=["wmean"])


def make_empirical_sf(resids: np.ndarray, bandwidth: float = 0.8):
    """Smoothed empirical survival of residuals (Gaussian-kernel ECDF)."""
    r = np.asarray(resids, dtype=float)

    def sf(x: float, loc: float = 0.0) -> float:
        return float(np.mean(norm.sf((x - loc - r) / bandwidth)))

    return sf


def price_frame(df: pd.DataFrame, prob_above) -> pd.DataFrame:
    """prob_above(row) -> callable sf(x) = P(final > x). Returns priced rows."""
    fair, keep = [], []
    for i, row in enumerate(df.itertuples(index=False)):
        sf = prob_above(row)
        p = bracket_yes_prob(sf, row.strike_type, row.floor_strike, row.cap_strike)
        if p is None:
            continue
        fair.append(float(p))
        keep.append(i)
    out = df.iloc[keep].copy()
    out["fair"] = fair
    return out


def run(train_frac: float = 0.5) -> None:
    df = load_joined()
    train, test = temporal_split(df, train_frac)
    sd_train = station_days(train)
    logger.info("markets train=%d test=%d  station-days train=%d",
                len(train), len(test), len(sd_train))

    biases, gbias = fit_station_bias(sd_train)
    centered = sd_train["resid"].values - np.array(
        [biases.get(s, gbias) for s in sd_train["station"]])
    gsigma = float(np.std(centered))
    st_sigma = {st: float(g["resid"].std()) for st, g in sd_train.groupby("station")
                if len(g) >= MIN_STATION_DAYS}

    a_emos, b_emos = fit_emos(sd_train, biases, gbias)
    logger.info("EMOS: sigma^2 = %.3f + %.3f * spread^2", a_emos, b_emos)
    t_df, t_scale = fit_student_t(sd_train, biases, gbias)
    logger.info("Student-t: df=%.1f scale=%.2f", t_df, t_scale)
    weights = fit_model_weights(train)
    logger.info("inverse-MSE weights: %s", {m: round(w, 3) for m, w in weights.items()})
    emp_sf = make_empirical_sf(centered)
    per_station_resids = {st: (g["resid"].values - biases.get(st, gbias))
                          for st, g in sd_train.groupby("station")
                          if len(g) >= MIN_STATION_DAYS}

    def loc(row):
        return row.ens_mean + biases.get(row.station, gbias)

    variants = {
        "V0 gauss station bias, global sigma": lambda r: (
            lambda x: float(norm.sf(x, loc=loc(r), scale=gsigma))),
        "V1 gauss per-station sigma": lambda r: (
            lambda x: float(norm.sf(x, loc=loc(r), scale=st_sigma.get(r.station, gsigma)))),
        "V2 EMOS spread-skill sigma": lambda r: (
            lambda x: float(norm.sf(x, loc=loc(r),
                                    scale=max(np.sqrt(a_emos + b_emos * r.ens_std ** 2), 0.3)))),
        "V3 student-t tails": lambda r: (
            lambda x: float(student_t.sf(x, t_df, loc=loc(r), scale=t_scale))),
        "V4 empirical pooled ECDF": lambda r: (
            lambda x: emp_sf(x, loc=loc(r))),
        "V5 empirical per-station ECDF": lambda r: (
            (lambda x: make_empirical_sf(per_station_resids[r.station])(x, loc=loc(r)))
            if r.station in per_station_resids else (lambda x: emp_sf(x, loc=loc(r)))),
    }

    wtrain = weighted_mean_frame(train, weights)
    wtest = weighted_mean_frame(test, weights)
    sd_wtrain = station_days(wtrain)
    wbiases, wgbias = fit_station_bias(sd_wtrain)
    wa, wb = fit_emos(sd_wtrain, wbiases, wgbias)
    wcentered = sd_wtrain["resid"].values - np.array(
        [wbiases.get(s, wgbias) for s in sd_wtrain["station"]])
    wemp_sf = make_empirical_sf(wcentered)

    def wloc(row):
        return row.ens_mean + wbiases.get(row.station, wgbias)

    wvariants = {
        "V6 invMSE-mean + EMOS": lambda r: (
            lambda x: float(norm.sf(x, loc=wloc(r),
                                    scale=max(np.sqrt(wa + wb * r.ens_std ** 2), 0.3)))),
        "V7 invMSE-mean + empirical ECDF": lambda r: (
            lambda x: wemp_sf(x, loc=wloc(r))),
    }

    mkt_b_test = brier_score(test["d1_mid"].values, test["settlement"].values)
    mkt_b_train = brier_score(train["d1_mid"].values, train["settlement"].values)

    print("\n" + "=" * 78)
    print("ENSEMBLE DISTRIBUTION UPGRADE SWEEP (fresh same-day, fair lead)")
    print("=" * 78)
    print(f"  market Brier   train {mkt_b_train:.4f}   TEST {mkt_b_test:.4f}")
    print(f"  {'variant':<38} {'trainB':>8} {'testB':>8} {'beats mkt?':>11}")
    results = {}
    for name, pa in {**variants, **wvariants}.items():
        base_tr = train if not name.startswith(("V6", "V7")) else wtrain
        base_te = test if not name.startswith(("V6", "V7")) else wtest
        ptr = price_frame(base_tr, pa)
        pte = price_frame(base_te, pa)
        btr = brier_score(ptr["fair"].values, ptr["settlement"].values)
        bte = brier_score(pte["fair"].values, pte["settlement"].values)
        results[name] = (btr, bte)
        flag = "YES <---" if bte < mkt_b_test else "no"
        print(f"  {name:<38} {btr:>8.4f} {bte:>8.4f} {flag:>11}")
    best = min(results, key=lambda k: results[k][0])
    print(f"\n  best by TRAIN Brier: {best}  (test {results[best][1]:.4f} vs market {mkt_b_test:.4f})")
    print("=" * 78)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
