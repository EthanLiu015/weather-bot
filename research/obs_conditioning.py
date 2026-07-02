"""Morning-obs conditioning at the 14:00 UTC cutoff (autoresearch cycle 5) — FAIR.

The book's last pre-cutoff trade (d1_mid) sees the morning's actual temperatures;
our NBM-only model does not. Two mechanisms recover that information:

  * hard physics: the day's high cannot be below the running max already
    observed by 14:00 UTC -> truncate the predictive distribution at runmax
  * trajectory: current temp + warming rate vs the NBM forecast shifts the mean

Obs come from the station Kalshi settles on (KNYC/KMDW for NY/CHI — cycle 4).

  M0  NBM12Z + wf bias + EMOS(xnd)             (cycle-3 baseline on this sample)
  M1  M0 + truncation at morning runmax
  M2  wf OLS mu(nbm, runmax, t_cut) + EMOS + truncation
  M3  M2 + warming-rate regressor (t_cut - t_11utc)

Walk-forward: every date d priced with params fit on station-days < d.
Variant selection on TRAIN window; TEST scored once; block bootstrap over dates.

    PYTHONPATH=. python -m research.obs_conditioning
Data: obs_hourly.parquet, nbm_station_12z.parquet, kalshi_prices.parquet,
      features.parquet
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

from config.stations import get_station
from strategies.bracket_pricing import bracket_yes_prob
from backtest.real_market_eval import (EVAL_END, EVAL_START, _load_eval_markets,
                                       brier_score)
from research.ensemble_upgrade import temporal_split
from research.ensemble_walkforward import bootstrap_dates

logger = logging.getLogger(__name__)

OBS_PATH = "data/historical/obs_hourly.parquet"
NBM12_PATH = "data/historical/nbm_station_12z.parquet"
FEAT_PATH = "data/historical/features.parquet"
PRICES_PATH = "data/historical/kalshi_prices.parquet"
CUTOFF_UTC_HOUR = 14
SLOPE_FROM_UTC_HOUR = 11
MIN_STATION_DAYS = 12
BURN_IN_DATES = 10
SIGMA_FLOOR = 0.55


def obs_features(obs: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per (station, date): runmax / t_cut / t_11 using only obs <= 14:00 UTC.

    runmax counts obs falling on the settlement date in station-local time,
    so it is a lower bound on the CLI daily high by construction."""
    rows = []
    for station, grp in obs.groupby("station"):
        tz = get_station(station).timezone
        local = grp["valid_utc"].dt.tz_localize("UTC").dt.tz_convert(tz)
        local_date = pd.to_datetime(local.dt.date)
        for d in dates:
            cutoff = d + pd.Timedelta(hours=CUTOFF_UTC_HOUR)
            before_cut = grp[grp["valid_utc"] <= cutoff]
            same_day = grp[(local_date == d) & (grp["valid_utc"] <= cutoff)]
            if before_cut.empty:
                continue
            last = before_cut.sort_values("valid_utc").iloc[-1]
            if (cutoff - last["valid_utc"]) > pd.Timedelta(hours=2):
                continue
            slope_base = before_cut[before_cut["valid_utc"] <= d + pd.Timedelta(
                hours=SLOPE_FROM_UTC_HOUR)]
            rows.append({
                "station": station, "date": d,
                "runmax": float(same_day["tmpf"].max()) if len(same_day) else np.nan,
                "t_cut": float(last["tmpf"]),
                "t_11": float(slope_base.sort_values("valid_utc").iloc[-1]["tmpf"])
                        if len(slope_base) else np.nan,
            })
    return pd.DataFrame(rows)


def load_all() -> pd.DataFrame:
    markets = _load_eval_markets(PRICES_PATH, EVAL_START, EVAL_END)
    markets["date"] = pd.to_datetime(markets["date"])
    feats = pd.read_parquet(FEAT_PATH)[["station", "date", "actual_tmax"]].dropna()
    feats["date"] = pd.to_datetime(feats["date"])
    actual = feats.drop_duplicates(["station", "date"])
    nbm = pd.read_parquet(NBM12_PATH)
    nbm["date"] = pd.to_datetime(nbm["date"])
    obs = pd.read_parquet(OBS_PATH)
    ofeat = obs_features(obs, pd.DatetimeIndex(np.sort(markets["date"].unique())))
    df = (markets
          .merge(nbm[["station", "date", "nbm_max_f", "nbm_sigma_f"]],
                 on=["station", "date"], how="inner")
          .merge(ofeat, on=["station", "date"], how="left")
          .merge(actual, on=["station", "date"], how="left"))
    df["warm_rate"] = df["t_cut"] - df["t_11"]
    return df[df["actual_tmax"].notna() & df["nbm_max_f"].notna()].reset_index(drop=True)


def station_day_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(["station", "date"])[
        ["station", "date", "nbm_max_f", "nbm_sigma_f", "runmax", "t_cut",
         "warm_rate", "actual_tmax"]].reset_index(drop=True)


def _per_station_bias(resid: pd.Series, stations: pd.Series) -> tuple[dict, float]:
    g = float(resid.mean())
    per = {st: float(r.mean()) for st, r in resid.groupby(stations)
           if len(r) >= MIN_STATION_DAYS}
    return per, g


def _fit_emos(resid_centered: np.ndarray, driver_std: np.ndarray) -> tuple[float, float]:
    d2 = np.nan_to_num(driver_std, nan=2.3) ** 2
    ok = ~np.isnan(resid_centered)

    def nll(p):
        a, b = p
        var = np.maximum(a + b * d2[ok], 0.05)
        return float(np.sum(0.5 * np.log(2 * np.pi * var)
                            + resid_centered[ok] ** 2 / (2 * var)))

    a, b = minimize(nll, x0=[1.0, 0.3], method="Nelder-Mead").x
    return float(a), float(b)


def _fit_ols(hist: pd.DataFrame, cols: list[str]) -> np.ndarray | None:
    h = hist.dropna(subset=cols + ["actual_tmax"])
    if len(h) < 10 * (len(cols) + 1):
        return None
    x = np.column_stack([np.ones(len(h))] + [h[c].values for c in cols])
    beta, *_ = np.linalg.lstsq(x, h["actual_tmax"].values, rcond=None)
    return beta


def fit_wf_params(hist: pd.DataFrame) -> dict:
    h = hist.copy()
    nbm_bias, nbm_g = _per_station_bias(h["actual_tmax"] - h["nbm_max_f"], h["station"])
    cen = (h["actual_tmax"] - h["nbm_max_f"]).values - np.array(
        [nbm_bias.get(s, nbm_g) for s in h["station"]])
    emos_nbm = _fit_emos(cen, h["nbm_sigma_f"].values)

    params = {"nbm_bias": (nbm_bias, nbm_g), "emos_nbm": emos_nbm}
    for key, cols in (("ols2", ["nbm_max_f", "runmax", "t_cut"]),
                      ("ols3", ["nbm_max_f", "runmax", "t_cut", "warm_rate"])):
        beta = _fit_ols(h, cols)
        if beta is None:
            params[key] = None
            continue
        hh = h.dropna(subset=cols + ["actual_tmax"])
        x = np.column_stack([np.ones(len(hh))] + [hh[c].values for c in cols])
        resid = hh["actual_tmax"].values - x @ beta
        params[key] = (beta, _fit_emos(resid, hh["nbm_sigma_f"].values))
    return params


def _truncate(sf, runmax: float):
    """Condition on T >= runmax (the high cannot be below what was observed)."""
    tail = sf(runmax)
    if tail is None or tail <= 1e-9:
        return sf
    return lambda x: 1.0 if x < runmax else min(float(sf(x)) / tail, 1.0)


def make_sf(variant: str, row, p: dict):
    xnd = row.nbm_sigma_f if not pd.isna(row.nbm_sigma_f) else 2.3
    per, g = p["nbm_bias"]
    a, b = p["emos_nbm"]
    loc0 = row.nbm_max_f + per.get(row.station, g)
    s0 = max(np.sqrt(max(a + b * xnd ** 2, 0.05)), SIGMA_FLOOR)
    base = lambda x: float(norm.sf(x, loc=loc0, scale=s0))

    if variant == "M0":
        return base
    if variant == "M1":
        return base if pd.isna(row.runmax) else _truncate(base, float(row.runmax))

    key = "ols2" if variant == "M2" else "ols3"
    cols = (["nbm_max_f", "runmax", "t_cut"] if variant == "M2"
            else ["nbm_max_f", "runmax", "t_cut", "warm_rate"])
    fitted = p.get(key)
    vals = [getattr(row, c) for c in cols]
    if fitted is None or any(pd.isna(v) for v in vals):
        return make_sf("M1", row, p)
    beta, (ea, eb) = fitted
    loc = float(beta[0] + np.dot(beta[1:], vals))
    s = max(np.sqrt(max(ea + eb * xnd ** 2, 0.05)), SIGMA_FLOOR)
    sf = lambda x: float(norm.sf(x, loc=loc, scale=s))
    return _truncate(sf, float(row.runmax))


def walk_forward(df: pd.DataFrame, sd: pd.DataFrame, eval_dates, variant: str) -> pd.DataFrame:
    priced = []
    for d in eval_dates:
        hist = sd[(sd["date"] < d) & sd["actual_tmax"].notna()]
        if hist["date"].nunique() < BURN_IN_DATES:
            continue
        params = fit_wf_params(hist)
        day = df[df["date"] == d]
        fair, keep = [], []
        for i, row in enumerate(day.itertuples(index=False)):
            sf = make_sf(variant, row, params)
            pr = bracket_yes_prob(sf, row.strike_type, row.floor_strike, row.cap_strike)
            if pr is None:
                continue
            fair.append(float(pr))
            keep.append(i)
        out = day.iloc[keep].copy()
        out["fair"] = fair
        priced.append(out)
    return pd.concat(priced, ignore_index=True) if priced else pd.DataFrame()


def run() -> None:
    df = load_all()
    sd = station_day_table(df)
    train, test = temporal_split(df)
    train_dates = np.sort(train["date"].unique())
    test_dates = np.sort(test["date"].unique())
    logger.info("markets train=%d test=%d  station-days=%d  runmax coverage=%.2f",
                len(train), len(test), len(sd), sd["runmax"].notna().mean())

    variants = {
        "M0 NBM12Z + wf bias + EMOS (baseline)": "M0",
        "M1 M0 + truncate at morning runmax": "M1",
        "M2 wf OLS(nbm, runmax, t_cut) + trunc": "M2",
        "M3 M2 + warming rate": "M3",
    }

    print("\n" + "=" * 84)
    print("MORNING-OBS CONDITIONING (cycle 5) — obs <= 14:00 UTC, FAIR")
    print("=" * 84)
    sel = {}
    print(f"  {'variant':<44} {'selB(train-wf)':>14}")
    for name, v in variants.items():
        prc = walk_forward(df, sd, train_dates, v)
        bsel = brier_score(prc["fair"].values, prc["settlement"].values) if len(prc) else np.nan
        sel[name] = bsel
        print(f"  {name:<44} {bsel:>14.4f}")
    best = min(sel, key=lambda k: sel[k])
    print(f"\n  selected on train walk-forward: {best}")

    print(f"\n  {'variant':<44} {'testB':>8} {'mktB':>8} {'beats?':>7}")
    for name, v in variants.items():
        prc = walk_forward(df, sd, test_dates, v)
        bte = brier_score(prc["fair"].values, prc["settlement"].values)
        mkt = brier_score(prc["d1_mid"].values, prc["settlement"].values)
        flag = "YES <--" if bte < mkt else "no"
        star = " *SELECTED*" if name == best else ""
        print(f"  {name:<44} {bte:>8.4f} {mkt:>8.4f} {flag:>7}{star}")
        if name == best:
            lo, hi, p_win = bootstrap_dates(prc)
            print(f"      bootstrap over {prc['date'].nunique()} test dates: "
                  f"90% CI [{lo:+.4f}, {hi:+.4f}]   P(model better) {p_win:.2f}")
    print("=" * 84)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
