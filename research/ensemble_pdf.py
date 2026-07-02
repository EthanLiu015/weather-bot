"""True ensemble-member PDF edge test (autoresearch cycle 5) — FAIR lead.

Cycle 1-3 distributions were Gaussians with sigma from scalar regressors
(cross-model spread, NBM xnd). Here the distribution comes from actual ensemble
members (ECMWF ENS 50+ctl, GEFS 30+ctl; previous_day1 = fair/stale lead): the
member cloud carries flow-dependent spread AND shape (skew, bimodality) that no
scalar sigma can. Variants:

  P0  raw pooled member CDF, kernel-smoothed, zero fitting
  P1  wf per-station bias + spread rescale on member cloud
  P2  wf EMOS Gaussian: mu = mem_mean + bias, sigma^2 = a + b*mem_var
  P3  wf blend: mu = w*NBM12Z + (1-w)*mem_mean + bias, member-spread EMOS sigma

Walk-forward: every test date d priced with params fit on station-days < d.
Variant selection on TRAIN window; TEST scored once; block bootstrap over dates.

    PYTHONPATH=. python -m research.ensemble_pdf
Data: ensemble_members.parquet, nbm_station_12z.parquet, kalshi_prices.parquet,
      features.parquet
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

from strategies.bracket_pricing import bracket_yes_prob
from backtest.real_market_eval import brier_score, _load_eval_markets
from backtest.real_market_eval import EVAL_START, EVAL_END
from research.ensemble_upgrade import temporal_split
from research.ensemble_walkforward import bootstrap_dates

logger = logging.getLogger(__name__)

MEMBERS_PATH = "data/historical/ensemble_members.parquet"
NBM12_PATH = "data/historical/nbm_station_12z.parquet"
FEAT_PATH = "data/historical/features.parquet"
PRICES_PATH = "data/historical/kalshi_prices.parquet"
MIN_STATION_DAYS = 12
BURN_IN_DATES = 10
SIGMA_FLOOR = 0.55
KERNEL_H = 1.0


def member_table(members_path: str = MEMBERS_PATH) -> pd.DataFrame:
    """One row per (station, date): pooled member array + summary stats."""
    mem = pd.read_parquet(members_path)
    mem["date"] = pd.to_datetime(mem["date"])
    rows = []
    for (station, date), grp in mem.groupby(["station", "date"]):
        vals = grp["tmax_f"].values.astype(float)
        rows.append({
            "station": station, "date": date, "members": vals,
            "mem_mean": float(vals.mean()), "mem_std": float(vals.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def load_all(members_path: str = MEMBERS_PATH) -> pd.DataFrame:
    markets = _load_eval_markets(PRICES_PATH, EVAL_START, EVAL_END)
    markets["date"] = pd.to_datetime(markets["date"])
    feats = pd.read_parquet(FEAT_PATH)[["station", "date", "actual_tmax"]].dropna()
    feats["date"] = pd.to_datetime(feats["date"])
    actual = feats.drop_duplicates(["station", "date"])
    nbm = pd.read_parquet(NBM12_PATH)
    nbm["date"] = pd.to_datetime(nbm["date"])
    df = (markets
          .merge(member_table(members_path), on=["station", "date"], how="inner")
          .merge(actual, on=["station", "date"], how="left")
          .merge(nbm[["station", "date", "nbm_max_f"]], on=["station", "date"], how="left"))
    return df[df["actual_tmax"].notna()].reset_index(drop=True)


def station_day_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(["station", "date"])[
        ["station", "date", "members", "mem_mean", "mem_std", "nbm_max_f",
         "actual_tmax"]].reset_index(drop=True)


def _per_station_bias(resid: pd.Series, stations: pd.Series) -> tuple[dict, float]:
    g = float(resid.mean())
    per = {st: float(r.mean()) for st, r in resid.groupby(stations)
           if len(r) >= MIN_STATION_DAYS}
    return per, g


def _fit_emos(resid_centered: np.ndarray, driver_std: np.ndarray) -> tuple[float, float]:
    d2 = driver_std ** 2
    ok = (~np.isnan(resid_centered)) & (~np.isnan(d2))

    def nll(p):
        a, b = p
        var = np.maximum(a + b * d2[ok], 0.05)
        return float(np.sum(0.5 * np.log(2 * np.pi * var)
                            + resid_centered[ok] ** 2 / (2 * var)))

    a, b = minimize(nll, x0=[1.0, 0.5], method="Nelder-Mead").x
    return float(a), float(b)


def fit_wf_params(hist: pd.DataFrame) -> dict:
    """All calibration from station-days strictly in the past."""
    h = hist.copy()
    mem_bias, mem_g = _per_station_bias(h["actual_tmax"] - h["mem_mean"], h["station"])
    cen_mem = (h["actual_tmax"] - h["mem_mean"]).values - np.array(
        [mem_bias.get(s, mem_g) for s in h["station"]])
    emos_mem = _fit_emos(cen_mem, h["mem_std"].values)

    spread_var = float(np.nanmean(h["mem_std"].values ** 2))
    resid_var = float(np.nanvar(cen_mem))
    c2 = max(resid_var - KERNEL_H ** 2, 0.1) / max(spread_var, 0.1)
    spread_scale = float(np.sqrt(c2))

    hn = h[h["nbm_max_f"].notna()]
    if len(hn) >= MIN_STATION_DAYS:
        mse_nbm = float(((hn["actual_tmax"] - hn["nbm_max_f"]) ** 2).mean())
        mse_mem = float(((hn["actual_tmax"] - hn["mem_mean"]) ** 2).mean())
        w_nbm = (1 / mse_nbm) / (1 / mse_nbm + 1 / mse_mem)
        blend_mu = w_nbm * hn["nbm_max_f"] + (1 - w_nbm) * hn["mem_mean"]
        blend_bias, blend_g = _per_station_bias(hn["actual_tmax"] - blend_mu, hn["station"])
        cen_blend = (hn["actual_tmax"] - blend_mu).values - np.array(
            [blend_bias.get(s, blend_g) for s in hn["station"]])
        emos_blend = _fit_emos(cen_blend, hn["mem_std"].values)
    else:
        w_nbm, blend_bias, blend_g, emos_blend = 0.0, {}, mem_g, emos_mem

    return {"mem_bias": (mem_bias, mem_g), "emos_mem": emos_mem,
            "spread_scale": spread_scale, "w_nbm": w_nbm,
            "blend_bias": (blend_bias, blend_g), "emos_blend": emos_blend}


def make_sf(variant: str, row, p: dict):
    members = np.asarray(row.members, dtype=float)

    if variant == "P0":
        return lambda x: float(np.mean(norm.sf(x, loc=members, scale=KERNEL_H)))

    if variant == "P1":
        per, g = p["mem_bias"]
        mu = row.mem_mean + per.get(row.station, g)
        scaled = mu + p["spread_scale"] * (members - row.mem_mean)
        return lambda x: float(np.mean(norm.sf(x, loc=scaled, scale=KERNEL_H)))

    if variant == "P2":
        per, g = p["mem_bias"]
        a, b = p["emos_mem"]
        loc = row.mem_mean + per.get(row.station, g)
        s = max(np.sqrt(max(a + b * row.mem_std ** 2, 0.05)), SIGMA_FLOOR)
        return lambda x: float(norm.sf(x, loc=loc, scale=s))

    # P3: NBM12Z-weighted mean, member-spread EMOS sigma
    if pd.isna(row.nbm_max_f) or p["w_nbm"] == 0.0:
        return make_sf("P2", row, p)
    per, g = p["blend_bias"]
    a, b = p["emos_blend"]
    loc = (p["w_nbm"] * row.nbm_max_f + (1 - p["w_nbm"]) * row.mem_mean
           + per.get(row.station, g))
    s = max(np.sqrt(max(a + b * row.mem_std ** 2, 0.05)), SIGMA_FLOOR)
    return lambda x: float(norm.sf(x, loc=loc, scale=s))


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


def run(members_path: str = MEMBERS_PATH, label: str = "FAIR") -> None:
    df = load_all(members_path)
    sd = station_day_table(df)
    train, test = temporal_split(df)
    train_dates = np.sort(train["date"].unique())
    test_dates = np.sort(test["date"].unique())
    logger.info("markets train=%d test=%d  station-days=%d", len(train), len(test), len(sd))

    variants = {
        "P0 raw pooled member CDF — zero fitting": "P0",
        "P1 wf bias + spread-rescaled member CDF": "P1",
        "P2 wf EMOS Gaussian (mem_mean, mem_var)": "P2",
        "P3 wf NBM12Z blend + member-spread EMOS": "P3",
    }

    print("\n" + "=" * 84)
    print(f"ENSEMBLE MEMBER PDF EDGE (cycle 7) — {label} — {members_path}")
    print("=" * 84)
    sel = {}
    print(f"  {'variant':<44} {'selB(train-wf)':>14}")
    for name, v in variants.items():
        prc = walk_forward(df, sd, train_dates, v)
        b = brier_score(prc["fair"].values, prc["settlement"].values) if len(prc) else np.nan
        sel[name] = b
        print(f"  {name:<44} {b:>14.4f}")
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
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--unfair", action="store_true",
                    help="score the Open-Meteo assembled ECMWF members "
                         "(post-cutoff look-ahead; ceiling diagnostic only)")
    args = ap.parse_args()
    if args.unfair:
        run("data/historical/ensemble_members_unfair.parquet",
            label="UNFAIR CEILING (post-cutoff assembled ECMWF)")
    else:
        run()


if __name__ == "__main__":
    main()
