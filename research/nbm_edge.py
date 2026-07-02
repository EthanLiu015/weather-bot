"""NBM station-bulletin edge test (autoresearch cycle 3) — FAIR lead.

The 07Z NBM NBS bulletin (posted ~08:30 UTC, before the 14:00 UTC `d1_mid`
decision cutoff) gives a station-calibrated daytime-max forecast `nbm_max_f`
AND the blend's own uncertainty `nbm_sigma_f`. Fair-lead MAE 1.86degF beats every
source we have. Question: does a walk-forward calibrated distribution around it
out-Brier the book?

Every test date d is priced with parameters fit only on station-days < d.
Variant selection on the TRAIN window walk-forward; TEST reported once.

    PYTHONPATH=. python -m research.nbm_edge
Data: nbm_station.parquet, openmeteo_multimodel.parquet (24h, stale/fair),
      kalshi_prices.parquet, features.parquet
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

from strategies.bracket_pricing import bracket_yes_prob
from backtest.real_market_eval import brier_score
from research.ensemble_upgrade import load_joined, temporal_split
from research.ensemble_walkforward import bootstrap_dates

logger = logging.getLogger(__name__)

NBM_PATH = "data/historical/nbm_station.parquet"
MM24_PATH = "data/historical/openmeteo_multimodel.parquet"
MIN_STATION_DAYS = 12
BURN_IN_DATES = 10
SIGMA_FLOOR = 0.55


def load_all() -> pd.DataFrame:
    df = load_joined(mm_path=MM24_PATH, lead_hour=24)  # markets + stale fair ens + actuals
    nbm = pd.read_parquet(NBM_PATH)
    nbm["date"] = pd.to_datetime(nbm["date"])
    df = df.merge(nbm[["station", "date", "nbm_max_f", "nbm_sigma_f"]],
                  on=["station", "date"], how="inner")
    return df[df["nbm_max_f"].notna()].reset_index(drop=True)


def station_day_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(["station", "date"])[
        ["station", "date", "nbm_max_f", "nbm_sigma_f", "ens_mean", "ens_std",
         "actual_tmax"]].reset_index(drop=True)


def fit_wf_params(hist: pd.DataFrame) -> dict:
    """Bias + blend weight + EMOS sigma from station-days strictly in the past."""
    h = hist.copy()
    e_nbm = h["actual_tmax"] - h["nbm_max_f"]
    e_ens = h["actual_tmax"] - h["ens_mean"]
    mse_nbm = float((e_nbm ** 2).mean())
    mse_ens = float((e_ens ** 2).mean())
    w_nbm = (1 / mse_nbm) / (1 / mse_nbm + 1 / mse_ens)

    def wf_bias(col_mu):
        resid = h["actual_tmax"] - h[col_mu] if isinstance(col_mu, str) else h["actual_tmax"] - col_mu
        g = float(resid.mean())
        per = {st: float(r.mean()) for st, r in resid.groupby(h["station"])
               if len(r) >= MIN_STATION_DAYS}
        return per, g

    h["blend_mu"] = w_nbm * h["nbm_max_f"] + (1 - w_nbm) * h["ens_mean"]
    nbm_b, nbm_g = wf_bias("nbm_max_f")
    blend_b, blend_g = wf_bias("blend_mu")

    def fit_emos(resid_centered, driver):
        d2 = driver.values ** 2
        ok = (~np.isnan(resid_centered)) & (~np.isnan(d2))

        def nll(p):
            a, b = p
            var = np.maximum(a + b * d2[ok], 0.05)
            return float(np.sum(0.5 * np.log(2 * np.pi * var)
                                + resid_centered[ok] ** 2 / (2 * var)))

        a, b = minimize(nll, x0=[1.0, 0.3], method="Nelder-Mead").x
        return float(a), float(b)

    cen_nbm = (h["actual_tmax"] - h["nbm_max_f"]).values - np.array(
        [nbm_b.get(s, nbm_g) for s in h["station"]])
    cen_blend = (h["actual_tmax"] - h["blend_mu"]).values - np.array(
        [blend_b.get(s, blend_g) for s in h["station"]])
    emos_nbm = fit_emos(cen_nbm, h["nbm_sigma_f"])
    emos_blend = fit_emos(cen_blend, h["nbm_sigma_f"])
    ok = ~np.isnan(cen_nbm)
    a, b = emos_nbm
    std_resid = cen_nbm[ok] / np.maximum(
        np.sqrt(a + b * h["nbm_sigma_f"].values[ok] ** 2), SIGMA_FLOOR)
    return {"w_nbm": w_nbm, "nbm_bias": (nbm_b, nbm_g), "blend_bias": (blend_b, blend_g),
            "emos_nbm": emos_nbm, "emos_blend": emos_blend,
            "sigma_nbm": float(np.nanstd(cen_nbm)), "std_resid": std_resid}


def make_sf(variant: str, row, p: dict):
    xnd = row.nbm_sigma_f if not pd.isna(row.nbm_sigma_f) else 2.3
    if variant == "N0":  # raw public product, zero fitting
        return lambda x: float(norm.sf(x, loc=row.nbm_max_f, scale=max(xnd, SIGMA_FLOOR)))
    if variant == "N1":  # wf bias + EMOS on xnd
        per, g = p["nbm_bias"]
        a, b = p["emos_nbm"]
        loc = row.nbm_max_f + per.get(row.station, g)
        s = max(np.sqrt(a + b * xnd ** 2), SIGMA_FLOOR)
        return lambda x: float(norm.sf(x, loc=loc, scale=s))
    if variant == "N2":  # blend with stale multimodel ensemble + EMOS
        if pd.isna(row.ens_mean):
            return make_sf("N1", row, p)
        per, g = p["blend_bias"]
        a, b = p["emos_blend"]
        loc = (p["w_nbm"] * row.nbm_max_f + (1 - p["w_nbm"]) * row.ens_mean
               + per.get(row.station, g))
        s = max(np.sqrt(a + b * xnd ** 2), SIGMA_FLOOR)
        return lambda x: float(norm.sf(x, loc=loc, scale=s))
    # N3: standardized empirical residuals scaled by EMOS sigma
    per, g = p["nbm_bias"]
    a, b = p["emos_nbm"]
    loc = row.nbm_max_f + per.get(row.station, g)
    s = max(np.sqrt(a + b * xnd ** 2), SIGMA_FLOOR)
    sr = p["std_resid"]
    return lambda x: float(np.mean(norm.sf((x - loc - s * sr) / 0.6)))


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
    logger.info("markets train=%d test=%d  station-days=%d", len(train), len(test), len(sd))

    variants = {
        "N0 raw NBM (txn, xnd) — zero fitting": "N0",
        "N1 wf bias + EMOS(xnd)": "N1",
        "N2 wf blend NBM+ens24 + EMOS": "N2",
        "N3 wf bias + std-empirical": "N3",
    }

    print("\n" + "=" * 84)
    print("NBM STATION EDGE (cycle 3) — 07Z bulletin, FAIR lead vs 14:00 UTC d1_mid")
    print("=" * 84)
    sel = {}
    print(f"  {'variant':<44} {'selB(train-wf)':>14}")
    for name, v in variants.items():
        p = walk_forward(df, sd, train_dates, v)
        b = brier_score(p["fair"].values, p["settlement"].values) if len(p) else np.nan
        sel[name] = b
        print(f"  {name:<44} {b:>14.4f}")
    best = min(sel, key=lambda k: sel[k])
    print(f"\n  selected on train walk-forward: {best}")

    print(f"\n  {'variant':<44} {'testB':>8} {'mktB':>8} {'beats?':>7}")
    for name, v in variants.items():
        p = walk_forward(df, sd, test_dates, v)
        bte = brier_score(p["fair"].values, p["settlement"].values)
        mkt = brier_score(p["d1_mid"].values, p["settlement"].values)
        flag = "YES <--" if bte < mkt else "no"
        star = " *SELECTED*" if name == best else ""
        print(f"  {name:<44} {bte:>8.4f} {mkt:>8.4f} {flag:>7}{star}")
        if name == best or v == "N0":
            lo, hi, p_win = bootstrap_dates(p)
            print(f"      bootstrap over {p['date'].nunique()} test dates: "
                  f"90% CI [{lo:+.4f}, {hi:+.4f}]   P(model better) {p_win:.2f}")
    print("=" * 84)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
