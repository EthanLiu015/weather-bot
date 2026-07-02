"""Walk-forward ensemble refit (autoresearch cycle 2).

Cycle 1 found inverse-MSE model weights + EMOS closes most of the gap
(test Brier 0.0984 vs market 0.0951) but the model degrades from train to
test while the market doesn't — parameter drift. Here every test date d is
priced with parameters fit on ALL station-days strictly before d (no
look-ahead), so weights/bias/sigma adapt.

Variant selection happens on the TRAIN window (walk-forward within train,
first 10 dates as burn-in); the TEST window is scored once per variant and
reported. Test dates match research.ensemble_upgrade exactly.

    PYTHONPATH=. python -m research.ensemble_walkforward
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

from strategies.bracket_pricing import bracket_yes_prob
from backtest.real_market_eval import brier_score
from research.ensemble_upgrade import load_joined, temporal_split, FRESH_PATH, FRESH_LEAD

logger = logging.getLogger(__name__)

MODELS = ["aifs", "graphcast", "gfs", "icon", "ecmwf"]
MIN_STATION_DAYS = 12
BURN_IN_DATES = 10


def model_wide(mm_path: str = FRESH_PATH, lead_hour: int = FRESH_LEAD) -> pd.DataFrame:
    mm = pd.read_parquet(mm_path)
    d = mm[mm.lead_hour == lead_hour]
    wide = d.pivot_table(index=["station", "date"], columns="model", values="tmax_f").reset_index()
    wide["date"] = pd.to_datetime(wide["date"])
    return wide


def fit_params(hist: pd.DataFrame, shrink: float = 0.5) -> dict:
    """Fit weights / bias / EMOS on station-days strictly before the target date.

    hist: one row per (station, date) with model columns + actual_tmax.
    shrink: per-station weight shrinkage toward global (0 = global only).
    """
    cols = [m for m in MODELS if m in hist]
    global_w = {}
    for m in cols:
        err = (hist["actual_tmax"] - hist[m]).dropna()
        if len(err) >= 20:
            global_w[m] = 1.0 / max(float((err ** 2).mean()), 1e-6)
    tot = sum(global_w.values()) or 1.0
    global_w = {m: w / tot for m, w in global_w.items()}

    station_w = {}
    for st, g in hist.groupby("station"):
        if len(g) < MIN_STATION_DAYS:
            continue
        sw = {}
        for m in global_w:
            err = (g["actual_tmax"] - g[m]).dropna()
            if len(err) >= MIN_STATION_DAYS:
                sw[m] = 1.0 / max(float((err ** 2).mean()), 1e-6)
        if len(sw) == len(global_w):
            tot = sum(sw.values())
            sw = {m: w / tot for m, w in sw.items()}
            station_w[st] = {m: shrink * sw[m] + (1 - shrink) * global_w[m]
                             for m in global_w}

    def wmean(row, st) -> float:
        w = station_w.get(st, global_w)
        num = den = 0.0
        for m, wi in w.items():
            v = row.get(m, np.nan) if isinstance(row, dict) else getattr(row, m, np.nan)
            if not pd.isna(v):
                num += wi * float(v)
                den += wi
        return num / den if den > 0 else np.nan

    h = hist.copy()
    h["wmean"] = [wmean(r._asdict(), r.station) for r in h.itertuples(index=False)]
    h["resid"] = h["actual_tmax"] - h["wmean"]
    h["spread"] = h[cols].std(axis=1)

    gbias = float(h["resid"].mean())
    biases = {st: float(g["resid"].mean()) for st, g in h.groupby("station")
              if len(g) >= MIN_STATION_DAYS}
    centered = h["resid"].values - np.array(
        [biases.get(s, gbias) for s in h["station"]])
    spread2 = h["spread"].values ** 2
    ok = ~np.isnan(centered) & ~np.isnan(spread2)

    def nll(params):
        a, b = params
        var = np.maximum(a + b * spread2[ok], 0.05)
        return float(np.sum(0.5 * np.log(2 * np.pi * var) + centered[ok] ** 2 / (2 * var)))

    a, b = minimize(nll, x0=[1.0, 0.3], method="Nelder-Mead").x
    std_resid = centered[ok] / np.maximum(np.sqrt(a + b * spread2[ok]), 0.55)
    return {"weights": (global_w, station_w), "wmean": wmean, "gbias": gbias,
            "biases": biases, "emos": (float(a), float(b)),
            "gsigma": float(np.std(centered[ok])), "std_resid": std_resid}


def price_day(day_rows: pd.DataFrame, params: dict, dist: str) -> pd.DataFrame:
    a, b = params["emos"]
    out_fair = []
    keep = []
    sr = params["std_resid"]
    for i, row in enumerate(day_rows.itertuples(index=False)):
        mu = params["wmean"](row._asdict(), row.station)
        if np.isnan(mu):
            continue
        loc = mu + params["biases"].get(row.station, params["gbias"])
        spread = getattr(row, "ens_std", np.nan)
        if dist == "emos":
            s = max(np.sqrt(a + b * spread ** 2), 0.55) if not np.isnan(spread) else params["gsigma"]
            sf = lambda x, loc=loc, s=s: float(norm.sf(x, loc=loc, scale=s))
        elif dist == "gauss":
            s = params["gsigma"]
            sf = lambda x, loc=loc, s=s: float(norm.sf(x, loc=loc, scale=s))
        else:  # standardized empirical ECDF scaled by EMOS sigma
            s = max(np.sqrt(a + b * spread ** 2), 0.55) if not np.isnan(spread) else params["gsigma"]
            sf = lambda x, loc=loc, s=s: float(np.mean(norm.sf((x - loc - s * sr) / 0.6)))
        p = bracket_yes_prob(sf, row.strike_type, row.floor_strike, row.cap_strike)
        if p is None:
            continue
        out_fair.append(float(p))
        keep.append(i)
    out = day_rows.iloc[keep].copy()
    out["fair"] = out_fair
    return out


def walk_forward(df: pd.DataFrame, wide: pd.DataFrame, eval_dates: np.ndarray,
                 dist: str, shrink: float) -> pd.DataFrame:
    sd_all = df.drop_duplicates(["station", "date"])[
        ["station", "date", "actual_tmax", "ens_std"]].merge(
        wide, on=["station", "date"], how="left")
    priced = []
    for d in eval_dates:
        hist = sd_all[sd_all["date"] < d]
        if hist["date"].nunique() < BURN_IN_DATES:
            continue
        params = fit_params(hist, shrink=shrink)
        day = df[df["date"] == d].merge(
            wide, on=["station", "date"], how="left")
        priced.append(price_day(day, params, dist))
    return pd.concat(priced, ignore_index=True) if priced else pd.DataFrame()


def bootstrap_dates(priced: pd.DataFrame, n_boot: int = 2000, seed: int = 7):
    """Block-bootstrap Brier difference (market - model) over settlement dates."""
    rng = np.random.default_rng(seed)
    per_date = {d: g for d, g in priced.groupby("date")}
    dates = list(per_date.keys())
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(dates, size=len(dates), replace=True)
        g = pd.concat([per_date[d] for d in pick])
        diffs.append(brier_score(g["d1_mid"].values, g["settlement"].values)
                     - brier_score(g["fair"].values, g["settlement"].values))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95)), float((diffs > 0).mean())


def run(mm_path: str = FRESH_PATH, lead_hour: int = FRESH_LEAD) -> None:
    df = load_joined(mm_path=mm_path, lead_hour=lead_hour)
    wide = model_wide(mm_path=mm_path, lead_hour=lead_hour)
    train, test = temporal_split(df)
    train_dates = np.sort(train["date"].unique())
    test_dates = np.sort(test["date"].unique())

    variants = {
        "W1 walkfwd invMSE + EMOS (global w)": ("emos", 0.0),
        "W2 walkfwd invMSE + EMOS (shrunk station w)": ("emos", 0.5),
        "W3 walkfwd invMSE + gauss": ("gauss", 0.0),
        "W4 walkfwd invMSE + std-empirical": ("emp", 0.0),
    }

    print("\n" + "=" * 82)
    print("WALK-FORWARD ENSEMBLE (cycle 2) — params refit before every date, no look-ahead")
    print("=" * 82)
    sel = {}
    print(f"  {'variant':<46} {'selB(train-wf)':>14}")
    for name, (dist, shrink) in variants.items():
        p = walk_forward(df, wide, train_dates, dist, shrink)
        b = brier_score(p["fair"].values, p["settlement"].values) if len(p) else np.nan
        sel[name] = b
        print(f"  {name:<46} {b:>14.4f}")
    best = min(sel, key=lambda k: sel[k])
    print(f"\n  selected on train walk-forward: {best}")

    print(f"\n  {'variant':<46} {'testB':>8} {'mktB':>8} {'beats?':>7}")
    for name, (dist, shrink) in variants.items():
        p = walk_forward(df, wide, test_dates, dist, shrink)
        bte = brier_score(p["fair"].values, p["settlement"].values)
        mkt = brier_score(p["d1_mid"].values, p["settlement"].values)
        flag = "YES <--" if bte < mkt else "no"
        star = " *SELECTED*" if name == best else ""
        print(f"  {name:<46} {bte:>8.4f} {mkt:>8.4f} {flag:>7}{star}")
        if name == best:
            lo, hi, p_win = bootstrap_dates(p)
            n_dates = p["date"].nunique()
            print(f"      bootstrap (mktB - modelB) over {n_dates} test dates: "
                  f"90% CI [{lo:+.4f}, {hi:+.4f}]   P(model better) {p_win:.2f}")
    print("=" * 82)


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale24", action="store_true",
                    help="use the 24h-lead Previous Runs data (strictly staler than the book)")
    args = ap.parse_args()
    if args.stale24:
        run(mm_path="data/historical/openmeteo_multimodel.parquet", lead_hour=24)
    else:
        run()


if __name__ == "__main__":
    main()
