"""Market-price shrinkage blend (autoresearch cycle 6) — approved fallback.

p_blend = w * p_model + (1 - w) * d1_mid, per bracket. If the model carries ANY
information orthogonal to the book, the train-optimal w is > 0 and the blend
out-Briers the market on test. Circular for trading (it consumes the market
price) but answers the research question: does our fair-information model add
anything at all on top of the 14:00 UTC price?

Model distribution = cycle-5 M1 (NBM12Z + wf bias + EMOS, truncated at morning
runmax) — the best fair distribution that uses every legal source. w has a
closed-form optimum (Brier is quadratic in w) and is fit walk-forward: w for
test date d comes from priced days strictly before d (train days + earlier test
days), so no test information leaks into its own weight.

    PYTHONPATH=. python -m research.blend_fallback
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backtest.real_market_eval import brier_score
from research.ensemble_upgrade import temporal_split
from research.ensemble_walkforward import bootstrap_dates
from research.obs_conditioning import load_all, station_day_table, walk_forward

logger = logging.getLogger(__name__)

MODEL_VARIANT = "M1"


def optimal_w(fair: np.ndarray, mkt: np.ndarray, y: np.ndarray) -> float:
    """argmin_w Brier(w*fair + (1-w)*mkt): w* = <fair-mkt, y-mkt> / |fair-mkt|^2,
    clipped to [0, 1]."""
    d = fair - mkt
    denom = float(np.dot(d, d))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(np.dot(d, y - mkt) / denom, 0.0, 1.0))


def walk_forward_blend(priced: pd.DataFrame, warmup_days: int = 10) -> pd.DataFrame:
    """Per-date blend weight from all priced days strictly before that date."""
    priced = priced.sort_values("date").reset_index(drop=True)
    dates = np.sort(priced["date"].unique())
    out = []
    for d in dates[warmup_days:]:
        hist = priced[priced["date"] < d]
        w = optimal_w(hist["fair"].values, hist["d1_mid"].values,
                      hist["settlement"].values)
        day = priced[priced["date"] == d].copy()
        day["blend"] = w * day["fair"] + (1 - w) * day["d1_mid"]
        day["w"] = w
        out.append(day)
    return pd.concat(out, ignore_index=True)


def run() -> None:
    df = load_all()
    sd = station_day_table(df)
    train, test = temporal_split(df)
    all_dates = np.sort(df["date"].unique())
    test_dates = np.sort(test["date"].unique())

    priced = walk_forward(df, sd, all_dates, MODEL_VARIANT)
    logger.info("priced %d markets over %d dates", len(priced), priced["date"].nunique())

    blended = walk_forward_blend(priced)
    bl_test = blended[blended["date"].isin(test_dates)]

    print("\n" + "=" * 84)
    print(f"MARKET-SHRINKAGE BLEND (cycle 6) — model = cycle-5 {MODEL_VARIANT}, wf weight")
    print("=" * 84)
    w_train = optimal_w(*(priced[priced['date'] < test_dates[0]][c].values
                          for c in ("fair", "d1_mid", "settlement")))
    print(f"  train-window optimal w (model share): {w_train:.3f}")
    print(f"  wf w range on test dates: [{bl_test['w'].min():.3f}, {bl_test['w'].max():.3f}]")

    b_model = brier_score(bl_test["fair"].values, bl_test["settlement"].values)
    b_mkt = brier_score(bl_test["d1_mid"].values, bl_test["settlement"].values)
    b_blend = brier_score(bl_test["blend"].values, bl_test["settlement"].values)
    print(f"\n  {'source':<24} {'testB':>8}")
    print(f"  {'model (fair only)':<24} {b_model:>8.4f}")
    print(f"  {'market d1_mid':<24} {b_mkt:>8.4f}")
    print(f"  {'blend':<24} {b_blend:>8.4f}   beats market: {'YES <--' if b_blend < b_mkt else 'no'}")

    boot = bl_test.rename(columns={"fair": "fair_model"}).rename(columns={"blend": "fair"})
    lo, hi, p_win = bootstrap_dates(boot)
    print(f"\n  bootstrap over {bl_test['date'].nunique()} test dates (blend vs market): "
          f"90% CI [{lo:+.4f}, {hi:+.4f}]   P(blend better) {p_win:.2f}")
    print("=" * 84)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
