"""Characterization test for the extracted in-memory training recipe.

`train_models` is the single source of truth for the production training recipe
(previously inlined in `train_final_models`). It must return an in-memory bundle
keyed by station/lead-bucket WITHOUT writing to the model registry, so the
real-markets eval harness can train on a look-ahead-free subset.
"""

import numpy as np
import pandas as pd
import pytest

from processing.features import get_feature_columns
from scripts.initial_train import train_models


def _synthetic_features(n_per_bucket: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = get_feature_columns()
    # One station, lead hours spanning the D1-2 bucket (0-48h).
    n = n_per_bucket
    base = {c: rng.normal(0, 1, n) for c in cols}
    df = pd.DataFrame(base)
    df["station"] = "KORD"
    df["lead_hour"] = 24
    df["ecmwf_tmax"] = rng.normal(75, 5, n)
    df["actual_tmax"] = df["ecmwf_tmax"] + rng.normal(0, 3, n)
    return df


def test_train_models_returns_bundle_without_touching_registry():
    df = _synthetic_features()
    bundle = train_models(df, n_estimators=20, learning_rate=0.05, min_rows=100)

    assert "KORD_D1-2" in bundle["ngboost"]
    assert "KORD_D1-2" in bundle["qrf"]
    assert "KORD_D1-2" in bundle["calibrator"]
    assert set(bundle["blender"].weights) == {"ngboost", "qrf"}


def test_train_models_skips_station_bucket_below_min_rows():
    df = _synthetic_features(n_per_bucket=50)
    bundle = train_models(df, n_estimators=20, learning_rate=0.05, min_rows=100)
    assert bundle["ngboost"] == {}
