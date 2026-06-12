import json

import numpy as np
import pytest

from processing.climatology import climo_tmax_normal


def test_climo_tmax_normal_returns_value_from_table(tmp_path):
    path = tmp_path / "tmax_normals.json"
    path.write_text(json.dumps({"KORD": {"6": 82.5}}))

    assert climo_tmax_normal("KORD", 6, path=path) == pytest.approx(82.5)


def test_climo_tmax_normal_returns_nan_for_unknown_station(tmp_path):
    path = tmp_path / "tmax_normals.json"
    path.write_text(json.dumps({"KORD": {"6": 82.5}}))

    assert np.isnan(climo_tmax_normal("KXXX", 6, path=path))


def test_climo_tmax_normal_returns_nan_when_file_missing(tmp_path):
    path = tmp_path / "does_not_exist.json"

    assert np.isnan(climo_tmax_normal("KORD", 6, path=path))
