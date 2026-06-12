"""Tests for pruning old GEFS/NBM grib cache run directories.

data/gefs and data/nbm accumulate one ~3-4GB (gefs) or ~0.5GB (nbm) directory
per run (data/<base>/<date>/<cycle>/...) and are never cleaned up, eventually
filling the disk and causing download failures mid-cycle. These tests cover
a pure function that decides which run directories are stale, and a
filesystem helper that deletes them while keeping the most recent runs.
"""
from pathlib import Path

from ingestion.cache_pruning import stale_run_dirs, prune_run_cache


# ── stale_run_dirs (pure) ────────────────────────────────────────────────────

def test_stale_run_dirs_returns_all_but_most_recent_n():
    run_ids = ["20260605/12", "20260608/12", "20260610/12", "20260610/18", "20260611/00"]

    stale = stale_run_dirs(run_ids, keep=2)

    assert stale == ["20260605/12", "20260608/12", "20260610/12"]


def test_stale_run_dirs_empty_when_fewer_than_keep():
    run_ids = ["20260610/18", "20260611/00"]

    assert stale_run_dirs(run_ids, keep=2) == []


def test_stale_run_dirs_handles_empty_input():
    assert stale_run_dirs([], keep=2) == []


def test_stale_run_dirs_sorts_unordered_input():
    run_ids = ["20260611/00", "20260605/12", "20260610/18"]

    stale = stale_run_dirs(run_ids, keep=1)

    assert stale == ["20260605/12", "20260610/18"]


# ── prune_run_cache (filesystem) ─────────────────────────────────────────────

def _make_run_dir(base: Path, date_str: str, cycle: str, *files: str) -> Path:
    run_dir = base / date_str / cycle
    run_dir.mkdir(parents=True)
    for name in files:
        (run_dir / name).write_bytes(b"data")
    return run_dir


def test_prune_run_cache_deletes_old_runs_and_keeps_recent(tmp_path):
    _make_run_dir(tmp_path, "20260605", "12", "gep01_f024.grib2")
    _make_run_dir(tmp_path, "20260610", "12", "gep01_f024.grib2")
    kept = _make_run_dir(tmp_path, "20260610", "18", "gep01_f024.grib2")

    prune_run_cache(tmp_path, keep=2)

    remaining = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.glob("*/*"))
    assert remaining == ["20260610/12", "20260610/18"]
    assert (kept / "gep01_f024.grib2").exists()


def test_prune_run_cache_removes_now_empty_date_dir(tmp_path):
    _make_run_dir(tmp_path, "20260605", "12", "gep01_f024.grib2")
    _make_run_dir(tmp_path, "20260610", "12", "gep01_f024.grib2")
    _make_run_dir(tmp_path, "20260610", "18", "gep01_f024.grib2")

    prune_run_cache(tmp_path, keep=2)

    assert not (tmp_path / "20260605").exists()


def test_prune_run_cache_returns_removed_run_ids(tmp_path):
    _make_run_dir(tmp_path, "20260605", "12", "gep01_f024.grib2")
    _make_run_dir(tmp_path, "20260610", "12", "gep01_f024.grib2")
    _make_run_dir(tmp_path, "20260610", "18", "gep01_f024.grib2")

    removed = prune_run_cache(tmp_path, keep=2)

    assert removed == ["20260605/12"]


def test_prune_run_cache_noop_when_base_dir_missing(tmp_path):
    missing = tmp_path / "does_not_exist"

    assert prune_run_cache(missing, keep=2) == []


def test_prune_run_cache_noop_when_within_keep_limit(tmp_path):
    _make_run_dir(tmp_path, "20260610", "12", "gep01_f024.grib2")
    _make_run_dir(tmp_path, "20260610", "18", "gep01_f024.grib2")

    removed = prune_run_cache(tmp_path, keep=2)

    assert removed == []
    assert (tmp_path / "20260610" / "12").exists()
