import shutil
from pathlib import Path


def stale_run_dirs(run_ids: list[str], keep: int = 2) -> list[str]:
    """Run ids (e.g. "20260610/18") for runs older than the most recent `keep`."""
    sorted_ids = sorted(run_ids)
    if len(sorted_ids) <= keep:
        return []
    return sorted_ids[: len(sorted_ids) - keep]


def prune_run_cache(base_dir: Path | str, keep: int = 2) -> list[str]:
    """Delete grib cache run directories under `base_dir` older than the most
    recent `keep`, returning the removed run ids ("<date>/<cycle>")."""
    base = Path(base_dir)
    if not base.is_dir():
        return []

    run_dirs = {
        f"{date_dir.name}/{cycle_dir.name}": cycle_dir
        for date_dir in base.iterdir()
        if date_dir.is_dir()
        for cycle_dir in date_dir.iterdir()
        if cycle_dir.is_dir()
    }

    removed = stale_run_dirs(list(run_dirs), keep=keep)
    for run_id in removed:
        shutil.rmtree(run_dirs[run_id])

    for date_dir in base.iterdir():
        if date_dir.is_dir() and not any(date_dir.iterdir()):
            date_dir.rmdir()

    return removed
