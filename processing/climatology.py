import json
from functools import lru_cache
from pathlib import Path

CLIMATOLOGY_PATH = Path("data/climatology/tmax_normals.json")


@lru_cache(maxsize=4)
def _load_normals(path_str: str) -> dict[str, dict[str, float]]:
    path = Path(path_str)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def climo_tmax_normal(station: str, month: int, path: Path = CLIMATOLOGY_PATH) -> float:
    """Climatological mean Tmax (°F) for a station/month, or NaN if unknown."""
    normals = _load_normals(str(path))
    return float(normals.get(station, {}).get(str(month), float("nan")))
