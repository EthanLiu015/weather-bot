"""Recompute final per-station models and the global blend-weight artifact
without re-running the ~1.5hr walk-forward backtest (Step 2 of initial_train.py).

Usage: PYTHONPATH=. python scripts/recompute_blend_weights.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from db.session import init_db
from scripts.initial_train import load_feature_data, train_final_models


def main() -> None:
    settings = get_settings()
    init_db(settings.DB_URL)

    df = load_feature_data()
    if df.empty:
        return

    train_final_models(df)


if __name__ == "__main__":
    main()
