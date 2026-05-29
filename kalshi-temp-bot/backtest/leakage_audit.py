import logging
from datetime import date
import pandas as pd

logger = logging.getLogger(__name__)


def audit_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "date",
    train_end: date = None,
) -> tuple[bool, list[str]]:
    issues: list[str] = []

    if date_col in train_df.columns and date_col in test_df.columns:
        train_dates = set(pd.to_datetime(train_df[date_col]).dt.date)
        test_dates = set(pd.to_datetime(test_df[date_col]).dt.date)
        overlap = train_dates & test_dates
        if overlap:
            issues.append(f"Date overlap between train and test: {sorted(overlap)[:5]} ...")

    if train_end is not None and date_col in train_df.columns:
        future_rows = train_df[pd.to_datetime(train_df[date_col]).dt.date > train_end]
        if not future_rows.empty:
            issues.append(f"Training data contains {len(future_rows)} rows after train_end {train_end}")

    if issues:
        for issue in issues:
            logger.error("LEAKAGE AUDIT FAILED: %s", issue)
        return False, issues

    logger.info("Leakage audit passed")
    return True, []
