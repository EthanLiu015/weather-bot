"""
CLI: run full walk-forward backtest.

Usage: python scripts/run_backtest.py --start 2022-01-01 --end 2024-12-31
"""
import argparse
import logging
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--train-years", type=int, default=3, help="Training window in years")
    parser.add_argument("--out-csv", default="data/backtest_results.csv")
    parser.add_argument("--out-html", default="data/backtest_report.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    from config.settings import get_settings
    from db.session import init_db
    settings = get_settings()
    init_db(settings.DB_URL)

    from backtest.runner import BacktestRunner
    runner = BacktestRunner(
        settings=settings,
        start_date=start,
        end_date=end,
        train_window_years=args.train_years,
    )
    logger.info("Starting backtest %s → %s (train window: %dy)", start, end, args.train_years)
    report = runner.run()
    summary = report.summary()

    logger.info("=== BACKTEST SUMMARY ===")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)

    report.to_csv(args.out_csv)
    report.to_html(args.out_html)
    logger.info("Results saved: %s, %s", args.out_csv, args.out_html)


if __name__ == "__main__":
    main()
