import csv
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold_month: date
    crps: float
    mae: float
    brier_score: float
    reliability_slope: float
    simulated_pnl_usd: float
    num_simulated_trades: int
    edge_above_threshold_pct: float


@dataclass
class BacktestReport:
    folds: list[FoldResult] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.folds:
            return {}
        import numpy as np
        return {
            "num_folds": len(self.folds),
            "mean_crps": float(np.mean([f.crps for f in self.folds])),
            "mean_mae": float(np.mean([f.mae for f in self.folds])),
            "mean_brier": float(np.mean([f.brier_score for f in self.folds])),
            "mean_reliability_slope": float(np.mean([f.reliability_slope for f in self.folds if f.reliability_slope == f.reliability_slope])),
            "total_simulated_pnl": float(sum(f.simulated_pnl_usd for f in self.folds)),
            "total_trades": sum(f.num_simulated_trades for f in self.folds),
            "mean_edge_pct": float(np.mean([f.edge_above_threshold_pct for f in self.folds])),
        }

    def to_csv(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "fold_month", "crps", "mae", "brier_score",
                "reliability_slope", "simulated_pnl_usd",
                "num_simulated_trades", "edge_above_threshold_pct",
            ])
            writer.writeheader()
            for fold in self.folds:
                writer.writerow({
                    "fold_month": fold.fold_month.isoformat(),
                    "crps": fold.crps,
                    "mae": fold.mae,
                    "brier_score": fold.brier_score,
                    "reliability_slope": fold.reliability_slope,
                    "simulated_pnl_usd": fold.simulated_pnl_usd,
                    "num_simulated_trades": fold.num_simulated_trades,
                    "edge_above_threshold_pct": fold.edge_above_threshold_pct,
                })
        logger.info("Backtest CSV saved to %s", path)

    def to_html(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        rows_html = "\n".join(
            f"<tr>"
            f"<td>{f.fold_month}</td>"
            f"<td>{f.crps:.4f}</td>"
            f"<td>{f.mae:.2f}</td>"
            f"<td>{f.brier_score:.4f}</td>"
            f"<td>{f.reliability_slope:.3f}</td>"
            f"<td>${f.simulated_pnl_usd:.2f}</td>"
            f"<td>{f.num_simulated_trades}</td>"
            f"<td>{f.edge_above_threshold_pct:.1%}</td>"
            f"</tr>"
            for f in self.folds
        )
        html = f"""<!DOCTYPE html>
<html>
<head><title>Backtest Report</title>
<style>body{{font-family:monospace;background:#111;color:#eee}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #333;padding:6px;text-align:right}}
th{{background:#222}}</style></head>
<body>
<h1>Backtest Report</h1>
<h2>Summary</h2>
<pre>{summary}</pre>
<h2>Fold Results</h2>
<table>
<tr><th>Month</th><th>CRPS</th><th>MAE</th><th>Brier</th>
<th>Rel.Slope</th><th>Sim.PnL</th><th>Trades</th><th>Edge%</th></tr>
{rows_html}
</table>
</body></html>"""
        with open(path, "w") as f:
            f.write(html)
        logger.info("Backtest HTML report saved to %s", path)
