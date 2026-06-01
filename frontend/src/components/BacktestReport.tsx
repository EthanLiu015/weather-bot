import { useEffect, useState } from 'react'
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'
import type { BacktestReport as BacktestReportType, BacktestFold } from '../types'

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className="text-xl font-bold text-white">{value}</div>
      {sub && <div className="text-xs text-gray-400 mt-1">{sub}</div>}
    </div>
  )
}

function slopeColor(slope: number | null) {
  if (slope === null) return 'text-gray-500'
  if (slope >= 0.9 && slope <= 1.1) return 'text-green-400'
  if (slope >= 0.75 && slope <= 1.25) return 'text-yellow-400'
  return 'text-red-400'
}

export function BacktestReport() {
  const [report, setReport] = useState<BacktestReportType | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/backtest/results')
      .then(r => r.json())
      .then(data => { setReport(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-gray-500">Loading backtest results...</div>
  )

  if (!report || report.folds.length === 0) return (
    <div className="flex items-center justify-center h-64 text-gray-500">
      No backtest results found. Run <code className="mx-1 bg-[#2a2a2a] px-1 rounded">python scripts/initial_train.py</code> first.
    </div>
  )

  const { summary, folds } = report
  const validFolds = folds.filter(f => f.crps !== null)

  // Cumulative PnL series for chart
  let cumPnl = 0
  const pnlData = validFolds.map(f => {
    cumPnl += f.simulated_pnl_usd ?? 0
    return {
      month: f.fold_month.slice(0, 7),
      daily_pnl: f.simulated_pnl_usd ?? 0,
      cumulative_pnl: cumPnl,
      crps: f.crps,
      mae: f.mae,
      reliability_slope: f.reliability_slope,
    }
  })

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-white text-xl font-bold">Backtest Report</h2>
      <p className="text-gray-400 text-xs">Walk-forward cross-validation · NGBoost · {summary.num_folds} monthly folds</p>

      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Mean CRPS (lower = better)"
          value={summary.mean_crps?.toFixed(3) ?? '—'}
          sub="Continuous Ranked Probability Score"
        />
        <StatCard
          label="Mean MAE"
          value={`${summary.mean_mae?.toFixed(2) ?? '—'}°F`}
          sub="Mean Absolute Error vs actual Tmax"
        />
        <StatCard
          label="Mean Reliability Slope"
          value={summary.mean_reliability_slope?.toFixed(3) ?? '—'}
          sub="1.0 = perfectly calibrated"
        />
        <StatCard
          label="Total Simulated PnL"
          value={`$${summary.total_simulated_pnl?.toFixed(2) ?? '—'}`}
          sub={`${summary.total_trades} simulated trades`}
        />
      </div>

      {/* Cumulative PnL chart */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-white font-semibold mb-3 text-sm">Simulated Cumulative PnL by Fold</h3>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={pnlData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
            <XAxis dataKey="month" stroke="#555" tick={{ fill: '#666', fontSize: 9 }}
              interval={Math.floor(pnlData.length / 8)} />
            <YAxis stroke="#555" tick={{ fill: '#888', fontSize: 10 }}
              tickFormatter={v => `$${v.toFixed(0)}`} />
            <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', color: '#eee', fontSize: 11 }}
              formatter={(v: number, n: string) => [`$${v.toFixed(2)}`, n]} />
            <Legend wrapperStyle={{ color: '#888', fontSize: 11 }} />
            <ReferenceLine y={0} stroke="#444" />
            <Bar dataKey="daily_pnl" fill="#334155" name="Fold PnL" />
            <Line type="monotone" dataKey="cumulative_pnl" stroke="#22d3ee"
              dot={false} strokeWidth={2} name="Cumulative PnL" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* CRPS over time */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-white font-semibold mb-3 text-sm">Forecast Skill (CRPS) by Fold</h3>
        <ResponsiveContainer width="100%" height={160}>
          <ComposedChart data={pnlData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
            <XAxis dataKey="month" stroke="#555" tick={{ fill: '#666', fontSize: 9 }}
              interval={Math.floor(pnlData.length / 8)} />
            <YAxis stroke="#555" tick={{ fill: '#888', fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', color: '#eee', fontSize: 11 }} />
            <Line type="monotone" dataKey="crps" stroke="#f59e0b"
              dot={false} strokeWidth={1.5} name="CRPS" />
            <Line type="monotone" dataKey="mae" stroke="#a78bfa"
              dot={false} strokeWidth={1.5} name="MAE (°F)" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Fold table */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg overflow-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[#2a2a2a] text-gray-500 uppercase tracking-wide">
              <th className="px-3 py-2 text-left">Fold</th>
              <th className="px-3 py-2 text-right">CRPS</th>
              <th className="px-3 py-2 text-right">MAE (°F)</th>
              <th className="px-3 py-2 text-right">Brier</th>
              <th className="px-3 py-2 text-right">Rel. Slope</th>
              <th className="px-3 py-2 text-right">Sim. PnL</th>
              <th className="px-3 py-2 text-right">Trades</th>
            </tr>
          </thead>
          <tbody>
            {validFolds.map((fold: BacktestFold) => (
              <tr key={fold.fold_month} className="border-b border-[#2a2a2a] hover:bg-[#222]">
                <td className="px-3 py-1.5 text-gray-300">{fold.fold_month}</td>
                <td className="px-3 py-1.5 text-right text-white">{fold.crps?.toFixed(3)}</td>
                <td className="px-3 py-1.5 text-right text-white">{fold.mae?.toFixed(2)}</td>
                <td className="px-3 py-1.5 text-right text-white">{fold.brier_score?.toFixed(4)}</td>
                <td className={`px-3 py-1.5 text-right font-semibold ${slopeColor(fold.reliability_slope)}`}>
                  {fold.reliability_slope?.toFixed(3)}
                </td>
                <td className={`px-3 py-1.5 text-right font-semibold ${(fold.simulated_pnl_usd ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${fold.simulated_pnl_usd?.toFixed(2)}
                </td>
                <td className="px-3 py-1.5 text-right text-gray-400">{fold.num_simulated_trades}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
