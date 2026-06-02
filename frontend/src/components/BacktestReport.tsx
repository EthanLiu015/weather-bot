import { useEffect, useState } from 'react'
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'
import type { BacktestReport as BacktestReportType, BacktestFold } from '../types'

function StatCard({ label, value, sub, highlight }: {
  label: string; value: string; sub?: string; highlight?: 'green' | 'red' | 'blue' | null
}) {
  const color = highlight === 'green' ? 'text-green-400'
    : highlight === 'red' ? 'text-red-400'
    : highlight === 'blue' ? 'text-cyan-400'
    : 'text-white'
  return (
    <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-xl font-bold ${color}`}>{value}</div>
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
      No backtest results. Run <code className="mx-1 bg-[#2a2a2a] px-1 rounded">PYTHONPATH=. python scripts/initial_train.py</code>
    </div>
  )

  const { summary, folds } = report
  const hasReal = summary.total_real_price_trades > 0

  // Build cumulative chart data
  let cumTotal = 0, cumReal = 0, cumClim = 0
  const pnlData = folds.map(f => {
    cumTotal += f.simulated_pnl_usd ?? 0
    cumReal  += f.real_price_pnl ?? 0
    cumClim  += f.clim_price_pnl ?? 0
    return {
      month:        f.fold_month.slice(0, 7),
      fold_pnl:     f.simulated_pnl_usd ?? 0,
      real_pnl:     f.real_price_pnl ?? 0,
      clim_pnl:     f.clim_price_pnl ?? 0,
      cum_total:    cumTotal,
      cum_real:     cumReal,
      cum_clim:     cumClim,
      crps:         f.crps,
      mae:          f.mae,
      has_real:     (f.real_price_trades ?? 0) > 0,
    }
  })

  const validFolds = folds.filter(f => f.crps !== null)

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-white text-xl font-bold">Backtest Report</h2>
          <p className="text-gray-400 text-xs mt-1">
            Walk-forward CV · NGBoost · {summary.num_folds} monthly folds ·{' '}
            <span className="text-cyan-400">{summary.total_real_price_trades} real Kalshi trades</span>
            {' '}·{' '}
            <span className="text-purple-400">{summary.total_clim_price_trades} climatological trades</span>
          </p>
        </div>
        <div className="flex gap-3 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-3 h-1 bg-cyan-400 inline-block rounded" /> Real Kalshi prices
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-1 bg-purple-400 inline-block rounded" /> Climatological baseline
          </span>
        </div>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          label="Mean CRPS"
          value={summary.mean_crps?.toFixed(3) ?? '—'}
          sub="Lower is better"
        />
        <StatCard
          label="Mean MAE"
          value={`${summary.mean_mae?.toFixed(2) ?? '—'}°F`}
          sub="vs actual Tmax"
        />
        <StatCard
          label="Mean Reliability Slope"
          value={summary.mean_reliability_slope?.toFixed(3) ?? '—'}
          sub="1.0 = perfectly calibrated"
        />
        <StatCard
          label="Total Simulated PnL"
          value={`$${summary.total_simulated_pnl?.toFixed(2) ?? '—'}`}
          sub={`${summary.total_trades} trades`}
          highlight={summary.total_simulated_pnl > 0 ? 'green' : 'red'}
        />
      </div>

      {/* Real vs Clim PnL comparison */}
      {hasReal && (
        <div className="grid grid-cols-2 gap-4">
          <StatCard
            label="Real Kalshi Prices — PnL"
            value={`$${summary.total_real_price_pnl?.toFixed(2)}`}
            sub={`${summary.total_real_price_trades} trades (Mar–May 2026)`}
            highlight={summary.total_real_price_pnl > 0 ? 'green' : 'red'}
          />
          <StatCard
            label="Climatological Baseline — PnL"
            value={`$${summary.total_clim_price_pnl?.toFixed(2)}`}
            sub={`${summary.total_clim_price_trades} trades (historical folds)`}
            highlight="blue"
          />
        </div>
      )}

      {/* Cumulative PnL — split by source */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-white font-semibold mb-1 text-sm">Cumulative Simulated PnL</h3>
        {hasReal && (
          <p className="text-xs text-gray-500 mb-3">
            Cyan = total · Teal = real Kalshi prices · Purple = climatological baseline
          </p>
        )}
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={pnlData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
            <XAxis dataKey="month" stroke="#555" tick={{ fill: '#666', fontSize: 9 }}
              interval={Math.floor(pnlData.length / 8)} />
            <YAxis stroke="#555" tick={{ fill: '#888', fontSize: 10 }}
              tickFormatter={v => `$${v.toFixed(0)}`} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', color: '#eee', fontSize: 11 }}
              formatter={(v: number, n: string) => [`$${v.toFixed(2)}`, n]}
            />
            <ReferenceLine y={0} stroke="#444" />
            <Bar dataKey="fold_pnl" fill="#1e3a5f" name="Fold PnL" opacity={0.6} />
            <Line type="monotone" dataKey="cum_total" stroke="#22d3ee"
              dot={false} strokeWidth={2} name="Cumulative (total)" />
            {hasReal && (
              <Line type="monotone" dataKey="cum_real" stroke="#14b8a6"
                dot={false} strokeWidth={1.5} strokeDasharray="4 2" name="Real Kalshi prices" />
            )}
            {hasReal && (
              <Line type="monotone" dataKey="cum_clim" stroke="#a855f7"
                dot={false} strokeWidth={1.5} strokeDasharray="4 2" name="Climatological baseline" />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* CRPS & MAE over time */}
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-white font-semibold mb-3 text-sm">Forecast Skill by Fold</h3>
        <ResponsiveContainer width="100%" height={160}>
          <ComposedChart data={pnlData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
            <XAxis dataKey="month" stroke="#555" tick={{ fill: '#666', fontSize: 9 }}
              interval={Math.floor(pnlData.length / 8)} />
            <YAxis stroke="#555" tick={{ fill: '#888', fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', color: '#eee', fontSize: 11 }} />
            <Legend wrapperStyle={{ color: '#888', fontSize: 11 }} />
            <Line type="monotone" dataKey="crps" stroke="#f59e0b" dot={false} strokeWidth={1.5} name="CRPS" />
            <Line type="monotone" dataKey="mae" stroke="#a78bfa" dot={false} strokeWidth={1.5} name="MAE (°F)" />
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
              <th className="px-3 py-2 text-right">MAE °F</th>
              <th className="px-3 py-2 text-right">Rel.Slope</th>
              <th className="px-3 py-2 text-right">Total PnL</th>
              <th className="px-3 py-2 text-right text-cyan-600">Real PnL</th>
              <th className="px-3 py-2 text-right text-cyan-600">Real Trades</th>
              <th className="px-3 py-2 text-right text-purple-600">Clim PnL</th>
              <th className="px-3 py-2 text-right text-purple-600">Clim Trades</th>
            </tr>
          </thead>
          <tbody>
            {folds.map((fold: BacktestFold) => (
              <tr key={fold.fold_month}
                className={`border-b border-[#2a2a2a] hover:bg-[#222] ${fold.real_price_trades > 0 ? 'border-l-2 border-l-cyan-800' : ''}`}>
                <td className="px-3 py-1.5 text-gray-300">{fold.fold_month}</td>
                <td className="px-3 py-1.5 text-right text-white">
                  {fold.crps != null ? fold.crps.toFixed(3) : '—'}
                </td>
                <td className="px-3 py-1.5 text-right text-white">
                  {fold.mae != null ? fold.mae.toFixed(2) : '—'}
                </td>
                <td className={`px-3 py-1.5 text-right font-semibold ${slopeColor(fold.reliability_slope)}`}>
                  {fold.reliability_slope?.toFixed(3) ?? '—'}
                </td>
                <td className={`px-3 py-1.5 text-right font-semibold ${(fold.simulated_pnl_usd ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${(fold.simulated_pnl_usd ?? 0).toFixed(2)}
                </td>
                <td className={`px-3 py-1.5 text-right ${fold.real_price_pnl >= 0 ? 'text-teal-400' : 'text-red-400'}`}>
                  {fold.real_price_trades > 0 ? `$${fold.real_price_pnl.toFixed(2)}` : '—'}
                </td>
                <td className="px-3 py-1.5 text-right text-cyan-600">
                  {fold.real_price_trades > 0 ? fold.real_price_trades : '—'}
                </td>
                <td className={`px-3 py-1.5 text-right ${fold.clim_price_pnl >= 0 ? 'text-purple-400' : 'text-red-400'}`}>
                  ${fold.clim_price_pnl.toFixed(2)}
                </td>
                <td className="px-3 py-1.5 text-right text-purple-600">
                  {fold.clim_price_trades}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
