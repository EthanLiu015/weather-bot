import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import type { DailyPnL } from '../types'

type Props = {
  series: DailyPnL[]
}

export function PnLChart({ series }: Props) {
  const data = series.map((d) => ({
    date: d.date.slice(5),
    daily: parseFloat(d.daily_pnl.toFixed(2)),
    cumulative: parseFloat(d.cumulative_pnl.toFixed(2)),
  }))

  if (data.length === 0) {
    return (
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4 flex items-center justify-center h-48">
        <p className="text-gray-500 text-sm">No PnL data yet</p>
      </div>
    )
  }

  return (
    <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
      <h3 className="text-white font-semibold mb-3">Cumulative PnL</h3>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
          <XAxis dataKey="date" stroke="#555" tick={{ fill: '#888', fontSize: 10 }} />
          <YAxis stroke="#555" tick={{ fill: '#888', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', color: '#eee' }}
          />
          <Legend wrapperStyle={{ color: '#888', fontSize: 12 }} />
          <Bar dataKey="daily" fill="#334155" name="Daily PnL" />
          <Line
            type="monotone"
            dataKey="cumulative"
            stroke="#22d3ee"
            dot={false}
            strokeWidth={2}
            name="Cumulative PnL"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
