import { useEffect, useRef, useState } from 'react'
import { LineChart, Line, ResponsiveContainer } from 'recharts'
import type { MarketState } from '../types'

type Props = {
  market: MarketState
}

function edgeColor(edge: number | null, locked: boolean): string {
  if (locked) return 'text-red-400'
  if (edge === null) return 'text-gray-500'
  if (edge >= 0.04) return 'text-green-400'
  if (edge >= 0.02) return 'text-yellow-400'
  return 'text-gray-500'
}

export function MarketRow({ market }: Props) {
  const [history, setHistory] = useState<{ v: number }[]>([])
  const [flash, setFlash] = useState<'up' | 'down' | null>(null)
  const prevFair = useRef<number | null>(null)

  useEffect(() => {
    const mid = market.market_mid
    if (mid !== null) {
      setHistory((h) => [...h.slice(-19), { v: mid }])
    }
  }, [market.market_mid])

  useEffect(() => {
    const fair = market.blended_fair
    if (fair !== null && prevFair.current !== null) {
      setFlash(fair > prevFair.current ? 'up' : 'down')
      const t = setTimeout(() => setFlash(null), 500)
      prevFair.current = fair
      return () => clearTimeout(t)
    }
    if (fair !== null) prevFair.current = fair
  }, [market.blended_fair])

  const mid = market.market_mid
  const fair = market.blended_fair
  const edge = mid !== null && fair !== null ? Math.abs(fair - mid) : null

  const rowBg =
    flash === 'up' ? 'bg-green-900/30' :
    flash === 'down' ? 'bg-red-900/30' :
    'bg-transparent'

  return (
    <tr className={`border-b border-[#2a2a2a] text-xs transition-colors duration-300 ${rowBg}`}>
      <td className="px-3 py-2 font-mono text-cyan-400">{market.ticker}</td>
      <td className="px-3 py-2 text-right text-white">
        {mid !== null ? (mid * 100).toFixed(1) + '¢' : '—'}
      </td>
      <td className="px-3 py-2 text-right text-white">
        {fair !== null ? (fair * 100).toFixed(1) + '¢' : '—'}
      </td>
      <td className={`px-3 py-2 text-right font-semibold ${edgeColor(edge, market.strategy_lock)}`}>
        {market.strategy_lock
          ? '🔒'
          : edge !== null
          ? (edge * 100).toFixed(1) + '¢'
          : '—'}
      </td>
      <td className="px-3 py-2 text-right text-gray-400">
        {market.ci_width !== null ? (market.ci_width * 100).toFixed(1) + '%' : '—'}
      </td>
      <td className="px-3 py-2 text-right text-gray-400">
        {market.horizon_days ?? '—'}d
      </td>
      <td className="px-3 py-2 text-right text-white">
        {market.net_contracts ?? 0}
      </td>
      <td className="px-3 py-2 w-20">
        {history.length > 1 && (
          <ResponsiveContainer width="100%" height={30}>
            <LineChart data={history}>
              <Line
                type="monotone"
                dataKey="v"
                stroke="#22d3ee"
                dot={false}
                strokeWidth={1}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </td>
    </tr>
  )
}
