import { useState } from 'react'
import { MarketRow } from './MarketRow'
import type { MarketState } from '../types'

type SortKey = 'edge' | 'horizon' | 'ci_width'

type Props = {
  markets: MarketState[]
}

export function MarketTable({ markets }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('edge')
  const [sortDesc, setSortDesc] = useState(true)

  const sorted = [...markets].sort((a, b) => {
    let av: number, bv: number
    if (sortKey === 'edge') {
      av = a.market_mid !== null && a.blended_fair !== null ? Math.abs(a.blended_fair - a.market_mid) : -1
      bv = b.market_mid !== null && b.blended_fair !== null ? Math.abs(b.blended_fair - b.market_mid) : -1
    } else if (sortKey === 'horizon') {
      av = a.horizon_days ?? 999
      bv = b.horizon_days ?? 999
    } else {
      av = a.ci_width ?? 999
      bv = b.ci_width ?? 999
    }
    return sortDesc ? bv - av : av - bv
  })

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDesc((d) => !d)
    else { setSortKey(key); setSortDesc(true) }
  }

  function SortTh({ label, k }: { label: string; k: SortKey }) {
    return (
      <th
        className="px-3 py-2 text-right cursor-pointer select-none hover:text-white text-gray-400"
        onClick={() => toggleSort(k)}
      >
        {label} {sortKey === k ? (sortDesc ? '↓' : '↑') : ''}
      </th>
    )
  }

  return (
    <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg overflow-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-[#2a2a2a] text-gray-500 uppercase tracking-wide">
            <th className="px-3 py-2 text-left">Ticker</th>
            <th className="px-3 py-2 text-right">Mid</th>
            <th className="px-3 py-2 text-right">Fair</th>
            <SortTh label="Edge" k="edge" />
            <SortTh label="CI%" k="ci_width" />
            <SortTh label="Horizon" k="horizon" />
            <th className="px-3 py-2 text-right">Pos</th>
            <th className="px-3 py-2">Sparkline</th>
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 ? (
            <tr>
              <td colSpan={8} className="px-3 py-8 text-center text-gray-500">
                No active markets
              </td>
            </tr>
          ) : (
            sorted.map((m) => <MarketRow key={m.ticker} market={m} />)
          )}
        </tbody>
      </table>
    </div>
  )
}
