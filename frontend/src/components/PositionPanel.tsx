import type { Position } from '../types'

type Props = {
  positions: Position[]
}

export function PositionPanel({ positions }: Props) {
  const open = positions.filter((p) => p.net_contracts !== 0)
  const totalUnrealized = open.reduce((s, p) => s + p.unrealized_pnl, 0)
  const totalRealized = positions.reduce((s, p) => s + p.realized_pnl, 0)

  return (
    <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
      <h3 className="text-white font-semibold mb-3">Open Positions</h3>
      <div className="flex gap-6 mb-3 text-xs text-gray-400">
        <div>
          Unrealized:{' '}
          <span className={totalUnrealized >= 0 ? 'text-green-400' : 'text-red-400'}>
            ${totalUnrealized.toFixed(2)}
          </span>
        </div>
        <div>
          Realized:{' '}
          <span className={totalRealized >= 0 ? 'text-green-400' : 'text-red-400'}>
            ${totalRealized.toFixed(2)}
          </span>
        </div>
      </div>
      {open.length === 0 ? (
        <p className="text-gray-500 text-xs">No open positions</p>
      ) : (
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[#2a2a2a] text-gray-500 uppercase">
              <th className="py-1 text-left">Ticker</th>
              <th className="py-1 text-right">Contracts</th>
              <th className="py-1 text-right">Avg Entry</th>
              <th className="py-1 text-right">Unrealized</th>
            </tr>
          </thead>
          <tbody>
            {open.map((p) => (
              <tr key={p.ticker} className="border-b border-[#2a2a2a]">
                <td className="py-1 font-mono text-cyan-400">{p.ticker}</td>
                <td className="py-1 text-right text-white">{p.net_contracts}</td>
                <td className="py-1 text-right text-gray-300">
                  {(p.avg_entry_price * 100).toFixed(1)}¢
                </td>
                <td className={`py-1 text-right ${p.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${p.unrealized_pnl.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
