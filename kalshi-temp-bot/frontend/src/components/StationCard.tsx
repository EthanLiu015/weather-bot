import type { MarketState, CalibrationHealth } from '../types'

type Props = {
  station: string
  markets: MarketState[]
  calibration: CalibrationHealth | undefined
  lastRun: string | null
  d0Active: boolean
}

function HealthDot({ status }: { status: CalibrationHealth['status'] }) {
  const color =
    status === 'green' ? 'bg-green-500' :
    status === 'amber' ? 'bg-yellow-500' :
    status === 'red' ? 'bg-red-500' :
    'bg-gray-500'
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} />
}

export function StationCard({ station, markets, calibration, lastRun, d0Active }: Props) {
  const shortName = station.replace('K', '')
  const activeCount = markets.filter((m) => m.ticker.includes(shortName)).length

  return (
    <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-white font-bold text-lg">{station}</span>
        <div className="flex items-center gap-2">
          {calibration && <HealthDot status={calibration.status} />}
          {d0Active && (
            <span className="text-xs bg-blue-900 text-blue-300 px-2 py-0.5 rounded">D-0</span>
          )}
        </div>
      </div>
      <div className="text-xs text-gray-400 space-y-1">
        <div>
          Active markets:{' '}
          <span className="text-white">{activeCount}</span>
        </div>
        {lastRun && (
          <div>
            Last run:{' '}
            <span className="text-white">
              {new Date(lastRun).toLocaleTimeString()}
            </span>
          </div>
        )}
        {calibration && (
          <div>
            Brier:{' '}
            <span className="text-white">
              {calibration.brier_score?.toFixed(4) ?? 'N/A'}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
