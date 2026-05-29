import { useEffect, useState } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'
import { MarketTable } from './MarketTable'
import { PnLChart } from './PnLChart'
import { PositionPanel } from './PositionPanel'
import { ModelStatus } from './ModelStatus'
import { StationCard } from './StationCard'
import { KillSwitch } from './KillSwitch'
import type { MarketState, Position, DailyPnL, ModelStatus as ModelStatusType } from '../types'

const WS_URL = `ws://${window.location.host}/ws/live`
const STATIONS = ['KORD', 'KJFK', 'KLAX']

export function Dashboard() {
  const { lastMessage, connected } = useWebSocket(WS_URL)
  const [markets, setMarkets] = useState<MarketState[]>([])
  const [positions, setPositions] = useState<Position[]>([])
  const [pnlSeries, setPnlSeries] = useState<DailyPnL[]>([])
  const [botActive, setBotActive] = useState(true)
  const [modelStatus, setModelStatus] = useState<ModelStatusType | null>(null)

  useEffect(() => {
    if (!lastMessage) return
    if (lastMessage.markets) setMarkets(lastMessage.markets)
    if (lastMessage.positions) setPositions(lastMessage.positions)
    if (lastMessage.pnl) {
      const pnl = lastMessage.pnl
      setPnlSeries(Array.isArray(pnl) ? pnl : (pnl as { series: DailyPnL[] }).series ?? [])
    }
    if (typeof lastMessage.bot_active === 'boolean') setBotActive(lastMessage.bot_active)
    if (lastMessage.model_status) setModelStatus(lastMessage.model_status)
  }, [lastMessage])

  async function handleKill() {
    await fetch('/api/controls/kill', { method: 'POST' })
    setBotActive(false)
  }

  async function handleResume() {
    await fetch('/api/controls/resume', { method: 'POST' })
    setBotActive(true)
  }

  return (
    <div className="min-h-screen bg-[#0f0f0f] text-white font-mono">
      {/* Header */}
      <header className="border-b border-[#2a2a2a] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold tracking-tight">Kalshi Temp Bot</span>
          <span className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded ${connected ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`} />
            {connected ? 'LIVE' : 'DISCONNECTED'}
          </span>
          {!botActive && (
            <span className="text-xs px-2 py-0.5 rounded bg-red-900 text-red-300 font-bold">
              KILL SWITCH ACTIVE
            </span>
          )}
        </div>
        <KillSwitch botActive={botActive} onKill={handleKill} onResume={handleResume} />
      </header>

      <main className="p-6 space-y-6">
        {/* Top row: Station Cards + Model Status + PnL Chart */}
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-4 grid grid-cols-1 gap-3">
            {STATIONS.map((station) => (
              <StationCard
                key={station}
                station={station}
                markets={markets}
                calibration={modelStatus?.calibration_health.find((h) => h.station === station)}
                lastRun={modelStatus?.last_forecast_run ?? null}
                d0Active={false}
              />
            ))}
          </div>
          <div className="col-span-2">
            <ModelStatus status={modelStatus} />
          </div>
          <div className="col-span-6">
            <PnLChart series={pnlSeries} />
          </div>
        </div>

        {/* Market Table */}
        <MarketTable markets={markets} />

        {/* Bottom row: Positions */}
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-12">
            <PositionPanel positions={positions} />
          </div>
        </div>
      </main>
    </div>
  )
}
