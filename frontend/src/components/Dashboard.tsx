import { useEffect, useState } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'
import { PnLChart } from './PnLChart'
import { PositionPanel } from './PositionPanel'
import { ModelStatus } from './ModelStatus'
import { KillSwitch } from './KillSwitch'
import { AlertBanner } from './AlertBanner'
import { CityCard, groupMarketsByCity } from './CityCard'
import type { MarketState, Position, DailyPnL, ModelStatus as ModelStatusType, Alert } from '../types'

const WS_URL = `ws://${window.location.host}/ws/live`

type Tab = 'markets' | 'performance' | 'positions'

export function Dashboard() {
  const { lastMessage, connected } = useWebSocket(WS_URL)
  const [markets, setMarkets] = useState<MarketState[]>([])
  const [positions, setPositions] = useState<Position[]>([])
  const [pnlSeries, setPnlSeries] = useState<DailyPnL[]>([])
  const [botActive, setBotActive] = useState(true)
  const [paperTrading, setPaperTrading] = useState(true)
  const [modelStatus, setModelStatus] = useState<ModelStatusType | null>(null)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [tab, setTab] = useState<Tab>('markets')
  const [filter, setFilter] = useState('')

  useEffect(() => {
    if (!lastMessage) return
    if (lastMessage.markets) {
      const raw = lastMessage.markets
      setMarkets(Array.isArray(raw) ? raw : Object.entries(raw as Record<string, Omit<MarketState,'ticker'>>).map(([t, v]) => ({ ticker: t, ...v })))
    }
    if (lastMessage.positions) setPositions(lastMessage.positions)
    if (lastMessage.pnl) {
      const pnl = lastMessage.pnl
      setPnlSeries(Array.isArray(pnl) ? pnl : (pnl as { series: DailyPnL[] }).series ?? [])
    }
    if (typeof lastMessage.bot_active === 'boolean') setBotActive(lastMessage.bot_active)
    if (lastMessage.model_status) setModelStatus(lastMessage.model_status)
    if (lastMessage.alerts) setAlerts(lastMessage.alerts)
  }, [lastMessage])

  useEffect(() => {
    fetch('/api/controls/status').then(r => r.json()).then(d => {
      if (typeof d.paper_trading === 'boolean') setPaperTrading(d.paper_trading)
    }).catch(() => {})
  }, [])

  async function handleKill() {
    await fetch('/api/controls/kill', { method: 'POST' })
    setBotActive(false)
  }
  async function handleResume() {
    await fetch('/api/controls/resume', { method: 'POST' })
    setBotActive(true)
  }

  const cityGroups = groupMarketsByCity(markets)
  const filteredGroups = filter
    ? new Map([...cityGroups].filter(([, v]) =>
        v.city.toLowerCase().includes(filter.toLowerCase()) ||
        v.badge.toLowerCase().includes(filter.toLowerCase())
      ))
    : cityGroups

  const TABS: { id: Tab; label: string }[] = [
    { id: 'markets',     label: 'Markets' },
    { id: 'performance', label: 'Performance' },
    { id: 'positions',   label: 'Positions' },
  ]

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white" style={{ fontFamily: 'system-ui, sans-serif' }}>

      {/* Header */}
      <header className="px-6 py-4 flex items-center justify-between border-b border-[#1a1a1a]">
        <div className="flex items-center gap-4">
          <span className="text-xl font-bold tracking-tight">Climate</span>
          <span className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium
            ${connected ? 'bg-green-900/60 text-green-300' : 'bg-red-900/60 text-red-300'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`} />
            {connected ? 'Live' : 'Disconnected'}
          </span>
          {paperTrading && (
            <span className="text-xs px-2.5 py-1 rounded-full bg-yellow-900/60 text-yellow-300 font-medium">
              Paper Trading
            </span>
          )}
          {!botActive && (
            <span className="text-xs px-2.5 py-1 rounded-full bg-red-900/60 text-red-300 font-bold">
              Kill Switch Active
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Model blend weights */}
          {modelStatus && (
            <span className="text-xs text-gray-500">
              NGBoost {Math.round(modelStatus.blend_weights.ngboost * 100)}% ·
              QRF {Math.round(modelStatus.blend_weights.qrf * 100)}%
            </span>
          )}
          <KillSwitch botActive={botActive} onKill={handleKill} onResume={handleResume} />
        </div>
      </header>

      <AlertBanner
        alerts={alerts}
        onDismiss={(ts) => setAlerts(prev => prev.filter(a => a.timestamp !== ts))}
      />

      {/* Tab bar */}
      <div className="px-6 pt-4 flex items-center gap-6 border-b border-[#1a1a1a]">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
              tab === t.id
                ? 'border-white text-white'
                : 'border-transparent text-gray-500 hover:text-gray-300'
            }`}
          >
            {t.label}
            {t.id === 'markets' && markets.length > 0 && (
              <span className="ml-2 text-xs bg-[#2a2a2a] text-gray-400 px-1.5 py-0.5 rounded-full">
                {filteredGroups.size}
              </span>
            )}
          </button>
        ))}

        {tab === 'markets' && (
          <div className="ml-auto mb-3">
            <input
              type="text"
              placeholder="Filter city..."
              value={filter}
              onChange={e => setFilter(e.target.value)}
              className="bg-[#1a1a1a] border border-[#2a2a2a] text-white text-sm rounded-lg px-3 py-1.5 w-40 focus:outline-none focus:border-[#444] placeholder-gray-600"
            />
          </div>
        )}
      </div>

      <main className="p-6">

        {/* Markets tab — city card grid */}
        {tab === 'markets' && (
          <>
            {filteredGroups.size === 0 ? (
              <div className="flex flex-col items-center justify-center h-64 text-gray-600 gap-2">
                <span className="text-4xl">🌡</span>
                <p>No active temperature markets</p>
                <p className="text-xs">Markets seed on startup — check the backend logs</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {[...filteredGroups.values()].map(group => (
                  <CityCard
                    key={`${group.station}-${group.isHigh}-${group.resolveDate}`}
                    city={group.city}
                    badge={group.badge}
                    isHigh={group.isHigh}
                    resolveDate={group.resolveDate}
                    markets={group.rows}
                  />
                ))}
              </div>
            )}
          </>
        )}

        {/* Performance tab */}
        {tab === 'performance' && (
          <div className="space-y-6 max-w-4xl">
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-[#111] border border-[#222] rounded-2xl p-5">
                <div className="text-xs text-gray-500 mb-1">Active Markets</div>
                <div className="text-2xl font-bold text-white">{filteredGroups.size}</div>
                <div className="text-xs text-gray-600 mt-1">{markets.length} total contracts</div>
              </div>
              <div className="bg-[#111] border border-[#222] rounded-2xl p-5">
                <div className="text-xs text-gray-500 mb-1">Open Positions</div>
                <div className="text-2xl font-bold text-white">
                  {positions.filter(p => p.net_contracts !== 0).length}
                </div>
              </div>
              <div className="bg-[#111] border border-[#222] rounded-2xl p-5">
                <div className="text-xs text-gray-500 mb-1">Model CI Width</div>
                <div className={`text-2xl font-bold ${(modelStatus?.mean_ci_width ?? 0) > 0.12 ? 'text-red-400' : 'text-green-400'}`}>
                  {modelStatus ? `${(modelStatus.mean_ci_width * 100).toFixed(1)}%` : '—'}
                </div>
              </div>
            </div>
            <PnLChart series={pnlSeries} />
            {modelStatus && <ModelStatus status={modelStatus} />}
          </div>
        )}

        {/* Positions tab */}
        {tab === 'positions' && (
          <div className="max-w-3xl">
            <PositionPanel positions={positions} />
          </div>
        )}

      </main>
    </div>
  )
}
