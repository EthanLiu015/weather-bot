import { useState } from 'react'
import { Dashboard } from './components/Dashboard'
import { BacktestReport } from './components/BacktestReport'

type Tab = 'live' | 'backtest'

export default function App() {
  const [tab, setTab] = useState<Tab>('live')

  return (
    <div className="min-h-screen bg-[#0f0f0f]">
      <nav className="border-b border-[#2a2a2a] px-6 flex gap-1 pt-2">
        {(['live', 'backtest'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
              tab === t
                ? 'bg-[#1a1a1a] text-white border border-b-0 border-[#2a2a2a]'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {t === 'live' ? 'Live Trading' : 'Backtest Report'}
          </button>
        ))}
      </nav>

      {tab === 'live' ? <Dashboard /> : <BacktestReport />}
    </div>
  )
}
