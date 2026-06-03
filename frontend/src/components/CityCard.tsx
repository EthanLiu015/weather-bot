import type { MarketState } from '../types'

// Series prefix → {badge, city, station}
const SERIES_META: Record<string, { badge: string; city: string; station: string }> = {
  KXHIGHCHI:  { badge: 'CHI', city: 'Chicago',       station: 'KORD' },
  KXLOWTCHI:  { badge: 'CHI', city: 'Chicago',       station: 'KORD' },
  KXHIGHNY:   { badge: 'NYC', city: 'New York City', station: 'KLGA' },
  KXHIGHNY0:  { badge: 'NYC', city: 'New York City', station: 'KLGA' },
  KXHIGHLAX:  { badge: 'LA',  city: 'Los Angeles',   station: 'KLAX' },
  KXLOWTLAX:  { badge: 'LA',  city: 'Los Angeles',   station: 'KLAX' },
  KXHIGHMIA:  { badge: 'MIA', city: 'Miami',         station: 'KMIA' },
  KXLOWMIA:   { badge: 'MIA', city: 'Miami',         station: 'KMIA' },
  KXHIGHOU:   { badge: 'HOU', city: 'Houston',       station: 'KIAH' },
  KXHIGHHOU:  { badge: 'HOU', city: 'Houston',       station: 'KIAH' },
  KXHIGHPHIL: { badge: 'PHI', city: 'Philadelphia',  station: 'KPHL' },
  KXHIGHATL:  { badge: 'ATL', city: 'Atlanta',       station: 'KATL' },
  KXHIGHAUS:  { badge: 'AUS', city: 'Austin',        station: 'KAUS' },
  KXDENHIGH:  { badge: 'DEN', city: 'Denver',        station: 'KDEN' },
  KXHIGHDEN:  { badge: 'DEN', city: 'Denver',        station: 'KDEN' },
  KXHIGHTPHX: { badge: 'PHX', city: 'Phoenix',       station: 'KPHX' },
  KXHIGHTSFO: { badge: 'SF',  city: 'San Francisco', station: 'KSFO' },
  KXHIGHTSEA: { badge: 'SEA', city: 'Seattle',       station: 'KSEA' },
  KXHIGHTBOS: { badge: 'BOS', city: 'Boston',        station: 'KBOS' },
  KXHIGHTDAL: { badge: 'DAL', city: 'Dallas',        station: 'KDFW' },
  KXHIGHTDC:  { badge: 'DC',  city: 'Washington DC', station: 'KDCA' },
  KXHIGHTLV:  { badge: 'LV',  city: 'Las Vegas',     station: 'KLAS' },
  KXHIGHTMIN: { badge: 'MIN', city: 'Minneapolis',   station: 'KMSP' },
  KXHIGHTOKC: { badge: 'OKC', city: 'Oklahoma City', station: 'KOKC' },
  KXHIGHTSATX:{ badge: 'SA',  city: 'San Antonio',   station: 'KSAT' },
  KXHIGHTNOLA:{ badge: 'NOL', city: 'New Orleans',   station: 'KMSY' },
}

export function parseTicker(ticker: string): {
  series: string
  badge: string
  city: string
  station: string
  threshold: number | null
  isHigh: boolean
  resolveDate: string   // YYYY-MM-DD
} | null {
  const parts = ticker.split('-')
  if (parts.length < 3) return null
  const series = parts[0]
  const meta = SERIES_META[series]
  if (!meta) return null

  const threshMatch = ticker.match(/-[TB]([\d.]+)$/)
  const threshold = threshMatch ? parseFloat(threshMatch[1]) : null
  const isHigh = series.toLowerCase().includes('high')

  // Parse date from YYMMMDD e.g. "26JUN03" → "2026-06-03"
  let resolveDate = 'unknown'
  if (parts.length >= 2) {
    try {
      const d = new Date(parts[1].replace(/(\d{2})([A-Z]{3})(\d{2})/, '20$1-$2-$3'))
      if (!isNaN(d.getTime())) {
        resolveDate = d.toISOString().slice(0, 10)
      }
    } catch { /* leave as unknown */ }
  }

  return { series, ...meta, threshold, isHigh, resolveDate }
}

function multiplier(prob: number): string {
  if (prob <= 0 || prob >= 1) return '—'
  return `${(1 / prob).toFixed(2)}x`
}


type MarketRow = {
  ticker: string
  threshold: number
  market_mid: number | null
  blended_fair: number | null
  net_contracts: number | null
}

type Props = {
  city: string
  badge: string
  isHigh: boolean
  resolveDate: string
  markets: MarketRow[]
  totalVolume?: number
}

export function CityCard({ city, badge, isHigh, resolveDate, markets, totalVolume }: Props) {
  // High markets: sort descending (highest threshold first — most interesting to traders)
  // Low markets: sort ascending (lowest threshold first)
  const sorted = [...markets].sort((a, b) =>
    isHigh ? b.threshold - a.threshold : a.threshold - b.threshold
  )
  const featured = sorted.slice(0, 2)
  const rest = sorted.slice(2)

  // Visual identity: orange sun = high temp, blue snowflake = low temp
  const badgeBg   = isHigh ? 'bg-orange-500'  : 'bg-blue-600'
  const accentBar = isHigh ? 'bg-green-400'   : 'bg-cyan-400'
  const accentLow = isHigh ? 'bg-blue-500'    : 'bg-purple-500'
  const borderAccent = isHigh ? 'border-[#222]' : 'border-blue-900/40'
  const typeLabel = isHigh ? 'HIGH TEMPERATURE' : 'LOW TEMPERATURE'
  const typeIcon  = isHigh ? '☀' : '❄'
  const today = new Date().toISOString().slice(0, 10)
  const dateLabel = resolveDate === today ? 'today'
    : resolveDate === new Date(Date.now() + 86400000).toISOString().slice(0, 10) ? 'tomorrow'
    : resolveDate !== 'unknown' ? new Date(resolveDate + 'T12:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    : ''
  const question  = isHigh
    ? `Highest temperature in ${city} ${dateLabel}?`
    : `Lowest temperature in ${city} ${dateLabel}?`

  // Compute the display range for each contract by looking at the adjacent threshold.
  // Kalshi shows "82° to 83°" between consecutive contracts, not "above 82°".
  // For HIGH (sorted descending: [87, 83, 80]):
  //   index 0 (87) → "above 87°"
  //   index 1 (83) → "83° to 87°"
  //   index 2 (80) → "80° to 83°"
  // For LOW (sorted ascending: [46, 50, 54]):
  //   index 0 (46) → "below 46°"
  //   index 1 (50) → "46° to 50°"
  //   index 2 (54) → "50° to 54°"
  function rangeLabel(index: number): string {
    const cur = sorted[index].threshold
    if (isHigh) {
      if (index === 0) return `above ${cur}°`
      const upper = sorted[index - 1].threshold
      return `${cur}° to ${upper}°`
    } else {
      if (index === 0) return `below ${cur}°`
      const lower = sorted[index - 1].threshold
      return `${lower}° to ${cur}°`
    }
  }

  return (
    <div className={`bg-[#111] border ${borderAccent} rounded-2xl p-5 flex flex-col gap-3`}>
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className={`w-9 h-9 rounded-xl ${badgeBg} flex items-center justify-center text-white text-xs font-bold shrink-0`}>
          {badge}
        </div>
        <div>
          <div className="flex items-center gap-1.5 text-[10px] text-gray-500 uppercase tracking-widest">
            <span>{typeIcon}</span>
            <span>{typeLabel}</span>
          </div>
          <div className="text-sm font-semibold text-white leading-tight">
            {question}
          </div>
        </div>
      </div>

      {/* Featured market rows */}
      <div className="space-y-3">
        {featured.map((m, featuredIdx) => {
          const mid = m.market_mid ?? 0.5
          const fair = m.blended_fair
          const pct = Math.round(mid * 100)
          const hasEdge = fair !== null && Math.abs(fair - mid) > 0.04
          const edgeColor = fair !== null && fair > mid ? 'text-green-400' : 'text-orange-400'
          // High markets: green for likely-yes (high prob), blue for unlikely
          // Low markets: cyan for likely-yes, purple for unlikely
          const barColor = pct >= 50 ? accentBar : accentLow
          const circleColor = pct >= 50
            ? (isHigh ? 'border-green-500 text-green-400' : 'border-cyan-500 text-cyan-400')
            : (isHigh ? 'border-blue-500 text-blue-400'  : 'border-purple-500 text-purple-400')
          const label = rangeLabel(featuredIdx)

          return (
            <div key={m.ticker}>
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-white text-sm font-medium">
                    {label}
                  </span>
                  {fair !== null && hasEdge && (
                    <span className={`text-[10px] font-bold ${edgeColor}`}>
                      {fair > mid ? '↑' : '↓'} {Math.round(fair * 100)}¢
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs">{multiplier(mid)}</span>
                  <div className={`w-12 h-7 rounded-full border text-xs font-bold flex items-center justify-center ${circleColor}`}>
                    {pct}%
                  </div>
                </div>
              </div>
              {/* Probability bar */}
              <div className="h-0.5 bg-[#2a2a2a] rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${barColor}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {/* Collapsed remaining rows */}
      {rest.length > 0 && (
        <div className="space-y-1 pt-1 border-t border-[#1e1e1e]">
          {rest.map((m, restIdx) => {
            const mid = m.market_mid ?? 0.5
            const pct = Math.round(mid * 100)
            // rest starts at index featured.length in sorted array
            const label = rangeLabel(featured.length + restIdx)
            return (
              <div key={m.ticker} className="flex items-center justify-between text-xs text-gray-500 py-0.5">
                <span>{label}</span>
                <div className="flex items-center gap-2">
                  <span>{multiplier(mid)}</span>
                  <span className="w-8 text-right">{pct}%</span>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between text-[11px] text-gray-600 pt-1 border-t border-[#1a1a1a]">
        <span>{totalVolume ? `$${totalVolume.toLocaleString()} vol` : ''}</span>
        <span>{markets.length} market{markets.length !== 1 ? 's' : ''}</span>
      </div>
    </div>
  )
}

type CityGroup = {
  city: string
  badge: string
  isHigh: boolean
  station: string
  resolveDate: string
  rows: MarketRow[]
}

/** Group markets by (station, isHigh, resolveDate) — one card per city per day */
export function groupMarketsByCity(markets: MarketState[]): Map<string, CityGroup> {
  const groups = new Map<string, CityGroup>()

  for (const m of markets) {
    const parsed = parseTicker(m.ticker)
    if (!parsed || parsed.threshold === null) continue

    // Key includes date so today's and tomorrow's cards are separate
    const key = `${parsed.station}-${parsed.isHigh ? 'high' : 'low'}-${parsed.resolveDate}`
    if (!groups.has(key)) {
      groups.set(key, {
        city: parsed.city,
        badge: parsed.badge,
        isHigh: parsed.isHigh,
        station: parsed.station,
        resolveDate: parsed.resolveDate,
        rows: [],
      })
    }
    groups.get(key)!.rows.push({
      ticker: m.ticker,
      threshold: parsed.threshold,
      market_mid: m.market_mid,
      blended_fair: m.blended_fair,
      net_contracts: m.net_contracts,
    })
  }

  // Sort: today's markets first, then by city name
  return new Map([...groups.entries()].sort(([, a], [, b]) => {
    if (a.resolveDate !== b.resolveDate) return a.resolveDate.localeCompare(b.resolveDate)
    return a.city.localeCompare(b.city)
  }))
}
