import type { MarketState } from '../types'

// Series prefix → {badge, city, station}
const SERIES_META: Record<string, { badge: string; city: string; station: string }> = {
  KXHIGHCHI:   { badge: 'CHI', city: 'Chicago',       station: 'KORD' },
  KXLOWTCHI:   { badge: 'CHI', city: 'Chicago',       station: 'KORD' },
  KXLOWCHI:    { badge: 'CHI', city: 'Chicago',       station: 'KORD' },
  KXHIGHNY:    { badge: 'NYC', city: 'New York City', station: 'KLGA' },
  KXHIGHNY0:   { badge: 'NYC', city: 'New York City', station: 'KLGA' },
  KXLOWTNYC:   { badge: 'NYC', city: 'New York City', station: 'KLGA' },
  KXLOWNYC:    { badge: 'NYC', city: 'New York City', station: 'KLGA' },
  KXHIGHLAX:   { badge: 'LA',  city: 'Los Angeles',   station: 'KLAX' },
  KXLOWTLAX:   { badge: 'LA',  city: 'Los Angeles',   station: 'KLAX' },
  KXHIGHMIA:   { badge: 'MIA', city: 'Miami',         station: 'KMIA' },
  KXLOWTMIA:   { badge: 'MIA', city: 'Miami',         station: 'KMIA' },
  KXLOWMIA:    { badge: 'MIA', city: 'Miami',         station: 'KMIA' },
  KXHIGHOU:    { badge: 'HOU', city: 'Houston',       station: 'KIAH' },
  KXHIGHHOU:   { badge: 'HOU', city: 'Houston',       station: 'KIAH' },
  KXHOUHIGH:   { badge: 'HOU', city: 'Houston',       station: 'KIAH' },
  KXHIGHTHOU:  { badge: 'HOU', city: 'Houston',       station: 'KIAH' },
  KXLOWTHOU:   { badge: 'HOU', city: 'Houston',       station: 'KIAH' },
  KXHIGHPHIL:  { badge: 'PHI', city: 'Philadelphia',  station: 'KPHL' },
  KXLOWTPHIL:  { badge: 'PHI', city: 'Philadelphia',  station: 'KPHL' },
  KXLOWPHIL:   { badge: 'PHI', city: 'Philadelphia',  station: 'KPHL' },
  KXHIGHATL:   { badge: 'ATL', city: 'Atlanta',       station: 'KATL' },
  KXHIGHTATL:  { badge: 'ATL', city: 'Atlanta',       station: 'KATL' },
  KXLOWTATL:   { badge: 'ATL', city: 'Atlanta',       station: 'KATL' },
  KXHIGHAUS:   { badge: 'AUS', city: 'Austin',        station: 'KAUS' },
  KXLOWTAUS:   { badge: 'AUS', city: 'Austin',        station: 'KAUS' },
  KXLOWAUS:    { badge: 'AUS', city: 'Austin',        station: 'KAUS' },
  KXDENHIGH:   { badge: 'DEN', city: 'Denver',        station: 'KDEN' },
  KXHIGHDEN:   { badge: 'DEN', city: 'Denver',        station: 'KDEN' },
  KXLOWTDEN:   { badge: 'DEN', city: 'Denver',        station: 'KDEN' },
  KXHIGHTPHX:  { badge: 'PHX', city: 'Phoenix',       station: 'KPHX' },
  KXLOWTPHX:   { badge: 'PHX', city: 'Phoenix',       station: 'KPHX' },
  KXHIGHTSFO:  { badge: 'SF',  city: 'San Francisco', station: 'KSFO' },
  KXLOWTSFO:   { badge: 'SF',  city: 'San Francisco', station: 'KSFO' },
  KXHIGHTSEA:  { badge: 'SEA', city: 'Seattle',       station: 'KSEA' },
  KXLOWTSEA:   { badge: 'SEA', city: 'Seattle',       station: 'KSEA' },
  KXHIGHTBOS:  { badge: 'BOS', city: 'Boston',        station: 'KBOS' },
  KXLOWTBOS:   { badge: 'BOS', city: 'Boston',        station: 'KBOS' },
  KXHIGHTDAL:  { badge: 'DAL', city: 'Dallas',        station: 'KDFW' },
  KXLOWTDAL:   { badge: 'DAL', city: 'Dallas',        station: 'KDFW' },
  KXHIGHTDC:   { badge: 'DC',  city: 'Washington DC', station: 'KDCA' },
  KXLOWTDC:    { badge: 'DC',  city: 'Washington DC', station: 'KDCA' },
  KXHIGHTLV:   { badge: 'LV',  city: 'Las Vegas',     station: 'KLAS' },
  KXLOWTLV:    { badge: 'LV',  city: 'Las Vegas',     station: 'KLAS' },
  KXHIGHTMIN:  { badge: 'MIN', city: 'Minneapolis',   station: 'KMSP' },
  KXLOWTMIN:   { badge: 'MIN', city: 'Minneapolis',   station: 'KMSP' },
  KXHIGHTOKC:  { badge: 'OKC', city: 'Oklahoma City', station: 'KOKC' },
  KXLOWTOKC:   { badge: 'OKC', city: 'Oklahoma City', station: 'KOKC' },
  KXHIGHTSATX: { badge: 'SA',  city: 'San Antonio',   station: 'KSAT' },
  KXLOWTSATX:  { badge: 'SA',  city: 'San Antonio',   station: 'KSAT' },
  KXHIGHTNOLA: { badge: 'NOL', city: 'New Orleans',   station: 'KMSY' },
  KXLOWTNOLA:  { badge: 'NOL', city: 'New Orleans',   station: 'KMSY' },
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
  const MONTHS: Record<string, string> = {
    JAN:'01',FEB:'02',MAR:'03',APR:'04',MAY:'05',JUN:'06',
    JUL:'07',AUG:'08',SEP:'09',OCT:'10',NOV:'11',DEC:'12',
  }
  let resolveDate = 'unknown'
  if (parts.length >= 2) {
    const dm = parts[1].match(/^(\d{2})([A-Z]{3})(\d{2})$/)
    if (dm && MONTHS[dm[2]]) {
      resolveDate = `20${dm[1]}-${MONTHS[dm[2]]}-${dm[3].padStart(2,'0')}`
    }
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

  // Visual identity: orange sun = high temp, blue snowflake = low temp
  const badgeBg   = isHigh ? 'bg-orange-500'  : 'bg-blue-600'
  const accentBar = isHigh ? 'bg-green-400'   : 'bg-cyan-400'
  const accentLow = isHigh ? 'bg-blue-500'    : 'bg-purple-500'
  const borderAccent = isHigh ? 'border-[#222]' : 'border-blue-900/40'
  const typeLabel = isHigh ? 'HIGH TEMPERATURE' : 'LOW TEMPERATURE'
  const typeIcon  = isHigh ? '☀' : '❄'
  // Compute a human date label relative to local time
  const now = new Date()
  const todayStr = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`
  const tomorrowDate = new Date(now); tomorrowDate.setDate(now.getDate() + 1)
  const tomorrowStr = `${tomorrowDate.getFullYear()}-${String(tomorrowDate.getMonth()+1).padStart(2,'0')}-${String(tomorrowDate.getDate()).padStart(2,'0')}`
  const dateLabel = resolveDate === todayStr ? 'today'
    : resolveDate === tomorrowStr ? 'tomorrow'
    : resolveDate !== 'unknown'
      ? new Date(resolveDate + 'T12:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
      : 'today'

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

      {/* All market rows — each shown as a bar */}
      <div className="space-y-2.5">
        {sorted.map((m, i) => {
          const mid = m.market_mid ?? 0.5
          const fair = m.blended_fair
          const pct = Math.round(mid * 100)
          const hasEdge = fair !== null && Math.abs(fair - mid) > 0.04
          const edgeColor = fair !== null && fair > mid ? 'text-green-400' : 'text-orange-400'
          const barColor = pct >= 50 ? accentBar : accentLow
          const circleColor = pct >= 50
            ? (isHigh ? 'border-green-500 text-green-400' : 'border-cyan-500 text-cyan-400')
            : (isHigh ? 'border-blue-500 text-blue-400'  : 'border-purple-500 text-purple-400')
          const label = rangeLabel(i)

          return (
            <div key={m.ticker}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-white text-sm">{label}</span>
                  {hasEdge && fair !== null && (
                    <span className={`text-[10px] font-bold ${edgeColor}`}>
                      {fair > mid ? '↑' : '↓'} {Math.round(fair * 100)}¢
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 text-xs">{multiplier(mid)}</span>
                  <div className={`w-11 h-6 rounded-full border text-xs font-bold flex items-center justify-center ${circleColor}`}>
                    {pct}%
                  </div>
                </div>
              </div>
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

/** Group markets by (station, isHigh), keeping only the nearest resolution date
 *  per city+direction so there are no duplicates when Kalshi rolls to a new day. */
export function groupMarketsByCity(markets: MarketState[]): Map<string, CityGroup> {
  // First pass: find the nearest (earliest) resolve date per city+direction key
  const nearestDate = new Map<string, string>()
  for (const m of markets) {
    const parsed = parseTicker(m.ticker)
    if (!parsed || parsed.threshold === null || parsed.resolveDate === 'unknown') continue
    const key = `${parsed.station}-${parsed.isHigh ? 'high' : 'low'}`
    const existing = nearestDate.get(key)
    if (!existing || parsed.resolveDate < existing) {
      nearestDate.set(key, parsed.resolveDate)
    }
  }

  // Second pass: only include contracts from the nearest date for each key
  const groups = new Map<string, CityGroup>()
  for (const m of markets) {
    const parsed = parseTicker(m.ticker)
    if (!parsed || parsed.threshold === null) continue
    const key = `${parsed.station}-${parsed.isHigh ? 'high' : 'low'}`
    if (parsed.resolveDate !== nearestDate.get(key)) continue

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

  return new Map([...groups.entries()].sort(([, a], [, b]) =>
    a.city.localeCompare(b.city)
  ))
}
