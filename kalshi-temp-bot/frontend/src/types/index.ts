export type MarketState = {
  ticker: string
  market_mid: number | null
  fair_value_a: number | null
  fair_value_b: number | null
  blended_fair: number | null
  ci_width: number | null
  horizon_days: number | null
  net_contracts: number | null
  strategy_lock: boolean
  last_updated: string | null
}

export type Position = {
  ticker: string
  net_contracts: number
  avg_entry_price: number
  unrealized_pnl: number
  realized_pnl: number
  last_updated: string | null
}

export type DailyPnL = {
  date: string
  daily_pnl: number
  cumulative_pnl: number
  fees_paid: number
}

export type CalibrationHealth = {
  station: string
  lead_bucket: string
  brier_score: number | null
  reliability_slope: number | null
  status: 'green' | 'amber' | 'red' | 'unknown'
}

export type ModelStatus = {
  last_forecast_run: string | null
  last_model_trained: string | null
  blend_weights: { ngboost: number; qrf: number }
  mean_ci_width: number
  calibration_health: CalibrationHealth[]
  num_active_tickers: number
}

export type WsMessage = {
  type: string
  timestamp: string
  markets: MarketState[]
  positions: Position[]
  pnl: { series: DailyPnL[] } | DailyPnL[]
  bot_active: boolean
  model_status?: ModelStatus
}
