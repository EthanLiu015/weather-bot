import type { ModelStatus as ModelStatusType } from '../types'

type Props = {
  status: ModelStatusType | null
}

export function ModelStatus({ status }: Props) {
  if (!status) {
    return (
      <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
        <h3 className="text-white font-semibold mb-2">Model Status</h3>
        <p className="text-gray-500 text-sm">No data</p>
      </div>
    )
  }

  const { blend_weights, mean_ci_width, last_forecast_run, calibration_health } = status
  const ngPct = (blend_weights.ngboost * 100).toFixed(0)
  const qrfPct = (blend_weights.qrf * 100).toFixed(0)

  return (
    <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg p-4">
      <h3 className="text-white font-semibold mb-3">Model Status</h3>
      <div className="space-y-2 text-xs text-gray-400">
        <div className="flex justify-between">
          <span>Last forecast run</span>
          <span className="text-white">
            {last_forecast_run ? new Date(last_forecast_run).toLocaleString() : 'Never'}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Blend (NGBoost / QRF)</span>
          <span className="text-white">{ngPct}% / {qrfPct}%</span>
        </div>
        <div className="flex justify-between">
          <span>Mean CI width</span>
          <span className={mean_ci_width > 0.12 ? 'text-red-400' : 'text-green-400'}>
            {(mean_ci_width * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span>Active tickers</span>
          <span className="text-white">{status.num_active_tickers}</span>
        </div>
      </div>
      {calibration_health.length > 0 && (
        <div className="mt-3 pt-3 border-t border-[#2a2a2a]">
          <p className="text-xs text-gray-500 mb-2">Calibration Health</p>
          <div className="space-y-1">
            {calibration_health.slice(0, 4).map((h, i) => (
              <div key={i} className="flex justify-between text-xs">
                <span className="text-gray-400">{h.station} {h.lead_bucket}</span>
                <span className={
                  h.status === 'green' ? 'text-green-400' :
                  h.status === 'amber' ? 'text-yellow-400' :
                  'text-red-400'
                }>
                  {h.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
