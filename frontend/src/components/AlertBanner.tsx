import type { Alert } from '../types'

type Props = {
  alerts: Alert[]
  onDismiss: (timestamp: string) => void
}

const ALERT_STYLES: Record<string, string> = {
  gefs_coverage:   'bg-yellow-900 border-yellow-700 text-yellow-200',
  ingestion_error: 'bg-red-900 border-red-700 text-red-200',
  ensemble_error:  'bg-red-900 border-red-700 text-red-200',
  default:         'bg-orange-900 border-orange-700 text-orange-200',
}

const ALERT_ICONS: Record<string, string> = {
  gefs_coverage:   '⚠',
  ingestion_error: '✕',
  ensemble_error:  '✕',
  default:         '!',
}

export function AlertBanner({ alerts, onDismiss }: Props) {
  if (alerts.length === 0) return null

  return (
    <div className="px-6 pt-2 space-y-1">
      {alerts.map(alert => {
        const style = ALERT_STYLES[alert.type] ?? ALERT_STYLES.default
        const icon  = ALERT_ICONS[alert.type]  ?? ALERT_ICONS.default
        return (
          <div
            key={alert.timestamp}
            className={`flex items-center justify-between px-4 py-2 rounded border text-xs font-mono ${style}`}
          >
            <span>
              <span className="font-bold mr-2">{icon}</span>
              {alert.message}
              <span className="ml-3 opacity-50">
                {new Date(alert.timestamp).toLocaleTimeString()}
              </span>
            </span>
            <button
              onClick={() => onDismiss(alert.timestamp)}
              className="ml-4 opacity-60 hover:opacity-100 text-sm leading-none"
            >
              ×
            </button>
          </div>
        )
      })}
    </div>
  )
}
