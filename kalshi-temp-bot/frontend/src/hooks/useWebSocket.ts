import { useEffect, useRef, useState, useCallback } from 'react'
import type { WsMessage } from '../types'

const MAX_RETRIES = 10
const BACKOFF_MS = 2000

export function useWebSocket(url: string) {
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const retriesRef = useRef(0)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (retriesRef.current >= MAX_RETRIES) return

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      retriesRef.current = 0
      setConnected(true)
    }

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data) as WsMessage
        setLastMessage(msg)
      } catch {
        // ignore malformed messages
      }
    }

    ws.onclose = () => {
      setConnected(false)
      if (mountedRef.current) {
        retriesRef.current += 1
        const delay = BACKOFF_MS * Math.min(retriesRef.current, 5)
        setTimeout(connect, delay)
      }
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [url])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      wsRef.current?.close()
    }
  }, [connect])

  return { lastMessage, connected }
}
