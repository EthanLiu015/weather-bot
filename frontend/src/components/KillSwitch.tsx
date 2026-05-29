import { useState } from 'react'

type Props = {
  botActive: boolean
  onKill: () => Promise<void>
  onResume: () => Promise<void>
}

export function KillSwitch({ botActive, onKill, onResume }: Props) {
  const [showModal, setShowModal] = useState(false)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleKill() {
    if (input !== 'KILL') return
    setLoading(true)
    try {
      await onKill()
    } finally {
      setLoading(false)
      setShowModal(false)
      setInput('')
    }
  }

  return (
    <>
      {botActive ? (
        <button
          onClick={() => setShowModal(true)}
          className="bg-red-700 hover:bg-red-600 text-white font-bold px-4 py-2 rounded border border-red-500 text-sm transition-colors"
        >
          KILL SWITCH
        </button>
      ) : (
        <button
          onClick={onResume}
          className="bg-green-800 hover:bg-green-700 text-white font-bold px-4 py-2 rounded border border-green-600 text-sm transition-colors"
        >
          RESUME
        </button>
      )}

      {showModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-[#1a1a1a] border border-red-800 rounded-xl p-6 w-80">
            <h2 className="text-red-400 font-bold text-lg mb-2">Confirm Kill Switch</h2>
            <p className="text-gray-300 text-sm mb-4">
              This will cancel all resting orders and halt all trading. Type{' '}
              <span className="font-mono font-bold text-red-400">KILL</span> to confirm.
            </p>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type KILL"
              className="w-full bg-[#0f0f0f] border border-[#2a2a2a] text-white rounded px-3 py-2 text-sm mb-4 focus:outline-none focus:border-red-700"
              autoFocus
            />
            <div className="flex gap-3">
              <button
                onClick={handleKill}
                disabled={input !== 'KILL' || loading}
                className="flex-1 bg-red-700 disabled:bg-gray-800 disabled:text-gray-500 text-white font-bold py-2 rounded text-sm transition-colors"
              >
                {loading ? 'Killing...' : 'KILL'}
              </button>
              <button
                onClick={() => { setShowModal(false); setInput('') }}
                className="flex-1 bg-[#2a2a2a] hover:bg-[#333] text-white py-2 rounded text-sm transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
