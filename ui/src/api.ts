import type { Session, FileRow } from './types'

const j = (r: Response) => r.json()

export const api = {
  // These endpoints wrap their payload, so unwrap here rather than at each call.
  models:  (): Promise<{ id: string; label: string }[]> =>
    fetch('/api/models').then(j).then((d) => d.models ?? []),
  servers: (): Promise<{ id: string; label: string; default_on: boolean }[]> =>
    fetch('/api/mcp_servers').then(j).then((d) => d.servers ?? []),
  modes:   (): Promise<string[]> =>
    fetch('/api/modes').then(j).then((d) => d.modes ?? []),
  newSession: (): Promise<Session> =>
    fetch('/api/sessions', { method: 'POST' }).then(j),
  session: (id: string): Promise<Session> => fetch(`/api/sessions/${id}`).then(j),
  files:   (rel = ''): Promise<{ entries: FileRow[]; rel: string }> =>
    fetch(`/api/files?rel=${encodeURIComponent(rel)}`).then(j),
  viz:     (rel: string) => fetch(`/api/viz?rel=${encodeURIComponent(rel)}`).then(j),
}

/** One socket per session. The server speaks line delimited JSON both ways. */
export function connect(id: string, onEvent: (e: unknown) => void) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/ws/${id}`)
  ws.onmessage = (m) => {
    try { onEvent(JSON.parse(m.data)) } catch { /* a malformed frame is not fatal */ }
  }
  return ws
}
