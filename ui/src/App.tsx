import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { DUR, EASE } from './motion'
import { api, connect } from './api'
import type { Ev, FieldSeries, FileRow, Session } from './types'
import Nav from './components/Nav'
import Idle from './components/Idle'
import Stage from './components/Stage'
import Ledger, { toSteps, finalAnswer } from './components/Ledger'
import Answer from './components/Answer'
import Verdict, { type Figure } from './components/Verdict'
import SlideOver from './components/SlideOver'

export default function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [events, setEvents] = useState<Ev[]>([])
  const [prompt, setPrompt] = useState('')
  const [started, setStarted] = useState(false)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('')
  const [menu, setMenu] = useState(false)
  const [field, setField] = useState<FieldSeries | null>(null)
  const [files, setFiles] = useState<FileRow[]>([])
  const [cwd, setCwd] = useState('')
  const [models, setModels] = useState<{ id: string; label: string }[]>([])
  const [modes, setModes] = useState<string[]>([])
  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    api.models().then(setModels)
    api.modes().then(setModes)
    /* ?session=<id> reopens a finished run. Useful on its own, and it is what
       lets a recorder replay a real run instead of staging one. */
    const wanted = new URLSearchParams(location.search).get('session')
    const load = wanted ? api.session(wanted) : api.newSession()
    load.then((s) => {
      if (s.events?.length) {
        setEvents(s.events)
        const first = s.events.find((e) => e.type === 'user_msg')
        if (first?.text) { setPrompt(first.text); setStarted(true) }
        void findField()
      }
      setSession(s)
      ws.current = connect(s.id, (raw) => {
        const e = raw as Ev
        if (e.type === 'status') {
          if (e.session) setSession((p) => ({ ...(p as Session), ...e.session }))
          setStatus(e.message === 'connected' ? '' : e.message || '')
          return
        }
        if (e.type === 'tool_result' || e.type === 'done') {
          void findField()   // only ever this run's own output
        }
        if (e.type === 'done') { setBusy(false); setStatus('') }
        if (e.type === 'token_count') {
          setSession((p) => p && {
            ...p,
            tokens_in: p.tokens_in + (e.input || 0),
            tokens_out: p.tokens_out + (e.output || 0),
          })
        }
        setEvents((p) => [...p, e])
      })
    })
    return () => ws.current?.close()
  }, [])

  const refreshFiles = (rel = cwd) =>
    api.files(rel).then((r) => { setFiles(r.entries); setCwd(r.rel || '') })

  /* A run may only show what that run produced.

     This used to walk the whole sandbox and open the first field file it found.
     Any prompt at all then displayed whatever result happened to be lying
     there, including for a run that computed nothing. That is fabrication: the
     interface would assert a result the run never made. The search is now
     confined to this session's own working directory. */
  async function findField() {
    if (!session?.id) return false
    const root = `webui_${session.id}`
    const walk = async (rel: string, depth: number): Promise<boolean> => {
      if (depth > 2) return false
      const r = await api.files(rel).catch(() => null)
      if (!r) return false
      for (const f of r.entries) {
        if (f.is_dir) { if (await walk(f.rel_path, depth + 1)) return true; continue }
        if (!f.name.endsWith('.json')) continue
        const v = await api.viz(f.rel_path).catch(() => null)
        if (v?.kind === 'field_series') { setField(v as FieldSeries); return true }
      }
      return false
    }
    return walk(root, 0)
  }

  useEffect(() => { if (menu) refreshFiles() }, [menu])

  function send() {
    if (!prompt.trim() || !ws.current) return
    ws.current.send(JSON.stringify({ type: 'prompt', text: prompt }))
    setStarted(true); setBusy(true)
  }
  const decide = (call_id: string, ok: boolean) =>
    ws.current?.send(JSON.stringify(
      ok ? { type: 'approve', call_id } : { type: 'reject', call_id, reason: 'not now' }))

  async function openFile(f: FileRow) {
    if (f.is_dir) { refreshFiles(f.rel_path); return }
    const v = await api.viz(f.rel_path)
    if (v.kind === 'field_series') { setField(v as FieldSeries); setMenu(false) }
  }

  const [cut, setCut] = useState<number | null>(null)

  /* The other half of the capture hook: show the run as it stood after N
     steps. The events are the real ones; this only decides how many of them
     have arrived yet. */
  useEffect(() => {
    ;(window as unknown as Record<string, unknown>).__film = {
      start: (text: string) => { setPrompt(text); setStarted(true) },
      steps: (n: number | null) => setCut(n),
    }
  }, [])

  const allSteps = toSteps(events)
  const steps = cut === null ? allSteps : allSteps.slice(0, cut)
  const answer = cut === null || cut >= allSteps.length ? finalAnswer(events) : ''

  /* The verdict is read out of the run, never asserted by the interface. */
  const meshCalls = new Set(
    events.filter((e) => e.type === 'tool_call_pending'
                      && e.tool === 'verify_mesh_independence')
          .map((e) => e.call_id))
  const meshLine = events
    .filter((e) => e.type === 'tool_result' && meshCalls.has(e.call_id || ''))
    .map((e) => (e.result || '').replace(/\\n/g, '\n'))
    .map((r) => r.split('\n').find((l) => /\bNOT CONVERGED|\bCONVERGED/.test(l)) || '')
    .filter(Boolean).pop()
  const converged = meshLine ? !meshLine.includes('NOT CONVERGED') : null
  const figures: Figure[] = []

  return (
    <>
      <Nav onMenu={() => setMenu(true)} />

      {!started ? (
        <Idle value={prompt} onChange={setPrompt} onSend={send} busy={busy} />
      ) : (
        <>
          <section className="w-[1224px] mx-auto pt-12">
            <div className="eyebrow flex items-center gap-3">
              {busy && <span className="w-2 h-2 rounded-full bg-coral animate-pulse" />}
              <span>{busy ? (status || 'working') : 'finished'}</span>
              <span className="text-graphit">&middot;</span>
              <span>{session?.model ?? ''}</span>
              {steps.length > 0 && <>
                <span className="text-graphit">&middot;</span>
                <span>{steps.length} {steps.length === 1 ? "step" : "steps"}</span>
              </>}
            </div>
            <h1 className="mt-5 text-[40px] font-medium tracking-[-0.024em] leading-[1.12]
                           text-ink line-clamp-2 max-w-[1100px]">
              {prompt}
            </h1>
          </section>

          <AnimatePresence>
            {field && (
              <motion.div key="stage"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                transition={{ duration: 0.6, ease: 'linear' }}>
                <Stage series={field} />
              </motion.div>
            )}
          </AnimatePresence>

          <div className="w-[1224px] mx-auto mt-12 grid grid-cols-[1fr_476px] gap-16 items-start">
            <Ledger steps={steps}
                    onApprove={(id) => decide(id, true)}
                    onReject={(id) => decide(id, false)} />
            <div className="space-y-12">
            <Answer text={answer} />
            {converged !== null && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: DUR.panel, ease: EASE }}>
              <Verdict
                headline={converged ? 'The answer stopped changing.' : 'Not converged yet.'}
                sub={meshLine ? meshLine.slice(0, 180) : ''}
                figures={figures}
              />
              </motion.div>
            )}
            </div>
          </div>
          <div className="h-16" />
        </>
      )}

      <SlideOver
        open={menu} onClose={() => setMenu(false)} session={session}
        models={models} modes={modes} files={files} cwd={cwd}
        onModel={(m) => ws.current?.send(JSON.stringify({ type: 'set_model', model: m }))}
        onMode={(m) => ws.current?.send(JSON.stringify({ type: 'set_mode', mode: m }))}
        onOpenFile={openFile}
        onUp={() => refreshFiles(cwd.split('/').slice(0, -1).join('/'))}
      />
    </>
  )
}
