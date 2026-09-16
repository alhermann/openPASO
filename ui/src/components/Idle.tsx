import { motion } from 'motion/react'
import { heroIn } from '../motion'

const SOLVERS = ['FEniCSx', 'deal.II', '4C', 'NGSolve', 'scikit-fem',
                 'Kratos', 'DUNE', 'FEBio', 'SPARTA']

export default function Idle({
  value, onChange, onSend, busy,
}: {
  value: string
  onChange: (v: string) => void
  onSend: () => void
  busy: boolean
}) {
  return (
    <section className="pt-[176px]" data-testid="idle">
      <div className="w-[816px] mx-auto">
        <motion.div className="eyebrow" {...heroIn(0)}>Nine solvers online</motion.div>
        <motion.h1 {...heroIn(1)}
          className="mt-6 text-[72px] font-medium tracking-[-0.028em] leading-[1.02] text-ink">
          Describe the physics.
        </motion.h1>
        <motion.p {...heroIn(2)}
           className="mt-6 max-w-[680px] text-lg leading-[1.55] tracking-[-0.014em]">
          We choose the solver, write its input, run it for real, and prove the
          answer stopped changing.
        </motion.p>

        <motion.div {...heroIn(3)} className="mt-11 w-[816px] h-16 bg-s2 border line rounded-[10px] edge-top
                        flex items-center pl-5 pr-2 focus-within:border-coral
                        transition-colors duration-150">
          <label htmlFor="prompt" className="sr-only">Describe a physics problem</label>
          <input
            id="prompt" data-testid="prompt" value={value} autoComplete="off"
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) onSend() }}
            placeholder="Flow past a cylinder at Reynolds 100"
            className="flex-1 bg-transparent outline-none text-base tracking-[-0.011em]
                       text-ink2 placeholder:text-[#6E7B8B]"
          />
          <button
            onClick={onSend} disabled={busy || !value.trim()} data-testid="send"
            className="h-10 px-[18px] rounded-[6px] bg-coral text-canvas text-[13px] font-medium
                       transition-[background-color,opacity] duration-150 hover:bg-coral-h
                       disabled:opacity-40"
          >Run</button>
        </motion.div>

        <motion.div {...heroIn(4)}
             className="mt-16 flex gap-[22px] font-mono text-[13px] text-muted">
          {SOLVERS.map((s) => <span key={s}>{s}</span>)}
        </motion.div>
      </div>
    </section>
  )
}
