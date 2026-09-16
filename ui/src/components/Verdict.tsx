export type Figure = { value: string; label: string; note: string; ok?: boolean }

export default function Verdict({
  headline, sub, figures,
}: { headline: string; sub: string; figures: Figure[] }) {
  return (
    <section data-testid="verdict"
             className="bg-s2 border line rounded-[10px] edge-top p-8">
      <h2 className="text-[30px] font-medium tracking-[-0.021em] text-ink2">{headline}</h2>
      <p className="mt-2.5 text-sm text-muted">{sub}</p>
      <div className="mt-6 grid gap-[18px]">
        {figures.map((f) => (
          <div key={f.label}
               className="grid grid-cols-[118px_1fr] gap-x-[18px] gap-y-1 items-baseline">
            <span className="num text-[30px] font-[450] text-ink tracking-[-0.02em] row-span-2">
              {f.value}
            </span>
            <span className="text-[11px] font-medium uppercase tracking-[0.045em]
                             text-graphit self-end">{f.label}</span>
            <span className="num text-xs text-muted">
              {f.note}{f.ok && <span className="text-coral"> matched</span>}
            </span>
          </div>
        ))}
      </div>
    </section>
  )
}
