export default function Nav({ onMenu }: { onMenu: () => void }) {
  return (
    <nav className="h-16 flex items-center px-12" data-testid="nav">
      <span className="text-base font-medium text-ink tracking-[-0.014em]">openPASO</span>
      <div className="ml-auto flex items-center gap-7 text-[13px] font-medium text-muted">
        <a href="https://github.com/Hereon-InstituteMS/openPASO"
           className="transition-colors duration-150 hover:text-ink2">Solvers</a>
        <a href="https://github.com/Hereon-InstituteMS/openPASO"
           className="transition-colors duration-150 hover:text-ink2">Docs</a>
        <button onClick={onMenu} aria-label="Session, model and files"
                className="w-8 h-8 rounded-[6px] border line grid place-items-center
                           transition-colors duration-150 hover:text-ink2">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"
               stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path d="M1.5 3.5h11M1.5 7h11M1.5 10.5h11" />
          </svg>
        </button>
      </div>
    </nav>
  )
}
