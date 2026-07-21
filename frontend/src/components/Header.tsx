export function Header() {
  return (
    <header className="site-header">
      <a className="brand" href="/" aria-label="Video Analysis Agent — accueil">
        <span className="brand-mark" aria-hidden="true">
          <span />
        </span>
        <span>
          <strong>Video Analysis</strong>
          <small>Local-first agent</small>
        </span>
      </a>
      <div className="privacy-chip">
        <span className="pulse-dot" />
        Traitement local par défaut
      </div>
    </header>
  );
}
