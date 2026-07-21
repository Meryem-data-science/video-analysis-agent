import { useEffect, useRef, useState } from "react";
import type { VlmMode } from "../types";

interface Props {
  busy: boolean;
  onSubmit: (file: File, mode: VlmMode, provider?: string) => Promise<void>;
  onDemo: () => Promise<void>;
}

const formatBytes = (bytes: number) => {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`;
  return `${(bytes / 1024 / 1024).toFixed(1)} Mo`;
};

export function UploadPanel({ busy, onSubmit, onDemo }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<VlmMode>("manual");
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const choose = (candidate?: File) => {
    if (candidate) setFile(candidate);
  };

  return (
    <section className="upload-card card" aria-labelledby="upload-title">
      <div className="section-heading">
        <span className="eyebrow">Nouvelle analyse</span>
        <h2 id="upload-title">Déposez une vidéo, on prépare le reste.</h2>
        <p>Une vidéo courte est recommandée pour une première analyse.</p>
      </div>

      <div
        className={`drop-zone ${dragging ? "is-dragging" : ""} ${file ? "has-file" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (event.currentTarget === event.target) setDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          choose(event.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          className="sr-only"
          type="file"
          accept=".mp4,.mov,.mkv,.avi,.webm,.m4v,video/*"
          aria-label="Sélectionner une vidéo"
          onChange={(event) => choose(event.target.files?.[0])}
        />
        {file ? (
          <div className="file-preview">
            {previewUrl && <video src={previewUrl} muted controls preload="metadata" />}
            <div className="file-copy">
              <span className="file-icon">▶</span>
              <div>
                <strong>{file.name}</strong>
                <small>{formatBytes(file.size)}</small>
              </div>
              <button className="button button-ghost" type="button" onClick={() => inputRef.current?.click()}>
                Remplacer
              </button>
            </div>
          </div>
        ) : (
          <button className="drop-button" type="button" onClick={() => inputRef.current?.click()}>
            <span className="upload-glyph" aria-hidden="true">↑</span>
            <strong>Glissez-déposez votre vidéo</strong>
            <span>ou cliquez pour parcourir vos fichiers</span>
            <small>MP4, MOV, MKV, AVI, WEBM ou M4V</small>
          </button>
        )}
      </div>

      <fieldset className="mode-fieldset">
        <legend>Mode d’analyse visuelle</legend>
        <div className="mode-grid">
          {([
            ["manual", "Manuel", "Prépare les packs, sans appel cloud."],
            ["disabled", "Local uniquement", "Termine après les étapes locales."],
            ["automatic", "Automatique", "Utilise un fournisseur configuré."],
          ] as const).map(([value, label, detail]) => (
            <label key={value} className={`mode-option ${mode === value ? "selected" : ""}`}>
              <input
                type="radio"
                name="vlm-mode"
                value={value}
                checked={mode === value}
                onChange={() => setMode(value)}
              />
              <span className="radio-dot" />
              <strong>{label}</strong>
              <small>{detail}</small>
            </label>
          ))}
        </div>
      </fieldset>

      {mode === "automatic" && (
        <label className="provider-field">
          <span>Fournisseur VLM</span>
          <select aria-label="Fournisseur VLM" defaultValue="gemini">
            <option value="gemini">Gemini</option>
          </select>
          <small>Une configuration et une clé API valides seront nécessaires.</small>
        </label>
      )}

      <button
        className="button button-primary launch-button"
        type="button"
        disabled={!file || busy}
        onClick={() => file && onSubmit(file, mode, mode === "automatic" ? "gemini" : undefined)}
      >
        {busy ? "Création du job…" : "Lancer l’analyse"}
        <span aria-hidden="true">→</span>
      </button>
      <div className="demo-launch">
        <div>
          <strong>Pas de vidéo sous la main ?</strong>
          <span>Générez notre scénario synthétique et suivez le même pipeline local.</span>
        </div>
        <button
          className="button button-demo"
          type="button"
          disabled={busy}
          onClick={onDemo}
        >
          Lancer la démonstration
        </button>
      </div>
      <p className="demo-privacy">
        Mode démonstration local — aucune donnée envoyée vers un service externe.
      </p>
    </section>
  );
}
