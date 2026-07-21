import { useState } from "react";
import { artifactUrl } from "../api";
import type { EvaluationResponse, ReportResponse, RunBundle } from "../types";
import { EvaluationPanel } from "./EvaluationPanel";
import { ManualVlmPanel } from "./ManualVlmPanel";
import { NoticePanel } from "./NoticePanel";
import { SceneTimeline } from "./SceneTimeline";

interface Props {
  bundle: RunBundle;
  onRefresh: () => Promise<void>;
  demoMode?: boolean;
  evaluation?: EvaluationResponse | null;
}

const displayValue = (value: unknown) => {
  if (typeof value === "number") return Number.isInteger(value) ? `${value}` : value.toFixed(2);
  if (typeof value === "boolean") return value ? "Oui" : "Non";
  if (typeof value === "string") return value;
  return "—";
};

function transcriptText(value?: Record<string, unknown>) {
  if (!value) return null;
  if (typeof value.text === "string") return value.text;
  if (Array.isArray(value.segments)) {
    return value.segments
      .map((item) => item && typeof item === "object" ? (item as Record<string, unknown>).text : "")
      .filter((item): item is string => typeof item === "string")
      .join(" ");
  }
  return null;
}

export function ResultDashboard({
  bundle, onRefresh, demoMode = false, evaluation = null,
}: Props) {
  const [copied, setCopied] = useState(false);
  const [reportOverride, setReportOverride] = useState<ReportResponse | null>(null);
  const report = reportOverride || bundle.report;
  const status = report.status || bundle.run.status;
  const metadata = bundle.run.video_metadata || {};
  const transcript = transcriptText(bundle.transcription);
  const duration = typeof metadata.duration_s === "number" ? `${metadata.duration_s.toFixed(1)} s` : "—";
  const resolution =
    metadata.width && metadata.height ? `${displayValue(metadata.width)} × ${displayValue(metadata.height)}` : "—";

  const copyRun = async () => {
    await navigator.clipboard.writeText(bundle.run.run_id);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <div className="results">
      <section className="result-hero">
        <div>
          <span className={`status-pill status-${status}`}>{status}</span>
          <h2>Votre analyse est prête à consulter.</h2>
          <div className="run-id">
            Run <code>{bundle.run.run_id.slice(0, 12)}…</code>
            <button type="button" onClick={copyRun} aria-label="Copier le run ID">{copied ? "Copié" : "Copier"}</button>
          </div>
        </div>
        <div className="result-actions">
          <a className="button button-secondary" href={artifactUrl(`/api/runs/${bundle.run.run_id}`)} target="_blank" rel="noreferrer">
            Manifest ↗
          </a>
          <a className="button button-primary" href={artifactUrl(`/api/runs/${bundle.run.run_id}/report`)} target="_blank" rel="noreferrer">
            Rapport ↗
          </a>
        </div>
      </section>

      {status === "partial" && (
        <NoticePanel title="Rapport partiel" tone="warning">
          <p>Des segments doivent encore être corrigés. Le statut affiché reste celui retourné par l’API.</p>
        </NoticePanel>
      )}
      {status === "awaiting_vlm" && (
        <NoticePanel title="Analyse visuelle en attente" tone="info">
          <p>Les étapes locales sont terminées. Utilisez le panneau manuel pour compléter l’analyse.</p>
        </NoticePanel>
      )}

      <section className="stat-grid" aria-label="Statistiques du run">
        <article><span>Durée</span><strong>{duration}</strong><small>vidéo analysée</small></article>
        <article><span>Scènes</span><strong>{bundle.scenes.length}</strong><small>segments détectés</small></article>
        <article><span>Résolution</span><strong>{resolution}</strong><small>source vidéo</small></article>
        <article><span>Artefacts</span><strong>{bundle.artifacts.length}</strong><small>fichiers consultables</small></article>
      </section>

      {evaluation && <EvaluationPanel evaluation={evaluation} />}

      {(transcript || bundle.audio) && (
        <section className="insight-grid">
          <article className="card">
            <span className="eyebrow">Transcription</span>
            <h2>Paroles détectées</h2>
            <p>{transcript || "Aucun texte transcrit n’est disponible."}</p>
          </article>
          <article className="card">
            <span className="eyebrow">Audio</span>
            <h2>Signal local</h2>
            <dl>
              {Object.entries(bundle.audio || {}).slice(0, 6).map(([key, value]) => (
                <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{displayValue(value)}</dd></div>
              ))}
            </dl>
          </article>
        </section>
      )}

      <SceneTimeline
        scenes={bundle.scenes}
        artifacts={bundle.artifacts}
        awaitingVlm={status === "awaiting_vlm"}
      />

      {(status === "awaiting_vlm" || status === "partial" || status === "failed") && (
        <ManualVlmPanel
          runId={bundle.run.run_id}
          artifacts={bundle.artifacts}
          onReportBuilt={async (response) => {
            setReportOverride(response);
            await onRefresh();
          }}
          demoMode={demoMode}
        />
      )}

      <section className="artifact-section card">
        <div className="section-heading horizontal">
          <div><span className="eyebrow">Traçabilité</span><h2>Artefacts disponibles</h2></div>
          <span className="scene-count">{bundle.artifacts.length}</span>
        </div>
        <div className="artifact-table">
          {bundle.artifacts.map((item) => (
            <a key={item.name} href={artifactUrl(item.url)} target="_blank" rel="noreferrer">
              <span className="artifact-type">{item.filename.split(".").pop()?.toUpperCase()}</span>
              <span><strong>{item.filename}</strong><small>{item.category || item.name}</small></span>
              <span>Consulter ↗</span>
            </a>
          ))}
        </div>
      </section>

      <NoticePanel title="Warnings du run" tone="warning" issues={bundle.run.warnings || []} />
      <NoticePanel title="Erreurs du run" tone="error" issues={bundle.run.errors || []} />
    </div>
  );
}
