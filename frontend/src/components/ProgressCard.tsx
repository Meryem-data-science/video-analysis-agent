import type { Job } from "../types";
import { NoticePanel } from "./NoticePanel";

const labels: Record<string, string> = {
  queued: "En attente",
  running: "Analyse en cours",
  completed: "Traitement terminé",
  failed: "Analyse interrompue",
  cancelled: "Job annulé",
};

interface Props {
  job: Job;
  onCancel: () => Promise<void>;
}

export function ProgressCard({ job, onCancel }: Props) {
  const progress = Math.min(100, Math.max(0, job.progress_percent || 0));
  return (
    <section className="progress-card card" aria-live="polite">
      <div className="progress-topline">
        <div>
          <span className={`status-pill status-${job.status}`}>
            <span className="status-dot" />
            {labels[job.status] || job.status}
          </span>
          <h2>{job.current_step ? job.current_step.replaceAll("_", " ") : "Préparation"}</h2>
          <p>{job.message || (job.status === "queued" ? "Votre analyse est dans la file locale." : "Mise à jour du pipeline…")}</p>
        </div>
        <strong className="progress-number">{Math.round(progress)}<small>%</small></strong>
      </div>
      <div
        className="progress-track"
        role="progressbar"
        aria-label="Progression de l’analyse"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progress}
      >
        <span style={{ width: `${progress}%` }} />
      </div>
      <div className="progress-footer">
        <small>Job {job.job_id.slice(0, 12)}…</small>
        {job.status === "queued" && (
          <button className="button button-danger-ghost" type="button" onClick={onCancel}>
            Annuler le job
          </button>
        )}
      </div>
      <NoticePanel title="Erreur du pipeline" tone="error" issues={job.errors || []} />
      <NoticePanel title="Points d’attention" tone="warning" issues={job.warnings || []} />
    </section>
  );
}
