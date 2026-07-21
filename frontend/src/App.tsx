import { useEffect, useState } from "react";
import {
  ApiClientError,
  cancelJob,
  createDemoJob,
  createJob,
  evaluateDemoRun,
  getRunBundle,
  safeMessage,
} from "./api";
import { Header } from "./components/Header";
import { NoticePanel } from "./components/NoticePanel";
import { ProgressCard } from "./components/ProgressCard";
import { ResultDashboard } from "./components/ResultDashboard";
import { UploadPanel } from "./components/UploadPanel";
import { useJobPolling } from "./hooks/useJobPolling";
import type { EvaluationResponse, Job, RunBundle, VlmMode } from "./types";

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [initialJob, setInitialJob] = useState<Job | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<RunBundle | null>(null);
  const [loadingResults, setLoadingResults] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationAttempted, setEvaluationAttempted] = useState(false);
  const { job, setJob, pollError } = useJobPolling(jobId, initialJob);

  const loadResults = async (runId: string) => {
    setLoadingResults(true);
    setError(null);
    try {
      setBundle(await getRunBundle(runId));
    } catch (reason) {
      setError(safeMessage(reason));
    } finally {
      setLoadingResults(false);
    }
  };

  useEffect(() => {
    if (job?.status === "completed" && job.run_id && bundle?.run.run_id !== job.run_id) {
      void loadResults(job.run_id);
    }
  }, [job?.status, job?.run_id, bundle?.run.run_id]);

  useEffect(() => {
    if (!demoMode || !bundle || evaluation || evaluationLoading || evaluationAttempted) return;
    setEvaluationAttempted(true);
    setEvaluationLoading(true);
    evaluateDemoRun(bundle.run.run_id)
      .then(setEvaluation)
      .catch((reason) => setError(safeMessage(reason)))
      .finally(() => setEvaluationLoading(false));
  }, [demoMode, bundle, evaluation, evaluationLoading, evaluationAttempted]);

  const submit = async (file: File, mode: VlmMode, provider?: string) => {
    setSubmitting(true);
    setError(null);
    setBundle(null);
    setDemoMode(false);
    setEvaluation(null);
    setEvaluationAttempted(false);
    setJobId(null);
    setInitialJob(null);
    try {
      const created = await createJob(file, mode, provider);
      setInitialJob(created);
      setJobId(created.job_id);
    } catch (reason) {
      const message = reason instanceof ApiClientError && reason.action
        ? `${reason.message} ${reason.action}`
        : safeMessage(reason);
      setError(message);
    } finally {
      setSubmitting(false);
    }
  };

  const submitDemo = async () => {
    setSubmitting(true);
    setError(null);
    setBundle(null);
    setEvaluation(null);
    setEvaluationAttempted(false);
    setJobId(null);
    setInitialJob(null);
    setDemoMode(true);
    try {
      const created = await createDemoJob("manual");
      setInitialJob(created);
      setJobId(created.job_id);
    } catch (reason) {
      setDemoMode(false);
      setError(safeMessage(reason));
    } finally {
      setSubmitting(false);
    }
  };

  const cancel = async () => {
    if (!jobId) return;
    try {
      setJob(await cancelJob(jobId));
    } catch (reason) {
      setError(safeMessage(reason));
    }
  };

  return (
    <>
      <Header />
      <main>
        <section className="intro">
          <div>
            <span className="eyebrow light">Comprendre chaque séquence</span>
            <h1>De la vidéo brute à une analyse <em>structurée.</em></h1>
          </div>
          <p>
            Une expérience locale, lisible et vérifiable pour explorer les scènes,
            l’audio et les résultats VLM — sans ligne de commande.
          </p>
        </section>

        <div className="workspace">
          {!jobId && (
            <UploadPanel busy={submitting} onSubmit={submit} onDemo={submitDemo} />
          )}
          {demoMode && jobId && (
            <div className="demo-mode-banner" role="status">
              <strong>Mode démonstration local</strong>
              <span>Aucune donnée envoyée vers un service externe.</span>
            </div>
          )}
          {error && <NoticePanel title="Action impossible" tone="error"><p>{error}</p></NoticePanel>}
          {pollError && (
            <NoticePanel title="Connexion momentanément interrompue" tone="warning">
              <p>{safeMessage(pollError)} Une nouvelle tentative sera effectuée.</p>
            </NoticePanel>
          )}
          {job && <ProgressCard job={job} onCancel={cancel} />}
          {loadingResults && <div className="loading-card card">Chargement des résultats du run…</div>}
          {bundle && (
            <ResultDashboard
              bundle={bundle}
              onRefresh={() => loadResults(bundle.run.run_id)}
              demoMode={demoMode}
              evaluation={evaluation}
            />
          )}
          {evaluationLoading && (
            <div className="evaluation-loading">Calcul des métriques locales…</div>
          )}
          {job && ["failed", "cancelled"].includes(job.status) && (
            <button
              className="button button-secondary restart-button"
              type="button"
              onClick={() => {
                setJobId(null);
                setInitialJob(null);
                setBundle(null);
                setDemoMode(false);
                setEvaluation(null);
                setEvaluationAttempted(false);
                setError(null);
              }}
            >
              Démarrer une nouvelle analyse
            </button>
          )}
        </div>
      </main>
      <footer>
        <span>Video Analysis Agent · 0.1.0</span>
        <span>Données conservées sur votre machine en modes manual et disabled.</span>
      </footer>
    </>
  );
}
