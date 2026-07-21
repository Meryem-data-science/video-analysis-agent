import type {
  Artifact,
  EvaluationResponse,
  Job,
  ReportResponse,
  RunBundle,
  RunDetails,
  Scene,
  ValidationResponse,
  VlmMode,
} from "./types";

const configuredBase = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE_URL = (configuredBase || "http://127.0.0.1:8000").replace(/\/+$/, "");

export class ApiClientError extends Error {
  code: string;
  step?: string | null;
  action?: string | null;

  constructor(message: string, code = "request_failed", step?: string | null, action?: string | null) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.step = step;
    this.action = action;
  }
}

export function safeMessage(value: unknown): string {
  const fallback = "Une erreur inattendue est survenue.";
  const source = value instanceof Error ? value.message : typeof value === "string" ? value : fallback;
  return source
    .replace(/[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]*/g, "[chemin masqué]")
    .replace(/(?:\/[^/\s]+){2,}/g, "[chemin masqué]")
    .replace(/\s+at\s+[\w$.<>]+\s*\([^)]*\)/g, "")
    .slice(0, 500);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiClientError(
      "Le serveur local est indisponible. Vérifiez qu’il est démarré sur le port configuré.",
      "server_unavailable",
    );
  }
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("json") ? await response.json() : null;
  if (!response.ok) {
    const issue = body?.error;
    throw new ApiClientError(
      safeMessage(issue?.message || `La requête a échoué (${response.status}).`),
      issue?.code || "request_failed",
      issue?.step,
      issue?.recommended_action,
    );
  }
  return body as T;
}

export function createJob(
  video: File,
  mode: VlmMode,
  provider?: string,
): Promise<Job> {
  const form = new FormData();
  form.append("video", video);
  form.append("mode", mode);
  if (mode === "automatic" && provider) form.append("provider", provider);
  return request<Job>("/api/jobs", { method: "POST", body: form });
}

export function createDemoJob(mode: "manual" | "disabled" = "manual"): Promise<Job> {
  const form = new FormData();
  form.append("mode", mode);
  return request<Job>("/api/demo/jobs", { method: "POST", body: form });
}

export const getJob = (jobId: string, signal?: AbortSignal) =>
  request<Job>(`/api/jobs/${encodeURIComponent(jobId)}`, { signal });

export const cancelJob = (jobId: string) =>
  request<Job>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });

export const getRun = (runId: string) =>
  request<RunDetails>(`/api/runs/${encodeURIComponent(runId)}`);

export const getScenes = (runId: string) =>
  request<{ source: string; scenes: Scene[] }>(`/api/runs/${encodeURIComponent(runId)}/scenes`);

export const getReport = (runId: string) =>
  request<ReportResponse>(`/api/runs/${encodeURIComponent(runId)}/report`);

export const getArtifacts = (runId: string) =>
  request<{ artifacts: Artifact[] }>(`/api/runs/${encodeURIComponent(runId)}/artifacts`);

async function getOptionalArtifact(artifacts: Artifact[], categories: string[]) {
  const artifact = artifacts.find((item) =>
    categories.some((category) =>
      `${item.category || ""} ${item.name}`.toLowerCase().includes(category),
    ),
  );
  if (!artifact || !artifact.media_type?.includes("json")) return undefined;
  try {
    return await request<Record<string, unknown>>(artifact.url);
  } catch {
    return undefined;
  }
}

export async function getRunBundle(runId: string): Promise<RunBundle> {
  const [run, scenesResponse, report, artifactResponse] = await Promise.all([
    getRun(runId),
    getScenes(runId),
    getReport(runId),
    getArtifacts(runId),
  ]);
  const artifacts = artifactResponse.artifacts || [];
  const [transcription, audio] = await Promise.all([
    getOptionalArtifact(artifacts, ["transcript", "transcription"]),
    getOptionalArtifact(artifacts, ["audio_light", "audio_features", "audio"]),
  ]);
  return {
    run,
    scenes: scenesResponse.scenes || [],
    sceneSource: scenesResponse.source,
    report,
    artifacts,
    transcription,
    audio,
  };
}

export const importVlmResults = (runId: string, results: Record<string, unknown>[]) =>
  request<ValidationResponse>(`/api/runs/${encodeURIComponent(runId)}/vlm-results`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, results }),
  });

export const validateRun = (runId: string) =>
  request<ValidationResponse>(`/api/runs/${encodeURIComponent(runId)}/validate`, {
    method: "POST",
  });

export const buildReport = (runId: string) =>
  request<ReportResponse>(`/api/runs/${encodeURIComponent(runId)}/build-report`, {
    method: "POST",
  });

export const loadDemoFixtures = (runId: string) =>
  request<ValidationResponse>(`/api/demo/runs/${encodeURIComponent(runId)}/vlm-fixtures`, {
    method: "POST",
  });

export const evaluateDemoRun = (runId: string) =>
  request<EvaluationResponse>(`/api/demo/runs/${encodeURIComponent(runId)}/evaluate`, {
    method: "POST",
  });

export function artifactUrl(path: string): string {
  return path.startsWith("http") ? path : `${API_BASE_URL}${path}`;
}
