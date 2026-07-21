export type VlmMode = "manual" | "disabled" | "automatic";
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type PipelineStatus =
  | "complete"
  | "partial"
  | "failed"
  | "awaiting_vlm"
  | "local_complete";

export interface ApiIssue {
  code?: string;
  message?: string;
  step?: string | null;
  details?: unknown;
  recommended_action?: string | null;
}

export interface Job {
  job_id: string;
  status: JobStatus;
  progress_percent: number;
  current_step?: string | null;
  message?: string | null;
  run_id?: string;
  run_dir?: string;
  pipeline_status?: PipelineStatus;
  artifacts?: Record<string, string>;
  links?: Record<string, string>;
  errors?: ApiIssue[];
  warnings?: string[] | ApiIssue[];
  status_url?: string;
}

export interface Artifact {
  name: string;
  category?: string;
  filename: string;
  media_type?: string;
  size_bytes?: number;
  url: string;
}

export interface Scene {
  segment_id?: number | string;
  scene_id?: number | string;
  start_s?: number;
  end_s?: number;
  duration_s?: number;
  keyframe?: string;
  key_frame?: string;
  scene_description?: string;
  description?: string;
  people_objects?: string[];
  actions?: string[];
  texte_ecran?: string[] | string;
  confidence?: number;
  uncertainties?: string[];
  combined_text?: string;
  [key: string]: unknown;
}

export interface RunDetails {
  run_id: string;
  status: PipelineStatus;
  current_step?: string | null;
  video_metadata?: Record<string, unknown>;
  configuration?: Record<string, unknown>;
  artifacts?: Record<string, string>;
  errors?: ApiIssue[];
  warnings?: string[] | ApiIssue[];
}

export interface ReportResponse {
  run_id: string;
  status: PipelineStatus;
  report: Record<string, unknown> | null;
  next_step?: unknown;
}

export interface ValidationResponse {
  status: PipelineStatus;
  expected_segments?: number | number[];
  valid_segments?: number | number[];
  missing_segments?: number[];
  invalid_segments?: number[];
  extra_segments?: number[];
  errors?: unknown;
  warnings?: unknown;
  next_step?: unknown;
}

export interface RunBundle {
  run: RunDetails;
  scenes: Scene[];
  sceneSource: string;
  report: ReportResponse;
  artifacts: Artifact[];
  transcription?: Record<string, unknown>;
  audio?: Record<string, unknown>;
}

export interface EvaluationReport {
  schema_version: string;
  run_id: string;
  sources: {
    pipeline_outputs: string;
    annotations: string;
    vlm_results: string;
  };
  summary: {
    status: PipelineStatus;
    expected_scenes: number;
    detected_scenes: number;
    boundary_f1: number;
    temporal_coverage_ratio: number | null;
    frame_count: number;
    artifact_size_bytes: number;
    error_count: number;
    warning_count: number;
    total_elapsed_s: number | null;
  };
  scene_metrics: {
    unmatched_expected_boundaries: number;
    unmatched_detected_boundaries: number;
    gap_count: number;
    overlap_count: number;
  };
  methodology?: {
    reliable?: string[];
    approximate?: string[];
  };
}

export interface EvaluationResponse {
  run_id: string;
  report: EvaluationReport;
  links: { json?: string; html?: string };
}
