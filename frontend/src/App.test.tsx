import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const RUN_ID = "a".repeat(64);

function json(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  }));
}

function completedBundle(path: string, status = "awaiting_vlm") {
  if (path.endsWith(`/api/runs/${RUN_ID}`)) {
    return json({
      run_id: RUN_ID,
      status,
      video_metadata: { duration_s: 18.4, width: 1080, height: 1920 },
      artifacts: {},
      errors: [],
      warnings: [],
    });
  }
  if (path.endsWith(`/api/runs/${RUN_ID}/scenes`)) {
    return json({
      source: "segmentation",
      scenes: [{
        segment_id: 1,
        start_s: 0,
        end_s: 6.2,
        duration_s: 6.2,
        combined_text: "Parole locale de test",
      }],
    });
  }
  if (path.endsWith(`/api/runs/${RUN_ID}/report`)) {
    return json({ run_id: RUN_ID, status, report: null });
  }
  if (path.endsWith(`/api/runs/${RUN_ID}/artifacts`)) {
    return json({
      artifacts: [
        {
          name: "vlm_pack",
          category: "vlm_pack",
          filename: "vlm_pack.json",
          media_type: "application/json",
          size_bytes: 400,
          url: `/api/runs/${RUN_ID}/artifacts/vlm_pack`,
        },
        {
          name: "prompt_segment_001",
          category: "prompts",
          filename: "prompt_segment_001.md",
          media_type: "text/markdown",
          size_bytes: 200,
          url: `/api/runs/${RUN_ID}/artifacts/prompt_segment_001`,
        },
      ],
    });
  }
  if (path.includes("/artifacts/vlm_pack")) return json({ segments: [] });
  return null;
}

function uploadVideo() {
  const file = new File(["video"], "demo.mp4", { type: "video/mp4" });
  fireEvent.change(screen.getByLabelText("Sélectionner une vidéo"), {
    target: { files: [file] },
  });
  return file;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Video Analysis Agent UI", () => {
  it("affiche l’upload, l’aperçu et les trois modes avec manual par défaut", async () => {
    render(<App />);
    const file = uploadVideo();
    expect(await screen.findByText("demo.mp4")).toBeInTheDocument();
    expect(screen.getByText("0.0 Ko")).toBeInTheDocument();
    expect(URL.createObjectURL).toHaveBeenCalledWith(file);
    expect(screen.getByRole("radio", { name: /Manuel/ })).toBeChecked();

    await userEvent.click(screen.getByRole("radio", { name: /Automatique/ }));
    expect(screen.getByLabelText("Fournisseur VLM")).toHaveValue("gemini");
    await userEvent.click(screen.getByRole("radio", { name: /Local uniquement/ }));
    expect(screen.queryByLabelText("Fournisseur VLM")).not.toBeInTheDocument();
  });

  it("crée un job queued en mode manual puis permet son annulation", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith("/api/jobs") && init?.method === "POST") {
        expect((init.body as FormData).get("mode")).toBe("manual");
        return json({ job_id: "job-queued", status: "queued", progress_percent: 0 });
      }
      if (path.endsWith("/api/jobs/job-queued/cancel")) {
        return json({ job_id: "job-queued", status: "cancelled", progress_percent: 0 });
      }
      return json({ job_id: "job-queued", status: "queued", progress_percent: 0 });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    uploadVideo();
    await userEvent.click(screen.getByRole("button", { name: "Lancer l’analyse" }));
    expect(await screen.findByText("En attente")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Annuler le job" }));
    expect(await screen.findByText("Job annulé")).toBeInTheDocument();
  });

  it("suit queued → running → completed, charge le run puis arrête le polling", async () => {
    let polls = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith("/api/jobs") && init?.method === "POST") {
        return json({ job_id: "job-ok", status: "queued", progress_percent: 0 });
      }
      if (path.endsWith("/api/jobs/job-ok")) {
        polls += 1;
        return polls === 1
          ? json({ job_id: "job-ok", status: "running", progress_percent: 42, current_step: "segmentation", message: "Scènes détectées." })
          : new Promise<Response>((resolve) => setTimeout(() => {
            void json({
              job_id: "job-ok",
              status: "completed",
              progress_percent: 100,
              run_id: RUN_ID,
              pipeline_status: "awaiting_vlm",
            }).then(resolve);
          }, 35));
      }
      return completedBundle(path) || json({});
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    uploadVideo();
    await userEvent.click(screen.getByRole("button", { name: "Lancer l’analyse" }));
    expect(await screen.findByText("Analyse en cours")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
    expect(await screen.findByText("Votre analyse est prête à consulter.")).toBeInTheDocument();
    expect(screen.getByText("Analyse locale terminée — résultats VLM en attente.")).toBeInTheDocument();
    const stoppedAt = polls;
    await new Promise((resolve) => setTimeout(resolve, 40));
    expect(polls).toBe(stoppedAt);
  });

  it("affiche un échec nettoyé sans stack trace ni chemin local", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith("/api/jobs") && init?.method === "POST") {
        return json({ job_id: "job-fail", status: "queued", progress_percent: 0 });
      }
      return json({
        job_id: "job-fail",
        status: "failed",
        progress_percent: 20,
        errors: [{
          code: "pipeline_failed",
          message: "Échec sûr",
          details: "C:\\Users\\alice\\secret\\video.mp4",
        }],
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    uploadVideo();
    await userEvent.click(screen.getByRole("button", { name: "Lancer l’analyse" }));
    expect(await screen.findByText("Analyse interrompue")).toBeInTheDocument();
    expect(screen.getByText("Échec sûr")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("C:\\Users");
    expect(document.body.textContent).not.toContain("at pipeline");
  });

  it("affiche une erreur API structurée de fichier refusé", async () => {
    vi.stubGlobal("fetch", vi.fn(() => json({
      error: {
        code: "upload_extension_invalid",
        message: "Ce type de fichier est refusé.",
        recommended_action: "Choisissez un MP4.",
      },
    }, 415)));
    render(<App />);
    uploadVideo();
    await userEvent.click(screen.getByRole("button", { name: "Lancer l’analyse" }));
    expect(await screen.findByText(/Ce type de fichier est refusé/)).toBeInTheDocument();
    expect(screen.getByText(/Choisissez un MP4/)).toBeInTheDocument();
  });

  it("importe des JSON manuels et affiche une validation partielle", async () => {
    let polls = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith("/api/jobs") && init?.method === "POST") {
        return json({ job_id: "job-manual", status: "queued", progress_percent: 0 });
      }
      if (path.endsWith("/api/jobs/job-manual")) {
        polls += 1;
        return json({ job_id: "job-manual", status: "completed", progress_percent: 100, run_id: RUN_ID });
      }
      if (path.endsWith("/vlm-results")) {
        expect(init?.method).toBe("POST");
        return json({ status: "partial", valid_segments: [1], invalid_segments: [2], missing_segments: [3] });
      }
      return completedBundle(path) || json({});
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    uploadVideo();
    await userEvent.click(screen.getByRole("button", { name: "Lancer l’analyse" }));
    await screen.findByRole("heading", { name: "Compléter l’analyse VLM" });

    const resultFile = new File([JSON.stringify({ segment_id: 1 })], "result.json", { type: "application/json" });
    Object.defineProperty(resultFile, "text", { value: () => Promise.resolve(JSON.stringify({ segment_id: 1 })) });
    fireEvent.change(screen.getByLabelText("Importer des résultats VLM"), {
      target: { files: [resultFile] },
    });
    await userEvent.click(screen.getByRole("button", { name: "Envoyer les résultats" }));
    expect(await screen.findByText("Validation partielle")).toBeInTheDocument();
    expect(screen.getByLabelText("Résumé de validation")).toHaveTextContent("1valides");
    expect(screen.getByLabelText("Résumé de validation")).toHaveTextContent("1invalides");
  });

  it("construit un rapport complete uniquement après la réponse API complete", async () => {
    let reportComplete = false;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith("/api/jobs") && init?.method === "POST") {
        return json({ job_id: "job-report", status: "queued", progress_percent: 0 });
      }
      if (path.endsWith("/api/jobs/job-report")) {
        return json({ job_id: "job-report", status: "completed", progress_percent: 100, run_id: RUN_ID });
      }
      if (path.endsWith("/build-report")) {
        reportComplete = true;
        return json({ run_id: RUN_ID, status: "complete", report: { status: "complete" } });
      }
      const response = completedBundle(path, reportComplete ? "complete" : "awaiting_vlm");
      return response || json({});
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    uploadVideo();
    await userEvent.click(screen.getByRole("button", { name: "Lancer l’analyse" }));
    await screen.findByRole("heading", { name: "Compléter l’analyse VLM" });
    expect(screen.queryByText("complete")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Construire le rapport" }));
    expect(await screen.findByText("complete")).toBeInTheDocument();
  });

  it("lance la démo locale, affiche l’évaluation et distingue les fixtures simulées", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith("/api/demo/jobs") && init?.method === "POST") {
        expect((init.body as FormData).get("mode")).toBe("manual");
        return json({
          job_id: "demo-job",
          status: "queued",
          progress_percent: 0,
          demo: true,
          network_used: false,
        });
      }
      if (path.endsWith("/api/jobs/demo-job")) {
        return json({
          job_id: "demo-job",
          status: "completed",
          progress_percent: 100,
          run_id: RUN_ID,
          pipeline_status: "awaiting_vlm",
        });
      }
      if (path.endsWith(`/api/demo/runs/${RUN_ID}/evaluate`)) {
        return json({
          run_id: RUN_ID,
          links: {
            json: `/api/runs/${RUN_ID}/evaluation`,
            html: `/api/runs/${RUN_ID}/evaluation/html`,
          },
          report: {
            schema_version: "1.0",
            run_id: RUN_ID,
            sources: {
              pipeline_outputs: "calculated",
              annotations: "reference",
              vlm_results: "absent",
            },
            summary: {
              status: "awaiting_vlm",
              expected_scenes: 3,
              detected_scenes: 3,
              boundary_f1: 1,
              temporal_coverage_ratio: 1,
              frame_count: 24,
              artifact_size_bytes: 1000,
              error_count: 0,
              warning_count: 0,
              total_elapsed_s: 4.2,
            },
            scene_metrics: {
              unmatched_expected_boundaries: 0,
              unmatched_detected_boundaries: 0,
              gap_count: 0,
              overlap_count: 0,
            },
          },
        });
      }
      if (path.endsWith(`/api/demo/runs/${RUN_ID}/vlm-fixtures`)) {
        return json({
          status: "complete",
          valid_segments: [1, 2, 3],
          invalid_segments: [],
          missing_segments: [],
          simulated: true,
          source: "demo_fixture",
        });
      }
      return completedBundle(path) || json({});
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Lancer la démonstration" }));
    expect(await screen.findByText("Mode démonstration local")).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Qualité locale mesurée" })).toBeInTheDocument();
    expect(screen.getByText("Calculé localement")).toBeInTheDocument();
    expect(screen.getByText("Annotation de référence")).toBeInTheDocument();
    expect(screen.getByText("Fixture VLM simulée")).toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", { name: "Charger les résultats de démonstration" }),
    );
    expect(await screen.findByLabelText("Résumé de validation")).toHaveTextContent("3valides");
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining(`/api/demo/runs/${RUN_ID}/vlm-fixtures`),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("nettoie le polling et les URLs d’aperçu au démontage", async () => {
    let resolvePoll: ((value: Response) => void) | undefined;
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith("/api/jobs") && init?.method === "POST") {
        return json({ job_id: "job-pending", status: "queued", progress_percent: 0 });
      }
      return new Promise<Response>((resolve) => { resolvePoll = resolve; });
    });
    vi.stubGlobal("fetch", fetchMock);
    const view = render(<App />);
    uploadVideo();
    await userEvent.click(screen.getByRole("button", { name: "Lancer l’analyse" }));
    await waitFor(() => expect(resolvePoll).toBeDefined());
    view.unmount();
    resolvePoll?.(new Response(JSON.stringify({ status: "running" }), {
      headers: { "Content-Type": "application/json" },
    }));
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:preview");
  });
});
