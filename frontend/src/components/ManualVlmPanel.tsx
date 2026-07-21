import { useRef, useState } from "react";
import {
  ApiClientError,
  artifactUrl,
  buildReport,
  importVlmResults,
  loadDemoFixtures,
  safeMessage,
  validateRun,
} from "../api";
import type { Artifact, ReportResponse, ValidationResponse } from "../types";
import { NoticePanel } from "./NoticePanel";

interface Props {
  runId: string;
  artifacts: Artifact[];
  onReportBuilt: (response: ReportResponse) => Promise<void>;
  demoMode?: boolean;
}

const count = (value?: number | number[]) =>
  Array.isArray(value) ? value.length : typeof value === "number" ? value : 0;

export function ManualVlmPanel({ runId, artifacts, onReportBuilt, demoMode = false }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const manualArtifacts = artifacts.filter((item) => {
    const value = `${item.name} ${item.category} ${item.filename}`.toLowerCase();
    return value.includes("vlm_pack") || value.includes("prompt");
  });

  async function parseFiles() {
    const results: Record<string, unknown>[] = [];
    for (const file of files) {
      let value: unknown;
      try {
        value = JSON.parse(await file.text());
      } catch {
        throw new ApiClientError(`Le fichier ${file.name} ne contient pas un JSON lisible.`, "json_invalid");
      }
      if (Array.isArray(value)) {
        results.push(...value.filter((item): item is Record<string, unknown> => !!item && typeof item === "object"));
      } else if (value && typeof value === "object" && Array.isArray((value as { results?: unknown }).results)) {
        results.push(...(value as { results: Record<string, unknown>[] }).results);
      } else if (value && typeof value === "object") {
        results.push(value as Record<string, unknown>);
      } else {
        throw new ApiClientError(`La structure de ${file.name} n’est pas acceptée.`, "json_invalid");
      }
    }
    if (!results.length) throw new ApiClientError("Aucun résultat JSON exploitable.", "json_empty");
    return results;
  }

  const perform = async (name: string, action: () => Promise<void>) => {
    setBusy(name);
    setError(null);
    try {
      await action();
    } catch (reason) {
      setError(safeMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="manual-panel card" aria-labelledby="manual-title">
      <div className="section-heading horizontal">
        <div>
          <span className="eyebrow">Mode manuel</span>
          <h2 id="manual-title">Compléter l’analyse VLM</h2>
          <p>Téléchargez les entrées, analysez-les avec le VLM de votre choix puis réimportez ses JSON.</p>
        </div>
        <span className="step-badge">3 étapes</span>
      </div>
      <div className="manual-grid">
        <article>
          <span className="step-number">01</span>
          <h3>Récupérer les packs</h3>
          <p>Les prompts contiennent le schéma exact attendu pour chaque segment.</p>
          <div className="artifact-links">
            {manualArtifacts.length ? manualArtifacts.map((item) => (
              <a key={item.name} href={artifactUrl(item.url)} target="_blank" rel="noreferrer">
                {item.filename} <span>↗</span>
              </a>
            )) : <small>Aucun pack déclaré.</small>}
          </div>
        </article>
        <article>
          <span className="step-number">02</span>
          <h3>Importer les résultats</h3>
          <p>Un ou plusieurs fichiers JSON. Leur contenu sera revalidé côté serveur.</p>
          <input
            ref={inputRef}
            className="sr-only"
            type="file"
            accept=".json,application/json"
            multiple
            aria-label="Importer des résultats VLM"
            onChange={(event) => setFiles(Array.from(event.target.files || []))}
          />
          <button className="button button-secondary" type="button" onClick={() => inputRef.current?.click()}>
            Choisir les JSON
          </button>
          {files.length > 0 && <small className="file-selection">{files.length} fichier(s) sélectionné(s)</small>}
          <button
            className="button button-primary compact"
            type="button"
            disabled={!files.length || !!busy}
            onClick={() => perform("import", async () => {
              const results = await parseFiles();
              setValidation(await importVlmResults(runId, results));
            })}
          >
            {busy === "import" ? "Import…" : "Envoyer les résultats"}
          </button>
        </article>
        <article>
          <span className="step-number">03</span>
          <h3>Valider et construire</h3>
          <p>Le rapport ne sera complet que si tous les segments sont valides.</p>
          <button
            className="button button-secondary"
            type="button"
            disabled={!!busy}
            onClick={() => perform("validate", async () => setValidation(await validateRun(runId)))}
          >
            {busy === "validate" ? "Validation…" : "Valider les segments"}
          </button>
          <button
            className="button button-primary compact"
            type="button"
            disabled={!!busy}
            onClick={() => perform("report", async () => onReportBuilt(await buildReport(runId)))}
          >
            {busy === "report" ? "Construction…" : "Construire le rapport"}
          </button>
        </article>
      </div>
      {demoMode && (
        <div className="simulated-fixture-box">
          <div>
            <span className="source-simulated">Fixture VLM simulée</span>
            <strong>Afficher une timeline enrichie sans appeler de modèle</strong>
            <p>Ces réponses sont écrites à la main et liées au hash de la vidéo synthétique.</p>
          </div>
          <button
            className="button button-demo"
            type="button"
            disabled={!!busy}
            onClick={() => perform("fixtures", async () => {
              setValidation(await loadDemoFixtures(runId));
            })}
          >
            {busy === "fixtures" ? "Chargement…" : "Charger les résultats de démonstration"}
          </button>
        </div>
      )}
      {validation && (
        <div className="validation-summary" aria-label="Résumé de validation">
          <div><strong>{count(validation.valid_segments)}</strong><span>valides</span></div>
          <div><strong>{count(validation.invalid_segments)}</strong><span>invalides</span></div>
          <div><strong>{count(validation.missing_segments)}</strong><span>manquants</span></div>
          <span className={`status-pill status-${validation.status}`}>{validation.status}</span>
        </div>
      )}
      {validation?.status === "partial" && (
        <NoticePanel title="Validation partielle" tone="warning">
          <p>Corrigez les segments invalides ou manquants avant de reconstruire le rapport.</p>
        </NoticePanel>
      )}
      {error && <NoticePanel title="Résultat VLM refusé" tone="error"><p>{error}</p></NoticePanel>}
    </section>
  );
}
