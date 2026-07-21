import { artifactUrl } from "../api";
import type { EvaluationResponse } from "../types";

interface Props {
  evaluation: EvaluationResponse;
}

const percent = (value: number | null | undefined) =>
  typeof value === "number" ? `${Math.round(value * 100)} %` : "—";

export function EvaluationPanel({ evaluation }: Props) {
  const { report, links } = evaluation;
  const { summary, scene_metrics: scenes } = report;
  return (
    <section className="evaluation-panel card" aria-labelledby="evaluation-title">
      <div className="section-heading horizontal">
        <div>
          <span className="eyebrow">Évaluation reproductible</span>
          <h2 id="evaluation-title">Qualité locale mesurée</h2>
          <p>Comparaison du run avec les annotations de référence du scénario synthétique.</p>
        </div>
        <div className="source-legend">
          <span className="source-calculated">Calculé localement</span>
          <span className="source-annotation">Annotation de référence</span>
          {report.sources.vlm_results === "simulated_fixture" && (
            <span className="source-simulated">Fixture VLM simulée</span>
          )}
        </div>
      </div>
      <div className="evaluation-grid">
        <article><small>Temps total</small><strong>{summary.total_elapsed_s?.toFixed(2) ?? "—"} s</strong></article>
        <article><small>Scènes attendues</small><strong>{summary.expected_scenes}</strong></article>
        <article><small>Scènes détectées</small><strong>{summary.detected_scenes}</strong></article>
        <article><small>F1 des limites</small><strong>{percent(summary.boundary_f1)}</strong></article>
        <article><small>Couverture</small><strong>{percent(summary.temporal_coverage_ratio)}</strong></article>
        <article><small>Erreurs de limites</small><strong>{scenes.unmatched_expected_boundaries + scenes.unmatched_detected_boundaries}</strong></article>
      </div>
      <div className="evaluation-footer">
        <p>
          Les scores de limites et les comptages sont déterministes. Les rapprochements
          textuels restent lexicaux et approximatifs.
        </p>
        <div>
          {links.json && <a className="button button-secondary" href={artifactUrl(links.json)} target="_blank" rel="noreferrer">Rapport JSON ↗</a>}
          {links.html && <a className="button button-primary" href={artifactUrl(links.html)} target="_blank" rel="noreferrer">Rapport HTML ↗</a>}
        </div>
      </div>
    </section>
  );
}
