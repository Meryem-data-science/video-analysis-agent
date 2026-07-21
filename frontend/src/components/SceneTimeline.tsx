import { artifactUrl } from "../api";
import type { Artifact, Scene } from "../types";

const time = (value?: number) => {
  if (typeof value !== "number") return "—";
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
};

function sceneImage(scene: Scene, artifacts: Artifact[]) {
  const direct = scene.keyframe || scene.key_frame;
  if (typeof direct === "string" && direct.startsWith("/api/")) return artifactUrl(direct);
  const id = Number(scene.segment_id ?? scene.scene_id);
  const tokens = Number.isFinite(id) ? [`${id}`.padStart(3, "0"), `segment_${id}`] : [];
  const match = artifacts.find((item) => {
    const haystack = `${item.name} ${item.filename}`.toLowerCase();
    return /\.(jpe?g|png|webp)$/i.test(item.filename) && tokens.some((token) => haystack.includes(token));
  });
  return match ? artifactUrl(match.url) : null;
}

interface Props {
  scenes: Scene[];
  artifacts: Artifact[];
  awaitingVlm: boolean;
}

export function SceneTimeline({ scenes, artifacts, awaitingVlm }: Props) {
  return (
    <section className="timeline-section">
      <div className="section-heading horizontal">
        <div>
          <span className="eyebrow">Découpage</span>
          <h2>Timeline des scènes</h2>
        </div>
        <span className="scene-count">{scenes.length} scène{scenes.length > 1 ? "s" : ""}</span>
      </div>
      {awaitingVlm && (
        <div className="local-banner">Analyse locale terminée — résultats VLM en attente.</div>
      )}
      {scenes.length === 0 ? (
        <div className="empty-state">Aucune scène n’est encore disponible.</div>
      ) : (
        <div className="timeline">
          {scenes.map((scene, index) => {
            const id = scene.segment_id ?? scene.scene_id ?? index + 1;
            const image = sceneImage(scene, artifacts);
            const screenText = Array.isArray(scene.texte_ecran)
              ? scene.texte_ecran.join(" · ")
              : scene.texte_ecran;
            return (
              <article className="scene-card" key={`${id}-${index}`}>
                <div className="timeline-marker"><span>{index + 1}</span></div>
                <div className="scene-media">
                  {image ? <img src={image} alt={`Image clé de la scène ${id}`} /> : <span>Image clé</span>}
                </div>
                <div className="scene-body">
                  <div className="scene-heading">
                    <div>
                      <span className="eyebrow">Scène {id}</span>
                      <h3>{time(scene.start_s)} — {time(scene.end_s)}</h3>
                    </div>
                    <span className="duration-chip">
                      {typeof scene.duration_s === "number"
                        ? `${scene.duration_s.toFixed(1)} s`
                        : typeof scene.start_s === "number" && typeof scene.end_s === "number"
                          ? `${(scene.end_s - scene.start_s).toFixed(1)} s`
                          : "durée —"}
                    </span>
                  </div>
                  <p className="scene-description">
                    {scene.scene_description || scene.description || scene.combined_text || "Description VLM en attente."}
                  </p>
                  <div className="tag-groups">
                    {scene.people_objects?.length ? <div><small>Objets & personnes</small><div className="tags">{scene.people_objects.map((item) => <span key={item}>{item}</span>)}</div></div> : null}
                    {scene.actions?.length ? <div><small>Actions</small><div className="tags accent">{scene.actions.map((item) => <span key={item}>{item}</span>)}</div></div> : null}
                  </div>
                  {screenText && <p className="screen-text"><strong>Texte détecté</strong> {screenText}</p>}
                  <div className="scene-footer">
                    {typeof scene.confidence === "number" && (
                      <span>Confiance <strong>{Math.round(scene.confidence * 100)} %</strong></span>
                    )}
                    {scene.uncertainties?.length ? (
                      <span className="uncertainty">À vérifier : {scene.uncertainties.join(" · ")}</span>
                    ) : null}
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
