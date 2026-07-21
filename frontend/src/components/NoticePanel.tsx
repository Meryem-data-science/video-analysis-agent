import type { ApiIssue } from "../types";

interface Props {
  title: string;
  tone?: "error" | "warning" | "info" | "success";
  issues?: Array<ApiIssue | string>;
  children?: React.ReactNode;
}

export function NoticePanel({ title, tone = "info", issues = [], children }: Props) {
  if (!children && issues.length === 0) return null;
  return (
    <section className={`notice notice-${tone}`} role={tone === "error" ? "alert" : "status"}>
      <strong>{title}</strong>
      {children}
      {issues.length > 0 && (
        <ul>
          {issues.map((issue, index) => (
            <li key={`${typeof issue === "string" ? issue : issue.code}-${index}`}>
              {typeof issue === "string" ? issue : issue.message || issue.code || "Diagnostic"}
              {typeof issue !== "string" && issue.step ? ` — étape ${issue.step}` : ""}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
