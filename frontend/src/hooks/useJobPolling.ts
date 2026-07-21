import { useEffect, useRef, useState } from "react";
import { getJob } from "../api";
import type { Job } from "../types";

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

export function useJobPolling(
  jobId: string | null,
  initialJob: Job | null,
  intervalMs = import.meta.env.MODE === "test" ? 10 : 1200,
) {
  const [job, setJob] = useState<Job | null>(initialJob);
  const [pollError, setPollError] = useState<Error | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    setJob(initialJob);
  }, [initialJob]);

  useEffect(() => {
    if (!jobId) return undefined;
    let active = true;

    const poll = async () => {
      controller.current = new AbortController();
      try {
        const next = await getJob(jobId, controller.current.signal);
        if (!active) return;
        setJob((previous) => {
          if (!previous) return next;
          if (TERMINAL.has(previous.status)) return previous;
          return {
            ...next,
            progress_percent: Math.max(previous.progress_percent, next.progress_percent),
          };
        });
        setPollError(null);
        if (!TERMINAL.has(next.status)) {
          timer.current = setTimeout(poll, intervalMs);
        }
      } catch (error) {
        if (!active || (error instanceof DOMException && error.name === "AbortError")) return;
        setPollError(error instanceof Error ? error : new Error("Polling interrompu."));
        timer.current = setTimeout(poll, intervalMs * 2);
      }
    };

    timer.current = setTimeout(poll, 0);
    return () => {
      active = false;
      if (timer.current) clearTimeout(timer.current);
      controller.current?.abort();
    };
  }, [jobId, intervalMs]);

  return { job, setJob, pollError };
}
