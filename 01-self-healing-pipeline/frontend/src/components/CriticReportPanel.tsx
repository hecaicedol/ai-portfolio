"use client";

import type { CriticReport, PrincipleScore } from "@/lib/types";
import { cn } from "@/lib/cn";

interface Props {
  report: CriticReport | null;
  passThreshold?: number;
}

const PRINCIPLE_LABEL: Record<PrincipleScore["principle"], string> = {
  completeness: "Completeness",
  accuracy: "Accuracy",
  consistency: "Consistency",
  format: "Format compliance",
};

export function CriticReportPanel({ report, passThreshold = 0.85 }: Props) {
  return (
    <section className="rounded-2xl border border-ink-700 bg-ink-800/40 p-5">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-medium tracking-tight text-ink-100">
          Critic report
        </h2>
        {report && (
          <span
            className={cn(
              "rounded-full px-2.5 py-0.5 text-xs font-medium",
              report.passes
                ? "bg-ok/15 text-ok"
                : "bg-bad/15 text-bad",
            )}
          >
            {(report.overall_score * 100).toFixed(0)}% ·{" "}
            {report.passes ? "passed" : "below threshold"}
          </span>
        )}
      </div>

      {!report && (
        <div className="rounded-md border border-dashed border-ink-600 px-4 py-8 text-center text-xs text-ink-400">
          Run the pipeline to see per-principle scoring.
        </div>
      )}

      {report && (
        <div className="space-y-3">
          {report.principles.map((p) => (
            <div key={p.principle}>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-medium text-ink-100">
                  {PRINCIPLE_LABEL[p.principle]}
                </span>
                <span
                  className={cn(
                    "font-mono",
                    p.score >= passThreshold ? "text-ok" : "text-warn",
                  )}
                >
                  {(p.score * 100).toFixed(0)}%
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-ink-700">
                <div
                  className={cn(
                    "h-full rounded-full transition-all duration-500",
                    p.score >= passThreshold ? "bg-ok" : "bg-warn",
                  )}
                  style={{ width: `${Math.max(p.score * 100, 4)}%` }}
                />
              </div>
              {p.feedback && (
                <div className="mt-1 text-[11px] leading-snug text-ink-300">
                  {p.feedback}
                </div>
              )}
            </div>
          ))}

          {report.similar_past_errors.length > 0 && (
            <div className="mt-4 border-t border-ink-700 pt-3">
              <div className="mb-2 text-[11px] uppercase tracking-wider text-ink-400">
                Similar past errors consulted
              </div>
              <div className="flex flex-wrap gap-1.5">
                {report.similar_past_errors.map((e, i) => (
                  <span
                    key={`${e}-${i}`}
                    className="rounded-full border border-ink-600 bg-ink-700/60 px-2 py-0.5 font-mono text-[10px] text-ink-200"
                  >
                    {e}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
