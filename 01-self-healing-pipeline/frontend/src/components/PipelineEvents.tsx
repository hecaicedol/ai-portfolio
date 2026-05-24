"use client";

import {
  Check,
  CircleDashed,
  Cog,
  FileText,
  Loader2,
  RotateCw,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { PipelineEvent } from "@/lib/types";
import { cn } from "@/lib/cn";

interface Props {
  events: PipelineEvent[];
  running: boolean;
  failed?: boolean;
}

const AGENT_META: Record<
  string,
  { label: string; icon: LucideIcon; tone: string }
> = {
  extract: { label: "Extract", icon: FileText, tone: "text-sky-300" },
  validate: { label: "Validate", icon: ShieldCheck, tone: "text-emerald-300" },
  critique: { label: "Critique", icon: Cog, tone: "text-amber-300" },
  reflect: { label: "Reflect", icon: RotateCw, tone: "text-fuchsia-300" },
  synthesize: { label: "Synthesize", icon: Sparkles, tone: "text-accent-400" },
};

interface NodeStep {
  agent: string;
  iteration: number;
  payload: Record<string, unknown>;
}

function collectSteps(events: PipelineEvent[]): NodeStep[] {
  return events
    .filter((e) => e.type === "agent_finished" && e.agent)
    .map((e) => ({
      agent: e.agent ?? "",
      iteration: e.iteration ?? 0,
      payload: e.payload ?? {},
    }));
}

export function PipelineEvents({ events, running, failed }: Props) {
  const steps = collectSteps(events);
  const started = events.some((e) => e.type === "run_started");
  const completed = events.some((e) => e.type === "run_completed");

  return (
    <section className="rounded-2xl border border-ink-700 bg-ink-800/40 p-5">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-medium tracking-tight text-ink-100">
          Pipeline
        </h2>
        <div className="flex items-center gap-2 text-xs text-ink-400">
          {running ? (
            <>
              <Loader2 className="h-3 w-3 animate-spin text-accent-400" />
              running
            </>
          ) : completed ? (
            <span className="text-ok">completed</span>
          ) : failed ? (
            <span className="text-bad">failed</span>
          ) : (
            <span>idle</span>
          )}
        </div>
      </div>

      {!started && (
        <div className="rounded-md border border-dashed border-ink-600 px-4 py-8 text-center text-xs text-ink-400">
          Submit a document to see the pipeline execute, step by step.
        </div>
      )}

      <ol className="space-y-2">
        {steps.map((step, idx) => {
          const meta = AGENT_META[step.agent] ?? {
            label: step.agent,
            icon: CircleDashed,
            tone: "text-ink-300",
          };
          const Icon = meta.icon;
          const detail = describeStep(step);
          return (
            <li
              key={idx}
              className="flex items-start gap-3 rounded-md border border-ink-700 bg-ink-800/60 px-3 py-2"
            >
              <div
                className={cn(
                  "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-ink-700",
                  meta.tone,
                )}
              >
                <Icon className="h-3.5 w-3.5" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-sm font-medium text-ink-100">
                    {meta.label}
                  </span>
                  {step.iteration > 0 && (
                    <span className="text-[10px] uppercase tracking-wider text-ink-400">
                      iter {step.iteration}
                    </span>
                  )}
                </div>
                {detail && (
                  <div className="mt-0.5 text-xs text-ink-300">{detail}</div>
                )}
              </div>
              <Check className="mt-1 h-4 w-4 shrink-0 text-ok" />
            </li>
          );
        })}
        {running && started && !completed && (
          <li className="flex items-center gap-3 rounded-md border border-ink-700 bg-ink-800/30 px-3 py-2 text-xs text-ink-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-accent-400" />
            waiting for next event…
          </li>
        )}
        {failed && (
          <li className="flex items-center gap-3 rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
            <X className="h-3.5 w-3.5" />
            pipeline returned an error — check the API logs
          </li>
        )}
      </ol>
    </section>
  );
}

function describeStep(step: NodeStep): string | null {
  if (step.agent === "extract") {
    const extracted = (step.payload as { extracted?: Record<string, unknown> })
      .extracted;
    if (extracted) {
      const keys = Object.keys(extracted);
      return `extracted ${keys.length} field${keys.length === 1 ? "" : "s"}`;
    }
  }
  if (step.agent === "validate") {
    const structural = (
      step.payload as {
        structural?: { structural_pass?: boolean; missing_fields?: string[] };
      }
    ).structural;
    if (structural) {
      if (structural.structural_pass) return "all required fields present";
      const missing = structural.missing_fields ?? [];
      return `missing: ${missing.join(", ")}`;
    }
  }
  if (step.agent === "critique") {
    const critic = (
      step.payload as {
        critic?: { overall_score?: number; passes?: boolean };
      }
    ).critic;
    if (critic) {
      const pct = ((critic.overall_score ?? 0) * 100).toFixed(0);
      return `score ${pct}% — ${critic.passes ? "passed" : "below threshold"}`;
    }
  }
  if (step.agent === "reflect") {
    const fb = (step.payload as { last_feedback?: string[] }).last_feedback;
    if (fb && fb.length > 0) {
      return `${fb.length} principle${fb.length === 1 ? "" : "s"} flagged — re-extracting`;
    }
  }
  if (step.agent === "synthesize") {
    const final = (
      step.payload as {
        final?: { audit?: { self_healed?: boolean; iterations?: number } };
      }
    ).final;
    if (final?.audit) {
      const it = final.audit.iterations ?? 0;
      return final.audit.self_healed
        ? `self-healed across ${it} iteration${it === 1 ? "" : "s"}`
        : "final output assembled";
    }
  }
  return null;
}
