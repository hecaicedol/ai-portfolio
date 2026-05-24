"use client";

import { useEffect, useState } from "react";
import { Play, Loader2, Sparkles } from "lucide-react";
import { SAMPLES, type SampleDoc } from "@/data/samples";
import { DEMO_SCENARIO } from "@/lib/demo";
import type { DocumentType } from "@/lib/types";
import { cn } from "@/lib/cn";

interface Props {
  onRun: (req: { document_type: DocumentType; content: string }) => void;
  running: boolean;
  mode: "demo" | "live";
}

const DOC_TYPES: DocumentType[] = [
  "invoice",
  "receipt",
  "contract",
  "purchase_order",
  "generic",
];

export function DocumentInput({ onRun, running, mode }: Props) {
  const [docType, setDocType] = useState<DocumentType>(
    mode === "demo" ? DEMO_SCENARIO.document_type : "invoice",
  );
  const [content, setContent] = useState<string>(
    mode === "demo" ? DEMO_SCENARIO.content : SAMPLES[0].content,
  );

  useEffect(() => {
    // Keep input aligned with mode changes — demo always uses the scripted doc
    if (mode === "demo") {
      setDocType(DEMO_SCENARIO.document_type);
      setContent(DEMO_SCENARIO.content);
    }
  }, [mode]);

  function pickSample(sample: SampleDoc) {
    setDocType(sample.document_type);
    setContent(sample.content);
  }

  const isDemo = mode === "demo";
  const inputsLocked = running || isDemo;

  return (
    <section className="rounded-2xl border border-ink-700 bg-ink-800/40 p-5">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-medium tracking-tight text-ink-100">
          Input
        </h2>
        <div className="text-xs text-ink-400">
          {content.length.toLocaleString()} chars
        </div>
      </div>

      {isDemo && (
        <div className="mb-4 flex items-start gap-2 rounded-md border border-accent-500/30 bg-accent-500/5 px-3 py-2 text-xs text-accent-400">
          <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <div className="text-ink-200">
            <span className="font-medium text-accent-400">Scripted demo.</span>{" "}
            The frame stream below is played client-side — no LLM is called.
            Showcases self-healing on the second iteration (the extractor first
            grabs a line item as the total, the critic catches it, reflection
            feeds the correction back).
          </div>
        </div>
      )}

      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="flex flex-col gap-1.5">
          <span className="text-xs uppercase tracking-wide text-ink-300">
            Document type
          </span>
          <select
            value={docType}
            onChange={(e) => setDocType(e.target.value as DocumentType)}
            disabled={inputsLocked}
            className="rounded-md border border-ink-600 bg-ink-700 px-3 py-2 text-sm text-ink-100 outline-none focus:border-accent-500 disabled:opacity-60"
          >
            {DOC_TYPES.map((t) => (
              <option key={t} value={t}>
                {t.replace("_", " ")}
              </option>
            ))}
          </select>
        </label>

        <div className="flex flex-col gap-1.5">
          <span className="text-xs uppercase tracking-wide text-ink-300">
            Load sample
          </span>
          <select
            onChange={(e) => {
              const sample = SAMPLES.find((s) => s.id === e.target.value);
              if (sample) pickSample(sample);
            }}
            disabled={inputsLocked}
            defaultValue=""
            className="rounded-md border border-ink-600 bg-ink-700 px-3 py-2 text-sm text-ink-100 outline-none focus:border-accent-500 disabled:opacity-60"
          >
            <option value="" disabled>
              choose one…
            </option>
            {SAMPLES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        disabled={inputsLocked}
        rows={14}
        spellCheck={false}
        className="w-full resize-none rounded-md border border-ink-600 bg-ink-900 p-3 font-mono text-xs leading-relaxed text-ink-100 outline-none focus:border-accent-500 disabled:opacity-70"
      />

      <div className="mt-4 flex items-center justify-end">
        <button
          type="button"
          onClick={() => onRun({ document_type: docType, content })}
          disabled={running || content.trim().length === 0}
          className={cn(
            "inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition",
            "bg-accent-500 text-white hover:bg-accent-600",
            "disabled:cursor-not-allowed disabled:bg-ink-600 disabled:text-ink-300",
          )}
        >
          {running ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : isDemo ? (
            <Sparkles className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          {running ? "Running…" : isDemo ? "Play scripted demo" : "Run pipeline"}
        </button>
      </div>
    </section>
  );
}
