"use client";

import { Check } from "lucide-react";

interface Props {
  data: Record<string, unknown> | null;
  iterations?: number;
  selfHealed?: boolean;
}

export function ExtractedDataPanel({ data, iterations, selfHealed }: Props) {
  return (
    <section className="rounded-2xl border border-ink-700 bg-ink-800/40 p-5">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-medium tracking-tight text-ink-100">
          Extracted data
        </h2>
        {data && iterations !== undefined && (
          <div className="flex items-center gap-2 text-xs text-ink-300">
            {selfHealed && (
              <span className="inline-flex items-center gap-1 rounded-full bg-accent-500/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-accent-400">
                <Check className="h-3 w-3" />
                self-healed
              </span>
            )}
            <span className="text-ink-400">
              {iterations} iter{iterations === 1 ? "" : "s"}
            </span>
          </div>
        )}
      </div>

      {!data && (
        <div className="rounded-md border border-dashed border-ink-600 px-4 py-8 text-center text-xs text-ink-400">
          The pipeline output will appear here.
        </div>
      )}

      {data && (
        <pre className="overflow-x-auto rounded-md border border-ink-700 bg-ink-900 p-3 font-mono text-xs leading-relaxed text-ink-100">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </section>
  );
}
