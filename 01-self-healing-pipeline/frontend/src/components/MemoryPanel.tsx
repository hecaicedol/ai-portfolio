"use client";

import { useEffect, useState } from "react";
import { Database, RefreshCw, Sparkles } from "lucide-react";
import type { EpisodicError } from "@/lib/types";
import { listRecentErrors } from "@/lib/api";
import { DEMO_RECENT_ERRORS } from "@/lib/demo";

interface Props {
  refreshToken: number;
  demoMode: boolean;
}

export function MemoryPanel({ refreshToken, demoMode }: Props) {
  const [errors, setErrors] = useState<EpisodicError[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    if (demoMode) {
      setError(null);
      setErrors(DEMO_RECENT_ERRORS);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const list = await listRecentErrors(8);
      setErrors(list);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshToken, demoMode]);

  return (
    <section className="rounded-2xl border border-ink-700 bg-ink-800/40 p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-accent-400" />
          <h2 className="text-sm font-medium tracking-tight text-ink-100">
            Episodic memory · recent errors
          </h2>
          {demoMode && (
            <span className="inline-flex items-center gap-1 rounded-full bg-accent-500/15 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-accent-400">
              <Sparkles className="h-3 w-3" />
              scripted
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={refresh}
          disabled={loading}
          className="inline-flex items-center gap-1 rounded-md border border-ink-600 px-2 py-1 text-xs text-ink-200 transition hover:border-accent-500 disabled:opacity-50"
        >
          <RefreshCw className={loading ? "h-3 w-3 animate-spin" : "h-3 w-3"} />
          refresh
        </button>
      </div>

      {error && (
        <div className="rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-xs text-bad">
          {error}
        </div>
      )}

      {!error && errors && errors.length === 0 && (
        <div className="rounded-md border border-dashed border-ink-600 px-4 py-6 text-center text-xs text-ink-400">
          No errors stored yet. Run the pipeline on a tricky document to
          populate this view.
        </div>
      )}

      {!error && errors && errors.length > 0 && (
        <ul className="space-y-2">
          {errors.map((e) => (
            <li
              key={e.id}
              className="flex items-start justify-between gap-3 rounded-md border border-ink-700 bg-ink-800/50 px-3 py-2"
            >
              <div className="min-w-0">
                <div className="font-mono text-xs text-ink-100">
                  {e.error_type}
                </div>
                <div className="mt-0.5 text-[11px] text-ink-400">
                  {e.document_type} · {e.principle} ·{" "}
                  {new Date(e.created_at).toLocaleString()}
                </div>
              </div>
              <span className="rounded-full bg-ink-700 px-2 py-0.5 text-[10px] font-medium text-ink-200">
                #{e.id}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
