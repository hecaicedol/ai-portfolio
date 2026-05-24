"use client";

import { Sparkles, Radio } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { cn } from "@/lib/cn";

export type Mode = "demo" | "live";

interface Props {
  mode: Mode;
  apiReachable: boolean | null;
  onToggleMode: () => void;
}

export function Header({ mode, apiReachable, onToggleMode }: Props) {
  const isDemo = mode === "demo";
  const Icon = isDemo ? Sparkles : Radio;

  const tooltip = isDemo
    ? apiReachable
      ? "Switch to live mode (API reachable)"
      : "Backend is offline — switch only available when API is reachable"
    : "Switch to scripted demo mode";

  return (
    <header className="border-b border-ink-700 bg-ink-800/50 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-ink-100">
            Self-Healing Multi-Agent Pipeline
          </h1>
          <p className="mt-0.5 text-xs text-ink-300">
            Constitutional AI critic · reflection loops · episodic memory
          </p>
        </div>
        <button
          type="button"
          onClick={onToggleMode}
          disabled={isDemo && !apiReachable}
          title={tooltip}
          className={cn(
            "flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition",
            "disabled:cursor-not-allowed",
            isDemo
              ? "border-accent-500/40 bg-accent-500/10 text-accent-400 hover:border-accent-500"
              : apiReachable
                ? "border-ok/40 bg-ok/10 text-ok hover:border-ok"
                : "border-bad/40 bg-bad/10 text-bad",
          )}
        >
          <Icon className="h-3.5 w-3.5" />
          <span className="font-medium">
            {isDemo ? "Demo mode" : apiReachable ? "Live mode" : "API offline"}
          </span>
          <span className="text-ink-400">·</span>
          <span className="font-mono text-ink-300">
            {isDemo ? "scripted scenario" : API_BASE}
          </span>
        </button>
      </div>
    </header>
  );
}
