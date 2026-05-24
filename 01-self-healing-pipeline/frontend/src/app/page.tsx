"use client";

import { useEffect, useState } from "react";
import { Header, type Mode } from "@/components/Header";
import { DocumentInput } from "@/components/DocumentInput";
import { PipelineEvents } from "@/components/PipelineEvents";
import { CriticReportPanel } from "@/components/CriticReportPanel";
import { ExtractedDataPanel } from "@/components/ExtractedDataPanel";
import { MemoryPanel } from "@/components/MemoryPanel";
import { checkHealth, streamPipeline } from "@/lib/api";
import { streamDemoPipeline } from "@/lib/demo";
import type {
  CriticReport,
  DocumentType,
  PipelineEvent,
} from "@/lib/types";

interface RunOutput {
  critic: CriticReport | null;
  extracted: Record<string, unknown> | null;
  iterations: number;
  selfHealed: boolean;
}

const EMPTY_OUTPUT: RunOutput = {
  critic: null,
  extracted: null,
  iterations: 0,
  selfHealed: false,
};

export default function HomePage() {
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [failed, setFailed] = useState(false);
  const [output, setOutput] = useState<RunOutput>(EMPTY_OUTPUT);
  const [memoryToken, setMemoryToken] = useState(0);
  const [apiReachable, setApiReachable] = useState<boolean | null>(null);
  const [userOverrodeMode, setUserOverrodeMode] = useState(false);
  const [mode, setMode] = useState<Mode>("demo");

  // Health-poll the backend; when it flips, optionally flip the mode.
  useEffect(() => {
    let live = true;
    const poll = async () => {
      const ok = await checkHealth();
      if (!live) return;
      setApiReachable(ok);
      if (!userOverrodeMode) {
        setMode(ok ? "live" : "demo");
      }
    };
    poll();
    const handle = setInterval(poll, 10_000);
    return () => {
      live = false;
      clearInterval(handle);
    };
  }, [userOverrodeMode]);

  function handleToggleMode() {
    setUserOverrodeMode(true);
    setMode((m) => (m === "demo" ? "live" : "demo"));
  }

  async function handleRun(req: {
    document_type: DocumentType;
    content: string;
  }) {
    setEvents([]);
    setOutput(EMPTY_OUTPUT);
    setRunning(true);
    setFailed(false);

    const source =
      mode === "demo" ? streamDemoPipeline() : streamPipeline(req);

    try {
      const collected: PipelineEvent[] = [];
      for await (const ev of source) {
        collected.push(ev);
        setEvents([...collected]);

        if (ev.type === "agent_finished" && ev.agent === "critique") {
          const payload = ev.payload as { critic?: CriticReport };
          if (payload.critic) {
            setOutput((prev) => ({ ...prev, critic: payload.critic ?? null }));
          }
        }

        if (ev.type === "agent_finished" && ev.agent === "synthesize") {
          const payload = ev.payload as {
            final?: {
              data?: Record<string, unknown>;
              audit?: { iterations?: number; self_healed?: boolean };
            };
          };
          if (payload.final) {
            setOutput((prev) => ({
              ...prev,
              extracted: payload.final?.data ?? null,
              iterations: payload.final?.audit?.iterations ?? 0,
              selfHealed: Boolean(payload.final?.audit?.self_healed),
            }));
          }
        }
      }
    } catch (e) {
      console.error(e);
      setFailed(true);
    } finally {
      setRunning(false);
      setMemoryToken((t) => t + 1);
    }
  }

  return (
    <div className="min-h-screen">
      <Header
        mode={mode}
        apiReachable={apiReachable}
        onToggleMode={handleToggleMode}
      />

      <main className="mx-auto max-w-7xl space-y-6 px-6 py-8">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <DocumentInput onRun={handleRun} running={running} mode={mode} />
          <PipelineEvents events={events} running={running} failed={failed} />
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <CriticReportPanel report={output.critic} />
          <ExtractedDataPanel
            data={output.extracted}
            iterations={output.iterations}
            selfHealed={output.selfHealed}
          />
        </div>

        <MemoryPanel refreshToken={memoryToken} demoMode={mode === "demo"} />

        <footer className="border-t border-ink-700 pt-6 text-xs text-ink-400">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              Project 1 of the{" "}
              <span className="text-ink-200">ai-portfolio</span> series — Heims
              Andrés Caicedo Lopera
            </div>
            <div className="font-mono">
              LangGraph · Claude Sonnet 4.5 · pgvector · FastAPI
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
