import type {
  EpisodicError,
  PipelineEvent,
  ProcessRequest,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}

export async function listRecentErrors(limit = 10): Promise<EpisodicError[]> {
  const res = await fetch(`${API_BASE}/api/memory/errors?limit=${limit}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`memory/errors ${res.status}`);
  const body = (await res.json()) as { errors: EpisodicError[] };
  return body.errors;
}

/**
 * Open an SSE-over-POST stream against the backend and yield each PipelineEvent
 * as it arrives. The backend wraps each event with `event:`/`data:` lines and
 * a `\n\n` terminator (sse-starlette format).
 */
export async function* streamPipeline(
  req: ProcessRequest,
  signal?: AbortSignal,
): AsyncGenerator<PipelineEvent> {
  const res = await fetch(`${API_BASE}/api/process/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) throw new Error(`process/stream ${res.status}`);
  if (!res.body) throw new Error("no response body for SSE stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const sep = buffer.indexOf("\n\n");
      if (sep === -1) break;
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("data:")) data += line.slice(5).trimStart();
      }
      if (!data) continue;
      try {
        yield JSON.parse(data) as PipelineEvent;
      } catch {
        // ignore malformed frames — server hiccup
      }
    }
  }
}
