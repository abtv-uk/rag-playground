// Client for the generation Worker (worker/src/index.ts) — the only backend
// this app has. Consumes the Worker's own SSE wire format (not either
// upstream provider's — see worker/src/sse.ts) and never throws: any
// failure resolves without further deltas and returns false, so callers
// fall back to the offline extractive answer.

import type { GenPhase, RagId } from "./types";

const ENDPOINT =
  process.env.NEXT_PUBLIC_LLM_ENDPOINT || "http://localhost:8787";

// Ceiling on a single request. The caller applies its own, much shorter
// deadline before falling back to the extractive answer; this only bounds
// how long the request itself may stay open.
const TIMEOUT_MS = 20000;

export interface LlmChunk {
  id: number;
  page: number;
  text: string;
}

export interface GenerateRequest {
  rag: RagId;
  query: string;
  /** "sample" only when state.doc.isSample is true — see
   *  worker/src/sample.ts for why the client can't just assert this for
   *  arbitrary text. */
  doc: "sample" | "upload";
  /** Bare chunk ids for the sample (the Worker resolves them against its
   *  own bundled copy); full {id,page,text} objects for anything else. */
  chunks: number[] | LlmChunk[];
}

type WireEvent =
  | { type: "delta"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

/** Consumes an SSE body, calling onDelta per delta frame, until the stream
 *  signals "done"/"error" or simply closes. */
async function consumeSse(
  body: ReadableStream<Uint8Array>,
  onDelta: (delta: string) => void,
): Promise<{ gotDelta: boolean }> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let gotDelta = false;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).replace(/\r$/, "");
        buf = buf.slice(idx + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        let evt: WireEvent;
        try {
          evt = JSON.parse(payload);
        } catch {
          continue;
        }
        if (evt.type === "delta" && evt.text) {
          gotDelta = true;
          onDelta(evt.text);
        } else if (evt.type === "done" || evt.type === "error") {
          return { gotDelta };
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  return { gotDelta };
}

export async function generateLlmAnswer(
  req: GenerateRequest,
  onDelta: (delta: string) => void,
  onPhase: (phase: GenPhase) => void,
  signal: AbortSignal,
): Promise<boolean> {
  if (!req.chunks.length) return false;

  const ac = new AbortController();
  const onAbort = () => ac.abort();
  signal.addEventListener("abort", onAbort, { once: true });
  const timer = setTimeout(() => ac.abort(), TIMEOUT_MS);

  try {
    onPhase("waiting");
    const res = await fetch(ENDPOINT + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal: ac.signal,
    });
    if (!res.ok || !res.body) return false;
    onPhase("generating");
    const { gotDelta } = await consumeSse(res.body, onDelta);
    return gotDelta;
  } catch {
    return false; // Worker down, network error, timeout, or abort
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
  }
}

export interface HealthStatus {
  ok: boolean;
  workersAiAvailable: boolean;
  geminiAvailable: boolean;
}

const OFFLINE_STATUS: HealthStatus = {
  ok: false,
  workersAiAvailable: false,
  geminiAvailable: false,
};

/** Checked once at mount so the UI can say "generator offline" up front
 *  instead of silently degrading 6s into every query. Never throws. */
export async function checkHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch(ENDPOINT + "/health", {
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return OFFLINE_STATUS;
    const data = await res.json();
    return {
      ok: !!data.ok,
      workersAiAvailable: data.workersAi?.tier !== "exhausted",
      geminiAvailable: data.gemini?.tier !== "exhausted",
    };
  } catch {
    return OFFLINE_STATUS;
  }
}
