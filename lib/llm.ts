// Client for the generation backend. Never throws — resolves `false` on any
// failure (endpoint down, timeout, error response, abort) so callers can
// fall back to the offline extractive answer.
//
// Transport is currently a single JSON response, surfaced through the same
// delta callback that token streaming will use, so swapping in SSE is a
// change to this file only.

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
  chunks: LlmChunk[];
}

export async function generateLlmAnswer(
  req: GenerateRequest,
  onDelta: (delta: string) => void,
  onPhase: (phase: GenPhase) => void,
  signal: AbortSignal,
): Promise<boolean> {
  if (!req.chunks.length) return false;

  // combine the caller's abort with our own timeout
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
    if (!res.ok) return false;
    const data = await res.json();
    const answer = typeof data.answer === "string" ? data.answer.trim() : "";
    if (!answer) return false;
    if (signal.aborted) return false;
    onPhase("generating");
    onDelta(answer);
    return true;
  } catch {
    return false; // endpoint down, network error, timeout, or abort
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
  }
}
