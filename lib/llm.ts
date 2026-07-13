// Client for the local-only LLM proxy (server/llm-proxy.mjs). Never throws —
// returns null on any failure (proxy not running, timeout, error response) so
// callers can fall back to the offline extractive answer. The proxy only
// ever listens on localhost, so this resolves to nothing for real visitors
// of the deployed static site — it's a local-dev enhancement only.

import type { RagId } from "./types";

const ENDPOINT =
  process.env.NEXT_PUBLIC_LLM_ENDPOINT || "http://localhost:8787";
// Real gemini-3.5-flash latency observed for a several-chunk grounded prompt
// runs 10-15s in this environment; a short timeout meant the offline
// extractive answer almost always won the race. This only affects local-dev
// pacing (a real visitor's fetch to localhost never connects at all), so a
// generous budget is fine.
const TIMEOUT_MS = 20000;

export interface LlmChunk {
  id: number;
  page: number;
  text: string;
}

export async function generateLlmAnswer(
  rag: RagId,
  query: string,
  chunks: LlmChunk[],
): Promise<string | null> {
  if (!chunks.length) return null;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(ENDPOINT + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rag, query, chunks }),
      signal: controller.signal,
    });
    if (!res.ok) return null;
    const data = await res.json();
    const answer = typeof data.answer === "string" ? data.answer.trim() : "";
    return answer || null;
  } catch {
    return null; // proxy not running, network error, or timeout — fall back silently
  } finally {
    clearTimeout(timer);
  }
}
