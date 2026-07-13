// Standalone local proxy for real LLM-generated answers.
//
// Why this exists as a separate process instead of a Next.js API route: the
// app builds with `output: "export"` (a static site served from GitHub
// Pages) so there is no server at runtime — a Route Handler simply cannot
// exist in that build. This script is never imported by any file under
// app/, components/, hooks/, or lib/, so Next's bundler never sees it and
// GEMINI_API_KEY can never end up in the client bundle or the exported
// static site. Run it only when you want real LLM answers during local
// development: `npm run dev:llm` (in a second terminal, alongside `npm run
// dev`). The playground UI falls back to its offline extractive answer
// whenever this proxy isn't reachable — including for every visitor of the
// deployed GitHub Pages site.
//
// Usage: node --env-file=.env server/llm-proxy.mjs

import { createServer } from "node:http";
import { GoogleGenAI } from "@google/genai";

const PORT = Number(process.env.LLM_PROXY_PORT) || 8787;
const MODEL = "gemini-3.1-flash-lite-preview";

if (!process.env.GEMINI_API_KEY) {
  console.error(
    "[llm-proxy] GEMINI_API_KEY is not set. Add it to .env and run with `node --env-file=.env server/llm-proxy.mjs`.",
  );
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const MODE_FRAME = {
  naive: "a single-pass vector search",
  hybrid: "a merged vector + knowledge-graph search",
  corrective: "a retrieval pass that graded chunks for relevance and re-retrieved to replace any that were rejected",
  agentic: "an agent that retrieved once, refined its query, and retrieved again",
};

function buildPrompt(rag, query, chunks) {
  const context = chunks
    .map((c, i) => `[${i + 1}] (chunk #${c.id}, p.${c.page}) ${c.text}`)
    .join("\n\n");
  const frame = MODE_FRAME[rag] || MODE_FRAME.naive;
  return (
    `You are answering a question using only the retrieved passages below, produced by ${frame}. ` +
    `Answer strictly from the passages — do not use outside knowledge, and do not mention the retrieval process itself. ` +
    `If the passages don't contain the answer, say so plainly. Keep the answer to 2-4 sentences.\n\n` +
    `PASSAGES:\n${context}\n\nQUESTION: ${query}\n\nANSWER:`
  );
}

function allowOrigin(origin) {
  return !!origin && /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin);
}

function send(res, status, body, origin) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    ...(allowOrigin(origin) ? { "Access-Control-Allow-Origin": origin } : {}),
    Vary: "Origin",
  });
  res.end(JSON.stringify(body));
}

const server = createServer(async (req, res) => {
  const origin = req.headers.origin;

  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      ...(allowOrigin(origin) ? { "Access-Control-Allow-Origin": origin } : {}),
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    });
    res.end();
    return;
  }

  if (req.method === "GET" && req.url === "/health") {
    send(res, 200, { ok: true }, origin);
    return;
  }

  if (req.method !== "POST" || req.url !== "/generate") {
    send(res, 404, { error: "not found" }, origin);
    return;
  }

  if (!allowOrigin(origin)) {
    send(res, 403, { error: "origin not allowed" }, origin);
    return;
  }

  let body = "";
  for await (const chunk of req) body += chunk;

  let payload;
  try {
    payload = JSON.parse(body);
  } catch {
    send(res, 400, { error: "invalid JSON" }, origin);
    return;
  }

  const { rag, query, chunks } = payload || {};
  if (!query || !Array.isArray(chunks) || !chunks.length) {
    send(res, 400, { error: "query and chunks are required" }, origin);
    return;
  }

  try {
    const response = await ai.models.generateContent({
      model: MODEL,
      contents: buildPrompt(rag, query, chunks.slice(0, 6)),
    });
    const text = (response.text || "").trim();
    if (!text) throw new Error("empty response");
    send(res, 200, { answer: text }, origin);
  } catch (err) {
    console.error("[llm-proxy] generate failed:", err.message || err);
    send(res, 502, { error: "generation failed" }, origin);
  }
});

server.listen(PORT, () => {
  console.log(`[llm-proxy] listening on http://localhost:${PORT} (model: ${MODEL})`);
});
