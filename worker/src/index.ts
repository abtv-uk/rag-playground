// Standalone Cloudflare Worker: the only backend this app has. Everything
// here is deployed independently of the static site (see wrangler.jsonc) —
// `wrangler dev` for local development, `wrangler deploy` for production.
//
// Routes:
//   GET  /health    quota/model status, so the client can know before
//                   asking a question rather than discovering it 6s in
//   POST /generate  SSE token stream, grounded + cited prose
//
// Provider routing (the sample/upload trust boundary): the client can only
// reach Gemini by naming the bundled sample and sending chunk *ids* — any
// request carrying inline chunk text is routed to Workers AI regardless of
// what it claims. See sample.ts for why that boundary exists.
import {
  GEMINI_MODELS,
  WORKERS_AI_MODELS,
  checkAndIncrementIpDaily,
  geminiTier,
  hashIp,
  recordGeminiCall,
  workersAiTier,
} from "./budget";
import type { Env } from "./env";
import { buildAnswerPrompt, type RagId } from "./prompts";
import { connectGemini, streamWorkersAi } from "./providers";
import { resolveSampleChunk } from "./sample";
import { sseHeaders } from "./sse";

const RAG_IDS: RagId[] = ["naive", "hybrid", "corrective", "agentic"];
const MAX_QUERY_CHARS = 400;
const MAX_CHUNKS = 8;
const MAX_CHUNK_CHARS = 1200;
const MAX_BODY_BYTES = 32 * 1024;

interface GenerateRequestBody {
  rag?: unknown;
  query?: unknown;
  doc?: unknown;
  chunks?: unknown;
}

interface Passage {
  page: number;
  text: string;
}

function allowedOrigins(env: Env): string[] {
  return env.ALLOWED_ORIGINS.split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function corsOrigin(env: Env, req: Request): string | null {
  const origin = req.headers.get("Origin");
  if (!origin) return null;
  return allowedOrigins(env).includes(origin) ? origin : null;
}

function json(status: number, body: unknown, origin: string | null): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...(origin ? { "Access-Control-Allow-Origin": origin, Vary: "Origin" } : {}),
    },
  });
}

/** The trust boundary. `doc:"sample"` only reaches Gemini when every chunk
 *  is a bare id resolved against the Worker's own bundled copy — a claim of
 *  "sample" carrying inline text (spoofed or from a stale client) falls
 *  through to the upload path below it, never to Gemini. */
function resolvePassages(body: GenerateRequestBody): {
  passages: Passage[];
  provider: "gemini" | "workers-ai";
} {
  const rawChunks = Array.isArray(body.chunks) ? body.chunks.slice(0, MAX_CHUNKS) : [];
  const allIdsOnly = rawChunks.length > 0 && rawChunks.every((c) => typeof c === "number");

  if (body.doc === "sample" && allIdsOnly) {
    const passages = (rawChunks as number[])
      .map((id) => resolveSampleChunk(id))
      .filter((c): c is NonNullable<typeof c> => !!c)
      .map((c) => ({ page: c.page, text: c.text }));
    return { passages, provider: "gemini" };
  }

  const passages = rawChunks
    .map((c): Passage | null => {
      if (typeof c === "number") {
        const s = resolveSampleChunk(c);
        return s ? { page: s.page, text: s.text } : null;
      }
      if (c && typeof c === "object" && typeof (c as Record<string, unknown>).text === "string") {
        const o = c as Record<string, unknown>;
        return {
          page: Number(o.page) || 0,
          text: String(o.text).slice(0, MAX_CHUNK_CHARS),
        };
      }
      return null;
    })
    .filter((c): c is Passage => !!c);
  return { passages, provider: "workers-ai" };
}

async function handleGenerate(
  req: Request,
  env: Env,
  ctx: ExecutionContext,
  origin: string | null,
): Promise<Response> {
  const raw = await req.text();
  if (raw.length > MAX_BODY_BYTES) return json(413, { error: "request too large" }, origin);

  let body: GenerateRequestBody;
  try {
    body = JSON.parse(raw);
  } catch {
    return json(400, { error: "invalid JSON" }, origin);
  }

  if (typeof body.query !== "string" || !Array.isArray(body.chunks)) {
    return json(400, { error: "query and chunks are required" }, origin);
  }
  if (!RAG_IDS.includes(body.rag as RagId)) {
    return json(400, { error: "invalid rag mode" }, origin);
  }
  const query = body.query.slice(0, MAX_QUERY_CHARS).trim();
  if (!query) return json(400, { error: "empty query" }, origin);

  const ip = req.headers.get("CF-Connecting-IP") || "unknown";
  const { success } = await env.GENERATE_LIMITER.limit({ key: ip });
  if (!success) return json(429, { error: "rate limited" }, origin);

  const ipHash = await hashIp(ip, env.IP_HASH_SALT);
  const withinDaily = await checkAndIncrementIpDaily(env.RATE, ipHash);
  if (!withinDaily) return json(429, { error: "daily limit reached" }, origin);

  const { passages, provider } = resolvePassages(body);
  if (!passages.length) return json(400, { error: "no resolvable passages" }, origin);

  const prompt = buildAnswerPrompt(body.rag as RagId, query, passages);

  if (provider === "gemini") {
    const tier = await geminiTier(env.QUOTA);
    if (tier !== "exhausted") {
      const stream = await connectGemini(env, tier, prompt);
      if (stream) {
        // Only a real successful connection counts against the daily
        // ladder — a live 429 from Google's own rate limit shouldn't burn
        // through our 200-calls budget for a request that produced no
        // answer (observed directly in testing: a burst of manual requests
        // tripped this).
        ctx.waitUntil(recordGeminiCall(env.QUOTA));
        return new Response(stream, { headers: sseHeaders(origin) });
      }
      // Gemini's connection failed (its own rate limit, a 5xx, a network
      // error) — fall through to Workers AI rather than surfacing a
      // transient upstream error to the client.
    }
    // Gemini's daily allowance is spent — same fall-through, different
    // reason: neither should fail the sample path outright.
  }

  const wTier = await workersAiTier(env.QUOTA);
  if (wTier === "exhausted") return json(503, { error: "quota exhausted" }, origin);
  return new Response(streamWorkersAi(env, ctx, wTier, prompt), { headers: sseHeaders(origin) });
}

async function handleHealth(env: Env, origin: string | null): Promise<Response> {
  const [wTier, gTier] = await Promise.all([workersAiTier(env.QUOTA), geminiTier(env.QUOTA)]);
  return json(
    200,
    {
      ok: true,
      workersAi: { tier: wTier, model: wTier === "exhausted" ? null : WORKERS_AI_MODELS[wTier] },
      gemini: { tier: gTier, model: gTier === "exhausted" ? null : GEMINI_MODELS[gTier] },
    },
    origin,
  );
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    const origin = corsOrigin(env, req);

    if (req.method === "OPTIONS") {
      return new Response(null, {
        status: origin ? 204 : 403,
        headers: origin
          ? {
              "Access-Control-Allow-Origin": origin,
              "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
              "Access-Control-Allow-Headers": "Content-Type",
              Vary: "Origin",
            }
          : {},
      });
    }

    // Hard 403 for a disallowed cross-origin request — the floor, not the
    // defense; payload caps and the quota ladders do the real work. A
    // request with no Origin header (curl, server-to-server) isn't a CORS
    // request at all and passes through.
    if (req.headers.get("Origin") && !origin) {
      return new Response("origin not allowed", { status: 403 });
    }

    if (req.method === "GET" && url.pathname === "/health") {
      return handleHealth(env, origin);
    }
    if (req.method === "POST" && url.pathname === "/generate") {
      return handleGenerate(req, env, ctx, origin);
    }
    return json(404, { error: "not found" }, origin);
  },
};
