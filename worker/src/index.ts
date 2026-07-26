// Standalone Cloudflare Worker: the only backend this app has. Everything
// here is deployed independently of the static site (see wrangler.jsonc) —
// `wrangler dev` for local development, `wrangler deploy` for production.
//
// Routes:
//   GET  /health    quota/model status, so the client can know before
//                   asking a question rather than discovering it 6s in
//   POST /embed     int8-quantized 768-d embeddings for uploaded chunks
//   POST /grade     JSON relevance verdicts for corrective mode
//   POST /generate  SSE token stream, grounded + cited prose
//
// Provider routing (the sample/upload trust boundary): the client can only
// reach Gemini by naming the bundled sample and sending chunk *ids* — any
// request carrying inline chunk text is routed to Workers AI regardless of
// what it claims. See sample.ts for why that boundary exists.
import {
  BGE_NEURONS_PER_TOKEN,
  GEMINI_MODELS,
  WORKERS_AI_MODELS,
  checkAndIncrementIpDaily,
  geminiTier,
  hashIp,
  recordGeminiCall,
  recordWorkersAiSpend,
  workersAiTier,
} from "./budget";
import type { Env } from "./env";
import {
  buildAnswerPrompt,
  buildGradePrompt,
  type GradeVerdict,
  type RagId,
} from "./prompts";
import { connectGemini, runAuxJson, streamWorkersAi } from "./providers";
import { encodeVectorBin, quantizeInt8 } from "./quantize";
import { resolveSampleChunk } from "./sample";
import { sseHeaders } from "./sse";

const RAG_IDS: RagId[] = ["naive", "hybrid", "corrective", "agentic"];
const MAX_QUERY_CHARS = 400;
const MAX_CHUNKS = 8;
const MAX_CHUNK_CHARS = 1200;
const MAX_BODY_BYTES = 32 * 1024;

const EMBED_MODEL = "@cf/baai/bge-base-en-v1.5";
const MAX_EMBED_TEXTS = 450;
const MAX_EMBED_TEXT_CHARS = 2000;
const MAX_EMBED_BODY_BYTES = 512 * 1024;

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

interface EmbedRequestBody {
  texts?: unknown;
}

/** Embeds uploaded-document chunks for semantic retrieval. Shares the same
 *  daily budget ladder as generation (both draw on the Workers AI account),
 *  checked up front so a document never half-embeds. Response is the same
 *  binary vectors.bin format the sample's precomputed file uses (see
 *  quantize.ts) — one decoder on the client handles both. */
async function handleEmbed(
  req: Request,
  env: Env,
  ctx: ExecutionContext,
  origin: string | null,
): Promise<Response> {
  const raw = await req.text();
  if (raw.length > MAX_EMBED_BODY_BYTES)
    return json(413, { error: "request too large" }, origin);

  let body: EmbedRequestBody;
  try {
    body = JSON.parse(raw);
  } catch {
    return json(400, { error: "invalid JSON" }, origin);
  }
  if (!Array.isArray(body.texts) || !body.texts.length || !body.texts.every((t) => typeof t === "string")) {
    return json(400, { error: "texts (string[]) is required" }, origin);
  }
  const texts = (body.texts as string[])
    .slice(0, MAX_EMBED_TEXTS)
    .map((t) => t.slice(0, MAX_EMBED_TEXT_CHARS));

  const ip = req.headers.get("CF-Connecting-IP") || "unknown";
  const { success } = await env.GENERATE_LIMITER.limit({ key: ip });
  if (!success) return json(429, { error: "rate limited" }, origin);

  const wTier = await workersAiTier(env.QUOTA);
  if (wTier === "exhausted") return json(503, { error: "quota exhausted" }, origin);

  let result: { data: number[][]; usage?: { prompt_tokens?: number } };
  try {
    result = (await env.AI.run(EMBED_MODEL, { text: texts })) as unknown as typeof result;
  } catch (err) {
    return json(502, { error: err instanceof Error ? err.message : "embedding failed" }, origin);
  }
  const vectors = result.data;
  if (!Array.isArray(vectors) || !vectors.length)
    return json(502, { error: "empty embedding response" }, origin);
  const dim = vectors[0].length;

  const promptTokens = result.usage?.prompt_tokens ?? texts.length * 60; // rough fallback estimate
  const neurons = promptTokens * BGE_NEURONS_PER_TOKEN;
  // Unlike /generate's streamed accounting, this response isn't a stream —
  // it's returned synchronously below, so the write completes as part of
  // normal request handling and doesn't strictly need waitUntil. Using it
  // anyway costs nothing and removes any doubt.
  ctx.waitUntil(recordWorkersAiSpend(env.QUOTA, neurons));

  const batch = quantizeInt8(vectors, dim);
  const bytes = encodeVectorBin(batch);
  return new Response(bytes, {
    headers: {
      "Content-Type": "application/octet-stream",
      ...(origin ? { "Access-Control-Allow-Origin": origin, Vary: "Origin" } : {}),
    },
  });
}

interface GradeRequestBody {
  query?: unknown;
  doc?: unknown;
  chunks?: unknown;
}

const MAX_GRADE_PASSAGES = 5;
const MAX_GRADE_PASSAGE_CHARS = 700;

/** Real relevance verdicts for corrective mode, replacing the client's
 *  absolute-cosine-floor placeholder.
 *
 *  Always Workers AI, never Gemini — resolvePassages' provider verdict is
 *  deliberately ignored here. That is not a gap in the trust boundary but
 *  the strictest possible reading of it: grading is the one call that
 *  necessarily sees passage text for BOTH the sample and uploads, so
 *  pinning it to Workers AI means a stranger's uploaded document can never
 *  reach a provider whose free tier permits human review. resolvePassages
 *  is still what turns bare sample ids into text, so the sample keeps
 *  costing no upload bandwidth.
 *
 *  Does NOT increment the per-IP daily counter: /grade only ever runs as
 *  the first half of a corrective query whose /generate call increments it,
 *  and charging a visitor two units for one question would silently halve
 *  corrective mode's daily allowance. The per-minute limiter and the neuron
 *  ladder both still apply, so the route can't be farmed for free work. */
async function handleGrade(
  req: Request,
  env: Env,
  ctx: ExecutionContext,
  origin: string | null,
): Promise<Response> {
  const raw = await req.text();
  if (raw.length > MAX_BODY_BYTES) return json(413, { error: "request too large" }, origin);

  let body: GradeRequestBody;
  try {
    body = JSON.parse(raw);
  } catch {
    return json(400, { error: "invalid JSON" }, origin);
  }
  if (typeof body.query !== "string" || !Array.isArray(body.chunks)) {
    return json(400, { error: "query and chunks are required" }, origin);
  }
  const query = body.query.slice(0, MAX_QUERY_CHARS).trim();
  if (!query) return json(400, { error: "empty query" }, origin);

  const ip = req.headers.get("CF-Connecting-IP") || "unknown";
  const { success } = await env.GENERATE_LIMITER.limit({ key: ip });
  if (!success) return json(429, { error: "rate limited" }, origin);

  const { passages } = resolvePassages(body as GenerateRequestBody);
  if (!passages.length) return json(400, { error: "no resolvable passages" }, origin);
  // Truncated harder than generation's passages: the grader only needs
  // enough text to judge topicality, and a shorter prompt is a faster
  // first token on the serial path in front of the answer.
  const graded = passages.slice(0, MAX_GRADE_PASSAGES).map((p) => ({
    page: p.page,
    text: p.text.slice(0, MAX_GRADE_PASSAGE_CHARS),
  }));

  const wTier = await workersAiTier(env.QUOTA);
  if (wTier === "exhausted") return json(503, { error: "quota exhausted" }, origin);

  // One call per passage, in parallel — see buildGradePrompt's note for the
  // measurements behind that. Wall-clock is one call's latency, not five.
  const settled = await Promise.all(
    graded.map((p) => runAuxJson(env, ctx, buildGradePrompt(query, p))),
  );

  // Validate every field rather than trusting JSON mode: `relevant` comes
  // back as the string "true" often enough to matter, and a missing `why`
  // would render as an empty reject card.
  const verdicts: GradeVerdict[] = [];
  settled.forEach((parsed, i) => {
    if (!parsed || typeof parsed !== "object") return;
    const o = parsed as Record<string, unknown>;
    if (o.relevant !== true && o.relevant !== false && o.relevant !== "true" && o.relevant !== "false")
      return;
    verdicts.push({
      i,
      relevant: o.relevant === true || o.relevant === "true",
      why: typeof o.why === "string" ? o.why.slice(0, 120).trim() : "",
    });
  });

  // Partial results are usable here in a way a partial batch never was:
  // each verdict is anchored to its own passage by position, so a passage
  // whose call failed simply keeps the client's existing score rather than
  // being silently rejected. Only a total washout is worth failing — that
  // means the grader is down, not merely fussy.
  if (!verdicts.length) {
    return json(502, { error: "grader returned no usable verdicts" }, origin);
  }
  return json(200, { verdicts }, origin);
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
    if (req.method === "POST" && url.pathname === "/embed") {
      return handleEmbed(req, env, ctx, origin);
    }
    if (req.method === "POST" && url.pathname === "/grade") {
      return handleGrade(req, env, ctx, origin);
    }
    if (req.method === "POST" && url.pathname === "/generate") {
      return handleGenerate(req, env, ctx, origin);
    }
    return json(404, { error: "not found" }, origin);
  },
};
