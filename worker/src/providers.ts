// Turns each provider's stream into the Worker's own wire format (see
// sse.ts). Gemini and Workers AI use two different chunk shapes on the
// wire — normalizing here means neither leaks past this file.
import {
  AUX_MODEL,
  GEMINI_MODELS,
  WORKERS_AI_MODELS,
  recordWorkersAiSpend,
  type GeminiTier,
  type WorkersAiTier,
} from "./budget";
import type { Env } from "./env";
import type { BuiltPrompt } from "./prompts";
import { logUpstreamError } from "./log";
import { frame, readUpstreamSseData } from "./sse";

const GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models";
const MAX_OUTPUT_TOKENS = 400;
const TEMPERATURE = 0.3;

/** Workers AI's newer `-fast` chat-completion models emit
 *  `choices[0].delta.content`; older ones emit a plain `response` string.
 *  Handling both means the primary/downshift tiers don't need to share a
 *  model family. The final chunk of either shape carries a cumulative
 *  `usage.neurons` total — the actual cost, not an estimate. */
function parseWorkersAiChunk(json: any): { delta: string; neurons?: number } {
  const delta =
    typeof json.response === "string"
      ? json.response
      : typeof json.choices?.[0]?.delta?.content === "string"
        ? json.choices[0].delta.content
        : "";
  const neurons = typeof json.usage?.neurons === "number" ? json.usage.neurons : undefined;
  return { delta, neurons };
}

export function streamWorkersAi(
  env: Env,
  ctx: ExecutionContext,
  tier: Exclude<WorkersAiTier, "exhausted">,
  prompt: BuiltPrompt,
): ReadableStream<Uint8Array> {
  const model = WORKERS_AI_MODELS[tier];
  return new ReadableStream({
    async start(controller) {
      let totalNeurons = 0;
      try {
        // `model` is a runtime string (chosen by budget tier), not a
        // literal, so it can't match the SDK's per-model overloads —
        // stream:true does return a ReadableStream at runtime regardless
        // (verified directly against the binding), hence the double cast.
        const upstream = (await env.AI.run(model, {
          messages: [
            { role: "system", content: prompt.system },
            { role: "user", content: prompt.user },
          ],
          stream: true,
          max_tokens: MAX_OUTPUT_TOKENS,
          temperature: TEMPERATURE,
        })) as unknown as ReadableStream<Uint8Array>;

        for await (const raw of readUpstreamSseData(upstream)) {
          if (raw === "[DONE]") break;
          let json: any;
          try {
            json = JSON.parse(raw);
          } catch {
            continue;
          }
          const { delta, neurons } = parseWorkersAiChunk(json);
          if (typeof neurons === "number") totalNeurons = neurons;
          if (delta) controller.enqueue(frame({ type: "delta", text: delta }));
        }
      } catch (err) {
        controller.enqueue(
          frame({ type: "error", message: err instanceof Error ? err.message : String(err) }),
        );
      }
      controller.enqueue(frame({ type: "done" }));
      controller.close();
      // The KV write happens after controller.close(), by which point the
      // client may already be gone and the runtime free to tear down this
      // request's execution context — waitUntil is what keeps it alive
      // long enough for the write to land. Without it this accounting
      // silently drops under real traffic (caught via a direct KV read
      // during Phase 1 testing: the counter never advanced past 0).
      // Fixed fallback keeps the ladder moving even if the chunk shape
      // above changes upstream and neurons is never captured.
      ctx.waitUntil(recordWorkersAiSpend(env.QUOTA, totalNeurons || 20));
    },
  });
}

// ---------- auxiliary JSON reasoning (grading, and later planning) ----------

const AUX_MAX_TOKENS = 500;
// Near-zero: grading is a classification, and letting a 1B model be
// creative about it is exactly how you get invented justifications.
const AUX_TEMPERATURE = 0.1;
// Fallback when the response carries no usage.neurons, mirroring the
// stream path's `|| 20`. Lower because this is the 1B model on a short
// prompt — over-charging the ladder would refuse service early.
const AUX_FALLBACK_NEURONS = 8;

/** A 1B model asked for JSON will sometimes still wrap it in prose, or open
 *  with a markdown fence, even under JSON mode. Pull out the first balanced
 *  object rather than trusting the whole string to parse — cheaper and far
 *  more reliable than a retry, which would double this call's latency
 *  inside an already latency-critical serial step. */
function extractJson(text: string): unknown {
  const trimmed = text.trim().replace(/^```(?:json)?/i, "").replace(/```$/, "");
  try {
    return JSON.parse(trimmed);
  } catch {
    // fall through to brace scanning
  }
  const start = trimmed.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < trimmed.length; i++) {
    const c = trimmed[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (c === "\\") {
      escaped = true;
      continue;
    }
    if (c === '"') inString = !inString;
    else if (!inString && c === "{") depth++;
    else if (!inString && c === "}" && --depth === 0) {
      try {
        return JSON.parse(trimmed.slice(start, i + 1));
      } catch {
        return null;
      }
    }
  }
  return null;
}

/** Runs the 1B aux model for a structured answer. Returns the raw parsed
 *  value (callers validate its shape — see index.ts) or null on any
 *  failure, so an aux step can never fail a request: every caller is
 *  expected to degrade to its own non-LLM behavior instead. */
export async function runAuxJson(
  env: Env,
  ctx: ExecutionContext,
  prompt: BuiltPrompt,
): Promise<unknown | null> {
  let result: any;
  try {
    result = await env.AI.run(AUX_MODEL as never, {
      messages: [
        { role: "system", content: prompt.system },
        { role: "user", content: prompt.user },
      ],
      max_tokens: AUX_MAX_TOKENS,
      temperature: AUX_TEMPERATURE,
      // json_object, NOT json_schema: the 1B aux model rejects schema mode
      // outright (Workers AI error 5025). Even json_object is only a hint
      // here — it is accepted but not enforced for this model — so the real
      // guarantees are the explicit format contract in the prompt and the
      // extractJson fallback plus per-field validation downstream.
      response_format: { type: "json_object" },
    } as never);
  } catch (err) {
    // Silent null is the contract callers rely on, but a persistently
    // failing aux model would otherwise be invisible — every mode simply
    // degrades to its non-LLM path and still answers.
    logUpstreamError("aux", AUX_MODEL, err);
    return null;
  }

  const neurons = typeof result?.usage?.neurons === "number" ? result.usage.neurons : 0;
  ctx.waitUntil(recordWorkersAiSpend(env.QUOTA, neurons || AUX_FALLBACK_NEURONS));

  // JSON mode may hand back an already-parsed object; without it, a string.
  // json_object mode hands back an already-parsed object on the 8B model;
  // on models where it's only a hint, a string that may be wrapped in prose.
  const raw = result?.response;
  if (raw && typeof raw === "object") return raw;
  if (typeof raw === "string") return extractJson(raw);
  return null;
}

/** Connects to Gemini eagerly and returns a ready-to-consume stream only on
 *  a successful (2xx) response — null on any failure, including Gemini's
 *  own free-tier rate limit (a live 429, observed directly in testing: a
 *  burst of requests trips Google's own per-minute cap independently of
 *  our daily call-count ladder). Structuring it this way — connect first,
 *  decide after — is what lets the caller fall through to Workers AI
 *  within the same request instead of the client's deadline having to
 *  catch a Gemini-side failure and show the weaker extractive answer. */
export async function connectGemini(
  env: Env,
  tier: Exclude<GeminiTier, "exhausted">,
  prompt: BuiltPrompt,
): Promise<ReadableStream<Uint8Array> | null> {
  const model = GEMINI_MODELS[tier];
  let res: Response;
  try {
    res = await fetch(
      `${GEMINI_ENDPOINT}/${model}:streamGenerateContent?alt=sse&key=${env.GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: prompt.system }] },
          contents: [{ parts: [{ text: prompt.user }] }],
          generationConfig: {
            maxOutputTokens: MAX_OUTPUT_TOKENS,
            temperature: TEMPERATURE,
            // Gemini 3.x's extended-thinking traces otherwise burn the
            // whole token budget as visible chain-of-thought before any
            // answer text — this is a short grounded-QA prompt with no
            // need for it.
            thinkingConfig: { thinkingBudget: 0 },
          },
        }),
      },
    );
  } catch (err) {
    logUpstreamError("/generate", `gemini:${model}`, err);
    return null; // network failure reaching Gemini at all
  }
  if (!res.ok || !res.body) {
    // e.g. the 429 above — Google's own per-minute cap, not our daily quota
    logUpstreamError("/generate", `gemini:${model}`, `HTTP ${res.status}`);
    return null;
  }
  return toWireStream(res.body);
}

function toWireStream(upstream: ReadableStream<Uint8Array>): ReadableStream<Uint8Array> {
  return new ReadableStream({
    async start(controller) {
      try {
        for await (const raw of readUpstreamSseData(upstream)) {
          let json: any;
          try {
            json = JSON.parse(raw);
          } catch {
            continue;
          }
          const text = (json.candidates?.[0]?.content?.parts || [])
            .map((p: any) => (typeof p.text === "string" ? p.text : ""))
            .join("");
          if (text) controller.enqueue(frame({ type: "delta", text }));
        }
      } catch (err) {
        controller.enqueue(
          frame({ type: "error", message: err instanceof Error ? err.message : String(err) }),
        );
      }
      controller.enqueue(frame({ type: "done" }));
      controller.close();
    },
  });
}
