// Turns each provider's stream into the Worker's own wire format (see
// sse.ts). Gemini and Workers AI use two different chunk shapes on the
// wire — normalizing here means neither leaks past this file.
import {
  GEMINI_MODELS,
  WORKERS_AI_MODELS,
  recordWorkersAiSpend,
  type GeminiTier,
  type WorkersAiTier,
} from "./budget";
import type { Env } from "./env";
import type { BuiltPrompt } from "./prompts";
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
  } catch {
    return null; // network failure reaching Gemini at all
  }
  if (!res.ok || !res.body) return null; // e.g. the 429 above — not our daily quota, Google's own
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
