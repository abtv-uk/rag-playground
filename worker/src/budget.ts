// Dual quota ladders, so the Worker can only ever cost $0: on the Workers
// Free plan, exceeding the 10k-neuron daily allocation makes requests fail
// rather than bill (verified against Cloudflare's docs), and this ladder
// downshifts to a cheaper model well before that ceiling, then refuses
// outright — never billing, and giving the client a clean signal to
// degrade to the extractive answer.
//
// KV read-then-write is not atomic, and KV reads are additionally
// eventually consistent across regions — so these counters can undercount
// under concurrent or geographically spread traffic, letting a request or
// two past a ceiling.
//
// That is deliberate, not unexamined. This ladder is NOT what keeps the
// Worker free: the Workers Free plan itself fails requests past its daily
// neuron allocation rather than billing for them. The ladder's job is to
// degrade *gracefully* — downshift to a cheaper model, then refuse with a
// clean signal the client can fall back on — before traffic hits that hard
// platform wall. The ceilings below sit well under the platform limit
// precisely so a small overshoot is absorbed by the margin.
//
// A SQLite-backed Durable Object would make the count exact and is free-
// plan eligible. It was considered and rejected: it puts a round-trip on
// every request's critical path (/grade and /plan already sit in front of
// generation, where latency is the scarce resource), and it introduces a
// second free-tier ceiling — DO compute — that can itself start failing.
// Trading a bounded, harmless overshoot for a new failure mode on the hot
// path is a bad deal at this scale. Revisit if traffic ever makes the
// overshoot material rather than theoretical.

export type WorkersAiTier = "primary" | "downshift" | "exhausted";
export type GeminiTier = "flash" | "flash-lite" | "exhausted";

const WORKERS_AI_PRIMARY_CEILING = 8000;
const WORKERS_AI_DOWNSHIFT_CEILING = 9500;
const GEMINI_FLASH_CEILING = 200;
const GEMINI_FLASH_LITE_CEILING = 900;

// two days: comfortably past UTC midnight rollover, so a slow cleanup never
// drops a counter mid-day
const COUNTER_TTL_SECONDS = 60 * 60 * 24 * 2;

export const WORKERS_AI_MODELS: Record<Exclude<WorkersAiTier, "exhausted">, string> = {
  primary: "@cf/meta/llama-3.1-8b-instruct-fast",
  downshift: "@cf/meta/llama-3.2-1b-instruct",
};

// @cf/baai/bge-base-en-v1.5 pricing: $0.067 / M input tokens. Neurons are
// Workers AI's unified cost unit ($0.011 / 1,000 neurons), so
// neurons-per-token = (0.067 / 1000) / 0.011 * 1000 = 6.09 neurons per 1k
// tokens, i.e. 6.09/1000 per token. Used to record embedding's *actual*
// reported cost (from the response's usage.prompt_tokens) against the same
// daily ladder generation draws on — both spend the same Workers AI budget.
export const BGE_NEURONS_PER_TOKEN = 6.09 / 1000;

// Auxiliary reasoning (relevance grading, and later query planning).
//
// This was meant to be the 1B model, to keep a 2-call mode near 1.2x the
// cost of a 1-call one. It cannot be. Measured on real sample passages, 1B
// marked EVERY passage relevant in both directions — including three
// Napster passages judged relevant to "how does weather affect crop
// yields", justified as "describes the relationship between weather and
// crop yields" — and for most passages it echoed the format example from
// the prompt verbatim instead of reasoning. Passing everything is the
// precise failure this grader exists to prevent (it is what made a
// relevance-only threshold useless in Phase 2), so a cheaper aux model
// here would be worse than shipping no grader at all.
//
// Real cost on 8B: ~11 neurons to grade five passages against ~13 for a
// generation call, so corrective is ~1.85x a single-call mode rather than
// 1.2x. That still allows ~330 corrective queries/day against the 8,000
// ladder — and for the bundled sample the generation half goes to Gemini,
// so grading is corrective's only draw on this budget.
//
// Deliberately NOT tier-dependent: downshifting this to 1B under load
// would silently turn grading back into "everything passes". When the
// ladder is exhausted the route refuses instead, and the client falls back
// to its own cosine floor.
export const AUX_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast";

export const GEMINI_MODELS: Record<Exclude<GeminiTier, "exhausted">, string> = {
  flash: "gemini-3.5-flash",
  "flash-lite": "gemini-3.5-flash-lite",
};

function utcDateKey(): string {
  return new Date().toISOString().slice(0, 10).replace(/-/g, "");
}

async function readCounter(kv: KVNamespace, key: string): Promise<number> {
  const v = await kv.get(key);
  return v ? Number(v) || 0 : 0;
}

async function bumpCounter(kv: KVNamespace, key: string, delta: number): Promise<void> {
  const current = await readCounter(kv, key);
  await kv.put(key, String(current + delta), { expirationTtl: COUNTER_TTL_SECONDS });
}

export async function workersAiTier(kv: KVNamespace): Promise<WorkersAiTier> {
  const spent = await readCounter(kv, `wai:${utcDateKey()}`);
  if (spent < WORKERS_AI_PRIMARY_CEILING) return "primary";
  if (spent < WORKERS_AI_DOWNSHIFT_CEILING) return "downshift";
  return "exhausted";
}

/** Records the actual neuron cost reported by Workers AI's final stream
 *  chunk, not an estimate — see the usage.neurons field captured in
 *  index.ts's stream transform. */
export async function recordWorkersAiSpend(kv: KVNamespace, neurons: number): Promise<void> {
  await bumpCounter(kv, `wai:${utcDateKey()}`, neurons);
}

export async function geminiTier(kv: KVNamespace): Promise<GeminiTier> {
  const calls = await readCounter(kv, `gem:${utcDateKey()}`);
  if (calls < GEMINI_FLASH_CEILING) return "flash";
  if (calls < GEMINI_FLASH_LITE_CEILING) return "flash-lite";
  return "exhausted";
}

export async function recordGeminiCall(kv: KVNamespace): Promise<void> {
  await bumpCounter(kv, `gem:${utcDateKey()}`, 1);
}

// ---------- per-IP daily ceiling ----------
// Legitimate exploration of the playground stays well under this; it exists
// to bound a single visitor's worst-case draw on the shared daily budgets
// above, not to be a precise abuse filter (see the per-minute ratelimit
// binding in index.ts for that).

const IP_DAILY_LIMIT = 60;

export async function hashIp(ip: string, salt: string): Promise<string> {
  const bytes = new TextEncoder().encode(`${salt}:${ip}`);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Returns false (and does not increment) once the caller's daily ceiling
 *  is reached. Never stores the raw IP — only its salted hash. */
export async function checkAndIncrementIpDaily(
  kv: KVNamespace,
  ipHash: string,
): Promise<boolean> {
  const key = `ip:${utcDateKey()}:${ipHash}`;
  const count = await readCounter(kv, key);
  if (count >= IP_DAILY_LIMIT) return false;
  await kv.put(key, String(count + 1), { expirationTtl: COUNTER_TTL_SECONDS });
  return true;
}
