// Dual quota ladders, so the Worker can only ever cost $0: on the Workers
// Free plan, exceeding the 10k-neuron daily allocation makes requests fail
// rather than bill (verified against Cloudflare's docs), and this ladder
// downshifts to a cheaper model well before that ceiling, then refuses
// outright — never billing, and giving the client a clean signal to
// degrade to the extractive answer.
//
// KV read-then-write is not atomic, so concurrent requests can race past a
// ceiling by a small margin. Acceptable for a free demo's scale; a Durable
// Object would close the race if this ever needed to be exact.

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
