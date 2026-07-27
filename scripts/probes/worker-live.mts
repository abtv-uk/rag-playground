// Opt-in live probe for the Worker's LLM-backed routes — the oracles that
// CANNOT run in CI, because they need a running Worker and every call
// draws real Workers AI quota. Run manually:
//
//   npm run dev:llm            # wrangler dev on :8787 (needs CLOUDFLARE_API_TOKEN)
//   node scripts/probes/worker-live.mts
//   WORKER_URL=https://... node scripts/probes/worker-live.mts   # against prod
//
// What it checks, and why these exact cases: during Phase 3 the graders'
// failure modes were all *quality* failures invisible to schema validation
// — the 1B model marked Napster passages relevant to a crop-yields query,
// and batched grading returned 1 verdict for 3 passages. These probes are
// the distilled versions of the measurements that caught that. Passages
// are deliberately fictional (Zorbex/quixnium) so a pass can't come from
// the model's world knowledge.
//
// Self-contained on purpose: no imports from lib/ (extensionless imports
// don't resolve under plain node — vitest handles that for tests/, but
// this script must run with nothing but node itself).
//
// Exit code 0 = all probes passed; 1 = at least one failed. Cost per full
// run: ~4 aux-model calls + 1 embed, single-digit neurons.

const ENDPOINT = process.env.WORKER_URL || "http://127.0.0.1:8787";

let failures = 0;

function check(name: string, ok: boolean, detail: string): void {
  const mark = ok ? "PASS" : "FAIL";
  console.log(`  [${mark}] ${name} — ${detail}`);
  if (!ok) failures++;
}

async function post(path: string, body: unknown): Promise<Response> {
  return fetch(ENDPOINT + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

const tokenize = (s: string): string[] =>
  (s.toLowerCase().match(/[a-z0-9][a-z0-9'-]{1,}/g) || []).filter((w) => w.length > 2);

const jaccard = (a: string[], b: string[]): number => {
  const A = new Set(a);
  const B = new Set(b);
  const inter = [...A].filter((x) => B.has(x)).length;
  return inter / (A.size + B.size - inter || 1);
};

console.log(`probing ${ENDPOINT}\n`);

// ---------- /health ----------
try {
  const res = await fetch(ENDPOINT + "/health", { signal: AbortSignal.timeout(6000) });
  const data = res.ok ? await res.json() : null;
  check("/health", !!data?.ok, res.ok ? `workersAi=${data.workersAi?.tier}, gemini=${data.gemini?.tier}` : `HTTP ${res.status}`);
} catch (err) {
  check("/health", false, `unreachable: ${err instanceof Error ? err.message : err}`);
  console.log("\nWorker unreachable — aborting remaining probes.");
  process.exit(1);
}

// ---------- /grade: on-topic passage must pass ----------
{
  const res = await post("/grade", {
    query: "What powers a Zorbex engine?",
    doc: "upload",
    chunks: [
      {
        id: 1,
        page: 1,
        text: "Quixnium crystals are valuable because they emit a faint blue glow and can power a Zorbex engine for up to eleven years without needing to be replaced.",
      },
    ],
  });
  const data = res.ok ? await res.json() : null;
  const v = data?.verdicts?.[0];
  check(
    "/grade on-topic",
    v?.relevant === true,
    res.ok ? `relevant=${v?.relevant} why="${v?.why}"` : `HTTP ${res.status}`,
  );
}

// ---------- /grade: off-topic passage must be rejected ----------
{
  const res = await post("/grade", {
    query: "how does weather affect crop yields",
    doc: "upload",
    chunks: [
      {
        id: 1,
        page: 1,
        text: "The music-sharing service faced copyright infringement lawsuits from record labels, and the industry eventually shifted to legal DRM-free download stores.",
      },
    ],
  });
  const data = res.ok ? await res.json() : null;
  const v = data?.verdicts?.[0];
  check(
    "/grade off-topic",
    v?.relevant === false,
    res.ok ? `relevant=${v?.relevant} why="${v?.why}"` : `HTTP ${res.status}`,
  );
}

// ---------- /plan: compound query decomposes into on-topic sub-queries ----------
{
  const query =
    "What is the difference between a patent and a trademark, and how long does each last?";
  const res = await post("/plan", { query });
  const data = res.ok ? await res.json() : null;
  const subs: string[] = Array.isArray(data?.subqueries) ? data.subqueries : [];
  const countOk = subs.length >= 1 && subs.length <= 3;
  const onTopic =
    subs.length > 0 && subs.some((s) => jaccard(tokenize(s), tokenize(query)) > 0);
  check(
    "/plan",
    countOk && onTopic,
    res.ok
      ? `${subs.length} subqueries: ${JSON.stringify(subs)} rationale="${data?.rationale}"`
      : `HTTP ${res.status}`,
  );
}

// ---------- /embed: one vector, unit norm, sane latency ----------
{
  const t0 = Date.now();
  const res = await post("/embed", { texts: ["a trade secret is confidential business information"] });
  const ms = Date.now() - t0;
  if (!res.ok) {
    check("/embed", false, `HTTP ${res.status}`);
  } else {
    const buf = await res.arrayBuffer();
    const view = new DataView(buf);
    const headerLen = view.getUint32(0, true);
    const header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)),
    ) as { dim: number; count: number; scales: number[] };
    const int8 = new Int8Array(buf, 4 + headerLen);
    let n = 0;
    for (let j = 0; j < header.dim; j++) {
      const v = int8[j] * header.scales[0]; // dequant is q*scale — never /127 again
      n += v * v;
    }
    const norm = Math.sqrt(n);
    check(
      "/embed",
      header.dim === 768 && header.count === 1 && norm > 0.95 && norm < 1.05,
      `dim=${header.dim} norm=${norm.toFixed(4)} in ${ms}ms`,
    );
  }
}

console.log(failures ? `\n${failures} probe(s) FAILED` : "\nall probes passed");
process.exit(failures ? 1 : 0);
