# RAG Playground — generation Worker

The only backend the app has. A standalone Cloudflare Worker, deployed
independently of the static site on GitHub Pages — this directory is its own
project (own `package.json`, own `wrangler.jsonc`), not part of the Next.js
build.

## Why it exists this way

The site is a static export (`output: "export"` in `next.config.ts`) served
from GitHub Pages — there's no server at runtime, so a Next.js API route
can't exist in this build. Real LLM generation needs somewhere to hold an
API key server-side, which is what this Worker is for.

## Routes

| Route | Purpose |
|---|---|
| `GET /health` | `{ok, workersAi:{tier,model}, gemini:{tier,model}}` — checked once by the client at mount |
| `POST /embed` | `{texts}` → int8-quantized 768-d vectors, in the binary `vectors.bin` wire format below |
| `POST /grade` | `{query, doc, chunks}` → `{verdicts:[{i, relevant, why}]}` — real relevance verdicts for corrective mode |
| `POST /plan` | `{query}` → `{subqueries, rationale}` — sub-query decomposition for agentic mode |
| `POST /generate` | SSE stream of `{"type":"delta"\|"done"\|"error", ...}` frames |

### `POST /embed` — wire format

One binary body, shared byte-for-byte with the precomputed sample sidecar
(`public/sample/*.vectors.bin`) so a single client decoder handles both:

```
[4 bytes]      uint32 LE — header byte length H
[H bytes]      UTF-8 JSON {"dim":768,"count":N,"scales":[N floats]}
[N*dim bytes]  Int8Array, row-major, one row per input text
```

Dequantization is **`v = q * scale`** — the ÷127 is already baked into each
scale at encode time. Dividing again yields vectors of norm ~0.007 instead
of ~1, which degrades retrieval silently rather than failing. Per-vector
scales (not one shared scale) cost a few bytes and recover most of the int8
recall loss. `tests/embeddings-wire.test.ts` pins all of this.

### `POST /grade` and `POST /plan` — the auxiliary calls

Both run on `AUX_MODEL` (see `src/budget.ts`) and both degrade to null on
any failure, so the client falls back to the behaviour that predates them.

`/grade` sends **one passage per call, in parallel** rather than batching.
This is deliberate and measured: asked to grade five passages in one call,
the model returned two verdicts and stopped — not truncation, it simply
declined to enumerate. Per-passage calls make the count structural.

Neither route increments the per-IP daily counter; only `/generate` does.
They are the first half of one logical question, and charging a visitor
twice would halve corrective and agentic mode's daily allowance.

`/plan` receives the query text only — never document content — so it
raises no provider-routing question at all.

## Provider routing (the trust boundary)

Two providers, chosen by the request's `doc` field and its `chunks` shape —
**not** by client assertion alone:

- `doc: "sample"` **and** every entry in `chunks` is a bare number → resolved
  against the Worker's own bundled copy of the sample (`src/sample.ts`,
  synced from `../public/sample/*.chunks.json` via `npm run sync:sample`),
  then sent to **Gemini**.
- Anything else — an upload, or a `doc: "sample"` claim that carries inline
  `{id, page, text}` objects instead of bare ids — is resolved from the
  request itself and sent to **Workers AI** (Llama), regardless of what
  `doc` claims.

This exists because free-tier Gemini permits Google to use prompts/responses
to improve their products; that's fine for the bundled OpenStax textbook
(public, CC BY 4.0) but not for a stranger's private upload. The only way to
reach Gemini is to name content this Worker already has bundled — a client
can never make it generate over arbitrary text.

## Cost — hard $0/month

Both KV-backed quota ladders downshift to a cheaper model before refusing
outright, so degradation to the client's offline extractive answer is the
worst case, never a bill:

```
Workers AI (uploads):  spent < 8,000 neurons/day → llama-3.1-8b-instruct-fast
                       spent < 9,500              → llama-3.2-1b-instruct
                       spent ≥ 9,500              → 503 → client degrades
Gemini (sample):       calls < 200/day  → gemini-3.5-flash
                       calls < 900      → gemini-3.5-flash-lite
                       else             → falls through to Workers AI
```

On the Workers **Free** plan (not Paid), exceeding the 10k-neuron daily
allocation makes further requests *fail*, not bill — verified against
Cloudflare's docs. The ladder above refuses at 9,500 specifically to stay
under that ceiling with margin.

Additional guards: a hard origin allowlist (`ALLOWED_ORIGINS` in
`wrangler.jsonc`), payload caps (`query ≤ 400` chars, `≤ 8` chunks, each
chunk's text truncated to 1,200 chars — enforced server-side, never trusting
the client), a 12/minute per-IP rate limit (the `ratelimits` binding), and a
60/day per-IP ceiling keyed on a salted hash (`src/budget.ts` — raw IPs are
never stored).

## Local development

```sh
cp .dev.vars.example .dev.vars   # fill in GEMINI_API_KEY + IP_HASH_SALT
npm install
npm run dev                      # wrangler dev, defaults to :8787
```

The app's `lib/llm.ts` defaults `NEXT_PUBLIC_LLM_ENDPOINT` to
`http://localhost:8787`, so no configuration is needed on that side for
local dev — just run `npm run dev:llm` from the repo root, which is this
same `npm run dev` under the hood.

`env.AI` (Workers AI) always hits the real remote service, even in local
dev — the binding is declared `remote: true` in `wrangler.jsonc` to state
that outright — so local testing against Llama draws from the real daily
quota. The opt-in probe at `../scripts/probes/worker-live.mts` exercises
`/health`, `/grade`, `/plan` and `/embed` against a running Worker and is
the fastest way to confirm the LLM routes still behave:

```sh
node ../scripts/probes/worker-live.mts
```

After changing bindings in `wrangler.jsonc`, regenerate the types:

```sh
npm run types      # writes worker-configuration.d.ts
```

`src/env.ts` re-exports the generated `Cloudflare.Env` rather than
restating the bindings, so a binding added to the config but forgotten in
the type — or vice versa — surfaces as a type error. CI typechecks against
the committed file.

## Observability

`observability` is enabled with `head_sampling_rate: 1` (keep everything —
this Worker's volume is bounded by the rate limiter and the daily ladders,
and a sampled-out request is exactly the one you want when chasing an
intermittent fallback).

Every request emits one structured JSON line: `route`, `status`, `ms`,
plus `provider` and `tier` once chosen, `reason` when a fallback or refusal
happened, and `n` for passage/subquery/verdict counts. Upstream failures
log separately with `upstream` and `error` so they can be filtered without
parsing every line.

This matters most for the failures that are *invisible from the outside*.
When Gemini returns a 503 the client still receives a perfectly good
answer — the Worker falls through to Workers AI within the same request —
so nothing surfaces that the preferred provider is down. That now shows up
as a pair of lines:

```json
{"route":"/generate","upstream":"gemini:gemini-3.5-flash","error":"HTTP 503"}
{"route":"/generate","status":200,"ms":270,"reason":"gemini connect failed, fell back","provider":"workers-ai","tier":"primary"}
```

**Not logged:** passage or chunk text. Uploaded documents are private —
the provider-routing boundary above exists to keep them from a provider
that trains on inputs, and writing them to logs would undo that from the
other end. The user's `query` is recorded (truncated to 120 characters),
since it is what makes a bad answer diagnosable.

## Deploying

```sh
npm run deploy
```

One-time setup (already done for this project, documented here for anyone
rebuilding it):

```sh
npx wrangler kv namespace create QUOTA
npx wrangler kv namespace create RATE
# paste the returned ids into wrangler.jsonc's kv_namespaces
npx wrangler secret put GEMINI_API_KEY
npx wrangler secret put IP_HASH_SALT
```

Deployment is manual/separate from the Pages build on purpose — coupling
them would mean every unrelated site change redeploys the Worker too, and
vice versa. After deploying, update `NEXT_PUBLIC_LLM_ENDPOINT` in
`.github/workflows/deploy.yml` to the Worker's URL if it ever changes.

## Regenerating the bundled sample

If `public/sample/introduction-intellectual-property.chunks.json` changes
(re-running the app's `npm run preprocess:sample`), sync the Worker's copy:

```sh
npm run sync:sample
```
