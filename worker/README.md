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
| `POST /generate` | SSE stream of `{"type":"delta"\|"done"\|"error", ...}` frames |

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
dev — see the `remote` mode noted in wrangler's own binding summary — so
local testing against Llama does draw from the real daily quota.

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
