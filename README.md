# RAG Playground

**Live demo → https://abtv-uk.github.io/rag-playground/**

An interactive, single-screen playground for exploring how four retrieval-augmented generation (RAG) architectures answer the same question over the same document.

Load a document — the bundled sample is OpenStax's *Introduction to Intellectual Property* (CC BY 4.0), a real 201-page textbook — and watch it get indexed — pages split into chunks, chunks fly into a vector scatter, and (for Hybrid) a 3D knowledge graph draws itself — then ask a question and watch the query travel through a live, canvas-rendered pipeline diagram that lights up stage by stage while the answer streams into the output panel with a retrieval trace.

## The four architectures

| Tab | Accent | What it demonstrates |
|---|---|---|
| **Basic** | blue | The baseline: embed the query → vector search → stuff top chunks into a prompt → generate |
| **Hybrid** | violet | Vector store **and** knowledge graph retrieved in parallel, contexts merged into one prompt |
| **Corrective** | emerald | Retrieved chunks are graded ✓/✕ for relevance; rejects trigger re-retrieval before generation |
| **Agentic** | amber | An agent plans, loops retrieval, refines its own query and calls MCP tools before answering |

Switching tabs while an answer exists re-runs the same query through the new architecture, so the pipelines are directly comparable.

Extras worth trying:

- **Hover / click the knowledge graph** (Hybrid tab) — nodes highlight their pathways, tooltips describe each entity, click pins the tooltip.
- **Re-index** replays the indexing hero moment; **light/dark theme** toggle is pinned to the sidebar.
- `prefers-reduced-motion` is respected: particles and pulses are skipped and pipelines jump to their lit end-states.

## Running it

```sh
npm install
npm run dev     # http://localhost:3000 (pass `-- --port 3100` to match .claude/launch.json)
```

`npm run build` produces a fully static production build (the site is deployed as a static export — see [Provenance](#provenance)).

### Real document ingestion

The sample loads instantly: its chunks are precomputed at build time (`npm run preprocess:sample` regenerates `public/sample/*.chunks.json` from the PDF using the exact same chunking code the app runs), so choosing the sample skips the multi-second client-side PDF parse while still exercising the full retrieval pipeline. The original PDF is served alongside the app and previewable from the empty state ("preview PDF ↗") or by clicking the document name in the sidebar.

Or drop in your own document: drag & drop, click to browse, or paste a URL (scraped via a reader proxy). PDFs are parsed client-side with `pdfjs-dist`; TXT/MD are read directly. The document is chunked, boilerplate (tables of contents, indexes, quiz blocks) is filtered out, and each of the four tabs runs a genuinely different retrieval strategy over the real chunks — with real scores and snippets.

Retrieval is **semantic**, not just lexical. Chunks and the query are embedded with `bge-base-en-v1.5` through the Worker's `/embed` route, and the dense ranking is fused with TF-IDF scoring via Reciprocal Rank Fusion; Hybrid adds entity-graph boosting, Corrective grades each chunk with a real LLM verdict and re-retrieves what it rejects, Agentic decomposes the question into sub-queries and merges their results. The sample ships pre-embedded, so loading it costs **zero** embedding calls — only one per question, for the query itself.

Every dense path is guarded: if the Worker is unreachable or out of quota, retrieval falls back to the lexical scoring that predates it and the app keeps working, degraded but silent. That fallback is the reason the demo still functions offline.

### Real, grounded, cited answers

Every public visitor gets an actual generated answer, not the extractive fallback — a small Cloudflare Worker (`worker/`) turns the retrieved chunks into a prompt, streams the response back over SSE, and the client renders `[1][2]`-style citations that scroll their source card into view on click. Two providers, chosen by which document is loaded:

- **The bundled sample** routes to **Gemini** (better prose). The Worker ships its own copy of the sample's chunks and only ever accepts *chunk ids* for this path — never client-supplied text — so there's no way to spend Gemini quota generating over arbitrary content.
- **Anything you upload or paste a URL for** routes to **Workers AI** (Llama), which has no data-training clause, so a private document's text never reaches a provider that might use it to improve their models.

If neither provider has quota left for the day, or the Worker is unreachable, the app falls back to the offline extractive answer and says so (`EXTRACTIVE FALLBACK`) rather than failing silently. The whole thing runs on Cloudflare's free tier — see `worker/README.md` for the cost model, quota ladders, and deployment.

For local development, run the Worker alongside the app:

```sh
cp worker/.dev.vars.example worker/.dev.vars   # add your GEMINI_API_KEY + a random IP_HASH_SALT
npm run dev:llm                                # wrangler dev, in a second terminal
```

`npm run dev` already defaults to `http://localhost:8787` for the Worker endpoint, so nothing else needs configuring locally.

### Honest visuals

The vector scatter plots the document's **actual** embeddings: dot positions are PCA components 1–2 of the real 768-d vectors, tints come from k-means over the full-dimensional vectors (not the projection — clustering after discarding most of the variance would group points that merely look close on screen), and the query is drawn as a ring in the same basis so you can watch it land inside the retrieved cluster.

The panel reports what those two axes actually capture — typically around 11% for a real document, which is simply what two components out of 768 explain. Stating it is the point: an earlier version positioned dots by `page % 5` plus a seeded PRNG, which looked like structure while meaning nothing. Where a document has no embeddings yet, the placeholder layout returns and the tints stay neutral rather than implying clusters that were never computed.

### Checks

```sh
npm test               # the measurement oracles: retrieval, projection, wire format, gating
npm run typecheck      # app
npm run typecheck:worker
```

CI gates the Pages deploy on all three — `main` cannot deploy without them passing.

The tests are deliberately the *measurements* that caught real defects during development, not coverage for its own sake: the int8 dequantization that would silently destroy retrieval, the prefix matcher that put trademark chunks in a trade-secret query's top results, the corrective gate that could never reject anything. Assertions on real-document fixtures are ranges and inequalities, so re-embedding the sample re-validates the suite instead of breaking it.

Oracles that need a live Worker — grader verdict quality, planner decomposition — can't run offline and draw real Workers AI quota, so they live as an opt-in probe:

```sh
npm run dev:llm                              # wrangler dev, in a second terminal
node scripts/probes/worker-live.mts          # or WORKER_URL=https://… for production
```

## Stack & architecture

Next.js 15 (App Router) · React 19 · TypeScript · Tailwind CSS 4 · Framer Motion.

The screen is a CSS grid of DOM chrome around one `<canvas>` that renders the pipeline diagram at 60fps:

```
components/
  RagPlayground.tsx    grid shell wiring state → sections
  TopNav.tsx           logo, architecture tabs, status chip
  DocumentSidebar.tsx  document card, page thumbnails, active-architecture blurb
  CanvasStage.tsx      the canvas + empty-state dropzone + live caption pill
  QueryBar.tsx         suggestion chips + query input
  OutputPanel.tsx      streaming answer + retrieval trace cards
hooks/
  usePlayground.ts     state machine: empty → indexing → ready → querying → answered
lib/
  renderer.ts          canvas engine: diagram layouts, routed pipes + particles,
                       vector scatter, interactive 3D graph, grading panel,
                       agent scene, indexing/query hero moments
  steps.ts             per-architecture query step sequences (560 ms per step)
  document.ts          ingestion: PDF/TXT/MD parsing, chunking, boilerplate
                       filtering, sample loading (chunks + vector sidecar)
  retrieval.ts         the four strategies, RRF fusion, entity graph,
                       corrective grading, extractive answers, trace cards
  embeddings.ts        int8 vectors.bin codec + /embed client + cosine
                       ranking; returns null rather than throwing
  projection.ts        power-iteration PCA and k-means — no dependency,
                       fully deterministic (no Math.random anywhere)
  scene.ts             SceneData: scatter layout, real projection, entity
                       graph, grade rows
  data.ts              seeded demo data (60 chunk dots, 11-entity graph)
  constants.ts         design tokens, copy, scripted answers/sources
  llm.ts               client for the Worker: SSE generation, /grade
                       verdicts, /plan decomposition, health check —
                       never throws, always degrades cleanly
worker/                standalone Cloudflare Worker — the only backend this
                       app has, deployed independently of the static site;
                       see worker/README.md
```

React owns the state machine and DOM chrome; the renderer runs its own `requestAnimationFrame` loop and reads a mutable view object that React updates on state changes — so streaming answers and step timers never force per-frame React renders.

Every document — the bundled sample included — runs the same real pipeline. The only thing scripted is the seeded placeholder scene shown behind the empty state before anything is loaded.

## Provenance

This app is a high-fidelity recreation of a design prototype produced with Claude Design, ported to Next.js from a single-file HTML/canvas handoff.
