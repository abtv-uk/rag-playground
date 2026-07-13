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
npm run dev     # http://localhost:3100 (or 3000 if unconfigured)
```

`npm run build` produces a fully static production build (the site is deployed as a static export — see [Provenance](#provenance)).

### Real document ingestion

The sample loads instantly: its chunks are precomputed at build time (`npm run preprocess:sample` regenerates `public/sample/*.chunks.json` from the PDF using the exact same chunking code the app runs), so choosing the sample skips the multi-second client-side PDF parse while still exercising the full retrieval pipeline. The original PDF is served alongside the app and previewable from the empty state ("preview PDF ↗") or by clicking the document name in the sidebar.

Or drop in your own document: drag & drop, click to browse, or paste a URL (scraped via a reader proxy). PDFs are parsed client-side with `pdfjs-dist`; TXT/MD are read directly. The document is chunked, boilerplate (tables of contents, indexes, quiz blocks) is filtered out, and each of the four tabs runs a genuinely different retrieval strategy over the real chunks — TF-IDF-style lexical scoring, entity-graph boosting (Hybrid), grade-and-re-retrieve (Corrective), or a query-refinement loop (Agentic) — with real scores, snippets and an extractive answer. No API key required for this path; it's fully client-side and works on the deployed static site.

### Real LLM-generated answers (optional, local dev only)

For a sharper answer than the offline extractive fallback, run a local proxy that calls the Gemini API with the same retrieved chunks as grounding context:

```sh
cp .env.example .env        # add your GEMINI_API_KEY
npm run dev:llm             # in a second terminal, alongside `npm run dev`
```

This never touches the deployed site. `output: "export"` (static site, no server) means a Next.js API route literally cannot exist in this build — so `server/llm-proxy.mjs` is a standalone Node process that the bundler never sees, holding the API key server-side. The client fetches `localhost:8787` with a short timeout and silently falls back to the extractive answer if the proxy isn't running, which is what happens for every real visitor of the public site — there is no code path by which the key can reach a browser bundle.

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
  data.ts              seeded demo data (60 chunk dots, 11-entity graph)
  constants.ts         design tokens, copy, scripted answers/sources
```

React owns the state machine and DOM chrome; the renderer runs its own `requestAnimationFrame` loop and reads a mutable view object that React updates on state changes — so streaming answers and step timers never force per-frame React renders.

Every document — the bundled sample included — runs the same real pipeline. The only thing scripted is the seeded placeholder scene shown behind the empty state before anything is loaded.

## Provenance

This app is a high-fidelity recreation of a design prototype produced with Claude Design, ported to Next.js from a single-file HTML/canvas handoff.
