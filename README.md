# RAG Playground

An interactive, single-screen playground for exploring how four retrieval-augmented generation (RAG) architectures answer the same question over the same document.

Load a document (a scripted demo of *Attention Is All You Need*), watch it get indexed — pages split into chunks, chunks fly into a vector scatter, and (for Hybrid) a 3D knowledge graph draws itself — then ask a question and watch the query travel through a live, canvas-rendered pipeline diagram that lights up stage by stage while the answer streams into the output panel with a retrieval trace.

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
npm run dev     # http://localhost:3000
```

`npm run build` produces a fully static production build.

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

The retrieval itself is a scripted demo (no API keys, no backend): answers, sources and grading results are canned per architecture. The pipeline visualization, state machine and interactions are the real subject.

## Design handoff

This app is a high-fidelity recreation of a design prototype produced with Claude Design. The original handoff bundle — including the self-contained HTML prototype you can open directly in a browser — lives in [`handoff/`](handoff/), with the spec in [`handoff/README.md`](handoff/README.md).
