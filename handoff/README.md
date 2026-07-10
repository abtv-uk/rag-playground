# Handoff: RAG Pipeline Playground

## Overview

An interactive, single-screen "RAG playground" teaching tool / portfolio showcase. The user loads a single document, then explores how four different RAG (retrieval-augmented generation) architectures retrieve from it. A canvas-based pipeline diagram — styled after node-flowchart RAG infographics — lights up stage by stage as a query travels through it. Two hero moments: **indexing** (document splits into pages → chunks fly into a 2D vector scatter; for Hybrid, a 3D knowledge graph draws itself) and **query** (the pipeline path glows and pulses per-architecture, answer streams into an output panel with a retrieval trace).

Target stack (per product owner): **Next.js + TypeScript + Tailwind + Framer Motion**; the diagram itself is best implemented as a `<canvas>` 2D rendering (as the prototype does) or SVG + Framer Motion.

## About the Design Files

The file in this bundle (`RAG Playground.dc.html`) is a **design reference created in HTML** — a working prototype showing intended look and behavior, not production code to copy directly. The task is to **recreate this design in the target codebase's environment** using its established patterns. The prototype opens directly in a browser; interact with it to see every state. All pipeline drawing lives in one canvas render function driven by `requestAnimationFrame`; all chrome (nav, sidebars, query bar, output) is DOM.

## Fidelity

**High-fidelity.** Colors, typography, spacing, animation timings and interaction behavior are final design intent. Recreate pixel-perfectly, adapting to the codebase's component conventions.

## Screen Anatomy (single workspace)

CSS grid: `grid-template-columns: 250px 1fr 360px; grid-template-rows: 58px 1fr 96px;`

1. **Top nav** (row 1, full width): logo mark + "RAG PLAYGROUND" wordmark (JetBrains Mono 12px, 0.14em tracking), segmented tab switcher for the 4 architectures, right-aligned status chip (`no document / indexing… / ready / retrieving… / answered`).
2. **Left sidebar** (250px, collapsible to 62px): loaded document card (filename, size, chunk/entity counts), 4×2 page-thumbnail grid, Re-index + Clear buttons, "Active architecture" blurb card, Light/Dark toggle pinned at bottom.
3. **Center canvas** (the star): pipeline diagram, empty-state upload dropzone overlay, live caption pill (bottom-left) describing the current stage.
4. **Query bar** (below canvas, 96px): suggestion chips row ("TRY …"), input with `● QUERY` prefix badge and accent-colored Ask → button.
5. **Right output sidebar** (360px): ANSWER section (streaming text with blinking block cursor), RETRIEVAL TRACE (source cards: chunk/entity/tool/rejected kinds with relevance score bars).

## The Four Architectures (tabs, in order)

Each tab has its own accent hue and its own diagram topology, sharing one visual grammar (icon nodes + routed orthogonal arrows + inset panels):

1. **Basic** (`#5B8DEF` blue, sub "BASELINE") — USER QUERY → EMBEDDING → VECTOR DB panel (DATA SOURCE doc node feeds it from above) → down into PROMPT TEMPLATE (query also routes directly to prompt down the left edge) → LLM → OUTPUT.
2. **Hybrid** (`#9B7BF0` violet, "VECTOR + GRAPH") — same skeleton, but two panels side by side: VECTOR DB (context 1) and KNOWLEDGE GRAPH (context 2). Embedding feeds the vector panel; a second route arcs over the top into the graph. Both panel outputs visibly **merge** into one pipe entering the prompt.
3. **Corrective** (`#2BB673` emerald, "GRADE · CORRECT") — vector panel feeds a RELEVANCE GRADING panel (chunk rows graded ✓/✕ with score bars, red `#d2655e` for rejects, dashed borders), a **dashed return arrow** back to the vector store (re-retrieval), SEARCH / CORRECT pills, then graded context → prompt → LLM.
4. **Agentic** (`#F2A93B` amber, "PLAN · LOOP") — AGENT node center (118×50, with live status text: planning / retrieval loop / refining query / reasoning), MEMORY (SHORT, LONG) and PLANNING (ReAct, CoT) chip groups above feeding into it, MCP TOOLS bracket right (SEARCH, LOCAL, CLOUD chips), dashed loop ellipse around the agent while looping, vector panel bottom-left with a dashed return edge, then agent → LLM → OUTPUT. No prompt-template node (agent talks to LLM directly).

Switching tabs while a query/answer exists **re-runs the same query** through the new architecture (no page reload; diagram cross-fades naturally since only the canvas redraws).

## Visual Grammar

- **Nodes**: 46×46 rounded-rect tiles (radius 13), 1.2px hairline border; lit = accent border 1.8px + accent glow (shadowBlur 18 @ 35% accent). Line-drawn icons inside (person = query, 3×3 dot grid = embedding, page = data source, speech bubble = prompt, 8-point star = LLM). Labels: JetBrains Mono 600 8.5px, uppercase, below the tile (wraps to 2 lines); lit nodes show a 7.5px accent status subline (e.g. `768-d`, `ctx #14 #22 + Q`, `generating…`).
- **Pipes**: orthogonal polylines with 12px rounded corners and a 7px arrowhead. Idle: 1.6px hairline (`rgba(255,255,255,0.20)` dark). Lit: 2.6px accent @ 92% + 9px accent glow; while flowing, an overlaid white dash (`[3,9]`, offset animated ~ -now/22 % 12) plus 3 white particle dots (2.6px, accent ring) traveling the route (period 900ms). Dashed pipes (`[4,4]`) = return/loop edges.
- **Panels** (vector store / graph / grading): rounded rect (radius 14), translucent fill `rgba(255,255,255,0.04)` dark / `rgba(255,255,255,0.55)` light, hairline stroke; lit = accent stroke @ 50% + soft glow. Title JetBrains Mono 600 10px + faint 9px subtitle, top-left, ellipsis-truncated to panel width − 28px. Content clipped to panel bounds.
- **Query origin**: the DOM query bar is the source — a pipe starts at bottom-center of the canvas, runs along the bottom, up the left gutter, into the USER QUERY node's left side.
- **OUTPUT ›** text tag at the canvas right edge, at LLM height, pointing toward the output sidebar.

## Hero Moment 1 — Indexing (~4.4s, on load / re-index)

Everything else dims to 16% alpha. Sequence (staged captions in the pill):
1. 0–12%: document card (38×48, accent border, accent glow, text lines) scales in at the source position; caption "separating 15 pages".
2. 8–38%: an accent scan line sweeps down the document.
3. 18–40%: three page cards fan out (±34px, ±0.12rad rotation) then fade; caption "splitting into 64 chunks".
4. 34–96%: 60 chunk dots fly from the document into the VECTOR DB panel, each with a random delay, ease-out cubic, settling into 5 similarity clusters; caption "embedding chunks → 768-d".
5. Hybrid only, 52–98%: knowledge-graph nodes scale in one by one and edges grow point-to-point; caption "extracting entities → graph".
For Basic, the source is the DATA SOURCE node; the doc→panel edge stays lit and flowing during indexing.

## Hero Moment 2 — Query

Steps advance every **560ms**; each step lights nodes + edges (cumulative — the path history stays lit) and only the **current** step's edges carry particles. Sequences:

- Basic: qn → embed → vector (panel lit) → highlight relevant chunks (6 dots enlarge to 3.6px accent with soft pulse halo; others dim) → prompt (vector + query edges) → llm → output/stream.
- Hybrid: … both panels light in parallel → graph active-subgraph highlights (6 of 11 entities + connecting edges in accent) → merge into prompt → …
- Corrective: … retrieve → grade rows animate ✓/✕ → SEARCH/CORRECT pills light + dashed re-retrieve edge → prompt → …
- Agentic: qn → agent → memory+planning chips light → agent→vector retrieve → dashed return + loop ellipse (sweeping dashed arc while querying) → tools light → llm → output.

When the answer starts streaming: output sidebar shows word-by-word streaming (34ms/word) with a blinking accent block cursor; retrieval trace cards slide in just before. **All looping motion stops once answered** — the path stays lit but frozen; the 3D graph only auto-rotates while querying or hovered.

## Interactive 3D Knowledge Graph (Hybrid)

11 named entity nodes (Transformer, Self-Attention, Multi-Head, Encoder, Decoder, Pos. Encoding, Feed-Forward, BLEU, WMT 2014, Adam, Dropout) on a unit sphere, Y-axis rotation projected with perspective (f=2.9, separate X/Y radii to fill the panel). 13 edges. Hover: cursor→pointer, node + its pathways stay lit, everything else dims to 30%, tooltip card (180px, surface bg, accent border, shadow) shows label, "N LINKS · ENTITY", and a description. Click pins the tooltip (adds "· PINNED"); click again or elsewhere unpins. Rotation slows ~2.7× while interacting. Labels: mono 8.5px below nodes, clamped inside the panel; on panels narrower than 180px labels show only for hovered/neighbor nodes.

## State Management

Phases: `empty → indexing → ready → querying → answered`. State: `rag` (tab), `phase`, `query`, `answer` (streaming), `sources[]`, `expanded` (sidebar), `dark`. The canvas reads phase + elapsed time each frame; query step index = `floor(elapsed / 560ms)`. Keep `lastQuery` so tab switches replay it.

## Design Tokens

**Accents**: Basic `#5B8DEF` · Hybrid `#9B7BF0` · Corrective `#2BB673` · Agentic `#F2A93B` · Reject `#d2655e`.

**Dark theme (default)**: bg `#0b0d11`; grid lines `rgba(255,255,255,.045)` at 26px; bars/sides `rgba(18,21,27,.74/.58)` + blur(10px); surface `#161b22`, surface2 `#1d232c`, canvas node fill `#1c232d`; ink `#e9ecf1`, ink2 `#b6bdc8`, sub `#838c9a`, faint `#626c79`; hairline `rgba(255,255,255,.11)`.
**Light theme**: bg `#FAFBFC`; grid `rgba(17,21,27,.035)`; surfaces white; ink `#11151b`; hairline `rgba(17,21,27,.10)`. Theme via CSS variables on the root + a parallel token set for canvas colors.

**Typography**: Space Grotesk (400–700) for UI text/headings; JetBrains Mono (400–600) for labels, badges, metadata, captions. Node labels 8.5px/600 mono uppercase; panel titles 10px/600 mono; answer body 14px/1.62 Space Grotesk.

**Radii**: nodes/panels 13–14px, cards 9–12px, chips/pills 8px. **Grid paper**: 26×26px hairline grid on the whole workspace background.

## Interactions & Behavior

- Empty state: dashed dropzone (380px, "Drop a document to begin", PDF·TXT·MD) + "Load sample document" primary button. Real implementation should accept an actual file drop and run real chunking/embedding, or keep the scripted demo.
- Query input disabled until a document is indexed; Enter or Ask submits; suggestion chips fill + submit.
- Sidebar collapse animates width 250→62px; document panel hides, theme toggle remains.
- Canvas is fully responsive: layout positions are fractions of canvas W/H; panels clip their content; grade panel drops rows/stacks pills below ~150px width; graph labels hide below 180px panel width.
- Respect `prefers-reduced-motion`: skip particles/pulses, jump to lit end-states.

## Assets

None — all iconography is line-drawn (canvas strokes in the prototype; reproduce as inline SVG). Fonts from Google Fonts: Space Grotesk, JetBrains Mono.

## Files

- `RAG Playground.dc.html` — the complete working prototype (open in a browser). Template = DOM chrome; the `Component` class inside the `data-dc-script` script tag contains all canvas drawing (`diagram()` = per-architecture node/edge layout, `steps()` = query step sequences, `draw()` = render loop, `drawIndexing()` = indexing hero, `drawGraph3D()` = interactive graph, `drawGradePanel()` / `drawAgentScene()` = type-specific panels) and the state machine (`runIndex()`, `runQuery()`, `streamAnswer()`).
