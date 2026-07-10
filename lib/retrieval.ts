// Retrieval over uploaded documents. Four genuinely different strategies
// share one lexical (TF-IDF-style) scoring core:
//   basic      — single pass, top-k
//   hybrid     — lexical pass merged with an entity-graph boost
//   corrective — grade top chunks, reject low scores, re-retrieve via PRF
//   agentic    — two-pass loop: retrieve, refine the query (PRF), retrieve again
// Plus: extractive answers, entity/co-occurrence graph extraction, trace-card
// builders, and generated suggestions.

import { ACCENTS } from "./constants";
import type { DocChunk } from "./document";
import type { SceneData } from "./scene";
import type { RagId, Source } from "./types";

const STOP = new Set(
  ("a an and are as at be but by for from has have if in into is it its of on or " +
    "that the their there these this to was were will with which what when where " +
    "who why how not no nor so than then too very can could should would may might " +
    "we you they he she i our your his her them us also more most other some such " +
    "only own same each few both all any about between through during before after " +
    "above below up down out off over under again further once here just because " +
    "does did do doing been being had having until while against").split(" "),
);

export function tokenize(s: string): string[] {
  return (s.toLowerCase().match(/[a-z0-9][a-z0-9'-]{1,}/g) || []).filter(
    (w) => w.length > 2 && !STOP.has(w),
  );
}

export interface ScoredChunk {
  chunk: DocChunk;
  raw: number;
  score: number; // normalized 0..0.95
}

export interface RetrievalResult {
  ranked: ScoredChunk[]; // final ranking, best first
  top: ScoredChunk[]; // alias of finalTop (top 6)
  answer: string;
  queryTerms: string[];
  initialTop: ScoredChunk[]; // pass-1 result (drives first-phase visuals)
  finalTop: ScoredChunk[]; // what the answer is actually built from
  rejected?: ScoredChunk[]; // corrective: graded-out chunks
  replacements?: ScoredChunk[]; // corrective: chunks found by re-retrieval
  refinedTerms?: string[]; // agentic: terms actually added in pass 2
  graphBoosted?: number[]; // hybrid: entity indices that boosted retrieval
  boostedChunkIds?: Set<number>; // hybrid: chunks whose rank the graph raised
}

export const PASS_THRESHOLD = 0.45;

// ---------- shared scoring core ----------

function scoreChunks(chunks: DocChunk[], terms: string[]): ScoredChunk[] {
  const N = chunks.length;
  const tokens = chunks.map((c) => tokenize(c.text));
  const df = new Map<string, number>();
  for (const q of terms) {
    let d = 0;
    for (const t of tokens) if (t.includes(q)) d++;
    df.set(q, d);
  }
  const idf = (q: string) => Math.log(1 + N / (1 + (df.get(q) || 0)));
  return chunks
    .map((chunk, i) => {
      const t = tokens[i];
      let raw = 0;
      for (const q of terms) {
        const tf = t.filter((w) => w === q || w.startsWith(q)).length;
        if (tf) raw += (1 + Math.log(tf)) * idf(q);
      }
      raw /= Math.sqrt(t.length || 1);
      return { chunk, raw, score: 0 };
    })
    .filter((s) => s.raw > 0)
    .sort((a, b) => b.raw - a.raw);
}

/** Normalize scores in place against the list's own maximum. */
function normalize(ranked: ScoredChunk[]): ScoredChunk[] {
  const max = ranked[0]?.raw || 1;
  ranked.forEach((s) => (s.score = (s.raw / max) * 0.95));
  return ranked;
}

/** Pseudo-relevance feedback: terms frequent in the trusted chunks but rare
 *  in the corpus (tf·idf), so boilerplate tokens (URLs, site chrome) that
 *  appear everywhere don't win. */
function prfTerms(
  from: ScoredChunk[],
  qTerms: string[],
  n: number,
  allChunks: DocChunk[],
): string[] {
  const known = new Set(qTerms);
  const freq = new Map<string, number>();
  for (const s of from) {
    for (const w of tokenize(s.chunk.text)) {
      if (known.has(w) || w.length <= 3) continue;
      freq.set(w, (freq.get(w) || 0) + 1);
    }
  }
  const N = allChunks.length || 1;
  const scored = [...freq.entries()]
    .filter(([, c]) => c >= 2)
    .map(([w, c]) => {
      let df = 0;
      for (const ch of allChunks) if (ch.text.toLowerCase().includes(w)) df++;
      return { w, weight: c * Math.log(N / (1 + df)) };
    })
    .filter((t) => t.weight > 0)
    .sort((a, b) => b.weight - a.weight);
  return scored.slice(0, n).map((t) => t.w);
}

function finish(
  ranked: ScoredChunk[],
  qTerms: string[],
  extras: Partial<RetrievalResult> & { initialTop: ScoredChunk[] },
): RetrievalResult {
  normalize(ranked);
  const finalTop = extras.finalTop ?? ranked.slice(0, 6);
  return {
    ranked,
    top: finalTop,
    answer: extractAnswer(finalTop.slice(0, 3), qTerms),
    queryTerms: qTerms,
    finalTop,
    ...extras,
  };
}

// ---------- the four strategies ----------

export function retrieveBasic(
  chunks: DocChunk[],
  query: string,
): RetrievalResult {
  const qTerms = [...new Set(tokenize(query))];
  const ranked = scoreChunks(chunks, qTerms);
  const initialTop = ranked.slice(0, 6);
  return finish(ranked, qTerms, { initialTop });
}

export interface GraphForRetrieval {
  nodes: { full?: string; label: string; chunkIds?: Set<number> }[];
  neighbors: Record<number, Set<number>>;
}

export function retrieveHybrid(
  chunks: DocChunk[],
  query: string,
  graph: GraphForRetrieval,
): RetrievalResult {
  const qTerms = [...new Set(tokenize(query))];
  const lexical = scoreChunks(chunks, qTerms);
  const initialTop = normalize(lexical.map((s) => ({ ...s }))).slice(0, 6);

  // graph side: entities whose label matches a query term, plus 1-hop neighbors
  const matched = new Set<number>();
  graph.nodes.forEach((n, i) => {
    const terms = tokenize(n.full || n.label);
    if (terms.some((t) => qTerms.some((q) => t === q || t.startsWith(q))))
      matched.add(i);
  });
  const active = new Set(matched);
  for (const i of matched)
    for (const j of graph.neighbors[i] || []) active.add(j);

  const boostChunkIds = new Set<number>();
  for (const i of active)
    for (const id of graph.nodes[i].chunkIds || []) boostChunkIds.add(id);

  const maxRaw = lexical[0]?.raw || 1;
  const byId = new Map(lexical.map((s) => [s.chunk.id, s]));
  // boost lexical hits; graph-only chunks enter the ranking at the boost floor
  for (const id of boostChunkIds) {
    const hit = byId.get(id);
    if (hit) hit.raw += 0.35 * maxRaw;
    else {
      const chunk = chunks.find((c) => c.id === id);
      if (chunk) lexical.push({ chunk, raw: 0.35 * maxRaw, score: 0 });
    }
  }
  lexical.sort((a, b) => b.raw - a.raw);
  const finalTop = normalize(lexical).slice(0, 6);
  return {
    ...finish(lexical, qTerms, { initialTop, finalTop }),
    graphBoosted: [...active].slice(0, 6),
    boostedChunkIds: boostChunkIds,
  };
}

export function retrieveCorrective(
  chunks: DocChunk[],
  query: string,
): RetrievalResult {
  const qTerms = [...new Set(tokenize(query))];
  const ranked = normalize(scoreChunks(chunks, qTerms));
  const initialTop = ranked.slice(0, 6).map((s) => ({ ...s }));
  const graded = ranked.slice(0, 5);
  const rejected = graded.filter((s) => s.score < PASS_THRESHOLD);
  const passing = graded.filter((s) => s.score >= PASS_THRESHOLD);
  if (!rejected.length) {
    return finish(ranked, qTerms, {
      initialTop,
      finalTop: ranked.slice(0, 6),
      rejected: [],
      replacements: [],
    });
  }
  // re-retrieve with query expanded by terms from the chunks that passed
  const trusted = passing.length ? passing : ranked.slice(0, 2);
  const expansion = prfTerms(trusted, qTerms, 3, chunks);
  const secondPass = scoreChunks(chunks, [...qTerms, ...expansion]);
  const excluded = new Set(graded.map((s) => s.chunk.id));
  const replacements = secondPass
    .filter((s) => !excluded.has(s.chunk.id))
    .slice(0, Math.max(rejected.length, 1));
  const merged = [...passing, ...replacements].sort((a, b) => b.raw - a.raw);
  const finalTop = normalize(merged).slice(0, 6);
  return finish([...merged], qTerms, {
    initialTop,
    finalTop,
    rejected,
    replacements,
  });
}

export function retrieveAgentic(
  chunks: DocChunk[],
  query: string,
): RetrievalResult {
  const qTerms = [...new Set(tokenize(query))];
  const pass1 = normalize(scoreChunks(chunks, qTerms));
  const initialTop = pass1.slice(0, 6).map((s) => ({ ...s }));
  const refinedTerms = prfTerms(pass1.slice(0, 2), qTerms, 3, chunks);
  if (!refinedTerms.length) {
    return finish(pass1, qTerms, {
      initialTop,
      finalTop: pass1.slice(0, 6),
      refinedTerms: [],
    });
  }
  const pass2 = scoreChunks(chunks, [...qTerms, ...refinedTerms]);
  // merge both passes, keeping each chunk's best raw score
  const byId = new Map<number, ScoredChunk>();
  for (const s of [...pass1, ...pass2]) {
    const prev = byId.get(s.chunk.id);
    if (!prev || s.raw > prev.raw) byId.set(s.chunk.id, { ...s });
  }
  const merged = [...byId.values()].sort((a, b) => b.raw - a.raw);
  const finalTop = normalize(merged).slice(0, 6);
  return finish(merged, qTerms, { initialTop, finalTop, refinedTerms });
}

// ---------- snippets & extractive answers ----------

export function snippet(chunk: DocChunk, qTerms: string[]): string {
  const lower = chunk.text.toLowerCase();
  let at = -1;
  for (const q of qTerms) {
    const i = lower.indexOf(q);
    if (i >= 0 && (at < 0 || i < at)) at = i;
  }
  const start = Math.max(0, (at < 0 ? 0 : at) - 30);
  const s = chunk.text.slice(start, start + 100).trim();
  return (start > 0 ? "…" : "") + s + (start + 100 < chunk.text.length ? "…" : "");
}

function sentences(text: string): string[] {
  return (text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [])
    .map((s) => s.trim())
    .filter((s) => s.split(/\s+/).length >= 5);
}

function extractAnswer(top: ScoredChunk[], qTerms: string[]): string {
  const cands: { s: string; score: number; order: number }[] = [];
  let order = 0;
  for (const sc of top) {
    for (const s of sentences(sc.chunk.text)) {
      const t = tokenize(s);
      let score = 0;
      for (const q of qTerms) if (t.some((w) => w === q || w.startsWith(q))) score++;
      score = score / Math.sqrt(t.length || 1) + sc.score * 0.1;
      cands.push({ s, score, order: order++ });
    }
  }
  if (!cands.length)
    return "No passage in this document matches the question closely — try rephrasing with terms that appear in the text.";
  const seen = new Set<string>();
  const picked: typeof cands = [];
  for (const c of cands.sort((a, b) => b.score - a.score)) {
    const key = tokenize(c.s).slice(0, 12).join(" ");
    if (seen.has(key)) continue;
    seen.add(key);
    picked.push(c);
    if (picked.length === 3) break;
  }
  const best = picked.sort((a, b) => a.order - b.order).map((c) => c.s);
  let out = best.join(" ");
  const words = out.split(/\s+/);
  if (words.length > 80) out = words.slice(0, 80).join(" ") + "…";
  return out;
}

// ---------- entity graph ----------

export interface ExtractedEntity {
  label: string;
  full: string;
  count: number;
  chunkIds: Set<number>;
  p: [number, number, number];
  desc: string;
}

export interface EntityGraph {
  nodes: ExtractedEntity[];
  edges: { a: number; b: number }[];
  neighbors: Record<number, Set<number>>;
}

const ENTITY_STOP = new Set(
  "The This That These Those There Here What When Where Which While With Without From Into After Before Because However Although Chapter Section Figure Table Page Note Also And But For Not You Your Our They Their".split(
    " ",
  ),
);

export function extractEntityGraph(chunks: DocChunk[]): EntityGraph {
  const counts = new Map<string, { count: number; chunkIds: Set<number> }>();
  const perChunk: string[][] = [];
  for (const c of chunks) {
    // capitalized runs not at sentence start: "Knowledge Graph", "Transformer"
    const found = new Set<string>();
    const re = /(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z0-9-]{2,}(?:\s[A-Z][a-zA-Z0-9-]{2,}){0,2})\b/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(c.text))) {
      const term = m[1];
      if (ENTITY_STOP.has(term.split(" ")[0])) continue;
      found.add(term);
      const e = counts.get(term) || { count: 0, chunkIds: new Set<number>() };
      e.count++;
      e.chunkIds.add(c.id);
      counts.set(term, e);
    }
    perChunk.push([...found]);
  }
  const topTerms = [...counts.entries()]
    .filter(([, v]) => v.count >= 2)
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 11);

  // golden-spiral placement on the unit sphere
  const nodes: ExtractedEntity[] = topTerms.map(([label, v], i) => {
    const n = topTerms.length;
    const y = n > 1 ? 1 - (i / (n - 1)) * 2 : 0;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const th = i * 2.39996;
    return {
      label: label.length > 16 ? label.slice(0, 15) + "…" : label,
      full: label,
      count: v.count,
      chunkIds: v.chunkIds,
      p: [Math.cos(th) * r, y * 0.85, Math.sin(th) * r] as [number, number, number],
      desc: `${v.count} mentions across ${v.chunkIds.size} chunk${v.chunkIds.size > 1 ? "s" : ""} of this document.`,
    };
  });

  const index = new Map(topTerms.map(([label], i) => [label, i]));
  const pairCounts = new Map<string, number>();
  for (const terms of perChunk) {
    const ids = terms
      .map((t) => index.get(t))
      .filter((i): i is number => i !== undefined);
    for (let a = 0; a < ids.length; a++)
      for (let b = a + 1; b < ids.length; b++) {
        const key = Math.min(ids[a], ids[b]) + ":" + Math.max(ids[a], ids[b]);
        pairCounts.set(key, (pairCounts.get(key) || 0) + 1);
      }
  }
  const edges = [...pairCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 13)
    .map(([key]) => {
      const [a, b] = key.split(":").map(Number);
      return { a, b };
    });
  const neighbors: Record<number, Set<number>> = {};
  nodes.forEach((_, i) => (neighbors[i] = new Set()));
  edges.forEach((e) => {
    neighbors[e.a].add(e.b);
    neighbors[e.b].add(e.a);
  });
  return { nodes, edges, neighbors };
}

// ---------- retrieval-trace cards for uploaded documents ----------

export function buildRealSources(
  rag: RagId,
  res: RetrievalResult,
  scene: SceneData,
): Source[] {
  const A = ACCENTS[rag];
  const chunkCard = (s: ScoredChunk, meta?: string): Source => ({
    kind: "chunk",
    label: "chunk #" + s.chunk.id,
    meta: meta ?? "p." + s.chunk.page,
    score: s.score.toFixed(2),
    scoreN: s.score,
    snippet: snippet(s.chunk, res.queryTerms),
    color: A,
  });
  const top = res.finalTop;
  if (!top.length)
    return [
      {
        kind: "reject",
        label: "no match",
        meta: "rejected",
        score: "0.00",
        scoreN: 0.02,
        snippet: "no chunk matched the query terms",
        color: A,
        rejected: true,
      },
    ];
  if (rag === "hybrid") {
    const boosted = res.boostedChunkIds ?? new Set<number>();
    const cards: Source[] = top
      .slice(0, 2)
      .map((s) =>
        chunkCard(
          s,
          boosted.has(s.chunk.id) ? "p." + s.chunk.page + " · graph" : undefined,
        ),
      );
    (res.graphBoosted ?? []).slice(0, 2).forEach((gi, k) => {
      const n = scene.gnodes[gi];
      if (!n) return;
      const nBoosted = top.filter((s) =>
        n.chunkIds?.has(s.chunk.id),
      ).length;
      const nbrs = [...(scene.gnbr[gi] || [])]
        .slice(0, 3)
        .map((j) => scene.gnodes[j]?.label)
        .filter(Boolean);
      const best = top.find((s) => n.chunkIds?.has(s.chunk.id));
      cards.splice(1 + k, 0, {
        kind: "node",
        label: n.label,
        meta: "entity",
        score: ((best?.score ?? 0.5) * 0.95).toFixed(2),
        scoreN: (best?.score ?? 0.5) * 0.95,
        snippet:
          (nBoosted
            ? `boosted ${nBoosted} chunk${nBoosted > 1 ? "s" : ""}`
            : "matched the query") +
          (nbrs.length ? " · linked to " + nbrs.join(", ") : ""),
        color: A,
      });
    });
    return cards;
  }
  if (rag === "agentic") {
    const cards: Source[] = [chunkCard(top[0])];
    const refined = res.refinedTerms ?? [];
    cards.push({
      kind: "tool",
      label: "refine query",
      meta: "agent step",
      score: refined.length ? "+" : "·",
      scoreN: refined.length ? 0.8 : 0.3,
      snippet: refined.length
        ? 'added "' + refined.join(" ") + '" and re-retrieved'
        : "first pass sufficient — no refinement needed",
      color: A,
    });
    // prefer showing a chunk the refinement surfaced (absent from pass 1)
    const initialIds = new Set(res.initialTop.map((s) => s.chunk.id));
    const surfaced = top.find((s) => !initialIds.has(s.chunk.id));
    const second = surfaced ?? top[1];
    if (second)
      cards.push(
        chunkCard(
          second,
          surfaced ? "p." + second.chunk.page + " · pass 2" : undefined,
        ),
      );
    return cards;
  }
  if (rag === "corrective") {
    const cards: Source[] = [];
    (res.rejected ?? []).slice(0, 1).forEach((s) => {
      cards.push({
        kind: "reject",
        label: "chunk #" + s.chunk.id,
        meta: "rejected",
        score: s.score.toFixed(2),
        scoreN: s.score,
        snippet: `graded ${s.score.toFixed(2)} — below threshold, re-retrieval triggered`,
        color: A,
        rejected: true,
      });
    });
    const replacementIds = new Set(
      (res.replacements ?? []).map((s) => s.chunk.id),
    );
    cards.push(
      ...top
        .slice(0, cards.length ? 2 : 3)
        .map((s) =>
          chunkCard(
            s,
            replacementIds.has(s.chunk.id)
              ? "p." + s.chunk.page + " · re-retrieved"
              : undefined,
          ),
        ),
    );
    return cards;
  }
  return top.slice(0, 3).map((s) => chunkCard(s));
}

// ---------- suggested questions for uploaded documents ----------

export function generateSuggestions(chunks: DocChunk[]): string[] {
  const graph = extractEntityGraph(chunks);
  const full = (i: number) => graph.nodes[i]?.full;
  const out: string[] = [];
  if (full(0)) out.push(`What is ${full(0)}?`);
  const edge = graph.edges.find(
    (e) => full(e.a) && full(e.b) && e.a !== e.b,
  );
  if (edge) out.push(`How does ${full(edge.a)} relate to ${full(edge.b)}?`);
  const third = graph.nodes.find(
    (n, i) => n.full && i !== 0 && i !== edge?.a && i !== edge?.b,
  );
  if (third) out.push(`What does it say about ${third.full}?`);
  if (out.length < 3) {
    // keyword fallback for documents with few capitalized entities
    const freq = new Map<string, number>();
    for (const c of chunks)
      for (const w of tokenize(c.text))
        freq.set(w, (freq.get(w) || 0) + 1);
    const kws = [...freq.entries()]
      .filter(([w]) => w.length > 3)
      .sort((a, b) => b[1] - a[1])
      .map(([w]) => w);
    for (const kw of kws) {
      if (out.length >= 3) break;
      if (out.some((q) => q.toLowerCase().includes(kw))) continue;
      out.push(
        out.length === 0
          ? `What is ${kw}?`
          : `What does it say about ${kw}?`,
      );
    }
  }
  return out.slice(0, 3);
}
