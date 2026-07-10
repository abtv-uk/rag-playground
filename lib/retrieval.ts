// Lexical (TF-IDF-style) retrieval over uploaded documents, extractive
// answers, and entity/co-occurrence graph extraction for the Hybrid tab.

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
  ranked: ScoredChunk[]; // all chunks with score > 0, best first
  top: ScoredChunk[]; // top 6
  answer: string;
  queryTerms: string[];
}

export function retrieve(chunks: DocChunk[], query: string): RetrievalResult {
  const qTerms = [...new Set(tokenize(query))];
  const N = chunks.length;
  const tokens = chunks.map((c) => tokenize(c.text));
  const df = new Map<string, number>();
  for (const q of qTerms) {
    let d = 0;
    for (const t of tokens) if (t.includes(q)) d++;
    df.set(q, d);
  }
  const idf = (q: string) => Math.log(1 + N / (1 + (df.get(q) || 0)));
  const scored = chunks.map((chunk, i) => {
    const t = tokens[i];
    let raw = 0;
    for (const q of qTerms) {
      const tf = t.filter((w) => w === q || w.startsWith(q)).length;
      if (tf) raw += (1 + Math.log(tf)) * idf(q);
    }
    raw /= Math.sqrt(t.length || 1);
    return { chunk, raw, score: 0 };
  });
  const ranked = scored.filter((s) => s.raw > 0).sort((a, b) => b.raw - a.raw);
  const max = ranked[0]?.raw || 1;
  ranked.forEach((s) => (s.score = (s.raw / max) * 0.95));
  const top = ranked.slice(0, 6);
  return {
    ranked,
    top,
    answer: extractAnswer(top.slice(0, 3), qTerms),
    queryTerms: qTerms,
  };
}

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
  const chunkCard = (s: ScoredChunk): Source => ({
    kind: "chunk",
    label: "chunk #" + s.chunk.id,
    meta: "p." + s.chunk.page,
    score: s.score.toFixed(2),
    scoreN: s.score,
    snippet: snippet(s.chunk, res.queryTerms),
    color: A,
  });
  const top = res.top;
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
    const cards: Source[] = top.slice(0, 2).map(chunkCard);
    const topIds = new Set(top.map((s) => s.chunk.id));
    scene.gActive.slice(0, 2).forEach((gi, k) => {
      const n = scene.gnodes[gi];
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
        snippet: nbrs.length
          ? "linked to " + nbrs.join(", ")
          : "mentioned in " + (n.chunkIds?.size ?? 0) + " chunks",
        color: A,
      });
      void topIds;
    });
    return cards;
  }
  if (rag === "agentic") {
    const cards: Source[] = [chunkCard(top[0])];
    cards.push({
      kind: "tool",
      label: "refine query",
      meta: "agent step",
      score: "+",
      scoreN: 0.8,
      snippet: 're-queried "' + res.queryTerms.slice(0, 3).join(" ") + '"',
      color: A,
    });
    if (top[1]) cards.push(chunkCard(top[1]));
    return cards;
  }
  if (rag === "corrective") {
    const rejectedRow = scene.gradeRows.find((r) => !r.pass);
    const cards: Source[] = [];
    if (rejectedRow) {
      cards.push({
        kind: "reject",
        label: "chunk #" + rejectedRow.n,
        meta: "rejected",
        score: rejectedRow.s.toFixed(2),
        scoreN: rejectedRow.s,
        snippet: "graded irrelevant — re-retrieval triggered",
        color: A,
        rejected: true,
      });
    }
    return cards.concat(top.slice(0, rejectedRow ? 2 : 3).map(chunkCard));
  }
  return top.slice(0, 3).map(chunkCard);
}
