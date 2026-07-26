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
import { cosineRank, type DenseIndex } from "./embeddings";
import type { SceneData } from "./scene";
import type { RagId, Source } from "./types";

export interface DenseOpts {
  dense?: DenseIndex;
  queryVec?: Float32Array;
}

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
  /** Background-normalized dense relevance (see backgroundStats below) —
   *  present only when this chunk was scored against embeddings. Never
   *  threshold on raw cosine or on `score` when this is set; use
   *  passScore() instead. */
  relevance?: number;
  /** Raw cosine similarity, preserved through RRF fusion (where `.raw`
   *  becomes the fused RRF weight, not cosine) specifically so
   *  passScore()'s absolute floor has something to test — see its
   *  docstring for why this exists alongside `relevance`. */
  cosine?: number;
}

// Below this raw cosine, nothing in the document is a genuine semantic
// match — see passScore()'s docstring for why relevance alone can never
// catch this case. Empirically calibrated against bge-base-en-v1.5 on the
// bundled sample: real on-topic queries topped out ~0.77-0.79 cosine; a
// deliberately off-topic control query ("how does weather affect crop
// yields" against a legal textbook) topped out ~0.52. This floor sits with
// margin on both sides of that gap. Expect to retune if the embedding
// model changes; Phase 3's LLM-based grader is the intended real fix —
// this is a best-effort placeholder until then.
const ABSOLUTE_COSINE_FLOOR = 0.55;

/** What corrective grading (and the grade-panel visual) should actually
 *  threshold against: relevance when dense scoring produced one, otherwise
 *  the existing TF-IDF-calibrated `score`. Exists because PASS_THRESHOLD
 *  (0.45) is calibrated for TF-IDF's 0..0.95 spread — bge cosines cluster
 *  ~0.6-0.85 for everything, so thresholding raw cosine directly would
 *  never reject anything.
 *
 *  Background-normalized relevance alone isn't sufficient, though: a
 *  top-ranked candidate is, by construction, ABOVE its own document's
 *  background mean, so its z-score is virtually always positive —  and
 *  sigmoid(positive) is mathematically always > 0.5, comfortably above
 *  PASS_THRESHOLD's 0.45. That makes it impossible for relevance alone to
 *  ever reject a top-ranked chunk, no matter how irrelevant the whole
 *  document actually is to the query (verified empirically against the
 *  bundled sample: an off-topic control query still scored 0.55-0.76
 *  relevance at every divisor tried). The absolute floor below is what
 *  actually catches that case; relevance remains useful as a *display*
 *  signal — it still differentiates a strong match from a merely adequate
 *  one once both clear the floor. */
export function passScore(s: ScoredChunk): number {
  if (s.relevance != null) {
    if (s.cosine != null && s.cosine < ABSOLUTE_COSINE_FLOOR) return 0;
    return s.relevance;
  }
  return s.score;
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

// ---------- dense scoring & fusion ----------

/** Mean/stddev of a query's cosine similarity against every chunk in the
 *  document. cosineRank already computes cosine for every row (there's no
 *  cheaper subset to sample once that work is done), so this uses the full
 *  distribution rather than the stride-sampled 96 the original design
 *  sketch assumed — strictly more accurate at the same cost. */
function backgroundStats(ranked: { cos: number }[]): { mu: number; sigma: number } {
  const n = ranked.length || 1;
  let mu = 0;
  for (const r of ranked) mu += r.cos;
  mu /= n;
  let variance = 0;
  for (const r of ranked) variance += (r.cos - mu) ** 2;
  variance /= n;
  return { mu, sigma: Math.sqrt(variance) };
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Dense-only ranking: cosine similarity against every chunk, converted to
 *  a background-normalized relevance ("N standard deviations more similar
 *  to this query than a random chunk in this document") so PASS_THRESHOLD
 *  stays meaningful — see passScore(). `raw` is left as the plain cosine;
 *  callers that mix this with a differently-scaled list (lexical TF-IDF,
 *  RRF weights) must rebase before merging — see fuseWithDense. */
function denseScoreChunks(
  chunks: DocChunk[],
  dense: DenseIndex,
  queryVec: Float32Array,
): ScoredChunk[] {
  const ranked = cosineRank(dense, queryVec);
  const { mu, sigma } = backgroundStats(ranked);
  const sigmaSafe = sigma || 1e-6;
  return ranked
    .map(({ i, cos }): ScoredChunk | null => {
      const chunk = chunks[i];
      if (!chunk) return null;
      return {
        chunk,
        raw: cos,
        score: 0,
        relevance: sigmoid((cos - mu) / sigmaSafe / 2.2),
        cosine: cos,
      };
    })
    .filter((s): s is ScoredChunk => !!s);
}

/** Reciprocal Rank Fusion: combines ranked lists by position, not raw
 *  score magnitude, so lists on incomparable scales (lexical TF-IDF, dense
 *  cosine, graph-boost membership) combine fairly. */
function rrf(lists: { id: number }[][], k = 60): Map<number, number> {
  const scores = new Map<number, number>();
  for (const list of lists) {
    list.forEach(({ id }, rank) => {
      scores.set(id, (scores.get(id) || 0) + 1 / (k + rank + 1));
    });
  }
  return scores;
}

/** RRF-fuses a lexical ranking with dense retrieval (plus any extra ranked
 *  lists, e.g. hybrid's graph-boost membership), returning an
 *  already-normalized (0..0.95 `.score`) result with dense `.relevance`
 *  carried through. The returned list's `.raw` is the RRF weight — a
 *  self-consistent scale, but NOT comparable to a fresh lexical pass's TF-IDF
 *  `.raw`. Callers that later merge this with such a pass (corrective's
 *  re-retrieval, agentic's second pass) must rebase on `.score` — see the
 *  comment at each call site for why mixing raw scales there would silently
 *  let one list's magnitude swamp the other's after the next normalize(). */
function fuseWithDense(
  chunks: DocChunk[],
  lexical: ScoredChunk[],
  dense: DenseIndex,
  queryVec: Float32Array,
  extraLists: { id: number }[][] = [],
): ScoredChunk[] {
  const denseScored = denseScoreChunks(chunks, dense, queryVec);
  const byIdChunk = new Map(chunks.map((c) => [c.id, c]));
  const relevanceById = new Map(denseScored.map((s) => [s.chunk.id, s.relevance]));
  const cosineById = new Map(denseScored.map((s) => [s.chunk.id, s.cosine]));
  const fused = rrf([
    lexical.map((s) => ({ id: s.chunk.id })),
    denseScored.map((s) => ({ id: s.chunk.id })),
    ...extraLists,
  ]);
  const ranked = [...fused.entries()]
    .map(([id, raw]): ScoredChunk | null => {
      const chunk = byIdChunk.get(id);
      if (!chunk) return null;
      return { chunk, raw, score: 0, relevance: relevanceById.get(id), cosine: cosineById.get(id) };
    })
    .filter((s): s is ScoredChunk => !!s)
    .sort((a, b) => b.raw - a.raw);
  return normalize(ranked);
}

// ---------- the four strategies ----------

export function retrieveBasic(
  chunks: DocChunk[],
  query: string,
  opts?: DenseOpts,
): RetrievalResult {
  const qTerms = [...new Set(tokenize(query))];
  // "basic vector RAG" means dense-only — when embeddings are available,
  // that's the real naive baseline; lexical is the fallback when they
  // aren't (quota exhausted, Worker unreachable, embed failure).
  if (opts?.dense && opts.queryVec) {
    const ranked = normalize(denseScoreChunks(chunks, opts.dense, opts.queryVec));
    const initialTop = ranked.slice(0, 6).map((s) => ({ ...s }));
    return finish(ranked, qTerms, { initialTop, finalTop: ranked.slice(0, 6) });
  }
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
  opts?: DenseOpts,
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

  let finalRanked: ScoredChunk[];
  if (opts?.dense && opts.queryVec) {
    // three-way fusion: lexical, dense, and graph-boost membership as a
    // third ranked signal — strengthens hybrid's story rather than
    // weakening it once real embeddings exist.
    const graphList = [...boostChunkIds].map((id) => ({ id }));
    finalRanked = fuseWithDense(chunks, lexical, opts.dense, opts.queryVec, [graphList]);
  } else {
    // lexical-only fallback: boost lexical hits directly (unchanged from
    // before dense retrieval existed)
    const maxRaw = lexical[0]?.raw || 1;
    const byId = new Map(lexical.map((s) => [s.chunk.id, s]));
    for (const id of boostChunkIds) {
      const hit = byId.get(id);
      if (hit) hit.raw += 0.35 * maxRaw;
      else {
        const chunk = chunks.find((c) => c.id === id);
        if (chunk) lexical.push({ chunk, raw: 0.35 * maxRaw, score: 0 });
      }
    }
    lexical.sort((a, b) => b.raw - a.raw);
    finalRanked = normalize(lexical);
  }
  const finalTop = finalRanked.slice(0, 6);
  return {
    ...finish(finalRanked, qTerms, { initialTop, finalTop }),
    graphBoosted: [...active].slice(0, 6),
    boostedChunkIds: boostChunkIds,
  };
}

export function retrieveCorrective(
  chunks: DocChunk[],
  query: string,
  opts?: DenseOpts,
): RetrievalResult {
  const qTerms = [...new Set(tokenize(query))];
  const lexical = normalize(scoreChunks(chunks, qTerms));
  const ranked =
    opts?.dense && opts.queryVec
      ? fuseWithDense(chunks, lexical, opts.dense, opts.queryVec)
      : lexical;

  const initialTop = ranked.slice(0, 6).map((s) => ({ ...s }));
  const graded = ranked.slice(0, 5);
  // never threshold on raw cosine — passScore() prefers the
  // background-normalized relevance when dense scoring produced one; see
  // its docstring for why PASS_THRESHOLD would otherwise pass everything
  const rejected = graded.filter((s) => passScore(s) < PASS_THRESHOLD);
  const passing = graded.filter((s) => passScore(s) >= PASS_THRESHOLD);
  if (!rejected.length) {
    return finish(ranked, qTerms, {
      initialTop,
      finalTop: ranked.slice(0, 6),
      rejected: [],
      replacements: [],
    });
  }
  // re-retrieve with query expanded by terms from the chunks that passed —
  // lexical PRF drives expansion even under dense retrieval, since it's
  // about vocabulary the corpus shares, not vector geometry
  const trusted = passing.length ? passing : ranked.slice(0, 2);
  const expansion = prfTerms(trusted, qTerms, 3, chunks);
  const secondPass = normalize(scoreChunks(chunks, [...qTerms, ...expansion]));
  const excluded = new Set(graded.map((s) => s.chunk.id));
  const replacements = secondPass
    .filter((s) => !excluded.has(s.chunk.id))
    .slice(0, Math.max(rejected.length, 1));
  // `passing` may carry RRF-fused `.raw` (dense path) while `replacements`
  // is always fresh lexical TF-IDF `.raw` — the two scales aren't
  // comparable, so sorting/renormalizing on raw would let whichever list
  // happens to have larger numbers swamp the other. `.score` is the one
  // scale both are guaranteed to share (each was independently normalized
  // to 0..0.95 above); rebase `.raw` onto it before the merge.
  const merged = [...passing, ...replacements]
    .map((s) => ({ ...s, raw: s.score }))
    .sort((a, b) => b.raw - a.raw);
  const finalTop = normalize(merged).slice(0, 6);
  return finish(merged, qTerms, {
    initialTop,
    finalTop,
    rejected,
    replacements,
  });
}

export function retrieveAgentic(
  chunks: DocChunk[],
  query: string,
  opts?: DenseOpts,
): RetrievalResult {
  const qTerms = [...new Set(tokenize(query))];
  const lexical = normalize(scoreChunks(chunks, qTerms));
  const pass1 =
    opts?.dense && opts.queryVec
      ? fuseWithDense(chunks, lexical, opts.dense, opts.queryVec)
      : lexical;

  const initialTop = pass1.slice(0, 6).map((s) => ({ ...s }));
  const refinedTerms = prfTerms(pass1.slice(0, 2), qTerms, 3, chunks);
  if (!refinedTerms.length) {
    return finish(pass1, qTerms, {
      initialTop,
      finalTop: pass1.slice(0, 6),
      refinedTerms: [],
    });
  }
  const pass2 = normalize(scoreChunks(chunks, [...qTerms, ...refinedTerms]));
  // merge both passes, keeping each chunk's best-scoring entry. Compared
  // on `.score`, not `.raw` — pass1 may be RRF-fused (dense path) while
  // pass2 is always fresh lexical TF-IDF, and those scales aren't
  // comparable (see the identical situation in retrieveCorrective above).
  const byId = new Map<number, ScoredChunk>();
  for (const s of [...pass1, ...pass2]) {
    const prev = byId.get(s.chunk.id);
    if (!prev || s.score > prev.score) byId.set(s.chunk.id, s);
  }
  const merged = [...byId.values()]
    .map((s) => ({ ...s, raw: s.score }))
    .sort((a, b) => b.raw - a.raw);
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
    .filter((s) => s.split(/\s+/).length >= 6);
}

// Textbook scaffolding that is term-dense but not explanatory: section
// headings, multiple-choice options, learning objectives, figure captions.
// Penalized (not excluded) so it only surfaces when nothing better matches.
function looksStructural(s: string): boolean {
  if (/^\s*\d/.test(s)) return true; // "1 Trade Secret Protection"
  if (/^\s*[a-eA-E][.)]\s/.test(s)) return true; // "b. …" quiz option
  if (/^\s*[•·▪◦-]\s/.test(s)) return true; // bullet
  if (
    /will be able to|learning objective|assessment question|\(credit:|figure\s*\d|table\s*\d|chapter\s*(summary|outline)|key terms/i.test(
      s,
    )
  )
    return true;
  // heading soup: mostly Capitalized words, few lowercase function words
  const words = s.split(/\s+/).filter(Boolean);
  const lower = words.filter((w) => /^[a-z]/.test(w)).length;
  return lower / (words.length || 1) < 0.35;
}

function extractAnswer(top: ScoredChunk[], qTerms: string[]): string {
  const cands: { s: string; score: number; order: number }[] = [];
  let order = 0;
  for (const sc of top) {
    for (const s of sentences(sc.chunk.text)) {
      const t = tokenize(s);
      let hits = 0;
      for (const q of qTerms) if (t.some((w) => w === q || w.startsWith(q))) hits++;
      if (!hits) continue; // only sentences that actually mention the query
      let score = hits / Math.sqrt(t.length || 1) + sc.score * 0.1;
      if (looksStructural(s)) score *= 0.15;
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
  ("The This That These Those There Here What When Where Which While With Without " +
    "From Into After Before Because However Although Chapter Section Figure Table " +
    "Page Note Also And But For Not You Your Our They Their " +
    // citation / reference artifacts (legal & academic texts)
    "See Rule Fed Ibid Cir Supp Vol Pub Sec Art Reporter Nutshell Eds Trans Rev " +
    "Stat Reg Ann App Ch Pt Ed Cf Id No Press University Journal Review Rev'd " +
    "Appendix Index Contents").split(/\s+/),
);

/** An entity must be a real name, not a citation abbreviation. Short single
 *  tokens ("Fed", "Ct", "Id") are citation noise; multi-word names and longer
 *  acronyms ("DMCA", "USPTO") are kept. */
function isCitationArtifact(term: string): boolean {
  const single = !term.includes(" ");
  if (single && term.length < 4) return true;
  if (/^\d/.test(term)) return true;
  return false;
}

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
      const wordsInTerm = term.split(" ");
      if (wordsInTerm.some((w) => ENTITY_STOP.has(w))) continue;
      if (isCitationArtifact(term)) continue;
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
  const top = res.finalTop;
  // The generator is handed `finalTop` in order and cites passages by their
  // 1-based position, so that position is the citation number a card must
  // claim for an inline [n] to resolve to it.
  const citationOf = new Map(top.map((s, i) => [s.chunk.id, i + 1]));
  const chunkCard = (s: ScoredChunk, meta?: string): Source => ({
    kind: "chunk",
    label: "chunk #" + s.chunk.id,
    meta: meta ?? "p." + s.chunk.page,
    score: s.score.toFixed(2),
    scoreN: s.score,
    snippet: snippet(s.chunk, res.queryTerms),
    color: A,
    chunkId: s.chunk.id,
    citation: citationOf.get(s.chunk.id),
  });
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
      // Report the number that actually drove the rejection, not `.score` —
      // under dense retrieval `.score` is the normalized RRF weight, so the
      // top-ranked chunk always reads 0.95 and the card would claim "graded
      // 0.95 — below threshold". Same reason scene.ts's grade panel uses
      // passScore(); the bar keeps a floor of 0.05 so a hard 0 (cosine below
      // ABSOLUTE_COSINE_FLOOR) still renders as a visible sliver.
      const graded = passScore(s);
      cards.push({
        kind: "reject",
        label: "chunk #" + s.chunk.id,
        meta: "rejected",
        score: graded.toFixed(2),
        scoreN: Math.max(0.05, graded),
        snippet: `graded ${graded.toFixed(2)} — below threshold, re-retrieval triggered`,
        color: A,
        rejected: true,
        chunkId: s.chunk.id,
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
//
// Three questions that tour the document: its overall topic, a theme from
// the middle chapters, and what the final section covers. Retrieval is
// lexical, so every question must carry terms that actually occur in the
// region it points at — "What does Chapter 3 discuss?" is unanswerable if
// no chunk says "Chapter 3", but "patent cases" or "trade secrets" is.

// tokens that make a phrase useless as a question subject: site chrome,
// citation abbreviations, textbook scaffolding
const PHRASE_NOISE =
  /\b(www|org|com|edu|net|http|https|html|pdf|isbn|openstax|access|free|page|pages|figure|table|chapter|section|appendix|index|contents|answer|key|credit|rule|rules|fed|civ|supp|cir|inc|llc|ibid|seq|stat|reg|vol|pub|press|review|journal|university|learning|objectives?|completing|question|questions|assessment)\b/;

/** Distinct bigrams of content tokens per chunk → chunk-frequency map. */
function bigramDf(chunks: DocChunk[]): Map<string, number> {
  const df = new Map<string, number>();
  for (const c of chunks) {
    const t = tokenize(c.text);
    const seen = new Set<string>();
    for (let i = 0; i < t.length - 1; i++) seen.add(t[i] + " " + t[i + 1]);
    for (const b of seen) df.set(b, (df.get(b) || 0) + 1);
  }
  return df;
}

const cleanPhrase = (b: string) =>
  !PHRASE_NOISE.test(b) && !/\d/.test(b) && b.split(" ").every((w) => w.length > 3);

/** The document's overall topic: the title-case run that opens the first
 *  chunk (title pages / H1s), cross-checked against the text; falls back to
 *  the filename, then to the most document-wide bigram. */
function detectTopic(
  chunks: DocChunk[],
  name: string | undefined,
  allDf: Map<string, number>,
): string {
  const inText = (phrase: string) => allDf.get(phrase.toLowerCase()) || 0;
  const strip = (s: string) =>
    s
      .replace(/^(an?\s+)?(brief\s+)?(introduction|guide|primer)\s+to\s+/i, "")
      .replace(/^(the\s+)?(complete\s+|beginner'?s\s+)?(guide|handbook|basics)\s+(to|of)\s+/i, "")
      .trim();

  // (a) leading title-case run of the first chunk, stopped at ALL-CAPS words
  // ("SENIOR CONTRIBUTING AUTHORS") and capped at 8 words
  const m = chunks[0]?.text.match(
    /^((?:[A-Z][a-z][\w'-]*|to|of|the|and|in|for|on|a|an)(?:\s+(?:[A-Z][a-z][\w'-]*|to|of|the|and|in|for|on|a|an)){0,7})/,
  );
  if (m) {
    const t = strip(m[1]);
    const toks = tokenize(t);
    // must be a real recurring subject, not a one-off heading
    if (toks.length >= 1 && toks.length <= 4 && inText(toks.slice(0, 2).join(" ")) >= 3)
      return t.toLowerCase();
  }

  // (b) filename: "introduction-intellectual-property.pdf" → best bigram of
  // its tokens that the text itself uses often
  if (name) {
    const toks = tokenize(
      name.replace(/\.[a-z0-9]+$/i, "").replace(/[-_./]+/g, " "),
    ).filter((w) => !/^(intro|introduction|guide|notes|draft|final|copy|v\d+)$/.test(w));
    let best = "";
    let bestDf = 2; // require at least 3 occurrences
    for (let i = 0; i < toks.length - 1; i++) {
      const b = toks[i] + " " + toks[i + 1];
      const d = inText(b);
      if (d > bestDf) [best, bestDf] = [b, d];
    }
    if (best) return best;
    if (toks.length === 1 && inText(toks[0])) return toks[0];
  }

  // (c) most widespread clean bigram, then unigram
  const top = [...allDf.entries()]
    .filter(([b]) => cleanPhrase(b))
    .sort((a, b) => b[1] - a[1])[0];
  if (top && top[1] >= 3) return top[0];
  const freq = new Map<string, number>();
  for (const c of chunks)
    for (const w of tokenize(c.text)) freq.set(w, (freq.get(w) || 0) + 1);
  const kw = [...freq.entries()]
    .filter(([w]) => w.length > 3 && !PHRASE_NOISE.test(w))
    .sort((a, b) => b[1] - a[1])[0];
  return kw ? kw[0] : "";
}

/** Best clean bigram of a document region, scored for being frequent in the
 *  region AND concentrated there (df²ᵣₑ𝓰ᵢₒₙ / df𝒹ₒ𝒸). */
function regionPhrase(
  region: DocChunk[],
  allDf: Map<string, number>,
  exclude: string[],
): string {
  const used = new Set(exclude.flatMap((p) => tokenize(p)));
  const local = bigramDf(region);
  const scored = [...local.entries()]
    .filter(
      ([b, d]) =>
        d >= 2 && cleanPhrase(b) && !b.split(" ").some((w) => used.has(w)),
    )
    .map(([b, d]) => [b, (d * d) / (allDf.get(b) || 1)] as const)
    .sort((a, b) => b[1] - a[1]);
  return scored[0]?.[0] ?? "";
}

export function generateSuggestions(
  chunks: DocChunk[],
  name?: string,
): string[] {
  if (!chunks.length) return [];
  const allDf = bigramDf(chunks);
  const out: string[] = [];

  // 1 — the document's own subject
  const topic = detectTopic(chunks, name, allDf);
  if (topic) out.push(`What is ${topic}?`);

  // 2 — a theme from the middle of the document
  const mid = chunks.slice(
    Math.floor(chunks.length * 0.35),
    Math.ceil(chunks.length * 0.65),
  );
  const midPhrase = regionPhrase(mid, allDf, [topic]);
  if (midPhrase) out.push(`What does it say about ${midPhrase}?`);

  // 3 — what the final section covers ("conclusion" only if the text has one)
  const tail = chunks.slice(Math.floor(chunks.length * 0.88));
  const hasConclusion = tail.some((c) => /\bconclusions?\b/i.test(c.text));
  const tailPhrase = regionPhrase(tail, allDf, [topic, midPhrase]);
  if (hasConclusion) out.push("What is the conclusion of the document?");
  else if (tailPhrase)
    out.push(`What does the final section say about ${tailPhrase}?`);

  // entity / keyword fallback for anything still missing
  if (out.length < 3) {
    const graph = extractEntityGraph(chunks);
    for (const n of graph.nodes) {
      if (out.length >= 3) break;
      const t = tokenize(n.full);
      if (out.some((q) => t.some((w) => q.toLowerCase().includes(w)))) continue;
      out.push(`What does it say about ${n.full}?`);
    }
  }
  if (out.length < 3) {
    const freq = new Map<string, number>();
    for (const c of chunks)
      for (const w of tokenize(c.text)) freq.set(w, (freq.get(w) || 0) + 1);
    const kws = [...freq.entries()]
      .filter(([w]) => w.length > 3 && !PHRASE_NOISE.test(w))
      .sort((a, b) => b[1] - a[1]);
    for (const [kw] of kws) {
      if (out.length >= 3) break;
      if (out.some((q) => q.toLowerCase().includes(kw))) continue;
      out.push(`What does it say about ${kw}?`);
    }
  }
  return out.slice(0, 3);
}
