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
  /** A real model verdict from POST /grade, present only on the chunks
   *  corrective mode actually graded (pass 1's top 5) and only when the
   *  grader was reachable. When set it *replaces* the cosine floor as the
   *  pass/reject decision — see gradedPass(). `why` is the model's own
   *  wording, shown to the user verbatim on the reject card, and may be
   *  empty if the model omitted it. */
  verdict?: { relevant: boolean; why: string };
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

/** Whether a graded chunk passes — the single place that decides. A real
 *  model verdict wins outright when present; otherwise this falls back to
 *  the cosine-floor heuristic, which is still what runs whenever the grader
 *  is unreachable, quota-exhausted, or simply wasn't asked (every mode
 *  except corrective). */
export function gradedPass(s: ScoredChunk): boolean {
  if (s.verdict) return s.verdict.relevant;
  return passScore(s) >= PASS_THRESHOLD;
}

/** Bar length for the grade panel and trace cards. Under a model verdict
 *  there is no continuous score to show — the grader returns a boolean, not
 *  a confidence — so this reports the verdict itself rather than dressing a
 *  cosine up as the thing that made the decision. (Showing the underlying
 *  score next to a model rejection is exactly the mismatch that made the
 *  reject card read "graded 0.95 — below threshold".) */
export function gradedBar(s: ScoredChunk): number {
  if (s.verdict) return s.verdict.relevant ? 0.9 : 0.12;
  return Math.max(0.05, passScore(s));
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
  refinedTerms?: string[]; // agentic (PRF fallback): terms actually added in pass 2
  planSubqueries?: string[]; // agentic (planned): the sub-queries retrieved
  planRationale?: string; // agentic (planned): model's own decomposition note
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

/** Two words are the same term if they differ only by an inflectional
 *  ending ("patent"/"patents", "file"/"filed"). A bare prefix test is not
 *  enough here: the query "trade secret" would light up the entity
 *  "Trademark Office", because "trademark" starts with "trade". */
function sameWord(a: string, b: string): boolean {
  if (a === b) return true;
  const [long, short] = a.length > b.length ? [a, b] : [b, a];
  return short.length >= 4 && long.length - short.length <= 2 && long.startsWith(short);
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
    if (terms.some((t) => qTerms.some((q) => sameWord(t, q)))) matched.add(i);
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

/** Corrective's first pass, split out so the caller can send `graded` to
 *  POST /grade and hand the verdicts back to retrieveCorrective. Kept
 *  separate (rather than making retrieval async) for the same reason query
 *  embedding is: retrieval stays synchronous and pure, and the one network
 *  step lives in the hook — see hooks/usePlayground.ts. */
export interface CorrectivePass1 {
  ranked: ScoredChunk[];
  /** The chunks to grade: pass 1's top 5, in order. Verdict `i` refers to
   *  position `i` here. */
  graded: ScoredChunk[];
}

export function correctivePass1(
  chunks: DocChunk[],
  query: string,
  opts?: DenseOpts,
): CorrectivePass1 {
  const qTerms = [...new Set(tokenize(query))];
  const lexical = normalize(scoreChunks(chunks, qTerms));
  const ranked =
    opts?.dense && opts.queryVec
      ? fuseWithDense(chunks, lexical, opts.dense, opts.queryVec)
      : lexical;
  return { ranked, graded: ranked.slice(0, 5) };
}

export interface GradeVerdict {
  i: number;
  relevant: boolean;
  why: string;
}

/** Attaches model verdicts to pass 1's graded chunks, by position. Mutates
 *  in place because `graded` holds the same object references as `ranked`,
 *  which is what carries the verdicts through to the final result and the
 *  trace cards. A verdict whose call failed is simply absent — that chunk
 *  keeps the cosine-floor heuristic rather than being rejected by default. */
export function applyGradeVerdicts(p1: CorrectivePass1, verdicts: GradeVerdict[]): void {
  for (const v of verdicts) {
    const target = p1.graded[v.i];
    if (target) target.verdict = { relevant: v.relevant, why: v.why };
  }
}

export function retrieveCorrective(
  chunks: DocChunk[],
  query: string,
  opts?: DenseOpts,
  pass1?: CorrectivePass1,
): RetrievalResult {
  const qTerms = [...new Set(tokenize(query))];
  const { ranked, graded } = pass1 ?? correctivePass1(chunks, query, opts);

  const initialTop = ranked.slice(0, 6).map((s) => ({ ...s }));
  // gradedPass() prefers a real model verdict and falls back to the cosine
  // floor — never a raw threshold on `.score`, which under dense retrieval
  // is an RRF weight and would pass everything (see passScore()).
  const rejected = graded.filter((s) => !gradedPass(s));
  const passing = graded.filter((s) => gradedPass(s));
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

/** A validated /plan decomposition, handed in by the hook (the one place
 *  that can await — same seam as corrective's pass1/verdicts split).
 *  `subVecs` is parallel to `subqueries`; entries may be missing when the
 *  batched embed failed or the doc has no dense index, in which case that
 *  sub-query retrieves lexically only. */
export interface AgenticPlanOpts {
  subqueries: string[];
  subVecs?: (Float32Array | null)[];
  rationale: string;
}

// Per-sub-query ranked lists are capped before RRF so one broad sub-query
// (many weak lexical matches) can't out-vote the others by sheer length.
const SUBQUERY_LIST_CAP = 20;

export function retrieveAgentic(
  chunks: DocChunk[],
  query: string,
  opts?: DenseOpts,
  plan?: AgenticPlanOpts,
): RetrievalResult {
  const qTerms = [...new Set(tokenize(query))];
  const lexical = normalize(scoreChunks(chunks, qTerms));
  const pass1 =
    opts?.dense && opts.queryVec
      ? fuseWithDense(chunks, lexical, opts.dense, opts.queryVec)
      : lexical;

  const initialTop = pass1.slice(0, 6).map((s) => ({ ...s }));

  // Planned path: pass 2 retrieves per sub-query and RRF-merges everything.
  // Measured against the bundled sample this genuinely changes the result
  // set (Jaccard 0.09-0.71 vs basic dense top-6 across probe queries, new
  // chunks surfaced for every compound question) — it is not a cosmetic
  // relabel of the PRF loop.
  if (plan?.subqueries.length) {
    const lists: { id: number }[][] = [];
    plan.subqueries.forEach((sub, i) => {
      const subLex = scoreChunks(chunks, [...new Set(tokenize(sub))]).slice(0, SUBQUERY_LIST_CAP);
      if (subLex.length) lists.push(subLex.map((s) => ({ id: s.chunk.id })));
      const vec = plan.subVecs?.[i];
      if (opts?.dense && vec) {
        lists.push(
          cosineRank(opts.dense, vec)
            .slice(0, SUBQUERY_LIST_CAP)
            .map((r) => {
              const chunk = chunks[r.i];
              return { id: chunk ? chunk.id : -1 };
            })
            .filter((e) => e.id !== -1),
        );
      }
    });
    // Original-query ranking participates as one list among the sub-query
    // lists, so the merged set stays anchored to the actual question while
    // each sub-query pulls in its own aspect.
    const fused = rrf([pass1.map((s) => ({ id: s.chunk.id })), ...lists]);
    const byIdPass1 = new Map(pass1.map((s) => [s.chunk.id, s]));
    const byIdChunk = new Map(chunks.map((c) => [c.id, c]));
    const merged = [...fused.entries()]
      .map(([id, raw]): ScoredChunk | null => {
        const chunk = byIdChunk.get(id);
        if (!chunk) return null;
        const prev = byIdPass1.get(id);
        return { chunk, raw, score: 0, relevance: prev?.relevance, cosine: prev?.cosine };
      })
      .filter((s): s is ScoredChunk => !!s)
      .sort((a, b) => b.raw - a.raw);
    const finalTop = normalize(merged).slice(0, 6);
    return finish(merged, qTerms, {
      initialTop,
      finalTop,
      refinedTerms: [],
      planSubqueries: plan.subqueries,
      planRationale: plan.rationale,
    });
  }

  // PRF fallback — the pre-/plan behavior, still what runs whenever the
  // planner is unreachable, quota-exhausted, or returned a degenerate plan.
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
//
// The graph has two jobs: it is the constellation the user sees in hybrid
// mode, and it is what hybrid retrieval matches a query against (and so
// what feeds the generator's RELATED CONCEPTS line). Both jobs want the
// document's *subject matter*, which is why the harvest below has two
// halves. Capitalized runs alone — all this used to do — can only ever
// yield proper nouns, so on a textbook about intellectual property the
// entire graph came out as "United States", "America", "Wikimedia
// Commons": the real concepts are written lowercase mid-sentence and were
// invisible to it. Recurring content-word phrases supply those, and the
// two pools are merged under one score.

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
    "Appendix Index Contents " +
    // bibliography and image-credit boilerplate — capitalized on every
    // occurrence ("Retrieved from…", "credit: Wikimedia Commons"), so
    // frequency alone can never tell it apart from a real name. Only words
    // that are boilerplate in every genre belong here: "Copyright" and
    // "Rights" appear in the same credit lines but are subject matter in a
    // document about intellectual property, and blocking them would be
    // tuning this list to one book.
    "Retrieved Accessed Access Wikimedia Commons " +
    // learning-objective scaffolding, likewise capitalized by convention
    "Understand Describe Identify Explain Discuss Define Analyze Summarize " +
    "Outline Compare Evaluate Recognize Learning Objectives Exercises Terms " +
    "Summary Assessment Questions Glossary References Bibliography " +
    // the same convention in technical documentation
    "Example Examples Default Defaults Returns Parameter Parameters Usage " +
    "Argument Arguments Optional Required Deprecated").split(/\s+/),
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

interface Candidate {
  label: string;
  count: number;
  chunkIds: Set<number>;
  /** Variants are merged into one candidate, so `count` stops describing
   *  `label` alone; this is how often the surviving surface form itself
   *  appeared, which is what decides whether a variant takes the label
   *  over. */
  surfaceCount: number;
  /** Node identity: variants share it, so the two pools can be deduplicated
   *  against each other and co-occurrence counted once per entity. */
  key: string;
}

/** Singular-ish key so "trade secret"/"trade secrets" and "Patent"/"Patents"
 *  collapse into one node instead of occupying two of eleven slots. */
function conceptKey(label: string): string {
  const singular = (w: string) =>
    /ies$/.test(w)
      ? w.slice(0, -3) + "y"
      : // "business", "status", "basis" are not plurals
        /(?:ss|us|is)$/.test(w)
        ? w
        : w.replace(/s$/, "");
  return label.toLowerCase().split(" ").map(singular).join(" ");
}

/** Frequent AND concentrated — the same df²-style shape regionPhrase uses to
 *  pick a section's subject. Ranking on raw count instead puts whatever the
 *  document mentions everywhere (a country, a court) above what it is
 *  actually about. */
const concentration = (c: Candidate) => (c.count * c.count) / c.chunkIds.size;

/** Recurring content-word phrases: the document's subject matter, which is
 *  written lowercase and so never appears in the capitalized-run pool.
 *  Phrases are built only from words adjacent in the source sentence, so
 *  "Patent and Trademark Office" does not become "patent trademark". */
function harvestPhrases(chunks: DocChunk[]): Map<string, Candidate> {
  const out = new Map<string, Candidate>();
  for (const c of chunks) {
    // Clause boundaries end a phrase. Two words are only a phrase if they
    // were written as one: "…the patent. Trade secrets…" is not "patent
    // trade", a list of "patents, trademarks" is not "patent trademark",
    // and a URL path is not a phrase at all — that last one is what turns
    // a markdown document's own link targets into its top "concepts".
    for (const clause of c.text
      .toLowerCase()
      // underscores split identifiers (UND_ERR_SOCKET), and a dash with
      // space around it is punctuation rather than part of a compound word
      .split(/[.;:!?,()[\]{}"“”/|<>#*=_`\\]+|\s[-–—]+\s/)) {
      const words = clause.match(/[a-z0-9][a-z0-9'-]{1,}/g) || [];
      for (let i = 0; i < words.length - 1; i++) {
        const [a, b] = [words[i], words[i + 1]];
        if (a.length < 3 || b.length < 3 || STOP.has(a) || STOP.has(b)) continue;
        // "docs docs", "string string" — markup, never a concept
        if (sameWord(a, b)) continue;
        const label = a + " " + b;
        if (PHRASE_NOISE.test(label) || /\d/.test(label)) continue;
        const e = out.get(label) || {
          label,
          count: 0,
          surfaceCount: 0,
          chunkIds: new Set<number>(),
          key: conceptKey(label),
        };
        e.count++;
        e.surfaceCount++;
        e.chunkIds.add(c.id);
        out.set(label, e);
      }
    }
  }
  return out;
}

/** Capitalized runs not at sentence start: "Federal Circuit", "USPTO". */
function harvestNames(chunks: DocChunk[]): Map<string, Candidate> {
  const out = new Map<string, Candidate>();
  for (const c of chunks) {
    const re = /(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z0-9-]{2,}(?:\s[A-Z][a-zA-Z0-9-]{2,}){0,2})\b/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(c.text))) {
      const term = m[1];
      const wordsInTerm = term.split(" ");
      if (wordsInTerm.some((w) => ENTITY_STOP.has(w))) continue;
      if (isCitationArtifact(term)) continue;
      const e = out.get(term) || {
        label: term,
        count: 0,
        surfaceCount: 0,
        chunkIds: new Set<number>(),
        key: conceptKey(term),
      };
      e.count++;
      e.surfaceCount++;
      e.chunkIds.add(c.id);
      out.set(term, e);
    }
  }
  return out;
}

/** Merge candidates that mean the same thing, pooling their mentions:
 *  "Patent"/"Patents" and "America"/"American" become one node each, while
 *  "Patents"/"patent rights" stay distinct. Without this, one entity spends
 *  two of eleven slots on two spellings of itself. */
function mergeVariants(cands: Candidate[]): Candidate[] {
  const best = new Map<string, Candidate>();
  const variantOf = (key: string) => {
    if (best.has(key)) return key;
    const words = key.split(" ");
    for (const existing of best.keys()) {
      const other = existing.split(" ");
      if (other.length !== words.length) continue;
      if (words.every((w, i) => sameWord(w, other[i]))) return existing;
    }
    return null;
  };
  // strongest first, so the surviving surface form is the dominant spelling
  for (const c of [...cands].sort((a, b) => concentration(b) - concentration(a))) {
    const hit = variantOf(c.key);
    if (hit === null) {
      best.set(c.key, { ...c, chunkIds: new Set(c.chunkIds) });
      continue;
    }
    const prev = best.get(hit)!;
    // label with the spelling the document actually uses most — "Dispatcher"
    // over "Dispatch", not whichever variant happens to be shorter
    if (c.count > prev.surfaceCount) {
      prev.label = c.label;
      prev.surfaceCount = c.count;
    }
    prev.count += c.count;
    for (const id of c.chunkIds) prev.chunkIds.add(id);
  }
  return [...best.values()].sort((a, b) => concentration(b) - concentration(a));
}

/** True when a stronger pick already covers this phrase: it repeats one of
 *  that phrase's words and lives almost entirely in the same chunks, so as a
 *  node it would boost nothing new and only spend a slot. "trade secret law"
 *  behind "trade secret" is the case this exists for — the bigram harvest
 *  sees it as "secret law", which reads like a separate concept and is not
 *  one. */
function isEclipsedBy(c: Candidate, picked: Candidate[]): boolean {
  const words = new Set(c.key.split(" "));
  return picked.some((p) => {
    if (!p.key.split(" ").some((w) => words.has(w))) return false;
    let shared = 0;
    for (const id of c.chunkIds) if (p.chunkIds.has(id)) shared++;
    return shared / c.chunkIds.size >= 0.85;
  });
}

const titleCase = (s: string) =>
  s.replace(/\b[a-z]/g, (ch) => ch.toUpperCase());

// Eleven nodes, two thirds of them subject-matter phrases. A single merged
// ranking would not do: concentration scores are much larger for phrases
// (which recur far more often than any proper noun), so proper names —
// "USPTO", "Federal Circuit" — would be shut out entirely and the
// constellation would lose the concrete anchors that make it readable.
const GRAPH_NODES = 11;
const NAME_SLOTS = 4;

export function extractEntityGraph(chunks: DocChunk[]): EntityGraph {
  // a phrase must recur across a real slice of the document, not spike
  // inside one worked example — concentration alone would rank a single
  // case study's vocabulary above the document's actual themes
  const minPhraseDf = Math.max(3, Math.round(chunks.length * 0.015));
  const phrases = mergeVariants([...harvestPhrases(chunks).values()]).filter(
    (c) => c.count >= 3 && c.chunkIds.size >= minPhraseDf,
  );
  const names = mergeVariants([...harvestNames(chunks).values()]).filter(
    // df >= 2: a name confined to one chunk can form no edge, and at this
    // end of the distribution single-chunk terms are almost all artifacts
    // ("Whether" after a colon, a one-off case name)
    (c) => c.count >= 2 && c.chunkIds.size >= 2,
  );

  // a name and a phrase can be the same entity ("United States"); the name's
  // surface form wins and the phrase copy is dropped
  const takenNames = names.slice(0, NAME_SLOTS);
  const nameKeys = new Set(takenNames.map((c) => c.key));
  const takenPhrases: Candidate[] = [];
  for (const c of phrases) {
    if (takenPhrases.length >= GRAPH_NODES - takenNames.length) break;
    if (nameKeys.has(c.key)) continue;
    if (isEclipsedBy(c, [...takenNames, ...takenPhrases])) continue;
    takenPhrases.push(c);
  }
  const picked = [...takenNames, ...takenPhrases];
  // whichever pool the document is short on, the other fills the gap
  const pickedKeys = new Set(picked.map((c) => c.key));
  for (const c of [...names, ...phrases]) {
    if (picked.length >= GRAPH_NODES) break;
    if (pickedKeys.has(c.key)) continue;
    pickedKeys.add(c.key);
    picked.push(c);
  }
  picked.sort((a, b) => concentration(b) - concentration(a));

  const perChunk: string[][] = chunks.map((c) =>
    picked.filter((p) => p.chunkIds.has(c.id)).map((p) => p.key),
  );

  // golden-spiral placement on the unit sphere
  const nodes: ExtractedEntity[] = picked.map((cand, i) => {
    const n = picked.length;
    const y = n > 1 ? 1 - (i / (n - 1)) * 2 : 0;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const th = i * 2.39996;
    const label = /[A-Z]/.test(cand.label) ? cand.label : titleCase(cand.label);
    return {
      label: label.length > 16 ? label.slice(0, 15) + "…" : label,
      full: label,
      count: cand.count,
      chunkIds: cand.chunkIds,
      p: [Math.cos(th) * r, y * 0.85, Math.sin(th) * r] as [number, number, number],
      desc: `${cand.count} mentions across ${cand.chunkIds.size} chunk${cand.chunkIds.size > 1 ? "s" : ""} of this document.`,
    };
  });

  const index = new Map(picked.map((c, i) => [c.key, i]));
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
    const planned = res.planSubqueries ?? [];
    cards.push({
      kind: "tool",
      label: planned.length ? "plan queries" : "refine query",
      meta: "agent step",
      score: planned.length || refined.length ? "+" : "·",
      scoreN: planned.length || refined.length ? 0.8 : 0.3,
      // Planned path: the model's own decomposition, verbatim — the same
      // show-the-real-reasoning move as corrective's verdict cards. PRF
      // fallback keeps the pre-/plan wording.
      snippet: planned.length
        ? 'split into "' +
          planned.join('" · "') +
          '"' +
          (res.planRationale ? " — " + res.planRationale : "")
        : refined.length
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
      // Whatever is shown here must be the thing that actually drove the
      // rejection. With a model verdict that's the model's own sentence,
      // and there is no meaningful number to print — showing the cosine
      // beside a model rejection is what made this card once read "graded
      // 0.95 — below threshold". Without a verdict, fall back to the
      // heuristic and print the score it actually thresholded on.
      const graded = passScore(s);
      const why = s.verdict?.why?.trim();
      cards.push({
        kind: "reject",
        label: "chunk #" + s.chunk.id,
        meta: "rejected",
        score: s.verdict ? "✕" : graded.toFixed(2),
        scoreN: gradedBar(s),
        snippet: s.verdict
          ? (why || "the grader found nothing here that answers the question") +
            " — re-retrieval triggered"
          : `graded ${graded.toFixed(2)} — below threshold, re-retrieval triggered`,
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
  /\b(www|org|com|edu|net|http|https|html|pdf|isbn|openstax|access|accessed|retrieved|wikimedia|commons|free|page|pages|figure|table|chapter|section|appendix|index|contents|answer|key|credit|rule|rules|fed|civ|supp|cir|inc|llc|ibid|seq|stat|reg|vol|pub|press|review|journal|university|learning|objectives?|completing|question|questions|assessment|examples?|default|defaults|returns|parameter|parameters|usage|deprecated)\b/;

/** Chunk-frequency of each phrase the document actually contains. Shares
 *  harvestPhrases with the entity graph so "a phrase" means one thing here
 *  and there — a suggested question built from a URL path ("What is docs
 *  docs?") has the same root cause as a graph node built from one. */
function bigramDf(chunks: DocChunk[]): Map<string, number> {
  const df = new Map<string, number>();
  for (const [label, c] of harvestPhrases(chunks)) df.set(label, c.chunkIds.size);
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
