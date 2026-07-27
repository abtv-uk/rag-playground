// SceneData is everything the canvas renderer draws that depends on the
// loaded document: scatter dots, highlighted chunks, the knowledge graph,
// and the grading rows. The bundled sample document uses the scripted demo
// data; uploaded documents get a scene built from real chunks/entities.

import {
  CHUNKS,
  GEDGES,
  GNODES,
  G_ACTIVE,
  G_NEIGHBORS,
  REL_NAIVE,
  rng,
  type Chunk,
  type GraphEdge,
  type GraphNode,
} from "./data";
import type { LoadedDoc } from "./document";
import type { DenseIndex } from "./embeddings";
import { kmeans, pca2, projectPoint, type Pca2Basis } from "./projection";
import {
  extractEntityGraph,
  gradedBar,
  gradedPass,
  type RetrievalResult,
} from "./retrieval";

export interface GradeRow {
  n: number;
  pass: 0 | 1;
  s: number;
  graded?: boolean; // false = never scored (drawn in the idle "···" style)
}

/** Everything needed to place a *new* vector in the same 2-d frame the dots
 *  were laid out in — kept on the scene so the query marker shares the
 *  document's projection instead of being fitted separately (which would
 *  make its position meaningless to compare against the dots). */
export interface SceneProjection {
  basis: Pca2Basis;
  /** Component-score extents used to normalize into panel space. */
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  /** Share of total variance the two drawn axes actually capture. Reported
   *  in the panel subtitle rather than quietly omitted — for 768-d
   *  embeddings this is legitimately ~11%, and pretending otherwise would
   *  be the same dishonesty this phase exists to remove. */
  explained: [number, number];
}

export interface SceneData {
  dots: Chunk[];
  rel: number[]; // dot indices highlighted when a query lights the panel
  relFinal: number[]; // post-correction/loop context (corrective, agentic)
  chunkCount: number;
  docLabel: string;
  gnodes: (GraphNode & { chunkIds?: Set<number>; full?: string })[];
  gedges: GraphEdge[];
  gActive: number[];
  gnbr: Record<number, Set<number>>;
  gradeRows: GradeRow[];
  /** Present only once real embeddings have been projected — absent for the
   *  scripted demo scene, and for a document whose embedding failed or has
   *  not finished. Its absence is what tells the renderer the scatter is
   *  decorative rather than semantic. */
  proj?: SceneProjection;
  /** The current query's position in the same frame, or null when there
   *  isn't one (no query yet, or no projection to place it in). */
  queryDot?: { nx: number; ny: number } | null;
}

const SAMPLE_GRADE_ROWS: GradeRow[] = [
  { n: 14, pass: 1, s: 0.93 },
  { n: 7, pass: 0, s: 0.31 },
  { n: 22, pass: 1, s: 0.79 },
  { n: 31, pass: 0, s: 0.28 },
  { n: 9, pass: 1, s: 0.61 },
];

export function sampleScene(): SceneData {
  return {
    dots: CHUNKS,
    rel: REL_NAIVE,
    relFinal: REL_NAIVE,
    chunkCount: 64,
    docLabel: "attention.pdf",
    gnodes: GNODES,
    gedges: GEDGES,
    gActive: G_ACTIVE,
    gnbr: G_NEIGHBORS,
    gradeRows: SAMPLE_GRADE_ROWS.map((r) => ({ ...r })),
  };
}

const CLUSTERS: [number, number][] = [
  [0.27, 0.3],
  [0.7, 0.26],
  [0.5, 0.56],
  [0.26, 0.74],
  [0.74, 0.7],
];

// Panel-space bounds the scatter is normalized into. Shared by the PRNG
// fallback and the real projection so switching between them doesn't change
// the cloud's footprint.
const NX_MIN = 0.06;
const NX_MAX = 0.94;
const NY_MIN = 0.08;
const NY_MAX = 0.92;

/** Replaces the decorative layout with the document's real semantic
 *  structure: dot positions become PCA components 1-2 of the actual
 *  embeddings, and `cl` becomes a k-means label instead of `page % 5`.
 *
 *  Mutates `scene.dots` in place and deliberately does NOT reorder them —
 *  applyQueryToScene maps `chunk.id - 1` straight to a dot index, so any
 *  reordering here would silently highlight the wrong dots.
 *
 *  Safe to call late: uploads only get their embeddings after a background
 *  pass that finishes well after buildScene, so this runs a second time
 *  then (see hooks/usePlayground.ts). */
export function applyProjection(scene: SceneData, dense: DenseIndex): void {
  const n = Math.min(dense.count, scene.dots.length);
  if (!n || !dense.dim) return;

  const { xy, mean, pc1, pc2, explained } = pca2(dense.vectors, dense.count, dense.dim);
  const labels = kmeans(dense.vectors, dense.count, dense.dim, CLUSTERS.length);

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xy[i * 2];
    const y = xy[i * 2 + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const proj: SceneProjection = {
    basis: { mean, pc1, pc2, explained },
    minX,
    maxX,
    minY,
    maxY,
    explained,
  };
  for (let i = 0; i < n; i++) {
    const [nx, ny] = toPanel(proj, xy[i * 2], xy[i * 2 + 1]);
    scene.dots[i].nx = nx;
    scene.dots[i].ny = ny;
    scene.dots[i].cl = labels[i];
  }
  scene.proj = proj;
}

/** Min-max normalize a component score into the panel's box. Clamped, so a
 *  query that projects outside the document's own extent (an off-topic one
 *  often does) still renders at the edge rather than escaping the panel. */
function toPanel(p: SceneProjection, x: number, y: number): [number, number] {
  const sx = p.maxX - p.minX || 1;
  const sy = p.maxY - p.minY || 1;
  const nx = NX_MIN + ((x - p.minX) / sx) * (NX_MAX - NX_MIN);
  const ny = NY_MIN + ((y - p.minY) / sy) * (NY_MAX - NY_MIN);
  return [
    Math.max(NX_MIN, Math.min(NX_MAX, nx)),
    Math.max(NY_MIN, Math.min(NY_MAX, ny)),
  ];
}

/** Places the query in the document's own projection. Null when there is no
 *  projection (lexical-only doc) or no query vector — the renderer then
 *  simply omits the marker rather than inventing a position. */
export function projectQuery(
  scene: SceneData,
  queryVec: Float32Array | null | undefined,
): void {
  if (!scene.proj || !queryVec) {
    scene.queryDot = null;
    return;
  }
  const [x, y] = projectPoint(scene.proj.basis, queryVec);
  const [nx, ny] = toPanel(scene.proj, x, y);
  scene.queryDot = { nx, ny };
}

export function buildScene(doc: LoadedDoc): SceneData {
  const r = rng(doc.chunks.length * 31 + doc.name.length);
  const dots: Chunk[] = doc.chunks.map((c, i) => {
    const cl = c.page % 5;
    const cc = CLUSTERS[cl];
    const a = r() * Math.PI * 2;
    const rad = Math.pow(r(), 0.7) * 0.13;
    return {
      nx: Math.max(NX_MIN, Math.min(NX_MAX, cc[0] + Math.cos(a) * rad * 1.1)),
      ny: Math.max(NY_MIN, Math.min(NY_MAX, cc[1] + Math.sin(a) * rad)),
      cl,
      page: c.page,
      idx: i,
      delay: r() * 0.45,
    };
  });
  const graph = extractEntityGraph(doc.chunks);
  const shortName =
    doc.name.length > 22 ? doc.name.slice(0, 21) + "…" : doc.name;
  const scene: SceneData = {
    dots,
    rel: [],
    relFinal: [],
    chunkCount: doc.chunks.length,
    docLabel: shortName,
    gnodes: graph.nodes.map((n) => ({
      label: n.label,
      full: n.full,
      p: n.p,
      desc: n.desc,
      chunkIds: n.chunkIds,
    })),
    gedges: graph.edges,
    gActive: [],
    gnbr: graph.neighbors,
    gradeRows: doc.chunks
      .slice(0, 5)
      .map((c) => ({ n: c.id, pass: 1 as const, s: 0, graded: false })),
    queryDot: null,
  };
  // The sample ships pre-embedded, so its real layout is available
  // immediately. An upload has no vectors yet at this point — the hook
  // calls applyProjection again once background embedding lands.
  if (doc.dense) applyProjection(scene, doc.dense);
  return scene;
}

/** Update the scene with real retrieval results before a query animates.
 *  `queryVec` is optional: without one (lexical-only retrieval, or a doc
 *  with no projection) the query marker is simply cleared. */
export function applyQueryToScene(
  scene: SceneData,
  res: RetrievalResult,
  queryVec?: Float32Array | null,
) {
  const toDots = (list: { chunk: { id: number } }[]) =>
    list.map((s) => s.chunk.id - 1).filter((i) => i < scene.dots.length);
  scene.rel = toDots(res.initialTop);
  scene.relFinal = toDots(res.finalTop);
  projectQuery(scene, queryVec);

  if (res.graphBoosted) {
    scene.gActive = res.graphBoosted;
  } else {
    const topIds = new Set(res.finalTop.map((s) => s.chunk.id));
    scene.gActive = scene.gnodes
      .map((n, i) => ({
        i,
        hits: n.chunkIds ? [...n.chunkIds].filter((id) => topIds.has(id)).length : 0,
      }))
      .filter((x) => x.hits > 0)
      .sort((a, b) => b.hits - a.hits)
      .slice(0, 6)
      .map((x) => x.i);
  }

  // grading rows: only chunks that were actually scored in pass 1; pad the
  // panel with unscored rows drawn in the idle style. Both the tick/cross
  // and the bar go through gradedPass/gradedBar so a real model verdict
  // wins when corrective had one, and the cosine-floor heuristic is used
  // otherwise — never a raw threshold on `.score`, which under dense
  // retrieval is an RRF weight rather than anything PASS_THRESHOLD is
  // calibrated against. See retrieval.ts for both.
  const rows: GradeRow[] = res.initialTop.slice(0, 5).map((s) => ({
    n: s.chunk.id,
    pass: (gradedPass(s) ? 1 : 0) as 0 | 1,
    s: gradedBar(s),
    graded: true,
  }));
  for (const c of scene.dots) {
    if (rows.length >= 5) break;
    if (rows.some((r) => r.n === c.idx + 1)) continue;
    rows.push({ n: c.idx + 1, pass: 1, s: 0, graded: false });
  }
  scene.gradeRows = rows;
}
