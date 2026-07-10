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
import { extractEntityGraph, type RetrievalResult } from "./retrieval";

export interface GradeRow {
  n: number;
  pass: 0 | 1;
  s: number;
}

export interface SceneData {
  dots: Chunk[];
  rel: number[]; // dot indices highlighted when a query lights the panel
  chunkCount: number;
  docLabel: string;
  gnodes: (GraphNode & { chunkIds?: Set<number> })[];
  gedges: GraphEdge[];
  gActive: number[];
  gnbr: Record<number, Set<number>>;
  gradeRows: GradeRow[];
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

export function buildScene(doc: LoadedDoc): SceneData {
  const r = rng(doc.chunks.length * 31 + doc.name.length);
  const dots: Chunk[] = doc.chunks.map((c, i) => {
    const cl = c.page % 5;
    const cc = CLUSTERS[cl];
    const a = r() * Math.PI * 2;
    const rad = Math.pow(r(), 0.7) * 0.13;
    return {
      nx: Math.max(0.06, Math.min(0.94, cc[0] + Math.cos(a) * rad * 1.1)),
      ny: Math.max(0.08, Math.min(0.92, cc[1] + Math.sin(a) * rad)),
      cl,
      page: c.page,
      idx: i,
      delay: r() * 0.45,
    };
  });
  const graph = extractEntityGraph(doc.chunks);
  const shortName =
    doc.name.length > 22 ? doc.name.slice(0, 21) + "…" : doc.name;
  return {
    dots,
    rel: [],
    chunkCount: doc.chunks.length,
    docLabel: shortName,
    gnodes: graph.nodes.map((n) => ({
      label: n.label,
      p: n.p,
      desc: n.desc,
      chunkIds: n.chunkIds,
    })),
    gedges: graph.edges,
    gActive: [],
    gnbr: graph.neighbors,
    gradeRows: doc.chunks
      .slice(0, 5)
      .map((c) => ({ n: c.id, pass: 1 as const, s: 0 })),
  };
}

const PASS_THRESHOLD = 0.45;

/** Update the scene with real retrieval results before a query animates. */
export function applyQueryToScene(scene: SceneData, res: RetrievalResult) {
  scene.rel = res.top.map((s) => s.chunk.id - 1).filter((i) => i < scene.dots.length);
  const topIds = new Set(res.top.map((s) => s.chunk.id));
  scene.gActive = scene.gnodes
    .map((n, i) => ({
      i,
      hits: n.chunkIds ? [...n.chunkIds].filter((id) => topIds.has(id)).length : 0,
    }))
    .filter((x) => x.hits > 0)
    .sort((a, b) => b.hits - a.hits)
    .slice(0, 6)
    .map((x) => x.i);
  const rows = res.ranked.slice(0, 5);
  scene.gradeRows = rows.length
    ? rows.map((s) => ({
        n: s.chunk.id,
        pass: (s.score >= PASS_THRESHOLD ? 1 : 0) as 0 | 1,
        s: Math.max(0.05, s.score),
      }))
    : scene.gradeRows.map((r) => ({ ...r, pass: 0 as const, s: 0.05 }));
}
