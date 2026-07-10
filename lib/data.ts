// Scripted demo data: 60 chunk dots clustered for the vector scatter, and the
// 11-entity knowledge graph for the Hybrid tab. Deterministic (seeded RNG) so
// every load looks identical.

export interface Chunk {
  nx: number;
  ny: number;
  cl: number;
  page: number;
  idx: number;
  delay: number;
}

export interface GraphNode {
  label: string;
  p: [number, number, number];
  desc: string;
}

export interface GraphEdge {
  a: number;
  b: number;
}

export function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const r = rng(7);
const clusters: [number, number][] = [
  [0.27, 0.3],
  [0.7, 0.26],
  [0.5, 0.56],
  [0.26, 0.74],
  [0.74, 0.7],
];

export const CHUNKS: Chunk[] = [];
for (let i = 0; i < 60; i++) {
  const c = clusters[i % 5];
  const a = r() * Math.PI * 2;
  const rad = Math.pow(r(), 0.7) * 0.13;
  CHUNKS.push({
    nx: Math.max(0.06, Math.min(0.94, c[0] + Math.cos(a) * rad * 1.1)),
    ny: Math.max(0.08, Math.min(0.92, c[1] + Math.sin(a) * rad)),
    cl: i % 5,
    page: 1 + Math.floor(r() * 15),
    idx: i,
    delay: r() * 0.45,
  });
}

export const REL_NAIVE = CHUNKS.filter((c) => c.cl === 2)
  .slice(0, 6)
  .map((c) => c.idx);

export const GNODES: GraphNode[] = [
  { label: "Transformer", p: [0, 0, 0], desc: "Core architecture — attention only, no recurrence or convolution." },
  { label: "Self-Attention", p: [-0.55, -0.45, 0.35], desc: "Relates every position to all others in one sequence." },
  { label: "Multi-Head", p: [-0.9, 0.15, -0.15], desc: "8 parallel attention heads, 64 dims each." },
  { label: "Encoder", p: [-0.1, -0.78, -0.3], desc: "Stack of 6 self-attention + feed-forward layers." },
  { label: "Decoder", p: [0.55, -0.5, 0.25], desc: "Masked self-attention plus encoder-decoder attention." },
  { label: "Pos. Encoding", p: [0.86, 0.1, -0.4], desc: "Sinusoidal signal that injects token order." },
  { label: "Feed-Forward", p: [0.32, 0.55, 0.4], desc: "Position-wise two-layer MLP after attention." },
  { label: "BLEU", p: [-0.45, 0.66, -0.2], desc: "Translation metric — 28.4 on EN-DE." },
  { label: "WMT 2014", p: [-0.82, 0.5, 0.35], desc: "Benchmark dataset used in the experiments." },
  { label: "Adam", p: [0.76, 0.6, 0.1], desc: "Optimizer with a warmup learning-rate schedule." },
  { label: "Dropout", p: [0.86, -0.26, 0.42], desc: "Regularization, p=0.1 on each sub-layer." },
];

const E = (a: number, b: number): GraphEdge => ({ a, b });
export const GEDGES: GraphEdge[] = [
  E(0, 1), E(0, 3), E(0, 4), E(1, 2), E(3, 6), E(4, 5), E(0, 6),
  E(7, 8), E(0, 7), E(4, 10), E(3, 5), E(1, 4), E(0, 2),
];

export const G_ACTIVE = [0, 1, 2, 3, 4, 6];

export const G_NEIGHBORS: Record<number, Set<number>> = {};
GNODES.forEach((_, i) => (G_NEIGHBORS[i] = new Set()));
GEDGES.forEach((e) => {
  G_NEIGHBORS[e.a].add(e.b);
  G_NEIGHBORS[e.b].add(e.a);
});
