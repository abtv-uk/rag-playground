import type { DenseIndex } from "./embeddings";

export type RagId = "naive" | "hybrid" | "corrective" | "agentic";

export type Phase = "empty" | "indexing" | "ready" | "querying" | "answered";

export type SourceKind = "chunk" | "node" | "tool" | "reject";

export interface Source {
  kind: SourceKind;
  label: string;
  meta: string;
  score: string;
  scoreN: number;
  snippet: string;
  color: string;
  rejected?: boolean;
  /** Chunk this card came from, so an inline [n] citation in the answer can
   *  point at it. Absent for cards that aren't backed by a chunk (entity
   *  nodes, agent steps). */
  chunkId?: number;
  /** 1-based position in the passage list handed to the generator — the
   *  number the model actually cites. */
  citation?: number;
}

export interface QueryStep {
  lit: string[];
  e: string[];
  panel?: string;
  hl?: 1;
  merge?: 1;
  grade?: 1;
  correct?: 1;
  loop?: 1;
  refine?: 1;
  plan?: 1;
  stream?: 1;
}

export interface LoadedDocInfo {
  name: string;
  sizeLabel: string;
  pages: number;
  chunks: { id: number; page: number; text: string }[];
  sourceUrl?: string;
  /** Set only by loadSampleDoc — see LoadedDoc in lib/document.ts for why
   *  this is a trust boundary, not just a display flag. */
  isSample?: boolean;
  /** See LoadedDoc.dense in lib/document.ts. */
  dense?: DenseIndex;
}

/** What the generation backend is doing right now. Drives the output-panel
 *  status chip, the `llm` node's sub-label, and the wait ring — so a slow
 *  answer reads as progress rather than a hang. */
export type GenPhase =
  | "idle"
  | "embedding"
  | "planning"
  | "grading"
  | "waiting"
  | "generating";

export interface PlaygroundState {
  rag: RagId;
  phase: Phase;
  expanded: boolean;
  dark: boolean;
  query: string;
  answer: string;
  streaming: boolean;
  sources: Source[];
  sourcesVisible: boolean;
  idxStage: number;
  doc: LoadedDocInfo | null;
  loading: boolean;
  loadingMsg: string;
  loadError: string;
  suggestions: string[];
  /** True when the shown answer is the offline extractive fallback rather
   *  than generated prose — surfaced in the UI instead of failing silently. */
  degraded: boolean;
  genPhase: GenPhase;
  /** True once the mount-time health check confirms the generation Worker
   *  has no capacity left today (both providers exhausted, or unreachable).
   *  Lets the UI say so before the first query, instead of only discovering
   *  it 6s into every answer. */
  generatorOffline: boolean;
  /** Live progress while an uploaded document's chunks are being embedded
   *  in the background (the sample never sets this — it ships pre-embedded).
   *  Null once embedding finishes or isn't running. */
  embedProgress: { done: number; total: number } | null;
}
