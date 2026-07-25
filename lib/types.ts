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
}
