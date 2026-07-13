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
}
