import type { RagId, Source } from "./types";

export const ACCENTS: Record<RagId, string> = {
  naive: "#5B8DEF",
  hybrid: "#9B7BF0",
  agentic: "#F2A93B",
  corrective: "#2BB673",
};

export const REJECT = "#d2655e";

export const STEP_MS = 560;
export const INDEX_MS = 4400;
export const STREAM_WORD_MS = 34;

export const TABS: { id: RagId; label: string; sub: string }[] = [
  { id: "naive", label: "Basic", sub: "BASELINE" },
  { id: "hybrid", label: "Hybrid", sub: "VECTOR + GRAPH" },
  { id: "corrective", label: "Corrective", sub: "GRADE · CORRECT" },
  { id: "agentic", label: "Agentic", sub: "PLAN · LOOP" },
];

export const NAMES: Record<RagId, string> = {
  naive: "Basic RAG",
  hybrid: "Hybrid RAG",
  agentic: "Agentic RAG",
  corrective: "Corrective RAG",
};

export const BLURBS: Record<RagId, string> = {
  naive:
    "A single straight path — embed the query, search the vector store, stuff the top chunks into a prompt.",
  hybrid:
    "Indexes the document as both a vector store and a knowledge graph, then retrieves from both in parallel and merges.",
  agentic:
    "An agent plans, retrieves in a loop, refines its own query and can call tools before committing to an answer.",
  corrective:
    "Retrieves, grades each chunk for relevance, then re-retrieves or corrects before the answer is generated.",
};

// generic fallback — real suggestions are generated from the loaded document
export const SUGGESTIONS = [
  "What is this document about?",
  "What are the key terms defined here?",
  "Summarize the main argument.",
];

export const QUERY_CAPTIONS: Record<RagId, string> = {
  naive: "QUERY · embed → vector search → prompt → generate",
  hybrid: "QUERY · retrieving from vector store + knowledge graph in parallel",
  corrective: "QUERY · retrieve → grade chunks → search & correct → generate",
  agentic: "QUERY · agent plans → loops retrieval → calls tools → generate",
};

export function indexCaptions(
  rag: RagId,
  pages = 15,
  chunks = 64,
): string[] {
  return [
    `INDEXING · separating ${pages} page${pages === 1 ? "" : "s"}`,
    `INDEXING · splitting into ${chunks} chunks`,
    "INDEXING · embedding chunks → 768-d",
    rag === "hybrid"
      ? "INDEXING · extracting entities → graph"
      : "INDEXING · arranging by similarity",
  ];
}

