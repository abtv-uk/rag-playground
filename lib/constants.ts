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

export const ANSWERS: Record<RagId, string> = {
  naive:
    "Multi-head attention runs several attention functions in parallel. Each head projects the queries, keys and values into a different learned subspace, the outputs are concatenated and projected once more — letting the model attend to information from several representation subspaces at once.",
  hybrid:
    "Multi-head attention attends in parallel across several projected subspaces. The knowledge graph confirms it sits inside both the Encoder and Decoder stacks and feeds the position-wise Feed-Forward layer, so this answer is grounded in both the passage text and the linked Transformer components.",
  agentic:
    'The agent first retrieved the definition, noticed it referenced "heads" without a count, then re-queried for specifics. The paper uses 8 parallel attention heads, each of dimension 64, concatenated back to the model dimension of 512 before a final linear projection.',
  corrective:
    "Initial retrieval surfaced two weakly-relevant chunks which the grader rejected. After re-retrieval the answer was generated only from high-scoring passages: multi-head attention performs 8 parallel attention functions and concatenates them, attending to different subspaces simultaneously.",
};

export const SUGGESTIONS = [
  "What is multi-head attention?",
  "How does positional encoding work?",
  "Why drop recurrence?",
];

export const QUERY_CAPTIONS: Record<RagId, string> = {
  naive: "QUERY · embed → vector search → prompt → generate",
  hybrid: "QUERY · retrieving from vector store + knowledge graph in parallel",
  corrective: "QUERY · retrieve → grade chunks → search & correct → generate",
  agentic: "QUERY · agent plans → loops retrieval → calls tools → generate",
};

export function indexCaptions(rag: RagId): string[] {
  return [
    "INDEXING · separating 15 pages",
    "INDEXING · splitting into 64 chunks",
    "INDEXING · embedding chunks → 768-d",
    rag === "hybrid"
      ? "INDEXING · extracting entities → graph"
      : "INDEXING · arranging by similarity",
  ];
}

export function buildSources(rag: RagId): Source[] {
  const A = ACCENTS[rag];
  const chunk = (id: number, p: number, score: number, snip: string): Source => ({
    kind: "chunk",
    label: "chunk #" + id,
    meta: "p." + p,
    score: score.toFixed(2),
    scoreN: score,
    snippet: snip,
    color: A,
  });
  const node = (l: string, score: number, snip: string): Source => ({
    kind: "node",
    label: l,
    meta: "entity",
    score: score.toFixed(2),
    scoreN: score,
    snippet: snip,
    color: A,
  });
  if (rag === "hybrid")
    return [
      chunk(14, 3, 0.91, "…employ h = 8 parallel attention layers, or heads…"),
      node("Multi-Head", 0.88, "linked to Self-Attention, Encoder, Decoder"),
      node("Feed-Forward", 0.74, "position-wise, applied after attention"),
      chunk(22, 4, 0.69, "…each head outputs are concatenated and projected…"),
    ];
  if (rag === "agentic")
    return [
      chunk(14, 3, 0.92, "…employ h = 8 parallel attention layers, or heads…"),
      {
        kind: "tool",
        label: "refine query",
        meta: "agent step",
        score: "+",
        scoreN: 0.8,
        snippet: 're-queried "number of attention heads"',
        color: A,
      },
      chunk(31, 6, 0.83, "…d_model / h = 64 dimensions per head…"),
    ];
  if (rag === "corrective")
    return [
      {
        kind: "reject",
        label: "chunk #07",
        meta: "rejected",
        score: "0.31",
        scoreN: 0.31,
        snippet: "graded irrelevant — re-retrieval triggered",
        color: A,
        rejected: true,
      },
      chunk(14, 3, 0.93, "…employ h = 8 parallel attention layers, or heads…"),
      chunk(22, 4, 0.79, "…outputs are concatenated and once again projected…"),
    ];
  return [
    chunk(14, 3, 0.89, "…employ h = 8 parallel attention layers, or heads…"),
    chunk(22, 4, 0.72, "…the outputs are concatenated and projected…"),
    chunk(9, 2, 0.58, "…attention allows modeling of dependencies…"),
  ];
}
