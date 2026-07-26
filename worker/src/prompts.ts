// Prompt construction for grounded, cited generation. The contract here is
// the direct fix for the extractive answer's failure mode: given passages
// that DO contain a definition, extractAnswer (lib/retrieval.ts) could only
// concatenate top-scoring sentences in document order, never compose one.
// Rules 3-4 force connected prose with a definitional lead instead.

export type RagId = "naive" | "hybrid" | "corrective" | "agentic";

export interface PassageForPrompt {
  page: number;
  text: string;
}

const MODE_FRAME: Record<RagId, string> = {
  naive: "a single-pass vector search",
  hybrid: "a merged vector + knowledge-graph search",
  corrective:
    "a retrieval pass that graded chunks for relevance and re-retrieved to replace any that were rejected",
  agentic: "an agent that retrieved once, refined its query, and retrieved again",
};

const SYSTEM_RULES = [
  "1. Use only the passages. Never use outside knowledge.",
  "2. Cite every factual claim with its passage number in square brackets: [2]. A sentence may cite more than one: [1][4].",
  '3. Write connected prose that directly answers the question — do not list or quote passages, and never mention "the passages" or the retrieval process.',
  "4. If the passages define a term, lead with the definition.",
  "5. If the passages do not answer the question, say in one sentence what they do cover instead. Do not speculate.",
  "6. 3-6 sentences. No preamble, no headings, no bullet points, no markdown formatting (no asterisks, no bold/italics) — plain prose only.",
].join("\n");

export interface BuiltPrompt {
  system: string;
  user: string;
}

export function buildAnswerPrompt(
  rag: RagId,
  query: string,
  passages: PassageForPrompt[],
): BuiltPrompt {
  const frame = MODE_FRAME[rag] ?? MODE_FRAME.naive;
  const context = passages
    .map((p, i) => `[${i + 1}] (p.${p.page}) ${p.text}`)
    .join("\n\n");
  return {
    system: `You answer questions strictly from the numbered passages provided, produced by ${frame}.\n${SYSTEM_RULES}`,
    user: `PASSAGES\n${context}\n\nQUESTION: ${query}`,
  };
}
