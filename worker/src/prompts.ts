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

// ---------- relevance grading (corrective mode's real grader) ----------

export interface GradeVerdict {
  /** Index into the passage array exactly as sent, so the client can map a
   *  verdict back to its chunk without the model having to echo chunk ids
   *  (which it gets wrong far more often than a 0-based position). */
  i: number;
  relevant: boolean;
  why: string;
}

// ONE passage per call, deliberately — not a batch.
//
// A batched prompt ("grade all 5, return 5 verdicts") is the obvious design
// and it does not work: measured against real passages, the 8B model
// returned 1 verdict for 3 passages and 2 for 5, at 47 completion tokens
// against a 500-token ceiling — so it wasn't truncation, it just declined
// to enumerate. Restating the exact indices in the prompt didn't fix it.
// The 1B model was worse: malformed JSON, and it hallucinated a passage
// about Napster as "describes the relationship between weather and crop
// yields" — confidently wrong in exactly the way the grader exists to
// prevent.
//
// Grading one passage per call removes enumeration from the model's job
// entirely: the count is guaranteed by the caller's Promise.all, and a
// single passage that fails to grade degrades on its own instead of
// voiding the whole set. It costs roughly 2x the input tokens (the rules
// are re-sent per passage) — about 11 neurons per corrective query, which
// is nothing against the 8,000/day ladder.
const GRADE_RULES = [
  "A passage is relevant ONLY if it contains information that helps answer the question. Being about the same broad subject is NOT enough.",
  "Not relevant: tables of contents, chapter headings, learning objectives, quiz questions and their answer keys, citation lists, and passages that merely mention a term without explaining it.",
  'For "why", write at most 12 words describing what this passage actually contains, in lower case, with no trailing period. It is shown to the user verbatim, so never mention passage numbers, grading, or relevance scores.',
].join("\n");

// The aux model does NOT support JSON Schema — Workers AI rejects
// response_format json_schema with error 5025 — so the output contract has
// to live in the prompt, stated explicitly and shown as a literal example.
// json_object mode IS honored for the 8B model (it hands back an already
// parsed object) but is only a hint for the 1B one, which will answer
// "PASSAGE VERDICT: irrelevant" in prose. The Worker validates every field
// regardless. Do not "simplify" this back to a schema reference.
const GRADE_FORMAT = `Reply with JSON and nothing else. No prose, no explanation, no markdown code fences.
Use exactly this shape:
{"relevant":true,"why":"defines the term and gives an example"}
"why" is required and must never be empty, including when relevant is false — say what the passage does contain, e.g. {"relevant":false,"why":"a list of chapter learning objectives"}`;

/** Grades a SINGLE passage for relevance to the query — see the note above
 *  for why this is not batched. Much smaller than buildAnswerPrompt: this
 *  runs on the aux model and asks for a classification, not prose. */
export function buildGradePrompt(query: string, passage: PassageForPrompt): BuiltPrompt {
  return {
    system: `You judge whether one retrieved passage helps answer a question.\n${GRADE_RULES}\n\n${GRADE_FORMAT}`,
    // Question repeated after the passage: the model attends better to the
    // end of the prompt, and stated only up front it drifts into
    // summarizing the passage instead of judging it.
    user: `PASSAGE\n${passage.text}\n\nQUESTION: ${query}\n\nDoes that passage help answer that question? JSON only.`,
  };
}
