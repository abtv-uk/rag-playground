import type { QueryStep, RagId } from "./types";

const S = (lit: string[], e: string[], opt?: Partial<QueryStep>): QueryStep => ({
  lit,
  e,
  ...opt,
});

export function steps(rag: RagId): QueryStep[] {
  if (rag === "naive")
    return [
      S(["qn"], ["eq"]),
      S(["embed"], ["e1"]),
      S(["vector"], ["e2"], { panel: "v" }),
      S(["vector"], [], { hl: 1 }),
      S(["prompt"], ["e3", "eQP"]),
      S(["llm"], ["e6"]),
      S(["output"], ["e8"], { stream: 1 }),
    ];
  if (rag === "hybrid")
    return [
      S(["qn"], ["eq"]),
      S(["embed"], ["e1"]),
      S(["vector", "sec"], ["e2", "e4"], { panel: "b" }),
      S(["vector", "sec"], [], { hl: 1, merge: 1 }),
      S(["prompt"], ["e3", "e5", "eQP"]),
      S(["llm"], ["e6"]),
      S(["output"], ["e8"], { stream: 1 }),
    ];
  if (rag === "corrective")
    return [
      S(["qn"], ["eq"]),
      S(["embed"], ["e1"]),
      S(["vector"], ["e2"], { panel: "v" }),
      S(["vector"], [], { hl: 1 }),
      S(["sec"], ["e3g"], { grade: 1 }),
      S(["sec", "vector"], ["eRe"], { grade: 1, correct: 1 }),
      S(["prompt"], ["egp", "eQP"]),
      S(["llm"], ["e6"]),
      S(["output"], ["e8"], { stream: 1 }),
    ];
  return [
    S(["qn"], ["eq"]),
    S(["agent"], ["e1"]),
    S(["mem", "plan", "agent"], ["eM", "eP"], { plan: 1 }),
    S(["vector", "agent"], ["eAV"], { panel: "v" }),
    S(["vector", "agent"], ["eVA"], { hl: 1, loop: 1 }),
    S(["tools", "agent"], ["eT"], { loop: 1, refine: 1 }),
    S(["llm"], ["eL"]),
    S(["output"], ["e8"], { stream: 1 }),
  ];
}
