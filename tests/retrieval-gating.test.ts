// Corrective mode's pass/reject decision chain, as pure-function units.
//
// The layering these encode, from most to least authoritative: a real
// model verdict from POST /grade wins outright; without one, the absolute
// cosine floor catches off-topic documents that background-normalized
// relevance mathematically cannot (a top-ranked chunk is above its own
// document's mean by construction, so sigmoid(z) > 0.5 always — relevance
// alone can never reject); without dense scoring at all, the TF-IDF score
// is thresholded directly.
import { describe, expect, it } from "vitest";
import {
  PASS_THRESHOLD,
  gradedBar,
  gradedPass,
  passScore,
  type ScoredChunk,
} from "../lib/retrieval";

const stub = (over: Partial<ScoredChunk>): ScoredChunk => ({
  chunk: { id: 1, page: 1, text: "x" },
  raw: 0,
  score: 0.5,
  ...over,
});

describe("passScore", () => {
  it("returns 0 when cosine sits below the absolute floor, whatever relevance says", () => {
    expect(passScore(stub({ relevance: 0.7, cosine: 0.54 }))).toBe(0);
  });

  it("returns relevance once cosine clears the floor", () => {
    expect(passScore(stub({ relevance: 0.7, cosine: 0.56 }))).toBe(0.7);
  });

  it("falls back to the lexical score when no dense relevance exists", () => {
    expect(passScore(stub({ score: 0.62 }))).toBe(0.62);
  });
});

describe("gradedPass", () => {
  it("a model verdict overrides everything — reject despite strong scores", () => {
    expect(
      gradedPass(
        stub({ verdict: { relevant: false, why: "" }, relevance: 0.9, cosine: 0.9 }),
      ),
    ).toBe(false);
  });

  it("a model verdict overrides everything — pass despite weak scores", () => {
    expect(
      gradedPass(
        stub({ verdict: { relevant: true, why: "" }, relevance: 0.1, cosine: 0.1 }),
      ),
    ).toBe(true);
  });

  it("without a verdict, thresholds passScore against PASS_THRESHOLD", () => {
    expect(gradedPass(stub({ score: PASS_THRESHOLD + 0.01 }))).toBe(true);
    expect(gradedPass(stub({ score: PASS_THRESHOLD - 0.01 }))).toBe(false);
  });
});

describe("gradedBar", () => {
  it("verdicts render as fixed bar lengths — a boolean has no confidence to show", () => {
    expect(gradedBar(stub({ verdict: { relevant: true, why: "" } }))).toBe(0.9);
    expect(gradedBar(stub({ verdict: { relevant: false, why: "" } }))).toBe(0.12);
  });

  it("without a verdict, shows passScore with a visible-sliver floor", () => {
    expect(gradedBar(stub({ score: 0.6 }))).toBe(0.6);
    // hard zero (cosine below floor) still renders as a sliver, not nothing
    expect(gradedBar(stub({ relevance: 0.7, cosine: 0.3 }))).toBe(0.05);
  });
});
