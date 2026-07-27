// Lexical scoring's stemming contract, on the real sample document.
//
// History this pins down: scoreChunks once counted a hit whenever a chunk
// word merely started with a query term. Multi-term queries masked it (a
// second term's idf drowns the noise), but single-term queries showed the
// damage plainly — "trade" returned trademark-only chunks in 4 of its top
// 6, and "copy" returned copyright-only chunks in 6 of 6. sameWord's
// inflection rule fixed both at a measured recall cost of one bibliography
// line. These tests hold that line.
import { describe, expect, it } from "vitest";
import { retrieveBasic } from "../lib/retrieval";
import { chunks, textOf } from "./fixtures";

const top6 = (query: string): number[] =>
  retrieveBasic(chunks, query).top.map((s) => s.chunk.id);

const trademarkOnly = (t: string) =>
  /trademark/.test(t) && !/\btrades?\b/.test(t);
const copyrightOnly = (t: string) =>
  /copyright/.test(t) && !/\bcop(y|ies|ying)\b/.test(t);

describe("lexical stemming (single-term queries, where prefix leaks show)", () => {
  it('"trade" surfaces no trademark-only chunks (was 4/6 under bare startsWith)', () => {
    const bad = top6("trade").filter((id) => trademarkOnly(textOf(id)));
    expect(bad).toEqual([]);
  });

  it('"copy" surfaces no copyright-only chunks (was 6/6 under bare startsWith)', () => {
    const bad = top6("copy").filter((id) => copyrightOnly(textOf(id)));
    expect(bad).toEqual([]);
  });

  it('"patent" keeps inflection-family recall (patents/patenting still match)', () => {
    const ids = top6("patent");
    expect(ids.length).toBe(6);
    for (const id of ids) expect(textOf(id)).toMatch(/patent/);
    // at least one hit came through an inflected form, proving sameWord
    // still unifies the family rather than requiring exact matches
    expect(ids.some((id) => /patent(s|ing|ed|able)/.test(textOf(id)))).toBe(true);
  });
});

describe("lexical retrieval (multi-term sanity)", () => {
  it('"what is a trade secret" retrieves trade-secret chunks', () => {
    const ids = top6("what is a trade secret");
    expect(ids.some((id) => /trade secret/.test(textOf(id)))).toBe(true);
  });
});
