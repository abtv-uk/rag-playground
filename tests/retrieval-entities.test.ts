// Entity matching in hybrid retrieval, on the real sample graph.
//
// Two regressions this guards. First, extraction quality: the graph's top
// entities were once dominated by citation noise (Retrieved, Wikimedia
// Commons, an author's surname), so most queries matched nothing. Second,
// the matcher: a bare prefix rule made the query "trade secret" light up
// the entity "Trademark Office" ("trademark".startsWith("trade")). The
// graphMatched field these tests read is also what gates hybrid's RELATED
// CONCEPTS prompt line — a wrong match here steers generation, not just a
// trace card.
import { describe, expect, it } from "vitest";
import { extractEntityGraph, retrieveHybrid } from "../lib/retrieval";
import { chunks } from "./fixtures";

const graph = extractEntityGraph(chunks);

function matchedLabels(query: string): string[] {
  const res = retrieveHybrid(chunks, query, graph);
  return (res.graphMatched ?? []).map(
    (i) => graph.nodes[i].full || graph.nodes[i].label,
  );
}

describe("entity graph extraction", () => {
  it("extracts a non-empty graph with subject-matter entities", () => {
    expect(graph.nodes.length).toBeGreaterThan(0);
    const labels = graph.nodes.map((n) => (n.full || n.label).toLowerCase());
    expect(labels.some((l) => /trade secret/.test(l))).toBe(true);
    expect(labels.some((l) => /intellectual property/.test(l))).toBe(true);
  });
});

describe("query-to-entity matching", () => {
  it('"what is a trade secret" matches Trade Secret, never a trademark entity', () => {
    const labels = matchedLabels("what is a trade secret").map((l) => l.toLowerCase());
    expect(labels.some((l) => /trade secret/.test(l))).toBe(true);
    expect(labels.some((l) => /trademark/.test(l))).toBe(false);
  });

  it('"what is intellectual property" matches Intellectual Property', () => {
    const labels = matchedLabels("what is intellectual property").map((l) =>
      l.toLowerCase(),
    );
    expect(labels.some((l) => /intellectual property/.test(l))).toBe(true);
  });
});
