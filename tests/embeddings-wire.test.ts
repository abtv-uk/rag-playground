// The int8 vectors.bin wire format, tested against the committed sample
// file. The norm assertion is the important one: dequantization is
// `q * scale` — the /127 is baked into scale at encode time. Dividing
// again (the obvious mistake, made once in a throwaway verification
// script) yields norms around 0.007 instead of ~1, silently destroying
// retrieval quality rather than crashing. This pins it permanently.
import { describe, expect, it } from "vitest";
import { encodeVectorBin } from "../lib/embeddings";
import { dense, vectorsBuf } from "./fixtures";

describe("vectors.bin wire format", () => {
  it("decodes the committed sample to 400 x 768", () => {
    expect(dense.dim).toBe(768);
    expect(dense.count).toBe(400);
    expect(dense.vectors.length).toBe(400 * 768);
  });

  it("header arithmetic accounts for every byte", () => {
    const view = new DataView(vectorsBuf);
    const headerLen = view.getUint32(0, true);
    expect(4 + headerLen + dense.count * dense.dim).toBe(vectorsBuf.byteLength);
  });

  it("every decoded vector is unit-norm (dequant is q*scale, not q*scale/127)", () => {
    for (let i = 0; i < dense.count; i++) {
      let n = 0;
      for (let j = 0; j < dense.dim; j++) {
        const v = dense.vectors[i * dense.dim + j];
        n += v * v;
      }
      const norm = Math.sqrt(n);
      expect(norm).toBeGreaterThan(0.95);
      expect(norm).toBeLessThan(1.05);
    }
  });

  it("encode(decode parts) round-trips byte-identically", () => {
    const view = new DataView(vectorsBuf);
    const headerLen = view.getUint32(0, true);
    const header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(vectorsBuf, 4, headerLen)),
    ) as { dim: number; count: number; scales: number[] };
    const int8 = new Int8Array(vectorsBuf, 4 + headerLen);
    const encoded = encodeVectorBin(header.dim, header.scales, int8);
    expect(encoded.byteLength).toBe(vectorsBuf.byteLength);
    expect(Buffer.compare(Buffer.from(encoded), Buffer.from(vectorsBuf))).toBe(0);
  });
});
