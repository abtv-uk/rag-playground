// The PCA/k-means projection behind the vector scatter, tested two ways:
// a synthetic matrix with KNOWN principal axes (correctness has a ground
// truth there), and the real sample embeddings (properties, not exact
// values — every real-fixture assertion is a range or inequality so a
// re-embedded sample re-validates itself instead of breaking).
import { describe, expect, it } from "vitest";
import { kmeans, pca2, projectPoint } from "../lib/projection";
import { dense, row } from "./fixtures";

function synthetic(): { data: Float32Array; n: number; dim: number } {
  const n = 200;
  const dim = 8;
  const data = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    const t = (i / n) * 2 - 1;
    data[i * dim + 0] = t * 10; // dominant axis
    data[i * dim + 1] = Math.sin(i) * 3; // second axis
    data[i * dim + 2] = Math.cos(i * 3) * 0.2; // noise
  }
  return { data, n, dim };
}

const dominantDim = (v: Float32Array): number => {
  let best = 0;
  for (let j = 1; j < v.length; j++) if (Math.abs(v[j]) > Math.abs(v[best])) best = j;
  return best;
};

describe("pca2 on a matrix with known axes", () => {
  const { data, n, dim } = synthetic();
  const p = pca2(data, n, dim);

  it("recovers the dominant axes in order", () => {
    expect(dominantDim(p.pc1)).toBe(0);
    expect(dominantDim(p.pc2)).toBe(1);
  });

  it("components are orthogonal", () => {
    let dot = 0;
    for (let j = 0; j < dim; j++) dot += p.pc1[j] * p.pc2[j];
    expect(Math.abs(dot)).toBeLessThan(1e-6);
  });

  it("explained variance is ordered and positive", () => {
    expect(p.explained[0]).toBeGreaterThan(p.explained[1]);
    expect(p.explained[1]).toBeGreaterThan(0);
  });
});

describe("pca2 on the real sample embeddings", () => {
  const p = pca2(dense.vectors, dense.count, dense.dim);

  it("two components explain a small, honest share of 768-d variance", () => {
    const total = p.explained[0] + p.explained[1];
    expect(total).toBeGreaterThan(0.08);
    expect(total).toBeLessThan(0.16);
  });

  it("projectPoint reproduces a row's own component scores (basis round-trip)", () => {
    const [x, y] = projectPoint(p, row(0));
    expect(Math.abs(x - p.xy[0])).toBeLessThan(1e-4);
    expect(Math.abs(y - p.xy[1])).toBeLessThan(1e-4);
  });

  it("is deterministic — no Math.random anywhere in the pipeline", () => {
    const again = pca2(dense.vectors, dense.count, dense.dim);
    expect(again.xy).toEqual(p.xy);
    expect(again.pc1).toEqual(p.pc1);
  });
});

describe("kmeans on the real sample embeddings", () => {
  const labels = kmeans(dense.vectors, dense.count, dense.dim, 5);

  it("labels every vector and uses all five clusters", () => {
    expect(labels.length).toBe(dense.count);
    expect(new Set(labels).size).toBe(5);
  });

  it("clusters are semantically coherent (intra-cluster cosine beats overall)", () => {
    const cos = (a: number, b: number): number => {
      let d = 0;
      for (let j = 0; j < dense.dim; j++)
        d += dense.vectors[a * dense.dim + j] * dense.vectors[b * dense.dim + j];
      return d; // vectors are unit-norm (see embeddings-wire tests)
    };
    let intra = 0;
    let intraN = 0;
    let all = 0;
    let allN = 0;
    // strided pair sample — full pairwise is 80k pairs, unnecessary
    for (let i = 0; i < dense.count; i += 3) {
      for (let j = i + 1; j < dense.count; j += 7) {
        const c = cos(i, j);
        all += c;
        allN++;
        if (labels[i] === labels[j]) {
          intra += c;
          intraN++;
        }
      }
    }
    expect(intraN).toBeGreaterThan(0);
    expect(intra / intraN).toBeGreaterThan(all / allN);
  });

  it("is deterministic", () => {
    expect(kmeans(dense.vectors, dense.count, dense.dim, 5)).toEqual(labels);
  });
});
