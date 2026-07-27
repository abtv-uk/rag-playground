// Turns the document's real 768-d embeddings into the 2-d scatter the
// pipeline draws, so the vector-store panel shows actual semantic structure
// instead of a decorative point cloud. Before this, dot positions came from
// `page % 5` plus a seeded PRNG — pretty, and completely unrelated to what
// the vectors mean.
//
// No dependency and no matrix library: two power-iteration passes over a
// 400x768 matrix is a few million multiply-adds, which runs in well under
// the indexing animation's budget. Everything here is deterministic — the
// same document must lay out identically on every load, so there is no
// Math.random anywhere, including k-means seeding.

export interface Pca2Basis {
  /** Column means, subtracted before projecting. */
  mean: Float32Array;
  pc1: Float32Array;
  pc2: Float32Array;
  /** Fraction of total variance each component captures, for honest
   *  reporting in the UI. Two components out of 768 legitimately explain a
   *  small share; the number is shown rather than hidden. */
  explained: [number, number];
}

export interface Pca2Result extends Pca2Basis {
  /** n * 2, row-major: component scores per input vector, in input order. */
  xy: Float32Array;
}

const POWER_ITERS = 24;

/** Deterministic unit-ish seed vector. Any fixed non-degenerate direction
 *  works; using a fixed irrational-ish pattern avoids the pathological case
 *  of a seed exactly orthogonal to the leading component. */
function seedVector(dim: number): Float32Array {
  const v = new Float32Array(dim);
  for (let j = 0; j < dim; j++) v[j] = Math.sin(j * 12.9898 + 1.0);
  return normalize(v);
}

function normalize(v: Float32Array): Float32Array {
  let n = 0;
  for (let j = 0; j < v.length; j++) n += v[j] * v[j];
  n = Math.sqrt(n) || 1e-9;
  for (let j = 0; j < v.length; j++) v[j] /= n;
  return v;
}

/** One power-iteration pass for the leading eigenvector of XᵀX, computed as
 *  v ← normalize(Xᵀ(Xv)) so the dim x dim covariance matrix is never formed
 *  (768x768 floats would dwarf the data itself). `deflate` removes an
 *  already-found component from each row on the fly. */
function leadingComponent(
  centered: Float32Array,
  n: number,
  dim: number,
  deflate: Float32Array | null,
): Float32Array {
  let v = seedVector(dim);
  const next = new Float32Array(dim);
  for (let it = 0; it < POWER_ITERS; it++) {
    next.fill(0);
    // deflate·v is row-independent, so it is computed once per iteration
    // rather than per row — the inner loops are already the hot path.
    let corr = 0;
    if (deflate) for (let j = 0; j < dim; j++) corr += deflate[j] * v[j];
    for (let i = 0; i < n; i++) {
      const base = i * dim;
      // scalar projections of row i onto v and onto the deflated direction
      let dot = 0;
      let dotD = 0;
      for (let j = 0; j < dim; j++) {
        const x = centered[base + j];
        dot += x * v[j];
        if (deflate) dotD += x * deflate[j];
      }
      // (x - (x·d)d)·v  =  x·v - (x·d)(d·v)
      if (deflate) dot -= dotD * corr;
      for (let j = 0; j < dim; j++) {
        const x = deflate ? centered[base + j] - dotD * deflate[j] : centered[base + j];
        next[j] += dot * x;
      }
    }
    v = normalize(next.slice());
  }
  return v;
}

/** Two-component PCA. Returns component scores plus the basis itself, so a
 *  vector that wasn't in the input — notably the query — can be projected
 *  into the same space later (see projectPoint). */
export function pca2(vectors: Float32Array, n: number, dim: number): Pca2Result {
  const mean = new Float32Array(dim);
  for (let i = 0; i < n; i++) {
    const base = i * dim;
    for (let j = 0; j < dim; j++) mean[j] += vectors[base + j];
  }
  for (let j = 0; j < dim; j++) mean[j] /= n || 1;

  const centered = new Float32Array(n * dim);
  let totalVar = 0;
  for (let i = 0; i < n; i++) {
    const base = i * dim;
    for (let j = 0; j < dim; j++) {
      const x = vectors[base + j] - mean[j];
      centered[base + j] = x;
      totalVar += x * x;
    }
  }

  const pc1 = leadingComponent(centered, n, dim, null);
  const pc2 = leadingComponent(centered, n, dim, pc1);

  const xy = new Float32Array(n * 2);
  let var1 = 0;
  let var2 = 0;
  for (let i = 0; i < n; i++) {
    const base = i * dim;
    let a = 0;
    let b = 0;
    for (let j = 0; j < dim; j++) {
      const x = centered[base + j];
      a += x * pc1[j];
      b += x * pc2[j];
    }
    xy[i * 2] = a;
    xy[i * 2 + 1] = b;
    var1 += a * a;
    var2 += b * b;
  }
  const denom = totalVar || 1;
  return { xy, mean, pc1, pc2, explained: [var1 / denom, var2 / denom] };
}

/** Projects a vector into an existing basis — used for the query marker, so
 *  it lands in the same coordinate frame as the document's dots rather than
 *  a separately-fitted one (which would be meaningless to compare). */
export function projectPoint(basis: Pca2Basis, v: Float32Array): [number, number] {
  const { mean, pc1, pc2 } = basis;
  let a = 0;
  let b = 0;
  for (let j = 0; j < mean.length; j++) {
    const x = v[j] - mean[j];
    a += x * pc1[j];
    b += x * pc2[j];
  }
  return [a, b];
}

/** k-means over the full-dimensional vectors (not the 2-d projection —
 *  clustering after discarding ~70% of the variance would group points that
 *  merely look close on screen). Deterministic: centroids are seeded by
 *  even stride through the input rather than at random, and the loop is
 *  bounded, so a document always yields the same labels. */
export function kmeans(
  vectors: Float32Array,
  n: number,
  dim: number,
  k = 5,
  iters = 12,
): Uint8Array {
  const labels = new Uint8Array(n);
  if (!n) return labels;
  const kk = Math.min(k, n);
  const centroids = new Float32Array(kk * dim);
  for (let c = 0; c < kk; c++) {
    const src = Math.floor((c * n) / kk) * dim;
    for (let j = 0; j < dim; j++) centroids[c * dim + j] = vectors[src + j];
  }

  const counts = new Int32Array(kk);
  for (let it = 0; it < iters; it++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      const base = i * dim;
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < kk; c++) {
        const cb = c * dim;
        let d = 0;
        for (let j = 0; j < dim; j++) {
          const t = vectors[base + j] - centroids[cb + j];
          d += t * t;
          if (d >= bestD) break; // early exit: most comparisons lose quickly
        }
        if (d < bestD) {
          bestD = d;
          best = c;
        }
      }
      if (labels[i] !== best) {
        labels[i] = best;
        moved = true;
      }
    }
    if (!moved && it > 0) break;
    centroids.fill(0);
    counts.fill(0);
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      counts[c]++;
      const base = i * dim;
      const cb = c * dim;
      for (let j = 0; j < dim; j++) centroids[cb + j] += vectors[base + j];
    }
    for (let c = 0; c < kk; c++) {
      if (!counts[c]) continue;
      const cb = c * dim;
      for (let j = 0; j < dim; j++) centroids[cb + j] /= counts[c];
    }
  }
  return labels;
}
