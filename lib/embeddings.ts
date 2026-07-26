// Semantic retrieval's client half: decode the Worker's int8 vectors.bin
// wire format (shared with the precomputed sample file — see
// scripts/preprocess-sample.mts and worker/src/quantize.ts, which this
// module's decoder is kept in sync with), embed uploaded-document chunks
// through the Worker, and rank by cosine.
//
// Never throws: embedTexts resolves null on any failure (Worker down,
// quota exhausted, network error), which is the graceful-degradation seam
// — retrieval.ts's dense-retrieval path is only taken when this succeeds,
// so any failure here silently lands on today's lexical-only behavior.

const ENDPOINT =
  process.env.NEXT_PUBLIC_LLM_ENDPOINT || "http://localhost:8787";

const EMBED_BATCH_SIZE = 100;
const EMBED_TIMEOUT_MS = 15000;

export interface DenseIndex {
  dim: number;
  vectors: Float32Array; // count * dim, row-major
  count: number;
}

// ---------- wire/file format: shared with worker/src/quantize.ts ----------
// [4 bytes] uint32 LE header length H
// [H bytes] UTF-8 JSON {"dim":768,"count":N,"scales":[N floats]}
// [N*dim bytes] Int8Array, row-major

export function decodeVectorBin(buf: ArrayBuffer): DenseIndex {
  const view = new DataView(buf);
  const headerLen = view.getUint32(0, true);
  const headerText = new TextDecoder().decode(new Uint8Array(buf, 4, headerLen));
  const header = JSON.parse(headerText) as { dim: number; count: number; scales: number[] };
  const { dim, count, scales } = header;
  const int8 = new Int8Array(buf, 4 + headerLen, count * dim);
  const vectors = new Float32Array(count * dim);
  for (let i = 0; i < count; i++) {
    const scale = scales[i];
    const base = i * dim;
    for (let j = 0; j < dim; j++) vectors[base + j] = int8[base + j] * scale;
  }
  return { dim, count, vectors };
}

/** Used only by the build-time precompute script, which stitches several
 *  /embed responses into one file. Quantization itself lives in the Worker
 *  (worker/src/quantize.ts) — the script forwards the int8 bytes it already
 *  returned rather than round-tripping through float and requantizing, so
 *  there is deliberately no encoder-side quantizer here. */
export function encodeVectorBin(dim: number, scales: number[], data: Int8Array): Uint8Array {
  const header = JSON.stringify({ dim, count: scales.length, scales });
  const headerBytes = new TextEncoder().encode(header);
  const out = new Uint8Array(4 + headerBytes.length + data.length);
  new DataView(out.buffer).setUint32(0, headerBytes.length, true);
  out.set(headerBytes, 4);
  out.set(data, 4 + headerBytes.length);
  return out;
}

// ---------- ranking ----------

/** Proper cosine, not a raw dot product — quantization can leave vectors
 *  fractionally off unit norm, and the query vector comes from a separate
 *  request than the document's, so never assume either is exactly
 *  normalized. */
export function cosineRank(
  idx: DenseIndex,
  q: Float32Array,
): { i: number; cos: number }[] {
  const { dim, count, vectors } = idx;
  let qNorm = 0;
  for (let j = 0; j < dim; j++) qNorm += q[j] * q[j];
  qNorm = Math.sqrt(qNorm) || 1e-8;

  const out: { i: number; cos: number }[] = [];
  for (let i = 0; i < count; i++) {
    const base = i * dim;
    let dot = 0;
    let vNorm = 0;
    for (let j = 0; j < dim; j++) {
      const v = vectors[base + j];
      dot += v * q[j];
      vNorm += v * v;
    }
    const denom = Math.sqrt(vNorm) * qNorm || 1e-8;
    out.push({ i, cos: dot / denom });
  }
  return out.sort((a, b) => b.cos - a.cos);
}

export function vectorAt(idx: DenseIndex, i: number): Float32Array {
  return idx.vectors.subarray(i * idx.dim, (i + 1) * idx.dim);
}

// ---------- embedding via the Worker ----------

async function embedBatch(
  texts: string[],
  signal: AbortSignal,
): Promise<DenseIndex | null> {
  try {
    const res = await fetch(ENDPOINT + "/embed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts }),
      signal,
    });
    if (!res.ok) return null;
    return decodeVectorBin(await res.arrayBuffer());
  } catch {
    return null;
  }
}

/** Embeds a list of texts in batches of 100, concatenating results in
 *  order. Returns null — never throws — the instant any batch fails, so a
 *  document never half-embeds under a mix of dense and lexical-only
 *  chunks. */
export async function embedTexts(
  texts: string[],
  onProgress?: (done: number, total: number) => void,
): Promise<DenseIndex | null> {
  if (!texts.length) return { dim: 0, count: 0, vectors: new Float32Array(0) };

  const batches: string[][] = [];
  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    batches.push(texts.slice(i, i + EMBED_BATCH_SIZE));
  }

  const timer = AbortSignal.timeout(EMBED_TIMEOUT_MS * batches.length);
  let dim = 0;
  let done = 0;
  const parts: Float32Array[] = [];
  for (const batch of batches) {
    const idx = await embedBatch(batch, timer);
    if (!idx) return null;
    // A short batch would leave the tail of the concatenated buffer as
    // zero rows silently masquerading as embedded chunks (the Worker caps
    // a request at MAX_EMBED_TEXTS, well above the batch size, so this
    // shouldn't happen) — fail the whole document instead, per the
    // never-half-embed contract above.
    if (idx.count !== batch.length) return null;
    dim = idx.dim;
    parts.push(idx.vectors);
    done += batch.length;
    onProgress?.(done, texts.length);
  }

  const vectors = new Float32Array(done * dim);
  let offset = 0;
  for (const p of parts) {
    vectors.set(p, offset);
    offset += p.length;
  }
  return { dim, count: done, vectors };
}

/** Convenience wrapper for embedding a single query string. */
export async function embedQuery(text: string): Promise<Float32Array | null> {
  const idx = await embedBatch([text], AbortSignal.timeout(EMBED_TIMEOUT_MS));
  return idx ? vectorAt(idx, 0) : null;
}
