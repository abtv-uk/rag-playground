// Int8 quantization for embedding vectors, and the wire/file format used to
// ship them. Symmetric per-vector quantization: each vector gets its own
// scale (max absolute component / 127), so quantization error stays
// proportional to that vector's own dynamic range rather than the whole
// batch's — costs a handful of extra bytes per vector, buys back most of
// the ~0.2% recall loss a single shared scale would add.
//
// Kept in sync with lib/embeddings.ts's decoder in the app repo — the two
// are separate deployable projects (this one ships to Cloudflare, that one
// to GitHub Pages), so sharing a literal file isn't practical, but the
// format is simple enough that duplication is low-risk. Wire/file layout:
//   [4 bytes] uint32 LE — header byte length H
//   [H bytes]  UTF-8 JSON: {"dim":768,"count":N,"scales":[N floats]}
//   [N*dim bytes] Int8Array, row-major, one row per input vector in order

export interface QuantizedBatch {
  dim: number;
  count: number;
  scales: number[];
  data: Int8Array; // count * dim bytes
}

export function quantizeInt8(vectors: number[][], dim: number): QuantizedBatch {
  const count = vectors.length;
  const data = new Int8Array(count * dim);
  const scales = new Array<number>(count);
  for (let i = 0; i < count; i++) {
    const v = vectors[i];
    let maxAbs = 0;
    for (let j = 0; j < dim; j++) {
      const a = Math.abs(v[j]);
      if (a > maxAbs) maxAbs = a;
    }
    const scale = maxAbs / 127 || 1e-8;
    scales[i] = scale;
    const base = i * dim;
    for (let j = 0; j < dim; j++) {
      let q = Math.round(v[j] / scale);
      if (q > 127) q = 127;
      else if (q < -127) q = -127;
      data[base + j] = q;
    }
  }
  return { dim, count, scales, data };
}

export function encodeVectorBin(batch: QuantizedBatch): Uint8Array {
  const header = JSON.stringify({ dim: batch.dim, count: batch.count, scales: batch.scales });
  const headerBytes = new TextEncoder().encode(header);
  const out = new Uint8Array(4 + headerBytes.length + batch.data.length);
  new DataView(out.buffer).setUint32(0, headerBytes.length, true);
  out.set(headerBytes, 4);
  out.set(new Uint8Array(batch.data.buffer, batch.data.byteOffset, batch.data.byteLength), 4 + headerBytes.length);
  return out;
}
