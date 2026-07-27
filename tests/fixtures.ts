// Shared fixtures: the bundled sample's committed chunks + embeddings.
// Loaded and decoded ONCE at module scope — every oracle reads the same
// deterministic inputs, which is the whole point of testing against the
// shipped sample rather than synthetic data.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { decodeVectorBin, type DenseIndex } from "../lib/embeddings";
import type { DocChunk } from "../lib/document";

const root = fileURLToPath(new URL("..", import.meta.url));
const BASE = root + "public/sample/introduction-intellectual-property";

export const chunks: DocChunk[] = JSON.parse(
  readFileSync(BASE + ".chunks.json", "utf8"),
).chunks;

const raw = readFileSync(BASE + ".vectors.bin");
// readFileSync returns a Buffer whose backing ArrayBuffer can be larger
// than the file (Node pools small buffers) — a naive `raw.buffer` would
// hand decodeVectorBin trailing garbage. Slice to the exact byte range.
export const vectorsBuf: ArrayBuffer = raw.buffer.slice(
  raw.byteOffset,
  raw.byteOffset + raw.byteLength,
) as ArrayBuffer;

export const dense: DenseIndex = decodeVectorBin(vectorsBuf);

/** Row `i` of the decoded matrix as its own Float32Array view. */
export function row(i: number): Float32Array {
  return dense.vectors.subarray(i * dense.dim, (i + 1) * dense.dim);
}

/** Chunk text by 1-based chunk id, lowercased — for content classifiers. */
const textById = new Map<number, string>(
  chunks.map((c) => [c.id, c.text.toLowerCase()]),
);
export function textOf(id: number): string {
  return textById.get(id) ?? "";
}
