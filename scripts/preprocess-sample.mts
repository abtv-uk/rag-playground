// Precompute the sample document's chunks — and their embeddings — so
// "Load sample document" is an instant fetch instead of a ~14s client-side
// PDF parse plus per-visitor embedding cost. Reuses the app's own
// chunkPages (same boilerplate filtering and sampling) so the precomputed
// chunks are byte-identical to what a live parse of the same file would
// produce.
//
// Embeddings go through the *deployed* generation Worker's own /embed
// route — not Workers AI's REST API directly — because that route already
// holds valid Workers AI credentials via its binding, and reusing it means
// this script needs no separate auth path or token scope of its own; a
// bare `fetch` with no Origin header is exactly what the Worker's origin
// check already allows through for server-to-server calls (see
// worker/src/index.ts). This does draw a small amount (~390 neurons for
// the full document) from the SAME production quota real traffic uses —
// negligible against the 8,000/day ceiling, and only spent when you
// actually re-run this script.
//
// Run after replacing the sample PDF: npm run preprocess:sample
// (Node ≥22.6 required — the script imports the app's TypeScript directly.)

import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { encodeVectorBin } from "../lib/embeddings.ts";
import { chunkPages, formatSize } from "../lib/document.ts";

const WORKER_ENDPOINT =
  process.env.NEXT_PUBLIC_LLM_ENDPOINT ||
  "https://rag-playground-worker.abdulbosit-melikuziev.workers.dev";
const EMBED_BATCH_SIZE = 100;

const root = fileURLToPath(new URL("..", import.meta.url));
const PDF = root + "public/sample/introduction-intellectual-property.pdf";
const OUT = root + "public/sample/introduction-intellectual-property.chunks.json";
const VECTORS_OUT = root + "public/sample/introduction-intellectual-property.vectors.bin";

const data = new Uint8Array(await readFile(PDF));
// pdfjs transfers (detaches) the buffer, zeroing byteLength — capture it now
const pdfBytes = data.byteLength;

// legacy build runs in Node without a DOM or worker
const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
const doc = await pdfjs.getDocument({ data }).promise;

const pages: string[] = [];
for (let i = 1; i <= doc.numPages; i++) {
  const page = await doc.getPage(i);
  const tc = await page.getTextContent();
  pages.push(
    tc.items
      .map((it: { str?: string }) => ("str" in it ? it.str : ""))
      .join(" ")
      .replace(/\s+/g, " "),
  );
  if (i % 25 === 0 || i === doc.numPages)
    console.log(`  parsed page ${i}/${doc.numPages}`);
}

const chunks = chunkPages(pages);
const out = {
  name: "introduction-intellectual-property.pdf",
  sizeLabel: formatSize(pdfBytes) + " · PDF · OpenStax, CC BY 4.0",
  pages: pages.length,
  chunks,
};

await writeFile(OUT, JSON.stringify(out));
console.log(
  `wrote ${OUT.split("/").pop()} — ${chunks.length} chunks from ${pages.length} pages (${formatSize(JSON.stringify(out).length)})`,
);

// ---------- embeddings, via the deployed Worker's /embed route ----------

const texts = chunks.map((c) => c.text);
let dim = 0;
const scales: number[] = [];
const dataParts: Uint8Array[] = [];

for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
  const batch = texts.slice(i, i + EMBED_BATCH_SIZE);
  const res = await fetch(WORKER_ENDPOINT + "/embed", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts: batch }),
  });
  if (!res.ok) {
    throw new Error(
      `embed batch failed — HTTP ${res.status} (chunks ${i + 1}-${i + batch.length})`,
    );
  }
  const buf = await res.arrayBuffer();
  const view = new DataView(buf);
  const headerLen = view.getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)),
  ) as { dim: number; scales: number[] };
  dim = header.dim;
  scales.push(...header.scales);
  // raw int8 bytes, unmodified — decoding to float and requantizing would
  // add a second, avoidable rounding step
  dataParts.push(new Uint8Array(buf, 4 + headerLen));
  console.log(`  embedded ${Math.min(i + EMBED_BATCH_SIZE, texts.length)}/${texts.length} chunks`);
}

const totalBytes = dataParts.reduce((n, p) => n + p.length, 0);
const vectorData = new Int8Array(totalBytes);
{
  let offset = 0;
  for (const p of dataParts) {
    vectorData.set(new Int8Array(p.buffer, p.byteOffset, p.byteLength), offset);
    offset += p.length;
  }
}

const vectorsBytes = encodeVectorBin(dim, scales, vectorData);
await writeFile(VECTORS_OUT, vectorsBytes);
console.log(
  `wrote ${VECTORS_OUT.split("/").pop()} — ${scales.length} × ${dim}-d int8 vectors (${formatSize(vectorsBytes.length)})`,
);
