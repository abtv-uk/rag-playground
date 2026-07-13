// Precompute the sample document's chunks so "Load sample document" is an
// instant JSON fetch instead of a ~14s client-side parse of a 201-page PDF.
// Reuses the app's own chunkPages (same boilerplate filtering and sampling)
// so the precomputed chunks are byte-identical to what a live parse of the
// same file would produce.
//
// Run after replacing the sample PDF: npm run preprocess:sample
// (Node ≥22.6 required — the script imports the app's TypeScript directly.)

import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chunkPages, formatSize } from "../lib/document.ts";

const root = fileURLToPath(new URL("..", import.meta.url));
const PDF = root + "public/sample/introduction-intellectual-property.pdf";
const OUT = root + "public/sample/introduction-intellectual-property.chunks.json";

const data = new Uint8Array(await readFile(PDF));

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
  sizeLabel: formatSize(data.byteLength) + " · PDF · OpenStax, CC BY 4.0",
  pages: pages.length,
  chunks,
};

await writeFile(OUT, JSON.stringify(out));
console.log(
  `wrote ${OUT.split("/").pop()} — ${chunks.length} chunks from ${pages.length} pages (${formatSize(JSON.stringify(out).length)})`,
);
