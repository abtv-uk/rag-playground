// Copies the app's precomputed sample chunks into the worker so it can
// resolve chunk ids -> {page, text} itself, without trusting client-supplied
// text (see src/sample.ts for why that boundary matters). Re-run whenever
// `npm run preprocess:sample` regenerates the source file.
import { copyFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../..", import.meta.url));
const SRC = root + "public/sample/introduction-intellectual-property.chunks.json";
const OUT = root + "worker/src/data/sample-chunks.json";

await copyFile(SRC, OUT);
console.log("synced sample chunks -> worker/src/data/sample-chunks.json");
