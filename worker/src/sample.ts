// The sample document's chunks, bundled into the Worker itself.
//
// This is the trust boundary that makes the dual-provider split safe. The
// client already holds the sample's full text locally (it needs it for
// local lexical retrieval), so if the Gemini route trusted client-supplied
// text for "this is the sample" requests, anyone could spend Gemini quota
// generating over arbitrary text by lying about which document they're
// asking about. Resolving ids against this bundled copy instead means the
// only way to reach Gemini is to ask about content this file actually
// contains — see resolvePassages in index.ts for the enforcement.
//
// Regenerate with `npm run sync:sample` whenever
// `npm run preprocess:sample` (in the app) changes the source chunks.
import sampleChunksData from "./data/sample-chunks.json";

interface SampleChunk {
  id: number;
  page: number;
  text: string;
}

const { chunks } = sampleChunksData as { chunks: SampleChunk[] };

const byId = new Map<number, SampleChunk>(chunks.map((c) => [c.id, c]));

export function resolveSampleChunk(id: number): SampleChunk | undefined {
  return byId.get(id);
}
