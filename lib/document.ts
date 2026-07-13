// Client-side document ingestion: file parsing (PDF via pdfjs, TXT/MD via
// text()), URL scraping through a CORS-friendly reader proxy, and chunking
// with page attribution. Chunking filters out front matter (tables of
// contents, indexes, page-number runs, bibliographies) and samples across the
// whole document so late chapters are represented, not just the opening pages.

export interface DocChunk {
  id: number;
  page: number;
  text: string;
}

export interface LoadedDoc {
  name: string;
  sizeLabel: string;
  pages: number;
  chunks: DocChunk[];
  /** where the original document can be viewed (sample PDF) */
  sourceUrl?: string;
}

/** Progress hint shown in the dropzone while a document is being read. */
export type ProgressFn = (message: string) => void;

const WORDS_PER_PAGE = 500; // pseudo-pages for pageless sources (txt/md/url)
const WORDS_PER_CHUNK = 120;
const MAX_CHUNKS = 400; // retained chunks after whole-document sampling
const RAW_CHUNK_CEILING = 6000; // safety bound on pre-sampling work
const MAX_TEXT_CHARS = 4_000_000; // guard for pageless (txt/url) sources

const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";
export const SAMPLE_PDF_URL =
  BASE + "/sample/introduction-intellectual-property.pdf";
const SAMPLE_CHUNKS_URL =
  BASE + "/sample/introduction-intellectual-property.chunks.json";

/** The bundled sample (an OpenStax open-licensed textbook) ships with
 *  precomputed chunks — loading it is a small JSON fetch instead of a
 *  multi-second client-side parse of the 201-page PDF. Regenerate the JSON
 *  with `npm run preprocess:sample` if the PDF changes. */
export async function loadSampleDoc(
  onProgress?: ProgressFn,
): Promise<LoadedDoc> {
  onProgress?.("loading preprocessed sample…");
  const res = await fetch(SAMPLE_CHUNKS_URL);
  if (!res.ok) throw new Error("sample data missing — HTTP " + res.status);
  const data = await res.json();
  return {
    name: data.name,
    sizeLabel: data.sizeLabel,
    pages: data.pages,
    chunks: data.chunks,
    sourceUrl: SAMPLE_PDF_URL,
  };
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// Front-matter / boilerplate detection. Structural signals (dot leaders,
// runs of "Heading … 86" entries) dominate — those are unambiguous tables of
// contents / indexes. Numeric density only counts when the passage also lacks
// sentence structure, so genuine prose with statistics or citations survives.
export function boilerplateScore(text: string): number {
  const words = text.split(/\s+/).filter(Boolean);
  const n = words.length || 1;
  const digitRatio = (text.match(/\d/g) || []).length / (text.length || 1);
  const numericTokens = words.filter((w) =>
    /^\d+$/.test(w.replace(/[.,):;]/g, "")),
  ).length;
  const dotLeaders = /\.{4,}|(?:\.\s){4,}/.test(text);
  // "Heading Words 86" — a title run ending in a small page number (TOC/index)
  const tocEntries = (
    text.match(/[A-Za-z][A-Za-z ,'&/-]{4,}\s+\d{1,4}(?=\s|$)/g) || []
  ).length;
  const sentenceEnds = (text.match(/[.!?](?:\s|$)/g) || []).length;

  // multiple-choice / review blocks: "a. … b. … c. …" runs, or quiz prompts
  const optionMarkers = (text.match(/(?:^|\s)[a-d][.)]\s/g) || []).length;
  const quizPrompt =
    /which of the following|true or false|assessment question|review question|check your understanding|select all that apply/i.test(
      text,
    );

  let score = 0;
  if (dotLeaders) score += 2; // TOC dot leaders — unambiguous
  if (tocEntries >= 4) score += 2; // repeated "title … page-number" entries
  if (optionMarkers >= 3) score += 2; // multiple-choice option block
  if (quizPrompt) score += 2;
  // number-soup with almost no sentences (data tables, page-number runs)
  const numberHeavy = digitRatio > 0.06 && numericTokens / n > 0.1;
  if (numberHeavy && sentenceEnds < n / 40) score += 2;
  return score;
}

function isBoilerplate(text: string): boolean {
  return boilerplateScore(text) >= 2;
}

/** Chunk every page, drop boilerplate, then sample evenly to MAX_CHUNKS so the
 *  whole document is represented rather than just the first N characters. */
export function chunkPages(pageTexts: string[]): DocChunk[] {
  const raw: { page: number; text: string }[] = [];
  outer: for (let pi = 0; pi < pageTexts.length; pi++) {
    const words = pageTexts[pi].split(/\s+/).filter(Boolean);
    for (let i = 0; i < words.length; i += WORDS_PER_CHUNK) {
      const slice = words.slice(i, i + WORDS_PER_CHUNK).join(" ");
      if (slice.trim().length < 40) continue;
      if (isBoilerplate(slice)) continue;
      raw.push({ page: pi + 1, text: slice });
      if (raw.length >= RAW_CHUNK_CEILING) break outer;
    }
  }
  let kept = raw;
  if (raw.length > MAX_CHUNKS) {
    const step = raw.length / MAX_CHUNKS;
    kept = [];
    for (let i = 0; i < MAX_CHUNKS; i++) kept.push(raw[Math.floor(i * step)]);
  }
  return kept.map((c, i) => ({ id: i + 1, page: c.page, text: c.text }));
}

function paginate(text: string): string[] {
  const words = text.slice(0, MAX_TEXT_CHARS).split(/\s+/).filter(Boolean);
  const pages: string[] = [];
  for (let i = 0; i < words.length; i += WORDS_PER_PAGE) {
    pages.push(words.slice(i, i + WORDS_PER_PAGE).join(" "));
  }
  return pages.length ? pages : [""];
}

async function extractPdfPages(
  data: ArrayBuffer,
  onProgress?: ProgressFn,
): Promise<string[]> {
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    "pdfjs-dist/build/pdf.worker.min.mjs",
    import.meta.url,
  ).toString();
  const doc = await pdfjs.getDocument({ data }).promise;
  const total = doc.numPages;
  const pages: string[] = [];
  for (let i = 1; i <= total; i++) {
    if (onProgress && (i === 1 || i % 5 === 0 || i === total))
      onProgress(`reading page ${i} / ${total}…`);
    try {
      const page = await doc.getPage(i);
      const tc = await page.getTextContent();
      pages.push(
        tc.items
          .map((it) => ("str" in it ? it.str : ""))
          .join(" ")
          .replace(/\s+/g, " "),
      );
    } catch {
      pages.push(""); // a single unreadable page shouldn't abort the whole parse
    }
  }
  return pages;
}

export async function parseFile(
  file: File,
  onProgress?: ProgressFn,
): Promise<LoadedDoc> {
  const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
  const isText =
    /\.(txt|md|markdown)$/i.test(file.name) || file.type.startsWith("text/");
  if (!isPdf && !isText)
    throw new Error("unsupported file type — PDF, TXT or MD only");

  onProgress?.("reading document…");
  const pageTexts = isPdf
    ? await extractPdfPages(await file.arrayBuffer(), onProgress)
    : paginate(await file.text());
  onProgress?.("chunking…");
  const chunks = chunkPages(pageTexts);
  if (!chunks.length) throw new Error("no readable text found in document");
  return {
    name: file.name,
    sizeLabel: formatSize(file.size) + (isPdf ? " · PDF" : " · text"),
    pages: pageTexts.length,
    chunks,
  };
}

export async function fetchUrl(
  rawUrl: string,
  onProgress?: ProgressFn,
): Promise<LoadedDoc> {
  let url = rawUrl.trim();
  if (!/^https?:\/\//i.test(url)) url = "https://" + url;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error("that doesn't look like a valid URL");
  }
  onProgress?.("fetching page…");
  // r.jina.ai renders the page and returns readable markdown with permissive CORS
  const res = await fetch("https://r.jina.ai/" + url);
  if (!res.ok) throw new Error("fetch failed — HTTP " + res.status);
  const text = await res.text();
  onProgress?.("chunking…");
  const pageTexts = paginate(text);
  const chunks = chunkPages(pageTexts);
  if (!chunks.length) throw new Error("no readable text at that URL");
  const name =
    (parsed.hostname + parsed.pathname).replace(/\/$/, "").slice(0, 60) ||
    parsed.hostname;
  return {
    name,
    sizeLabel: formatSize(text.length) + " · scraped",
    pages: pageTexts.length,
    chunks,
  };
}
