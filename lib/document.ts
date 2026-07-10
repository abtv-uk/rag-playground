// Client-side document ingestion: file parsing (PDF via pdfjs, TXT/MD via
// text()), URL scraping through a CORS-friendly reader proxy, and chunking
// with page attribution.

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
  isSample: boolean;
}

const MAX_CHARS = 250_000;
const WORDS_PER_PAGE = 500; // pseudo-pages for pageless sources (txt/md/url)
const WORDS_PER_CHUNK = 120;

export const SAMPLE_DOC: LoadedDoc = {
  name: "attention-is-all-you-need.pdf",
  sizeLabel: "312 KB · arXiv:1706.03762",
  pages: 15,
  chunks: [],
  isSample: true,
};

export function formatSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function paginate(text: string): string[] {
  const words = text.slice(0, MAX_CHARS).split(/\s+/).filter(Boolean);
  const pages: string[] = [];
  for (let i = 0; i < words.length; i += WORDS_PER_PAGE) {
    pages.push(words.slice(i, i + WORDS_PER_PAGE).join(" "));
  }
  return pages.length ? pages : [""];
}

export function chunkPages(pageTexts: string[]): DocChunk[] {
  const chunks: DocChunk[] = [];
  let budget = MAX_CHARS;
  pageTexts.forEach((pt, pi) => {
    if (budget <= 0) return;
    const text = pt.slice(0, budget);
    budget -= text.length;
    const words = text.split(/\s+/).filter(Boolean);
    for (let i = 0; i < words.length; i += WORDS_PER_CHUNK) {
      const slice = words.slice(i, i + WORDS_PER_CHUNK).join(" ");
      if (slice.trim().length < 20) continue;
      chunks.push({ id: chunks.length + 1, page: pi + 1, text: slice });
    }
  });
  return chunks;
}

async function extractPdfPages(data: ArrayBuffer): Promise<string[]> {
  const pdfjs = await import("pdfjs-dist");
  pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    "pdfjs-dist/build/pdf.worker.min.mjs",
    import.meta.url,
  ).toString();
  const doc = await pdfjs.getDocument({ data }).promise;
  const pages: string[] = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i);
    const tc = await page.getTextContent();
    pages.push(
      tc.items
        .map((it) => ("str" in it ? it.str : ""))
        .join(" ")
        .replace(/\s+/g, " "),
    );
  }
  return pages;
}

export async function parseFile(file: File): Promise<LoadedDoc> {
  const isPdf =
    file.type === "application/pdf" || /\.pdf$/i.test(file.name);
  const isText =
    /\.(txt|md|markdown)$/i.test(file.name) || file.type.startsWith("text/");
  if (!isPdf && !isText)
    throw new Error("unsupported file type — PDF, TXT or MD only");

  const pageTexts = isPdf
    ? await extractPdfPages(await file.arrayBuffer())
    : paginate(await file.text());
  const chunks = chunkPages(pageTexts);
  if (!chunks.length) throw new Error("no readable text found in document");
  return {
    name: file.name,
    sizeLabel: formatSize(file.size) + (isPdf ? " · PDF" : " · text"),
    pages: pageTexts.length,
    chunks,
    isSample: false,
  };
}

export async function fetchUrl(rawUrl: string): Promise<LoadedDoc> {
  let url = rawUrl.trim();
  if (!/^https?:\/\//i.test(url)) url = "https://" + url;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error("that doesn't look like a valid URL");
  }
  // r.jina.ai renders the page and returns readable markdown with permissive CORS
  const res = await fetch("https://r.jina.ai/" + url);
  if (!res.ok) throw new Error("fetch failed — HTTP " + res.status);
  const text = await res.text();
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
    isSample: false,
  };
}
