// Structured logging for Workers Observability.
//
// `observability.enabled` was already on in wrangler.jsonc, but the Worker
// emitted nothing of its own — so the dashboard held platform invocation
// records and no way to answer the questions that actually come up: which
// provider served a request, why a fallback fired, how often the daily
// ladder downshifts, which route is slow.
//
// One JSON object per request, one `console.log` per line, because
// Observability indexes structured fields — a formatted string would have
// to be re-parsed to filter on.
//
// WHAT IS DELIBERATELY NOT LOGGED: passage and chunk text. Uploaded
// documents are private (the whole provider-routing boundary in index.ts
// exists to keep them away from a provider that trains on inputs), and
// writing them into logs would undo that from the other end. `query` IS
// recorded, truncated — it is what makes a bad answer diagnosable — but it
// is the only user-supplied content here, and it never carries document
// contents.

const MAX_LOGGED_QUERY_CHARS = 120;

export interface RequestLog {
  route: string;
  status: number;
  ms: number;
  /** Truncated user question. Absent on routes that receive none. */
  query?: string;
  /** Which upstream actually served it — the fallback chain means this is
   *  not predictable from the route alone. */
  provider?: "gemini" | "workers-ai" | "none";
  /** Daily-ladder position at decision time, so downshifts are visible
   *  before they become refusals. */
  tier?: string;
  /** Set when the response was an error or a degraded path was taken. */
  reason?: string;
  /** Passage/sub-query/verdict counts — shape, never content. */
  n?: number;
}

export function logRequest(entry: RequestLog): void {
  console.log(
    JSON.stringify({
      ...entry,
      query:
        entry.query === undefined
          ? undefined
          : entry.query.slice(0, MAX_LOGGED_QUERY_CHARS),
    }),
  );
}

/** Upstream failures worth alerting on, separated from ordinary request
 *  logs so they can be filtered without parsing every line. Takes the
 *  message only — an Error's stack from a provider SDK can embed request
 *  payloads, which is the content this module is careful to keep out. */
export function logUpstreamError(route: string, upstream: string, err: unknown): void {
  console.error(
    JSON.stringify({
      route,
      upstream,
      error: err instanceof Error ? err.message : String(err),
    }),
  );
}
