// SSE plumbing shared by both providers. The Worker defines its own wire
// format (WireEvent) rather than passing either provider's raw stream
// through — Gemini and Workers AI use two different chunk shapes (see
// providers.ts), and neither should leak into the client.

export type WireEvent =
  | { type: "delta"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

const encoder = new TextEncoder();

export function frame(event: WireEvent): Uint8Array {
  return encoder.encode(`data: ${JSON.stringify(event)}\n\n`);
}

/** Yields the raw payload after `data:` for each SSE line of an upstream
 *  stream, buffering partial lines across chunk boundaries. Works for both
 *  providers' upstream responses — only the payload shape differs. */
export async function* readUpstreamSseData(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).replace(/\r$/, "");
        buf = buf.slice(idx + 1);
        if (line.startsWith("data:")) yield line.slice(5).trim();
      }
    }
    const rest = buf.trim();
    if (rest.startsWith("data:")) yield rest.slice(5).trim();
  } finally {
    reader.releaseLock();
  }
}

export function sseHeaders(origin: string | null): HeadersInit {
  return {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
    ...(origin ? { "Access-Control-Allow-Origin": origin, Vary: "Origin" } : {}),
  };
}
