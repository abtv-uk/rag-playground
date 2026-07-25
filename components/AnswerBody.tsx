"use client";

// Renders a generated answer as prose with clickable inline citations.
//
// The generator is instructed to cite passages as [2] or [1][4] and to emit
// no headings, lists or other markdown — so the entire output grammar is
// "text runs plus bracketed integers", which a regex split covers. That's
// why there's no markdown dependency here. `pre-wrap` preserves the
// paragraph breaks a real model emits (the previous plain-text node
// collapsed them).

import type { Source } from "@/lib/types";

const CITE = /\[(\d+)\]/g;

interface Props {
  text: string;
  sources: Source[];
  accent: string;
  streaming: boolean;
  onCite: (chunkId: number) => void;
}

export default function AnswerBody({
  text,
  sources,
  accent,
  streaming,
  onCite,
}: Props) {
  // citation number → the trace card it points at
  const byCitation = new Map<number, Source>();
  sources.forEach((s, i) => {
    byCitation.set(s.citation ?? i + 1, s);
  });

  const parts: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  CITE.lastIndex = 0;
  while ((m = CITE.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const n = Number(m[1]);
    const src = byCitation.get(n);
    // a citation with no matching card is left as literal text rather than
    // rendered as a dead control
    parts.push(
      src?.chunkId != null ? (
        <sup key={`c${m.index}`} style={{ lineHeight: 0 }}>
          <button
            type="button"
            onClick={() => onCite(src.chunkId as number)}
            title={src.label + " · " + src.meta}
            style={{
              font: 'inherit',
              fontSize: 9.5,
              fontWeight: 600,
              color: accent,
              background: "transparent",
              // longhands only — mixing `border` with `borderBottom` makes
              // React warn on every streamed re-render
              borderWidth: "0 0 1px 0",
              borderStyle: "dotted",
              borderColor: accent,
              padding: "0 1px",
              margin: "0 1px",
              cursor: "pointer",
            }}
          >
            {n}
          </button>
        </sup>
      ) : (
        m[0]
      ),
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));

  return (
    <div
      style={{
        fontSize: 14,
        lineHeight: 1.62,
        color: "var(--ink2)",
        whiteSpace: "pre-wrap",
      }}
    >
      {parts}
      {streaming && (
        <span
          style={{
            display: "inline-block",
            width: 7,
            height: 15,
            background: accent,
            marginLeft: 2,
            transform: "translateY(2px)",
            animation: "rgblink .8s step-end infinite",
          }}
        />
      )}
    </div>
  );
}
