"use client";

import type { FormEvent } from "react";
import { SUGGESTIONS } from "@/lib/constants";
import type { PlaygroundActions } from "@/hooks/usePlayground";
import type { PlaygroundState } from "@/lib/types";

const MONO = "'JetBrains Mono',monospace";

export default function QueryBar({
  state,
  accent,
  actions,
}: {
  state: PlaygroundState;
  accent: string;
  actions: PlaygroundActions;
}) {
  const D = state.dark;
  const isEmpty = state.phase === "empty";
  const isIndexing = state.phase === "indexing";
  const isQuerying = state.phase === "querying";
  const inputDisabled = isEmpty || isIndexing;
  const sendDisabled = inputDisabled || !(state.query || "").trim();

  const send = D ? "#e9ecf1" : "#11151b";
  const sendTx = D ? "#0b0d11" : "#fff";
  const sendOff = D ? "rgba(255,255,255,0.08)" : "rgba(17,21,27,0.08)";
  const sendOffTx = D ? "#5f6875" : "#aab0bb";
  const border = D ? "rgba(255,255,255,0.12)" : "rgba(17,21,27,0.12)";

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    actions.submit();
  };

  return (
    <div
      style={{
        gridColumn: 2,
        gridRow: 3,
        borderTop: "1px solid var(--hair)",
        background: "var(--side)",
        backdropFilter: "blur(10px)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        padding: "0 26px",
        gap: 8,
        zIndex: 4,
        minWidth: 0,
      }}
    >
      <div
        style={{
          display: "flex",
          gap: 7,
          alignItems: "center",
          overflow: "hidden",
          minWidth: 0,
        }}
      >
        <span
          style={{
            fontFamily: MONO,
            fontSize: 9,
            letterSpacing: "0.12em",
            color: "var(--faint)",
            marginRight: 2,
          }}
        >
          TRY
        </span>
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => actions.pickSuggestion(s)}
            style={{
              height: 25,
              padding: "0 11px",
              borderRadius: 7,
              background: "var(--chip)",
              border: "1px solid var(--hair2)",
              fontSize: 11,
              color: "var(--sub)",
              whiteSpace: "nowrap",
            }}
          >
            {s}
          </button>
        ))}
      </div>
      <form
        onSubmit={onSubmit}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          height: 46,
          borderRadius: 13,
          background: "var(--surface)",
          border: "1px solid " + (isQuerying ? accent : border),
          padding: "0 7px 0 16px",
          transition: "border-color .2s",
        }}
      >
        <span
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: accent,
            flex: "none",
            animation: "rgpulse 2s ease-in-out infinite",
          }}
        />
        <span
          style={{
            fontFamily: MONO,
            fontSize: 9,
            letterSpacing: "0.14em",
            fontWeight: 600,
            color: accent,
            flex: "none",
          }}
        >
          QUERY
        </span>
        <span style={{ width: 1, height: 20, background: "var(--hair)", flex: "none" }} />
        <input
          value={state.query}
          onChange={(e) => actions.setQuery(e.target.value)}
          disabled={inputDisabled}
          placeholder={
            isEmpty ? "Load a document first…" : "Ask this document a question…"
          }
          style={{
            flex: 1,
            border: "none",
            background: "none",
            fontFamily: "'Space Grotesk',sans-serif",
            fontSize: 14.5,
            color: "var(--ink)",
            height: "100%",
          }}
        />
        <button
          type="submit"
          disabled={sendDisabled}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            height: 34,
            padding: "0 16px",
            borderRadius: 9,
            background: sendDisabled ? sendOff : send,
            color: sendDisabled ? sendOffTx : sendTx,
            transition: "all .18s",
          }}
        >
          <span style={{ fontSize: 13, fontWeight: 600 }}>Ask</span>
          <span style={{ fontSize: 14 }}>→</span>
        </button>
      </form>
    </div>
  );
}
