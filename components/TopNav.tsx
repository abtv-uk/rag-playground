"use client";

import type { CSSProperties } from "react";
import { ACCENTS, TABS } from "@/lib/constants";
import { rgba } from "@/lib/color";
import type { PlaygroundState, RagId } from "@/lib/types";

const MONO = "'JetBrains Mono',monospace";

export default function TopNav({
  state,
  accent,
  onTab,
}: {
  state: PlaygroundState;
  accent: string;
  onTab: (id: RagId) => void;
}) {
  const D = state.dark;
  const tabBg = D ? "#222a35" : "#fff";
  const tabSh = D ? "0 1px 4px rgba(0,0,0,.4)" : "0 1px 4px rgba(17,21,27,.12)";
  const tabOn = D ? "#e9ecf1" : "#11151b";
  const tabOff = D ? "#838c9a" : "#8a92a0";

  const statusMap: Record<PlaygroundState["phase"], [string, string]> = {
    empty: [D ? "#5f6875" : "#c2c8d0", "no document"],
    indexing: [ACCENTS.agentic, "indexing…"],
    ready: [accent, "ready"],
    querying: [accent, "retrieving…"],
    answered: [ACCENTS.corrective, "answered"],
  };
  const [stColor, stLabel] = statusMap[state.phase];

  return (
    <div
      style={{
        gridColumn: "1 / 4",
        gridRow: 1,
        display: "flex",
        alignItems: "center",
        gap: 18,
        padding: "0 20px",
        borderBottom: "1px solid var(--hair)",
        background: "var(--bar)",
        backdropFilter: "blur(10px)",
        zIndex: 5,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 188 }}>
        <div
          style={{
            width: 22,
            height: 22,
            borderRadius: 6,
            background: "var(--ink)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <div
            style={{
              width: 8,
              height: 8,
              background: accent,
              transform: "rotate(45deg)",
              borderRadius: 1,
            }}
          />
        </div>
        <div
          style={{
            fontFamily: MONO,
            fontSize: 12,
            letterSpacing: "0.14em",
            fontWeight: 600,
            color: "var(--ink)",
          }}
        >
          RAG&nbsp;PLAYGROUND
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "stretch",
          gap: 4,
          background: "var(--chip)",
          padding: 4,
          borderRadius: 11,
          border: "1px solid var(--hair2)",
        }}
      >
        {TABS.map((t) => {
          const on = t.id === state.rag;
          const c = ACCENTS[t.id];
          const style: CSSProperties = {
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 13px",
            borderRadius: 8,
            background: on ? tabBg : "transparent",
            boxShadow: on ? tabSh : "none",
            color: on ? tabOn : tabOff,
            transition: "all .18s",
          };
          return (
            <button key={t.id} onClick={() => onTab(t.id)} style={style}>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: c,
                  boxShadow: on ? "0 0 0 3px " + rgba(c, 0.22) : "none",
                  flex: "none",
                }}
              />
              <span
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "flex-start",
                  lineHeight: 1.05,
                }}
              >
                <span style={{ fontSize: 12.5, fontWeight: 600, letterSpacing: "-0.01em" }}>
                  {t.label}
                </span>
                <span
                  style={{
                    fontFamily: MONO,
                    fontSize: 8.5,
                    letterSpacing: "0.08em",
                    opacity: 0.6,
                  }}
                >
                  {t.sub}
                </span>
              </span>
            </button>
          );
        })}
      </div>

      <div
        style={{
          marginLeft: "auto",
          display: "flex",
          alignItems: "center",
          gap: 14,
          fontFamily: MONO,
          fontSize: 10.5,
          letterSpacing: "0.06em",
          color: "var(--sub)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: stColor,
              flex: "none",
            }}
          />
          {stLabel}
        </div>
      </div>
    </div>
  );
}
