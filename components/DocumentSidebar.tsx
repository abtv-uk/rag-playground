"use client";

import { motion } from "framer-motion";
import { BLURBS, NAMES } from "@/lib/constants";
import { rgba } from "@/lib/color";
import type { PlaygroundActions } from "@/hooks/usePlayground";
import type { PlaygroundState } from "@/lib/types";

const MONO = "'JetBrains Mono',monospace";

export default function DocumentSidebar({
  state,
  accent,
  actions,
  width,
  entityCount,
  dragging,
  onHandleDown,
  onHandleReset,
}: {
  state: PlaygroundState;
  accent: string;
  actions: PlaygroundActions;
  width: number;
  entityCount: number;
  dragging: boolean;
  onHandleDown: (e: React.PointerEvent) => void;
  onHandleReset: () => void;
}) {
  const hasDoc = state.phase !== "empty";
  const showDocPanel = state.expanded && hasDoc;
  const doc = state.doc;
  const pages = doc?.pages ?? 15;
  const chunkCount = doc ? doc.chunks.length : 0;
  const pageThumbs = Array.from(
    { length: Math.max(1, Math.min(pages, 8)) },
    (_, i) => String(i + 1).padStart(2, "0"),
  );
  const isPdf = !doc || /\.pdf$/i.test(doc.name);

  return (
    <motion.div
      initial={false}
      animate={{ width }}
      transition={dragging ? { duration: 0 } : { duration: 0.25, ease: "easeInOut" }}
      style={{
        position: "relative",
        gridColumn: 1,
        gridRow: "2 / 4",
        borderRight: "1px solid var(--hair)",
        background: "var(--side)",
        backdropFilter: "blur(10px)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        zIndex: 4,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "16px 14px 12px",
        }}
      >
        {state.expanded && (
          <span
            style={{
              fontFamily: MONO,
              fontSize: 10.5,
              letterSpacing: "0.14em",
              color: "var(--sub2)",
              fontWeight: 600,
            }}
          >
            DOCUMENT
          </span>
        )}
        <button
          onClick={actions.toggleCollapse}
          style={{
            width: 26,
            height: 26,
            borderRadius: 7,
            border: "1px solid var(--hair)",
            background: "var(--surface)",
            color: "var(--sub)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 13,
          }}
        >
          {state.expanded ? "‹" : "›"}
        </button>
      </div>

      {showDocPanel && (
        <div
          style={{
            padding: "0 14px",
            display: "flex",
            flexDirection: "column",
            gap: 14,
            overflowY: "auto",
            width: "100%",
            minWidth: 222,
          }}
        >
          <div
            style={{
              border: "1px solid var(--hair)",
              borderRadius: 12,
              background: "var(--surface)",
              overflow: "hidden",
            }}
          >
            <div style={{ display: "flex", gap: 11, padding: 13 }}>
              <div
                style={{
                  width: 42,
                  height: 54,
                  borderRadius: 5,
                  background: "var(--surface2)",
                  border: "1px solid var(--hair)",
                  position: "relative",
                  flex: "none",
                  overflow: "hidden",
                }}
              >
                <div style={{ position: "absolute", top: 6, left: 5, right: 5, height: 2, background: "var(--hair)" }} />
                <div style={{ position: "absolute", top: 11, left: 5, right: 9, height: 2, background: "var(--hair2)" }} />
                <div style={{ position: "absolute", top: 16, left: 5, right: 6, height: 2, background: "var(--hair2)" }} />
                <div style={{ position: "absolute", top: 21, left: 5, right: 12, height: 2, background: "var(--hair2)" }} />
                <div
                  style={{
                    position: "absolute",
                    bottom: 5,
                    right: 5,
                    fontFamily: MONO,
                    fontSize: 6,
                    color: "#fff",
                    background: accent,
                    padding: "1px 3px",
                    borderRadius: 2,
                  }}
                >
                  {isPdf ? "PDF" : "TXT"}
                </div>
              </div>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div
                  style={{
                    fontFamily: MONO,
                    fontSize: 11.5,
                    fontWeight: 600,
                    color: "var(--ink)",
                    lineHeight: 1.25,
                    wordBreak: "break-all",
                  }}
                >
                  {doc?.sourceUrl ? (
                    <a
                      href={doc.sourceUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="open the original PDF"
                      style={{ color: "var(--ink)", textDecoration: "underline" }}
                    >
                      {doc.name}
                    </a>
                  ) : (
                    (doc?.name ?? "")
                  )}
                </div>
                <div style={{ fontFamily: MONO, fontSize: 10, color: "var(--sub2)", marginTop: 5 }}>
                  {doc?.sizeLabel ?? ""}
                </div>
              </div>
            </div>
            <div style={{ display: "flex", borderTop: "1px solid var(--hair2)" }}>
              <div style={{ flex: 1, padding: "9px 12px", borderRight: "1px solid var(--hair2)" }}>
                <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: "0.1em", color: "var(--faint)" }}>
                  CHUNKS
                </div>
                <div style={{ fontFamily: MONO, fontSize: 15, fontWeight: 600, color: "var(--ink)", marginTop: 2 }}>
                  {chunkCount}
                </div>
              </div>
              <div style={{ flex: 1, padding: "9px 12px" }}>
                <div style={{ fontFamily: MONO, fontSize: 9, letterSpacing: "0.1em", color: "var(--faint)" }}>
                  ENTITIES
                </div>
                <div style={{ fontFamily: MONO, fontSize: 15, fontWeight: 600, color: accent, marginTop: 2 }}>
                  {state.rag === "hybrid" && entityCount > 0 ? entityCount : "—"}
                </div>
              </div>
            </div>
          </div>

          <div>
            <div
              style={{
                fontFamily: MONO,
                fontSize: 9.5,
                letterSpacing: "0.12em",
                color: "var(--faint)",
                marginBottom: 8,
              }}
            >
              PAGES · {pages}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 7 }}>
              {pageThumbs.map((p) => (
                <div
                  key={p}
                  style={{
                    aspectRatio: "0.78",
                    borderRadius: 4,
                    background: "var(--surface)",
                    border: "1px solid var(--hair)",
                    position: "relative",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      position: "absolute",
                      top: 4,
                      left: 3,
                      right: 3,
                      bottom: 3,
                      backgroundImage:
                        "repeating-linear-gradient(180deg,var(--hair2) 0,var(--hair2) 1px,transparent 1px,transparent 4px)",
                    }}
                  />
                  <div
                    style={{
                      position: "absolute",
                      bottom: 1,
                      right: 2,
                      fontFamily: MONO,
                      fontSize: 6,
                      color: "var(--faint)",
                    }}
                  >
                    {p}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display: "flex", gap: 7, marginTop: 2 }}>
            <button
              onClick={actions.reindex}
              style={{
                flex: 1,
                height: 34,
                borderRadius: 8,
                border: "1px solid var(--hair)",
                background: "var(--surface)",
                fontSize: 11.5,
                fontWeight: 600,
                color: "var(--ink)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 6,
              }}
            >
              ↻ Re-index
            </button>
            <button
              onClick={actions.clear}
              style={{
                width: 38,
                height: 34,
                borderRadius: 8,
                border: "1px solid var(--hair)",
                background: "var(--surface)",
                fontSize: 13,
                color: "#d2655e",
              }}
            >
              ✕
            </button>
          </div>

          <div
            style={{
              marginTop: 4,
              padding: "10px 11px",
              borderRadius: 9,
              background: "var(--chip)",
              border: "1px solid var(--hair2)",
            }}
          >
            <div
              style={{
                fontFamily: MONO,
                fontSize: 9,
                letterSpacing: "0.1em",
                color: "var(--faint)",
                marginBottom: 6,
              }}
            >
              ACTIVE ARCHITECTURE
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                style={{
                  width: 9,
                  height: 9,
                  borderRadius: "50%",
                  background: accent,
                  boxShadow: "0 0 0 3px " + rgba(accent, 0.2),
                }}
              />
              <span style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>
                {NAMES[state.rag]}
              </span>
            </div>
            <div style={{ fontSize: 11, color: "var(--sub)", lineHeight: 1.4, marginTop: 7 }}>
              {BLURBS[state.rag]}
            </div>
          </div>
        </div>
      )}

      {/* theme toggle (always visible, pinned bottom) */}
      <div style={{ marginTop: "auto", padding: "12px 14px", borderTop: "1px solid var(--hair)" }}>
        <button
          onClick={actions.toggleTheme}
          style={{
            width: "100%",
            height: 34,
            borderRadius: 8,
            border: "1px solid var(--hair)",
            background: "var(--surface)",
            color: "var(--ink)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 8,
            fontSize: 11.5,
            fontWeight: 600,
          }}
        >
          <span style={{ fontSize: 13 }}>{state.dark ? "☀" : "☾"}</span>
          {state.expanded && <span>{state.dark ? "Light mode" : "Dark mode"}</span>}
        </button>
      </div>

      <div
        className={"rg-handle" + (dragging ? " active" : "")}
        style={{ right: 0 }}
        onPointerDown={onHandleDown}
        onDoubleClick={onHandleReset}
        title="drag to resize · double-click to reset"
      />
    </motion.div>
  );
}
