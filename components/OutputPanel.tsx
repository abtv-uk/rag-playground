"use client";

import { motion } from "framer-motion";
import { NAMES, REJECT } from "@/lib/constants";
import { rgba } from "@/lib/color";
import type { PlaygroundState } from "@/lib/types";

const MONO = "'JetBrains Mono',monospace";

export default function OutputPanel({
  state,
  accent,
}: {
  state: PlaygroundState;
  accent: string;
}) {
  const isEmpty = state.phase === "empty";
  const isIndexing = state.phase === "indexing";
  const isReady = state.phase === "ready";
  const isQuerying = state.phase === "querying";
  const isAnswered = state.phase === "answered";

  const showGenChip = isQuerying || (isAnswered && state.streaming);
  const outIdle =
    (isReady || isEmpty || isIndexing) && !state.sourcesVisible && !state.answer;
  const outRetrieving = isQuerying && !state.answer;
  const showSources = state.sourcesVisible && state.sources.length > 0;

  return (
    <div
      style={{
        gridColumn: 3,
        gridRow: "2 / 4",
        borderLeft: "1px solid var(--hair)",
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
          padding: "16px 18px 12px",
          borderBottom: "1px solid var(--hair2)",
        }}
      >
        <span
          style={{
            fontFamily: MONO,
            fontSize: 10.5,
            letterSpacing: "0.14em",
            color: "var(--sub2)",
            fontWeight: 600,
          }}
        >
          OUTPUT
        </span>
        {showGenChip && (
          <span
            style={{
              fontFamily: MONO,
              fontSize: 9.5,
              letterSpacing: "0.06em",
              color: accent,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: accent,
                animation: "rgblink 1s ease-in-out infinite",
              }}
            />
            {state.streaming ? "generating" : "retrieving"}
          </span>
        )}
      </div>

      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 18,
          display: "flex",
          flexDirection: "column",
          gap: 18,
        }}
      >
        {outIdle && (
          <div style={{ marginTop: 30, textAlign: "center", color: "var(--faint)" }}>
            <div
              style={{
                width: 46,
                height: 46,
                borderRadius: 12,
                border: "1px dashed var(--hair)",
                margin: "0 auto 14px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 18,
                color: "var(--faint)",
              }}
            >
              ⌁
            </div>
            <div
              style={{
                fontSize: 12.5,
                lineHeight: 1.5,
                color: "var(--sub2)",
                maxWidth: 200,
                margin: "0 auto",
              }}
            >
              {isEmpty
                ? "Load a document, then ask a question to watch the pipeline light up."
                : "Ask a question to run the " + NAMES[state.rag] + " pipeline."}
            </div>
          </div>
        )}

        {outRetrieving && (
          <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
            <div className="rg-shim" style={{ height: 11, borderRadius: 5, width: "84%" }} />
            <div className="rg-shim" style={{ height: 11, borderRadius: 5, width: "96%" }} />
            <div className="rg-shim" style={{ height: 11, borderRadius: 5, width: "70%" }} />
          </div>
        )}

        {!!state.answer && (
          <div>
            <div
              style={{
                fontFamily: MONO,
                fontSize: 9.5,
                letterSpacing: "0.12em",
                color: "var(--faint)",
                marginBottom: 9,
              }}
            >
              ANSWER
            </div>
            <div style={{ fontSize: 14, lineHeight: 1.62, color: "var(--ink2)" }}>
              {state.answer}
              {state.streaming && (
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
          </div>
        )}

        {showSources && (
          <div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 10,
              }}
            >
              <span
                style={{
                  fontFamily: MONO,
                  fontSize: 9.5,
                  letterSpacing: "0.12em",
                  color: "var(--faint)",
                }}
              >
                RETRIEVAL TRACE
              </span>
              <span style={{ fontFamily: MONO, fontSize: 9.5, color: "var(--faint)" }}>
                {state.sources.length} sources
              </span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {state.sources.map((src, i) => {
                const col = src.rejected ? REJECT : src.color;
                return (
                  <motion.div
                    key={src.label + i}
                    initial={{ opacity: 0, x: 14 }}
                    animate={{ opacity: src.rejected ? 0.62 : 1, x: 0 }}
                    transition={{ duration: 0.25, delay: i * 0.06 }}
                    style={{
                      border: "1px solid var(--hair)",
                      borderRadius: 9,
                      background: "var(--surface)",
                      padding: "9px 11px",
                      borderStyle: src.rejected ? "dashed" : "solid",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span
                        style={{
                          width: 9,
                          height: 9,
                          flex: "none",
                          borderRadius: src.kind === "node" ? "50%" : 2,
                          background: rgba(col, 0.18),
                          border: "1.5px solid " + col,
                        }}
                      />
                      <span
                        style={{
                          fontFamily: MONO,
                          fontSize: 11,
                          fontWeight: 600,
                          color: "var(--ink)",
                        }}
                      >
                        {src.label}
                      </span>
                      <span
                        style={{
                          marginLeft: "auto",
                          fontFamily: MONO,
                          fontSize: 9.5,
                          color: "var(--faint)",
                        }}
                      >
                        {src.meta}
                      </span>
                    </div>
                    <div
                      style={{
                        fontSize: 11.5,
                        color: "var(--sub)",
                        lineHeight: 1.4,
                        marginTop: 6,
                      }}
                    >
                      {src.snippet}
                    </div>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        marginTop: 8,
                      }}
                    >
                      <div
                        style={{
                          flex: 1,
                          height: 4,
                          borderRadius: 2,
                          background: "var(--hair2)",
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            height: "100%",
                            width: Math.round(src.scoreN * 100) + "%",
                            background: col,
                            borderRadius: 2,
                          }}
                        />
                      </div>
                      <span style={{ fontFamily: MONO, fontSize: 9.5, color: col }}>
                        {src.score}
                      </span>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
