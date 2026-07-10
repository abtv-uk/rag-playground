"use client";

import { useEffect, useRef } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ACCENTS } from "@/lib/constants";
import type { PipelineRenderer } from "@/lib/renderer";
import type { PlaygroundState } from "@/lib/types";

const MONO = "'JetBrains Mono',monospace";

export default function CanvasStage({
  state,
  accent,
  renderer,
  caption,
  onLoadSample,
}: {
  state: PlaygroundState;
  accent: string;
  renderer: PipelineRenderer;
  caption: string | undefined;
  onLoadSample: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const centerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const center = centerRef.current;
    if (!canvas || !center) return;
    let cancelled = false;
    const attach = () => {
      if (!cancelled) renderer.attach(canvas, center);
    };
    // wait (briefly) for webfonts so canvas text measures correctly
    if (document.fonts?.ready) {
      Promise.race([
        document.fonts.ready,
        new Promise((r) => setTimeout(r, 800)),
      ]).then(attach);
    } else attach();
    return () => {
      cancelled = true;
      renderer.detach();
    };
  }, [renderer]);

  const isEmpty = state.phase === "empty";
  const isIndexing = state.phase === "indexing";
  const isQuerying = state.phase === "querying";

  return (
    <div
      ref={centerRef}
      style={{
        gridColumn: 2,
        gridRow: 2,
        position: "relative",
        overflow: "hidden",
        minWidth: 0,
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      />

      <AnimatePresence>
        {isEmpty && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 3,
            }}
          >
            <div style={{ width: 380, textAlign: "center" }}>
              <div
                style={{
                  width: "100%",
                  height: 196,
                  border: "1.5px dashed var(--hair)",
                  borderRadius: 16,
                  background: "var(--side)",
                  backdropFilter: "blur(2px)",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 14,
                }}
              >
                <div
                  style={{
                    width: 52,
                    height: 52,
                    borderRadius: 13,
                    background: "var(--surface)",
                    border: "1px solid var(--hair)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <div
                    style={{
                      width: 20,
                      height: 24,
                      border: "2px solid var(--ink)",
                      borderRadius: 3,
                      position: "relative",
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        left: "50%",
                        top: "50%",
                        width: 2,
                        height: 11,
                        background: "var(--ink)",
                        transform: "translate(-50%,-60%)",
                      }}
                    />
                    <div
                      style={{
                        position: "absolute",
                        left: "50%",
                        top: "50%",
                        width: 8,
                        height: 2,
                        background: "var(--ink)",
                        transform: "translate(-50%,140%)",
                      }}
                    />
                  </div>
                </div>
                <div style={{ fontSize: 15, fontWeight: 600, color: "var(--ink)" }}>
                  Drop a document to begin
                </div>
                <div style={{ fontFamily: MONO, fontSize: 10.5, color: "var(--sub2)" }}>
                  PDF · TXT · MD &nbsp;·&nbsp; one document at a time
                </div>
              </div>
              <button
                onClick={onLoadSample}
                style={{
                  marginTop: 16,
                  height: 42,
                  padding: "0 22px",
                  borderRadius: 11,
                  background: "var(--btnBg)",
                  color: "var(--btnTx)",
                  fontSize: 13.5,
                  fontWeight: 600,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 9,
                }}
              >
                <span
                  style={{ width: 7, height: 7, background: accent, borderRadius: "50%" }}
                />
                Load sample document
              </button>
              <div style={{ fontFamily: MONO, fontSize: 10, color: "var(--faint)", marginTop: 10 }}>
                attention-is-all-you-need.pdf · 15 pp
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {caption && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6 }}
            transition={{ duration: 0.18 }}
            style={{
              position: "absolute",
              left: 18,
              bottom: 14,
              zIndex: 3,
              display: "flex",
              alignItems: "center",
              gap: 9,
              padding: "8px 13px",
              borderRadius: 10,
              background: "var(--bar)",
              backdropFilter: "blur(8px)",
              border: "1px solid var(--hair)",
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: isIndexing ? ACCENTS.agentic : accent,
                flex: "none",
                animation:
                  isIndexing || isQuerying
                    ? "rgpulse 1.6s ease-in-out infinite"
                    : "none",
              }}
            />
            <span
              style={{
                fontFamily: MONO,
                fontSize: 11,
                color: "var(--ink)",
                letterSpacing: "0.02em",
              }}
            >
              {caption}
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
