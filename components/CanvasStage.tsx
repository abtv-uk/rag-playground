"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ACCENTS, REJECT } from "@/lib/constants";
import type { PlaygroundActions } from "@/hooks/usePlayground";
import type { PipelineRenderer } from "@/lib/renderer";
import type { PlaygroundState } from "@/lib/types";

const MONO = "'JetBrains Mono',monospace";

export default function CanvasStage({
  state,
  accent,
  renderer,
  caption,
  actions,
}: {
  state: PlaygroundState;
  accent: string;
  renderer: PipelineRenderer;
  caption: string | undefined;
  actions: PlaygroundActions;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const centerRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [url, setUrl] = useState("");

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
  const loading = state.loading;

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (loading) return;
    const file = e.dataTransfer.files?.[0];
    if (file) {
      actions.loadFile(file);
      return;
    }
    const text =
      e.dataTransfer.getData("text/uri-list") ||
      e.dataTransfer.getData("text/plain");
    const dropped = (text || "").trim().split("\n")[0];
    if (/^https?:\/\/\S+$/i.test(dropped)) actions.loadUrl(dropped);
  };

  const onUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const u = url.trim();
    if (u && !loading) actions.loadUrl(u);
  };

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
              // frost the idle pipeline so the focus is on loading a document
              background:
                "color-mix(in srgb, var(--bg) 55%, transparent)",
              backdropFilter: "blur(7px)",
              WebkitBackdropFilter: "blur(7px)",
            }}
          >
            <div style={{ width: 380, textAlign: "center" }}>
              <input
                ref={fileRef}
                type="file"
                accept=".pdf,.txt,.md,.markdown"
                style={{ display: "none" }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) actions.loadFile(f);
                  e.target.value = "";
                }}
              />
              <div
                role="button"
                aria-label="Upload a document"
                onClick={() => !loading && fileRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                style={{
                  width: "100%",
                  height: 196,
                  border:
                    "1.5px dashed " + (dragOver ? accent : "var(--hair)"),
                  borderRadius: 16,
                  background: "var(--side)",
                  backdropFilter: "blur(2px)",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 14,
                  cursor: loading ? "default" : "pointer",
                  transition: "border-color .15s",
                }}
              >
                {loading ? (
                  <>
                    <span
                      style={{
                        width: 9,
                        height: 9,
                        borderRadius: "50%",
                        background: accent,
                        animation: "rgpulse 1s ease-in-out infinite",
                      }}
                    />
                    <div
                      style={{
                        fontFamily: MONO,
                        fontSize: 11.5,
                        letterSpacing: "0.06em",
                        color: "var(--ink)",
                      }}
                    >
                      reading document…
                    </div>
                  </>
                ) : (
                  <>
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
                    <div
                      style={{ fontSize: 15, fontWeight: 600, color: "var(--ink)" }}
                    >
                      Drop a document to begin
                    </div>
                    <div
                      style={{
                        fontFamily: MONO,
                        fontSize: 10.5,
                        color: "var(--sub2)",
                      }}
                    >
                      PDF · TXT · MD &nbsp;·&nbsp; click to browse
                    </div>
                  </>
                )}
              </div>

              <form
                onSubmit={onUrlSubmit}
                style={{
                  marginTop: 12,
                  display: "flex",
                  alignItems: "center",
                  gap: 7,
                  height: 36,
                  borderRadius: 10,
                  background: "var(--surface)",
                  border: "1px solid var(--hair)",
                  padding: "0 5px 0 12px",
                }}
              >
                <span
                  style={{
                    fontFamily: MONO,
                    fontSize: 8.5,
                    letterSpacing: "0.12em",
                    fontWeight: 600,
                    color: "var(--sub2)",
                    flex: "none",
                  }}
                >
                  URL
                </span>
                <span
                  style={{
                    width: 1,
                    height: 16,
                    background: "var(--hair)",
                    flex: "none",
                  }}
                />
                <input
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  disabled={loading}
                  placeholder="or paste a URL — https://…"
                  style={{
                    flex: 1,
                    border: "none",
                    background: "none",
                    fontFamily: "'Space Grotesk',sans-serif",
                    fontSize: 12.5,
                    color: "var(--ink)",
                    height: "100%",
                    minWidth: 0,
                  }}
                />
                <button
                  type="submit"
                  disabled={loading || !url.trim()}
                  style={{
                    height: 26,
                    padding: "0 12px",
                    borderRadius: 7,
                    background:
                      loading || !url.trim() ? "var(--chip)" : "var(--btnBg)",
                    color:
                      loading || !url.trim() ? "var(--faint)" : "var(--btnTx)",
                    fontSize: 11.5,
                    fontWeight: 600,
                    flex: "none",
                  }}
                >
                  Fetch
                </button>
              </form>

              {state.loadError && (
                <div
                  style={{
                    fontFamily: MONO,
                    fontSize: 10.5,
                    color: REJECT,
                    marginTop: 10,
                    lineHeight: 1.5,
                  }}
                >
                  ✕ {state.loadError}
                </div>
              )}

              <button
                onClick={actions.loadSample}
                disabled={loading}
                style={{
                  marginTop: 14,
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
                  opacity: loading ? 0.5 : 1,
                }}
              >
                <span
                  style={{
                    width: 7,
                    height: 7,
                    background: accent,
                    borderRadius: "50%",
                  }}
                />
                Load sample document
              </button>
              <div
                style={{
                  fontFamily: MONO,
                  fontSize: 10,
                  color: "var(--faint)",
                  marginTop: 10,
                }}
              >
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
