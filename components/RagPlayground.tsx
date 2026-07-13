"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { usePlayground } from "@/hooks/usePlayground";
import { ACCENTS, NAMES, QUERY_CAPTIONS, indexCaptions } from "@/lib/constants";
import CanvasStage from "./CanvasStage";
import DocumentSidebar from "./DocumentSidebar";
import OutputPanel from "./OutputPanel";
import QueryBar from "./QueryBar";
import TopNav from "./TopNav";

export const LEFT_DEFAULT = 250;
export const RIGHT_DEFAULT = 360;
const LEFT_RAIL = 62;
const LEFT_MAX = 400;
const LEFT_SNAP = 100; // dragging below this snaps to the collapsed rail
const RIGHT_MIN = 280;
const RIGHT_MAX = 560;

const clamp = (v: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(hi, v));

export default function RagPlayground() {
  const { state, lastQuery, renderer, actions } = usePlayground();
  const accent = ACCENTS[state.rag];

  // --- resizable sidebars ---
  const [leftW, setLeftW] = useState(LEFT_DEFAULT);
  const [rightW, setRightW] = useState(RIGHT_DEFAULT);
  const [dragging, setDragging] = useState<"left" | "right" | null>(null);
  const widthsRef = useRef({ leftW: LEFT_DEFAULT, rightW: RIGHT_DEFAULT });
  widthsRef.current = { leftW, rightW };

  useEffect(() => {
    const l = parseInt(localStorage.getItem("rg.leftW") || "", 10);
    const r = parseInt(localStorage.getItem("rg.rightW") || "", 10);
    if (!Number.isNaN(l)) setLeftW(clamp(l, LEFT_SNAP, LEFT_MAX));
    if (!Number.isNaN(r)) setRightW(clamp(r, RIGHT_MIN, RIGHT_MAX));
  }, []);

  const persist = useCallback(() => {
    localStorage.setItem("rg.leftW", String(widthsRef.current.leftW));
    localStorage.setItem("rg.rightW", String(widthsRef.current.rightW));
  }, []);

  const startDrag = useCallback(
    (side: "left" | "right") => (e: React.PointerEvent) => {
      e.preventDefault();
      const startX = e.clientX;
      const startLeft = state.expanded ? widthsRef.current.leftW : LEFT_RAIL;
      const startRight = widthsRef.current.rightW;
      let expanded = state.expanded;
      setDragging(side);
      document.body.style.userSelect = "none";
      document.body.style.cursor = "col-resize";
      const move = (ev: PointerEvent) => {
        const dx = ev.clientX - startX;
        if (side === "left") {
          const w = clamp(startLeft + dx, LEFT_RAIL, LEFT_MAX);
          const wantExpanded = w >= LEFT_SNAP;
          if (wantExpanded !== expanded) {
            expanded = wantExpanded;
            actions.setExpanded(wantExpanded);
          }
          if (wantExpanded) setLeftW(w);
        } else {
          setRightW(clamp(startRight - dx, RIGHT_MIN, RIGHT_MAX));
        }
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        document.body.style.userSelect = "";
        document.body.style.cursor = "";
        setDragging(null);
        persist();
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    },
    [actions, persist, state.expanded],
  );

  const resetSide = useCallback(
    (side: "left" | "right") => () => {
      if (side === "left") {
        setLeftW(LEFT_DEFAULT);
        actions.setExpanded(true);
        localStorage.setItem("rg.leftW", String(LEFT_DEFAULT));
      } else {
        setRightW(RIGHT_DEFAULT);
        localStorage.setItem("rg.rightW", String(RIGHT_DEFAULT));
      }
    },
    [actions],
  );

  // --- captions ---
  const chunkCount = state.doc ? state.doc.chunks.length : 64;
  const pageCount = state.doc?.pages ?? 15;
  const captionMap: Partial<Record<typeof state.phase, string>> = {
    indexing:
      indexCaptions(state.rag, pageCount, chunkCount)[state.idxStage] ?? "",
    querying: QUERY_CAPTIONS[state.rag],
    answered: "TRACE · " + NAMES[state.rag] + " · " + lastQuery,
  };
  const caption = captionMap[state.phase];

  return (
    <div
      className={"rgt" + (state.dark ? "" : " light")}
      style={{
        position: "relative",
        width: "100vw",
        height: "100vh",
        background: "var(--bg)",
        backgroundImage:
          "linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px)",
        backgroundSize: "26px 26px",
        overflow: "hidden",
        display: "grid",
        gridTemplateColumns: "auto 1fr auto",
        gridTemplateRows: "58px 1fr 96px",
        color: "var(--ink)",
      }}
    >
      <TopNav state={state} accent={accent} onTab={actions.selectTab} />
      <DocumentSidebar
        state={state}
        accent={accent}
        actions={actions}
        width={state.expanded ? leftW : LEFT_RAIL}
        entityCount={renderer.view.scene.gnodes.length}
        dragging={dragging === "left"}
        onHandleDown={startDrag("left")}
        onHandleReset={resetSide("left")}
      />
      <CanvasStage
        state={state}
        accent={accent}
        renderer={renderer}
        caption={caption}
        actions={actions}
      />
      <QueryBar state={state} accent={accent} actions={actions} />
      <OutputPanel
        state={state}
        accent={accent}
        width={rightW}
        dragging={dragging === "right"}
        onHandleDown={startDrag("right")}
        onHandleReset={resetSide("right")}
      />
    </div>
  );
}
