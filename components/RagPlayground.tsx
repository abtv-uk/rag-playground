"use client";

import { usePlayground } from "@/hooks/usePlayground";
import { ACCENTS, NAMES, QUERY_CAPTIONS, indexCaptions } from "@/lib/constants";
import CanvasStage from "./CanvasStage";
import DocumentSidebar from "./DocumentSidebar";
import OutputPanel from "./OutputPanel";
import QueryBar from "./QueryBar";
import TopNav from "./TopNav";

export default function RagPlayground() {
  const { state, lastQuery, renderer, actions } = usePlayground();
  const accent = ACCENTS[state.rag];

  const captionMap: Partial<Record<typeof state.phase, string>> = {
    indexing: indexCaptions(state.rag)[state.idxStage] ?? "",
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
        gridTemplateColumns: "auto 1fr 360px",
        gridTemplateRows: "58px 1fr 96px",
        color: "var(--ink)",
      }}
    >
      <TopNav state={state} accent={accent} onTab={actions.selectTab} />
      <DocumentSidebar state={state} accent={accent} actions={actions} />
      <CanvasStage
        state={state}
        accent={accent}
        renderer={renderer}
        caption={caption}
        onLoadSample={actions.loadSample}
      />
      <QueryBar state={state} accent={accent} actions={actions} />
      <OutputPanel state={state} accent={accent} />
    </div>
  );
}
