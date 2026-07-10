"use client";

// State machine for the playground: empty → indexing → ready → querying →
// answered. Mirrors state into the canvas renderer's mutable view so the rAF
// loop reads fresh values without re-rendering React each frame.

import { useCallback, useEffect, useReducer, useRef } from "react";
import {
  ANSWERS,
  INDEX_MS,
  STEP_MS,
  STREAM_WORD_MS,
  buildSources,
} from "@/lib/constants";
import { PipelineRenderer } from "@/lib/renderer";
import { steps } from "@/lib/steps";
import type { PlaygroundState, RagId } from "@/lib/types";

const INITIAL: PlaygroundState = {
  rag: "naive",
  phase: "empty",
  expanded: true,
  dark: true,
  query: "",
  answer: "",
  streaming: false,
  sources: [],
  sourcesVisible: false,
  idxStage: 0,
};

export interface PlaygroundActions {
  loadSample: () => void;
  reindex: () => void;
  clear: () => void;
  toggleTheme: () => void;
  toggleCollapse: () => void;
  setQuery: (q: string) => void;
  submit: () => void;
  selectTab: (id: RagId) => void;
  pickSuggestion: (t: string) => void;
}

export function usePlayground() {
  const stateRef = useRef<PlaygroundState>(INITIAL);
  const [, force] = useReducer((c: number) => c + 1, 0);
  const rendererRef = useRef<PipelineRenderer | null>(null);
  if (!rendererRef.current) rendererRef.current = new PipelineRenderer();
  const renderer = rendererRef.current;

  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const lastQueryRef = useRef("");

  const set = useCallback(
    (patch: Partial<PlaygroundState>) => {
      const next = { ...stateRef.current, ...patch };
      stateRef.current = next;
      const view = renderer.view;
      view.rag = next.rag;
      view.phase = next.phase;
      view.dark = next.dark;
      view.streaming = next.streaming;
      force();
    },
    [renderer],
  );

  const clearTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }, []);

  const after = useCallback((ms: number, fn: () => void) => {
    timersRef.current.push(setTimeout(fn, ms));
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => {
      renderer.view.reducedMotion = mq.matches;
    };
    apply();
    mq.addEventListener("change", apply);
    return () => {
      mq.removeEventListener("change", apply);
      clearTimers();
    };
  }, [renderer, clearTimers]);

  const runIndex = useCallback(() => {
    clearTimers();
    lastQueryRef.current = "";
    renderer.view.indexStart = performance.now();
    set({
      answer: "",
      sources: [],
      sourcesVisible: false,
      streaming: false,
      idxStage: 0,
      phase: "indexing",
    });
    after(1100, () => set({ idxStage: 1 }));
    after(2200, () => set({ idxStage: 2 }));
    after(3300, () => set({ idxStage: 3 }));
    after(INDEX_MS, () => set({ phase: "ready" }));
  }, [after, clearTimers, renderer, set]);

  const streamAnswer = useCallback(
    (rag: RagId) => {
      const words = ANSWERS[rag].split(" ");
      set({ answer: "", streaming: true });
      let i = 0;
      const tick = () => {
        i++;
        set({ answer: words.slice(0, i).join(" ") });
        if (i < words.length) after(STREAM_WORD_MS, tick);
        else set({ streaming: false, phase: "answered" });
      };
      after(40, tick);
    },
    [after, set],
  );

  const runQuery = useCallback(
    (q: string) => {
      clearTimers();
      lastQueryRef.current = q;
      const rag = stateRef.current.rag;
      const qsteps = steps(rag);
      renderer.view.querySteps = qsteps;
      renderer.view.queryStart = performance.now();
      set({
        query: q,
        answer: "",
        sources: [],
        sourcesVisible: false,
        streaming: false,
        phase: "querying",
      });
      const streamIdx = qsteps.findIndex((s) => s.stream);
      after(STEP_MS * (streamIdx - 0.4), () =>
        set({ sources: buildSources(rag), sourcesVisible: true }),
      );
      after(STEP_MS * streamIdx, () => streamAnswer(rag));
    },
    [after, clearTimers, renderer, set, streamAnswer],
  );

  const actions: PlaygroundActions = {
    loadSample: runIndex,
    reindex: runIndex,
    clear: () => {
      clearTimers();
      lastQueryRef.current = "";
      set({
        query: "",
        answer: "",
        sources: [],
        sourcesVisible: false,
        streaming: false,
        phase: "empty",
      });
    },
    toggleTheme: () => set({ dark: !stateRef.current.dark }),
    toggleCollapse: () => set({ expanded: !stateRef.current.expanded }),
    setQuery: (q) => set({ query: q }),
    submit: () => {
      const s = stateRef.current;
      const q = (s.query || "").trim();
      if (!q || s.phase === "empty" || s.phase === "indexing") return;
      runQuery(q);
    },
    selectTab: (id) => {
      set({ rag: id });
      renderer.resetGraphInteraction();
      const ph = stateRef.current.phase;
      if (
        lastQueryRef.current &&
        (ph === "answered" || ph === "ready" || ph === "querying")
      ) {
        runQuery(lastQueryRef.current);
      }
    },
    pickSuggestion: (t) => {
      set({ query: t });
      const ph = stateRef.current.phase;
      if (ph !== "empty" && ph !== "indexing") runQuery(t);
    },
  };

  return {
    state: stateRef.current,
    lastQuery: lastQueryRef.current,
    renderer,
    actions,
  };
}
