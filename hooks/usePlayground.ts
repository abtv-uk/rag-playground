"use client";

// State machine for the playground: empty → indexing → ready → querying →
// answered. Mirrors state into the canvas renderer's mutable view so the rAF
// loop reads fresh values without re-rendering React each frame.
//
// Documents: the bundled sample keeps the scripted demo (canned answers,
// sources and scene); uploaded files / scraped URLs get real chunking,
// lexical retrieval and extractive answers.

import { useCallback, useEffect, useReducer, useRef } from "react";
import {
  ANSWERS,
  INDEX_MS,
  STEP_MS,
  STREAM_WORD_MS,
  SUGGESTIONS,
  buildSources,
} from "@/lib/constants";
import {
  SAMPLE_DOC,
  fetchUrl,
  parseFile,
  type LoadedDoc,
} from "@/lib/document";
import { PipelineRenderer } from "@/lib/renderer";
import {
  buildRealSources,
  generateSuggestions,
  retrieveAgentic,
  retrieveBasic,
  retrieveCorrective,
  retrieveHybrid,
} from "@/lib/retrieval";
import { applyQueryToScene, buildScene, sampleScene } from "@/lib/scene";
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
  doc: null,
  loading: false,
  loadError: "",
  suggestions: SUGGESTIONS,
};

export interface PlaygroundActions {
  loadSample: () => void;
  loadFile: (file: File) => void;
  loadUrl: (url: string) => void;
  reindex: () => void;
  clear: () => void;
  toggleTheme: () => void;
  toggleCollapse: () => void;
  setExpanded: (v: boolean) => void;
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
  const loadSeqRef = useRef(0);

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

  const adoptDoc = useCallback(
    (doc: LoadedDoc) => {
      renderer.view.scene = doc.isSample ? sampleScene() : buildScene(doc);
      renderer.resetGraphInteraction();
      const generated = doc.isSample ? [] : generateSuggestions(doc.chunks);
      set({
        doc,
        loading: false,
        loadError: "",
        suggestions: generated.length ? generated : SUGGESTIONS,
      });
      runIndex();
    },
    [renderer, runIndex, set],
  );

  const loadAsync = useCallback(
    (job: Promise<LoadedDoc>) => {
      const seq = ++loadSeqRef.current;
      set({ loading: true, loadError: "" });
      job.then(
        (doc) => {
          if (seq === loadSeqRef.current) adoptDoc(doc);
        },
        (err: unknown) => {
          if (seq !== loadSeqRef.current) return;
          const msg = err instanceof Error ? err.message : String(err);
          set({ loading: false, loadError: msg });
        },
      );
    },
    [adoptDoc, set],
  );

  const streamAnswer = useCallback(
    (text: string) => {
      const words = text.split(" ");
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
      const { rag, doc } = stateRef.current;
      const qsteps = steps(rag);
      renderer.view.querySteps = qsteps;
      renderer.view.queryStart = performance.now();

      let answer: string;
      let sources;
      if (doc && !doc.isSample) {
        const scene = renderer.view.scene;
        const res =
          rag === "hybrid"
            ? retrieveHybrid(doc.chunks, q, {
                nodes: scene.gnodes,
                neighbors: scene.gnbr,
              })
            : rag === "corrective"
              ? retrieveCorrective(doc.chunks, q)
              : rag === "agentic"
                ? retrieveAgentic(doc.chunks, q)
                : retrieveBasic(doc.chunks, q);
        applyQueryToScene(scene, res);
        answer = res.answer;
        sources = buildRealSources(rag, res, scene);
      } else {
        answer = ANSWERS[rag];
        sources = buildSources(rag);
      }

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
        set({ sources, sourcesVisible: true }),
      );
      after(STEP_MS * streamIdx, () => streamAnswer(answer));
    },
    [after, clearTimers, renderer, set, streamAnswer],
  );

  const actions: PlaygroundActions = {
    loadSample: () => adoptDoc(SAMPLE_DOC),
    loadFile: (file) => loadAsync(parseFile(file)),
    loadUrl: (url) => loadAsync(fetchUrl(url)),
    reindex: runIndex,
    clear: () => {
      clearTimers();
      loadSeqRef.current++;
      lastQueryRef.current = "";
      renderer.view.scene = sampleScene();
      renderer.resetGraphInteraction();
      set({
        query: "",
        answer: "",
        sources: [],
        sourcesVisible: false,
        streaming: false,
        doc: null,
        loading: false,
        loadError: "",
        suggestions: SUGGESTIONS,
        phase: "empty",
      });
    },
    toggleTheme: () => set({ dark: !stateRef.current.dark }),
    toggleCollapse: () => set({ expanded: !stateRef.current.expanded }),
    setExpanded: (v) => {
      if (v !== stateRef.current.expanded) set({ expanded: v });
    },
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
