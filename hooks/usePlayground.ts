"use client";

// State machine for the playground: empty → indexing → ready → querying →
// answered. Mirrors state into the canvas renderer's mutable view so the rAF
// loop reads fresh values without re-rendering React each frame.
//
// All documents run the real pipeline (chunking, per-mode retrieval,
// extractive or LLM-backed answers). The bundled sample is an OpenStax
// textbook whose chunks ship precomputed, so loading it skips the PDF parse.

import { useCallback, useEffect, useReducer, useRef } from "react";
import {
  INDEX_MS,
  STEP_MS,
  STREAM_WORD_MS,
  SUGGESTIONS,
} from "@/lib/constants";
import {
  fetchUrl,
  loadSampleDoc,
  parseFile,
  type LoadedDoc,
} from "@/lib/document";
import { checkHealth, generateLlmAnswer } from "@/lib/llm";
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
  loadingMsg: "",
  loadError: "",
  suggestions: SUGGESTIONS,
  degraded: false,
  genPhase: "idle",
  generatorOffline: false,
};

// How long to wait for generated prose before giving up and showing the
// offline extractive answer instead. Workers AI's time-to-first-token is
// well under a second; Gemini (used for the sample) is both the slower
// path and, in testing, a highly variable one — seven real calls against
// gemini-3.5-flash for the same small prompt ranged 4.5-19.1s (network
// transit to Google measured separately at ~0.2s, so this is inference
// latency, not transit). No deadline short enough for good UX catches the
// true tail, so 12s is a deliberate compromise: it catches most draws
// while keeping the pathological case bounded. When it doesn't land in
// time, showing the extractive answer with the EXTRACTIVE FALLBACK label
// is the correct, designed-for outcome — not a bug to chase away.
const GEN_DEADLINE_MS = 12000;

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
  const querySeqRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const set = useCallback(
    (patch: Partial<PlaygroundState>) => {
      const next = { ...stateRef.current, ...patch };
      stateRef.current = next;
      const view = renderer.view;
      view.rag = next.rag;
      view.phase = next.phase;
      view.dark = next.dark;
      view.streaming = next.streaming;
      view.genPhase = next.genPhase;
      force();
    },
    [renderer],
  );

  // Aborting here (rather than only on the llm.ts timeout) is what stops a
  // superseded query — tab switch, new question, clear, unmount — from
  // leaving an orphan request in flight.
  const clearTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    abortRef.current?.abort();
    abortRef.current = null;
    renderer.view.holdStep = null;
  }, [renderer]);

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

  // Checked once at mount so a total generator outage shows up before the
  // user asks anything, rather than only being discovered 6s into every
  // query. A single exhausted provider isn't "offline" — the Worker itself
  // falls through from Gemini to Workers AI — so this only trips when
  // neither provider has capacity, or the Worker is unreachable at all.
  useEffect(() => {
    let cancelled = false;
    checkHealth().then((status) => {
      if (cancelled) return;
      set({
        generatorOffline: !status.ok || (!status.workersAiAvailable && !status.geminiAvailable),
      });
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      renderer.view.scene = buildScene(doc);
      renderer.resetGraphInteraction();
      const generated = generateSuggestions(doc.chunks, doc.name);
      set({
        doc,
        loading: false,
        loadingMsg: "",
        loadError: "",
        suggestions: generated.length ? generated : SUGGESTIONS,
      });
      runIndex();
    },
    [renderer, runIndex, set],
  );

  const loadAsync = useCallback(
    (makeJob: (onProgress: (m: string) => void) => Promise<LoadedDoc>) => {
      const seq = ++loadSeqRef.current;
      set({ loading: true, loadingMsg: "reading document…", loadError: "" });
      const onProgress = (m: string) => {
        if (seq === loadSeqRef.current) set({ loadingMsg: m });
      };
      makeJob(onProgress).then(
        (doc) => {
          if (seq === loadSeqRef.current) adoptDoc(doc);
        },
        (err: unknown) => {
          if (seq !== loadSeqRef.current) return;
          const msg = err instanceof Error ? err.message : String(err);
          set({ loading: false, loadingMsg: "", loadError: msg });
        },
      );
    },
    [adoptDoc, set],
  );

  /** Synthetic word-by-word pacer, used only for the offline extractive
   *  fallback so the degraded path still looks alive. Real generated text
   *  arrives as deltas and is appended verbatim — pacing it here would
   *  collapse the newlines a model actually emits. */
  const streamExtractive = useCallback(
    (text: string) => {
      const words = text.split(" ");
      set({ answer: "", streaming: true, genPhase: "generating" });
      let i = 0;
      const tick = () => {
        i++;
        set({ answer: words.slice(0, i).join(" ") });
        if (i < words.length) after(STREAM_WORD_MS, tick);
        else set({ streaming: false, phase: "answered", genPhase: "idle" });
      };
      after(40, tick);
    },
    [after, set],
  );

  const runQuery = useCallback(
    (q: string) => {
      clearTimers();
      const seq = ++querySeqRef.current;
      lastQueryRef.current = q;
      const { rag, doc } = stateRef.current;
      const qsteps = steps(rag);
      renderer.view.querySteps = qsteps;
      renderer.view.queryStart = performance.now();

      if (!doc) return;
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
      const sources = buildRealSources(rag, res, scene);

      // The generator must only ever be handed passages that have a
      // visible trace card — buildRealSources curates a smaller, mode-
      // specific display (e.g. naive shows 3 of finalTop's up to 6 chunks),
      // so sending finalTop wholesale would let the model cite a passage
      // the user has no card for, rendering as dead "[5]" text instead of
      // a clickable citation. Recompute each card's citation number to
      // match its position in this exact list, so what the model numbers
      // is what the UI can resolve.
      const chunkById = new Map(doc.chunks.map((c) => [c.id, c]));
      const citable = sources.filter(
        (s): s is typeof s & { chunkId: number } =>
          s.kind === "chunk" && !s.rejected && s.chunkId != null,
      );
      citable.forEach((s, i) => {
        s.citation = i + 1;
      });

      // The extractive answer resolves synchronously and is the fallback of
      // last resort. Real generation runs against it with a deadline: first
      // token wins, and if nothing arrives in time we show the extractive
      // answer and say so, rather than shimmering indefinitely.
      const extractive = res.answer;

      set({
        query: q,
        answer: "",
        sources: [],
        sourcesVisible: false,
        streaming: false,
        degraded: false,
        genPhase: "waiting",
        phase: "querying",
      });

      const streamIdx = qsteps.findIndex((s) => s.stream);
      after(STEP_MS * (streamIdx - 0.4), () =>
        set({ sources, sourcesVisible: true }),
      );
      // hold the pipeline on the generate step until something arrives
      renderer.view.holdStep = streamIdx;

      const ac = new AbortController();
      abortRef.current = ac;

      let released = false;
      const release = (fallback: string | null) => {
        if (released || seq !== querySeqRef.current) return;
        released = true;
        renderer.view.holdStep = null;
        if (fallback != null) {
          set({ degraded: true });
          streamExtractive(fallback);
        } else {
          set({ answer: "", streaming: true, genPhase: "generating" });
        }
      };

      const deadline = setTimeout(() => release(extractive), GEN_DEADLINE_MS);
      timersRef.current.push(deadline);

      const isSample = !!doc.isSample;
      generateLlmAnswer(
        {
          rag,
          query: q,
          doc: isSample ? "sample" : "upload",
          // Sample: bare ids — the Worker resolves them against its own
          // bundled copy (see worker/src/sample.ts). Anything else: full
          // chunk objects, which always route to Workers AI regardless of
          // this "doc" label. Either way, exactly `citable`, in order —
          // see the comment above where citations were recomputed.
          chunks: isSample
            ? citable.map((s) => s.chunkId)
            : citable.map((s) => {
                const c = chunkById.get(s.chunkId)!;
                return { id: c.id, page: c.page, text: c.text };
              }),
        },
        (delta) => {
          if (seq !== querySeqRef.current) return;
          release(null); // first token releases the hold
          clearTimeout(deadline);
          set({ answer: stateRef.current.answer + delta });
        },
        (phase) => {
          if (seq === querySeqRef.current) set({ genPhase: phase });
        },
        ac.signal,
      ).then(
        (ok) => {
          clearTimeout(deadline);
          if (seq !== querySeqRef.current) return;
          if (!ok) release(extractive);
          else if (released)
            set({ streaming: false, phase: "answered", genPhase: "idle" });
        },
        () => {
          clearTimeout(deadline);
          release(extractive);
        },
      );
    },
    [after, clearTimers, renderer, set, streamExtractive],
  );

  const actions: PlaygroundActions = {
    loadSample: () => loadAsync((p) => loadSampleDoc(p)),
    loadFile: (file) => loadAsync((p) => parseFile(file, p)),
    loadUrl: (url) => loadAsync((p) => fetchUrl(url, p)),
    reindex: runIndex,
    clear: () => {
      clearTimers();
      loadSeqRef.current++;
      querySeqRef.current++;
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
        loadingMsg: "",
        loadError: "",
        suggestions: SUGGESTIONS,
        degraded: false,
        genPhase: "idle",
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
