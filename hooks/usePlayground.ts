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
import { embedQuery, embedTexts, vectorAt } from "@/lib/embeddings";
import { checkHealth, generateLlmAnswer, gradePassages, planQuery } from "@/lib/llm";
import { PipelineRenderer } from "@/lib/renderer";
import {
  applyGradeVerdicts,
  buildRealSources,
  correctivePass1,
  generateSuggestions,
  retrieveAgentic,
  retrieveBasic,
  retrieveCorrective,
  retrieveHybrid,
  type AgenticPlanOpts,
  type CorrectivePass1,
} from "@/lib/retrieval";
import { applyProjection, applyQueryToScene, buildScene, sampleScene } from "@/lib/scene";
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
  embedProgress: null,
};

// Bounds a single query's embedding call so a slow/hung Worker degrades
// that one query to lexical-only rather than stalling the whole
// interaction — much shorter than embedTexts' own per-document timeout,
// since a query is one short string, not a whole document.
const QUERY_EMBED_TIMEOUT_MS = 3000;

// Agentic embeds the original query AND its sub-queries in one batched
// /embed call (≤4 short strings — never one call per sub-query), so its
// bound is slightly looser than the single-query one above.
const AGENTIC_EMBED_TIMEOUT_MS = 4500;

// Quality bar on hybrid's RELATED CONCEPTS: send only entities whose OWN
// label matched a query term, never the 1-hop graph neighbors that
// graphBoosted also carries.
//
// Why: on the deployed site "What does it say about patent cases?" sent
// ["Patents", "America", "Khan"] and the answer duly reorganized itself
// around an America-vs-Britain historical contrast instead of patent-case
// substance. None of that noise was a match — it arrived as neighbors of
// "Patents". The neighbor hop is right for boosting *retrieval*
// (co-occurrence points at relevant chunks) and wrong for steering *prose*,
// which is exactly the distinction graphMatched draws.
//
// Improved entity extraction has since removed the worst offenders (Khan,
// Wikimedia Commons, Retrieved) but NOT the geographic ones: for that same
// query the neighbor set is still Patent System + America + United States,
// so this gate still earns its place. Measured against the current
// extraction, matched-only yields exactly the subject of each question —
// Intellectual Property, Patent System, Trade Secret.
//
// Deliberately trades recall for precision: a genuinely useful neighbor
// would be dropped too. Revisit if neighbors ever become trustworthy.
//
// Falls to undefined when a query names nothing the graph knows, which the
// Worker treats as "omit the RELATED CONCEPTS line entirely" — an empty
// list would be worse than none.
function conceptsForPrompt(
  res: { graphMatched?: number[] },
  scene: { gnodes: { full?: string; label: string }[] },
): string[] | undefined {
  const labels = (res.graphMatched ?? [])
    .map((gi) => scene.gnodes[gi]?.full || scene.gnodes[gi]?.label)
    .filter((s): s is string => !!s);
  return labels.length ? labels.slice(0, 6) : undefined;
}

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

      // The sample ships pre-embedded (loadSampleDoc); an upload needs it
      // computed once, here, overlapping with the indexing animation
      // rather than adding to the wait — embedding ~400 chunks takes ~1s
      // in practice, well under INDEX_MS. `doc` is mutated in place once
      // it resolves (mirroring the existing renderer.view mutation
      // pattern), since `stateRef.current.doc` is this same object and the
      // next query reads it directly, no further `set()` required for
      // correctness. Any failure (quota exhausted, Worker down, network)
      // just leaves `doc.dense` unset — every retrieval mode already
      // degrades to lexical-only when that's the case.
      if (!doc.dense) {
        const loadSeq = loadSeqRef.current;
        set({ embedProgress: { done: 0, total: doc.chunks.length } });
        embedTexts(
          doc.chunks.map((c) => c.text),
          (done, total) => {
            if (loadSeq === loadSeqRef.current) set({ embedProgress: { done, total } });
          },
        ).then((dense) => {
          if (loadSeq !== loadSeqRef.current) return; // superseded by a newer load
          set({ embedProgress: null });
          if (dense) {
            doc.dense = dense;
            // Swap the placeholder scatter for the document's real semantic
            // layout now that there are vectors to project. buildScene ran
            // long before this, with nothing to work from.
            applyProjection(renderer.view.scene, dense);
          }
        });
      }
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

  // Split from runQuery below so runQuery's dependency array can reference
  // this by name — the two together are one logical operation, separated
  // only because query embedding (runQuery's job) is the one async step in
  // an otherwise synchronous pipeline.
  const runRetrieval = useCallback(
    (
      seq: number,
      q: string,
      doc: LoadedDoc,
      rag: RagId,
      qsteps: ReturnType<typeof steps>,
      queryVec: Float32Array | null,
      pass1?: CorrectivePass1,
      planOpts?: AgenticPlanOpts,
    ) => {
      const scene = renderer.view.scene;
      const denseOpts = doc.dense && queryVec ? { dense: doc.dense, queryVec } : undefined;
      const res =
        rag === "hybrid"
          ? retrieveHybrid(
              doc.chunks,
              q,
              { nodes: scene.gnodes, neighbors: scene.gnbr },
              denseOpts,
            )
          : rag === "corrective"
            ? // pass1 carries any model verdicts already applied; absent, this
              // recomputes pass 1 itself and falls back to the cosine floor
              retrieveCorrective(doc.chunks, q, denseOpts, pass1)
            : rag === "agentic"
              ? // planOpts carries a validated /plan decomposition; absent,
                // this falls back to the PRF refine loop
                retrieveAgentic(doc.chunks, q, denseOpts, planOpts)
              : retrieveBasic(doc.chunks, q, denseOpts);
      applyQueryToScene(scene, res, queryVec);
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
          // Hybrid: entities the graph linked to this query, so the prose
          // reflects the graph half of the retrieval instead of reading
          // like naive's. Full labels, not the truncated display ones,
          // since the Worker matches them against passage text.
          concepts: rag === "hybrid" ? conceptsForPrompt(res, scene) : undefined,
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
    [after, renderer, set, streamExtractive],
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

      // Agentic plans BEFORE embedding, because the decomposition decides
      // what gets embedded: the original query and every sub-query go to
      // /embed together as one batched call — never one call per
      // sub-query. Serial cost is the plan call (~0.5-0.9s measured)
      // ahead of the embed; both are bounded, and every failure along the
      // way degrades to a working path (plan null → PRF refine loop;
      // embed null → lexical-only retrieval, plan still applied).
      if (rag === "agentic") {
        const ac = new AbortController();
        abortRef.current = ac;
        set({ genPhase: "planning" });
        planQuery(q, ac.signal).then(async (plan) => {
          if (seq !== querySeqRef.current) return;
          let queryVec: Float32Array | null = null;
          let planOpts: AgenticPlanOpts | undefined = plan
            ? { subqueries: plan.subqueries, rationale: plan.rationale }
            : undefined;
          if (doc.dense) {
            set({ genPhase: "embedding" });
            const idx = await Promise.race([
              embedTexts([q, ...(plan?.subqueries ?? [])]),
              new Promise<null>((resolve) =>
                setTimeout(() => resolve(null), AGENTIC_EMBED_TIMEOUT_MS),
              ),
            ]);
            if (seq !== querySeqRef.current) return;
            if (idx) {
              queryVec = vectorAt(idx, 0);
              if (planOpts)
                planOpts = {
                  ...planOpts,
                  subVecs: planOpts.subqueries.map((_, i) => vectorAt(idx, i + 1)),
                };
            }
          }
          runRetrieval(seq, q, doc, rag, qsteps, queryVec, undefined, planOpts);
        });
        return;
      }

      // Query embedding is a network call, so it's the one part of
      // retrieval that isn't synchronous. It's fast (a few hundred ms for
      // one short string) against a 3.1-5s animation, so no holdStep is
      // needed here the way generation needs one — the visual pipeline
      // just keeps playing underneath while this resolves. Bounded by
      // QUERY_EMBED_TIMEOUT_MS so a slow Worker degrades this one query to
      // lexical-only rather than stalling the interaction.
      const embedded = doc.dense
        ? Promise.race([
            embedQuery(q),
            new Promise<null>((resolve) =>
              setTimeout(() => resolve(null), QUERY_EMBED_TIMEOUT_MS),
            ),
          ])
        : Promise.resolve(null);
      if (doc.dense) set({ genPhase: "embedding" });

      embedded.then(async (queryVec) => {
        if (seq !== querySeqRef.current) return;

        // Corrective's grading step: a real model verdict per candidate,
        // replacing the cosine-floor heuristic. Runs here rather than
        // inside retrieval for the same reason query embedding does —
        // retrieval stays synchronous and pure, and this is the one place
        // that can await. Everything about it degrades: if the grader is
        // unreachable, quota-exhausted or slow, `verdicts` is null and
        // retrieveCorrective falls back to exactly the Phase 2 behavior.
        let pass1: CorrectivePass1 | undefined;
        if (rag === "corrective") {
          const denseOpts =
            doc.dense && queryVec ? { dense: doc.dense, queryVec } : undefined;
          pass1 = correctivePass1(doc.chunks, q, denseOpts);
          const ac = new AbortController();
          abortRef.current = ac;
          set({ genPhase: "grading" });
          const isSample = !!doc.isSample;
          const verdicts = await gradePassages(
            {
              query: q,
              doc: isSample ? "sample" : "upload",
              // Same id-vs-text split as generation, for the same reason —
              // see the comment at the generateLlmAnswer call below.
              chunks: isSample
                ? pass1.graded.map((s) => s.chunk.id)
                : pass1.graded.map((s) => ({
                    id: s.chunk.id,
                    page: s.chunk.page,
                    text: s.chunk.text,
                  })),
            },
            ac.signal,
          );
          if (seq !== querySeqRef.current) return;
          if (verdicts) applyGradeVerdicts(pass1, verdicts);
        }

        runRetrieval(seq, q, doc, rag, qsteps, queryVec, pass1);
      });
    },
    [clearTimers, renderer, set, runRetrieval],
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
        embedProgress: null,
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
