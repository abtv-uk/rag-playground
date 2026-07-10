// Canvas pipeline renderer — a direct port of the handoff prototype's drawing
// code. Owns the rAF loop, hover/click state for the 3D graph, and all
// per-architecture diagram layout. React feeds it a mutable `view` each render.

import { ACCENTS, INDEX_MS, REJECT, STEP_MS } from "./constants";
import {
  CHUNKS,
  GEDGES,
  GNODES,
  G_ACTIVE,
  G_NEIGHBORS,
  REL_NAIVE,
} from "./data";
import { steps } from "./steps";
import type { Phase, QueryStep, RagId } from "./types";

export interface RendererView {
  rag: RagId;
  phase: Phase;
  dark: boolean;
  streaming: boolean;
  indexStart: number;
  queryStart: number;
  querySteps: QueryStep[] | null;
  reducedMotion: boolean;
}

interface Pt {
  x: number;
  y: number;
}

interface DiagramNode {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  icon: string;
  sub?: string;
  labelSide?: "top" | "bottom" | "right";
  kind?: string;
}

interface Panel {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  cx: number;
  cy: number;
  title: string;
  sub: string;
}

interface Diagram {
  nodes: Record<string, DiagramNode>;
  panels: Record<string, Panel>;
  edges: Record<string, { pts: Pt[]; dash?: 1 }>;
}

interface ThemeColors {
  panelFill: string;
  panelStroke: string;
  boxFill: string;
  surface: string;
  ink: string;
  ink2: string;
  sub: string;
  faint: string;
  hair: string;
  hair2: string;
  track: string;
  dot: string;
  dotDim: string;
  nodeStroke: string;
  pipeBase: string;
  shadow: string;
}

const TC_DARK: ThemeColors = {
  panelFill: "rgba(255,255,255,0.04)",
  panelStroke: "rgba(255,255,255,0.12)",
  boxFill: "#1c232d",
  surface: "#222a35",
  ink: "#e9ecf1",
  ink2: "#aeb6c2",
  sub: "#838c9a",
  faint: "#5f6875",
  hair: "rgba(255,255,255,0.16)",
  hair2: "rgba(255,255,255,0.08)",
  track: "rgba(255,255,255,0.10)",
  dot: "rgba(255,255,255,0.32)",
  dotDim: "rgba(255,255,255,0.16)",
  nodeStroke: "rgba(255,255,255,0.30)",
  pipeBase: "rgba(255,255,255,0.20)",
  shadow: "rgba(0,0,0,0.55)",
};

const TC_LIGHT: ThemeColors = {
  panelFill: "rgba(255,255,255,0.55)",
  panelStroke: "rgba(17,21,27,0.10)",
  boxFill: "#ffffff",
  surface: "#ffffff",
  ink: "#11151b",
  ink2: "#3a414b",
  sub: "#8a92a0",
  faint: "#aab0bb",
  hair: "rgba(17,21,27,0.13)",
  hair2: "rgba(17,21,27,0.07)",
  track: "rgba(17,21,27,0.07)",
  dot: "rgba(17,21,27,0.22)",
  dotDim: "rgba(17,21,27,0.16)",
  nodeStroke: "rgba(17,21,27,0.30)",
  pipeBase: "rgba(17,21,27,0.17)",
  shadow: "rgba(17,21,27,0.20)",
};

export class PipelineRenderer {
  view: RendererView = {
    rag: "naive",
    phase: "empty",
    dark: true,
    streaming: false,
    indexStart: 0,
    queryStart: 0,
    querySteps: null,
    reducedMotion: false,
  };

  private cv: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private container: HTMLElement | null = null;
  private ro: ResizeObserver | null = null;
  private raf = 0;
  private W = 0;
  private H = 0;
  private dpr = 1;
  private T: ThemeColors = TC_DARK;

  private hoverNode = -1;
  private focusNode = -1;
  private gScreen: { i: number; x: number; y: number; r: number }[] = [];
  private rot = 0;
  private lastNow = 0;
  private mouse: Pt | null = null;
  private drawErrLogged = false;

  attach(canvas: HTMLCanvasElement, container: HTMLElement) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.container = container;
    this.resize();
    canvas.addEventListener("mousemove", this.onMouse);
    canvas.addEventListener("mouseleave", this.onLeave);
    canvas.addEventListener("click", this.onClick);
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.startLoop();
  }

  detach() {
    cancelAnimationFrame(this.raf);
    this.ro?.disconnect();
    if (this.cv) {
      this.cv.removeEventListener("mousemove", this.onMouse);
      this.cv.removeEventListener("mouseleave", this.onLeave);
      this.cv.removeEventListener("click", this.onClick);
    }
    this.cv = null;
    this.ctx = null;
    this.container = null;
  }

  resetGraphInteraction() {
    this.hoverNode = -1;
    this.focusNode = -1;
    if (this.cv) this.cv.style.cursor = "default";
  }

  private onMouse = (e: MouseEvent) => {
    if (!this.cv) return;
    const r = this.cv.getBoundingClientRect();
    this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
    this.updateHover();
  };

  private onLeave = () => {
    this.hoverNode = -1;
    this.mouse = null;
    if (this.cv) this.cv.style.cursor = "default";
  };

  private onClick = (e: MouseEvent) => {
    if (this.view.rag !== "hybrid" || !this.cv) return;
    const r = this.cv.getBoundingClientRect();
    this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
    this.updateHover();
    this.focusNode =
      this.hoverNode >= 0
        ? this.focusNode === this.hoverNode
          ? -1
          : this.hoverNode
        : -1;
  };

  private updateHover() {
    if (this.view.rag !== "hybrid" || !this.mouse || !this.gScreen.length) {
      this.hoverNode = -1;
      if (this.cv) this.cv.style.cursor = "default";
      return;
    }
    let best = -1;
    let bd = 1e9;
    for (const g of this.gScreen) {
      const d = Math.hypot(g.x - this.mouse.x, g.y - this.mouse.y);
      if (d < g.r + 11 && d < bd) {
        bd = d;
        best = g.i;
      }
    }
    this.hoverNode = best;
    if (this.cv) this.cv.style.cursor = best >= 0 ? "pointer" : "default";
  }

  private resize() {
    const cv = this.cv;
    const el = this.container;
    if (!cv || !el || !this.ctx) return;
    const r = el.getBoundingClientRect();
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.W = r.width;
    this.H = r.height;
    cv.width = Math.round(r.width * this.dpr);
    cv.height = Math.round(r.height * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    // setting canvas.width clears it — repaint now so a resize while rAF is
    // throttled (hidden tab) doesn't leave the diagram blank
    try {
      this.draw(performance.now());
    } catch {
      /* first-frame draw may race attach; the loop recovers */
    }
  }

  private startLoop() {
    const loop = (ts: number) => {
      try {
        this.draw(ts || 0);
      } catch (e) {
        if (!this.drawErrLogged) {
          this.drawErrLogged = true;
          console.error("PipelineRenderer draw error:", e);
        }
      }
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  // ---------- layout ----------

  private diagram(): Diagram {
    const W = this.W;
    const H = this.H;
    const rag = this.view.rag;
    const pt = (x: number, y: number): Pt => ({ x, y });
    const N = (
      id: string,
      fx: number,
      fy: number,
      label: string,
      icon: string,
      opt?: Partial<DiagramNode>,
    ): DiagramNode =>
      Object.assign(
        { id, x: fx * W, y: fy * H, w: 46, h: 46, label, icon },
        opt || {},
      );
    const RECT = (
      x0: number,
      y0: number,
      x1: number,
      y1: number,
      title: string,
      sub: string,
    ): Panel => ({
      x0: x0 * W,
      y0: y0 * H,
      x1: x1 * W,
      y1: y1 * H,
      cx: ((x0 + x1) / 2) * W,
      cy: ((y0 + y1) / 2) * H,
      title,
      sub,
    });
    const nodes: Record<string, DiagramNode> = {};
    const panels: Record<string, Panel> = {};
    const edges: Record<string, { pts: Pt[]; dash?: 1 }> = {};
    const add = (n: DiagramNode) => {
      nodes[n.id] = n;
    };
    const r = 25;
    if (rag === "naive") {
      add(N("qn", 0.085, 0.42, "USER QUERY", "query", { labelSide: "top" }));
      add(N("embed", 0.26, 0.22, "EMBEDDING", "embed", { sub: "768-d" }));
      add(N("doc", 0.585, 0.07, "DATA SOURCE", "doc", { labelSide: "right" }));
      panels.vector = RECT(0.4, 0.17, 0.775, 0.6, "VECTOR DB", "64 chunks · 768-d");
      add(N("prompt", 0.5, 0.82, "PROMPT TEMPLATE", "prompt", { sub: "ctx #14 #22 + Q" }));
      add(N("llm", 0.7, 0.82, "LLM", "llm", { sub: "generate" }));
      const n = nodes;
      const v = panels.vector;
      edges.e1 = { pts: [pt(n.qn.x + r, n.qn.y), pt(n.embed.x, n.qn.y), pt(n.embed.x, n.embed.y + r)] };
      edges.e2 = { pts: [pt(n.embed.x + r, n.embed.y), pt(v.x0, n.embed.y)] };
      edges.eDoc = { pts: [pt(n.doc.x, n.doc.y + r), pt(n.doc.x, v.y0)] };
      edges.e3 = { pts: [pt(n.prompt.x, v.y1), pt(n.prompt.x, n.prompt.y - r)] };
      edges.eQP = { pts: [pt(n.qn.x, n.qn.y + r), pt(n.qn.x, n.prompt.y), pt(n.prompt.x - r, n.prompt.y)] };
      edges.e6 = { pts: [pt(n.prompt.x + r, n.prompt.y), pt(n.llm.x - r, n.llm.y)] };
      edges.e8 = { pts: [pt(n.llm.x + r, n.llm.y), pt(W - 68, n.llm.y)] };
    } else if (rag === "hybrid" || rag === "corrective") {
      add(N("qn", 0.075, 0.4, "USER QUERY", "query", { labelSide: "top" }));
      add(N("embed", 0.225, 0.2, "EMBEDDING", "embed", { sub: "768-d" }));
      panels.vector = RECT(0.325, 0.1, 0.625, 0.555, "VECTOR DB", rag === "hybrid" ? "context 1" : "64 chunks · 768-d");
      panels.sec =
        rag === "hybrid"
          ? RECT(0.685, 0.1, 0.965, 0.555, "KNOWLEDGE GRAPH", "context 2 · hover & click")
          : RECT(0.685, 0.1, 0.965, 0.555, "RELEVANCE GRADING", "grade · correct");
      add(N("prompt", 0.475, 0.82, "PROMPT TEMPLATE", "prompt", { sub: "ctx #14 #22 + Q" }));
      add(N("llm", 0.675, 0.82, "LLM", "llm", { sub: "generate" }));
      const n = nodes;
      const v = panels.vector;
      const sc = panels.sec;
      const mY = n.prompt.y - 42;
      edges.e1 = { pts: [pt(n.qn.x + r, n.qn.y), pt(n.embed.x, n.qn.y), pt(n.embed.x, n.embed.y + r)] };
      edges.e2 = { pts: [pt(n.embed.x + r, n.embed.y), pt(v.x0, n.embed.y)] };
      edges.eQP = { pts: [pt(n.qn.x, n.qn.y + r), pt(n.qn.x, n.prompt.y), pt(n.prompt.x - r, n.prompt.y)] };
      edges.e6 = { pts: [pt(n.prompt.x + r, n.prompt.y), pt(n.llm.x - r, n.llm.y)] };
      edges.e8 = { pts: [pt(n.llm.x + r, n.llm.y), pt(W - 68, n.llm.y)] };
      if (rag === "hybrid") {
        edges.e4 = { pts: [pt(n.embed.x, n.embed.y - r), pt(n.embed.x, H * 0.045), pt(sc.cx, H * 0.045), pt(sc.cx, sc.y0)] };
        edges.e3 = { pts: [pt(n.prompt.x, v.y1), pt(n.prompt.x, n.prompt.y - r)] };
        edges.e5 = { pts: [pt(sc.cx, sc.y1), pt(sc.cx, mY), pt(n.prompt.x, mY), pt(n.prompt.x, n.prompt.y - r)] };
      } else {
        edges.e3g = { pts: [pt(v.x1, v.cy - 20), pt(sc.x0, v.cy - 20)] };
        edges.eRe = { pts: [pt(sc.x0, v.cy + 24), pt(v.x1, v.cy + 24)], dash: 1 };
        edges.egp = { pts: [pt(sc.cx, sc.y1), pt(sc.cx, mY), pt(n.prompt.x, mY), pt(n.prompt.x, n.prompt.y - r)] };
      }
    } else {
      add(N("qn", 0.075, 0.46, "USER QUERY", "query", { labelSide: "top" }));
      add(Object.assign(N("agent", 0.42, 0.46, "AGENT", "agent"), { w: 118, h: 50, kind: "agent" }));
      panels.vector = RECT(0.055, 0.6, 0.375, 0.9, "VECTOR DB", "64 chunks · 768-d");
      add(N("llm", 0.615, 0.82, "LLM", "llm", { sub: "generate" }));
      const n = nodes;
      const v = panels.vector;
      const ag = n.agent;
      edges.e1 = { pts: [pt(n.qn.x + r, n.qn.y), pt(ag.x - ag.w / 2, ag.y)] };
      edges.eM = { pts: [pt(0.295 * W, 0.1 * H + 26), pt(0.295 * W, ag.y - 52), pt(ag.x - 18, ag.y - 52), pt(ag.x - 18, ag.y - ag.h / 2)] };
      edges.eP = { pts: [pt(0.545 * W, 0.1 * H + 26), pt(0.545 * W, ag.y - 52), pt(ag.x + 18, ag.y - 52), pt(ag.x + 18, ag.y - ag.h / 2)] };
      edges.eAV = { pts: [pt(ag.x - 34, ag.y + ag.h / 2), pt(ag.x - 34, H * 0.575), pt(v.cx, H * 0.575), pt(v.cx, v.y0)] };
      edges.eVA = { pts: [pt(v.x1, v.y0 + 26), pt(ag.x, v.y0 + 26), pt(ag.x, ag.y + ag.h / 2)], dash: 1 };
      edges.eT = { pts: [pt(ag.x + ag.w / 2, ag.y), pt(0.765 * W, ag.y), pt(0.765 * W, 0.42 * H), pt(0.805 * W, 0.42 * H)] };
      edges.eL = { pts: [pt(ag.x + 34, ag.y + ag.h / 2), pt(ag.x + 34, n.llm.y), pt(n.llm.x - r, n.llm.y)] };
      edges.e8 = { pts: [pt(n.llm.x + r, n.llm.y), pt(W - 68, n.llm.y)] };
    }
    return { nodes, panels, edges };
  }

  // ---------- helpers ----------

  private hx(h: string): [number, number, number] {
    const n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  private rgba(h: string, a: number) {
    const c = this.hx(h);
    return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  }
  private rr(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
    if (w <= 0 || h <= 0) {
      ctx.beginPath();
      return;
    }
    r = Math.max(0, Math.min(r, h / 2, w / 2));
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  private pn(panel: Panel, nx: number, ny: number): Pt {
    return { x: panel.x0 + nx * (panel.x1 - panel.x0), y: panel.y0 + ny * (panel.y1 - panel.y0) };
  }
  private truncate(ctx: CanvasRenderingContext2D, text: string, maxw: number) {
    if (ctx.measureText(text).width <= maxw) return text;
    let t = text;
    while (t.length > 1 && ctx.measureText(t + "…").width > maxw) t = t.slice(0, -1);
    return t + "…";
  }
  private cl(t: number) {
    return Math.max(0, Math.min(1, t));
  }
  private easeOut(t: number) {
    return 1 - Math.pow(1 - t, 3);
  }

  // ---------- main draw ----------

  private draw(ts: number) {
    const ctx = this.ctx;
    if (!ctx || !this.W) return;
    const { rag, phase: ph, reducedMotion } = this.view;
    const T = (this.T = this.view.dark ? TC_DARK : TC_LIGHT);
    const W = this.W;
    const H = this.H;
    ctx.clearRect(0, 0, W, H);
    const A = ACCENTS[rag];
    const L = this.diagram();
    const now = ts;

    let settle = 1;
    let gapp = 1;
    let idxP = 0;
    if (ph === "indexing") {
      idxP = Math.min(1, (performance.now() - this.view.indexStart) / INDEX_MS);
      if (reducedMotion) {
        settle = 1;
        gapp = 1;
      } else {
        settle = this.easeOut(Math.max(0, Math.min(1, (idxP - 0.34) / 0.62)));
        gapp = Math.max(0, Math.min(1, (idxP - 0.52) / 0.46));
      }
    }
    if (ph === "empty") {
      settle = 0;
      gapp = 0;
    }

    let step = -1;
    let qsteps: QueryStep[] | null = null;
    const curE = new Set<string>();
    if (ph === "querying" || ph === "answered") {
      qsteps = this.view.querySteps || steps(rag);
      const el = performance.now() - this.view.queryStart;
      step = Math.min(qsteps.length - 1, Math.floor(el / STEP_MS));
      if (ph === "answered" || reducedMotion) step = qsteps.length - 1;
      if (ph === "querying" && qsteps[step])
        (qsteps[step].e || []).forEach((x) => curE.add(x));
    }

    const lit = new Set<string>();
    const litEdges = new Set<string>();
    const F: Record<string, boolean> = {};
    if (qsteps) {
      for (let i = 0; i <= step; i++) {
        const st = qsteps[i];
        st.lit.forEach((b) => lit.add(b));
        st.e.forEach((x) => litEdges.add(x));
        (["hl", "grade", "correct", "loop", "refine", "plan", "merge"] as const).forEach(
          (k) => {
            if (st[k]) F[k] = true;
          },
        );
      }
    }
    const anim = ph === "querying" && !reducedMotion;
    const idxDoc =
      rag === "naive" && L.nodes.doc
        ? { x: L.nodes.doc.x, y: L.nodes.doc.y }
        : { x: W * 0.52, y: H * 0.76 };
    this.gScreen = [];

    const v = L.panels.vector;
    this.drawPanel(ctx, v, v.title, v.sub, A, lit.has("vector"));
    this.drawScatter(ctx, v, A, settle, now, {
      highlightOn: !!F.hl,
      active: ph === "querying" || ph === "answered",
      anim,
      src: ph === "indexing" && !reducedMotion ? idxDoc : null,
    });
    const sec = L.panels.sec;
    if (sec) {
      this.drawPanel(ctx, sec, sec.title, sec.sub, A, lit.has("sec"));
      if (rag === "hybrid")
        this.drawGraph3D(ctx, sec, A, gapp, now, lit.has("sec"), anim || (ph === "indexing" && !reducedMotion));
      else
        this.drawGradePanel(ctx, sec, A, now, { gradeOn: !!F.grade, correctOn: !!F.correct });
    }

    const dimming = ph === "indexing" && !reducedMotion;
    if (dimming) ctx.globalAlpha = 0.16;
    Object.keys(L.edges).forEach((id) => {
      const e = L.edges[id];
      if (dimming && id === "eDoc") {
        ctx.globalAlpha = 1;
        this.drawRoute(ctx, e.pts, A, { lit: true, flowing: true, now });
        ctx.globalAlpha = 0.16;
        return;
      }
      this.drawRoute(ctx, e.pts, A, {
        lit: litEdges.has(id),
        flowing: litEdges.has(id) && anim,
        now,
        dash: e.dash,
      });
    });
    this.drawQueryPipe(ctx, L, A, litEdges.has("eq"), litEdges.has("eq") && anim, now);
    Object.keys(L.nodes).forEach((id) => {
      const n = L.nodes[id];
      if (n.kind === "agent") return;
      if (id === "llm") n.sub = this.view.streaming ? "generating…" : "generate";
      this.drawNode(ctx, n, A, { lit: lit.has(id) });
    });
    if (rag === "agentic") this.drawAgentScene(ctx, L, A, now, { lit, anim, F });
    this.drawOutputTag(ctx, L, A, lit.has("output"));
    if (dimming) {
      ctx.globalAlpha = 1;
      this.drawIndexing(ctx, A, idxP, idxDoc);
    }
    if (ph === "querying" && !reducedMotion) {
      curE.forEach((id) => {
        if (L.edges[id]) this.drawParticles(ctx, L.edges[id].pts, A, now);
      });
    }
  }

  // ---------- pieces ----------

  private drawQueryPipe(
    ctx: CanvasRenderingContext2D,
    L: Diagram,
    A: string,
    lit: boolean,
    flowing: boolean,
    now: number,
  ) {
    const T = this.T;
    const qn = L.nodes.qn;
    if (!qn) return;
    const gx = Math.max(24, qn.x - 46);
    const pts = [
      { x: this.W * 0.5, y: this.H - 4 },
      { x: gx, y: this.H - 4 },
      { x: gx, y: qn.y },
      { x: qn.x - (qn.w || 46) / 2 - 2, y: qn.y },
    ];
    this.drawRoute(ctx, pts, A, { lit, flowing, now });
    ctx.save();
    ctx.beginPath();
    ctx.arc(this.W * 0.5, this.H - 4, 4, 0, 7);
    ctx.fillStyle = lit ? A : T.faint;
    ctx.fill();
    ctx.restore();
  }

  private drawPanel(
    ctx: CanvasRenderingContext2D,
    p: Panel,
    title: string,
    sub: string,
    A: string,
    glow: boolean,
  ) {
    const T = this.T;
    ctx.save();
    if (glow) {
      ctx.shadowColor = this.rgba(A, 0.2);
      ctx.shadowBlur = 24;
    }
    this.rr(ctx, p.x0, p.y0, p.x1 - p.x0, p.y1 - p.y0, 14);
    ctx.fillStyle = T.panelFill;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.lineWidth = 1;
    ctx.strokeStyle = glow ? this.rgba(A, 0.5) : T.panelStroke;
    ctx.stroke();
    ctx.restore();
    ctx.save();
    this.rr(ctx, p.x0 + 1, p.y0 + 1, p.x1 - p.x0 - 2, p.y1 - p.y0 - 2, 13);
    ctx.clip();
    ctx.font = '600 10px "JetBrains Mono",monospace';
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    const maxw = p.x1 - p.x0 - 28;
    ctx.fillStyle = glow ? A : T.sub;
    ctx.fillText(this.truncate(ctx, title, maxw), p.x0 + 14, p.y0 + 20);
    ctx.font = '400 9px "JetBrains Mono",monospace';
    ctx.fillStyle = T.faint;
    ctx.fillText(this.truncate(ctx, sub, maxw), p.x0 + 14, p.y0 + 33);
    ctx.restore();
  }

  private drawScatter(
    ctx: CanvasRenderingContext2D,
    p: Panel,
    A: string,
    settle: number,
    now: number,
    o: { highlightOn: boolean; active: boolean; anim: boolean; src: Pt | null },
  ) {
    const T = this.T;
    const src = o.src || { x: p.x0 + (p.x1 - p.x0) * 0.5, y: p.y0 - 40 };
    const rel = new Set(o.highlightOn ? REL_NAIVE : []);
    ctx.save();
    this.rr(ctx, p.x0 + 1, p.y0 + 1, p.x1 - p.x0 - 2, p.y1 - p.y0 - 2, 13);
    ctx.clip();
    CHUNKS.forEach((c) => {
      const local = Math.max(0, Math.min(1, (settle - c.delay) / (1 - c.delay || 1)));
      const t = this.easeOut(local);
      const tp = this.pn(p, c.nx, c.ny);
      const x = src.x + (tp.x - src.x) * t;
      const y = src.y + (tp.y - src.y) * t;
      const isRel = rel.has(c.idx);
      let R = 2.1;
      let fill = T.dot;
      if (o.active && !o.highlightOn) fill = T.dotDim;
      if (isRel) {
        R = 3.6;
        fill = A;
        const pulse = o.anim ? 0.5 + 0.5 * Math.sin(now / 300) : 0.5;
        ctx.beginPath();
        ctx.arc(x, y, 7 + pulse * 2, 0, 7);
        ctx.fillStyle = this.rgba(A, 0.13);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(x, y, R, 0, 7);
      ctx.fillStyle = fill;
      ctx.fill();
    });
    ctx.restore();
  }

  private drawGraph3D(
    ctx: CanvasRenderingContext2D,
    p: Panel,
    A: string,
    app: number,
    now: number,
    active: boolean,
    anim: boolean,
  ) {
    const T = this.T;
    ctx.save();
    this.rr(ctx, p.x0 + 1, p.y0 + 1, p.x1 - p.x0 - 2, p.y1 - p.y0 - 2, 13);
    ctx.clip();
    const cx = p.cx;
    const cy = (p.y0 + p.y1) / 2 + 6;
    const radX = (p.x1 - p.x0) * 0.42;
    const radY = (p.y1 - p.y0) * 0.4;
    const interact = this.focusNode >= 0 ? this.focusNode : this.hoverNode;
    const rotSpeed = interact >= 0 ? 0.00006 : 0.00016;
    const dt = Math.min(60, now - (this.lastNow || now));
    this.lastNow = now;
    if ((anim || this.hoverNode >= 0) && !this.view.reducedMotion)
      this.rot += dt * rotSpeed;
    const ang = this.rot;
    const proj = GNODES.map((nd, i) => {
      const [x, y, z] = nd.p;
      const xr = x * Math.cos(ang) + z * Math.sin(ang);
      const zr = -x * Math.sin(ang) + z * Math.cos(ang);
      const f = 2.9;
      const sc = f / (f - zr);
      return { i, x: cx + xr * radX * sc, y: cy + y * radY * sc, z: zr, sc, depth: (zr + 1) / 2 };
    });
    const activeN = new Set(active ? G_ACTIVE : []);
    const nbr = interact >= 0 ? G_NEIGHBORS[interact] : null;
    GEDGES.forEach((e, ei) => {
      const a = proj[e.a];
      const b = proj[e.b];
      const grow = Math.max(0, Math.min(1, (app - (ei / GEDGES.length) * 0.5) / 0.5));
      if (grow <= 0) return;
      const mx = a.x + (b.x - a.x) * grow;
      const my = a.y + (b.y - a.y) * grow;
      let sc2 = T.hair;
      let lw = 1;
      if (activeN.has(e.a) && activeN.has(e.b)) {
        sc2 = this.rgba(A, 0.42);
        lw = 1.4;
      }
      if (interact >= 0) {
        if (e.a === interact || e.b === interact) {
          sc2 = this.rgba(A, 0.9);
          lw = 2.2;
        } else {
          sc2 = T.hair2;
          lw = 1;
        }
      }
      ctx.strokeStyle = sc2;
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(mx, my);
      ctx.stroke();
    });
    this.gScreen = [];
    proj
      .slice()
      .sort((a, b) => a.z - b.z)
      .forEach((q) => {
        const i = q.i;
        const nd = GNODES[i];
        const scale = Math.max(0, Math.min(1, (app - (i / GNODES.length) * 0.4) / 0.5));
        if (scale <= 0) return;
        const isAct = activeN.has(i);
        const foc = i === interact;
        const dim = interact >= 0 && !foc && !(nbr && nbr.has(i));
        const R = (isAct || foc ? 6.5 : 5) * q.sc * this.easeOut(scale);
        this.gScreen.push({ i, x: q.x, y: q.y, r: R });
        if ((isAct || foc) && !dim) {
          const pulse = anim || foc ? 0.5 + 0.5 * Math.sin(now / 300 + i) : 0.6;
          ctx.beginPath();
          ctx.arc(q.x, q.y, R + 6 + pulse * 2, 0, 7);
          ctx.fillStyle = this.rgba(A, 0.13);
          ctx.fill();
        }
        ctx.globalAlpha = dim ? 0.3 : 1;
        ctx.beginPath();
        ctx.arc(q.x, q.y, R, 0, 7);
        ctx.fillStyle = isAct || foc ? A : T.boxFill;
        ctx.fill();
        ctx.lineWidth = foc ? 2.4 : 1.5;
        ctx.strokeStyle = isAct || foc ? A : T.nodeStroke;
        ctx.stroke();
        if (q.depth > 0.52 || foc || isAct || (nbr && nbr.has(i))) {
          const narrowPanel = p.x1 - p.x0 < 180;
          if (!narrowPanel || foc || (nbr && nbr.has(i))) {
            ctx.font =
              (foc ? "600 " : "500 ") +
              (8.5 * Math.max(0.85, q.sc)).toFixed(1) +
              'px "JetBrains Mono",monospace';
            ctx.textBaseline = "top";
            ctx.fillStyle = dim ? T.faint : isAct || foc ? T.ink : T.sub;
            const halfW = ctx.measureText(nd.label).width / 2;
            const pad = 7;
            const lx = Math.max(p.x0 + pad + halfW, Math.min(p.x1 - pad - halfW, q.x));
            ctx.textAlign = "center";
            ctx.fillText(nd.label, lx, Math.min(q.y + R + 3, p.y1 - 14));
          }
        }
        ctx.globalAlpha = 1;
      });
    ctx.restore();
    if (interact >= 0)
      this.drawNodeTooltip(ctx, p, proj[interact], interact, A, this.focusNode >= 0);
  }

  private drawNodeTooltip(
    ctx: CanvasRenderingContext2D,
    p: Panel,
    q: { x: number; y: number },
    i: number,
    A: string,
    locked: boolean,
  ) {
    const T = this.T;
    const nd = GNODES[i];
    const tw = 180;
    const lines = this.wrap(ctx, nd.desc, tw - 24);
    const th = 44 + lines.length * 13;
    let tx = q.x + 16;
    if (tx + tw > p.x1 - 6) tx = q.x - 16 - tw;
    tx = Math.max(p.x0 + 6, tx);
    const ty = Math.max(p.y0 + 6, Math.min(q.y - th / 2, p.y1 - th - 6));
    ctx.save();
    ctx.shadowColor = T.shadow;
    ctx.shadowBlur = 20;
    ctx.shadowOffsetY = 5;
    this.rr(ctx, tx, ty, tw, th, 11);
    ctx.fillStyle = T.surface;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;
    ctx.lineWidth = 1;
    ctx.strokeStyle = this.rgba(A, 0.5);
    ctx.stroke();
    ctx.textAlign = "left";
    ctx.font = '600 12.5px "Space Grotesk",sans-serif';
    ctx.fillStyle = T.ink;
    ctx.textBaseline = "top";
    ctx.fillText(nd.label, tx + 13, ty + 11);
    ctx.font = '500 8px "JetBrains Mono",monospace';
    ctx.fillStyle = A;
    ctx.fillText(
      G_NEIGHBORS[i].size + " LINKS · ENTITY" + (locked ? " · PINNED" : ""),
      tx + 13,
      ty + 27,
    );
    ctx.font = '400 10.5px "Space Grotesk",sans-serif';
    ctx.fillStyle = T.ink2;
    lines.forEach((ln, k) => ctx.fillText(ln, tx + 13, ty + 39 + k * 13));
    ctx.restore();
  }

  private wrap(ctx: CanvasRenderingContext2D, text: string, maxw: number) {
    ctx.font = '400 10.5px "Space Grotesk",sans-serif';
    const words = text.split(" ");
    const lines: string[] = [];
    let cur = "";
    for (const w of words) {
      const t = cur ? cur + " " + w : w;
      if (ctx.measureText(t).width > maxw && cur) {
        lines.push(cur);
        cur = w;
      } else cur = t;
    }
    if (cur) lines.push(cur);
    return lines;
  }

  private drawGradePanel(
    ctx: CanvasRenderingContext2D,
    p: Panel,
    A: string,
    now: number,
    o: { gradeOn: boolean; correctOn: boolean },
  ) {
    const T = this.T;
    const pw = p.x1 - p.x0;
    const narrow = pw < 150;
    const pad = narrow ? 10 : 18;
    const x0 = p.x0 + pad;
    const x1 = p.x1 - pad;
    const stackPills = pw < 170;
    const pillH = 26;
    const pillBlock = stackPills ? pillH * 2 + 8 : pillH + 4;
    const top = p.y0 + (narrow ? 44 : 54);
    const bottom = p.y1 - pillBlock - 16;
    const rows = [
      { n: 14, pass: 1, s: 0.93 },
      { n: 7, pass: 0, s: 0.31 },
      { n: 22, pass: 1, s: 0.79 },
      { n: 31, pass: 0, s: 0.28 },
      { n: 9, pass: 1, s: 0.61 },
    ];
    const gap = narrow ? 6 : 10;
    let rows2 = rows;
    let rh = Math.max(16, Math.min(42, (bottom - top - (rows.length - 1) * gap) / rows.length));
    const fit = Math.max(1, Math.floor((bottom - top + gap) / (rh + gap)));
    if (fit < rows.length) {
      rows2 = rows.slice(0, fit);
      rh = Math.max(16, Math.min(42, (bottom - top - (rows2.length - 1) * gap) / rows2.length));
    }
    const shown = o.gradeOn || o.correctOn;
    ctx.save();
    this.rr(ctx, p.x0 + 1, p.y0 + 1, pw - 2, p.y1 - p.y0 - 2, 13);
    ctx.clip();
    const showBar = !narrow && rh >= 26;
    const fs = narrow ? 8.5 : 11;
    const inPad = narrow ? 7 : 12;
    rows2.forEach((r, i) => {
      const y = top + i * (rh + gap);
      ctx.save();
      this.rr(ctx, x0, y, x1 - x0, rh, narrow ? 6 : 9);
      ctx.fillStyle = T.boxFill;
      ctx.fill();
      ctx.lineWidth = 1.2;
      if (shown && !r.pass) ctx.setLineDash([4, 3]);
      ctx.strokeStyle = shown ? (r.pass ? this.rgba(A, 0.5) : this.rgba(REJECT, 0.6)) : T.hair;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 " + fs + 'px "JetBrains Mono",monospace';
      ctx.fillStyle = T.ink;
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      const mx = x1 - (narrow ? 12 : 22);
      const my = y + rh / 2;
      const label = this.truncate(ctx, (narrow ? "#" : "chunk #") + r.n, mx - x0 - inPad - 14);
      ctx.fillText(label, x0 + inPad, showBar ? y + rh * 0.38 : my);
      if (showBar) {
        const bw = Math.max(10, mx - x0 - inPad - 14);
        const by = y + rh * 0.62;
        this.rr(ctx, x0 + inPad, by, bw, 4, 2);
        ctx.fillStyle = T.track;
        ctx.fill();
        this.rr(ctx, x0 + inPad, by, bw * r.s, 4, 2);
        ctx.fillStyle = shown ? (r.pass ? A : REJECT) : T.dotDim;
        ctx.fill();
      }
      const mk = narrow ? 3.5 : 5;
      if (shown) {
        ctx.lineWidth = 1.8;
        ctx.strokeStyle = r.pass ? A : REJECT;
        if (r.pass) {
          ctx.beginPath();
          ctx.moveTo(mx - mk, my + 1);
          ctx.lineTo(mx - 1, my + mk);
          ctx.lineTo(mx + mk + 1, my - mk);
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.moveTo(mx - mk, my - mk);
          ctx.lineTo(mx + mk, my + mk);
          ctx.moveTo(mx + mk, my - mk);
          ctx.lineTo(mx - mk, my + mk);
          ctx.stroke();
        }
      } else {
        ctx.font = "600 " + fs + 'px "JetBrains Mono",monospace';
        ctx.fillStyle = T.faint;
        ctx.textAlign = "center";
        ctx.fillText("···", mx, my);
      }
      ctx.restore();
    });
    const py = p.y1 - pillBlock - 4;
    if (stackPills) {
      this.drawPill(ctx, x0, py, "SEARCH", o.correctOn, A, x1 - x0, pillH);
      this.drawPill(ctx, x0, py + pillH + 8, "CORRECT", o.correctOn, A, x1 - x0, pillH);
    } else {
      const half = (x1 - x0 - 12) / 2;
      this.drawPill(ctx, x0, py, "SEARCH", o.correctOn, A, half, pillH);
      this.drawPill(ctx, x0 + half + 12, py, "CORRECT", o.correctOn, A, half, pillH);
    }
    ctx.restore();
  }

  private drawPill(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    text: string,
    on: boolean,
    A: string,
    w: number,
    h = 30,
  ) {
    const T = this.T;
    ctx.save();
    this.rr(ctx, x, y, w, h, 8);
    ctx.fillStyle = on ? this.rgba(A, 0.12) : T.boxFill;
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = on ? A : T.hair;
    ctx.stroke();
    ctx.font = '600 9px "JetBrains Mono",monospace';
    ctx.fillStyle = on ? A : T.sub;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillText(this.truncate(ctx, text, w - 10), x + w / 2, y + h / 2 + 0.5);
    ctx.restore();
  }

  private drawChip(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    w: number,
    h: number,
    text: string,
    on: boolean,
    A: string,
  ) {
    const T = this.T;
    ctx.save();
    this.rr(ctx, x, y, w, h, 8);
    ctx.fillStyle = on ? this.rgba(A, 0.12) : T.boxFill;
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = on ? this.rgba(A, 0.55) : T.hair;
    ctx.stroke();
    ctx.font = '600 9.5px "JetBrains Mono",monospace';
    ctx.fillStyle = on ? A : T.sub;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillText(text, x + w / 2, y + h / 2 + 0.5);
    ctx.restore();
  }

  private drawAgentScene(
    ctx: CanvasRenderingContext2D,
    L: Diagram,
    A: string,
    now: number,
    o: { lit: Set<string>; anim: boolean; F: Record<string, boolean> },
  ) {
    const T = this.T;
    const W = this.W;
    const H = this.H;
    const ag = L.nodes.agent;
    const litMem = o.lit.has("mem");
    const litPlan = o.lit.has("plan");
    const litT = o.lit.has("tools");
    const litA = o.lit.has("agent");
    const groups = [
      { cx: 0.295 * W, label: "MEMORY", chips: ["SHORT", "LONG"], on: litMem },
      { cx: 0.545 * W, label: "PLANNING", chips: ["ReAct", "CoT"], on: litPlan },
    ];
    ctx.textBaseline = "alphabetic";
    groups.forEach((g) => {
      const cw = Math.min(62, W * 0.083);
      const chh = 26;
      const gap = 8;
      const total = g.chips.length * cw + gap;
      let x = g.cx - total / 2;
      ctx.font = '600 8px "JetBrains Mono",monospace';
      ctx.textAlign = "center";
      ctx.fillStyle = g.on ? A : T.sub;
      ctx.fillText(g.label, g.cx, 0.1 * H - 9);
      g.chips.forEach((c) => {
        this.drawChip(ctx, x, 0.1 * H, cw, chh, c, g.on, A);
        x += cw + gap;
      });
    });
    const tx = 0.805 * W;
    const tw = 0.16 * W;
    const rows = [0.26, 0.42, 0.58];
    ctx.font = '600 8px "JetBrains Mono",monospace';
    ctx.textAlign = "center";
    ctx.fillStyle = litT ? A : T.sub;
    ctx.fillText("MCP TOOLS", tx + tw / 2, rows[0] * H - 26);
    ctx.strokeStyle = litT ? this.rgba(A, 0.5) : T.hair;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(tx - 13, rows[0] * H);
    ctx.lineTo(tx - 13, rows[2] * H);
    ctx.stroke();
    ["SEARCH", "LOCAL", "CLOUD"].forEach((t, i) => {
      const y = rows[i] * H;
      ctx.beginPath();
      ctx.moveTo(tx - 13, y);
      ctx.lineTo(tx, y);
      ctx.stroke();
      this.drawChip(ctx, tx, y - 15, tw, 30, t, litT, A);
    });
    if (o.F.loop) {
      ctx.save();
      ctx.strokeStyle = this.rgba(A, 0.5);
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 5]);
      if (o.anim) {
        const sweep = (now / 420) % (Math.PI * 2);
        ctx.lineDashOffset = -(now / 30) % 18;
        ctx.beginPath();
        ctx.ellipse(ag.x, ag.y, ag.w * 0.72, ag.h * 1.05, 0, sweep, sweep + Math.PI * 1.5);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.ellipse(ag.x, ag.y, ag.w * 0.72, ag.h * 1.05, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();
    }
    ctx.save();
    if (litA) {
      ctx.shadowColor = this.rgba(A, 0.35);
      ctx.shadowBlur = 20;
    }
    this.rr(ctx, ag.x - ag.w / 2, ag.y - ag.h / 2, ag.w, ag.h, 13);
    ctx.fillStyle = T.boxFill;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.lineWidth = litA ? 1.9 : 1.3;
    ctx.strokeStyle = litA ? A : T.hair;
    ctx.stroke();
    ctx.font = '600 13px "Space Grotesk",sans-serif';
    ctx.fillStyle = litA ? T.ink : T.sub;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("AGENT", ag.x, ag.y - 6);
    ctx.font = '500 8px "JetBrains Mono",monospace';
    ctx.fillStyle = litA ? A : T.faint;
    ctx.fillText(
      o.F.refine
        ? "refining query"
        : o.F.loop
          ? "retrieval loop"
          : o.F.plan
            ? "planning"
            : "reasoning",
      ag.x,
      ag.y + 9,
    );
    ctx.restore();
  }

  private drawRoute(
    ctx: CanvasRenderingContext2D,
    pts: Pt[],
    A: string,
    o: { lit: boolean; flowing?: boolean; now: number; dash?: 1 },
  ) {
    const T = this.T;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const trace = () => {
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length - 1; i++)
        ctx.arcTo(pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y, 12);
      ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
    };
    if (o.dash && !o.lit) ctx.setLineDash([4, 4]);
    trace();
    ctx.lineWidth = o.lit ? 2.6 : 1.6;
    if (o.lit) {
      ctx.shadowColor = this.rgba(A, 0.5);
      ctx.shadowBlur = 9;
      ctx.strokeStyle = this.rgba(A, 0.92);
      ctx.stroke();
      ctx.shadowBlur = 0;
      if (o.flowing) {
        ctx.setLineDash([3, 9]);
        ctx.lineDashOffset = -(o.now / 22) % 12;
        ctx.strokeStyle = "rgba(255,255,255,0.85)";
        ctx.lineWidth = 1.7;
        ctx.stroke();
      }
    } else {
      ctx.strokeStyle = T.pipeBase;
      ctx.stroke();
    }
    ctx.setLineDash([]);
    const a = pts[pts.length - 2];
    const b = pts[pts.length - 1];
    const ang = Math.atan2(b.y - a.y, b.x - a.x);
    ctx.fillStyle = o.lit ? this.rgba(A, 0.95) : T.pipeBase;
    ctx.beginPath();
    ctx.moveTo(b.x, b.y);
    ctx.lineTo(b.x - 7 * Math.cos(ang - 0.42), b.y - 7 * Math.sin(ang - 0.42));
    ctx.lineTo(b.x - 7 * Math.cos(ang + 0.42), b.y - 7 * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  private routeAt(pts: Pt[], t: number): Pt {
    let total = 0;
    const segs: number[] = [];
    for (let i = 1; i < pts.length; i++) {
      const l = Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
      segs.push(l);
      total += l;
    }
    let d = t * total;
    for (let i = 0; i < segs.length; i++) {
      if (d <= segs[i]) {
        const u = segs[i] ? d / segs[i] : 0;
        return {
          x: pts[i].x + (pts[i + 1].x - pts[i].x) * u,
          y: pts[i].y + (pts[i + 1].y - pts[i].y) * u,
        };
      }
      d -= segs[i];
    }
    return pts[pts.length - 1];
  }

  private drawParticles(ctx: CanvasRenderingContext2D, pts: Pt[], A: string, now: number) {
    const n = 3;
    for (let i = 0; i < n; i++) {
      const t = (now / 900 + i / n) % 1;
      const p = this.routeAt(pts, t);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.6, 0, 7);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.6, 0, 7);
      ctx.strokeStyle = this.rgba(A, 0.9);
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }
  }

  private drawNode(
    ctx: CanvasRenderingContext2D,
    n: DiagramNode,
    A: string,
    o: { lit: boolean },
  ) {
    const T = this.T;
    const w = n.w || 46;
    const h = n.h || 46;
    const x = n.x - w / 2;
    const y = n.y - h / 2;
    ctx.save();
    if (o.lit) {
      ctx.shadowColor = this.rgba(A, 0.35);
      ctx.shadowBlur = 18;
    }
    this.rr(ctx, x, y, w, h, 13);
    ctx.fillStyle = T.boxFill;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.lineWidth = o.lit ? 1.8 : 1.2;
    ctx.strokeStyle = o.lit ? A : T.hair;
    ctx.stroke();
    this.drawIcon(ctx, n.icon, n.x, n.y, o.lit ? A : T.sub);
    ctx.font = '600 8.5px "JetBrains Mono",monospace';
    ctx.textAlign = "center";
    ctx.fillStyle = o.lit ? T.ink : T.sub;
    const side = n.labelSide || "bottom";
    if (side === "bottom") {
      ctx.textBaseline = "top";
      const words = n.label.split(" ");
      let ly = y + h + 7;
      if (words.length > 1 && ctx.measureText(n.label).width > w + 34) {
        words.forEach((wd) => {
          ctx.fillText(wd, n.x, ly);
          ly += 11;
        });
      } else {
        ctx.fillText(n.label, n.x, ly);
        ly += 11;
      }
      if (o.lit && n.sub) {
        ctx.font = '500 7.5px "JetBrains Mono",monospace';
        ctx.fillStyle = A;
        ctx.fillText(n.sub, n.x, ly);
      }
    } else if (side === "top") {
      ctx.textBaseline = "alphabetic";
      const hw = ctx.measureText(n.label).width / 2;
      ctx.fillText(n.label, Math.max(hw + 4, Math.min(this.W - hw - 4, n.x)), y - 8);
    } else {
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(n.label, x + w + 9, n.y);
    }
    ctx.restore();
  }

  private drawIcon(
    ctx: CanvasRenderingContext2D,
    k: string,
    x: number,
    y: number,
    col: string,
  ) {
    ctx.save();
    ctx.translate(x, y);
    ctx.strokeStyle = col;
    ctx.fillStyle = col;
    ctx.lineWidth = 1.6;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (k === "query") {
      ctx.beginPath();
      ctx.arc(0, -4, 3.4, 0, 7);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 8, 6.6, Math.PI * 1.08, Math.PI * 1.92);
      ctx.stroke();
    } else if (k === "embed") {
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) {
          ctx.beginPath();
          ctx.arc(-5 + j * 5, -5 + i * 5, 1.2, 0, 7);
          ctx.fill();
        }
    } else if (k === "doc") {
      this.rr(ctx, -6, -8, 12, 16, 2);
      ctx.stroke();
      for (let l = 0; l < 3; l++) {
        ctx.beginPath();
        ctx.moveTo(-3, -4 + l * 4);
        ctx.lineTo(3, -4 + l * 4);
        ctx.stroke();
      }
    } else if (k === "prompt") {
      this.rr(ctx, -7, -7, 14, 11, 3);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(-2, 4);
      ctx.lineTo(-4.5, 8);
      ctx.lineTo(1.5, 4);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(-4, -4);
      ctx.lineTo(4, -4);
      ctx.moveTo(-4, -1);
      ctx.lineTo(2, -1);
      ctx.stroke();
    } else if (k === "llm") {
      ctx.beginPath();
      for (let i = 0; i <= 8; i++) {
        const a = (i * Math.PI) / 4 - Math.PI / 2;
        const rr2 = i % 2 ? 3 : 6.5;
        const px = Math.cos(a) * rr2;
        const py = Math.sin(a) * rr2;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(0, 0, 1.3, 0, 7);
      ctx.fill();
    }
    ctx.restore();
  }

  private drawOutputTag(ctx: CanvasRenderingContext2D, L: Diagram, A: string, lit: boolean) {
    const T = this.T;
    const y = L.nodes.llm.y;
    ctx.save();
    ctx.font = '600 10px "JetBrains Mono",monospace';
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    ctx.fillStyle = lit ? A : T.faint;
    ctx.fillText("OUTPUT ›", this.W - 8, y);
    ctx.restore();
  }

  private drawIndexing(ctx: CanvasRenderingContext2D, A: string, P: number, dc: Pt) {
    const T = this.T;
    const w = 38;
    const h = 48;
    const appear = this.easeOut(this.cl(P / 0.12));
    const fade = 1 - this.cl((P - 0.44) / 0.18);
    if (fade <= 0) return;
    ctx.save();
    // pages fanning out (split moment)
    const fan = this.cl((P - 0.2) / 0.18);
    const pFade = this.cl((P - 0.18) / 0.08) * (1 - this.cl((P - 0.4) / 0.14));
    if (pFade > 0) {
      ctx.globalAlpha = pFade * fade;
      for (let i = 0; i < 3; i++) {
        const fi = i - 1;
        const px = dc.x + fi * 34 * fan;
        const py = dc.y - 6 * fan + Math.abs(fi) * 3;
        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(fi * 0.12 * fan);
        this.rr(ctx, -13, -17, 26, 34, 3);
        ctx.fillStyle = T.boxFill;
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = T.nodeStroke;
        ctx.stroke();
        ctx.fillStyle = T.hair;
        for (let l = 0; l < 4; l++) {
          this.rr(ctx, -9, -12 + l * 5, l === 3 ? 11 : 18, 1.5, 1);
          ctx.fill();
        }
        ctx.restore();
      }
    }
    // central document with downward embedding scan
    ctx.globalAlpha = fade * appear;
    ctx.save();
    ctx.translate(dc.x, dc.y);
    ctx.scale(0.82 + 0.18 * appear, 0.82 + 0.18 * appear);
    ctx.shadowColor = this.rgba(A, 0.4);
    ctx.shadowBlur = 22;
    this.rr(ctx, -w / 2, -h / 2, w, h, 5);
    ctx.fillStyle = T.boxFill;
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = A;
    ctx.stroke();
    ctx.fillStyle = this.rgba(A, 0.42);
    for (let l = 0; l < 7; l++) {
      this.rr(ctx, -13, -18 + l * 5, l === 6 ? 15 : 26, 1.8, 1);
      ctx.fill();
    }
    const scan = this.cl((P - 0.08) / 0.3);
    if (scan > 0 && scan < 1) {
      const sy = -h / 2 + 2 + scan * (h - 4);
      ctx.strokeStyle = A;
      ctx.lineWidth = 2;
      ctx.shadowColor = this.rgba(A, 0.7);
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.moveTo(-w / 2 + 2, sy);
      ctx.lineTo(w / 2 - 2, sy);
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
    ctx.restore();
    // label
    ctx.globalAlpha = fade;
    ctx.font = '600 8.5px "JetBrains Mono",monospace';
    ctx.fillStyle = A;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("attention.pdf", dc.x, dc.y + h * 0.5 * appear + 7);
    ctx.restore();
  }
}
