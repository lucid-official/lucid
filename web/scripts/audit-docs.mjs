/**
 * Docs contract audit — exhaustive conformance check for the built API site.
 *
 *   pnpm build && node scripts/audit-docs.mjs
 *
 * Crawls every generated page in ``out/`` (the complete page set), cross-refs
 * the ``public/api-data`` JSON, and checks each COMPONENT TYPE against a fixed
 * CONTRACT (a set of invariants).  Every deviation is classified as:
 *   - conforming            (passes)
 *   - registered-exception  (matches a declarative waiver in audit/exceptions)
 *   - VIOLATION             (unregistered → fails the audit, exit 1)
 *
 * Plus DEAD-WAIVER detection (registry entries that never matched) so the
 * exception list can't rot.  This turns the design rules in
 * obsidian/architecture/arch-docs-site-rules.md into executable, regression-
 * proof gates — the production-level alternative to ad-hoc spot-fixing.
 */

import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, relative } from "node:path";
import { parse } from "node-html-parser";
import { WAIVERS, matchWaiver } from "./audit/exceptions.mjs";

const ROOT = process.cwd();                       // run from web/
const OUT = join(ROOT, "out");
const API_DATA = join(ROOT, "public", "api-data");
const LABELS_TS = join(ROOT, "src", "lib", "labels.ts");
const API_SECTIONS_JSON = join(ROOT, "data", "api-sections.json");
const BASE = "/lucid";

// ── reST / markdown leak patterns (unambiguous — used by prose contracts) ────
const REST_LEAKS = [
  { name: "cross-ref role", re: /:(?:class|func|meth|mod|attr|obj|data|exc|ref|file|cite|term|doc|envvar|samp|command|option|paramref|py:[a-z]+):`?/ },
  { name: "directive", re: /\.\.[ \t]+[a-z][a-z-]*::/ },
  { name: "reST hyperlink", re: /<https?:\/\/[^>]+>`_/ },
  { name: "double-backtick", re: /``[^`]/ },
];
// Inline math that escaped KaTeX: a $…\cmd…$ visible as text.
const RAW_MATH = /\$[^$\n]{0,200}?\\[a-zA-Z]+[^$\n]{0,200}?\$/;

// ───────────────────────────── helpers ──────────────────────────────────────

function walkHtml(dir, acc = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walkHtml(p, acc);
    else if (name.endsWith(".html")) acc.push(p);
  }
  return acc;
}

/** out/api/lucid.fft.html → /api/lucid.fft ; out/index.html → / */
function fileToRoute(file) {
  let r = "/" + relative(OUT, file).replace(/\\/g, "/");
  r = r.replace(/\/index\.html$/, "").replace(/\.html$/, "");
  return r === "" ? "/" : r;
}

/** /lucid/api/lucid.fft#x?y → /api/lucid.fft (or null if not base-prefixed) */
function linkToRoute(href) {
  let h = href.split("#")[0].split("?")[0];
  if (h === BASE) return "/";
  if (!h.startsWith(BASE + "/")) return null;
  h = h.slice(BASE.length);
  if (h.length > 1 && h.endsWith("/")) h = h.slice(0, -1);
  return h;
}

const SKIP_HREF = (h) =>
  !h ||
  /^(https?:|mailto:|#|\/\/)/.test(h) ||
  h.startsWith("/_next") ||
  h.startsWith(BASE + "/_next") ||
  /\.(svg|png|jpe?g|ico|xml|txt|json|css|js|woff2?|webmanifest|bib)$/.test(h);

function stripScripts(html) {
  // Until stable: removing one block can join the halves of another.
  let prev;
  do {
    prev = html;
    html = html.replace(/<script\b[^>]*>[\s\S]*?<\/script\b[^>]*>/gi, "");
  } while (html !== prev);
  return html;
}

function loadJson(p) {
  try { return JSON.parse(readFileSync(p, "utf8")); } catch { return null; }
}

// ───────────────────────── load corpus ──────────────────────────────────────

if (!existsSync(OUT)) {
  console.error("✗ out/ not found — run `pnpm build` first.");
  process.exit(2);
}

const files = walkHtml(OUT);
const routeSet = new Set(files.map(fileToRoute));
// Case-insensitive view: macOS/APFS collides e.g. AlexNet.html / alexnet.html
// (model class vs factory) into one file, so the class route looks "missing"
// locally though both exist on case-sensitive CI.  Resolve case-insensitively
// — Next generates pages and links from the same member names, so a real case
// mismatch is impossible by construction.
const routeSetLower = new Set([...routeSet].map((r) => r.toLowerCase()));

const apiBySlug = new Map();      // slug → module JSON
for (const f of readdirSync(API_DATA)) {
  if (!f.endsWith(".json") || f.startsWith("_")) continue;
  const d = loadJson(join(API_DATA, f));
  if (d) apiBySlug.set(f.slice(0, -5), d);
}

// PACKAGE_LABELS keys declared in labels.ts (for the label.coverage contract).
const labelKeys = new Set();
if (existsSync(LABELS_TS)) {
  const src = readFileSync(LABELS_TS, "utf8");
  const blk = src.slice(src.indexOf("const PACKAGE_LABELS"));
  for (const m of blk.matchAll(/"(lucid[\w.]*)":/g)) labelKeys.add(m[1]);
}
// Packages grouped in data/api-sections.json are labelled there (labels.ts
// spreads them into PACKAGE_LABELS), so their slugs count as covered.
const sectionsJson = loadJson(API_SECTIONS_JSON);
for (const parent of Object.values(sectionsJson ?? {}))
  for (const section of parent.sections ?? [])
    for (const pkg of section.packages ?? []) labelKeys.add(pkg.slug);

const isFamilyLeaf = (slug) => {
  const p = slug.split(".");
  return p.length === 4 && p[0] === "lucid" && p[1] === "models" &&
    ["vision", "text", "generative", "multimodal"].includes(p[2]);
};

// A model FACTORY returns a model instance and (when summarizable) carries a
// ``model_summary``.  A config-BUILDER such as ``efficientdet_config`` is also
// a module-level function but returns a ``*Config`` and never has a summary by
// design — the summary-coverage / summary-gaps contracts must count only true
// factories, else every family's config helper reads as a missing card.
const isModelFactory = (m) =>
  m.kind === "function" && !/Config$/.test((m.returns?.annotation || "").trim());

const global = { files, routeSet, apiBySlug, labelKeys };

// ─────────────────────────── contracts ──────────────────────────────────────
//
// Each contract: { id, component, kind: "page"|"global", applies?(ctx),
//   check(ctxOrGlobal) → [{ scope, detail }] }.  ``scope`` is the slug/route a
// waiver matches against.

// ── UI design-language schema (the design language AS DATA) ──────────────────
// Each entry: a CSS selector + the canonical token classes its matches MUST
// carry (`all`), optionally MUST NOT (`none`), plus an any-of regex (`any`).
// One generic checker turns this table into per-id contracts, so the visual
// spec stays a declarative table rather than 15 hand-written functions.  Recipes
// reverse-engineered from the built pages; prose in arch-docs-ui-design-language.
function classSet(el) {
  return new Set((el.getAttribute("class") || "").split(/\s+/).filter(Boolean));
}
const UI_CLASS_SCHEMA = [
  { id: "ui.card.shape", sel: "a.group.block",
    all: ["rounded-xl", "border-lucid-border", "bg-lucid-surface/40", "transition-colors", "hover:bg-lucid-surface", "hover:border-lucid-primary/40"] },
  { id: "ui.member-card.shape", sel: "a.compact-card",
    all: ["group", "flex", "items-start", "justify-between", "rounded-xl", "border-lucid-border", "bg-lucid-surface", "px-4", "py-3.5"] },
  { id: "ui.member-card.hover-kind", sel: "a.compact-card",
    all: ["hover:bg-lucid-elevated"], any: /^hover:border-api-[a-z-]+\/40$/ },
  { id: "ui.section-heading.recipe", sel: 'section[id^="section-"] > h2',
    all: ["text-xs", "font-semibold", "tracking-widest", "uppercase", "text-lucid-text-disabled"] },
  { id: "ui.family-card.title", sel: "a.group.block header h3",
    all: ["font-mono", "text-lg", "font-semibold", "text-api-class"] },
  { id: "ui.type-link.style", sel: "a.decoration-dotted",
    all: ["text-lucid-primary-light", "underline-offset-2"] },
  { id: "ui.seealso.list", sel: 'ul[class*="border-lucid-primary/40"]',
    all: ["rounded-xl", "border-l-2", "bg-lucid-primary/5", "space-y-1.5"] },
];
const uiContracts = UI_CLASS_SCHEMA.map((s) => ({
  id: s.id, component: "UI", kind: "page", severity: s.sev || "error",
  check(ctx) {
    const out = [];
    let els;
    try { els = ctx.dom.querySelectorAll(s.sel); } catch { return out; }
    for (const el of els) {
      const cs = classSet(el);
      const missing = (s.all || []).filter((c) => !cs.has(c));
      const forbidden = (s.none || []).filter((c) => cs.has(c));
      const anyOk = !s.any || [...cs].some((c) => s.any.test(c));
      if (missing.length || forbidden.length || !anyOk)
        out.push({ scope: ctx.scope, detail: `${s.sel}: ${[missing.length && "missing " + missing.join(" "), forbidden.length && "forbidden " + forbidden.join(" "), !anyOk && "need " + s.any].filter(Boolean).join("; ")}` });
    }
    return out;
  },
}));

const PAGE_CONTRACTS = [
  // ── Links (every page) ─────────────────────────────────────────────────────
  {
    id: "link.basepath", component: "InternalLink", kind: "page",
    check(ctx) {
      const bad = [];
      for (const a of ctx.dom.querySelectorAll("a")) {
        const h = a.getAttribute("href");
        if (SKIP_HREF(h) || !h.startsWith("/")) continue;
        if (h !== BASE && !h.startsWith(BASE + "/"))
          bad.push({ scope: ctx.scope, detail: `<a href="${h}"> missing ${BASE} basePath` });
      }
      return bad;
    },
  },
  {
    id: "link.resolves", component: "InternalLink", kind: "page",
    check(ctx) {
      const bad = [];
      for (const a of ctx.dom.querySelectorAll("a")) {
        const h = a.getAttribute("href");
        if (SKIP_HREF(h) || !h.startsWith(BASE)) continue;
        const r = linkToRoute(h);
        if (r && !routeSet.has(r) && !routeSetLower.has(r.toLowerCase()))
          bad.push({ scope: ctx.scope, detail: `link → ${h} has no generated page` });
      }
      return bad;
    },
  },
  // ── Prose / rendering (every page) ─────────────────────────────────────────
  {
    id: "prose.no-visible-rest", component: "ProseField", kind: "page",
    check(ctx) {
      const out = [];
      for (const { name, re } of REST_LEAKS) {
        const m = ctx.visible.match(re);
        if (m) out.push({ scope: ctx.scope, detail: `${name} leaked: "${m[0].slice(0, 40)}"` });
      }
      return out;
    },
  },
  {
    id: "math.no-raw-leak", component: "MathBlock", kind: "page",
    check(ctx) {
      const m = ctx.visible.match(RAW_MATH);
      return m ? [{ scope: ctx.scope, detail: `unrendered math: "${m[0].slice(0, 40)}"` }] : [];
    },
  },
  {
    id: "doctest.not-blockquote", component: "ExamplesBlock", kind: "page",
    check(ctx) {
      const bad = ctx.dom.querySelectorAll("blockquote").some((b) => b.text.includes(">>>"));
      return bad ? [{ scope: ctx.scope, detail: "doctest >>> rendered as <blockquote> (should be a code fence)" }] : [];
    },
  },
  // ── Module overview page ───────────────────────────────────────────────────
  {
    id: "module.has-heading", component: "ModulePage", kind: "page",
    applies: (ctx) => ctx.kind === "api-module",
    check(ctx) {
      return ctx.dom.querySelector("h1") ? [] : [{ scope: ctx.scope, detail: "no <h1> module heading" }];
    },
  },
  {
    id: "card.well-formed", component: "SubpackageCard", kind: "page",
    applies: (ctx) => ctx.kind === "api-module",
    check(ctx) {
      const out = [];
      const secs = ctx.dom.querySelectorAll("section").filter((s) =>
        /^(Model Families|Sub-packages)$/.test(s.querySelector("h2")?.text?.trim() || ""));
      for (const sec of secs) {
        for (const card of sec.querySelectorAll("a")) {
          const title = card.querySelector("h3")?.text?.trim();
          if (!title) out.push({ scope: ctx.scope, detail: `sub-package card with empty title (href=${card.getAttribute("href")})` });
        }
      }
      return out;
    },
  },
  {
    id: "ui.main-column.full-width", component: "DetailLayout", kind: "page",
    applies: (ctx) => ctx.kind === "api-member" || ctx.kind === "api-module",
    check(ctx) {
      // Two-column detail/overview layout (main content column + ToC aside):
      // the content column is a flex item, so without a grow rule it shrinks to
      // its content and short pages render a NARROW card.  Enforce that the main
      // column always fills the available width up to its ``max-w`` cap
      // (``flex-1`` / ``w-full``) — every main card is a fixed full width
      // regardless of content length.  Single-column pages (no ToC, e.g. the
      // C++ engine landing) have no ``xl:flex-row`` row and are skipped: a block
      // element already fills its parent there.
      const row = ctx.dom.querySelectorAll("div").find((d) => d.classList.contains("xl:flex-row"));
      if (!row) return [];
      const col = row.querySelector(".max-w-4xl");
      if (!col) return [{ scope: ctx.scope, detail: "two-column layout has no max-w-4xl main column" }];
      const fills =
        col.classList.contains("xl:flex-1") ||
        col.classList.contains("flex-1") ||
        col.classList.contains("w-full");
      return fills
        ? []
        : [{ scope: ctx.scope, detail: "main content column lacks flex-1/w-full — card shrinks to content instead of a fixed full width" }];
    },
  },
  // ── UI design language (rendered-class conformance) ────────────────────────
  // Made executable as a DECLARATIVE SCHEMA (UI_CLASS_SCHEMA above) plus the
  // custom checks below: every component must carry its canonical token recipe
  // so the visual language stays uniform.  Drift (a card with rounded-md, an
  // off-recipe heading, an off-palette text-gray-400) fails the build instead
  // of relying on code review.  Full prose: arch-docs-ui-design-language.
  ...uiContracts,
  {
    id: "ui.classes-before-functions", component: "ModulePage", kind: "page",
    applies: (ctx) => ctx.kind === "api-module",
    check(ctx) {
      const ids = ctx.dom.querySelectorAll('section[id^="section-"]').map((s) => s.getAttribute("id"));
      const c = ids.indexOf("section-classes"), f = ids.indexOf("section-functions");
      return c !== -1 && f !== -1 && c > f
        ? [{ scope: ctx.scope, detail: "Functions section precedes Classes (Classes must render first)" }] : [];
    },
  },
  {
    id: "ui.kind-pill.color-trio", component: "ApiKindBadge", kind: "page",
    check(ctx) {
      // A kind pill's solid text token (text-api-X) must be backed by the
      // matching bg-api-X/10 + border-api-X/{30,35,40} (one hue per pill).
      const out = [];
      for (const el of ctx.dom.querySelectorAll('span[class*="text-api-"]')) {
        const cs = classSet(el);
        if (!cs.has("select-none")) continue;   // scope to actual kind pills, not text-coloured spans
        const tok = [...cs].find((c) => /^text-api-[a-z0-9-]+$/.test(c));  // solid only (no /opacity)
        if (!tok) continue;
        const kind = tok.slice("text-api-".length);
        const bgOk = [...cs].some((c) => new RegExp(`^bg-api-${kind}/(8|10)$`).test(c));
        const borderOk = [...cs].some((c) => new RegExp(`^border-api-${kind}/(30|35|40)$`).test(c));
        if (!bgOk || !borderOk)
          out.push({ scope: ctx.scope, detail: `kind pill text-api-${kind} lacks matching bg-api-${kind}/{8,10} or border-api-${kind}/{30,35,40}` });
      }
      return out;
    },
  },
  {
    id: "ui.uppercase-tracking", component: "DesignToken", kind: "page",
    check(ctx) {
      // Every all-caps element pairs with a tracking utility (typographic rule).
      for (const el of ctx.dom.querySelectorAll('[class*="uppercase"]')) {
        const cs = classSet(el);
        // Any tracking-* utility counts (incl. arbitrary tracking-[0.12em]).
        if (cs.has("uppercase") && ![...cs].some((c) => c.startsWith("tracking-")))
          return [{ scope: ctx.scope, detail: `uppercase without tracking-*: "${(el.getAttribute("class") || "").slice(0, 60)}"` }];
      }
      return [];
    },
  },
  {
    id: "ui.palette", component: "DesignToken", kind: "page",
    check(ctx) {
      // No off-design-system colours: only lucid-* / api-* / task-* tokens are
      // allowed.  Catches drift to stock Tailwind palettes or arbitrary hex.
      const m = ctx.mainHtml.match(/\b(?:text|bg|border|ring|fill|stroke|decoration)-(?:gray|slate|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b/)
        || ctx.mainHtml.match(/\b(?:text|bg|border)-\[#[0-9a-fA-F]{3,8}\]/);
      return m ? [{ scope: ctx.scope, detail: `off-palette colour class: ${m[0]}` }] : [];
    },
  },
  {
    id: "ui.weights-section", component: "ModulePage", kind: "page",
    applies: (ctx) => ctx.kind === "api-module",
    check(ctx) {
      // Pretrained-weight enums (``<Family>Weights``) render under a dedicated
      // "Weights" section, never mixed into "Classes".  Model-zoo-wide rule:
      // the lucid.models.weights aggregator is not a package — each family owns
      // its weight classes and surfaces them in its own Weights slot/section.
      const classesSec = ctx.dom.querySelectorAll("section")
        .find((s) => s.querySelector("h2")?.text?.trim() === "Classes");
      if (!classesSec) return [];
      const out = [];
      for (const a of classesSec.querySelectorAll("a")) {
        const href = (a.getAttribute("href") || "").split("#")[0];
        if (href.endsWith("Weights"))
          out.push({ scope: ctx.scope, detail: `weight class in "Classes" (must be a "Weights" section): ${href}` });
      }
      return out;
    },
  },
  {
    id: "model.size-card-rendered", component: "ModelSizeCard", kind: "page",
    applies: (ctx) =>
      ctx.kind === "api-member" && isFamilyLeaf(ctx.scope.split("/")[0]),
    check(ctx) {
      // A model factory whose api-data carries a ``model_summary`` MUST render
      // the expandable "Model Size" card (param count + collapsible layer tree)
      // on its page.  Catches a broken data→render path; the missing-summary
      // case is tracked by ``model.summary-gaps`` instead.
      const member = (ctx.data?.members || []).find((m) => m.name === ctx.memberName);
      if (!member?.model_summary) return [];
      // Skip when this local file is a case-collision clobber (see
      // ``pageMatchesMember``) — the rendered page is the sibling class, not
      // this factory.  The model_summary data invariant above is still
      // verified, and case-sensitive CI checks the real render.
      if (!ctx.pageMatchesMember) return [];
      return /Model Size/.test(ctx.visible)
        ? []
        : [{ scope: ctx.scope, detail: "has model_summary but no 'Model Size' card rendered" }];
    },
  },
];

const GLOBAL_CONTRACTS = [
  {
    id: "models.weights-not-a-package", component: "ModulePage", kind: "global",
    check(g) {
      // The per-family weight-enum aggregator must NOT be a documented package
      // (excluded in build-api-data); each family owns its <Family>Weights.
      return g.apiBySlug.has("lucid.models.weights")
        ? [{ scope: "lucid.models.weights", detail: "weights aggregator is documented as a package — must be excluded (each family owns its Weights)" }]
        : [];
    },
  },
  {
    id: "model.summary-coverage", component: "ModelSizeCard", kind: "global",
    check(g) {
      // Guard against a wiped _summaries.json cache (the regression that made
      // every variant's param/structure card vanish): the share of model-family
      // factories carrying a model_summary must stay above a floor.
      let total = 0, withSummary = 0;
      for (const [slug, d] of g.apiBySlug) {
        if (!isFamilyLeaf(slug)) continue;
        for (const m of d.members || [])
          if (isModelFactory(m)) { total++; if (m.model_summary) withSummary++; }
      }
      if (total === 0) return [];
      const pct = Math.round((withSummary / total) * 100);
      return withSummary / total < 0.5
        ? [{ scope: "lucid.models", detail: `model_summary coverage ${pct}% (${withSummary}/${total}) < 50% — likely a wiped _summaries.json cache` }]
        : [];
    },
  },
  {
    id: "model.summary-gaps", component: "ModelSizeCard", kind: "global", severity: "warn",
    check(g) {
      // Advisory: model factories with no computed model_summary (run
      // `python -m tools.build_model_summaries` to fill — needs the engine).
      const out = [];
      for (const [slug, d] of g.apiBySlug) {
        if (!isFamilyLeaf(slug)) continue;
        for (const m of d.members || [])
          if (isModelFactory(m) && !m.model_summary)
            out.push({ scope: `${slug}/${m.name}`, detail: "factory has no model_summary (no param/structure card)" });
      }
      return out;
    },
  },
  {
    id: "design.no-raw-hex", component: "DesignToken", kind: "global",
    check() {
      // Components reference colour only via tokens — never a raw hex literal.
      // (globals.css IS the token SSOT, so it is excluded.)
      const out = [];
      const walk = (dir) => {
        for (const n of readdirSync(dir)) {
          const p = join(dir, n);
          if (statSync(p).isDirectory()) walk(p);
          else if (n.endsWith(".tsx")) {
            const m = readFileSync(p, "utf8").match(/#[0-9a-fA-F]{6}\b/);
            if (m) out.push({ scope: relative(ROOT, p), detail: `raw hex colour ${m[0]} (use a lucid-/api- token)` });
          }
        }
      };
      try { walk(join(ROOT, "src")); } catch { /* no src */ }
      return out;
    },
  },
  {
    id: "module.has-summary", component: "ModulePage", kind: "global", severity: "warn",
    check(g) {
      const out = [];
      for (const [slug, d] of g.apiBySlug)
        if (d.kind === "module" && !(d.summary && d.summary.trim()))
          out.push({ scope: slug, detail: `${slug}: module has no summary` });
      return out;
    },
  },
  {
    id: "member.has-signature", component: "MemberPage", kind: "global",
    check(g) {
      const out = [];
      for (const [slug, d] of g.apiBySlug) {
        if (d.kind !== "module") continue;
        for (const m of d.members || [])
          if ((m.kind === "function" || m.kind === "class") && !m.signature && !(m.methods))
            out.push({ scope: `${slug}/${m.name}`, detail: `${slug}.${m.name}: no signature` });
      }
      return out;
    },
  },
  {
    id: "prose.no-rest-leak", component: "ProseField", kind: "global",
    check(g) {
      const out = [];
      const fields = ["summary", "extended", "theory", "citation"];
      const scan = (slug, s) => {
        if (typeof s !== "string") return;
        // Ignore protected regions — fenced code (```…```) and math ($$…$$ / $…$)
        // legitimately contain `` (LaTeX left-quotes) and \commands.
        const prose = s
          .replace(/```[\s\S]*?```/g, "")
          .replace(/\$\$[\s\S]*?\$\$/g, "")
          .replace(/\$[^$\n]+?\$/g, "");
        for (const { name, re } of REST_LEAKS)
          if (re.test(prose)) out.push({ scope: slug, detail: `${slug}: ${name} in prose` });
      };
      const walk = (slug, o) => {
        if (Array.isArray(o)) o.forEach((x) => walk(slug, x));
        else if (o && typeof o === "object")
          for (const [k, v] of Object.entries(o)) {
            if (fields.includes(k) || k === "description") scan(slug, v);
            else if (k === "notes" || k === "warns") (v || []).forEach((x) => scan(slug, x));
            else walk(slug, v);
          }
      };
      for (const [slug, d] of g.apiBySlug) walk(slug, d);
      return out;
    },
  },
  {
    id: "label.coverage", component: "SidebarEntry", kind: "global",
    check(g) {
      const out = [];
      for (const slug of g.apiBySlug.keys()) {
        if (isFamilyLeaf(slug)) continue;        // family leaves use canonical_name
        if (slug === "lucid") continue;
        if (!g.labelKeys.has(slug))
          out.push({ scope: slug, detail: `${slug}: no PACKAGE_LABELS alias (add to labels.ts or waive)` });
      }
      return out;
    },
  },
  {
    id: "sidebar.complete", component: "SidebarEntry", kind: "global",
    check(g) {
      // Use the /api landing page's sidebar as the canonical rendering.
      const landing = join(OUT, "api.html");
      const idx = join(OUT, "api", "index.html");
      const file = existsSync(landing) ? landing : existsSync(idx) ? idx : null;
      if (!file) return [{ scope: "/api", detail: "cannot locate /api page to read sidebar" }];
      const dom = parse(readFileSync(file, "utf8"));
      const hrefs = new Set(dom.querySelectorAll("a").map((a) => a.getAttribute("href")));
      const out = [];
      for (const slug of g.apiBySlug.keys()) {
        const depth = slug.split(".").length;
        if (depth !== 2) continue;               // top-level packages must be in the sidebar
        if (slug === "lucid") continue;
        if (!hrefs.has(`${BASE}/api/${slug}`))
          out.push({ scope: slug, detail: `${slug}: top-level package not linked in sidebar` });
      }
      return out;
    },
  },
];

// ─────────────────────────── run ────────────────────────────────────────────

const hitWaivers = new Set();
const findings = [];   // { contractId, component, scope, detail, status }

function record(contract, raw) {
  const sev = contract.severity || "error";   // "error" = hard violation; "warn" = advisory
  for (const f of raw) {
    const waiver = matchWaiver(contract.id, f.scope, hitWaivers);
    const status = waiver ? "exception" : sev === "warn" ? "warning" : "violation";
    findings.push({ contractId: contract.id, component: contract.component, scope: f.scope, detail: f.detail, status, waiver });
  }
}

// Per-page contracts.
let pageCount = 0;
for (const file of files) {
  const route = fileToRoute(file);
  let kind = "other";
  let scope = route, data = null, memberName = null;
  if (route === "/") kind = "home";
  else if (route === "/api") kind = "api-landing";
  else if (route === "/changelog") kind = "changelog";
  else if (route.startsWith("/docs")) kind = "docs";
  else if (route.startsWith("/api/")) {
    const rest = route.slice(5).split("/");
    scope = rest.length === 1 ? rest[0] : `${rest[0]}/${rest.slice(1).join("/")}`;
    data = apiBySlug.get(rest[0]) || null;
    if (rest.length === 1) kind = "api-module";
    else { kind = "api-member"; memberName = rest.slice(1).join("/"); }
  }
  const dom = parse(readFileSync(file, "utf8"));
  dom.querySelectorAll("script,style,template,noscript").forEach((n) => n.remove());
  // Visible-text checks scope to <main> only: Next's RSC hydration payload
  // (serialized props) lives in <script> blocks at the end of <body>, which
  // node-html-parser doesn't always strip cleanly — scoping to the rendered
  // content area is the robust way to ignore it.
  const main = dom.querySelector("#main-content") || dom.querySelector("main") || dom;
  const ctx = { route, scope, kind, data, memberName, dom, visible: main.text, mainHtml: main.toString() };
  // macOS/APFS is case-insensitive: a factory page (``zfnet``) and its
  // case-only sibling class page (``ZFNet``) collide into ONE output file, so
  // whichever Next writes last clobbers the other — locally the file under a
  // member route may actually hold the sibling's render.  A page genuinely for
  // ``memberName`` anchors a heading element with ``id===memberName`` (see
  // FunctionSignature / ClassDoc).  When that anchor is absent the file is the
  // colliding sibling's page; on case-sensitive CI the two files are distinct
  // so this never triggers there.  Member-content contracts gate on this so a
  // local FS artifact can't masquerade as a render regression.
  ctx.pageMatchesMember =
    kind !== "api-member" || dom.getElementById(memberName) != null;
  for (const c of PAGE_CONTRACTS) {
    if (c.applies && !c.applies(ctx)) continue;
    record(c, c.check(ctx));
  }
  pageCount++;
}

// Global contracts.
for (const c of GLOBAL_CONTRACTS) record(c, c.check(global));

// ─────────────────────────── report ─────────────────────────────────────────

const violations = findings.filter((f) => f.status === "violation");
const warnings = findings.filter((f) => f.status === "warning");
const exceptions = findings.filter((f) => f.status === "exception");
const deadWaivers = WAIVERS.filter((w) => !hitWaivers.has(w.id));

const byContract = (list) => {
  const m = new Map();
  for (const f of list) m.set(f.contractId, (m.get(f.contractId) || 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1]);
};

console.log("─".repeat(72));
console.log(`Docs contract audit — ${pageCount} pages, ${apiBySlug.size} api modules`);
console.log("─".repeat(72));
console.log(`contracts: ${PAGE_CONTRACTS.length + GLOBAL_CONTRACTS.length}   waivers: ${WAIVERS.length}`);
console.log(`exceptions: ${exceptions.length}   warnings: ${warnings.length}   VIOLATIONS: ${violations.length}\n`);

if (exceptions.length) {
  console.log("Registered exceptions (waived):");
  for (const [c, n] of byContract(exceptions)) console.log(`  · ${c}: ${n}`);
  console.log();
}

if (warnings.length) {
  console.log("⚠️  Warnings (advisory — do not fail the audit):");
  for (const [c, n] of byContract(warnings)) {
    console.log(`  · ${c}: ${n}`);
    for (const w of warnings.filter((x) => x.contractId === c).slice(0, 10))
      console.log(`        ${w.scope}: ${w.detail}`);
  }
  console.log();
}

if (deadWaivers.length) {
  console.log("⚠️  DEAD waivers (registered but never matched — remove or fix scope):");
  for (const w of deadWaivers) console.log(`  · ${w.id}`);
  console.log();
}

if (violations.length) {
  console.log("✗ VIOLATIONS:");
  for (const [c, n] of byContract(violations)) {
    console.log(`\n  [${c}] — ${n}`);
    const sample = violations.filter((v) => v.contractId === c).slice(0, 12);
    for (const v of sample) console.log(`      ${v.scope}: ${v.detail}`);
    if (n > sample.length) console.log(`      … +${n - sample.length} more`);
  }
  console.log(`\n✗ AUDIT FAILED — ${violations.length} unregistered violation(s).`);
  process.exit(1);
}

console.log("✓ AUDIT PASSED — every component conforms (or is a registered exception).");
process.exit(0);
