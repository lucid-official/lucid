import type { Metadata } from "next";
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  Binary,
  Boxes,
  BrainCircuit,
  Cpu,
  Database,
  Dices,
  FunctionSquare,
  Gauge,
  GitBranch,
  Layers,
  Network,
  Puzzle,
  Rocket,
  Save,
  Shuffle,
  Sigma,
  SlidersHorizontal,
  Sparkles,
  Spline,
  Star,
  Timer,
  TrendingDown,
  Type,
  Waves,
  Wrench,
  Workflow,
} from "lucide-react";
import { FadeIn, FadeInStagger } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/Card";
import { loadApiData } from "@/lib/api-loader";
import { isApiModule, isApiClassModule } from "@/lib/types";
import { canonical } from "@/lib/site";
import { API_SECTIONS, sectionedSlugs } from "@/lib/api-sections";

export const metadata: Metadata = {
  title: "API Reference",
  description: "Complete API reference for the Lucid ML framework.",
  alternates: canonical("/api"),
};

interface ModuleCard {
  name: string;
  slug: string;          // for href; "/api/<slug>"
  description: string;
  icon: LucideIcon;
  badge?: string;
  detail?: string;       // full text for the tooltip when `description` is shortened
}

interface ModuleGroup {
  category: string;
  modules: ModuleCard[];
}

// Build-time count of a module's emitted members.  Single source of
// truth — every blurb / badge below interpolates from this so the page
// stays in sync as the API surface grows.  Skipped (returns 0) when the
// JSON is missing.
function _memberCount(slug: string): number {
  try {
    const d = loadApiData(slug);
    if (isApiClassModule(d)) return d.methods.length;
    // Module-level constants documented in the docstring's ``Attributes``
    // section are public names on the page too, so they count — otherwise a
    // module that exports mostly constants advertises a misleadingly low
    // number.  No-op for modules that document none.
    if (isApiModule(d))       return d.members.length + d.attributes.length;
    return 0;
  } catch {
    return 0;
  }
}

// `lucid.models` doesn't expose families as direct members — they live as
// ``family_groups`` entries inside each task-category JSON.
function _modelFamilyCount(): number {
  let total = 0;
  for (const cat of ["vision", "text", "generative", "multimodal"]) {
    try {
      const d = loadApiData(`lucid.models.${cat}`);
      if (isApiModule(d) && d.family_groups) {
        total += d.family_groups.length;
      }
    } catch {
      // category missing — skip
    }
  }
  return total;
}

// `lucid.optim` houses both optimisers and LR schedulers — partition by
// subcategory the build attaches to each class.
function _optimCounts(): { optimizers: number; lrSchedulers: number } {
  let optimizers = 0;
  let lrSchedulers = 0;
  try {
    const d = loadApiData("lucid.optim");
    if (isApiModule(d)) {
      for (const m of d.members) {
        if (m.kind !== "class") continue;
        const sub = (m.subcategory || "").toLowerCase();
        if (sub.includes("lr_scheduler") || sub.includes("scheduler")) {
          lrSchedulers += 1;
        } else {
          optimizers += 1;
        }
      }
    }
  } catch {
    // optim file missing — leave at 0
  }
  return { optimizers, lrSchedulers };
}

// ``lucid.peft`` documents its tuner configs, models and layers on one page per
// method (``lucid.peft.tuners.<method>``), so its public names are spread over
// the package page and those pages — count them all, or the card undersells it.
function _peftCounts(): { methods: number; names: number } {
  const tuners = (API_SECTIONS["lucid.peft"]?.sections ?? []).flatMap(
    (s) => s.packages ?? [],
  );
  const methods = tuners.filter((p) => _memberCount(p.slug) > 0).length;
  const names = sectionedSlugs().reduce((n, slug) => n + _memberCount(slug), 0);
  return { methods, names: names + _memberCount("lucid.peft") };
}

// Count badge shared by every card.  ``undefined`` (no badge) when the count is
// 0 — the card keeps an empty badge slot so titles still line up.
function _badge(count: number, unit: string): string | undefined {
  return count > 0 ? `${count} ${unit}` : undefined;
}

function buildModuleGroups(): ModuleGroup[] {
  const tensorMethods = _memberCount("lucid.tensor");
  const nnClasses     = _memberCount("lucid.nn");
  const nnFn          = _memberCount("lucid.nn.functional");
  const nnInit        = _memberCount("lucid.nn.init");
  const families      = _modelFamilyCount();
  const { optimizers, lrSchedulers } = _optimCounts();
  const linalg        = _memberCount("lucid.linalg");
  const fft           = _memberCount("lucid.fft");
  const special       = _memberCount("lucid.special");
  const signal        = _memberCount("lucid.signal");
  const distributions = _memberCount("lucid.distributions");
  const diffeq        = _memberCount("lucid.diffeq");
  const coreml        = _memberCount("lucid.coreml");
  const peft          = _peftCounts();
  const engineMembers = _memberCount("lucid._C.engine");
  const names = (slug: string) => _badge(_memberCount(slug), "names");

  return [
    {
      category: "Core",
      modules: [
        { name: "lucid.Tensor", slug: "lucid.tensor", description: `Primary tensor class — ${tensorMethods} methods and properties.`, icon: Boxes, badge: _badge(tensorMethods, "methods") },
      ],
    },
    {
      category: "Neural Networks",
      modules: [
        { name: "lucid.nn",            slug: "lucid.nn",            description: `Module, Parameter, and ${nnClasses} layer classes.`, icon: Network,        badge: _badge(nnClasses, "modules") },
        { name: "lucid.nn.functional", slug: "lucid.nn.functional", description: "Stateless functional operations.",                   icon: FunctionSquare, badge: _badge(nnFn, "functions") },
        { name: "lucid.nn.init",       slug: "lucid.nn.init",       description: "Weight initialization strategies.",                  icon: Sparkles,       badge: _badge(nnInit, "functions") },
        { name: "lucid.nn.utils",      slug: "lucid.nn.utils",      description: "Gradient clipping, weight norm, RNN packing.",       icon: Wrench,         badge: names("lucid.nn.utils") },
      ],
    },
    {
      category: "Model Zoo",
      modules: [
        { name: "lucid.models", slug: "lucid.models", description: `${families} paper-cited vision / text / generative families with pretrained factories.`, icon: BrainCircuit, badge: _badge(families, "families") },
      ],
    },
    {
      category: "Optimization",
      modules: [
        { name: "lucid.optim", slug: "lucid.optim", description: `${optimizers} optimizers + ${lrSchedulers} learning rate schedulers.`, icon: TrendingDown, badge: _badge(optimizers + lrSchedulers, "names") },
      ],
    },
    {
      category: "Fine-Tuning",
      modules: [
        {
          name: "lucid.peft", slug: "lucid.peft", icon: Puzzle, badge: _badge(peft.names, "names"),
          description: "Parameter-efficient fine-tuning: get_peft_model, PeftModel, merge and save.",
          detail: `Parameter-efficient fine-tuning in the reference adapter library's API and file format — ${peft.methods} methods behind one entry point: get_peft_model, PeftModel, the AutoPeftModel loaders, multiple adapters, merge and save.`,
        },
        {
          name: "lucid.peft.tuners", slug: "lucid.peft.tuners", icon: Layers, badge: _badge(peft.methods, "methods"),
          description: `The ${peft.methods} methods by family, each with its config, model and layers.`,
          detail: `The ${peft.methods} methods by family — LoRA, orthogonal, LyCORIS, prompt learning, shared-table and more — each with its config, model and layers.`,
        },
      ],
    },
    {
      category: "Differentiation",
      modules: [
        { name: "lucid.autograd", slug: "lucid.autograd", description: "Function, grad, gradcheck, functional transforms.", icon: GitBranch, badge: names("lucid.autograd") },
        { name: "lucid.func",     slug: "lucid.func",     description: "vmap, grad, vjp, jvp, jacrev, jacfwd, hessian.",    icon: Workflow,  badge: names("lucid.func") },
      ],
    },
    {
      category: "Deployment",
      modules: [
        {
          name: "lucid.coreml", slug: "lucid.coreml", icon: Rocket, badge: _badge(coreml, "names"),
          description: "Export to Core ML and run on the Neural Engine.",
          detail: `Export to Core ML and run on the Neural Engine — ${coreml} names covering precision, weight compression, flexible shapes, and state.`,
        },
      ],
    },
    {
      category: "Math",
      modules: [
        { name: "lucid.linalg",        slug: "lucid.linalg",        description: "Decomposition, norm, and solve operations.",         icon: Sigma,    badge: _badge(linalg, "functions") },
        { name: "lucid.fft",           slug: "lucid.fft",           description: "DFT / Hermitian / shift / frequency functions.",     icon: Waves,    badge: _badge(fft, "functions") },
        { name: "lucid.special",       slug: "lucid.special",       description: "Special-math functions (erf, sinc, gamma, …).",      icon: Star,     badge: _badge(special, "functions") },
        { name: "lucid.signal",        slug: "lucid.signal",        description: "Spectral window functions.",                         icon: Activity, badge: _badge(signal, "functions") },
        { name: "lucid.einops",        slug: "lucid.einops",        description: "Einops-style tensor manipulation.",                  icon: Shuffle,  badge: names("lucid.einops") },
        { name: "lucid.distributions", slug: "lucid.distributions", description: "Distributions, base class, and KL registry.",        icon: Dices,    badge: _badge(distributions, "names") },
        { name: "lucid.diffeq",        slug: "lucid.diffeq",        description: "Fixed-step explicit Runge-Kutta ODE integration.",   icon: Spline,   badge: _badge(diffeq, "names") },
      ],
    },
    {
      category: "Utilities",
      modules: [
        { name: "lucid.utils.data",       slug: "lucid.utils.data",       description: "Dataset, Sampler, DataLoader.",                                             icon: Database,          badge: names("lucid.utils.data") },
        { name: "lucid.utils.cache",      slug: "lucid.utils.cache",      description: "DynamicCache, StaticCache, EncoderDecoderCache for incremental decoding.", icon: Layers,            badge: names("lucid.utils.cache") },
        { name: "lucid.utils.tokenizer",  slug: "lucid.utils.tokenizer",  description: "BPE, WordPiece, Unigram, and byte-level tokenizers.",                       icon: Type,              badge: names("lucid.utils.tokenizer") },
        { name: "lucid.utils.transforms", slug: "lucid.utils.transforms", description: "Image and tensor augmentation transforms.",                                 icon: SlidersHorizontal, badge: names("lucid.utils.transforms") },
      ],
    },
    {
      category: "Others",
      modules: [
        { name: "lucid.amp",           slug: "lucid.amp",           description: "Automatic mixed precision.",                                         icon: Gauge,  badge: names("lucid.amp") },
        { name: "lucid.quantization",  slug: "lucid.quantization",  description: "Post-training + QAT quantization (int8 / int4 weights, MLX GEMM).", icon: Binary, badge: names("lucid.quantization") },
        { name: "lucid.profiler",      slug: "lucid.profiler",      description: "Performance profiling utilities.",                                   icon: Timer,  badge: names("lucid.profiler") },
        { name: "lucid.serialization", slug: "lucid.serialization", description: "State dict save / load.",                                            icon: Save,   badge: names("lucid.serialization") },
      ],
    },
    {
      category: "Engine",
      modules: [
        { name: "lucid._C.engine", slug: "lucid._C.engine", description: "C++ compute core — storage, ops, autograd graph, backend dispatch.", icon: Cpu, badge: _badge(engineMembers, "members") },
      ],
    },
  ];
}

const ACCENT = "var(--color-lucid-primary)";

// One card rhythm for the whole page: fixed padding, 36px icon box, a 24px
// title row (name + badge slot, empty when there is no count) and a
// description clamped to two lines whose box always reserves both lines, so
// every card is the same height whatever it says.  ``h-full`` + the stretched
// grid cell make a row's cards equal; the full text rides on ``title``.
function ModuleCardLink({ mod }: { mod: ModuleCard }) {
  const Icon = mod.icon;
  const bg = `color-mix(in srgb, ${ACCENT} 14%, transparent)`;
  const ring = `color-mix(in srgb, ${ACCENT} 38%, transparent)`;
  return (
    <Card href={`/api/${mod.slug}`} className="h-full">
      <div className="h-full px-5 py-4 flex items-start gap-3">
        <span
          className="shrink-0 inline-flex h-9 w-9 items-center justify-center rounded-lg border"
          style={{ backgroundColor: bg, borderColor: ring, color: ACCENT }}
          aria-hidden
        >
          <Icon className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex h-6 items-center gap-2">
            <code
              className="truncate font-mono text-sm font-semibold"
              style={{ color: ACCENT }}
            >
              {mod.name}
            </code>
            {mod.badge && (
              <Badge variant="secondary" className="shrink-0 font-mono text-[10px]">
                {mod.badge}
              </Badge>
            )}
          </div>
          <p
            className="mt-1 min-h-[3.25em] line-clamp-2 text-xs text-lucid-text-low leading-relaxed"
            title={mod.detail ?? mod.description}
          >
            {mod.description}
          </p>
        </div>
      </div>
    </Card>
  );
}

export default function ApiPage() {
  const moduleGroups = buildModuleGroups();
  return (
    <FadeIn>
      <div className="max-w-4xl">
        <header className="mb-12">
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <span className="text-xs font-semibold tracking-widest uppercase text-lucid-text-disabled">
              API Reference
            </span>
            <h1 className="font-mono text-3xl font-bold text-lucid-text-high">
              Lucid
            </h1>
          </div>
          <p className="max-w-3xl text-base text-lucid-text-mid leading-relaxed">
            Complete reference documentation, auto-generated from Lucid's
            Python sources via{" "}
            <a
              href="https://github.com/mkdocstrings/griffe"
              className="text-lucid-primary hover:text-lucid-primary-light transition-colors"
            >
              Griffe
            </a>{" "}
            and the C++ engine via libclang.  Browse by module below, or jump
            straight into a surface from the sidebar.
          </p>
        </header>

        <div className="space-y-10">
          {moduleGroups.map(({ category, modules }) => (
            <section key={category}>
              <FadeIn>
                <h2 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase mb-3">
                  {category}
                </h2>
              </FadeIn>
              <FadeInStagger
                staggerDelay={0.04}
                // Every section is the same 2-column grid (a lone card takes one
                // column, not the row) so all cards share one width and height.
                className="grid grid-cols-1 sm:grid-cols-2 gap-3"
                itemClassName="h-full"
              >
                {modules.map((mod) => (
                  <ModuleCardLink key={mod.slug} mod={mod} />
                ))}
              </FadeInStagger>
            </section>
          ))}
        </div>
      </div>
    </FadeIn>
  );
}
