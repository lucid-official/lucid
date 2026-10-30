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
import { cn } from "@/lib/utils";
import { loadApiData } from "@/lib/api-loader";
import { isApiModule, isApiClassModule } from "@/lib/types";
import { canonical } from "@/lib/site";

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
  const peft          = _memberCount("lucid.peft");
  const engineMembers = _memberCount("lucid._C.engine");

  return [
    {
      category: "Core",
      modules: [
        { name: "lucid.Tensor", slug: "lucid.tensor", description: `Primary tensor class — ${tensorMethods} methods and properties.`, icon: Boxes, badge: `${tensorMethods} methods` },
      ],
    },
    {
      category: "Neural Networks",
      modules: [
        { name: "lucid.nn",            slug: "lucid.nn",            description: `Module, Parameter, and ${nnClasses} layer classes.`,        icon: Network,        badge: `${nnClasses} modules` },
        { name: "lucid.nn.functional", slug: "lucid.nn.functional", description: `Stateless functional operations (${nnFn} functions).`,    icon: FunctionSquare },
        { name: "lucid.nn.init",       slug: "lucid.nn.init",       description: `Weight initialization strategies (${nnInit} functions).`, icon: Sparkles       },
        { name: "lucid.nn.utils",      slug: "lucid.nn.utils",      description: "Gradient clipping, weight norm, RNN packing.",            icon: Wrench         },
      ],
    },
    {
      category: "Model Zoo",
      modules: [
        { name: "lucid.models", slug: "lucid.models", description: `${families} paper-cited vision / text / generative families with pretrained factories.`, icon: BrainCircuit, badge: `${families} families` },
      ],
    },
    {
      category: "Optimization",
      modules: [
        { name: "lucid.optim", slug: "lucid.optim", description: `${optimizers} optimizers + ${lrSchedulers} learning rate schedulers.`, icon: TrendingDown },
      ],
    },
    {
      category: "Fine-Tuning",
      modules: [
        { name: "lucid.peft", slug: "lucid.peft", description: `LoRA fine-tuning in the reference adapter library's API and file format — ${peft} names: LoraConfig, get_peft_model, PeftModel, merge and save.`, icon: Puzzle, badge: `${peft} names` },
      ],
    },
    {
      category: "Differentiation",
      modules: [
        { name: "lucid.autograd", slug: "lucid.autograd", description: "Function, grad, gradcheck, functional transforms.", icon: GitBranch },
        { name: "lucid.func",     slug: "lucid.func",     description: "vmap, grad, vjp, jvp, jacrev, jacfwd, hessian.",    icon: Workflow  },
      ],
    },
    {
      category: "Deployment",
      modules: [
        { name: "lucid.coreml", slug: "lucid.coreml", description: `Export to Core ML and run on the Neural Engine — ${coreml} names covering precision, weight compression, flexible shapes, and state.`, icon: Rocket, badge: `${coreml} names` },
      ],
    },
    {
      category: "Math",
      modules: [
        { name: "lucid.linalg",        slug: "lucid.linalg",        description: `${linalg} decomposition, norm, and solve operations.`,     icon: Sigma    },
        { name: "lucid.fft",           slug: "lucid.fft",           description: `${fft} DFT / Hermitian / shift / frequency functions.`,    icon: Waves    },
        { name: "lucid.special",       slug: "lucid.special",       description: `${special} special-math functions (erf, sinc, gamma, …).`, icon: Star     },
        { name: "lucid.signal",        slug: "lucid.signal",        description: `${signal} spectral window functions.`,                     icon: Activity },
        { name: "lucid.einops",        slug: "lucid.einops",        description: "Einops-style tensor manipulation.",                         icon: Shuffle  },
        { name: "lucid.distributions", slug: "lucid.distributions", description: `${distributions} distributions + base class + KL registry.`, icon: Dices  },
        { name: "lucid.diffeq",        slug: "lucid.diffeq",        description: `${diffeq} names — fixed-step explicit Runge-Kutta ODE integration.`,        icon: Spline   },
      ],
    },
    {
      category: "Utilities",
      modules: [
        { name: "lucid.utils.data",       slug: "lucid.utils.data",       description: "Dataset, Sampler, DataLoader.",                          icon: Database          },
        { name: "lucid.utils.cache",      slug: "lucid.utils.cache",      description: "DynamicCache, StaticCache, EncoderDecoderCache for incremental decoding.", icon: Layers },
        { name: "lucid.utils.tokenizer",  slug: "lucid.utils.tokenizer",  description: "BPE, WordPiece, Unigram, and byte-level tokenizers.",    icon: Type              },
        { name: "lucid.utils.transforms", slug: "lucid.utils.transforms", description: "Image and tensor augmentation transforms.",              icon: SlidersHorizontal },
      ],
    },
    {
      category: "Others",
      modules: [
        { name: "lucid.amp",           slug: "lucid.amp",           description: "Automatic mixed precision.",       icon: Gauge },
        { name: "lucid.quantization",  slug: "lucid.quantization",  description: "Post-training + QAT quantization (int8 / int4 weights, MLX GEMM).", icon: Binary },
        { name: "lucid.profiler",      slug: "lucid.profiler",      description: "Performance profiling utilities.", icon: Timer },
        { name: "lucid.serialization", slug: "lucid.serialization", description: "State dict save / load.",          icon: Save  },
      ],
    },
    {
      category: "Engine",
      modules: [
        { name: "lucid._C.engine", slug: "lucid._C.engine", description: "C++ compute core — storage, ops, autograd graph, backend dispatch.", icon: Cpu, badge: `${engineMembers} members` },
      ],
    },
  ];
}

const ACCENT = "var(--color-lucid-primary)";

function ModuleCardLink({ mod }: { mod: ModuleCard }) {
  const Icon = mod.icon;
  const bg = `color-mix(in srgb, ${ACCENT} 14%, transparent)`;
  const ring = `color-mix(in srgb, ${ACCENT} 38%, transparent)`;
  return (
    <Card href={`/api/${mod.slug}`}>
      <div className="px-5 pt-4 pb-4 flex items-start gap-3">
        <span
          className="shrink-0 inline-flex h-9 w-9 items-center justify-center rounded-lg border"
          style={{ backgroundColor: bg, borderColor: ring, color: ACCENT }}
          aria-hidden
        >
          <Icon className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-2">
            <code
              className="font-mono text-sm font-semibold"
              style={{ color: ACCENT }}
            >
              {mod.name}
            </code>
            {mod.badge && (
              <Badge variant="secondary" className="font-mono text-[10px]">
                {mod.badge}
              </Badge>
            )}
          </div>
          <p className="mt-1 text-xs text-lucid-text-low leading-relaxed">
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
                className={cn(
                  "grid gap-3",
                  // A single-module category renders one card; in a 2-col grid
                  // that card fills only one column and reads as a stray
                  // half-width tile.  Give lone cards the full row.
                  modules.length === 1 ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-2",
                )}
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
