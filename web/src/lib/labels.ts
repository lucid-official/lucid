/**
 * Unified display-label resolution — the single source of truth for turning
 * raw Python slugs / source-file segment names into the friendly titles shown
 * in the sidebar, breadcrumbs, and sub-package cards.
 *
 * Three layers, applied in precedence order by each resolver:
 *   1. Explicit alias map (``PACKAGE_LABELS`` for slugs, ``SEGMENT_LABELS``
 *      for subcategory/file segments) — for names that can't be derived.
 *   2. Acronym-aware titleization (``titleize``) — capitalises words and
 *      upper-cases known acronyms (``bpe`` → ``BPE``, ``fft`` → ``FFT``).
 *   3. Data-driven canonical names (model families carry ``canonical_name``
 *      from ``@model_family_meta``) — resolved by the caller, which prefers
 *      that over ``packageLabel`` when present.
 *
 * Add a new friendly name in EXACTLY one place here — never inline at a call
 * site.  If a name is a pure acronym, add it to ``ACRONYMS`` instead of the
 * override maps so every derived form (sidebar + card + breadcrumb) agrees.
 */

// Tokens that should render fully upper-cased when they appear as a
// whole word inside a slug/segment (case-insensitive match).  Keeps
// ``titleize`` general — a new acronym here fixes the label everywhere.
const ACRONYMS = new Set<string>([
  "bpe", "fft", "rnn", "lstm", "gru", "sgd", "kl", "lr", "mlp", "api",
  "cpu", "gpu", "mps", "io", "svd", "qr", "lu", "vae", "ncsn", "ddpm",
  "gan", "amp", "relu", "gelu", "selu", "elu", "rms", "wgan", "uri",
]);

// Words with irregular internal casing that titleization can't derive.
const MIXED_CASE: Record<string, string> = {
  wordpiece:  "WordPiece",
  bytelevel:  "ByteLevel",
  bytepair:   "BytePair",
  adamw:      "AdamW",
  lora:       "LoRA",
  resnext:    "ResNeXt",
  convnext:   "ConvNeXt",
};

/** Capitalise each ``_``/``-``-separated word, upper-casing known acronyms
 *  and applying irregular mixed-case forms.  ``byte_bpe`` → ``Byte BPE``,
 *  ``fft`` → ``FFT``, ``wordpiece`` → ``WordPiece``. */
export function titleize(slug: string): string {
  return slug
    .split(/[_\-]/)
    .map((w) => {
      if (w.length === 0) return w;
      const lower = w.toLowerCase();
      if (ACRONYMS.has(lower)) return lower.toUpperCase();
      if (MIXED_CASE[lower]) return MIXED_CASE[lower];
      return w[0].toUpperCase() + w.slice(1);
    })
    .join(" ");
}

// ---------------------------------------------------------------------------
// Segment labels — keyed by a source-file / subcategory segment (the bucket
// names produced by build-api-data.py's ``_subcategory``).  Used for the
// in-page member tree and any per-file grouping.
// ---------------------------------------------------------------------------

const SEGMENT_LABELS: Record<string, string> = {
  // lucid top-level
  tensor: "Tensor",
  types: "Types & Protocols",
  dtypes: "Data Types",
  device: "Device",
  factories: "Tensor Creation",
  ops: "Tensor Operations",
  composite: "Composite Ops",
  autograd: "Autograd",
  predicates: "Predicates",
  serialization: "Serialization",
  globals: "Global Settings",
  threads: "Threading",
  dispatch: "Dispatch",
  vmap: "vmap",
  // nn / nn.functional file basenames
  modules: "Modules",
  hooks: "Hooks",
  activations: "Activations",
  activation: "Activations",
  attention: "Attention",
  container: "Containers",
  conv: "Convolution",
  dropout: "Dropout",
  flatten: "Flatten",
  linear: "Linear",
  loss: "Loss",
  normalization: "Normalization",
  padding: "Padding",
  pooling: "Pooling",
  rnn: "Recurrent",
  sparse: "Sparse",
  transformer: "Transformer",
  upsampling: "Upsampling",
  sampling: "Sampling",
  // optim
  sgd: "SGD",
  adam: "Adam / AdamW",
  lbfgs: "L-BFGS",
  lr_scheduler: "LR Schedulers",
  others: "Other Optimizers",
  // autograd
  function: "Function",
  graph: "Graph",
  gradcheck: "Gradient Check",
  checkpoint: "Checkpointing",
  profiler: "Profiler",
  // utils.data
  dataloader: "DataLoader",
  dataset: "Dataset",
  sampler: "Sampler",
  // utils.tokenizer — algorithm modules (named things, like model families)
  base: "Base",
  bpe: "BPE",
  byte: "Byte",
  byte_bpe: "ByteLevel",
  char: "Char",
  regex: "Regex",
  unigram: "Unigram",
  whitespace: "Whitespace",
  word: "Word",
  wordpiece: "WordPiece",
  // peft — file stems (``_peft_model.py``) and the tuners/lora tree
  peft_model: "PeftModel",
  state_dict: "State Dict",
  // amp
  autocast: "Autocast",
  grad_scaler: "Gradient Scaler",
  // nn.utils
  clip_grad: "Gradient Clipping",
  convert_parameters: "Parameter Conversion",
  fusion: "Fusion",
  parametrize: "Parametrization",
  parametrizations: "Parametrizations",
  prune: "Pruning",
  skip_init: "Skip Init",
  spectral_norm: "Spectral Norm",
  weight_norm: "Weight Norm",
  // lucid.ops arity buckets
  unary: "Unary",
  binary: "Binary",
  ternary: "Ternary",
  variadic: "Variadic",
  // C++ engine subcategory buckets (mirror the lucid/_C/ directory tree)
  core: "Core",
  nn: "Neural Networks",
  ufunc: "Unary Ops",
  bfunc: "Binary Ops",
  gfunc: "Generic Ops",
  cfunc: "Composite Ops",
  kernel: "Kernels",
  primitives: "Primitives",
  backend: "Backend",
  cpu: "CPU (Accelerate)",
  gpu: "GPU (MLX)",
  mps: "MPS",
  bindings: "Bindings",
  helpers: "Helpers",
  random: "Random",
  complex: "Complex",
  einops: "Einops",
  fft: "FFT",
  linalg: "Linear Algebra",
  optim: "Optimizers",
  test: "Test",
  utils: "Utils",
  // distributions
  transforms: "Transforms",
  kl: "KL Divergence",
  normal: "Normal",
  bernoulli: "Bernoulli",
  categorical: "Categorical",
  uniform: "Uniform",
  exponential: "Exponential",
  gamma: "Gamma",
  student: "Student-t",
  multivariate: "Multivariate",
  matrix: "Matrix",
  discrete: "Discrete",
  continuous_extra: "Continuous (extra)",
  independent: "Independent",
  mixture: "Mixture",
  relaxed: "Relaxed",
  extra: "Extra",
};

/** Friendly label for a subcategory/file segment (sidebar member tree). */
export function segmentLabel(segment: string): string {
  return SEGMENT_LABELS[segment] ?? titleize(segment);
}

// ---------------------------------------------------------------------------
// Package (slug) labels — keyed by the full dotted slug.  Used for every
// sidebar entry, breadcrumb step, and sub-package card title.  The slug (and
// therefore the URL) is unchanged; only the displayed title is aliased.
// ---------------------------------------------------------------------------

const PACKAGE_LABELS: Record<string, string> = {
  "lucid.tensor":            "Tensor",
  "lucid.creation":          "Tensor Creation",
  "lucid.ops":               "Tensor Operations",
  "lucid.ops.composite":     "Composite Ops",
  "lucid.nn":                "Neural Networks",
  "lucid.nn.functional":     "Functional",
  "lucid.nn.init":           "Init",
  "lucid.nn.utils":          "NN Utils",
  "lucid.nn.quantized":      "Quantized",
  "lucid.nn.qat":            "QAT",
  "lucid.nn.intrinsic":      "Fused Modules",
  "lucid.optim":             "Optimizers",
  "lucid.autograd":          "Autograd",
  "lucid.compile":           "Compile",
  "lucid.func":              "Functional Transforms",
  "lucid.linalg":            "Linear Algebra",
  "lucid.fft":               "FFT",
  "lucid.signal":            "Signal",
  "lucid.special":           "Special Functions",
  "lucid.distributions":     "Distributions",
  "lucid.diffeq":            "Differential Equations",
  "lucid.einops":            "Einops",
  "lucid.amp":               "Mixed Precision",
  "lucid.quantization":      "Quantization",
  "lucid.profiler":          "Profiler",
  "lucid.serialization":     "Serialization",
  "lucid.utils":             "Utils",        // synthetic-ish parent
  "lucid.utils.cache":       "KV Cache",
  "lucid.utils.data":        "Data",
  "lucid.utils.rollout":     "Rollout",
  "lucid.utils.tokenizer":   "Tokenizers",
  "lucid.utils.transforms":  "Transforms",
  "lucid.weights":           "Weights",
  "lucid.coreml":            "Core ML",
  "lucid.peft":              "PEFT",
  "lucid.peft.tuners":       "Tuners",
  "lucid.peft.tuners.lora":  "LoRA",
  "lucid.models":            "Model Zoo",
  "lucid.models.vision":     "Vision Models",
  "lucid.models.text":       "Text Models",
  "lucid.models.generative": "Generative Models",
  "lucid.models.multimodal": "Multimodal Models",
  // C++ engine — generated by web/scripts/build-cpp-data.py (libclang).
  "lucid._C.engine":         "C++ Engine",
};

/** Friendly label for a package slug (sidebar / card / breadcrumb).  Falls
 *  back to an acronym-aware titleization of the last meaningful segment. */
export function packageLabel(slug: string): string {
  if (PACKAGE_LABELS[slug]) return PACKAGE_LABELS[slug];
  const stripped = slug.startsWith("lucid.") ? slug.slice(6) : slug;
  return titleize(stripped);
}
