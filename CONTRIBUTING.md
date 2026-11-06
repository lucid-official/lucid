# Contributing to Lucid

Lucid is a production-grade ML framework for Apple Silicon, with a PyTorch-compatible Python surface backed by a custom C++ engine running on MLX (GPU) and Apple Accelerate (CPU). This guide covers everything you need to contribute correctly.

**Read this document fully before opening a PR.** Most rejections come from violating the hard rules in §3, which are non-negotiable.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Development environment](#2-development-environment)
3. [Hard rules — non-negotiable](#3-hard-rules--non-negotiable)
4. [Coding conventions](#4-coding-conventions)
5. [Architecture invariants](#5-architecture-invariants)
6. [Adding a new op](#6-adding-a-new-op)
7. [Testing](#7-testing)
8. [Changelog and commits](#8-changelog-and-commits)
9. [Pull request checklist](#9-pull-request-checklist)
10. [Out-of-scope](#10-out-of-scope)

---

## 1. Project overview

Lucid targets Apple Silicon exclusively (M1–M4, macOS arm64). Linux, Windows, and x86 are out of scope and will not be supported in this major version.

### Layer stack

```
Python public API       lucid.* / lucid.nn.* / lucid.optim.*
Python composite/dispatch  lucid/_ops/composite/, _dispatch.py
pybind11 boundary       lucid/_C/engine.cpython-*.so
C++ engine
  tensor/    TensorImpl, AutogradMeta, view semantics
  core/      Dtype, Shape, Device, Error, Allocator
  backend/
    cpu/     Apple Accelerate (BLAS / LAPACK / vDSP / vForce)
    gpu/     MLX (mlx::core::*) + MetalAllocator
  kernel/    IKernel, UnaryKernel, BinaryKernel, primitives
  autograd/  Node, Engine, AutogradNode<Derived,N>
  ops/       bfunc / ufunc / gfunc / utils / linalg / einops / nn
  optim/     SGD / Adam / AdamW / RMSprop / …
  registry/  OpSchema, OpRegistry, BindingGen
  bindings/  bind_*.cpp (auto-generated where possible)
```

### Layer dependency DAG (top → bottom; no reverse imports allowed)

```
bindings → ops → kernel → autograd → backend → tensor → core
                       ↑
                  registry  (orthogonal; used by bindings + kernel)
                       ↑
                  primitives (kernel/primitives/; may use backend)
```

`tools/check_layers.py` checks four Python import boundaries in CI; it is not
a complete validator of this C++ dependency DAG. Review the remaining edges
when changing engine code.

### Backend rule

| Stream | Backend | Notes |
|--------|---------|-------|
| CPU | Apple Accelerate only | vDSP / vForce / BLAS / LAPACK |
| GPU | MLX only | mlx::core::* |
| `lucid.linalg` on CPU | MLX (exception) | MLX itself is CPU-backed here; wrap result back as GPU |
| Data-dependent output shapes | CPU round-trip | Unavoidable; document the reason in a comment |

---

## 2. Development environment

### Requirements

- macOS 26.2 Tahoe (arm64) or later, M1 or later — the minos of the MLX build
  the engine links (`[tool.lucid.mlx]`), which the build fetches itself
- Python 3.14 only (PEP 649 lazy annotations — H1/H7 require it)
- CMake ≥ 3.24
- Ninja ≥ 1.11
- Xcode Command Line Tools

### Install for development

```bash
git clone https://github.com/lucid-official/lucid.git
cd lucid
uv venv --python 3.14
uv pip install --python .venv/bin/python setuptools wheel cmake ninja "pybind11>=3.0,<3.1"
uv pip install --python .venv/bin/python -e ".[dev]" --no-build-isolation
.venv/bin/python -m tools.doctor
```

For parity tests against the reference framework, install it separately and run:

```bash
uv pip install --python .venv/bin/python -e ".[test]" --no-build-isolation
# install reference framework separately
.venv/bin/python -m pytest lucid/test/parity/ -m parity
```

For documentation:

```bash
uv pip install --python .venv/bin/python -e ".[docs]" --no-build-isolation
```

### Build the C++ engine

```bash
uv pip install --python .venv/bin/python -e . --no-build-isolation
.venv/bin/python -m tools.doctor
```

The build backend is setuptools; `setup.py` drives CMake. Pin the interpreter
explicitly: a uv environment need not contain pip, and invoking another pip
installs into another environment. The engine links the MLX build pinned in
`pyproject.toml` `[tool.lucid.mlx]`, which the build downloads, verifies and
stages under `~/Library/Caches/lucid/mlx-metal` (`tools/fetch_mlx.py`); the
`mlx` package is needed only for interop (the `[mlx]` extra). Rebuild after
source updates when the import guard reports an ABI mismatch. `tools.doctor
--json` reports interpreter, dependencies, source ABI agreement, the pinned
MLX's cache entry and the engine's link to it, and an isolated native
import/CPU smoke check, even when `import lucid` itself fails.

### Published checkpoint checks

```bash
.venv/bin/python -m tools.check_pretrained_parity --list
.venv/bin/python -m tools.check_pretrained_parity --model resnet_18_cls \
  --clean-downloads --json build/pretrained.json
```

The selected checkpoint comes from the factory's actual `pretrained=True`
resolution, not necessarily the shared enum's `DEFAULT`. `--clean-downloads`
uses an owned temporary cache in a child process: existing user caches are
neither reused nor swept. The normal HF authentication file location is preserved
without copying token contents; explicit authentication settings and implicit-auth
opt-out remain respected. Authentication files are outside download cleanup.
Reports checkpoint completed comparisons atomically;
`finished` and `complete` are distinct, and missing or oversized oracles remain
unverified. The default 500M-parameter bound applies to every adapter. Raise it
only on a machine able to hold both implementations and loading temporaries.

Adapters compare classification/text/CLIP outputs, segmentation query or semantic
scores, raw DETR queries, and deterministic diffusion components. Diffusion
checks do not validate stochastic sampling or image quality; none of these checks
establishes a dataset metric. DETR uses its original upstream implementation at
a pinned revision. Optional oracle packages belong only in the development
environment, never in Lucid's internal compute dependencies.

Reference loading disables automatic remote format conversion: validation reads
existing checkpoints and must not start publishing-service jobs. Socket defaults
are capped at 60 seconds during each comparison, preserving stricter caller
defaults and restoring them afterward; explicit SDK timeouts are unaffected.

For split runs, merge explicitly ordered reports before attaching them:

```bash
.venv/bin/python -m tools.merge_pretrained_evidence \
  build/pretrained-first.json build/pretrained-recheck.json \
  --output build/pretrained-merged.json
```

Later observations replace earlier ones even when they fail or cannot be
verified. Missing factories, unexpected names, invalid numbers and report hashes
remain explicit. A complete aggregate covers the current discovery registry's
default checkpoints; it does not establish working-tree equivalence or coverage
of every enum variant. CLIP zero-shot wrappers have fully pretrained trunks and
no random classifier; text wrappers with random task heads are not advertised as
fully pretrained checkpoints.

To snapshot the actual API/model/native registrations and attach verification
evidence without interpreting registrations as universal backend support:

```bash
.venv/bin/python -m tools.support_manifest --output build/support.json
# Reports are optional, scoped evidence; absent evidence means unverified.
.venv/bin/python -m tools.support_manifest --audit build/audit.json \
  --pretrained build/pretrained.json --output build/support.json
```

Attached reports retain failures, skips, environment and scope; the manifest
does not assume they were generated from the current source revision. Checkpoint
output parity is not a dataset accuracy or training-convergence result.

The manifest's `api_contracts` links each enumerated export to its implementation,
stub declarations (or explicitly unverified inline typing), generated documentation,
static test calls and attached audit findings. Empty links stay empty: a test call
is not a passed assertion, and a documented name is not a verified implementation.

Compile and quantized-development benchmarks share `tools/_bench_timing.py`:
every measured call materializes its returned output before synchronization.
These are end-to-end latencies, including host observation, not kernel timings.
Use `python -m tools.bench_quantized_dispatch --output build/quantized.json` to
compare packed, cached-dense and per-call dequantization on identical weights.

The pytest `bench` fixture uses the same output-observation contract. Return
the tensors being measured, including gradients for backward timings. Its
`last_elapsed` is the plugin median or a one-shot fallback/disabled-plugin
observation, so configured thresholds apply with either provider. Do not compare
these different sampling protocols as a speedup measurement.

On memory-constrained Apple Silicon, run large suites sequentially. The existing
sharded model runner starts a fresh interpreter for each chunk and reports any
memory-related omissions explicitly:

```bash
.venv/bin/python -m lucid.test.audit --tests-only \
  --suite-path lucid/test/unit/models --no-line-coverage --no-doctests
```

To verify the model zoo family by family, each in its own process under a memory
budget taken from the machine, run `lucid-audit --models`. It verifies the zoo on
Metal and then on the CPU (`--zoo-device metal` or `cpu` narrows it), and skips what
the machine can't hold, saying so. Add `--family <name>` for one family. Use a scratch clone: a moved HEAD stops the run.

Sanitizer builds for memory/UB checking:

```bash
LUCID_BUILD_MODE=debug-asan  uv pip install --python .venv/bin/python -e . --no-build-isolation
LUCID_BUILD_MODE=debug-ubsan uv pip install --python .venv/bin/python -e . --no-build-isolation
```

### Static analysis tools

```bash
ruff check lucid/           # Python linting
mypy --strict lucid/        # Python type checking (see mypy.ini for rationale)
bash tools/check_format.sh  # clang-format + clang-tidy for C++
```

---

## 3. Hard rules — non-negotiable

These rules are mandatory. Automated checks cover parts of them; a green CI run
does not replace review of the remaining requirements.

### H1 — No `from __future__ import annotations`

This import is forbidden in every file under `lucid/`. Python 3.14's lazy annotations make it redundant and it interferes with runtime type inspection.

Detection: `grep -r "from __future__ import annotations" lucid/`

### H2 — `lucid._C` imports must use `_C_{name}` aliases

```python
# Correct
from lucid._C import engine as _C_engine

# Wrong — all of these are banned
import lucid._C.engine as engine
import lucid._C.engine as _eng
import lucid._C.engine as _e
import lucid._C.engine as _ce
```

### H3 — CPU = Accelerate only; GPU = MLX only

Never mix backends. The two permitted exceptions are `lucid.linalg` (MLX on CPU stream) and ops with data-dependent output shapes (CPU round-trip). Every other case must use the canonical backend for its stream.

### H4 — No external library imports inside Lucid internals

Lucid is a standalone framework. The only dependencies permitted inside `lucid/` are the C++ engine and the Python standard library. **`numpy`, `scipy`, and any other third-party package are forbidden** in all compute paths:

- `lucid/_ops/composite/`
- `lucid/_tensor/` (except the 6 bridge points below)
- `lucid/nn/`
- `lucid/optim/`
- `lucid/autograd/`
- `lucid/linalg/`
- `lucid/fft/`
- `lucid/signal/`
- `lucid/special/`
- `lucid/distributions/`
- `lucid/einops/`
- `lucid/amp/`
- `lucid/profiler/`

**The 6 permitted bridge boundaries** (and only these):

| # | Location | What it may import |
|---|----------|--------------------|
| 1 | `lucid/_factories/converters.py` | numpy — external tensor → Lucid conversion |
| 2 | `lucid/_tensor/tensor.py` — `.numpy()`, `__dlpack__`, `_to_impl` only | numpy / dlpack |
| 3 | `lucid/_tensor/_repr.py` | numpy — display only |
| 4 | `lucid/_types.py` | typing protocols — no runtime compute |
| 5 | `lucid/serialization/` + `lucid/optim/{optimizer,lbfgs}.py` state_dict paths | numpy — checkpoint serialization only |
| 6 | `lucid/utils/data/dataloader.py` | numpy — external data ingest |

If you see `import numpy` anywhere outside these six locations, it is a violation.

### H5 — No "torch" or "PyTorch" in source

The words "torch" and "PyTorch" are banned from source code, comments, docstrings, and error messages. Use "reference framework" or similar neutral phrasing instead.

**Only exception:** `lucid/test/_fixtures/ref_framework.py` — the test infrastructure is opt-in (`pip install lucid[test]` + `pytest -m parity`) and may use the literal name. All other test files receive the reference framework only through the `ref` fixture.

Detection: `grep -ri "torch\|pytorch" lucid/ --include="*.py" --exclude-dir=test`

### H6 — No "cuda"

Lucid is Apple Silicon only. Use `metal` everywhere. "cuda" is banned.

Detection: `grep -ri "cuda" lucid/`

### H7 — No string type hints

Forward references as string literals are banned. Use a `TYPE_CHECKING` block instead:

```python
# Wrong
def forward(self, x: "Tensor") -> "Tensor": ...

# Correct
from __future__ import annotations  # NO — see H1
# Instead:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

def forward(self, x: Tensor) -> Tensor: ...
```

Python 3.14 lazy annotations make `TYPE_CHECKING`-style imports sufficient for type checkers without string literals.

### H8 — One canonical path per op; no sub-package shortcuts

- `lucid.linalg` ops are accessed only as `lucid.linalg.<name>`.
- `lucid.einops` ops are accessed only as `lucid.einops.<name>`.
- Do not add top-level aliases like `lucid.norm`, `lucid.cross`, `lucid.einsum`, `lucid.vander`.
- Do not add Tensor method shortcuts like `tensor.norm()`, `tensor.cross()`.

Each op has exactly one canonical path. Duplicating it creates maintenance debt.

### H9 — No `*args` / `**kwargs` in `.pyi` stubs

Every function and method in a `.pyi` file must have an explicit signature that matches the implementation 1:1 — parameter names, types, defaults, positional-only `/`, and keyword-only `*` separators.

```python
# Wrong
def add(self, *args: Any, **kwargs: Any) -> Tensor: ...

# Correct
def add(self, other: Tensor | float, /, *, alpha: float = 1.0) -> Tensor: ...
```

The only exception is a genuinely variadic API such as `def cat(*tensors: Tensor) -> Tensor`.

Detection: `grep -rn "\*args\|\*\*kwargs" lucid/ --include="*.pyi"`

### H10 — Spell model size names in full

Use `tiny`, `small`, `base`, `medium`, `large` and `xlarge`, not abbreviated
suffixes. This applies to factories, exports, registry names and test IDs.

### H11 — Register only paper-defined variants

Every variant needs a basis in its original paper. A single-size family uses its
nominal name without an invented size suffix. For small tests, override the config
of an existing real factory instead of registering an imaginary tiny variant.

### H12 — Complete the new-family contract

New families require the maintainer's explicit implementation approval after
paper review; discovering a missing family is not authorization to add it. Follow
the project's full 16-step family procedure. Its required structural gates include:

- `_config.py`, `_model.py`, `_pretrained.py`, and `__init__.py` in the family directory.
- `@model_family_meta(canonical_name=, citation=, theory=)` above a frozen dataclass
  config, with its `model_type: ClassVar[str]`.
- Config, direct model, task wrapper, output dataclass and private building blocks.
- Every factory declares task, family, model type/class, default config,
  paper-grounded integer parameter count and `summary="auto"`.
- Pass `python -m tools.validate_model_zoo --family <family> --runtime`,
  `pytest lucid/test/unit/models/test_family_contract.py -k <family>`, and
  `python -m tools.build_model_summaries --family <family>` using shadow allocation.

### H13 — No `goto` statements in native code

Under `lucid/_C/`, use structured control flow, extracted functions and RAII
instead of `goto` in `.cpp`, `.h` or `.mm` files.

---

## 4. Coding conventions

### Python

| Rule | Detail |
|------|--------|
| **S1** | 4-space indent, 100-column lines |
| **S2** | `ruff check lucid/` and `mypy --strict lucid/` must pass |
| **S6** | Only `_wrap` / `_unwrap` in `lucid/_dispatch.py` may cross the `Tensor ↔ TensorImpl` boundary |
| **S7** | Tier 1 (`lucid.*`) exposes only ops, factories, dtypes, grad-control, and sub-packages. `Module`, `Parameter`, `Linear`, `Adam`, `DataLoader`, etc. live in Tier 2 (`lucid.nn.*`, `lucid.optim.*`, etc.) and must not appear in `lucid.__all__` |
| **S8** | Tier 3 (`lucid._*`) is private — never import it from public API |
| **S9** | dtype aliases are module attributes: `lucid.float = lucid.float32` — never shadow Python builtins |
| **S18** | Write no comments by default. Add one only when the **why** is non-obvious: a hidden constraint, a subtle invariant, or a workaround for a specific bug |
| **S19** | Implement only what the task requires. No speculative abstractions. Three similar lines is better than a premature helper |

### C++

| Rule | Detail |
|------|--------|
| **S1** | 4-space indent, brace-attached (Google base), 100-column — enforced by `.clang-format` |
| **S3** | `clang-format --dry-run --Werror` + `clang-tidy` must pass (`bash tools/check_format.sh`) |
| **S10** | Never `throw std::runtime_error` or `std::invalid_argument` directly — use the `LucidError` hierarchy + `ErrorBuilder` chain from `lucid/_C/core/Error.h` |
| **S11** | Every public C++ type must be tagged with the `LUCID_API` macro (default visibility is hidden) |
| **S12** | Every op must register an `OpSchema` (name, version, AMP policy, determinism flag, complexity class) via `lucid/_C/registry/` |
| **S20** | Every in-place op must call `bump_version()` on the tensor; every forward must call `set_saved_versions(...)` — both are required for backward version checks to work |

### Strict type hints everywhere

Every function in `lucid/` (public, private, helper) must annotate every parameter and return type. No bare `def f(x):`. This is verified by `tools/check_stubs.py`.

---

## 5. Architecture invariants

These must hold at all times. A PR that breaks any of them cannot merge.

| # | Invariant |
|---|-----------|
| **A1** | `Tensor._impl` is always a `TensorImpl`. Nothing else may be stored there. |
| **A2** | Direct access to `._impl` is allowed only inside `lucid/_tensor/tensor.py` and `lucid/_dispatch.py`. Everywhere else must go through `_wrap` / `_unwrap`. |
| **A3** | Scalar operands are coerced via `_to_tensor` — this is how `a + 3` works. |
| **A4** | `Parameter` is a `Tensor` subclass — `isinstance(p, Tensor)` is always `True`. |
| **A5** | `Optimizer` extracts `_impl` at construction time. Post-construction parameter changes go through `add_param_group`. |
| **A6** | `Module.training` propagates to all children via `train()`. |
| **A7** | `nn.functional` is stateless — Tensor in, Tensor out. No module state. |
| **A8** | Import order: `_dispatch → _tensor → _ops → _factories → autograd → nn → optim → lucid`. No cycles. |
| **A9** | `state_dict` returns `OrderedDict` with a `_metadata` attribute (PyTorch-compatible). `lucid.save` separates and re-attaches it. |
| **A10** | `_load_from_state_dict` signature: `(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)`. |

---

## 6. Adding a new op

### Composite op (pure Python, no C++ changes)

A composite op is implemented entirely in Python on top of existing engine primitives. Use this path whenever possible.

1. Add the implementation to the appropriate file in `lucid/_ops/composite/`:
   - `elementwise.py`, `reductions.py`, `shape.py`, `blas.py`, `statistics.py`,
     `indexing.py`, `predicates.py`, `dtype.py`, `constants.py`
2. Add the name to that file's `__all__`.
3. The name is automatically included in `COMPOSITE_NAMES` (via `lucid/_ops/composite/__init__.py`) and exposed through the top-level lazy loader.
4. Add an `OpEntry` in `lucid/_ops/_registry.py` with `method_name` and `free_fn_name`.
5. Write a unit test in `lucid/test/unit/` and a parity test in `lucid/test/parity/`.

### Engine op (new C++ op)

Use the scaffolding CLI to generate the boilerplate:

```bash
python tools/new_op.py <OpName> --kind <unary|binary|composite>
# preview without writing:
python tools/new_op.py <OpName> --kind unary --dry-run
```

This generates 9 files automatically. Then:

1. Implement the op in `lucid/_C/ops/<family>/MyOp.{h,cpp}`.
2. Register an `OpSchema` in `lucid/_C/registry/`.
3. Add a pybind11 binding in `lucid/_C/bindings/bind_<family>.cpp`.
4. Update `lucid/_ops/_registry.py` with an `OpEntry`.
5. Update the relevant `.pyi` stub (or run `python tools/gen_pyi.py`).
6. Rebuild the engine: `cmake --build build/temp.macosx-*/lucid__C_engine/ -j$(sysctl -n hw.ncpu)`
7. Write unit + parity tests.

### Parity test requirement

Every new public API **must** ship both:
- A unit test in `lucid/test/unit/` (no reference framework dependency)
- A parity test in `lucid/test/parity/` using the `ref` fixture

```python
# Example parity test
@pytest.mark.parity
def test_my_op_parity(ref):
    x = lucid.tensor([1.0, 2.0, 3.0])
    x_ref = ref.tensor([1.0, 2.0, 3.0])
    assert_close(lucid.my_op(x), ref.my_op(x_ref))
```

---

## 7. Testing

### Test suite layout

```
lucid/test/
├── unit/        Pure Lucid tests — no reference framework
├── nn/          nn.Module / nn.functional
├── autograd/    backward / gradcheck / higher-order
├── linalg/      decomposition correctness
├── parity/      @pytest.mark.parity — numerical parity vs reference framework
├── integration/ @pytest.mark.slow — end-to-end (model training)
└── helpers/     Numerics / parity utilities
```

### Running tests

```bash
# Fast unit tests only (no reference framework needed)
pytest lucid/test/unit/ -q

# Full suite (excluding parity)
pytest lucid/test/ --ignore=lucid/test/parity -q

# Full suite including parity (requires reference framework)
pytest lucid/test/ -q

# C++ Google Test suite
ctest --test-dir build/temp.macosx-*/lucid__C_engine/ --output-on-failure
```

### Test markers

| Marker | Meaning |
|--------|---------|
| `parity` | Numerical parity vs reference framework; auto-skips if not installed |
| `smoke` | Quick sanity check (< 1 s) |
| `slow` | End-to-end, > 5 s (training loops, full model forward) |
| `gpu` | Requires Apple Silicon GPU (Metal); auto-skips otherwise |
| `perf` | `pytest-benchmark` timing test; opt-in via `--benchmark-only` |
| `stability` | Numerical edge cases (inf / nan / subnormal) |
| `f64_only` | Float64-only path; skipped when only float32 is available |

### Quick smoke test (MLP XOR)

```python
import lucid
import lucid.nn as nn
import lucid.optim as optim

m = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid())
opt = optim.Adam(m.parameters(), lr=0.01)
X = lucid.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=lucid.float32)
Y = lucid.tensor([[0],[1],[1],[0]], dtype=lucid.float32)
for _ in range(500):
    loss = lucid.nn.functional.mse_loss(m(X), Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
assert loss.item() < 0.05
```

### Phase exit gate

Before any PR can merge, **all of the following must pass**:

```bash
ruff check lucid/
mypy --strict lucid/
pytest lucid/test/ -q
ctest --test-dir build/temp.macosx-*/lucid__C_engine/ --output-on-failure

python tools/check_layers.py      # layer dependency DAG
python tools/check_stubs.py       # .pyi stub freshness
python tools/check_op_api.py      # OpEntry → engine_fn consistency
python tools/check_doxygen.py     # C++ doc coverage
bash   tools/check_format.sh      # clang-format + clang-tidy
bash   scripts/ci_full.sh         # full CI gate
```

Before a push, `scripts/ci_local.sh` runs what CI's push gate runs — the
`core`, `coreml` and `checks` parts of `ci_full.sh`, side by side — in
interpreters held to the packages CI installs (`scripts/ci_mirror/`), so a
test that would skip on the runner skips here too. `ci_full.sh` takes the
same parts (`core coreml checks zoo native`, or `zoo:K/N` for one shard);
with none it runs all of them in sequence.

---

## 8. Changelog and commits

Lucid uses [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Adding a changelog entry

Use the helper tool:

```bash
python tools/changelog.py add --section Added "Brief description of the change"
```

Or edit `CHANGELOG.md` manually under the `[Unreleased]` section. Omitting the changelog entry from a PR will block merge.

### Changelog sections

Standard sections from Keep a Changelog (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`) plus two project-specific ones:

- **Performance** — measured speed or memory wins (include numbers)
- **Tooling** — dev-only changes that don't affect runtime (CI, lints, scaffolding)

### Commit style — enforced

Commit subjects follow **Conventional Commits**, and the convention is
**enforced**: a malformed subject is rejected at commit time (the `commit-msg`
git hook) and in CI (`.github/workflows/commit-convention.yml`). The validator
is `tools/check_commit_msg.py`. Activate the hooks once with
`./tools/install_hooks.sh`. Each commit should still be scoped to a single
logical change — phase-level work lands as a series of focused commits, not one
giant squash.

**Format:** `<type>(<scope>): <subject>`

**Rules (hard-blocked):**

- **`type`** — one of (lower-case): `feat` `fix` `perf` `refactor` `revert`
  `remove` `deprecate` `security` `docs` `style` `test` `build` `ci` `chore`
  `release`. (Note the full names `deprecate` / `security`.)
- **`scope`** — *optional*; a lower-case **dotted source path** mirroring the
  tree (`models.text.bert`, `nn.functional`, `compile`, `gpu`, `weights`,
  `utils.transforms`, `tools`, …). Comma-separated multi-scope allowed. Must
  match `[a-z0-9._,-]+`.
- **`!`** — *optional* breaking-change marker: `type(scope)!: …` (or a
  `BREAKING CHANGE:` footer).
- **`subject`** — non-empty, **no trailing period**, header **≤ 100 chars**
  (≤ 72 recommended → warning).
- Merge / `Revert "…"` / `fixup!` / `squash!` commits are exempt.
- Bypass one local commit (discouraged): `LUCID_SKIP_COMMIT_CONVENTION=1 git
  commit …` or `--no-verify` — CI still checks it.

The first eight types (`feat` … `security`) are *user-facing* and feed the
CHANGELOG pipeline (see *Adding a changelog entry* above): the `commit-msg` hook
warns if such a commit doesn't touch `CHANGELOG.md`, and the `post-commit` hook
auto-folds an entry via `tools/changelog.py`.

Examples:

```
feat(models.text.bert): add SQuAD v1.1 fine-tuned weights
fix(nn.functional.grid_sample): honor mode / padding_mode
perf(gpu)!: fuse layernorm kernel (breaking buffer layout)
ci: enforce strict commit convention
```

---

### API stability

Semantic Versioning is a promise to the people running Lucid: code written
against one 3.x release runs on every later 3.x. A minor or patch release may
add names and optional parameters. It may not remove a public name, rename or
reorder a parameter, make one keyword-only, or make an optional one required.
3.x broke this in minor releases — `mobilenet_v1` became `mobilenet`,
`GenerationMixin` became `CausalLMMixin`, 43 names left `lucid.models` — and
code written a month earlier stopped importing.

- **A test holds the surface.** `lucid/test/audit/released_surface.json`
  records every public name in the last release — the audited namespaces,
  `lucid.models`, and the public methods of public classes — with the
  parameters a caller could pass. `lucid/test/unit/audit/test_released_surface.py`
  fails when a released call would stop binding. Defaults and annotations are
  not recorded, since no call site spells them; a changed default is still a
  change of behaviour and goes under `Changed`.
- **Retire a name, don't delete it.** Keep the old spelling working and mark
  it with `lucid._deprecation.deprecated(since=, removal=, alternative=)` — a
  renamed function becomes a thin wrapper, a renamed class a subclass of its
  replacement. Every use raises `LucidDeprecationWarning`, a `FutureWarning`
  so the person running the code sees it. `removal` is at least two minor
  releases after `since`, or the next major version; the helper refuses
  anything sooner. The deprecation goes under `Deprecated` in the CHANGELOG,
  the removal under `Removed` in the release it reaches.
- **The release commit rewrites the snapshot**, after bumping the version:
  `python -m lucid.test.audit._released --update`. It refuses while any
  released call would break. Nothing else writes the file.

### Linking work to the backlog

The backlog lives in Linear (team key `CHA`), connected to this repository.
Name the issue in the commit body — `Refs CHA-5` for work toward it,
`Fixes CHA-5` when the commit resolves it. The integration links the commit
to the issue, and a `Fixes` reaching `main` closes it. A finding that does not
fit the change at hand goes into a new issue rather than a TODO comment.

## 9. Pull request checklist

Before opening a PR, verify every item:

- [ ] Hard rules H1–H13 are not violated (run the detection commands from §3)
- [ ] `ruff check lucid/` passes
- [ ] `mypy --strict lucid/` passes (error count must not increase)
- [ ] `pytest lucid/test/ -q` passes, or the documented sequential tiers cover the suite with explicit skips
- [ ] C++ Google Test suite passes (if C++ was modified)
- [ ] `tools/check_layers.py` passes (no new layer dependency violations)
- [ ] `tools/check_stubs.py` passes (stubs are up to date)
- [ ] `tools/check_doxygen.py` passes at 100% if C++ public API was added
- [ ] `bash tools/check_format.sh` passes (if C++ was modified)
- [ ] Unit test added for new public API
- [ ] Parity test added for new public API (via `ref` fixture)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No released call stops binding (`test_released_surface.py`); a name on its way out is marked with `lucid._deprecation.deprecated`
- [ ] No `torch` / `pytorch` / `cuda` in any new or modified source
- [ ] No `from __future__ import annotations` introduced
- [ ] No `*args` / `**kwargs` in any new `.pyi` signatures
- [ ] No external library imports outside the 6 permitted bridge boundaries

---

## 10. Out-of-scope

The following will not be accepted in this major version:

- Linux / Windows / x86_64 cross-platform support
- CUDA or NCCL distributed training
- ONNX export (Lucid's own `.lucid` format is in scope; ONNX is not)
- TorchScript / FX graph
- `__torch_function__` protocol

PRs adding any of these will be closed.

Quantization, `lucid.compile`, Core ML export and multi-process `DataLoader`
already exist. Their supported cases and limits must be checked against current
APIs and scoped tests; older "future phase" descriptions are not current policy.

---

## Questions?

If something in this guide is unclear or contradicts the code, the code is the ground truth. Open an issue and ask — do not guess.
