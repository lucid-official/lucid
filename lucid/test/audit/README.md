# `lucid-audit` — the whole diagnosis, one command

Four stages, one verdict, one exit code.

| stage | asks | catches |
|---|---|---|
| **self-check** | can the instruments still go red? | one or more negative controls for every axis, including an isolated fatal-call probe |
| **sweep** | does each reachable symbol keep its contract? | the runtime public-symbol census × all applicable axes |
| **suite** | are the specific values the right values? | `pytest lucid/test` + a line-coverage floor |
| **doctests** | does the documentation run? | executable examples + a per-module floor |

They fail independently, and none is a substitute for another.

Use the report's counts instead of historical totals. `python -m tools.support_manifest`
can attach a scoped audit report and link exports to source, stubs, generated
documentation and static test references. References are not proof that tests ran.
The smoke self-check terminates only a dedicated subprocess with a known exit
code; a successful child or an unrelated import failure does not count as caught.

The self-check runs **first**, on a clean interpreter, and the order is
load-bearing: run after the sweep, four mutants stopped being caught,
because the sweep leaves device and module state behind. An instrument
check the run itself perturbs proves nothing.

Measured on one session's defects: the sweep found the gradient that was
never wired and the sampler drawing at the wrong concentration; the
suite found the assignment writing a rectangle instead of a diagonal,
the histogram binning onto the wrong grid, the transform whose inverse
was NaN, and the window that deleted its own frame. A gate that runs
only one of them reports "clean" over half a framework.

```bash
lucid-audit                 # every stage, one verdict     (~15 min)
lucid-audit --audit-only    # self-check + sweep
lucid-audit --tests-only    # suite + line coverage + doctests
lucid-audit --self-check    # the instruments alone, seconds
```

```
── verdict ──────────────────────────────────────────────
  blind axes                      0
  audit defects                   0
  audit coverage regressions      0
  suite failures                  0
  line coverage regressions       0
  docstring regressions           0

  clean on every stage that ran
```

Exit is `0` only when every stage that ran is clean, so this works as a
gate without reading the output. `2` means the harness itself broke —
distinct from `1`, which means the framework did.

### Why the suite runs in a subprocess

By the time the sweep is done the audit has imported the whole package
and, on some axes, patched parts of it. Collecting the suite into that
interpreter would let one stage's state decide the other stage's result,
which is the one thing a gate made of two independent checks must not
allow.

### Why line coverage is measured by default

Because it is free. The suite takes 9m09s uninstrumented and 8m29s under
`coverage` — the wall clock is MLX and Accelerate, not Python line
tracing. A floor that costs nothing to check should be checked on every
run rather than remembered. `--no-line-coverage` turns it off anyway.

The floor lives in `suite.json` and is compared per module, with a
2-point tolerance and a 20-statement minimum: adding statements to a
well-covered file lowers its percentage honestly, and a gate that fires
on that is a gate people learn to ignore. Record a new floor
deliberately:

```bash
lucid-audit --tests-only --update-suite
```

---

## Setup on a fresh machine

From the **repository root** (the directory holding `pyproject.toml`):

```bash
cd /path/to/lucid

# 1. Build the engine and install the package.  The engine links the MLX
#    build pinned in pyproject.toml [tool.lucid.mlx], which the build
#    fetches and stages under ~/Library/Caches/lucid/mlx-metal; the target
#    is that build's macOS floor.
#    ⚠️ `uv pip`, never plain `pip` — .venv has no pip, so `pip` leaks to
#    the system pip3 and installs into that interpreter instead.
#    See [[debug-build-wrong-venv-rpath]].
VIRTUAL_ENV=.venv MACOSX_DEPLOYMENT_TARGET=$(python3 tools/fetch_mlx.py --print-macos) \
    uv pip install -e ".[audit]" --no-build-isolation

# 2. Confirm the engine links the pinned MLX (liblucid_mlx, every symbol
#    present, built for the floor).
.venv/bin/python tools/check_engine_link.py

# 3. Confirm the tool is on PATH and the surface enumerates.
lucid-audit --coverage
```

`--coverage` runs no probes at all — it walks the package and prints what
the sweep *can* reach. It is the fastest way to tell a working install
from a broken one.

### What the `[audit]` extra adds

`rich`, `coverage` and `pytest`. `rich` buys the live per-subsystem
display and nothing else — without it the sweep runs identically on a
stdlib ANSI console, because a correctness tool should not fail to start
over a presentation dependency. `coverage` backs the line-coverage floor
and `pytest` runs the suite stage; without either, that stage degrades
rather than fails: it says what is missing and the sweep still reports.

### If `lucid-audit` is not found

The console script is registered at install time, so it appears only
after step 1. Until then — and always, as a fallback — this is the same
program:

```bash
python -m lucid.test.audit --coverage
```

### Requirements

Inherited from Lucid itself: macOS 26+ on Apple Silicon, Python 3.14,
MLX ≥ 0.31. There is nothing the audit needs beyond what building the
engine already needs.

---

## Running it

```bash
lucid-audit                      # everything; minutes
lucid-audit --quick              # fewer domains, smaller probes
lucid-audit --json report.json   # machine-readable, with every SKIP reason
lucid-audit --fail-fast          # stop at the first defect
```

Scoping, when you are working on one area:

```bash
lucid-audit --subsystem nn.functional
lucid-audit --subsystem linalg,fft --axis grad,grad2
lucid-audit --select 'conv|pool'
```

Exit status is `0` when no defect survived, `1` when one did, `2` when
the harness itself broke.

### The transcript

Every run mirrors itself to `lucid/test/audit.log`, without colour, as it
happens:

```bash
tail -f lucid/test/audit.log     # while it runs
lucid-audit --log /tmp/run.log   # somewhere else
lucid-audit --no-log             # not at all
```

Three properties it is built to have, each for a reason:

- **Written line by line, flushed each time.** The stages this gate is
  slowest in are the ones most likely to be killed before the end, and a
  buffered file loses exactly those. After a `SIGKILL` the log still ends
  at the axis the run died in, which is the only record of where it got
  to.
- **Complete under `--quiet`.** The screen is reduced to the summary; the
  transcript is not. A run worth sharing is worth recording in full.
- **Truncated per run, and not written by the informational modes.**
  `--list-axes` and its neighbours answer a question about the harness
  rather than running the gate, so they leave the last real run's log
  alone — otherwise the feature could destroy what it exists to preserve.

The log is git-ignored. It is the one file the gate writes into the
checkout on purpose; everything else it needs, including the coverage
scratch, goes to a temporary directory and is gone when the stage ends,
whether or not the run survived to clean up after itself.

### Through pytest

Every cell is also a collectable pytest case, deselected by default
because the sweep takes minutes:

```bash
pytest lucid/test/ -m audit
pytest lucid/test/audit/test_audit_smoke.py    # the harness's own tests, ~1s
```

---

## Reading the output

```
⠙ overall ━━━━━╺━━━━━━━━━━━━━━━━━  230/10841  16% 0:00:12 eta 0:04:31

  lucid         ━━━━━━╺━━━━━━━━━━ 230/807    221 ok     9 –
  special       ━━━━━━━━━━╺━━━━━━  53/114     47 ok    6 fail
```

One bar per subsystem, each with its own running counts. Defects print
above the live region as they are found.

### Statuses

| | |
|---|---|
| `PASS` | the axis's question was asked and answered correctly |
| `FAIL` | a defect |
| `TRNC` | the finite-difference probe was the limit, not the op — the disagreement shrank like `h²` when the step was refined |
| `GAUG` | correct only up to a symmetry (sign of an eigenvector, phase of an FFT, the constant a softmax is invariant to) |
| `VAC` | the check ran but could not have failed — reported separately because a vacuous pass reads as coverage and is not |
| `UNSP` | the op refused, loudly and by design |
| `SKIP` | **the harness could not build inputs.** Not "no problem" — "not checked" |
| `KNWN` | a defect accepted in `known.json` |

### Proving an axis can fail

```bash
lucid-audit --self-check
```

Breaks the framework on purpose, once per axis, in exactly the way that
axis exists to notice — a gradient that is not the derivative, a NaN that
gets swallowed, a handle whose `remove` does not remove — and reports
whether the axis said so. The 2026-09-20 validation caught 35/35 mutants
covering 34/34 axes, including constant/extreme inputs and a fatal-call
negative control in a dedicated subprocess. Always use the current report's
counts: an axis without a mutant is printed as **unproven**, not silently
treated as covered. A caught mutant proves sensitivity to that deliberate
defect, not exhaustive correctness of every operation on the axis.

It is not decoration. Three findings so far, all about this tool rather
than the framework:

* `broadcast` compared output *shapes* and never values, so an op that
  broadcast to the right extent and computed the wrong numbers passed all
  nine directions;
* the tokenizer round trip passed a decoder that silently dropped a
  character, on the grounds that dropping it was at least repeatable;
* `layout` **could not fail at all**. It compared a packed operand against
  a strided view, and the engine then materialised every view —
  `t[..., ::2]`, `T`, `expand`, `broadcast_to`, `diagonal` and `unfold`
  all came back packed — so the two operands were byte-identical. 688
  cells per run were reporting agreement between an op and itself. Those
  are views now and `t[..., ::2]` is the strided operand — on metal too,
  since metal slices became members of their base's view family
  (730ef06c5). If a slice ever comes back packed again, the axis says the
  layout half is vacuous instead of claiming agreement.

`SKIP` is the number to watch. The summary's *applicable cells* line
reports how many cells produced a verdict at all, and that is the honest
measure of what a run actually checked.

### `SKIP` is not `NOT_APPLICABLE`

They were conflated, and it cost the SKIP list its meaning. "The scale
sweep found no float output" is not a gap — `argmax` returns integers and
a scale sweep has nothing to measure on it. Nor is "the op takes a module
and returns a module": `nn.utils.weight_norm` is checked, thoroughly, by
the `nnutils` axis. Filing either under SKIP put ~500 cells into a work
queue that was already done, and buried the ones that were not.

The rule is the axis's *question*, not its outcome:

* the question cannot be asked here → `NOT_APPLICABLE`, and the cell
  leaves the denominator;
* the question applies and the harness could not set it up → `SKIP`, and
  it stays in the denominator as a debt.

### `UNSUPPORTED` is a claim about the op, so check it is one

`UNSP` reads as "the op refused, loudly and by design", and 318 cells
were making that claim about refusals the *probe* had provoked: a float64
tensor handed to the `fft` family, which accepts only `F16/F32/C64`; a
`Tensor` handed to `pad_packed_sequence`, which wants a `PackedSequence`;
and 135 cells where the axis read `.grad.numpy()` on a `None` and
reported its own `AttributeError`. The substitution now keeps the
original argument's dtype, the axes name a missing gradient as a missing
gradient, and a non-tensor output is `NOT_APPLICABLE` rather than a
refusal — 318 down to **zero**.

### An out-of-range index differs by device, on purpose

An index outside its axis is an error on the CPU and is isolated inside
the graph on Metal (policy B, LCD-228):

| | CPU | Metal |
|---|---|---|
| gather, `index_select`, `take`, `x[i]`, embedding, class-loss target | `IndexError` | NaN at that position (0 for an integer or bool result) |
| `scatter`, `scatter_add`, `scatter_reduce`, `index_add` / `_copy` / `_fill` / `_put_` | `IndexError` | the update is dropped; the base keeps its value |
| `one_hot` | `IndexError` | a zero row |

Refusing on Metal would mean reading the indices back to the host, a
pipeline stall on every gather, scatter and embedding. Neither device
reads or writes outside the buffer. An index into an *empty* axis is
decided from the shapes, so both devices raise. One known gap remains:
`x[i] = v` on Metal derives its positions through an integer gather, so
an out-of-range key writes position 0 instead of being dropped (LCD-209).
For the `device` axis,
a cell that reports `UNSP` with `IndexError` therefore means the probe
built an out-of-range index. That is a bug in the spec, not in the op.
The contract is pinned by `lucid/test/unit/ops/test_index_bounds_contract.py`.
Its owner is `gpu_axis_index` in `lucid/_C/backend/gpu/AxisIndex.h`.

---

## The three coverage numbers

They fail independently and are routinely conflated:

| | meaning |
|---|---|
| **reach** | fraction of public objects the surface enumerates. Checked against `independent_walk()`, which shares none of the enumeration's logic. Currently 100%. |
| **depth** | fraction of symbols with an axis that can actually fail, rather than only the smoke axis. |
| **verdict** | fraction of cells that answer rather than SKIP. **Only a real run knows this**, and it is well below the other two. Currently 75.7%, over 1,476 of 1,512 symbols. It *fell* from 81.7% when `--self-check` showed the layout axis could not fail: 688 cells moved from PASS to VACUOUS, and the lower number is the true one. |

"Mutating process state" is not a reason to stop, and was used as one for
79 symbols. Mutating state is what makes them checkable: seed twice and
the draws must match, set a default and a new tensor must follow it,
register a hook and it must fire — then `remove()` and it must stop.
That is the `state` and `hook` axes, and none of it is a numeric
comparison.

Reaching a symbol is not verifying it, and `depth` is an upper bound, not
a result. Quote the verdict rate from `--json` when you want to say how
much of the framework a run actually checked.

---

## Did my change break something?

An absolute count cannot answer that. An op that stops being reachable
moves one cell from `pass` to `unsupported` among fifteen hundred
already-unsupported ones, and nothing in the summary changes visibly.

So every run is diffed against `coverage.json`, which records which
`(axis, symbol)` cells produce a verdict:

```bash
lucid-audit                 # runs, then reports what moved
```

```
── coverage regressions · 1 ──────────────────────────────
  LOST  grad::lucid.expm1  (was pass, now unanswered)
```

The exit code is non-zero for a regression as well as for a defect, so
this works as a gate without reading the output. A cell that used to
answer and no longer does is the interesting direction: the op is still
exported, the audit simply cannot reach it any more — which is what a
refactor breaks without breaking a test.

A symbol that has left the public surface is not that, so its cells are
reported as `REMOVED` and do not fail the run. Removing a name is judged
by the released-surface gate (`released_surface.json`), not here: LCD-473
took out `lucid.pad` and `Tensor.pad` on purpose, and their 24 cells as
`LOST` held the gate red over a deliberate change (LCD-532). `entry::F.pad`
went with them, because the entry axis compares a symbol with its other
spellings and now only one is left. Nothing counts as removed while any
module fails to import, since the sweep skips an unimportable module
without a word.

Newly answered cells are reported too, as progress rather than failure.
When the new state is the intended one:

```bash
lucid-audit --update-coverage
```

Do that deliberately. Re-recording to make a red run go green silences
exactly the signal the file exists for.

Only verdicts are recorded. `SKIP` and `UNSUPPORTED` carry probe details
that move for reasons unrelated to the framework, and a baseline that
churns is a baseline nobody re-reads.

`--no-coverage-diff` turns the comparison off.

> `coverage.json` answers "is this still reachable". `known.json` answers
> "is this failure accepted". A failing cell belongs in the second; a cell
> that stopped being asked belongs in neither — fix it or record it as the
> new floor on purpose.

---

## Extending it

**A symbol is skipped.** Read the reason in the report. If the block is a
required parameter the derivation has no value for, add one entry to
`_autospec._BY_NAME` — that closes it for every other op spelling the
parameter the same way. Writing a per-op spec in `_specs.py` is the last
resort, not the first.

**A whole subsystem is thin.** `--coverage` prints per-subsystem depth. A
low number there means the symbols need an axis, not a spec — see
`_axes_data.py` for the shape of one.

**A scoped run crashes and the full one does not.** That is not a
selection bug. `--axis grad,grad2` SIGBUSes about three runs in five,
inside LAPACK, reached from `lucid.linalg.ldl_solve`; the full sweep
dies about one run in eight, and `ldl_solve` on its own survives 120 consecutive runs of the
same axis. An out-of-bounds *write* elsewhere lands harmlessly when the
other axes have already shaped the heap. Three such writes have been
found and closed at the probe end — `ctc_loss` reading past `targets`,
`lu_solve` reading past a flat `pivots`, `LPPool1d` unfolding a rank-2
input — and at least one remains unlocated. When a run dies, re-run
under `-X faulthandler` and bisect by axis, not by symbol: the symbol it
dies in is the victim, not the cause.

**A finding looks wrong.** Check the harness before the framework. Of the
first twelve failures the newest axes reported, twelve were the
instrument: an abstract base raising `NotImplementedError` by contract, a
word-level tokenizer handed a vocabulary of single letters, a regex
parameter filled with the literal `"constant"`, a scheduler resumed
without restoring the optimizer it drives.

---

## Layout

| file | |
|---|---|
| `__main__.py` | CLI, the run loop, the summary |
| `_surface.py` | what there is to audit; the recursive walk and its independent check |
| `_axes.py` | the numeric axes and the `Axis` contract |
| `_axes_stability.py` | scale sweeps, extreme values, strides, determinism |
| `_axes_subsystem.py` | distributions, diffeq, quantization, serialization, compile, smoke |
| `_axes_data.py` | transforms, data, tokenizers, schedulers |
| `_axes_state.py` | global state, hooks, dtype algebra, functional transforms, `nn.utils` |
| `_specs.py` | hand-written invocations, and the tier order |
| `_builders.py` | hand-built instances of classes whose arguments are related objects no signature describes (`LoraModel(model, config)`) |
| `_autospec.py` | invocations derived from signatures |
| `_probe.py` | domains, sampling, finite differences |
| `_console.py` | the stdlib ANSI display |
| `_console_rich.py` | the `rich` display, when installed |
| `_mutants.py` | the deliberate defects, one per axis, and `--self-check` |
| `_doctests.py` | the documentation stage and its per-module floor |
| `_result.py` | `Status`, `Finding`, `Report`, `Baseline` |
| `known.json` | accepted defects, reported as `KNWN` rather than `FAIL` |
