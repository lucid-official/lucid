# Lucid test suite

A production-grade test infrastructure for the Lucid ML framework.
Every op is exercised under a single canonical template (shape /
value / dtype / device / edge / backward / determinism), every device
stream (CPU + Metal) is validated for numerical agreement, and every
numerically meaningful op has a parity check against a reference
framework.

## Layout

```
lucid/test/
├── _fixtures/    fixture modules (devices, dtypes, tensors, ref, perf)
├── _helpers/     comparison / seeding / shapes / gradcheck / golden I/O
├── unit/         focused per-op CPU+GPU behaviour
├── parity/       value comparison vs the reference framework
├── numerical/    invariants / stability / precision / golden tensors
├── integration/  multi-component flows (train loops, checkpoints)
├── perf/         pytest-benchmark powered timing (opt-in)
├── stubs/        type-stub regression
└── conftest.py   global wiring, autouse seed
```

## Markers

| Marker | Use |
|---|---|
| `smoke` | < 1 s sanity check |
| `gpu` | Metal-only; collect-skip when unavailable |
| `parity` | Reference-framework parity; collect-skip when missing |
| `slow` | > 5 s (integration / training loops) |
| `perf` | Benchmark; opt-in via `--benchmark-only` |
| `stability` | Extreme-value / numerical-edge |
| `f64_only` | Float64-only path |

## Running

```bash
# fast tier (unit + numerical + stubs; no reference framework)
pytest lucid/test/ \
    --ignore=lucid/test/parity \
    --ignore=lucid/test/integration \
    --ignore=lucid/test/perf

# GPU-only sweep (Apple Silicon)
pytest lucid/test/ -m gpu

# parity tier (auto-skips when reference framework isn't installed)
pytest lucid/test/parity/

# integration tier (training loops + checkpoint round-trip)
pytest lucid/test/integration/

# perf tier — install ``pytest-benchmark`` for full timing tables
pytest lucid/test/perf/ -m perf
pytest lucid/test/perf/ --benchmark-only          # with pytest-benchmark
```

`scripts/ci_full.sh` runs all four tiers in order and treats parity /
perf failures as warnings, so a fresh checkout without the reference
framework or `pytest-benchmark` still passes the gate.

## Reference framework

Parity tests use a reference ML framework as ground truth. The import
is **lazy and string-concat-obfuscated** so the literal name doesn't
appear in Lucid source (H5). End users who never run `pytest` never
trigger this import — Lucid stays usable without `pip install <ref>`.

When the reference isn't installed, every test under `parity/` is
auto-skipped at collection time with a single message.

## Determinism

`conftest.py` sets `lucid.manual_seed(0)` automatically before every
test (autouse fixture). When a test additionally needs the reference
framework's RNG seeded the same way, use
`lucid.test._helpers.seeding.seed_all(0, ref=ref)`.

## CPU + GPU coverage

The `device` fixture parametrizes over `["cpu", "metal"]` whenever
Metal is detected. Most op-level tests don't have to do anything
special — they pull `device` and the cross-product runs automatically.

For tests that need to compare CPU vs GPU output of the *same* op
(detecting "device drift"), use the `cross_device_pair` fixture.

## Correctness matrices

Three tests share one table of small cases — every compile emitter, over
float32 / int64 / int32 / bool — in `unit/compile/_op_matrix.py`. They run
in the fast tier, about half a minute together.

| test | holds | against |
|---|---|---|
| `unit/compile/test_op_dtype_replay.py` | `lucid.compile` output, traced on one input and replayed on another | eager on the second input |
| `unit/compile/test_op_grad_replay.py` | `make_step` gradients through each op | eager backward |
| `unit/ops/test_numpy_oracle.py` | eager, on CPU and on Metal | numpy (`unit/ops/_numpy_oracle.py`) |

Values are read through `Tensor.numpy()` on the tensor as produced — the
path that once reported compiled integer outputs as float32. A fallback to
eager is allowed only where the harness lists it with a reason, so a graph
that stops compiling fails the day it happens — and a listed case that
compiles fails too, so the lists cannot go stale.

`test_op_grad_replay.py` runs every case twice: once as training runs, and
once with `LUCID_MANUAL_VJP_REQUIRE=1`, where MPSGraph's autodiff may not
stand in for a missing VJP. A train-mode batch norm or an interpolation
anywhere in a model rules that autodiff out, so an op without a VJP sent
the whole model's step eager — twenty ops were in that state and nothing
said so.

Combinations the single-op cases cannot see are covered by a few composite
cases (pad then pool, pool or resize then transpose) and, per model, by
`unit/compile/_zoo_matrix.py`: the smallest factory of every zoo family,
shrunk through its config, compiled with `make_step` and compared with eager
in eval and in train mode (loss, every gradient, batch-norm buffers). It is
a harness, not a test — one family per process, since an MPSGraph abort
kills the interpreter:

```
python -m lucid.test.unit.compile._zoo_matrix <family> <task>
```

Agreement between two NaNs is no agreement, so a family whose loss or
gradients are not finite, or whose gradients are all zero, is `nonfinite`
and fails whatever `EXPECTED` says (`_zoo_matrix.nonfinite`, pinned by
`unit/compile/test_zoo_matrix_status.py` without running the matrix). A
density over a bounded domain gets inputs inside it (`Spec.domain`).

Adding a case means adding it to `CASES` and giving it a numpy reference in
`REFS`, or a reason in `NO_REF` — `test_every_case_is_accounted_for` fails
otherwise. numpy is used there only to compute expected values (H4 covers
the compute path, not the test tree).

## Adding a new test

Drop the file under the right `unit/<area>/` subdirectory. Match the
canonical template:

```python
import pytest
import lucid

class TestMyOp:
    def test_basic_shape(self, device, float_dtype): ...
    def test_known_values(self, device, float_dtype): ...
    def test_broadcasting(self, device): ...
    def test_zero_dim(self, device): ...

    @pytest.mark.stability
    def test_inf_nan(self, device): ...

    def test_backward(self, device): ...
    def test_gradcheck(self, device): ...
    def test_reproducible(self, device): ...
    def test_cpu_gpu_match(self): ...
```

When the op is also expected to match the reference framework, add a
mirror file under `parity/<same-path>/`:

```python
@pytest.mark.parity
class TestMyOpParity:
    def test_forward(self, ref, device, float_dtype): ...
    def test_backward(self, ref, device): ...
```
