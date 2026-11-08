"""A rank-5+ graph output is materialised at rank 4 and wrapped at its logical shape (LCD-920c).

MPSGraph runs the copy of a rank >= 5 result into its output buffer as a loop over the dims
above the last four, even when that result is only a reshape of a faster value: a rank-5
feed -> rank-4 add -> rank-5 output took 27.7 ms against 2.2 ms with a rank-4 output.
``MpsBuilder`` now names such an output through a view at the canonical rank (leading axes
merged until the rank is 4); ``run_executable`` binds the result ``MPSGraphTensorData`` at
the executable's own target shape and only the host wrap uses the logical one.  The gate is
``rank_canon_wants`` (lead >= ``LUCID_COMPILE_RANK_CANON_MIN_LEAD``, default 64).

Guards:

* values: rank 5 and 6 outputs equal eager's in float32 / float16 / bfloat16, with the shape
  the user wrote, both as a reshape result and as a computed value, and both gate
  settings (default and ``0``, where every rank-5+ output is merged);
* in-place writes: a rank-5 parameter's new value, momentum / Adam state, and gradient are
  right through ``make_step``, ``fused_step`` and ``compile_optimizer`` (the swap-buffer
  paths keep their checks on the logical shape);
* an AOT export / import round trip returns the same bits;
* a symbolic batch is left as it was;
* a coarse timing bound against the rank-4 twin.
"""

import os
import time
from collections.abc import Callable, Iterator
from unittest import mock

import pytest

import lucid
import lucid.compile as lc
import lucid.nn as nn
import lucid.optim as optim
from lucid.compile import compile_optimizer, fused_step, make_step
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, to_metal

ENV = "LUCID_COMPILE_RANK_CANON_MIN_LEAD"

pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="compile needs Metal"),
    # A fallback to eager would make every compiled == eager row below vacuous.
    pytest.mark.filterwarnings("error::lucid.compile._core.fallback.CompileFallbackWarning"),
]


@pytest.fixture(autouse=True)
def _default_gate() -> Iterator[None]:
    """Every test starts from the default lead threshold, whatever the shell exports."""
    with mock.patch.dict(os.environ):
        os.environ.pop(ENV, None)
        yield


def _filled(shape: tuple[int, ...], phase: float = 0.0) -> lucid.Tensor:
    n = 1
    for s in shape:
        n *= s
    return (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 0.37 + phase) + 1.5).reshape(*shape)


def _equal(got: lucid.Tensor, want: lucid.Tensor, why: str, tol: float = 0.0) -> None:
    assert tuple(got.shape) == tuple(want.shape), why
    g, w = got.to(lucid.float32).reshape(-1).tolist(), want.to(lucid.float32).reshape(-1).tolist()
    worst = max((abs(a - b) for a, b in zip(g, w, strict=True)), default=0.0)
    assert worst <= tol, f"{why}: max abs diff {worst}"


# ── values ────────────────────────────────────────────────────────────────────────────

_DTYPES = [lucid.float32, lucid.float16, lucid.bfloat16]

#: ``(input shape, output shape)``: rank-4 -> rank-5/6 and rank-5/6 -> same, all above the
#: default lead gate (lead 64 / 128).
_OUTPUTS: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    ((64, 6, 5, 4), (64, 2, 3, 5, 4)),
    ((64, 2, 3, 5, 4), (64, 2, 3, 5, 4)),
    ((8, 8, 3, 5, 4), (8, 8, 3, 5, 4)),
    ((128, 6, 10), (8, 8, 2, 3, 5, 4)),
]


@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d))
@pytest.mark.parametrize(
    "shapes", _OUTPUTS, ids=lambda s: "->".join("x".join(map(str, v)) for v in s)
)
def test_a_rank_5_or_6_output_equals_eager(
    shapes: tuple[tuple[int, ...], tuple[int, ...]], dtype: lucid.dtype
) -> None:
    src, dst = shapes

    def reshaped(a: lucid.Tensor) -> lucid.Tensor:
        return (a * 2.0 + 1.0).reshape(*dst)

    def computed(a: lucid.Tensor) -> lucid.Tensor:
        return (a.reshape(*dst) * 0.5 - 0.25) * a.reshape(*dst)

    x = _filled(src, 0.3).to(dtype).to(COMPILE_DEVICE)
    tol = 0.0 if dtype == lucid.float32 else 1e-2
    for name, fn in (("reshape result", reshaped), ("computed", computed)):
        for gate in (None, "0"):
            with mock.patch.dict(os.environ):
                if gate is not None:
                    os.environ[ENV] = gate
                got = lucid.compile(fn)(x)  # a fresh executable per gate setting (LCD-922)
            want = fn(x)
            assert got.dtype == dtype
            _equal(got, want, f"{name} gate={gate}", tol)


@pytest.mark.usefixtures("inference")
def test_a_rank_5_output_keeps_the_shape_the_user_wrote_and_is_usable() -> None:
    x = _filled((64, 6, 5, 4)).to(COMPILE_DEVICE)

    def fn(a: lucid.Tensor) -> lucid.Tensor:
        return (a + 1.0).reshape(64, 2, 3, 5, 4)

    got = lucid.compile(fn)(x)
    assert tuple(got.shape) == (64, 2, 3, 5, 4)
    # A downstream eager op reads the wrapped buffer at the logical shape.
    _equal(got.sum(dim=1), fn(x).sum(dim=1), "sum over axis 1", 1e-4)
    _equal(got[3, 1], fn(x)[3, 1], "indexed slice")


@pytest.mark.usefixtures("inference")
def test_a_passthrough_rank_5_feed_is_returned_unchanged() -> None:
    x = _filled((64, 2, 3, 5, 4)).to(COMPILE_DEVICE)
    y = lucid.compile(lambda a: (a, a * 2.0))(x)
    _equal(y[0], x, "passthrough")
    _equal(y[1], x * 2.0, "computed")


# ── in-place writes ───────────────────────────────────────────────────────────────────

_PSHAPE = (64, 2, 3, 4, 5)


class _Fit(nn.Module):
    """Pulls a rank-5 parameter towards ``1 / x``, weighted element by element."""

    def __init__(self, shape: tuple[int, ...], dtype: lucid.dtype) -> None:
        super().__init__()
        self.p = nn.Parameter(_filled(shape, 0.1).to(dtype).to(COMPILE_DEVICE))
        self.register_buffer("w", _filled(shape, 0.9).to(dtype).to(COMPILE_DEVICE))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return ((self.p * x - 1.0) ** 2 * self.w).sum()


_OPTIMIZERS: dict[str, Callable[[list[nn.Parameter]], optim.Optimizer]] = {
    "SGD-momentum": lambda ps: optim.SGD(ps, lr=0.05, momentum=0.9),
    "Adam": lambda ps: optim.Adam(ps, lr=0.01),
}


def _loss(out: lucid.Tensor) -> lucid.Tensor:
    return out


def _state(opt: optim.Optimizer) -> list[lucid.Tensor]:
    out: list[lucid.Tensor] = []
    for st in opt.state.values():
        for v in st.values():
            if isinstance(v, lucid.Tensor):
                out.append(v)
    return out


def _train(entry: str, model: _Fit, opt: optim.Optimizer, x: lucid.Tensor, steps: int) -> None:
    if entry == "fused":
        fused = fused_step(model, _loss, opt)
        for _ in range(steps):
            fused(x)
    elif entry == "compile_optimizer":
        copt = compile_optimizer(opt)
        for _ in range(steps):
            copt.zero_grad()
            model(x).backward()
            copt.step()
    elif entry == "make":
        step = make_step(model, _loss)
        for _ in range(steps):
            opt.zero_grad()
            step(x).backward()
            opt.step()
    else:
        for _ in range(steps):
            opt.zero_grad()
            model(x).backward()
            opt.step()


@pytest.mark.parametrize("gate", [None, "0"], ids=["default", "gate0"])
@pytest.mark.parametrize("opt_name", list(_OPTIMIZERS))
@pytest.mark.parametrize("entry", ["fused", "compile_optimizer", "make"])
def test_a_rank_5_parameter_trains_like_eager(entry: str, opt_name: str, gate: str | None) -> None:
    x = lucid.tensor([0.5]).to(COMPILE_DEVICE)
    with mock.patch.dict(os.environ):
        if gate is not None:
            os.environ[ENV] = gate
        model = _Fit(_PSHAPE, lucid.float32)
        opt = _OPTIMIZERS[opt_name](list(model.parameters()))
        _train(entry, model, opt, x, 3)
    ref = _Fit(_PSHAPE, lucid.float32)
    ref_opt = _OPTIMIZERS[opt_name](list(ref.parameters()))
    _train("eager", ref, ref_opt, x, 3)

    assert tuple(model.p.shape) == _PSHAPE
    _equal(model.p, ref.p, "parameter", 1e-5)
    got_state, ref_state = _state(opt), _state(ref_opt)
    assert len(got_state) == len(ref_state)
    for k, (g, r) in enumerate(zip(got_state, ref_state, strict=True)):
        if g.shape == r.shape and g.ndim >= 1:
            _equal(g, r, f"state[{k}]", 1e-5)


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=lambda d: str(d))
def test_a_half_precision_rank_5_gradient_is_written_without_an_assertion(
    dtype: lucid.dtype,
) -> None:
    """An F16/BF16 result bound at a rank other than its target's asserts in MPSGraph."""
    x = lucid.tensor([0.5]).to(dtype).to(COMPILE_DEVICE)
    for gate in (None, "0"):
        with mock.patch.dict(os.environ):
            if gate is not None:
                os.environ[ENV] = gate
            model = _Fit(_PSHAPE, dtype)
            step = make_step(model, _loss)
            loss = step(x)
            loss.backward()
        ref = _Fit(_PSHAPE, dtype)
        want = _loss(ref(x))
        want.backward()
        assert model.p.grad is not None and ref.p.grad is not None
        assert tuple(model.p.grad.shape) == _PSHAPE
        scale = max(abs(v) for v in ref.p.grad.to(lucid.float32).reshape(-1).tolist())
        _equal(model.p.grad, ref.p.grad, f"grad gate={gate}", 5e-2 * scale)


@pytest.mark.parametrize("gate", [None, "0"], ids=["default", "gate0"])
def test_a_rank_5_parameter_gradient_equals_eager(gate: str | None) -> None:
    x = lucid.tensor([0.5]).to(COMPILE_DEVICE)
    with mock.patch.dict(os.environ):
        if gate is not None:
            os.environ[ENV] = gate
        model = _Fit(_PSHAPE, lucid.float32)
        step = make_step(model, _loss)
        step(x).backward()
    ref = _Fit(_PSHAPE, lucid.float32)
    _loss(ref(x)).backward()
    assert model.p.grad is not None and ref.p.grad is not None
    assert tuple(model.p.grad.shape) == _PSHAPE
    _equal(model.p.grad, ref.p.grad, "grad", 1e-4)


# ── AOT export / import ───────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("gate", [None, "0"], ids=["default", "gate0"])
def test_an_aot_round_trip_of_a_rank_5_output_returns_the_same_bits(
    tmp_path: object, gate: str | None
) -> None:
    x = _filled((64, 2, 3, 5, 4), 0.2).to(COMPILE_DEVICE)

    def fn(a: lucid.Tensor) -> lucid.Tensor:
        return a * 2.0 + 1.0

    with mock.patch.dict(os.environ):
        if gate is not None:
            os.environ[ENV] = gate
        compiled = lc.compile(fn)
        want = compiled(x)
        path = os.path.join(str(tmp_path), "r5")
        assert lc.save_compiled(compiled, path) is True
        restored = lc.load_compiled(path)
        got = restored(x)
    assert tuple(got.shape) == (64, 2, 3, 5, 4)
    _equal(got, want, "restored", 0.0)
    _equal(got, fn(x), "eager", 1e-6)


# ── dynamic batch ─────────────────────────────────────────────────────────────────────


class _Rank5Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(12, 12)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y = self.lin(x)
        return y.reshape(y.shape[0], 2, 3, 2, 1)


@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("gate", [None, "0"], ids=["default", "gate0"])
def test_a_dynamic_batch_rank_5_output_is_unchanged(gate: str | None) -> None:
    with mock.patch.dict(os.environ):
        if gate is not None:
            os.environ[ENV] = gate
        m = to_metal(_Rank5Head()).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (4, 9, 16):
            x = _filled((bs, 12), 0.1 * bs).to(COMPILE_DEVICE)
            got = cm(x)
            assert tuple(got.shape) == (bs, 2, 3, 2, 1)
            _equal(got, m(x), f"bs={bs}", 1e-5)


# ── time ──────────────────────────────────────────────────────────────────────────────


def _ms(fn: Callable[[lucid.Tensor], lucid.Tensor], x: lucid.Tensor) -> float:
    cm = lucid.compile(fn)
    for _ in range(3):
        cm(x)
    best = float("inf")
    for _ in range(15):
        t0 = time.perf_counter()
        float(cm(x).reshape(-1)[0].item())
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


@pytest.mark.usefixtures("inference")
def test_a_rank_5_output_is_not_slower_than_its_rank_4_twin() -> None:
    """Unfixed: 26 ms against 2 ms.  Each rank-5 output is held to its own rank-4 twin.

    The twin differs from the case in the output alone, so a cost the *input* side pays
    (a rank-5 feed) is in both.
    """
    r5 = (6144, 16, 2, 16, 2)
    r4 = (6144, 16, 2, 32)
    x5, x4 = _filled(r5).to(COMPILE_DEVICE), _filled(r4).to(COMPILE_DEVICE)

    # Wall-clock on a shared GPU: keep each case's best over up to five rounds and stop at
    # the first round where both ratios hold.  The unfixed output is ~10x its twin, so a
    # noisy round cannot hide the defect.
    cases: dict[str, tuple[Callable[[lucid.Tensor], lucid.Tensor], lucid.Tensor]] = {
        "twin5": (lambda a: (a + 1.0).reshape(*r4), x5),
        "out5": (lambda a: a + 1.0, x5),
        # A graph output that is a reshape of a computed value costs a rank-4 reshape too
        # (+25% here, whatever the rank), so the twin is a rank-4 reshape.
        "twin4": (lambda a: (a + 1.0).reshape(98304, 2, 16, 2), x4),
        "via_reshape": (lambda a: (a + 1.0).reshape(*r5), x4),
    }
    best = dict.fromkeys(cases, float("inf"))
    for _ in range(5):
        for name, (fn, x) in cases.items():
            best[name] = min(best[name], _ms(fn, x))
        if best["out5"] <= 1.3 * best["twin5"] and best["via_reshape"] <= 1.3 * best["twin4"]:
            return
    assert best["out5"] <= 1.3 * best["twin5"], f"{best['out5']:.2f} vs {best['twin5']:.2f} ms"
    assert best["via_reshape"] <= 1.3 * best["twin4"], (
        f"{best['via_reshape']:.2f} vs {best['twin4']:.2f} ms"
    )
