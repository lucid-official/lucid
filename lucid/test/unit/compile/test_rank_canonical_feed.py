"""A rank-5+ graph feed read only through views is declared at its views' rank (LCD-928).

MPSGraph materialises an in-graph ``reshape`` of a placeholder (``x * w`` over a rank-5 feed
runs at 1.24 ms against 0.65 for the same elements fed flat), where the same reshape of a computed
tensor is an alias.  ``canonical_feed_shapes`` (``lucid/_C/compile/RankCanon.h``) names the
rank-5+ feeds read by nothing but pure views, the builder declares their placeholders at the
rank <= 4 shape most of those views produce, and the executable binds the caller's buffer at that
shape.  The logical shape stays everywhere outside the executable.

What is held here:

* values: the result equals eager's and the unrewritten graph's, at three dtypes and over several
  calls with different buffers (the bound-tensor cache), for a feed the pass declares at another
  shape and for the feeds it must leave alone (a feed also read by a permute, a reduction, or
  returned whole; a parameter);
* a ``make_step`` gradient through a rank-5 feed and parameter;
* a symbolic batch keeps its feeds as they are;
* an AOT round trip binds the restored executable's feed at the shape it was declared at;
* the cache key is unchanged: the same executable serves a second call and a second tensor;
* a rank-5 feed is no slower than its rank-4 twin.
"""

import os
import time
from collections.abc import Callable, Iterator
from unittest import mock

import pytest

import lucid
import lucid.compile as lc
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile import make_step
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, to_metal

ENV = "LUCID_COMPILE_RANK_CANON_MIN_LEAD"

needs_metal = pytest.mark.skipif(not metal_available(), reason="compile needs Metal")

# A fallback to eager would make every compiled == eager row below vacuous.
pytestmark = pytest.mark.filterwarnings(
    "error::lucid.compile._core.fallback.CompileFallbackWarning"
)

Shape = tuple[int, ...]


@pytest.fixture(autouse=True)
def _default_gate() -> Iterator[None]:
    with mock.patch.dict(os.environ):
        os.environ.pop(ENV, None)
        yield


def _filled(shape: Shape, phase: float = 0.0) -> lucid.Tensor:
    n = 1
    for s in shape:
        n *= s
    return (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 0.37 + phase) + 1.5).reshape(*shape)


def _on(t: lucid.Tensor, dtype: lucid.dtype = lucid.float32) -> lucid.Tensor:
    return t.to(COMPILE_DEVICE).to(dtype)


def _values(t: lucid.Tensor) -> list[float]:
    return t.to(lucid.float32).reshape(-1).tolist()


def _alike(a: float, b: float) -> bool:
    return a == b or (a != a and b != b)


def _same(got: lucid.Tensor, want: lucid.Tensor, why: str) -> None:
    g, w = _values(got), _values(want)
    assert len(g) == len(w), why
    bad = [i for i, (a, b) in enumerate(zip(g, w, strict=True)) if not _alike(a, b)]
    assert not bad, f"{why}: {len(bad)} values differ, first at {bad[0]}: {g[bad[0]]} != {w[bad[0]]}"


def _close(got: lucid.Tensor, want: lucid.Tensor, tol: float, why: str) -> None:
    g, w = _values(got), _values(want)
    assert len(g) == len(w), why
    for i, (a, b) in enumerate(zip(g, w, strict=True)):
        assert _alike(a, b) or a == pytest.approx(b, rel=tol, abs=tol), f"{why}[{i}]: {a} != {b}"


_DTYPES = [lucid.float32, lucid.float16, lucid.bfloat16]
_TOL = {lucid.float32: 3e-4, lucid.float16: 4e-3, lucid.bfloat16: 3e-2}

R5: Shape = (6, 4, 2, 3, 2)
R6: Shape = (3, 2, 2, 3, 2, 2)


def _compiled(fn: Callable[..., lucid.Tensor], gate: str | None = None) -> Callable[..., lucid.Tensor]:
    with mock.patch.dict(os.environ):
        if gate is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = gate
        _C_engine.compile.session_cache_clear()
        return lucid.compile(fn)


# ── values ────────────────────────────────────────────────────────────────────────────

#: Each reads its rank-5+ feeds through views only (the pass wraps every op).
_VIA_VIEWS: dict[str, Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]] = {
    "mul": lambda x, w: x * w,
    "where": lambda x, w: lucid.where(x > w, x, w * 2.0),
    "broadcast": lambda x, w: x + w[:1] * 0.5,
    "chain": lambda x, w: lucid.exp(lucid.sin(x) * w) + x,
}


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("shape", [R5, R6], ids=["rank5", "rank6"])
@pytest.mark.parametrize("name", sorted(_VIA_VIEWS))
def test_a_feed_read_through_views_matches_eager_and_the_unrewritten_graph(
    name: str, shape: Shape, dtype: lucid.dtype
) -> None:
    fn = _VIA_VIEWS[name]
    x, w = _on(_filled(shape, 0.1), dtype), _on(_filled(shape, 0.9), dtype)
    want = fn(x, w)
    # The gate at 0 makes the pass rewrite these small shapes; -1 is the unrewritten graph.
    on, off = _compiled(fn, "0"), _compiled(fn, "-1")
    _close(on(x, w), want, _TOL[dtype], "vs eager")
    _close(on(x, w), off(x, w), _TOL[dtype], "vs unrewritten")


@needs_metal
@pytest.mark.usefixtures("inference")
def test_the_same_executable_serves_other_buffers() -> None:
    fn = _VIA_VIEWS["mul"]
    cm = _compiled(fn, "0")
    for phase in (0.0, 0.5, 1.0, 1.5):
        x, w = _on(_filled(R5, phase)), _on(_filled(R5, phase + 2.0))
        _close(cm(x, w), fn(x, w), 3e-4, f"phase {phase}")
    # the same buffers again hit the bound-tensor cache
    _close(cm(x, w), fn(x, w), 3e-4, "repeat")


#: Feeds the pass must not re-declare: each is also read, or returned, as it is.
_KEPT: dict[str, Callable[[lucid.Tensor, lucid.Tensor], object]] = {
    "returned_whole": lambda x, w: (x, x * w),
    "also_permuted": lambda x, w: (x * w).sum() + x.permute(4, 3, 2, 1, 0).sum(),
    "also_reduced": lambda x, w: x * w + x.sum(dim=(1, 3), keepdim=True),
    "also_sliced": lambda x, w: x[:, 1:3] * w[:, 1:3] + x[:, :2],
}


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("name", sorted(_KEPT))
def test_a_feed_that_is_also_read_whole_keeps_its_logical_shape(name: str) -> None:
    fn = _KEPT[name]
    x, w = _on(_filled(R5, 0.1)), _on(_filled(R5, 0.9))
    got, want = _compiled(fn, "0")(x, w), fn(x, w)
    if isinstance(want, tuple):
        assert isinstance(got, tuple) and len(got) == len(want)
        for g, w_ in zip(got, want, strict=True):
            assert tuple(g.shape) == tuple(w_.shape)
            _close(g, w_, 3e-4, name)
    else:
        assert isinstance(got, lucid.Tensor)
        _close(got, want, 3e-4, name)


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_feed_returned_whole_comes_back_unchanged() -> None:
    x, w = _on(_filled(R5, 0.1)), _on(_filled(R5, 0.9))
    back, _ = _compiled(_KEPT["returned_whole"], "0")(x, w)
    assert tuple(back.shape) == R5
    _same(back, x, "passthrough")


# ── a step ──────────────────────────────────────────────────────


class _Scaled(nn.Module):
    """A rank-5 data feed and a rank-5 parameter, both read through views."""

    def __init__(self, shape: Shape, dtype: lucid.dtype) -> None:
        super().__init__()
        self.w = nn.Parameter(_on(_filled(shape, 0.7), dtype))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x * self.w * 1.5


def _loss(y: lucid.Tensor) -> lucid.Tensor:
    return (y * y).sum()


@needs_metal
@pytest.mark.parametrize("dtype", _DTYPES, ids=str)
@pytest.mark.parametrize("gate", ["0", "-1"], ids=["on", "off"])
def test_a_step_over_a_rank_5_feed_and_parameter_matches_eager(
    gate: str, dtype: lucid.dtype
) -> None:
    x = _on(_filled(R5, 0.3), dtype)
    eager = to_metal(_Scaled(R5, dtype))
    want = _loss(eager(x))
    want.backward()
    model = to_metal(_Scaled(R5, dtype))
    with mock.patch.dict(os.environ, {ENV: gate}):
        _C_engine.compile.session_cache_clear()
        loss = make_step(model, _loss)(x)
        loss.backward()
    _close(loss, want, _TOL[dtype] * 4, "loss")
    assert model.w.grad is not None and eager.w.grad is not None
    scale = max(max(abs(v) for v in _values(eager.w.grad)), 1.0)
    _close(model.w.grad / scale, eager.w.grad / scale, _TOL[dtype] * 8, "grad")


# ── a symbolic batch ──────────────────────────────────────────────────────────────────


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_symbolic_batch_keeps_its_rank_5_feeds_as_they_are() -> None:
    fn = _VIA_VIEWS["mul"]
    with mock.patch.dict(os.environ, {ENV: "0"}):
        _C_engine.compile.session_cache_clear()
        cm = lucid.compile(fn, dynamic=True)
        for batch in (4, 7, 3):
            x, w = _on(_filled((batch, 4, 2, 3, 2), 0.1 * batch)), _on(
                _filled((batch, 4, 2, 3, 2), 0.9)
            )
            _close(cm(x, w), fn(x, w), 3e-4, f"batch {batch}")


# ── AOT ───────────────────────────────────────────────────────────────────────────────


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("gate", [None, "0"], ids=["default", "gate0"])
def test_an_aot_round_trip_binds_a_declared_feed_at_its_declared_shape(
    tmp_path: object, gate: str | None
) -> None:
    fn = _VIA_VIEWS["mul"]
    shape: Shape = (4096, 4, 2, 3, 2)
    x, w = _on(_filled(shape, 0.2)), _on(_filled(shape, 0.8))
    cm = _compiled(fn, gate)
    want = cm(x, w)
    path = os.path.join(str(tmp_path), "feed")
    assert lc.save_compiled(cm, path) is True
    restored = lc.load_compiled(path)
    assert restored.input_shapes == [list(shape), list(shape)]
    got = restored(x, w)
    assert tuple(got.shape) == shape
    _same(got, want, "restored")
    _close(got, fn(x, w), 3e-4, "eager")


# ── time ──────────────────────────────────────────────────────────────────────────────


def _ms(fn: Callable[..., lucid.Tensor], *args: lucid.Tensor) -> float:
    cm = lucid.compile(fn)
    for _ in range(3):
        cm(*args)
    best = float("inf")
    for _ in range(15):
        t0 = time.perf_counter()
        float(cm(*args).reshape(-1)[0].item())
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


#: ``(case, twin)``: the rank-5 feeds and the same elements fed at rank 4.
_TWINS: dict[str, tuple[Callable[..., lucid.Tensor], Shape, Shape]] = {
    "mul": (lambda x, w: (x * w).sum(), (6144, 16, 2, 16, 2), (6144 * 16, 2, 16, 2)),
    "where": (
        lambda x, w: lucid.where(x > w, x, w).sum(),
        (1024, 2, 8, 49, 49),
        (1024 * 2, 8, 49, 49),
    ),
}


@needs_metal
@pytest.mark.usefixtures("inference")
@pytest.mark.parametrize("name", sorted(_TWINS))
def test_a_rank_5_feed_is_not_slower_than_its_rank_4_twin(name: str) -> None:
    """Held to 1.3x the twin; the views of a rank-5 feed ran 1.9x to 2.6x it."""
    fn, wide, twin = _TWINS[name]
    xs5 = (_on(_filled(wide, 0.1)), _on(_filled(wide, 0.9)))
    xs4 = (_on(_filled(twin, 0.1)), _on(_filled(twin, 0.9)))
    best5, best4 = float("inf"), float("inf")
    for _ in range(5):
        _C_engine.compile.session_cache_clear()
        best5 = min(best5, _ms(fn, *xs5))
        best4 = min(best4, _ms(fn, *xs4))
        if best5 <= 1.3 * best4 + 0.1:
            return
    assert best5 <= 1.3 * best4 + 0.1, f"{name}: rank-5 {best5:.2f} ms vs twin {best4:.2f} ms"
