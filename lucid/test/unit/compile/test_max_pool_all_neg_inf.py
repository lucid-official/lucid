"""A compiled max pool keeps a window of nothing but ``-inf`` at ``-inf`` (LCD-916).

MPSGraph's pooling kernel starts its running maximum at the dtype's lowest finite value and
takes only an element *greater* than it, so a window whose every element is ``-inf`` — or
exactly that lowest finite value — came back ``0``; eager and the reference say ``-inf`` (and
the lowest value itself).  A masked attention score or a log-probability pooled over a masked
span is such a window.  The pool is now a stencil with a prologue and an epilogue around it
(``OpEmitters/_MaxPool.h``), and its VJP routes the gradient of a ``-inf`` window to the first
element *inside* the input, never to padding, as eager does.

Tabled: every pool kind (``max_pool`` and ``adaptive_max_pool`` in 1, 2 and 3 dims) over a
window of each kind — all ``-inf``, ``-inf`` with one finite value, all the dtype's lowest
finite value, ordinary — with and without padding and in ceil mode, in float32, float16 and
bfloat16.  The forward value is held to eager's bit for bit; the gradient through ``make_step``
(and ``fused_step`` for the 2-D pool) to eager's.
"""

import itertools
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step, make_step
from lucid.compile._core.fallback import CompileFallbackWarning

_NINF = -math.inf
#: The lowest finite value of each dtype: what a pooling kernel treats as "nothing yet".
_LOWEST = {
    lucid.float32: -3.4028234663852886e38,
    lucid.float16: -65504.0,
    lucid.bfloat16: -3.3895313892515355e38,
}
_DTYPE_IDS = {lucid.float32: "f32", lucid.float16: "f16", lucid.bfloat16: "bf16"}
_LR = 0.5
_STEPS = 2


@dataclass(frozen=True)
class _Pool:
    dims: int
    op: Callable[[lucid.Tensor], lucid.Tensor]
    #: Spatial extent per axis of the test input.
    extent: int


def _grid(dims: int, extent: int, lowest: float) -> list[float]:
    """Flat ``(extent,) * dims`` values in 2-wide blocks, one kind of window per block.

    Block ``b`` (row-major over the blocks) is ``b % 4``: all ``-inf``; ``-inf`` but for one
    finite value; ordinary; all the dtype's lowest finite value.  A k2 s2 window is one block.
    """
    out = []
    for coord in itertools.product(range(extent), repeat=dims):
        block = 0
        for c in coord:
            block = block * (extent // 2) + c // 2
        kind = block % 4
        last = all(c % 2 == 1 for c in coord)
        if kind == 0:
            out.append(_NINF)
        elif kind == 1:
            out.append(2.0 + sum(coord) if last else _NINF)
        elif kind == 2:
            out.append(float((7 * len(out)) % 11) - 5.0)
        else:
            out.append(lowest)
    return out


_POOLS: dict[str, _Pool] = {}


def _register(
    name: str, dims: int, extent: int, op: Callable[..., lucid.Tensor], *a: object, **kw: object
) -> None:
    _POOLS[name] = _Pool(dims, lambda x: op(x, *a, **kw), extent)


_POOL_FN = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}
_ADAPTIVE_FN = {1: F.adaptive_max_pool1d, 2: F.adaptive_max_pool2d, 3: F.adaptive_max_pool3d}
for _d, _ext in ((1, 8), (2, 4), (3, 4)):
    _register(f"max_pool{_d}d-k2s2", _d, _ext, _POOL_FN[_d], 2)
    _register(f"max_pool{_d}d-k3s1p1", _d, _ext, _POOL_FN[_d], 3, 1, 1)
    _register(f"max_pool{_d}d-k3s2-ceil", _d, _ext, _POOL_FN[_d], 3, 2, 0, ceil_mode=True)
    _register(f"max_pool{_d}d-k3s2p1", _d, _ext, _POOL_FN[_d], 3, 2, 1)
    _register(f"max_pool{_d}d-k3s2p1-ceil", _d, _ext, _POOL_FN[_d], 3, 2, 1, ceil_mode=True)
    _register(f"max_pool{_d}d-k2s1-ceil-odd", _d, _ext + 1, _POOL_FN[_d], 2, 2, 0, ceil_mode=True)
    _register(f"adaptive_max_pool{_d}d", _d, _ext, _ADAPTIVE_FN[_d], 2 if _d > 1 else 4)


def _input(pool: _Pool, dtype: lucid.dtype, device: str) -> lucid.Tensor:
    # Two channels: the second the first reversed, so a window kind lands on another position.
    flat = _grid(pool.dims, pool.extent, _LOWEST[dtype])
    spatial = (pool.extent,) * pool.dims
    first = lucid.tensor(flat, dtype=dtype, device=device).reshape((1, 1) + spatial)
    second = lucid.tensor(flat[::-1], dtype=dtype, device=device).reshape((1, 1) + spatial)
    return lucid.cat([first, second], 1)


class _Apply(nn.Module):
    def __init__(self, op: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.op = op

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.op(x)


def _same(got: list[float], want: list[float], why: str) -> None:
    assert len(got) == len(want), why
    for g, w in zip(got, want, strict=True):
        assert (math.isnan(g) and math.isnan(w)) or g == w, f"{why}: compiled {got} != eager {want}"


_CASES = [
    pytest.param(name, dtype, id=f"{name}-{dt}")
    for name in _POOLS
    for dtype, dt in _DTYPE_IDS.items()
]


@pytest.mark.parametrize(("name", "dtype"), _CASES)
def test_compiled_forward_matches_eager(
    name: str, dtype: lucid.dtype, device_gpu_only: str
) -> None:
    pool = _POOLS[name]
    x = _input(pool, dtype, device_gpu_only)
    want = pool.op(x)
    compiled = lucid.compile(_Apply(pool.op))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        with lucid.no_grad():
            got = [compiled(x) for _ in range(2)]
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"]
    assert tuple(got[0].shape) == tuple(want.shape)
    for k, g in enumerate(got):
        _same(
            g.to(lucid.float32).cpu().reshape(-1).tolist(),
            want.to(lucid.float32).cpu().reshape(-1).tolist(),
            f"call {k}",
        )


class _Pooled(nn.Module):
    """``pool(x)`` weighted element by element, so a gradient sent to the wrong place shows."""

    def __init__(self, pool: _Pool, dtype: lucid.dtype, device: str) -> None:
        super().__init__()
        self.x = nn.Parameter(_input(pool, dtype, device))
        self.pool = pool
        with lucid.no_grad():
            shape = tuple(pool.op(self.x).shape)
        weight = lucid.tensor(
            [1.0 + k for k in range(math.prod(shape))], dtype=dtype, device=device
        )
        self.register_buffer("w", weight.reshape(shape))

    def forward(self, s: lucid.Tensor) -> lucid.Tensor:
        return (self.pool.op(self.x) * self.w * s).sum().reshape(1)


def _loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return (out * target).sum()


def _steps(
    entry: str, model: _Pooled, opt: optim.Optimizer, s: lucid.Tensor
) -> list[tuple[list[float], list[float]]]:
    steps = []
    fused = fused_step(model, _loss, opt) if entry == "fused_step" else None
    step = make_step(model, _loss) if entry == "make_step" else None
    for _ in range(_STEPS):
        opt.zero_grad()
        if fused is not None:
            fused(s, s)
            grad: list[float] = []
        else:
            loss = step(s, s) if step is not None else _loss(model(s), s)
            loss.backward()
            assert model.x.grad is not None
            grad = model.x.grad.detach().cpu().reshape(-1).tolist()
            opt.step()
        steps.append((grad, model.x.detach().to(lucid.float32).cpu().reshape(-1).tolist()))
    if step is not None:
        assert not step.eager_only
    if fused is not None:
        assert not fused.cache_info()["eager_only"]
    return steps


_GRAD_CASES = [
    pytest.param(name, entry, dtype, id=f"{name}-{entry}-{dt}")
    for name in _POOLS
    for entry in (("make_step", "fused_step") if name.startswith("max_pool2d") else ("make_step",))
    for dtype, dt in _DTYPE_IDS.items()
]


@pytest.mark.parametrize(("name", "entry", "dtype"), _GRAD_CASES)
def test_compiled_gradient_matches_eager(
    name: str, entry: str, dtype: lucid.dtype, device_gpu_only: str
) -> None:
    pool = _POOLS[name]
    s = lucid.ones(1, dtype=dtype, device=device_gpu_only)
    eager = _Pooled(pool, dtype, device_gpu_only)
    want = _steps("eager", eager, optim.SGD(eager.parameters(), lr=_LR), s)
    model = _Pooled(pool, dtype, device_gpu_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        got = _steps(entry, model, optim.SGD(model.parameters(), lr=_LR), s)
    for k, ((gg, gx), (wg, wx)) in enumerate(zip(got, want, strict=True)):
        if entry != "fused_step":
            _same(gg, wg, f"step {k} gradient")
        _same(gx, wx, f"step {k} input after SGD")


def test_the_reported_case(device_gpu_only: str) -> None:
    """LCD-916's own numbers: ``[[-inf, -inf, 1, 2], [-inf, -inf, 3, 4]]``."""
    x = lucid.tensor(
        [[[[_NINF, _NINF, 1.0, 2.0], [_NINF, _NINF, 3.0, 4.0]]]], device=device_gpu_only
    )
    with lucid.no_grad():
        k2 = lucid.compile(lambda t: F.max_pool2d(t, 2, 2))(x)
        k3 = lucid.compile(lambda t: F.max_pool2d(t, 3, 1, 1))(x)
    assert k2.reshape(-1).tolist() == [_NINF, 4.0]
    assert k3.reshape(-1).tolist() == [_NINF, 3.0, 4.0, 4.0, _NINF, 3.0, 4.0, 4.0]
