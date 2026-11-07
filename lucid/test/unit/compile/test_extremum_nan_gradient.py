"""A compiled max / min backward shapes a NaN slice's gradient as eager does (LCD-878).

Eager gives the gradient of a slice that holds a NaN three ways, by the call (see
``lucid/test/unit/ops/reductions/test_extremum_nan_gradient.py``): ``x.max()`` hands it to
the NaN elements, ``amax`` / ``aminmax`` / ``matrix_norm`` make the whole slice NaN and
``vector_norm(inf)`` gives zeros.  The compiled VJP of the ``max`` / ``min`` reductions used
to answer zeros for all of them (LCD-865's copy of eager's old rule).  It now reads the
``nan_grad`` attribute the forward traces, so the same table gives the same gradients — and
the second call, which runs the cached executable, the same again.
"""

import math
import warnings

import pytest

import lucid
import lucid.nn as nn
from lucid.compile import make_step
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.test._fixtures.extremum_nan import DATA, REDUCTIONS

_DTYPES = {"f32": lucid.float32, "f16": lucid.float16}
_TOL = {"f32": 1e-6, "f16": 2e-3}


class _Weighted(nn.Module):
    """``op(a)`` weighted per element, so a share sent to the wrong position shows."""

    def __init__(self, name: str, data: list[list[float]], dtype: lucid.dtype, device: str) -> None:
        super().__init__()
        self.a = nn.Parameter(lucid.tensor(data, dtype=dtype, device=device))
        self.op = REDUCTIONS[name]
        with lucid.no_grad():
            shape = tuple(self.op(lucid, self.a).shape)
        weights = [1.0 + k for k in range(math.prod(shape))]
        self.register_buffer("w", lucid.tensor(weights, dtype=dtype, device=device).reshape(shape))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return (self.op(lucid, self.a) * self.w * x).sum().reshape(1)


def _loss(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return (out * target).sum()


def _grad(model: _Weighted) -> list[float]:
    grad = model.a.grad
    assert grad is not None
    return grad.detach().cpu().to(lucid.float32).reshape(-1).tolist()


def _assert_same(got: list[float], want: list[float], tol: float, why: str) -> None:
    assert len(got) == len(want)
    for p, q in zip(got, want, strict=True):
        ok = math.isnan(p) if math.isnan(q) else math.isclose(p, q, rel_tol=tol, abs_tol=tol)
        assert ok, f"{why}: compiled {got} != eager {want}"


def _cases() -> list[object]:
    return [
        pytest.param(name, data, dt, id=f"{name}-{data}-{dt}")
        for name in REDUCTIONS
        for data in DATA
        for dt in _DTYPES
    ]


@pytest.mark.parametrize(("name", "data", "dt"), _cases())
def test_compiled_gradient_through_a_nan_slice_matches_eager(
    name: str, data: str, dt: str, device_gpu_only: str
) -> None:
    dtype = _DTYPES[dt]
    x = lucid.ones(1, dtype=dtype, device=device_gpu_only)
    t = lucid.ones(1, dtype=dtype, device=device_gpu_only)

    eager = _Weighted(name, DATA[data], dtype, device_gpu_only)
    _loss(eager(x), t).backward()
    want = _grad(eager)

    model = _Weighted(name, DATA[data], dtype, device_gpu_only)
    step = make_step(model, _loss)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        for call in range(2):
            model.a.grad = None
            step(x, t).backward()
            _assert_same(_grad(model), want, _TOL[dt], f"call {call}")
    assert not step.eager_only


_NAN = math.nan

#: The weighted gradient of the "nan-mid" rows ``[[1, nan, 1], [2, 2, 0]]``, written out —
#: independent of eager, so a compiled backward that agreed with a wrong eager one still fails.
_PINNED = {
    "max": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "amax-dim": [_NAN, _NAN, _NAN, 1.0, 1.0, 0.0],
    "vector_norm-inf": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


@pytest.mark.parametrize("name", list(_PINNED))
def test_compiled_nan_slice_gradient_is_the_documented_one(name: str, device_gpu_only: str) -> None:
    model = _Weighted(name, DATA["nan-mid"], lucid.float32, device_gpu_only)
    step = make_step(model, _loss)
    one = lucid.ones(1, device=device_gpu_only)
    step(one, one).backward()
    assert not step.eager_only
    _assert_same(_grad(model), _PINNED[name], 1e-6, name)
