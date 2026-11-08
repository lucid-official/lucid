"""The gradient of max / min through a slice that holds a NaN (LCD-878).

A NaN that reached ``max`` / ``amax`` / ``aminmax`` / ``vector_norm(inf)`` left no trace in
the backward pass: the equality mask was empty and the share ``mask / (ties + 1 - mask)``
was zero.  The reference differentiates it three ways, by the call — ``x.max()`` gives the
NaN elements the gradient, ``amax`` / ``amin`` / ``aminmax`` and the matrix norms make the
whole slice NaN, ``vector_norm(inf)`` gives zeros — and ``max(dim)`` gathers at its index.
The table (``lucid/test/_fixtures/extremum_nan.py``) holds every one of them to the
reference's own CPU gradient, on CPU and Metal, over a NaN in each position, two NaNs, an
all-NaN row, ties and an ordinary row.  Gradients are weighted per element so a share sent
to the wrong position shows.
"""

import math
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.extremum_nan import DATA, REDUCTIONS, SPECTRAL

pytestmark = pytest.mark.parity

_DTYPES = ("float32", "float16", "bfloat16")
_TOL = {"float32": 1e-6, "float16": 2e-3, "bfloat16": 2e-2}


def _grad_pair(
    ref: ModuleType, name: str, data: list[list[float]], dtype: str, device: str
) -> tuple[list[float], list[float]]:
    op = {**REDUCTIONS, **SPECTRAL}[name]
    x = lucid.tensor(data, dtype=getattr(lucid, dtype), device=device, requires_grad=True)
    want_x = ref.tensor(data, dtype=getattr(ref, dtype), requires_grad=True)
    out, want = op(lucid, x), op(ref, want_x)
    n = math.prod(want.shape)
    weights = [1.0 + k for k in range(n)]
    (
        out * lucid.tensor(weights, dtype=out.dtype, device=device).reshape(out.shape)
    ).sum().backward()
    (want * ref.tensor(weights, dtype=want.dtype).reshape(want.shape)).sum().backward()
    assert x.grad is not None and want_x.grad is not None
    return (x.grad.to(lucid.float32).reshape(-1).tolist(), want_x.grad.float().reshape(-1).tolist())


def _assert_same(got: list[float], want: list[float], tol: float) -> None:
    assert len(got) == len(want)
    for p, q in zip(got, want, strict=True):
        ok = math.isnan(p) if math.isnan(q) else math.isclose(p, q, rel_tol=tol, abs_tol=tol)
        assert ok, f"lucid {got} != reference {want}"


def _cases() -> list[object]:
    out = []
    for name in (*REDUCTIONS, *SPECTRAL):
        for data in DATA:
            for dtype in _DTYPES:
                if name in SPECTRAL and dtype != "float32":
                    continue
                out.append(pytest.param(name, data, dtype, id=f"{name}-{data}-{dtype}"))
    return out


@pytest.mark.parametrize(("name", "data", "dtype"), _cases())
def test_gradient_through_a_nan_slice_matches_the_reference(
    ref: ModuleType, device: str, name: str, data: str, dtype: str
) -> None:
    got, want = _grad_pair(ref, name, DATA[data], dtype, device)
    _assert_same(got, want, _TOL[dtype])


def test_the_defects_own_numbers(device: str) -> None:
    """Independent of the reference: the two rows of LCD-878."""
    x = lucid.tensor([1.0, math.nan, 1.0], device=device, requires_grad=True)
    x.max().backward()
    assert x.grad is not None and x.grad.tolist() == [0.0, 1.0, 0.0]
    y = lucid.tensor([[1.0, math.nan, 1.0], [2.0, 2.0, 0.0]], device=device, requires_grad=True)
    lucid.amax(y, dim=1).sum().backward()
    assert y.grad is not None
    row0, row1 = y.grad.tolist()
    assert all(math.isnan(v) for v in row0) and row1 == [0.5, 0.5, 0.0]
