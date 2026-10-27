"""CPU kernels that move or add elements cover every dtype they claim.

LCD-248 / LCD-155.  The CPU dispatches on dtype by name, and several
switches named float16 without bfloat16, or had no case for the 8- and
16-byte complex types: gradient accumulation refused bfloat16 ("dtype not
yet supported in Phase 2"), and the backward of every transpose / permute
refused bfloat16 and complex.  bfloat16 could not train on the CPU at all.

Permutation now moves elements by width alone, accumulation names every
dtype, and ``tools/check_dtype_dispatch.py`` (rule ``half-pair``) fails
any switch that names float16 without bfloat16.  Each case below checks a
value, not just that the call runs: a byte-moving kernel at the wrong
width answers with the wrong numbers, not an error.
"""

import pytest

import lucid

_HALVES = [lucid.float16, lucid.bfloat16]
_COMPLEX = [lucid.complex64, lucid.complex128]
_FLOATS = [*_HALVES, *_COMPLEX, lucid.float32, lucid.float64]


def _flat(values: object) -> list[object]:
    if isinstance(values, list):
        return [v for item in values for v in _flat(item)]
    return [values]


def _complex_grid(shape: tuple[int, ...], dtype: lucid.dtype) -> lucid.Tensor:
    """Distinct complex values with distinct real and imaginary parts."""
    flat = [complex(v, 7 - v) for v in _flat(_grid(shape, lucid.float64).tolist())]
    return lucid.tensor(flat, dtype=dtype).reshape(*shape)


def _grid(shape: tuple[int, ...], dtype: lucid.dtype) -> lucid.Tensor:
    """Small distinct integers — exact in every float format here."""
    n = 1
    for s in shape:
        n *= s
    return (lucid.arange(n).reshape(*shape) % 13 + 1).to(dtype)


# ── accumulation: a second backward adds into the leaf's .grad ──────────────


@pytest.mark.parametrize("dtype", _FLOATS, ids=str)
def test_grad_accumulates_in_its_own_dtype(dtype: lucid.dtype) -> None:
    w = _complex_grid((2, 3), dtype) if dtype in _COMPLEX else _grid((2, 3), dtype)
    w.requires_grad_()
    for _ in range(2):
        loss = w.abs().sum() if dtype in _COMPLEX else (w * w).sum()
        loss.backward()
    assert w.grad is not None and w.grad.dtype == dtype
    once = w.detach() / w.detach().abs() if dtype in _COMPLEX else 2 * w.detach()
    got, want = _flat(w.grad.tolist()), _flat((once + once).tolist())
    assert got == pytest.approx(want, rel=1e-6)


# ── permute / transpose backward ────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [*_HALVES, lucid.float32, lucid.float64], ids=str)
@pytest.mark.parametrize(
    "perm", [(1, 0, 2), (2, 0, 1), (0, 2, 1)], ids=lambda p: "".join(map(str, p))
)
def test_permute_backward_real(dtype: lucid.dtype, perm: tuple[int, ...]) -> None:
    w = lucid.zeros(2, 3, 4).to(dtype).requires_grad_()
    out_shape = tuple(w.shape[p] for p in perm)
    k = _grid(out_shape, dtype)
    (w.permute(*perm).contiguous() * k).sum().backward()
    inverse = [perm.index(d) for d in range(3)]
    assert w.grad is not None and w.grad.dtype == dtype
    assert w.grad.double().tolist() == k.permute(*inverse).double().tolist()


@pytest.mark.parametrize("dtype", _HALVES + _COMPLEX, ids=str)
def test_transpose_backward_matches_direct_path(dtype: lucid.dtype) -> None:
    """``f(w.T)`` and ``f(w)`` have the same gradient when ``f`` is elementwise."""
    base = _complex_grid((3, 4), dtype) if dtype in _COMPLEX else _grid((3, 4), dtype)
    a = base.detach().clone().requires_grad_()
    b = base.detach().clone().requires_grad_()
    a.T.contiguous().abs().sum().backward()
    b.abs().sum().backward()
    assert a.grad is not None and b.grad is not None
    assert a.grad.dtype == dtype
    assert a.grad.tolist() == b.grad.tolist()


@pytest.mark.parametrize(
    "dtype", [*_FLOATS, lucid.int8, lucid.int16, lucid.int32, lucid.int64, lucid.bool], ids=str
)
def test_tril_every_dtype(dtype: lucid.dtype) -> None:
    x = lucid.ones(3, 3).to(dtype)
    got = lucid.tril(x)
    assert got.dtype == dtype
    mask = [[1 if j <= i else 0 for j in range(3)] for i in range(3)]
    assert lucid.tensor(mask).to(dtype).tolist() == got.tolist()


@pytest.mark.parametrize("dtype", _HALVES, ids=str)
def test_ones_half(dtype: lucid.dtype) -> None:
    assert lucid.ones(4, dtype=dtype).float().tolist() == [1.0] * 4


# ── LCD-253: an autocast op's gradient into a non-leaf f32 input ────────────


def test_autocast_bf16_grad_through_view_lands_as_float32() -> None:
    x = lucid.randn(2, 4).requires_grad_()
    with lucid.amp.autocast("cpu", dtype=lucid.bfloat16):
        y = (x[:, 0] * lucid.ones(2)).sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.tolist() == [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
