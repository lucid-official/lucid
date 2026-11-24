"""Complex dtypes in the kernel tables of ``where``, ``outer``, ``pow`` and ``solve_triangular``.

LCD-1070.  Each of these ops listed the real dtypes in its kernel dispatch and
left complex out, so the same call that ran on one stream raised
``NotImplementedError: <op>: dtype not supported`` on the other:

* ``where`` — the CPU select had no 16-byte case (complex128), and its gradient
  routing took float32 / float64 only, so a complex64 ``where`` could not be
  differentiated on the CPU;
* ``outer`` / ``ger`` / ``addr`` — an ``(M, 1) @ (1, N)`` product, and the CPU
  ``matmul`` had no complex GEMM;
* ``pow(complex, complex)`` — the CPU had no complex power, and its gradient
  needs a complex ``log`` and a real-scalar add there;
* ``solve_triangular`` — the op refused complex at its dtype gate, and the
  backend had no ``ctrsm`` / ``ztrsm``.

The same table pattern left complex out of the CPU's ``masked_fill`` and
complex128 out of ``inner`` and ``tensordot``'s CPU kernels; they are in the
sweep below.  Metal holds complex64 only (MLX has no complex128), so complex128
cells run on the CPU stream.

Three guards, over both devices and both widths:

* every op against a closed form (NumPy, from inputs rounded to the dtype);
* every op, and its gradient, against the reference — the gradient convention is
  the conjugate one of ``autograd/ComplexGrad.h``;
* the boundaries: empty operands, a broadcast condition, a transposed view.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params

_COMPLEX = (lucid.complex64, lucid.complex128)
_PLACEMENTS = device_dtype_params(_COMPLEX)

#: dtype name -> comparison tolerance.  The closed forms are taken from the
#: inputs rounded to the dtype, so what is left is the kernel's own rounding.
_TOL = {"complex64": {"rtol": 2e-4, "atol": 2e-5}, "complex128": {"rtol": 1e-11, "atol": 1e-12}}

_U = np.array([1 + 2j, -0.5j, 3 - 1j])
_V = np.array([2 - 1j, 1j])
_M = np.array([[1 + 1j, 2], [0.5j, -1], [3, 1 - 2j]])
_BASE = np.array([1 + 2j, -0.5 + 0.3j, 0.7 - 1.1j])
_EXPO = np.array([0.6 - 0.4j, 1.3 + 0.2j, -0.8 + 0.9j])
_MASK = np.array([True, False, True])
_X = np.array([1 + 2j, 3 - 1j, -2j])
_Y = np.array([0.5 + 0.5j, -1, 2 + 3j])
# The strict lower triangle holds values the upper solve must not read.
_A = np.array([[2 + 1j, 1 - 1j, 0.5j], [9 + 9j, 1 + 2j, -1 + 0.5j], [7 - 3j, 8 + 1j, 3 - 1j]])
_B = np.array([[1 + 1j, 2], [3j, 1 - 1j], [0.5, -2j]])
_ROW_B = np.array([[1 + 1j, 2, 3j], [1 - 1j, 0.5, -2j]])
_P = np.array([[1 + 1j, 2, 0.5j], [3, 1 - 1j, 2j]])
_Q = np.array([[1j, 2 - 1j], [0.5, 1 + 1j], [-1, 3j]])
_BATCH_P = np.stack([_P, _P * 1j])
_BATCH_Q = np.stack([_Q, _Q.conj()])
_COND_2D = np.array([[True], [False]])
_WIDE_X = np.stack([_X, _X * 1j])
_MASK_2D = np.array([[True, False, True], [False, True, False]])


def _tri(a: np.ndarray, upper: bool, unit: bool) -> np.ndarray:
    """The triangle ``solve_triangular`` reads, with the diagonal it assumes."""
    t = np.triu(a) if upper else np.tril(a)
    return t - np.diag(np.diag(t)) + np.eye(len(t)) if unit else t


def _rounded(arr: np.ndarray, dtype_name: str) -> np.ndarray:
    if arr.dtype != np.complex128:
        return arr
    return arr.astype(dtype_name).astype(np.complex128)


def _tensor(lib: ModuleType, arr: np.ndarray, dtype_name: str, device: str) -> Any:
    if arr.dtype == np.bool_:
        if lib is lucid:
            return lucid.tensor(arr, dtype=lucid.bool, device=device)
        return lib.tensor(arr)
    if lib is lucid:
        return lucid.tensor(arr, dtype=getattr(lucid, dtype_name), device=device)
    return lib.tensor(arr, dtype=getattr(lib, dtype_name))


def _numpy(t: Any) -> np.ndarray:
    """A tensor of either framework as a NumPy array."""
    if isinstance(t, lucid.Tensor):
        return np.asarray(t.detach().to("cpu").numpy())
    return np.asarray(t.detach().resolve_conj().resolve_neg().numpy())


@dataclass(frozen=True)
class _Case:
    """One op over its inputs: ``fn(lib, *tensors)`` and the closed form."""

    name: str
    inputs: tuple[np.ndarray, ...]
    fn: Callable[..., Any]
    want: Callable[..., np.ndarray]
    #: Indices of the inputs a gradient is taken for; empty = forward only.
    grad: tuple[int, ...] = ()


def _outer(lib: ModuleType, a: Any, b: Any) -> Any:
    return lib.linalg.outer(a, b) if lib is lucid else lib.outer(a, b)


def _inner(lib: ModuleType, a: Any, b: Any) -> Any:
    return lib.linalg.inner(a, b) if lib is lucid else lib.inner(a, b)


def _solve(upper: bool, unit: bool = False, left: bool = True) -> Callable[..., Any]:
    def run(lib: ModuleType, a: Any, b: Any) -> Any:
        return lib.linalg.solve_triangular(a, b, upper=upper, left=left, unitriangular=unit)

    return run


_CASES = [
    _Case("where", (_MASK, _X, _Y), lambda lib, c, x, y: lib.where(c, x, y), np.where, grad=(1, 2)),
    _Case(
        "where-broadcast",
        (_COND_2D, _WIDE_X, _Y),
        lambda lib, c, x, y: lib.where(c, x, y),
        np.where,
        grad=(1, 2),
    ),
    _Case(
        "masked_fill",
        (_WIDE_X, _MASK_2D),
        lambda lib, x, m: lib.masked_fill(x, m, 2.5),
        lambda x, m: np.where(m, 2.5, x),
        grad=(0,),
    ),
    _Case("outer", (_U, _V), _outer, np.outer, grad=(0, 1)),
    _Case("ger", (_U, _V), lambda lib, a, b: lib.ger(a, b), np.outer, grad=(0, 1)),
    _Case(
        "addr",
        (_M, _U, _V),
        lambda lib, m, u, v: lib.addr(m, u, v, beta=2, alpha=3),
        lambda m, u, v: 2 * m + 3 * np.outer(u, v),
        grad=(0, 1, 2),
    ),
    _Case("pow", (_BASE, _EXPO), lambda lib, a, b: lib.pow(a, b), np.power, grad=(0, 1)),
    _Case(
        "solve_triangular-upper",
        (_A, _B),
        _solve(upper=True),
        lambda a, b: np.linalg.solve(_tri(a, True, False), b),
        grad=(0, 1),
    ),
    _Case(
        "solve_triangular-lower",
        (_A, _B),
        _solve(upper=False),
        lambda a, b: np.linalg.solve(_tri(a, False, False), b),
        grad=(0, 1),
    ),
    _Case(
        "solve_triangular-unit",
        (_A, _B),
        _solve(upper=True, unit=True),
        lambda a, b: np.linalg.solve(_tri(a, True, True), b),
        grad=(0, 1),
    ),
    _Case(
        "solve_triangular-right",
        (_A, _ROW_B),
        _solve(upper=True, left=False),
        lambda a, b: b @ np.linalg.inv(_tri(a, True, False)),
        grad=(0, 1),
    ),
    _Case(
        "solve_triangular-vector",
        (_A, _B[:, 0]),
        _solve(upper=True),
        lambda a, b: np.linalg.solve(_tri(a, True, False), b),
        grad=(0, 1),
    ),
    # The matmul family reaches the CPU's complex GEMM, which ``outer`` and the
    # gradient of ``solve_triangular`` need; its forward and gradient are held
    # to the same standard.
    _Case("matmul", (_P, _Q), lambda lib, a, b: lib.matmul(a, b), np.matmul, grad=(0, 1)),
    _Case("mm", (_P, _Q), lambda lib, a, b: lib.mm(a, b), np.matmul, grad=(0, 1)),
    _Case("bmm", (_BATCH_P, _BATCH_Q), lambda lib, a, b: lib.bmm(a, b), np.matmul, grad=(0, 1)),
    _Case("log", (_BASE,), lambda lib, a: lib.log(a), np.log, grad=(0,)),
    # complex128 left out of the CPU kernels of ``inner`` and ``tensordot``.
    _Case("inner", (_U, _U[::-1].copy()), _inner, np.inner, grad=(0, 1)),
    _Case(
        "tensordot",
        (_P, _Q),
        lambda lib, a, b: lib.tensordot(a, b, dims=1) if lib is lucid else lib.tensordot(a, b, 1),
        lambda a, b: np.tensordot(a, b, 1),
    ),
]

_IDS = [c.name for c in _CASES]


def _build(
    case: _Case, lib: ModuleType, dtype_name: str, device: str, *, grad: bool = False
) -> list[Any]:
    tensors = [_tensor(lib, _rounded(arr, dtype_name), dtype_name, device) for arr in case.inputs]
    if grad:
        for i in case.grad:
            tensors[i].requires_grad_(True)
    return tensors


# ── closed forms ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_forward_matches_the_closed_form(case: _Case, device: str, dtype: lucid.dtype) -> None:
    name = str(dtype).removeprefix("lucid.")
    got = case.fn(lucid, *_build(case, lucid, name, device))
    assert got.dtype == dtype
    assert got.device.type == device
    want = case.want(*(_rounded(a, name) for a in case.inputs))
    assert got.shape == want.shape
    np.testing.assert_allclose(_numpy(got), want, **_TOL[name])


@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
def test_where_routes_the_whole_complex_gradient(device: str, dtype: lucid.dtype) -> None:
    """Each branch receives the gradient where it was chosen, imaginary part included."""
    name = str(dtype).removeprefix("lucid.")
    x = _tensor(lucid, _rounded(_X, name), name, device).requires_grad_(True)
    y = _tensor(lucid, _rounded(_Y, name), name, device).requires_grad_(True)
    cond = _tensor(lucid, _MASK, name, device)
    out = lucid.where(cond, x, y)
    lucid.real(out * lucid.conj(out)).sum().backward()
    # L = |out|^2, so the gradient reaching out is 2 * out.
    np.testing.assert_allclose(_numpy(x.grad), 2 * _rounded(_X, name) * _MASK, **_TOL[name])
    np.testing.assert_allclose(_numpy(y.grad), 2 * _rounded(_Y, name) * ~_MASK, **_TOL[name])


@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
@pytest.mark.parametrize(
    ("upper", "unit"), [(True, False), (False, False), (True, True), (False, True)]
)
def test_solve_triangular_gradient_is_the_conjugate_transpose_rule(
    upper: bool, unit: bool, device: str, dtype: lucid.dtype
) -> None:
    """dB = A^-H G and dA = -Pi(dB X^H), with Pi the triangle the solve read."""
    name = str(dtype).removeprefix("lucid.")
    a_np, b_np = _rounded(_A, name), _rounded(_B, name)
    a = _tensor(lucid, a_np, name, device).requires_grad_(True)
    b = _tensor(lucid, b_np, name, device).requires_grad_(True)
    x = lucid.linalg.solve_triangular(a, b, upper=upper, unitriangular=unit)
    lucid.real(x * lucid.conj(x)).sum().backward()

    tri = _tri(a_np, upper, unit)
    x_np = np.linalg.solve(tri, b_np)
    d_b = np.linalg.solve(tri.conj().T, 2 * x_np)
    d_a = -d_b @ x_np.conj().T
    keep = (
        np.triu(np.ones((3, 3)), 1 if unit else 0)
        if upper
        else np.tril(np.ones((3, 3)), -1 if unit else 0)
    )
    np.testing.assert_allclose(_numpy(b.grad), d_b, **_TOL[name])
    np.testing.assert_allclose(_numpy(a.grad), d_a * keep, **_TOL[name])


# ── the reference ────────────────────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_forward_matches_the_reference(
    ref: ModuleType, case: _Case, device: str, dtype: lucid.dtype
) -> None:
    name = str(dtype).removeprefix("lucid.")
    got = case.fn(lucid, *_build(case, lucid, name, device))
    want = case.fn(ref, *_build(case, ref, name, device))
    assert got.shape == tuple(want.shape)
    np.testing.assert_allclose(_numpy(got), _numpy(want), **_TOL[name])


def _norm_sq(lib: ModuleType, y: Any) -> Any:
    return lib.real(y * lib.conj(y)).sum()


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
@pytest.mark.parametrize("case", [c for c in _CASES if c.grad], ids=lambda c: c.name)
def test_gradient_matches_the_reference(
    ref: ModuleType, case: _Case, device: str, dtype: lucid.dtype
) -> None:
    name = str(dtype).removeprefix("lucid.")
    xs = _build(case, lucid, name, device, grad=True)
    rs = _build(case, ref, name, device, grad=True)
    got = lucid.autograd.grad(
        _norm_sq(lucid, case.fn(lucid, *xs)), [xs[i] for i in case.grad], allow_unused=True
    )
    want = ref.autograd.grad(
        _norm_sq(ref, case.fn(ref, *rs)), [rs[i] for i in case.grad], allow_unused=True
    )
    for i, g, w in zip(case.grad, got, want, strict=True):
        assert g is not None and w is not None, f"input {i} received no gradient"
        assert g.dtype == dtype
        np.testing.assert_allclose(_numpy(g), _numpy(w), **_TOL[name])


@pytest.mark.parity
@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
def test_pow_agrees_with_the_reference_at_a_zero_base(
    ref: ModuleType, device: str, dtype: lucid.dtype
) -> None:
    """``0 ** y`` for a real, positive ``y`` is 0, and the reference's CPU kernel and
    Lucid's both come from ``std::pow`` — NaN, not 1, at ``0 ** 0``."""
    name = str(dtype).removeprefix("lucid.")
    base = np.zeros(3, dtype=np.complex128)
    expo = np.array([1 + 0j, 2.5 + 0j, 2 + 0j])
    got = lucid.pow(_tensor(lucid, base, name, device), _tensor(lucid, expo, name, device))
    want = ref.pow(_tensor(ref, base, name, device), _tensor(ref, expo, name, device))
    np.testing.assert_allclose(_numpy(got), _numpy(want), **_TOL[name])


# ── boundaries ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
def test_empty_operands(device: str, dtype: lucid.dtype) -> None:
    empty = lucid.zeros(0, dtype=dtype, device=device)
    three = lucid.ones(3, dtype=dtype, device=device)
    assert lucid.linalg.outer(empty, three).shape == (0, 3)
    assert lucid.linalg.outer(three, empty).shape == (3, 0)
    assert lucid.pow(empty, empty).shape == (0,)
    none = lucid.zeros(0, dtype=lucid.bool, device=device)
    assert lucid.where(none, empty, empty).shape == (0,)
    solved = lucid.linalg.solve_triangular(
        lucid.zeros(0, 0, dtype=dtype, device=device),
        lucid.zeros(0, 2, dtype=dtype, device=device),
        upper=True,
    )
    assert solved.shape == (0, 2) and solved.dtype == dtype
    # An empty contraction sums over nothing.
    k0 = lucid.zeros(2, 0, dtype=dtype, device=device) @ lucid.zeros(
        0, 3, dtype=dtype, device=device
    )
    assert k0.tolist() == [[0j] * 3] * 2


@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
def test_zero_dim_where(device: str, dtype: lucid.dtype) -> None:
    name = str(dtype).removeprefix("lucid.")
    one, two = 1 + 2j, 3 - 1j
    got = lucid.where(
        lucid.tensor(True, device=device),
        lucid.tensor(one, dtype=dtype, device=device),
        lucid.tensor(two, dtype=dtype, device=device),
    )
    assert got.shape == () and got.dtype == dtype
    np.testing.assert_allclose(_numpy(got), _rounded(np.array(one), name), **_TOL[name])


@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
def test_non_contiguous_operands(device: str, dtype: lucid.dtype) -> None:
    """A transposed matrix and a strided vector reach the kernels in their logical order."""
    name = str(dtype).removeprefix("lucid.")
    a_np = _rounded(_A, name)
    # The transpose of A's transpose: a view whose values are A's, laid out column-major.
    a_view = _tensor(lucid, np.ascontiguousarray(a_np.T), name, device).mT
    b = _tensor(lucid, _rounded(_B, name), name, device)
    got = lucid.linalg.solve_triangular(a_view, b, upper=True)
    np.testing.assert_allclose(
        _numpy(got), np.linalg.solve(np.triu(a_np), _rounded(_B, name)), **_TOL[name]
    )

    ramp = _rounded(np.arange(6) * (1 + 1j), name)
    strided = _tensor(lucid, ramp, name, device)[::2]
    np.testing.assert_allclose(
        _numpy(lucid.linalg.outer(strided, strided)), np.outer(ramp[::2], ramp[::2]), **_TOL[name]
    )


@pytest.mark.parametrize(("device", "dtype"), _PLACEMENTS)
def test_batched_a_broadcasts_against_a_shared_b(device: str, dtype: lucid.dtype) -> None:
    name = str(dtype).removeprefix("lucid.")
    batch = np.stack([_A, _A.T])
    got = lucid.linalg.solve_triangular(
        _tensor(lucid, _rounded(batch, name), name, device),
        _tensor(lucid, _rounded(_B, name), name, device),
        upper=True,
    )
    want = np.stack(
        [np.linalg.solve(np.triu(_rounded(a, name)), _rounded(_B, name)) for a in batch]
    )
    assert got.shape == (2, 3, 2)
    np.testing.assert_allclose(_numpy(got), want, **_TOL[name])


@pytest.mark.parametrize("dtype", [lucid.int32, lucid.float16, lucid.bool])
def test_the_dtype_gate_admits_complex_and_no_other_non_float(dtype: lucid.dtype) -> None:
    """Complex joined float32 / float64 at the gate; an integer, half or bool ``A`` stays refused."""
    a = lucid.tensor([[1, 2], [0, 1]]).to(dtype)
    with pytest.raises(NotImplementedError):
        lucid.linalg.solve_triangular(a, a, upper=True)
