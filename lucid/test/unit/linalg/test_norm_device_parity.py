"""``linalg.norm`` — device parity, and the keepdim heap-overflow regression.

Two bugs, both found by the metal-vs-cpu backward sweep (2026-07-13) and both
invisible to the tests of the time, which only covered 1-D vectors:

1. **Matrix semantics diverged by device.**  The CPU stream applied the
   *entrywise* p-norm to a matrix while the GPU stream forwarded to MLX with no
   axis, which applies *matrix* semantics: ``ord=2`` became the largest singular
   value and ``ord=1`` the max column sum.  The same call returned different
   numbers on different devices (3.892 vs 2.812 on a 4x5).

2. **CPU keepdim wrote out of bounds.**  ``norm_elementwise_loop`` mapped a
   surviving input axis to the *packed* output-axis counter even under keepdims,
   where the reduced axes are retained as size-1 and the ranks match.  For
   ``(2,3,4)`` reduced over dim 1 with ``keepdim=True`` that indexes up to 16 into
   an 8-element accumulator — a heap overflow that returned partly-zeroed data
   and then **segfaulted** (exit 139).  A 2-D input survived by luck (the
   mis-picked stride equals 1 there), which is why nothing caught it.

The first was settled the wrong way round.  Both streams were made entrywise, so
the divergence went away and the GPU — which had been *closer to right* — was
brought down to the CPU's reading.  ``lucid.linalg.norm`` now dispatches by rank
and axes the way its docstring always said, so ``ord`` on a matrix means the
matrix order; see ``test_norm_dispatch.py``.  What is still worth asserting here
is that the two devices agree, whatever the semantics are.

The second is guarded against the engine op directly.  ``norm`` reaches its
answer through ``sum``/``sqrt``/``max`` now and no longer calls the engine's
``linalg.norm`` at all, so testing it through the wrapper would no longer
execute the loop that overflowed.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
from lucid._dispatch import _unwrap, _wrap
from lucid.linalg import _la
from lucid.test._fixtures.devices import metal_available

_SHAPES = [(6,), (4, 5), (2, 3, 4), (2, 3, 4, 5)]
_DIMS = [0, 1, [0, 1]]

#: ``(shape, dim)`` pairs whose every axis exists.  A 1-D input has no axis 1,
#: so those pairs are not generated at all; what such a call does instead is
#: asserted by ``test_norm_rejects_an_axis_the_input_lacks``.
_IN_RANGE = [
    (shape, dim)
    for shape in _SHAPES
    for dim in _DIMS
    if max([dim] if isinstance(dim, int) else dim) < len(shape)
]


def _entrywise(x: np.ndarray, ord_: float, axis: object, keepdims: bool) -> np.ndarray:
    """The vector p-norm applied over ``axis`` — never a matrix norm."""
    if ord_ == 1:
        return np.abs(x).sum(axis=axis, keepdims=keepdims)
    return np.sqrt((x**2).sum(axis=axis, keepdims=keepdims))


# ── bug 2: the engine's elementwise loop, reached directly ────────────────────


@pytest.mark.parametrize(("shape", "dim"), _IN_RANGE)
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("ord_", [1.0, 2.0])
def test_engine_norm_keepdim_indexing(
    shape: tuple, dim: object, keepdim: bool, ord_: float
) -> None:
    """This segfaulted at 3-D once ``keepdim`` kept the ranks equal."""
    dims = [dim] if isinstance(dim, int) else dim
    x = np.random.default_rng(7).standard_normal(shape).astype(np.float32)
    axis = dim if isinstance(dim, int) else tuple(dim)
    ref = _entrywise(x, ord_, axis, keepdim)

    got = _wrap(_la.norm(_unwrap(lucid.tensor(x)), ord_, dims, keepdim)).numpy()
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_norm_rejects_an_axis_the_input_lacks(device: str) -> None:
    """An axis the input does not have is refused, the same way on both streams.

    This is the public entry point, which validates the axes before any kernel
    runs.  The engine op above does not validate on the CPU stream, which is
    why the out-of-range pairs are never generated against it.

    The engine's ``IndexError`` derives from ``RuntimeError``, not from the
    builtin ``IndexError``, so that is the class to catch.
    """
    x = lucid.tensor(np.ones(6, dtype=np.float32), device=device)
    with pytest.raises(RuntimeError, match="axis out of range"):
        LA.norm(x, dim=1)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        LA.norm(x, dim=(0, 1))


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("dim", [0, 1, [0, 1]])
@pytest.mark.parametrize("keepdim", [False, True])
def test_engine_norm_metal_matches_cpu(dim: object, keepdim: bool) -> None:
    dims = [dim] if isinstance(dim, int) else dim
    x = np.random.default_rng(5).standard_normal((2, 3, 4)).astype(np.float32)

    def run(device: str) -> np.ndarray:
        t = lucid.tensor(x, device=device)
        return _wrap(_la.norm(_unwrap(t), 2.0, dims, keepdim)).numpy()

    cpu, mtl = run("cpu"), run("metal")
    assert cpu.shape == mtl.shape
    np.testing.assert_allclose(mtl, cpu, atol=1e-4, rtol=1e-4)


# ── bug 1: the two devices must agree, whatever the semantics ─────────────────


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("kwargs", [{}, {"ord": 2, "dim": 0}, {"ord": 1, "dim": 0}])
def test_norm_metal_matches_cpu(shape: tuple, kwargs: dict) -> None:
    """The same call must return the same number on both streams."""
    x = np.random.default_rng(3).standard_normal(shape).astype(np.float32)
    cpu = LA.norm(lucid.tensor(x, device="cpu"), **kwargs).numpy()
    mtl = LA.norm(lucid.tensor(x, device="metal"), **kwargs).numpy()
    np.testing.assert_allclose(mtl, cpu, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("ord_", [1, -1, 2, -2, "fro", "nuc", float("inf"), float("-inf")])
def test_matrix_norm_metal_matches_cpu(ord_: object) -> None:
    """Every matrix order, including the three that need an SVD."""
    x = np.random.default_rng(11).standard_normal((4, 5)).astype(np.float32)
    cpu = LA.matrix_norm(lucid.tensor(x, device="cpu"), ord=ord_).numpy()
    mtl = LA.matrix_norm(lucid.tensor(x, device="metal"), ord=ord_).numpy()
    np.testing.assert_allclose(mtl, cpu, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("dim", [0, 1, [0, 1]])
@pytest.mark.parametrize("keepdim", [False, True])
def test_norm_metal_matches_cpu_reduced(dim: object, keepdim: bool) -> None:
    """Axis-reduced norms (incl. the two-axis matrix path) agree device-wise."""
    x = np.random.default_rng(5).standard_normal((2, 3, 4)).astype(np.float32)
    cpu = LA.norm(lucid.tensor(x, device="cpu"), dim=dim, keepdim=keepdim).numpy()
    mtl = LA.norm(lucid.tensor(x, device="metal"), dim=dim, keepdim=keepdim).numpy()
    assert cpu.shape == mtl.shape
    np.testing.assert_allclose(mtl, cpu, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_norm_backward_metal_matches_cpu() -> None:
    """The wrong forward also poisoned the gradient (dA = A/|A| uses the value)."""
    x = np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)

    def grad(device: str) -> np.ndarray:
        t = lucid.tensor(x, dtype=lucid.float32, device=device)
        t.requires_grad = True
        LA.norm(t).backward()
        return t.grad.numpy()

    np.testing.assert_allclose(grad("metal"), grad("cpu"), atol=1e-5, rtol=1e-4)


# ── the wrapper's own full reduction ──────────────────────────────────────────


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("keepdim", [False, True])
def test_norm_with_neither_ord_nor_dim_is_entrywise_at_every_rank(
    shape: tuple, keepdim: bool
) -> None:
    """No ``ord`` and no ``dim`` still flattens — that part did not change."""
    x = np.random.default_rng(len(shape)).standard_normal(shape).astype(np.float32)
    ref = _entrywise(x, 2.0, None, keepdim)
    got = LA.norm(lucid.tensor(x), keepdim=keepdim).numpy()
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)
