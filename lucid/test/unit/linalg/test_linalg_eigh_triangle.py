"""``eigh`` / ``eigvalsh`` read the triangle ``UPLO`` names, on every path.

The defect class (LCD-287): which triangle the ``eigh`` kernel reads was
kept as a per-device table in ``lucid.linalg`` — ``{"cpu": "L", "metal":
"U"}`` — written down from the MLX of the day.  MLX 0.32.1 fixed its
``UPLO`` (ml-explore/mlx#3834), so an engine built against it read the
lower triangle on Metal too, the table flipped ``UPLO="L"`` onto the
upper triangle, and ``eigvalsh([[1, 2], [0, 0]])`` gave the spectrum of
``[[1, 2], [2, 0]]`` on Metal.  The backend owns the convention now: every
backend's kernel reads the lower triangle, and the Metal backend picks the
MLX flag that does so for the MLX it was built against.

These tests check that against numpy, with no reference framework, so they
run in every default suite.  The input's two triangles describe matrices
with different spectra, so reading the wrong one cannot pass by accident.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.linalg as LA
from lucid._dispatch import _unwrap, _wrap
from lucid.test._fixtures.devices import device_dtype_params


def _lopsided(batch: tuple[int, ...]) -> np.ndarray:
    """SPD matrices with +10 added to their upper triangle only."""
    rng = np.random.default_rng(0)
    m = rng.standard_normal(batch + (4, 4))
    spd = m @ np.swapaxes(m, -1, -2) + 4 * np.eye(4)
    return spd + np.triu(np.full((4, 4), 10.0), 1)


#: The inputs.  ``singular_triangle`` is the CI failure of LCD-287: its lower
#: triangle has the spectrum ``[0, 1]``, its upper ``(1 ± √17) / 2``.
_INPUTS: dict[str, np.ndarray] = {
    "lopsided": _lopsided(()),
    "lopsided_batch": _lopsided((3,)),
    "lopsided_batch_2d": _lopsided((2, 2)),
    "singular_triangle": np.array([[1.0, 2.0], [0.0, 0.0]]),
}


def _from_triangle(a: np.ndarray, uplo: str) -> np.ndarray:
    """The symmetric matrix that ``a``'s ``uplo`` triangle describes."""
    tri = np.tril if uplo == "L" else np.triu
    half = tri(a, -1 if uplo == "L" else 1)
    return tri(a) + np.swapaxes(half, -1, -2)


def _eigh_values(x: lucid.Tensor, uplo: str) -> lucid.Tensor:
    return LA.eigh(x, UPLO=uplo)[0]


def _eigh_rebuilt(x: lucid.Tensor, uplo: str) -> lucid.Tensor:
    # ``V diag(w) Vᵀ`` is the matrix the kernel decomposed, free of the
    # eigenvectors' sign ambiguity.
    w, v = LA.eigh(x, UPLO=uplo)
    return (v * w.unsqueeze(-2)) @ v.mT


def _eigvalsh(x: lucid.Tensor, uplo: str) -> lucid.Tensor:
    return LA.eigvalsh(x, UPLO=uplo)


#: Every way into the kernel, and whether it answers with the spectrum or
#: the rebuilt matrix.  ``requires_grad`` takes ``eigvalsh`` through
#: ``eigh`` and both through the gradient wrappers.
_PATHS: dict[str, tuple[Callable[[lucid.Tensor, str], lucid.Tensor], bool, bool]] = {
    "eigh_values": (_eigh_values, False, False),
    "eigh_vectors": (_eigh_rebuilt, True, False),
    "eigvalsh": (_eigvalsh, False, False),
    "eigh_values_grad": (_eigh_values, False, True),
    "eigh_vectors_grad": (_eigh_rebuilt, True, True),
    "eigvalsh_grad": (_eigvalsh, False, True),
}


def _tol(dtype: lucid.dtype) -> dict[str, float]:
    if dtype is lucid.float64:
        return {"rtol": 1e-10, "atol": 1e-10}
    return {"rtol": 1e-4, "atol": 1e-3}


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params([lucid.float32, lucid.float64]))
@pytest.mark.parametrize("name", list(_INPUTS))
@pytest.mark.parametrize("path", list(_PATHS))
@pytest.mark.parametrize("uplo", ["L", "U"])
def test_uplo_names_the_triangle_read(
    uplo: str, path: str, name: str, device: str, dtype: lucid.dtype
) -> None:
    a = _INPUTS[name]
    fn, rebuilds, requires_grad = _PATHS[path]
    x = lucid.tensor(a, dtype=dtype, device=device, requires_grad=requires_grad)
    got = fn(x, uplo).detach().to("cpu").numpy()
    want = _from_triangle(a, uplo) if rebuilds else np.linalg.eigvalsh(a, UPLO=uplo)
    np.testing.assert_allclose(got, want, **_tol(dtype))


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params([lucid.float32, lucid.float64]))
@pytest.mark.parametrize("name", list(_INPUTS))
@pytest.mark.parametrize("uplo", ["L", "U"])
def test_the_kernel_reads_the_triangle_linalg_assumes(
    uplo: str, name: str, device: str, dtype: lucid.dtype
) -> None:
    """``_C_engine.linalg.eigh`` passes ``UPLO`` straight to the backend
    kernel on every device without a Python transpose."""
    a = _INPUTS[name]
    w, _ = LA._la.eigh(_unwrap(lucid.tensor(a, dtype=dtype, device=device)), uplo)
    got = _wrap(w).to("cpu").numpy()
    want = np.linalg.eigvalsh(a, UPLO=uplo)
    np.testing.assert_allclose(got, want, **_tol(dtype))
