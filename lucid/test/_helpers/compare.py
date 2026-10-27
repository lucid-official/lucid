"""Numerical comparison utilities.

``assert_close`` is the canonical entry point for value comparisons.
It accepts ``lucid.Tensor`` / ``np.ndarray`` / reference-framework
tensors and does the right thing per pairing — pulling everything to
host memory through a single, well-defined path.

Default tolerances mirror the reference framework's
``allclose`` defaults (``atol=1e-5``, ``rtol=1e-4``) but every test
should pick a tolerance appropriate to the op (e.g. tighter for
elementwise math, looser for accumulating reductions).
"""

from typing import Any

import numpy as np

import lucid


def _to_numpy(x: Any) -> np.ndarray:
    """Coerce ``x`` to a numpy array regardless of source framework."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, lucid.Tensor):
        return x.numpy()
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        # Reference-framework Tensor.
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def assert_close(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    msg: str = "",
) -> None:
    """Element-wise close comparison with framework-agnostic operands."""
    a = _to_numpy(actual)
    e = _to_numpy(expected)
    if a.shape != e.shape:
        raise AssertionError(
            f"{msg + ': ' if msg else ''}shape mismatch — " f"actual={a.shape} expected={e.shape}"
        )
    np.testing.assert_allclose(a, e, atol=atol, rtol=rtol, err_msg=msg or "values do not match")


def assert_equal_int(actual: Any, expected: Any, *, msg: str = "") -> None:
    """Exact equality for integer / boolean operands (no tolerance)."""
    a = _to_numpy(actual)
    e = _to_numpy(expected)
    if a.shape != e.shape:
        raise AssertionError(
            f"{msg + ': ' if msg else ''}shape mismatch — " f"actual={a.shape} expected={e.shape}"
        )
    np.testing.assert_array_equal(a, e, err_msg=msg or "values do not match")


def assert_distributions_close(
    d1: Any,
    d2: Any,
    *,
    n_samples: int = 4096,
    atol_mean: float = 0.05,
    atol_var: float = 0.10,
    msg: str = "",
) -> None:
    """Loose Monte-Carlo comparison: draw ``n_samples`` from each
    distribution and check sample mean/variance agree."""
    s1 = _to_numpy(d1.sample((n_samples,)))
    s2 = _to_numpy(d2.sample((n_samples,)))
    np.testing.assert_allclose(
        s1.mean(axis=0),
        s2.mean(axis=0),
        atol=atol_mean,
        err_msg=f"{msg or ''}: sample mean mismatch".lstrip(": "),
    )
    np.testing.assert_allclose(
        s1.var(axis=0),
        s2.var(axis=0),
        atol=atol_var,
        err_msg=f"{msg or ''}: sample variance mismatch".lstrip(": "),
    )
