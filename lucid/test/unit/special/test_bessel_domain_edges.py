"""The Bessel family keeps one domain rule and a clean gradient in both dtypes
(LCD-418).

The ``float64`` path gave ``Y`` / ``K`` the reference's edges (``-inf`` /
``+inf`` at 0, NaN below) in LCD-110; ``float32`` kept finite garbage there:
``bessel_y0(0) = -0.0738``, ``modified_bessel_k0(0) = 0.1159``,
``scaled_modified_bessel_k0(-inf) = 0.84``.  Both paths now run through
``_bessel._half_line_split``.

The ``float32`` branches also ran on arguments outside their interval — the
large branch at 0, the small polynomial at ``1e8`` — and ``where`` carried
the unused branch's ``nan`` into the gradient: ``K0'`` and ``I0'`` were NaN
near 0, ``J0'`` NaN at large ``x``.  Each branch now runs on its own clamped
interval; values inside the domain are unchanged bit for bit (checked over a
sorted 4600-point grid, CPU and Metal, before landing).
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.special as S
from lucid.test._fixtures.devices import metal_available

UnaryFn = Callable[[lucid.Tensor], lucid.Tensor]

#: name -> (Lucid function, reference ``special`` attribute)
_FUNCS: dict[str, tuple[UnaryFn, str]] = {
    "i0": (lucid.i0, "i0"),
    "i0e": (S.i0e, "i0e"),
    "i1": (S.i1, "i1"),
    "i1e": (S.i1e, "i1e"),
    "j0": (S.bessel_j0, "bessel_j0"),
    "j1": (S.bessel_j1, "bessel_j1"),
    "y0": (S.bessel_y0, "bessel_y0"),
    "y1": (S.bessel_y1, "bessel_y1"),
    "k0": (S.modified_bessel_k0, "modified_bessel_k0"),
    "k1": (S.modified_bessel_k1, "modified_bessel_k1"),
    "k0e": (S.scaled_modified_bessel_k0, "scaled_modified_bessel_k0"),
    "k1e": (S.scaled_modified_bessel_k1, "scaled_modified_bessel_k1"),
}
_HALF_LINE = ["y0", "y1", "k0", "k1", "k0e", "k1e"]
_EDGES = [0.0, -0.0, -1e-6, -1.0, -2.5, -10.0, math.inf, -math.inf, math.nan]
_DTYPES = {"float32": lucid.float32, "float64": lucid.float64}


def _cases() -> list[tuple[str, str]]:
    """``(device, dtype)`` pairs; Metal has no ``float64``."""
    devices = ["cpu", "metal"] if metal_available() else ["cpu"]
    return [(d, t) for d in devices for t in _DTYPES if (d, t) != ("metal", "float64")]


def _edge_run(name: str, device: str, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    x = lucid.tensor(_EDGES, dtype=_DTYPES[dtype], device=device, requires_grad=True)
    y = _FUNCS[name][0](x)
    y.sum().backward()
    return y.to("cpu").numpy(), x.grad.to("cpu").numpy()


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_FUNCS))
@pytest.mark.parametrize("device,dtype", _cases())
def test_edges_match_the_reference(name: str, device: str, dtype: str, ref: Any) -> None:
    got, _ = _edge_run(name, device, dtype)
    want = getattr(ref.special, _FUNCS[name][1])(
        ref.tensor(_EDGES, dtype=getattr(ref, dtype))
    ).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("name", _HALF_LINE)
@pytest.mark.parametrize("device,dtype", _cases())
def test_edge_gradients_follow_one_rule(name: str, device: str, dtype: str) -> None:
    # The value at 0 and below is a constant of the domain rule, so it carries
    # no gradient; NaN in, NaN out.  The same in either dtype.
    _, grad = _edge_run(name, device, dtype)
    _, f64 = _edge_run(name, "cpu", "float64")
    np.testing.assert_array_equal(grad, f64)
    assert np.all(grad[:6] == 0.0) and np.all(grad[7] == 0.0)


# Tiny arguments, both sides of every branch edge (float32: 2, 3, 3.75, 8;
# float64: 2, 5, 8), and large ones where the other branch overflows.
_GRID = [1e-30, 1e-6, 1e-3, 0.7, 1.99, 2.0, 2.01, 2.99, 3.0, 3.01, 3.74, 3.75]
_GRID += [3.76, 4.99, 5.0, 5.01, 7.99, 8.0, 8.01, 20.0, 60.0, 107.0, 1e4, 1e8]


def _identity(name: str, v: dict[str, Any]) -> Any:
    """``f'(x)`` from the reference's values ``v`` (closed-form identities)."""
    x = v["x"]
    return {
        "i0": v["i1"],
        "i0e": v["i1e"] - np.sign(x) * v["i0e"],
        "i1": v["i0"] - v["i1"] / x,
        "i1e": v["i0e"] - v["i1e"] / x - np.sign(x) * v["i1e"],
        "j0": -v["j1"],
        "j1": v["j0"] - v["j1"] / x,
        "y0": -v["y1"],
        "y1": v["y0"] - v["y1"] / x,
        "k0": -v["k1"],
        "k1": -v["k0"] - v["k1"] / x,
        "k0e": v["k0e"] - v["k1e"],
        "k1e": v["k1e"] - v["k0e"] - v["k1e"] / x,
    }[name]


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_FUNCS))
@pytest.mark.parametrize("device", [d for d, t in _cases() if t == "float32"])
def test_float32_gradient_matches_the_identity(name: str, device: str, ref: Any) -> None:
    # Past ~1e4 the float32 phase of J / Y is rounding noise, and I overflows
    # float32 past ~88; the gradient there only has to be finite.
    xs = np.asarray([v for v in _GRID if name[0] in "k" or v <= 60.0])
    if name[0] in "ij":
        xs = np.concatenate([xs, -xs])
    xr = ref.tensor(xs, dtype=ref.float64)
    v = {n: getattr(ref.special, a)(xr).numpy() for n, (_, a) in _FUNCS.items()}
    v["x"] = xs
    x = lucid.tensor(xs.astype(np.float32), device=device, requires_grad=True)
    _FUNCS[name][0](x).sum().backward()
    # A derivative past float32's range (Y1', K1' at 1e-30) is its infinity;
    # the table also builds the identities this name does not use, at
    # arguments where those overflow.
    with np.errstate(over="ignore", invalid="ignore"):
        want = _identity(name, v).astype(np.float32)
    np.testing.assert_allclose(x.grad.to("cpu").numpy(), want, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize("name", list(_FUNCS))
@pytest.mark.parametrize("device", [d for d, t in _cases() if t == "float32"])
def test_float32_gradient_is_not_nan_inside_the_domain(name: str, device: str) -> None:
    # Where the value is finite the unused branch adds nothing: I0' was NaN
    # at 1e-6, J0' at 1e8, K0' at 107.
    xs = [v for v in _GRID if name[0] != "i" or v <= 60.0]
    x = lucid.tensor(xs, device=device, requires_grad=True)
    y = _FUNCS[name][0](x)
    y.sum().backward()
    finite = np.isfinite(y.to("cpu").numpy())
    assert not np.isnan(x.grad.to("cpu").numpy()[finite]).any()
