"""Every ``lucid.special`` function in float16 and bfloat16, against the reference.

The defect class: a composite special function evaluated its series,
continued fraction or polynomial fit in the 16-bit dtype it was handed.
Constants and intermediates outside float16's range turned into NaN —
``zeta(2, 1)``, ``gammainc(0.5, 0.3)``, ``bessel_y0`` on ``(0, 8]`` — or
``inf`` (``i1(12)``, representable at 18141), and bfloat16's 8 bits made
``bessel_j0(300)`` thirty times too small.  The reference evaluates its
kernels in float32 and rounds the answer; the composites now do the same
through ``lucid._ops.composite._opmath._in_float32``.

Each function is offered points whose answers fit the 16-bit range, on CPU
and Metal, and must agree to the dtype's rounding with the reference's
float32 answer at the same 16-bit points, rounded to 16 bits.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.special
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.parity

_X = [0.3, 1.5, 7.5, 12.0]
_Q = [1.0, 2.0, 3.0, 10.0]
_UNIT = [-0.7, -0.2, 0.4, 0.9]

#: name -> the arguments, with ``"x"`` / ``"q"`` / ``"u"`` standing for the
#: tensors above and anything else passed as is.
_CALLS: dict[str, tuple[Any, ...]] = {
    "zeta": ("q", "x"),
    "gammainc": ("x", "q"),
    "gammaincc": ("x", "q"),
    "xlog1py": ("x", "q"),
    "polygamma": (1, "x"),
    "multigammaln": ("q", 2),
    **{
        name: ("u", 3)
        for name in (
            "chebyshev_polynomial_t",
            "chebyshev_polynomial_u",
            "chebyshev_polynomial_v",
            "chebyshev_polynomial_w",
            "hermite_polynomial_h",
            "hermite_polynomial_he",
            "legendre_polynomial_p",
            "laguerre_polynomial_l",
        )
    },
    **{
        name: ("x", 3)
        for name in (
            "shifted_chebyshev_polynomial_t",
            "shifted_chebyshev_polynomial_u",
            "shifted_chebyshev_polynomial_v",
            "shifted_chebyshev_polynomial_w",
        )
    },
}

#: (function, dtype) pairs whose 16-bit path lives outside ``lucid.special``,
#: with why they still fail; strict, so a fix shows up as an XPASS.
_PENDING: dict[tuple[str, str], str] = {}

_TOLS = {"float16": (2e-3, 1e-3), "bfloat16": (1.6e-2, 1e-2)}


def _cases() -> list[Any]:
    params = []
    for name in sorted(lucid.special.__all__):
        for dtype in ("float16", "bfloat16"):
            reason = _PENDING.get((name, dtype))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            params.append(pytest.param(name, dtype, marks=marks, id=f"{name}-{dtype}"))
    return params


def _args(name: str, make: Callable[[list[float]], Any]) -> list[Any]:
    tensors = {"x": _X, "q": _Q, "u": _UNIT}
    spec = _CALLS.get(name, ("x",))
    return [make(tensors[a]) if isinstance(a, str) else a for a in spec]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "metal", marks=pytest.mark.skipif(not metal_available(), reason="needs Metal")
        ),
    ],
)
@pytest.mark.parametrize("name,dtype", _cases())
def test_sixteen_bit_answers_match(name: str, dtype: str, device: str, ref: Any) -> None:
    ref_fn = getattr(ref.special, name, None)
    if ref_fn is None:
        pytest.skip(f"the reference has no special.{name}")
    lucid_dtype, ref_dtype = getattr(lucid, dtype), getattr(ref, dtype)

    def ours_of(v: list[float]) -> lucid.Tensor:
        return lucid.tensor(np.asarray(v)).to(lucid_dtype).to(device)

    # The 16-bit points widened to float32, and the answer rounded back: the
    # reference's opmath, spelled out, since several of its special kernels
    # are not built for 16-bit dtypes at all.
    def theirs_of(v: list[float]) -> Any:
        return ref.tensor(np.asarray(v)).to(ref_dtype).float()

    out = getattr(lucid.special, name)(*_args(name, ours_of))
    assert out.dtype == lucid_dtype
    ours = out.to("cpu").to(lucid.float64).numpy().tolist()
    theirs = ref_fn(*_args(name, theirs_of)).to(ref_dtype).double().tolist()
    rtol, atol = _TOLS[dtype]
    wrong = [
        (o, t)
        for o, t in zip(ours, theirs, strict=True)
        if not ((math.isnan(o) and math.isnan(t)) or math.isclose(o, t, rel_tol=rtol, abs_tol=atol))
    ]
    assert not wrong, wrong
