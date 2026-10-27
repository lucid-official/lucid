"""A complex tensor converted to a real dtype, on both devices.

Metal's conversion to bool read only the real part, so ``2j`` was False
there and True on the CPU and in the reference (LCD-408).  A complex
number is true when either part is nonzero; every real target keeps the
real part.  ``any`` and ``all`` judge an element's truth the same way.
"""

import math
import warnings
from typing import Any

import pytest

import lucid

_INF = math.inf
_NAN = math.nan
# Zero of either sign in either part, infinities and NaN in either part,
# and ordinary values on each axis.
_REAL = [0.0, -0.0, 0.0, -0.0, 1.5, 0.0, -2.5, _INF, -_INF, 0.0, 0.0, _NAN, 0.0, _NAN]
_IMAG = [0.0, 0.0, -0.0, -0.0, 0.0, 2.0, 1.0, 0.0, 0.0, _INF, -_INF, 0.0, _NAN, _NAN]


def _same(got: list[Any], want: list[Any]) -> bool:
    def eq(a: Any, b: Any) -> bool:
        if isinstance(a, float) and isinstance(b, float) and math.isnan(b):
            return math.isnan(a)
        return bool(a == b)

    return len(got) == len(want) and all(eq(a, b) for a, b in zip(got, want, strict=True))


def _pair(ref: Any, device: str) -> tuple[lucid.Tensor, Any]:
    z = lucid.complex(lucid.tensor(_REAL, device=device), lucid.tensor(_IMAG, device=device))
    return z, ref.complex(ref.tensor(_REAL), ref.tensor(_IMAG))


@pytest.mark.parity
class TestComplexToReal:
    @pytest.mark.parametrize("dtype", ["bool", "int32", "float32"])
    def test_matches_the_reference(self, ref: Any, device: str, dtype: str) -> None:
        z, rz = _pair(ref, device)
        with warnings.catch_warnings():
            # The reference warns that the imaginary part is discarded.
            warnings.simplefilter("ignore")
            want = rz.to(getattr(ref, dtype)).tolist()
            got = z.to(getattr(lucid, dtype)).to("cpu").tolist()
        assert _same(got, want), (got, want)

    @pytest.mark.parametrize("reduction", ["any", "all"])
    def test_logical_reductions_read_both_parts(
        self, ref: Any, device: str, reduction: str
    ) -> None:
        z, rz = _pair(ref, device)
        for i in range(len(_REAL)):
            got = getattr(lucid, reduction)(z[i : i + 1]).item()
            want = getattr(ref, reduction)(rz[i : i + 1]).item()
            assert got == want, (reduction, _REAL[i], _IMAG[i])
