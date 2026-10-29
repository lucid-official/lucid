"""``ndtri`` and ``log_ndtr`` are double-precision accurate in float64 (LCD-664).

``ndtri`` is the Beasley-Springer-Moro rational approximation, good to 1.15e-9
relative: right for float32, 1e7 times too coarse for float64 (``ndtri(0.1)``
was ``-1.2815515641401563`` for ``-1.2815515655446004``).  float64 now takes
one Newton step on top of it, on ``Phi(x) - p`` in the centre and on
``log Phi(x) - log p`` in the tails, so it holds to the last digits from
``1e-300`` to ``1 - 1e-16``, a denormal ``p`` included.

``log_ndtr`` read ``erfcx`` through a 24-term continued fraction that is good to
1e-10 at ``x = 2`` and 1e-12 at ``x = 2.5``, so ``log_ndtr(3)`` was off by 3e-11
and ``log_ndtr(-3)`` by 5e-12.  float64 takes 60 terms.

Everything is compared with the reference at ``rtol = 1e-13``.  float64 lives on
the CPU stream only (MLX has no double).
"""

import math
from types import ModuleType

import pytest

import lucid
import lucid.special as S

_RTOL = 1e-13

#: Both tails, the central/tail seam at 0.02425, the middle, and the last
#: doubles on either side of 1.
_P = [
    5e-324, 1e-310, 1e-300, 1e-200, 1e-100, 1e-50, 1e-20, 1e-10, 1e-5, 1e-3, 0.01, 0.02,
    0.02424, 0.02425, 0.02426, 0.03, 0.1, 0.3, 0.4999, 0.5 - 1e-9, 0.5 - 1e-15, 0.5, 0.5 + 1e-15,
    0.7, 0.9, 0.97, 0.9757, 0.999, 1 - 1e-5, 1 - 1e-10, 1 - 1e-15, 1 - 2.0**-53,
]  # fmt: skip

#: Round and awkward ``x`` on both sides of every branch edge of ``log_ndtr``
#: (-1 and 2*sqrt(2)), and deep tails.
_X = [
    -38.0, -30.0, -20.0, -10.0, -5.0, -3.0, -2.5, -2.0, -1.5, -1.0001, -1.0, -0.9999, -0.5, -1e-9, 0.0,
    1e-9, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8, 2.8284, 2.82843, 2.8285, 2.9, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0,
    10.0, 15.0, 20.0, 30.0, 37.0,
]  # fmt: skip


def _close(got: list[float], want: list[float], rtol: float, what: str) -> None:
    for g, w in zip(got, want, strict=True):
        assert math.isclose(g, w, rel_tol=rtol, abs_tol=0.0), f"{what}: got {g!r}, want {w!r}"


def test_ndtri_float64_matches_the_reference(ref: ModuleType) -> None:
    got = S.ndtri(lucid.tensor(_P, dtype=lucid.float64)).tolist()
    want = ref.special.ndtri(ref.tensor(_P, dtype=ref.float64)).tolist()
    _close(got, want, _RTOL, "ndtri")
    assert got[_P.index(0.5)] == 0.0


def test_ndtri_float64_is_the_last_digits_not_the_approximation() -> None:
    # The card's case: the approximation alone is 1.4e-9 off here.
    got = S.ndtri(lucid.tensor(0.1, dtype=lucid.float64)).item()
    assert math.isclose(got, -1.2815515655446004, rel_tol=1e-15)


def test_ndtri_float64_is_odd_about_one_half() -> None:
    # ``1 - p`` is exact for p in [1/2, 1] (Sterbenz), so each pair is an exact
    # mirror and the two answers are the same number, signed.
    hi = [0.5 + 2.0**-53, 0.6, 0.7, 0.9, 0.97, 0.99, 0.9999, 1 - 1e-9, 1 - 1e-15]
    lo = [1.0 - v for v in hi]
    up = S.ndtri(lucid.tensor(hi, dtype=lucid.float64)).tolist()
    down = S.ndtri(lucid.tensor(lo, dtype=lucid.float64)).tolist()
    _close([-v for v in down], up, 3e-15, "odd about 1/2")


def test_ndtri_float64_keeps_its_boundaries() -> None:
    p = [0.0, 1.0, -0.1, 1.1, math.nan, math.inf, -math.inf, -0.0]
    got = S.ndtri(lucid.tensor(p, dtype=lucid.float64)).tolist()
    assert got[0] == -math.inf and got[1] == math.inf
    assert all(math.isnan(v) for v in got[2:7])
    assert got[7] == -math.inf


def test_ndtri_float64_gradient_is_the_exact_reciprocal_density() -> None:
    # d ndtri / dp = 1 / pdf(x) = sqrt(2 pi) exp(x^2 / 2), which the refined
    # step differentiates to; the approximation's own slope is 1e-9 off.
    p = lucid.tensor([0.1, 0.5, 0.02, 0.9, 1e-10, 0.999], dtype=lucid.float64, requires_grad=True)
    S.ndtri(p).sum().backward()
    assert p.grad is not None
    x = S.ndtri(p.detach()).tolist()
    want = [math.sqrt(2.0 * math.pi) * math.exp(0.5 * v * v) for v in x]
    _close(p.grad.tolist(), want, 1e-12, "d ndtri / dp")


def test_ndtri_float32_is_still_the_approximation() -> None:
    p = [1e-8, 1e-3, 0.02, 0.1, 0.3, 0.5, 0.9, 0.999, 1 - 1e-6]
    p32 = lucid.tensor(p, dtype=lucid.float32)
    got = S.ndtri(p32).tolist()
    exact = S.ndtri(p32.to(lucid.float64)).tolist()
    # float32 holds the approximation's 1.15e-9 many times over.
    _close(got, exact, 2e-6, "float32 ndtri")


def test_log_ndtr_float64_matches_the_reference(ref: ModuleType) -> None:
    got = S.log_ndtr(lucid.tensor(_X, dtype=lucid.float64)).tolist()
    want = ref.special.log_ndtr(ref.tensor(_X, dtype=ref.float64)).tolist()
    _close(got, want, _RTOL, "log_ndtr")


def test_log_ndtr_float64_round_trips_through_ndtri() -> None:
    # log Phi(ndtri(p)) = log p holds across both tails and the centre when
    # both halves are accurate: it is the property, not a table.
    p = lucid.tensor(
        [1e-300, 1e-100, 1e-20, 1e-5, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9], dtype=lucid.float64
    )
    back = S.log_ndtr(S.ndtri(p)).tolist()
    _close(back, [math.log(v) for v in p.tolist()], 1e-12, "log_ndtr(ndtri(p))")


def test_erfcx_float64_matches_the_reference(ref: ModuleType) -> None:
    # The continued fraction both log_ndtr tails read, from its cutoff at 2.
    x = [0.5, 1.0, 1.9, 1.999, 2.0, 2.001, 2.12, 2.5, 3.0, 4.0, 6.0, 10.0, 30.0]
    got = S.erfcx(lucid.tensor(x, dtype=lucid.float64)).tolist()
    want = ref.special.erfcx(ref.tensor(x, dtype=ref.float64)).tolist()
    _close(got, want, _RTOL, "erfcx")


@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float16, lucid.bfloat16])
def test_narrow_dtypes_keep_their_dtype_and_float32_path(dtype: lucid.dtype) -> None:
    p = lucid.tensor([0.025, 0.5, 0.975], dtype=dtype)
    out = S.ndtri(p)
    assert out.dtype == dtype
    assert abs(out.tolist()[0] + 1.96) < 0.02
