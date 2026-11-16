"""LCD-1037: ``StudentT`` is range-safe in ``df``.

``log_prob`` was ``lgamma((df+1)/2) - lgamma(df/2) - log(pi df)/2 - log(scale) - (df+1)/2 log1p(z^2/df)``:
the two ``lgamma`` of a large ``df`` are each the whole answer plus its rounding, so their
difference is noise while the density is an ordinary number.  In float32, ``StudentT(1e6)`` at 1 was
-1.457 and ``StudentT(1e8)`` -9.783 for -1.4189 (``df = 1e30`` -35.1; the Normal limit is -1.4189);
float64 was 1.2e-7 off at ``df = 1e8`` for the same reason.  ``entropy`` had the same pair plus a
difference of two ``digamma`` (float32, ``df = 1e8``: 9.21 for 1.4189).  Hand-writing ``log1p`` alone
does not repair it, and there is a second cancellation behind the first: the slope of
``(df+1)/2 log1p(z^2/df)`` in ``df`` is the difference of two ``z^2/(2 df)``, so even with a stable
value the gradient in ``df`` was noise from about ``df = 1e6`` (``_student_math``).  The entropy's
slope in ``df`` has its own: ``(a + 1/2) F(a)`` is 1/2 plus a small part, so its slope is a sum of
two ``1/(2a)`` that cancel.

The oracle is the closed form in 400-digit arithmetic, written out below (mpmath, scratch only),
at inputs that float32 holds exactly; the gradients in ``x``, ``loc`` and ``scale`` are the closed
forms ``-(df+1) z / (scale (df+z^2))`` and ``df (z^2 - 1) / (scale (df+z^2))``, the gradient in
``df`` ``[digamma((df+1)/2) - digamma(df/2)]/2 - 1/(2 df) - log1p(u)/2 + (df+1) u / (2 df (1+u))``
with ``u = z^2/df``.  Errors are in units of the dtype's epsilon.
"""

import math
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
from lucid.distributions._student_math import _digamma_half_gap, _gamma_half_excess
from lucid.test._fixtures.devices import device_dtype_params

_FLOATS = device_dtype_params([lucid.float32, lucid.float64])
_EPS = {lucid.float32: 2.0**-23, lucid.float64: 2.0**-52}
_TINY = {lucid.float32: 1.1754943508222875e-38, lucid.float64: 2.2250738585072014e-308}


def _table(text: str) -> tuple[tuple[float, ...], ...]:
    """Rows of whitespace-separated floats."""
    return tuple(tuple(float(v) for v in line.split()) for line in text.strip().splitlines())


# (df, x, loc, scale, log_prob, d/d df, d/d x, d/d scale); d/d loc = -d/d x.  Every df and every
# input is a float32 number, so one table serves both dtypes.
_LOG_PROB = _table("""
0.0009765625 0.0 0.0 1.0 -4.159559593000005 511.3076551510054 0.0 -1.0
0.0009765625 1.0 0.0 1.0 -7.629168523108317 1019.8414312052191 -1.0 0.0
0.0009765625 3.0 1.0 0.5 -8.32321153361254 1018.9243157619555 -0.500457735733903 0.00183094293561184
0.0009765625 -1.5 0.5 1.0 -8.322626260229274 1019.5235584818439 0.5003661215523554 0.000732243104710764
0.0009765625 0.75 0.25 2.0 -6.941938683541897 1013.8358461314422 -1.9711538461538463 -0.007211538461538462
0.0009765625 100.0 0.0 1.0 -12.238347474890961 1015.7366989645662 -0.010009764647483922 0.0009764647483921145
0.5 0.0 0.0 1.0 -1.3105329259115095 0.5707963267948967 0.0 -1.0
0.5 1.0 0.0 1.0 -2.1344921424125918 1.0214901824608418 -1.0 0.0
0.5 3.0 1.0 0.5 -3.2397664164514244 0.27708800060711103 -0.7272727272727273 0.9090909090909091
0.5 -1.5 0.5 1.0 -2.958451358913674 0.8055173714601203 0.6666666666666666 0.3333333333333333
0.5 0.75 0.25 2.0 -2.0920173832137423 0.6785714756333716 -0.3333333333333333 -0.4166666666666667
0.5 100.0 0.0 1.0 -8.738186089376137 -2.8810474450983756 -0.014999250037498125 0.4999250037498125
1.0 0.0 0.0 1.0 -1.1447298858494002 0.19314718055994531 0.0 -1.0
1.0 1.0 0.0 1.0 -1.8378770664093456 0.34657359027997264 -1.0 0.0
1.0 3.0 1.0 0.5 -3.284796049345671 -0.28228302087992746 -0.9411764705882353 1.7647058823529411
1.0 -1.5 0.5 1.0 -2.7541677982835004 0.18842822434289513 0.8 0.6
1.0 0.75 0.25 2.0 -1.8985016882257804 0.2216583990634926 -0.23529411764705882 -0.4411764705882353
1.0 100.0 0.0 1.0 -10.355170252825916 -3.4121729929293125 -0.019998000199980003 0.9998000199980002
3.0 0.0 0.0 1.0 -1.0008888496235098 0.026480513893278643 0.0 -1.0
3.0 1.0 0.0 1.0 -1.5762529945270716 0.04930614433405484 -1.0 0.0
3.0 3.0 1.0 0.5 -3.999395050060226 -0.3350293225839569 -1.6842105263157894 4.7368421052631575
3.0 -1.5 0.5 1.0 -2.695484570397917 -0.016216035347942213 1.1428571428571428 1.2857142857142858
3.0 0.75 0.25 2.0 -1.7352746045889265 0.02977631246878155 -0.16326530612244897 -0.45918367346938777
3.0 100.0 0.0 1.0 -17.224944926257653 -3.363066778616585 -0.039988003598920324 2.9988003598920323
30.0 0.0 0.0 1.0 -0.9272703253788457 0.0002776237981191988 0.0 -1.0
30.0 1.0 0.0 1.0 -1.4355125791352041 0.0005493790532904301 -1.0 0.0
30.0 3.0 1.0 0.5 -6.8595053746364645 -0.03373423868781438 -5.391304347826087 19.565217391304348
30.0 -1.5 0.5 1.0 -2.8672990411659387 -0.0015196339533936124 1.8235294117647058 2.6470588235294117
30.0 0.75 0.25 2.0 -1.6526755820980719 0.00031119177035798925 -0.1288981288981289 -0.4677754677754678
30.0 100.0 0.0 1.0 -91.01541706443315 -2.390670323090402 -0.3090727816550349 29.907278165503488
1000.0 0.0 0.0 1.0 -0.9191885331630061 2.4999987500025e-07 0.0 -1.0
1000.0 1.0 0.0 1.0 -1.4194384498713144 4.998333332336666e-07 -1.0 0.0
1000.0 3.0 1.0 0.5 -8.17065260532628 -5.4534814490546816e-05 -7.881889763779528 29.52755905511811
1000.0 -1.5 0.5 1.0 -2.9171951785665016 -1.73673051125613e-06 1.9940239043824701 2.9880478087649402
1000.0 0.75 0.25 2.0 -1.6436159862246178 2.8027144087179933e-07 -0.125117180176239 -0.46872070495594026
1000.0 100.0 0.0 1.0 -1201.0657725687474 -0.7439473863993102 -9.1 909.0
1000000.0 0.0 0.0 1.0 -0.9189387832046727 2.49999999999875e-13 0.0 -1.0
1000000.0 1.0 0.0 1.0 -1.4189390332045895 4.999998333333333e-13 -1.0 0.0
1000000.0 3.0 1.0 0.5 -8.225735603263386 -5.5748762689194405e-11 -7.999880001919969 29.999520007679877
1000000.0 -1.5 0.5 1.0 -2.9189367832113393 -1.7499866667307913e-12 1.999994000024 2.999988000048
1000000.0 0.75 0.25 2.0 -1.6433359940380545 2.802734356281303e-13 -0.12500011718749268 -0.46874997070312685
1000000.0 100.0 0.0 1.0 -4976.089340532672 -2.4665426334041423e-05 -99.01 9900.0
100000000.0 0.0 0.0 1.0 -0.9189385357046728 2.5e-17 0.0 -1.0
100000000.0 1.0 0.0 1.0 -1.4189385382046726 4.999999983333333e-17 -1.0 0.0
100000000.0 3.0 1.0 0.5 -8.225790795144789 -5.574998762666892e-15 -7.999998800000192 29.999995200000768
100000000.0 -1.5 0.5 1.0 -2.9189385157046734 -1.7499998666666732e-16 1.9999999400000024 2.9999998800000047
100000000.0 0.75 0.25 2.0 -1.6433357165673523 2.8027343748128255e-17 -0.12500000117187499 -0.46874999970703124
100000000.0 100.0 0.0 1.0 -5000.669005198622 -2.4991667291576674e-09 -99.99000199980001 9998.000199980002
10000000000.0 0.0 0.0 1.0 -0.9189385332296728 2.5e-21 0.0 -1.0
10000000000.0 1.0 0.0 1.0 -1.4189385332546727 4.999999999833333e-21 -1.0 0.0
10000000000.0 3.0 1.0 0.5 -8.225791347069727 -5.574999987626667e-19 -7.999999988 29.999999952
10000000000.0 -1.5 0.5 1.0 -2.9189385330296727 -1.7499999986666667e-20 1.9999999994 2.9999999988
10000000000.0 0.75 0.25 2.0 -1.6433357137926454 2.8027343749981284e-21 -0.12500000001171874 -0.4687499999970703
10000000000.0 100.0 0.0 1.0 -5000.916439034896 -2.4994966421704163e-13 -99.99990001009999 9998.99000101
1125899906842624.0 0.0 0.0 1.0 -0.918938533204673 1.9721522630525295e-31 0.0 -1.0
1125899906842624.0 1.0 0.0 1.0 -1.4189385332046731 3.9443045261050577e-31 -1.0 0.0
1125899906842624.0 3.0 1.0 0.5 -8.225791352644677 -4.397899546607054e-29 -7.999999999999893 29.999999999999574
1125899906842624.0 -1.5 0.5 1.0 -2.9189385332046713 -1.3805065841367614e-30 1.9999999999999947 2.9999999999999893
1125899906842624.0 0.75 0.25 2.0 -1.6433357137646183 2.2109675761565468e-31 -0.1250000000000001 -0.46875
1125899906842624.0 100.0 0.0 1.0 -5000.918938511005 -1.971757812855045e-23 -99.9999999991119 9998.999999911191
1.2676506002282294e+30 0.0 0.0 1.0 -0.9189385332046728 1.5557538194652854e-61 0.0 -1.0
1.2676506002282294e+30 1.0 0.0 1.0 -1.4189385332046727 3.111507638930571e-61 -1.0 0.0
1.2676506002282294e+30 3.0 1.0 0.5 -8.225791352644727 -3.4693310174075865e-59 -8.0 30.0
1.2676506002282294e+30 -1.5 0.5 1.0 -2.9189385332046727 -1.0890276736256998e-60 2.0 3.0
1.2676506002282294e+30 0.75 0.25 2.0 -1.643335713764618 1.74414588354116e-61 -0.125 -0.46875
1.2676506002282294e+30 100.0 0.0 1.0 -5000.918938533205 -1.5554426531438542e-53 -100.0 9999.0
""")

# (df, scale, entropy, d entropy / d df); d entropy / d scale = 1 / scale
_ENTROPY = _table("""
0.0009765625 1.0 1028.4665386259887 -1049087.1784896126
0.0009765625 2.0 1029.1596858065489 -1049087.1784896126
0.5 1.0 3.6667274161038543 -4.495793565063314
0.5 2.0 4.3598745966638 -4.495793565063314
1.0 1.0 2.5310242469692907 -1.1449340668482264
1.0 2.0 3.224171427529236 -1.1449340668482264
3.0 1.0 1.773477571863291 -0.12320146702978621
3.0 2.0 2.466624752423236 -0.12320146702978621
7.0 1.0 1.5663782617176052 -0.02164102929766765
7.0 2.0 2.2595254422775506 -0.02164102929766765
8.0 1.0 1.5474982728250088 -0.01647016857073614
8.0 2.0 2.2406454533849542 -0.01647016857073614
17.0 1.0 1.4785919128942644 -0.0035557043061433327
17.0 2.0 2.17173909345421 -0.0035557043061433327
18.0 1.0 1.4752360890704899 -0.003167171790651088
18.0 2.0 2.1683832696304353 -0.003167171790651088
30.0 1.0 1.4525433297872075 -0.0011289938824940592
30.0 2.0 2.1456905103471526 -0.0011289938824940592
1000.0 1.0 1.4199387830378813 -1.0004994995015015e-06
1000.0 2.0 2.1130859635978267 -1.0004994995015015e-06
10000.0 1.0 1.419038535704506 -1.0000499949995002e-08
10000.0 2.0 2.1121857162644515 -1.0000499949995002e-08
1000000.0 1.0 1.4189395332049228 -1.0000004999995e-12
1000000.0 2.0 2.1120867137648682 -1.0000004999995e-12
10000000.0 1.0 1.4189386332046752 -1.000000049999995e-14
10000000.0 2.0 2.1120858137646206 -1.000000049999995e-14
100000000.0 1.0 1.4189385432046728 -1.000000005e-16
100000000.0 2.0 2.112085723764618 -1.000000005e-16
1000000000.0 1.0 1.4189385342046728 -1.0000000005e-18
1000000000.0 2.0 2.112085714764618 -1.0000000005e-18
10000000000.0 1.0 1.4189385333046727 -1.00000000005e-20
10000000000.0 2.0 2.112085713864618 -1.00000000005e-20
1099511627776.0 1.0 1.4189385332055822 -8.271806125534038e-25
1099511627776.0 2.0 2.1120857137655276 -8.271806125534038e-25
1125899906842624.0 1.0 1.4189385332046736 -7.888609052210122e-31
1125899906842624.0 2.0 2.112085713764619 -7.888609052210122e-31
1.2676506002282294e+30 1.0 1.4189385332046727 -6.223015277861142e-61
1.2676506002282294e+30 2.0 2.112085713764618 -6.223015277861142e-61
""")

# (a, lgamma(a + 1/2) - lgamma(a) - log(a)/2, digamma(a + 1/2) - digamma(a))
_GAP = _table("""
9.5367431640625e-07 -6.359108184746584 1048574.6137087763
0.0009765625 -2.8947231964662516 1022.6169115365883
0.125 -0.6188680912249382 6.935783898719288
0.5 -0.22579135264472744 1.3862943611198906
1.0 -0.12078223763524522 0.6137056388801094
2.5 -0.049681055850051385 0.21962769445322394
3.0 -0.04147972254692593 0.18037230554677605
7.0 -0.017842049779024694 0.07397314914761965
8.0 -0.015614874604334552 0.06444933962381012
30.0 -0.004166473829678374 0.016805536276136207
1000.0 -0.00012499999479166822 0.000500124999984375
1000000.0 -1.2499999999999478e-07 5.00000125e-07
10000000000.0 -1.25e-11 5.000000000125e-11
1125899906842624.0 -1.1102230246251565e-16 4.440892098500627e-16
1.2676506002282294e+30 -9.860761315262648e-32 3.944304526105059e-31
""")

# (df, x, d2/d df2, d2/d df dx, d2/dx2) at loc 0, scale 1
_HESSIAN = _table("""
3.0 2.0 -0.007841183083977164 -0.12244897959183673 0.08163265306122448
1000.0 1.5 -2.225307639989873e-10 -1.8665908913724365e-06 -0.9942685082286482
1000000.0 3.0 3.09993925083839e-17 -2.399956800583193e-11 -0.9999740003779953
""")


def _t(value: float, device: str, dtype: lucid.dtype, grad: bool = False) -> lucid.Tensor:
    return lucid.tensor(value, dtype=dtype, device=device, requires_grad=grad)


def _within(
    got: float, want: float, dtype: lucid.dtype, k: float, floor: float | None = None
) -> bool:
    """``|got - want| <= k eps max(|want|, floor)``; ``floor`` defaults to the dtype's smallest normal."""
    return abs(got - want) <= k * _EPS[dtype] * max(
        abs(want), _TINY[dtype] if floor is None else floor
    )


def _gradients(
    df: float, x: float, loc: float, scale: float, device: str, dtype: lucid.dtype
) -> tuple[float, float, float, float, float]:
    """``(log_prob, d df, d x, d loc, d scale)`` of ``StudentT`` at the point, by autograd."""
    df_t, x_t, loc_t, scale_t = (_t(v, device, dtype, True) for v in (df, x, loc, scale))
    out = D.StudentT(df_t, loc_t, scale_t, validate_args=False).log_prob(x_t)
    out.backward()
    grads = [df_t.grad, x_t.grad, loc_t.grad, scale_t.grad]
    assert all(g is not None for g in grads)
    ddf, dx, dloc, dscale = (g.item() for g in grads if g is not None)
    return out.item(), ddf, dx, dloc, dscale


def _normal_gradients(
    x: float, loc: float, scale: float, device: str, dtype: lucid.dtype
) -> tuple[float, float, float, float]:
    """``(log_prob, d x, d loc, d scale)`` of ``Normal`` at the point."""
    x_t, loc_t, scale_t = (_t(v, device, dtype, True) for v in (x, loc, scale))
    out = D.Normal(loc_t, scale_t).log_prob(x_t)
    out.backward()
    grads = [x_t.grad, loc_t.grad, scale_t.grad]
    assert all(g is not None for g in grads)
    dx, dloc, dscale = (g.item() for g in grads if g is not None)
    return out.item(), dx, dloc, dscale


_POINTS = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (3.0, 1.0, 0.5), (0.75, 0.25, 2.0)]


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("df", sorted({row[0] for row in _LOG_PROB}))
def test_log_prob_and_gradients_match_the_closed_form(
    df: float, device: str, dtype: lucid.dtype
) -> None:
    """``df`` from 1e-3 to 1e30: the value and all four gradients to a few ulp.

    The value is held to 8 eps and the gradients to 16 (64 for ``df``, whose closed form is a small
    difference of large pieces on the direct side of the switch).  Where the true ``d scale`` is 0
    (``|z| = 1``) the floor is ``1 / scale``, as the gradient is a sum of two terms of that size.
    """
    for _, x, loc, scale, want_lp, want_ddf, want_dx, want_dscale in (
        r for r in _LOG_PROB if r[0] == df
    ):
        lp, ddf, dx, dloc, dscale = _gradients(df, x, loc, scale, device, dtype)
        where = f"df={df} x={x} loc={loc} scale={scale}"
        assert _within(lp, want_lp, dtype, 8, 1.0), (where, "log_prob", lp, want_lp)
        assert _within(ddf, want_ddf, dtype, 64), (where, "d df", ddf, want_ddf)
        assert _within(dx, want_dx, dtype, 16), (where, "d x", dx, want_dx)
        assert _within(dloc, -want_dx, dtype, 16), (where, "d loc", dloc, -want_dx)
        assert _within(dscale, want_dscale, dtype, 16, 1.0 / scale), (
            where,
            "d scale",
            dscale,
            want_dscale,
        )


@pytest.mark.parametrize(("device", "dtype"), device_dtype_params([lucid.float32]))
@pytest.mark.parametrize("df", [1e6, 1e8])
def test_float32_repro_of_the_card(df: float, device: str, dtype: lucid.dtype) -> None:
    """The reported points: ``StudentT(1e6)`` and ``StudentT(1e8)`` at 1.0 were -1.457 and -9.783."""
    got = D.StudentT(_t(df, device, dtype)).log_prob(_t(1.0, device, dtype)).item()
    want = next(r[4] for r in _LOG_PROB if r[:4] == (df, 1.0, 0.0, 1.0))
    assert _within(got, want, dtype, 8, 1.0), (got, want)
    assert abs(got - -1.4189385) < 1e-6, got  # the Normal's, to 1e-6


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("df", [1e12, 1e20, 2.0**100])
def test_the_normal_limit(df: float, device: str, dtype: lucid.dtype) -> None:
    """For a huge ``df`` the density is the Normal's, and so is each gradient but ``df``'s."""
    tol = 2e-6 if dtype == lucid.float32 else 2e-9
    for x, loc, scale in _POINTS:
        lp, ddf, dx, dloc, dscale = _gradients(df, x, loc, scale, device, dtype)
        n_lp, n_dx, n_dloc, n_dscale = _normal_gradients(x, loc, scale, device, dtype)
        assert abs(lp - n_lp) < tol, (x, loc, scale, lp, n_lp)
        assert abs(dx - n_dx) < tol, (x, loc, scale, dx, n_dx)
        assert abs(dloc - n_dloc) < tol, (x, loc, scale, dloc, n_dloc)
        assert abs(dscale - n_dscale) < tol, (x, loc, scale, dscale, n_dscale)
        assert abs(ddf) < 1e-10, ddf


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_infinite_df_is_the_normal(device: str, dtype: lucid.dtype) -> None:
    """``df = +inf`` is the Normal log-density, with its gradients, and a zero gradient in ``df``.

    A deliberate departure from the reference, which answers NaN there: its ``lgamma(inf) -
    lgamma(inf)`` is an artefact of the formula, not a defined behaviour, and the limit is the
    Normal.  It is the value every finite ``df`` above 1e12 already reaches.
    """
    tol = 2e-6 if dtype == lucid.float32 else 2e-15
    for x, loc, scale in _POINTS:
        lp, ddf, dx, dloc, dscale = _gradients(math.inf, x, loc, scale, device, dtype)
        n_lp, n_dx, n_dloc, n_dscale = _normal_gradients(x, loc, scale, device, dtype)
        assert abs(lp - n_lp) < tol, (x, loc, scale, lp, n_lp)
        assert abs(dx - n_dx) < tol, (x, loc, scale, dx, n_dx)
        assert abs(dloc - n_dloc) < tol, (x, loc, scale, dloc, n_dloc)
        assert abs(dscale - n_dscale) < tol, (x, loc, scale, dscale, n_dscale)
        assert ddf == 0.0, ddf


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("df", [0.0, -1.0, math.nan])
def test_df_outside_its_domain_is_nan_with_nan_gradients(
    df: float, device: str, dtype: lucid.dtype
) -> None:
    """``df <= 0`` and NaN are NaN in the value and in every gradient (without validation, which
    would raise first).  Masking only the output left the gradients 0; the old code had NaN."""
    for x in (0.0, 1.0):
        lp, _, dx, dloc, dscale = _gradients(df, x, 0.0, 1.0, device, dtype)
        assert all(math.isnan(v) for v in (lp, dx, dloc, dscale)), (df, x, lp, dx, dloc, dscale)
    entropy = D.StudentT(_t(df, device, dtype), validate_args=False).entropy()
    assert math.isnan(entropy.item()), df


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_a_bad_lane_does_not_poison_the_others(device: str, dtype: lucid.dtype) -> None:
    """Elementwise: lanes with ``df`` of -1 and 0 are NaN, the lane with ``df = 3`` is as alone."""
    df = lucid.tensor([-1.0, 3.0, 0.0], dtype=dtype, device=device)
    x = lucid.tensor([0.5, 0.5, 0.5], dtype=dtype, device=device, requires_grad=True)
    out = D.StudentT(df, validate_args=False).log_prob(x)
    out.sum().backward()
    assert x.grad is not None
    alone_lp, _, alone_dx, _, _ = _gradients(3.0, 0.5, 0.0, 1.0, device, dtype)
    got = [v.item() for v in out]
    grad = [v.item() for v in x.grad]
    assert math.isnan(got[0]) and math.isnan(got[2]) and math.isnan(grad[0]) and math.isnan(grad[2])
    assert got[1] == alone_lp and grad[1] == alone_dx


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_a_batch_mixes_every_regime_in_one_call(device: str, dtype: lucid.dtype) -> None:
    """One tensor of ``df`` across the shift, series and ``u`` switches, each lane as alone."""
    rows = [r for r in _LOG_PROB if r[1:4] == (3.0, 1.0, 0.5)]
    df = lucid.tensor([r[0] for r in rows], dtype=dtype, device=device)
    got = D.StudentT(df, _t(1.0, device, dtype), _t(0.5, device, dtype)).log_prob(
        _t(3.0, device, dtype)
    )
    for lane, row in enumerate(rows):
        assert _within(got[lane].item(), row[4], dtype, 8, 1.0), (row[0], got[lane].item(), row[4])


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("df", "scale", "want", "want_ddf"), _ENTROPY)
def test_entropy_and_its_gradients_match_the_closed_form(
    df: float, scale: float, want: float, want_ddf: float, device: str, dtype: lucid.dtype
) -> None:
    """The entropy over the same ``df`` range: it was 9.21 for 1.4189 at ``df = 1e8`` in float32.

    Its slope in ``df`` is ``-1/df^2`` for a large ``df``, the remainder of ``1/(2a) - 1/(2a)``
    in the digamma form: float32 was 1e-3 off at ``df = 1e3`` and 4 times off from ``1e7``,
    float64 6e-5 off at ``1e12``.
    """
    df_t, scale_t = _t(df, device, dtype, True), _t(scale, device, dtype, True)
    out = D.StudentT(df_t, _t(0.0, device, dtype), scale_t).entropy()
    out.backward()
    assert df_t.grad is not None and scale_t.grad is not None
    assert _within(out.item(), want, dtype, 8), (df, scale, out.item(), want)
    assert _within(df_t.grad.item(), want_ddf, dtype, 24), (df, df_t.grad.item(), want_ddf)
    assert _within(scale_t.grad.item(), 1.0 / scale, dtype, 4), (scale, scale_t.grad.item())


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_infinite_df_entropy_is_the_normal(device: str, dtype: lucid.dtype) -> None:
    """``df = +inf``: ``log(scale) + log(2 pi e)/2``, a zero gradient in ``df``, 1/scale in ``scale``.

    A deliberate departure from the reference, which answers NaN: its ``(df+1)/2 *
    (digamma(inf) - digamma(inf))`` is ``inf * 0``, an artefact of the formula; the limit is the
    Normal's entropy, as for ``log_prob``.
    """
    for scale in (1.0, 2.0):
        df_t, scale_t = _t(math.inf, device, dtype, True), _t(scale, device, dtype, True)
        out = D.StudentT(df_t, _t(0.0, device, dtype), scale_t).entropy()
        out.backward()
        want = math.log(scale) + 0.5 * math.log(2.0 * math.pi * math.e)
        assert _within(out.item(), want, dtype, 4), (scale, out.item(), want)
        assert df_t.grad is not None and df_t.grad.item() == 0.0
        assert scale_t.grad is not None and _within(scale_t.grad.item(), 1.0 / scale, dtype, 4)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("a", "excess", "gap"), _GAP)
def test_the_gamma_ratio_helpers(
    a: float, excess: float, gap: float, device: str, dtype: lucid.dtype
) -> None:
    """``lgamma(a + 1/2) - lgamma(a) - log(a)/2`` and ``digamma(a + 1/2) - digamma(a)``, to the ulp.

    Relative to their own value, which is as small as ``1e-32`` here: that is the point of them.
    """
    a_t = _t(a, device, dtype)
    assert _within(_gamma_half_excess(a_t).item(), excess, dtype, 4), (a, excess)
    assert _within(_digamma_half_gap(a_t).item(), gap, dtype, 4), (a, gap)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("df", "x", "d_df2", "d_df_dx", "d_x2"), _HESSIAN)
def test_second_derivatives(
    df: float, x: float, d_df2: float, d_df_dx: float, d_x2: float, device: str, dtype: lucid.dtype
) -> None:
    """The graph is differentiable twice (the helpers are built from ops, not a custom backward)."""
    df_t, x_t = _t(df, device, dtype, True), _t(x, device, dtype, True)
    out = D.StudentT(df_t, validate_args=False).log_prob(x_t)
    g_df, g_x = lucid.autograd.grad(out, [df_t, x_t], create_graph=True)
    h_df_df, h_df_x = lucid.autograd.grad(g_df, [df_t, x_t], retain_graph=True)
    (h_x_x,) = lucid.autograd.grad(g_x, [x_t])
    k = 256
    assert _within(h_df_df.item(), d_df2, dtype, k), (df, x, h_df_df.item(), d_df2)
    assert _within(h_df_x.item(), d_df_dx, dtype, k), (df, x, h_df_x.item(), d_df_dx)
    assert _within(h_x_x.item(), d_x2, dtype, k), (df, x, h_x_x.item(), d_x2)


# ── against the reference, where it is right (df below ~1e4, float64) ────────


@pytest.mark.parity
@pytest.mark.parametrize("df", [0.5, 3.0, 30.0, 1000.0])
def test_parity_log_prob_gradient_and_entropy(ref: Any, df: float) -> None:
    xs = np.array([-4.0, -1.0, 0.0, 0.3, 4.0])
    f64 = lucid.float64
    mine_df = lucid.tensor(df, dtype=f64, requires_grad=True)
    mine = D.StudentT(mine_df, lucid.tensor(0.2, dtype=f64), lucid.tensor(1.5, dtype=f64))
    mine_lp = mine.log_prob(lucid.tensor(xs.copy(), dtype=f64))
    mine_lp.sum().backward()

    ref_df = ref.tensor(df, dtype=ref.float64, requires_grad=True)
    theirs = ref.distributions.StudentT(
        ref_df, ref.tensor(0.2, dtype=ref.float64), ref.tensor(1.5, dtype=ref.float64)
    )
    ref_lp = theirs.log_prob(ref.tensor(xs.copy(), dtype=ref.float64))
    ref_lp.sum().backward()

    assert mine_df.grad is not None and ref_df.grad is not None
    np.testing.assert_allclose(
        mine_lp.detach().numpy(), ref_lp.detach().numpy(), rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(mine_df.grad.numpy(), ref_df.grad.numpy(), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(
        mine.entropy().detach().numpy(), theirs.entropy().detach().numpy(), rtol=1e-10, atol=1e-10
    )


@pytest.mark.parity
@pytest.mark.parametrize("df", [0.5, 3.0, 30.0])
def test_parity_entropy_gradient_in_df(ref: Any, df: float) -> None:
    mine_df = lucid.tensor(df, dtype=lucid.float64, requires_grad=True)
    D.StudentT(mine_df).entropy().backward()
    ref_df = ref.tensor(df, dtype=ref.float64, requires_grad=True)
    ref.distributions.StudentT(ref_df).entropy().backward()
    assert mine_df.grad is not None and ref_df.grad is not None
    # The reference's own trigamma is good to ~1e-9 here (the closed-form table above is exact).
    np.testing.assert_allclose(mine_df.grad.numpy(), ref_df.grad.numpy(), rtol=1e-7, atol=1e-12)
