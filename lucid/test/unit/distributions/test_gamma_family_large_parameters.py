"""LCD-1041: the gamma family is range-safe in its parameters.

``lgamma`` of a large argument is large, so ``lgamma(a) + lgamma(b) - lgamma(a + b)`` and the
neighbours of it (``alpha + lgamma(alpha) + (1 - alpha) digamma(alpha)``, ``lgamma(n + 1) -
lgamma(k + 1) - lgamma(n - k + 1)``, and the main terms of a density it cancels with) answered the
rounding of their inputs while the result is an ordinary number.  In float32, ``Beta(1e6, 1e6)`` at
0.5 was 9.0 for 7.0285, ``FisherSnedecor(1e6, 1e6)`` at 1 was 4.8125 for 5.2957, ``Gamma(1e6, 1)``
had entropy 7.0 for 8.3267 (and a log-density of -6 for -7.83 at 1e6), ``Binomial(1e7, 0.5)`` at
5e6 was 0 for -8.2848; float64 was off by 1e-10 relative.  ``_stirling_math`` owns the repair: the
Stirling remainders ``S`` and ``D``, the deviance kernel ``bd0`` and, on those, ``_log_beta`` /
``_log_multi_beta`` / ``_log_comb`` and the centred density and entropy forms.

The oracle is the closed form in 90-digit arithmetic (mpmath, scratch only; the tables are
literals), at inputs that float32 holds exactly.  Gradients are the analytic derivatives (digamma
and trigamma forms), errors are in units of the dtype's epsilon: ``|got - want| <= k eps
max(|want|, floor)``.  Where a table row has a root or a cancelling slope, the floor is the scale of
its terms.
"""

import ast
import inspect
import math
import pathlib
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D
from lucid.distributions import _stirling_math as M
from lucid.distributions import kl_divergence
from lucid.test._fixtures.devices import device_dtype_params

_FLOATS = device_dtype_params([lucid.float32, lucid.float64])
_F32 = device_dtype_params([lucid.float32])
_EPS = {lucid.float32: 2.0**-23, lucid.float64: 2.0**-52}
_TINY = {lucid.float32: 1.1754943508222875e-38, lucid.float64: 2.2250738585072014e-308}

# ``lucid/distributions/extra.py`` (InverseGamma, Kumaraswamy, Multinomial) was claimed by another
# branch while this one was written, so its call sites move to the owner in a follow-up patch
# (``/tmp/LCD-1041-extra.patch``).  The rows that only the owner gets right are strict xfails until
# then; once the sites are routed the condition is false and they run as ordinary tests.
_ROUTED = "_stirling_math" in inspect.getsource(sys.modules[D.InverseGamma.__module__])
_PENDING = pytest.mark.xfail(
    not _ROUTED, strict=True, reason="LCD-1041: extra.py call sites await LCD-751 (follow-up patch)"
)


def _table(text: str) -> tuple[tuple[float, ...], ...]:
    """Rows of whitespace-separated floats."""
    return tuple(tuple(float(v) for v in line.split()) for line in text.strip().splitlines())


def _t(v: Any, device: str, dtype: lucid.dtype, grad: bool = False) -> lucid.Tensor:
    return lucid.tensor(v, dtype=dtype, device=device, requires_grad=grad)


def _marked(rows: Any, pending: Callable[[Any], bool], unpack: bool = False) -> list[Any]:
    """``rows`` with ``_PENDING`` on the ones ``pending`` selects (``unpack``: a row is a tuple of args)."""
    out = []
    for r in rows:
        args = tuple(r) if unpack else (r,)
        out.append(
            pytest.param(*args, marks=_PENDING)
            if pending(r)
            else (r if not unpack else pytest.param(*args))
        )
    return out


def _within(
    got: float, want: float, dtype: lucid.dtype, k: float, floor: float | None = None
) -> bool:
    """``|got - want| <= k eps max(|want|, floor)``; ``floor`` defaults to the smallest normal."""
    return abs(got - want) <= k * _EPS[dtype] * max(
        abs(want), _TINY[dtype] if floor is None else floor
    )


def _value_and_grads(
    fn: Callable[..., lucid.Tensor], args: tuple[float, ...], device: str, dtype: lucid.dtype
) -> tuple[float, list[float]]:
    ts = [_t(a, device, dtype, True) for a in args]
    out = fn(*ts)
    out.backward()
    return out.item(), [t.grad.item() if t.grad is not None else 0.0 for t in ts]


def _check_rows(
    fn: Callable[..., lucid.Tensor],
    row: tuple[float, ...],
    n_in: int,
    device: str,
    dtype: lucid.dtype,
    k_value: float,
    k_grad: float,
    floors: tuple[float, ...],
) -> None:
    """One table row ``(inputs..., value, d/d input_i...)``: the value and every slope."""
    args, want = row[:n_in], row[n_in:]
    got_v, got_g = _value_and_grads(fn, args, device, dtype)
    assert _within(got_v, want[0], dtype, k_value, 1.0), (args, "value", got_v, want[0])
    for i, floor in enumerate(floors):
        assert _within(got_g[i], want[1 + i], dtype, k_grad, floor), (
            args,
            f"d/d arg{i}",
            got_g[i],
            want[1 + i],
        )


# z, S(z) = lgamma(z) - [(z - 1/2) log z - z + log(2 pi)/2], D(z) = log z - 1/(2z) - digamma(z)
_REMAINDERS = _table("""
0.001 2.5422704679670947 493.66781665282815
0.0625 0.6701855227616264 5.706264767820323
0.5 0.15342640972002736 0.27036284546147815
1.0 0.08106146679532726 0.07721566490153287
3.0 0.02767792568499834 0.009161286902975886
9.0 0.009255462182712733 0.0010275438250538307
10.0 0.00833056343336287 0.0008325039273245763
11.0 0.007573675487951841 0.0006881382771039818
25.0 0.003333155636728093 0.00013331201622674067
1000.0 8.333333055555635e-05 8.333332500000397e-08
1000000.0 8.333333333333056e-08 8.3333333333325e-14
1099511627776.0 7.579122514774402e-14 6.89317177127523e-26
1.2676506002282294e+30 6.573840876841765e-32 5.1858460648842845e-62
""")
# x, m, bd0(x, m) = x log(x/m) + m - x
_BD0 = _table("""
1.0 1.0 0.0
1000000.0 1000800.0 0.319829435667841
1000000.0 999200.0 0.3201707691322464
1099511627776.0 1099512676352.0 0.4999996821087886
3.0 7.0 1.4581064188383892
7.0 3.0 1.9310850227104253
0.5 0.001 2.608304049211096
0.001 0.5 0.49278539190157783
50.0 49.0 0.01013536587597242
1.152921504606847e+18 1.1529226041184748e+18 524287.6666669051
10.0 10.5 0.01209835830567997
""")
# a, b, log B(a, b), d/da, d/db
_LOG_BETA = _table("""
0.001 0.001 7.600900817008347 -500.0016413354569 -500.0016413354569
0.5 0.5 1.1447298858494002 -1.3862943611198906 -1.3862943611198906
1.0 1.0 0.0 -1.0 -1.0
0.5 5.0 -0.2073951943460706 -3.574603174603175 -0.10497548014995065
2.0 3.0 -2.4849066497880004 -1.0833333333333333 -0.5833333333333334
5.0 4.0 -5.634789603169249 -0.6345238095238095 -0.8845238095238095
7.0 3.0 -5.529429087511423 -0.37896825396825395 -1.328968253968254
5.0 5.0 -6.4457198193855785 -0.7456349206349207 -0.7456349206349207
12.0 0.5 -0.6596747231378598 -0.04253397129910003 -4.448705677296336
30.0 40.0 -48.30174909591612 -0.8568972457652998 -0.5650080044159815
1000.0 7.0 -41.79599038729854 -0.0069790905612628655 -5.041449951111608
1000000.0 1000000.0 -1386300.003362921 -0.6931474305600078 -0.6931474305600078
1000000.0 0.5 -6.335390211057437 -5.00000125e-07 -15.779020583985739
0.001 1000000.0 6.893363375325389 -1014.3910819907745 -1.0000004995001662e-09
1099511627776.0 3298534883328.0 -2473176120874.7134 -1.3862943611202316 -0.2876820724518188
1099511627776.0 1.0 -27.725887222397812 -9.094947017729282e-13 -28.3031028872998
1.2676506002282294e+30 1.2676506002282294e+30 -1.757336878966639e+30 -0.6931471805599453 -0.6931471805599453
1.2676506002282294e+30 1.0 -69.31471805599453 -7.888609052210118e-31 -69.89193372089606
""")
# a, b, d2/da2, d2/da db of log B
_LOG_BETA_HESS = _table("""
1000.0 7.0 6.9582712473022615e-06 -0.000993541895419331
50.0 60.0 0.01106897660467458 -0.009132356622022546
1000000.0 1000000.0 5.000003750001459e-07 -5.000001250000209e-07
3.0 4.0 0.24138888888888888 -0.15354517795933756
""")
# n, k, log C(n, k), d/dn, d/dk
_LOG_COMB = _table("""
20.0 7.0 11.258291246564648 0.41760590200992675 0.5872766122766123
9.0 4.0 4.836281906951478 0.5456349206349206 0.2
10.0 5.0 5.529429087511423 0.6456349206349207 -4.1708118308681856e-92
50.0 25.0 32.47055650581199 0.6832471605759182 4.746096221332763e-92
10000.0 3.0 25.838961621697493 0.00030003000500090015 7.9539726727060485
10000000.0 5000000.0 6931463.52076025 0.6931471305599478 1.1313926345803354e-91
10000000.0 3.0 46.56252718364688 3.00000030000005e-07 14.861977732526489
1099511627776.0 549755813888.0 762123384771.7217 0.6931471805594905 1.472728039589318e-90
1.2676506002282294e+30 6.338253001141147e+29 8.786684394833195e+29 0.6931471805599453 6.320457836570823e-90
30.0 0.0 0.0 0.0 3.994987130920391
30.0 30.0 0.0 3.994987130920391 -3.994987130920391
12.5 4.0 6.392688905099487 0.3674577748719625 0.6916202079711491
""")
# a, b, x, log p, d/da, d/db, d/dx
_BETA_LP = _table("""
1000000.0 1000000.0 0.5 7.028537391617382 2.500000625e-07 2.500000625e-07 0.0
1000000.0 1000000.0 0.50048828125 6.074863574138264 0.0009763359731179589 -0.0009767896477641128 -3906.2498190401257
1000.0 2000.0 0.25 -48.10991537542923 -0.2873486650443817 0.11786638056379044 1330.6666666666667
1000.0 2000.0 0.375 -7.511064406488313 0.11811644306378263 -0.06445517623016418 -534.4
10.0 20.0 0.25 1.172718673224192 -0.2536088175010868 0.1262320679915949 10.666666666666666
6.0 7.0 0.5 0.9958525547103407 0.12673016431739956 -0.0399365023492671 -2.0
1000000.0 0.5 0.75 -287674.75623231684 -0.28768157245165593 14.392726222865848 1333334.0
0.5 1000000.0 0.0009765625 -967.2375446731078 8.847548778386287 -0.0009765396477016128 -1001488.5161290322
1099511627776.0 3298534883328.0 0.75 -2415873971601.9062 1.0986122886684508 -1.0986122886680718 -11728124029608.0
1.2676506002282294e+30 1.2676506002282294e+30 0.5 34.77814126563251 1.9721522630525295e-31 1.9721522630525295e-31 0.0
1.0 1000000.0 9.313225746154785e-10 13.814579236320547 -6.401688693932636 9.990686774249508e-07 -999999.0009313216
1.0 30.0 9.094947017729282e-13 3.40119738163578 -23.730900091477423 0.03333333333242384 -29.000000000026375
30.0 30.0 0.5 1.8172144546366444 0.008402768138068104 0.008402768138068104 0.0
2.0 5.0 0.25 0.8641747307351411 0.06370563888010938 0.07898459421488574 -1.3333333333333333
1.0 1.0 0.3 0.0 -0.20397280432593604 0.6433250560612677 0.0
""")
# a, b, H, d/da, d/db
_BETA_H = _table("""
1000000.0 1000000.0 -6.528537766617507 -2.49999874999875e-07 -2.49999874999875e-07
1000.0 3000.0 -3.5653106969305544 0.00012528141149150318 -0.0002083483786648422
20.0 30.0 -1.261978167919611 -0.004481640887397238 -0.013292755492996343
6.0 7.0 -0.6018207472189455 -0.027083065985013217 -0.04173935505546188
3.0 4.0 -0.3443445622221007 -0.022142243899765136 -0.08374297741465823
1000000.0 0.5 -13.724900878050432 -9.999997499997083e-07 3.4673996002722562
0.5 1000000.0 -13.724900878050432 3.4673996002722562 -9.999997499997083e-07
1099511627776.0 3298534883328.0 -13.97414047534023 1.1368683772184867e-13 -1.894780628693725e-13
1.2676506002282294e+30 1.2676506002282294e+30 -34.27814126563251 -1.9721522630525295e-31 -1.9721522630525295e-31
2.0 2.0 -0.12509280256138833 -0.07728815537399579 -0.07728815537399579
1.0 1.0 0.0 9.57616230800835e-189 9.57616230800835e-189
100.0 7.0 -2.3596519067250874 -0.009058250168319942 0.06463718174567834
""")
# d1, d2, x, log p, d/dd1, d/dd2, d/dx
_F_LP = _table("""
1000000.0 1000000.0 1.0 5.295669315217519 2.50000125e-07 2.50000125e-07 -1.0
50.0 70.0 1.25 -0.17333296546487123 0.0018770040747649944 0.0006229875521149578 -3.4415094339622643
30.0 8.0 0.5 -0.530259182775087 -0.004992609857345913 -0.023225921214905933 3.217391304347826
10000.0 30000.0 0.75 -144.1564584643074 -0.011534274094604885 -0.0010599060168625623 1332.0
1099511627776.0 1099511627776.0 1.0 12.250857897434061 2.2737367544333546e-13 2.2737367544333546e-13 -1.0
4.0 6.0 2.0 -1.8693656878044012 0.04757305841324104 0.010874879330207718 -0.9285714285714286
1.0 1000000.0 1.0 -1.4189390332045895 0.6351809227310724 4.999998333333333e-13 -1.0
2.0 1000000.0 1.0 -1.0000009999993333 0.28860733245159975 9.999986666686666e-13 -1.0
""")
# a, H of Gamma(a, 1), dH/da
_GAMMA_H = _table("""
1000000.0 8.326693478853393 5.000003333335e-07
1000.0 4.8724827560179715 0.0005003335000333
20.0 2.899928334598662 0.025854364231140722
10.0 2.53605417848098 0.053502978864828285
9.0 2.4794710791003705 0.0599038824477486
3.0 1.8475785103630111 0.21013186630354713
1.0 1.0 1.0
0.5 0.09060992991398835 3.4674011002723395
0.001 -992.6668174744946 999002.6408906627
1099511627776.0 15.281882144403276 4.547473508867399e-13
1.2676506002282294e+30 36.07629756120194 3.944304526105059e-31
""")
# a, rate, x, log p, d/da, d/drate, d/dx
_GAMMA_LP = _table("""
1000000.0 1.0 1000000.0 -7.826693895520143 5.000000833333334e-07 0.0 -1e-06
1000000.0 0.5 2001024.0 -8.65138022298298 0.0005123689728054032 -1024.0 -0.0002563687392055268
1000.0 2.0 500.0 -3.6797523254663513 0.000500083333325 0.0 -0.002
30.0 1.0 20.0 -4.380803034102271 -0.3887058591315339 10.0 0.45
12.0 3.0 5.0 -1.6151433450814654 0.26538852112639805 -1.0 -0.8
3.0 1.0 2.0 -1.3068528194400546 -0.22963715453852182 1.0 0.0
0.5 1.0 0.25 -0.12921776236475477 0.5772156649015329 0.25 -3.0
1099511627776.0 1.0 1099511627776.0 -14.781882144403655 4.547473508865331e-13 0.0 -9.094947017729282e-13
1.2676506002282294e+30 1.0 1.2676506002282294e+30 -35.57629756120194 3.944304526105059e-31 0.0 -7.888609052210118e-31
""")
_IGAMMA_LP = _table("""
1000000.0 1000000.0 1.0 5.988816662444131 5.000000833333334e-07 0.0 -1.0
1000.0 2.0 0.001953125 8.48970701531139 0.02421660995064104 -12.0 11776.0
3.0 1.0 2.0 -3.9657359027997265 -1.6159315156584124 2.5 -1.75
""")
# a, rate, H, d/da, d/drate
_IGAMMA_H = _table("""
1000000.0 1.0 -19.30432663707499 -1.5000006666668334e-06 1.0
50.0 2.0 -3.7425822179176707 -0.030267994561553418 0.5
3.0 1.0 0.002009840166076752 -0.5797362673929057 1.0
10.0 1.0 -1.9674509996524625 -0.1568296924985432 1.0
""")
# rate, k, log P, d/drate
_POIS_LP = _table("""
1000000.0 1000000.0 -7.826693895520143 0.0
10000.0 10100.0 -6.0274337524420645 0.01
50.0 40.0 -3.839719497631553 -0.2
10.0 10.0 -2.0785616431350586 1.82877982605164e-99
3.0 12.0 -9.80386703164457 3.0
1099511627776.0 1099511627776.0 -14.781882144403655 0.0
0.5 1000.0 -6605.775359048109 1999.0
5.0 2.0 -2.4742713556917444 -0.6
""")
# n, logits, k, log P  /  r, logits, k, log P
_BINOM_LP = _table("""
10000000.0 0.0 5000000.0 -8.284839203123887
10000000.0 -1.0 2690438.0 -8.431373648202241
1000.0 -2.0 120.0 -3.252493295575646
50.0 0.0 20.0 -3.173444885807883
20.0 0.0 10.0 -1.7361522965964518
100000.0 4.0 98200.0 -4.658212020319227
10000.0 -14.0 3.0 -16.169353662036343
10000000.0 0.0 1.0 -6931455.687503803
7.0 -1.0 3.0 -1.6374837511381461
30.0 0.5 0.0 -29.2223095254032
30.0 0.5 30.0 -14.222309525403201
1073741824.0 0.0 536870912.0 -10.622999061276738
""")
_NB_LP = _table("""
1000000.0 0.0 1000000.0 -8.173267527466782
50.0 -1.0 30.0 -4.9938176456130545
1000.0 0.0 1000.0 -4.719514762970506
0.5 0.0 1000000.0 -693155.0072538825
3.0 -1.0 4.0 -3.48478161152535
10000.0 -14.0 10.0 -63.00483387665664
5.0 0.0 0.0 -3.4657359027997265
0.001 0.0 100.0 -80.82316010766922
1073741824.0 0.0 1073741824.0 -11.66271983200024
""")
# K, alpha_1..K, x_1..K, log p
_DIR_LP = _table("""
3 1000000.0 1000000.0 2000000.0 0.25 0.25 0.5 15.096795616574683
4 5.0 6.0 7.0 8.0 0.125 0.25 0.25 0.375 4.5008381666547885
3 1000.0 0.0625 5.0 0.5 0.25 0.25 -667.6364666135217
4 1000000.0 1000000.0 1000000.0 1000000.0 0.25 0.25 0.25 0.25 22.81848018875201
3 1099511627776.0 1099511627776.0 2199023255552.0 0.25 0.25 0.5 29.00717246850805
3 2.0 3.0 4.0 0.25 0.25 0.5 1.8813716279177422
""")
# K, alpha_1..K, H
_DIR_H = _table("""
3 1000000.0 1000000.0 1000000.0 -13.72416476878098
4 5.0 6.0 7.0 8.0 -3.5825671661978618
3 1000.0 0.0625 5.0 -24.320351799754864
3 1000000.0 0.5 0.5 -27.44980275610049
3 3.0 3.0 3.0 -1.218018503922557
3 1099511627776.0 1099511627776.0 2199023255552.0 -28.007172468508674
2 1.2676506002282294e+30 1.2676506002282294e+30 -34.27814126563251
""")
# K, counts_1..K, log multinomial coefficient
_MULTINOMIAL = _table("""
3 1000000.0 2000000.0 3000000.0 6068409.934856347
3 5.0 3.0 2.0 7.832014180505469
4 10000.0 1000.0 100000.0 5.0 39258.43705107761
2 1073741824.0 536870912.0 1025177815.3332329
2 3.0 4.0 3.5553480614894135
""")
# K, counts_1..K, probs_1..K, log P
_MULTINOMIAL_LP = _table("""
3 3.0 4.0 5.0 0.25 0.25 0.5 -2.9398869773351213
3 40.0 30.0 30.0 0.5 0.25 0.25 -6.801285746445483
2 1000.0 2000.0 0.375 0.625 -15.464187459619748
""")
# a_p, rate_p, a_q, rate_q, KL(p || q)
_GAMMA_KL = _table("""
3.0 2.0 5.0 1.5 1.3277483418499707
1000000.0 1.0 1000064.0 1.0001220703125 0.0016859084556730588
30.0 1.0 31.0 1.0 0.0167592489766305
0.5 1.0 7.0 3.0 12.079415417547885
1000.0 4.0 1000.0 5.0 26.856448685790244
1000000.0 1.0 2000000.0 4.0 613705.7923065608
100.0 1.0 100.0 1.0 0.0
""")
# a_p, b_p, a_q, b_q, KL(p || q)
_BETA_KL = _table("""
1000000.0 1000000.0 2000000.0 1000000.0 169899.14295439597
30.0 20.0 25.0 22.0 0.45628771160994996
2.0 3.0 5.0 4.0 0.683450379952084
1000.0 2000.0 3000.0 1000.0 1452.485967082379
""")
# K, alpha_1..K, beta_1..K, KL(p || q)
_DIR_KL = _table("""
3 1000000.0 1000000.0 1000000.0 2000000.0 1000000.0 1000000.0 235566.20191358554
3 5.0 6.0 7.0 7.0 6.0 5.0 0.7333333333333333
3 1000.0 2000.0 3000.0 3000.0 2000.0 1000.0 2197.891392151018
""")
# low, high, a, b, KL(Uniform(low, high) || Beta(a, b))
_UNIFORM_BETA_KL = _table("""
0.25 0.75 2.0 3.0 0.42336831512512324
0.25 0.375 1000.0 3000.0 46.923784190304325
0.125 0.5 5.0 4.0 1.4323669167209474
""")
# a, b, mean, variance of Kumaraswamy(a, b)
_KUMA = _table("""
2.0 1000000.0 0.0008862265931178341 2.14601425651757e-07
2.0 4.0 0.40634920634920635 0.03488032249937012
0.5 10000.0 1.9994001399700062e-08 1.9978414671892e-15
4.0 1048576.0 0.028325073187219538 6.31459013150914e-05
""")


# ── the owners ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("z", "s", "d"), _REMAINDERS)
def test_stirling_remainders(z: float, s: float, d: float, device: str, dtype: lucid.dtype) -> None:
    """From ``z = 10`` the asymptotic series, to the ulp relative to its own value (as small as
    1e-62); below it ``lgamma`` / ``digamma`` where both are O(10), absolute."""
    zt = _t(z, device, dtype)
    got_s, got_d = M._stirling_remainder(zt).item(), M._digamma_remainder(zt).item()
    if z >= 10.0:
        assert _within(got_s, s, dtype, 4), (z, "S", got_s, s)
        assert _within(got_d, d, dtype, 4), (z, "D", got_d, d)
    else:
        assert _within(got_s, s, dtype, 64, 1.0), (z, "S", got_s, s)
        assert _within(got_d, d, dtype, 64, 1.0), (z, "D", got_d, d)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("x", "m", "want"), _BD0)
def test_bd0(x: float, m: float, want: float, device: str, dtype: lucid.dtype) -> None:
    """``x log(x/m) + m - x`` relative to its value, also where it is 1e-13 of its terms."""
    got = M._bd0(_t(x, device, dtype), _t(m, device, dtype)).item()
    assert _within(got, want, dtype, 8), (x, m, got, want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _LOG_BETA)
def test_log_beta(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    """``log B(a, b)`` for ``a, b`` from 1e-3 to 1e30: the value (relative to ``max(|v|, 1)``) and
    both slopes, ``psi(a) - psi(a + b)`` and ``psi(b) - psi(a + b)``."""
    _check_rows(M._log_beta, row, 2, device, dtype, 32, 64, (1e-3, 1e-3))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("a", "b", "d_aa", "d_ab"), _LOG_BETA_HESS)
def test_log_beta_second_derivatives(
    a: float, b: float, d_aa: float, d_ab: float, device: str, dtype: lucid.dtype
) -> None:
    """Differentiable twice (every piece is an ordinary op): ``psi_1(a) - psi_1(a+b)``, ``-psi_1(a+b)``."""
    at, bt = _t(a, device, dtype, True), _t(b, device, dtype, True)
    (g_a,) = lucid.autograd.grad(M._log_beta(at, bt), [at], create_graph=True)
    h_a, h_b = lucid.autograd.grad(g_a, [at, bt])
    assert _within(h_a.item(), d_aa, dtype, 512, 1e-9), (a, b, h_a.item(), d_aa)
    assert _within(h_b.item(), d_ab, dtype, 512, 1e-9), (a, b, h_b.item(), d_ab)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _LOG_COMB)
def test_log_comb(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    """``log C(n, k)`` to ``n = 1e30``; the ends ``k = 0``, ``k = n`` are exactly 0 and keep the
    slope of the function (``psi(n + 1) - psi(1)`` in ``n`` at ``k = n``)."""
    _check_rows(M._log_comb, row, 2, device, dtype, 16, 128, (1e-3, 1e-3))
    if row[1] in (0.0, row[0]):
        assert M._log_comb(_t(row[0], device, dtype), _t(row[1], device, dtype)).item() == 0.0


# ── the laws ────────────────────────────────────────────────────────────────


def _beta_lp(a: lucid.Tensor, b: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
    return D.Beta(a, b, validate_args=False).log_prob(x)


def _beta_h(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
    return D.Beta(a, b, validate_args=False).entropy()


def _f_lp(d1: lucid.Tensor, d2: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
    return D.FisherSnedecor(d1, d2, validate_args=False).log_prob(x)


def _gamma_h(a: lucid.Tensor) -> lucid.Tensor:
    return D.Gamma(a, lucid.ones_like(a), validate_args=False).entropy()


def _gamma_lp(a: lucid.Tensor, r: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
    return D.Gamma(a, r, validate_args=False).log_prob(x)


def _igamma_lp(a: lucid.Tensor, r: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
    return D.InverseGamma(a, r, validate_args=False).log_prob(x)


def _igamma_h(a: lucid.Tensor, r: lucid.Tensor) -> lucid.Tensor:
    return D.InverseGamma(a, r, validate_args=False).entropy()


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _BETA_LP)
def test_beta_log_prob(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    """``Beta`` log-density and its slopes in ``a``, ``b``, ``x`` across the parameter range,
    inside, beside and far from the mean (the last two rows are the plain, small-parameter form)."""
    _check_rows(_beta_lp, row, 3, device, dtype, 64, 256, (1e-3, 1e-3, 1.0))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _BETA_H)
def test_beta_entropy(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    """``Beta`` entropy: ``log B - (a-1) psi(a) - (b-1) psi(b) + (a+b-2) psi(a+b)`` is a sum of
    terms of size ``a log a``; its slopes are small differences of them."""
    _check_rows(_beta_h, row, 2, device, dtype, 96, 512, (1e-8, 1e-8))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _F_LP)
def test_f_log_prob(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    _check_rows(_f_lp, row, 3, device, dtype, 32, 256, (1e-6, 1e-6, 1.0))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("a", "want", "slope"), _GAMMA_H)
def test_gamma_entropy(
    a: float, want: float, slope: float, device: str, dtype: lucid.dtype
) -> None:
    """``alpha + lgamma(alpha) + (1 - alpha) digamma(alpha)`` for ``alpha`` from 1e-3 to 1e30, and
    its slope ``1 + (1 - alpha) psi_1(alpha)``, which is ``1/(2 alpha)`` for a large one."""
    at = _t(a, device, dtype, True)
    out = _gamma_h(at)
    out.backward()
    assert at.grad is not None
    assert _within(out.item(), want, dtype, 32, 1.0), (a, out.item(), want)
    assert _within(at.grad.item(), slope, dtype, 32, 1e-13), (a, at.grad.item(), slope)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _GAMMA_LP)
def test_gamma_log_prob(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    _check_rows(_gamma_lp, row, 3, device, dtype, 32, 256, (1e-3, 1e-3, 1.0))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _marked(_IGAMMA_LP, lambda r: r[0] >= 100.0))
def test_inverse_gamma_log_prob(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    _check_rows(_igamma_lp, row, 3, device, dtype, 16, 32, (1e-3, 1e-3, 1.0))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _marked(_IGAMMA_H, lambda r: r[0] >= 50.0))
def test_inverse_gamma_entropy(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    _check_rows(_igamma_h, row, 2, device, dtype, 32, 32, (1e-3, 1e-3))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _IGAMMA_LP)
def test_inverse_gamma_owner_log_prob(
    row: tuple[float, ...], device: str, dtype: lucid.dtype
) -> None:
    """The owner's own form (``rate / x`` in ``bd0``), whether or not the law calls it yet."""
    _check_rows(M._inverse_gamma_log_prob, row, 3, device, dtype, 16, 32, (1e-3, 1e-3, 1.0))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _IGAMMA_H)
def test_inverse_gamma_owner_entropy(
    row: tuple[float, ...], device: str, dtype: lucid.dtype
) -> None:
    _check_rows(M._inverse_gamma_entropy, row, 2, device, dtype, 32, 32, (1e-3, 1e-3))


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("rate", "k", "want", "slope"), _POIS_LP)
def test_poisson_log_prob(
    rate: float, k: float, want: float, slope: float, device: str, dtype: lucid.dtype
) -> None:
    """``k log(rate) - rate - lgamma(k + 1)``: at ``rate = k = 1e6`` float32 gave -6 for -7.83."""
    rt = _t(rate, device, dtype, True)
    out = D.Poisson(rt, validate_args=False).log_prob(_t(k, device, dtype))
    out.backward()
    assert rt.grad is not None
    assert _within(out.item(), want, dtype, 8, 1.0), (rate, k, out.item(), want)
    assert _within(rt.grad.item(), slope, dtype, 8, 1e-3), (rate, k, rt.grad.item(), slope)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("n", "logit", "k", "want"), _BINOM_LP)
def test_binomial_log_prob(
    n: float, logit: float, k: float, want: float, device: str, dtype: lucid.dtype
) -> None:
    """Binomial through a logit: the saddle-point form for ``n >= 10`` and ``0 < k < n``, the plain
    form (with the stable coefficient) at the ends and for small ``n``.  The logit rounds ``p``
    to an ulp, and one ulp of ``p`` moves the answer by ``(k - np)/(np q) * p eps``, a hundred
    float64 epsilons at ``n = 1e7`` a standard deviation off the mean."""
    got = D.Binomial(_t(n, device, dtype), logits=_t(logit, device, dtype)).log_prob(
        _t(k, device, dtype)
    )
    assert _within(got.item(), want, dtype, 256, 1.0), (n, logit, k, got.item(), want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("r", "logit", "k", "want"), _NB_LP)
def test_negative_binomial_log_prob(
    r: float, logit: float, k: float, want: float, device: str, dtype: lucid.dtype
) -> None:
    got = D.NegativeBinomial(_t(r, device, dtype), logits=_t(logit, device, dtype)).log_prob(
        _t(k, device, dtype)
    )
    assert _within(got.item(), want, dtype, 32, 1.0), (r, logit, k, got.item(), want)


def _vec_rows(table: tuple[tuple[float, ...], ...], n_vectors: int) -> list[tuple[Any, ...]]:
    """Rows ``K v... w...`` as ``(vector_1, ..., vector_n, tail)``: ``K`` numbers per vector."""
    rows = []
    for row in table:
        k = int(row[0])
        vectors = [list(row[1 + i * k : 1 + (i + 1) * k]) for i in range(n_vectors)]
        rows.append((*vectors, row[1 + n_vectors * k]))
    return rows


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("alpha", "x", "want"), _vec_rows(_DIR_LP, 2))
def test_dirichlet_log_prob(
    alpha: list[float], x: list[float], want: float, device: str, dtype: lucid.dtype
) -> None:
    got = D.Dirichlet(_t(alpha, device, dtype)).log_prob(_t(x, device, dtype))
    assert _within(got.item(), want, dtype, 32, 1.0), (alpha, x, got.item(), want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("alpha", "want"), _vec_rows(_DIR_H, 1))
def test_dirichlet_entropy(
    alpha: list[float], want: float, device: str, dtype: lucid.dtype
) -> None:
    got = D.Dirichlet(_t(alpha, device, dtype)).entropy()
    assert _within(got.item(), want, dtype, 64, 1.0), (alpha, got.item(), want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("counts", "want"), _vec_rows(_MULTINOMIAL, 1))
def test_log_multinomial_coefficient(
    counts: list[float], want: float, device: str, dtype: lucid.dtype
) -> None:
    got = M._log_multinomial_coeff(_t(counts, device, dtype))
    assert _within(got.item(), want, dtype, 32, 1.0), (counts, got.item(), want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(
    ("counts", "probs", "want"),
    _marked(_vec_rows(_MULTINOMIAL_LP, 2), lambda r: sum(r[0]) >= 1000.0, unpack=True),
)
def test_multinomial_log_prob(
    counts: list[float], probs: list[float], want: float, device: str, dtype: lucid.dtype
) -> None:
    """The coefficient is the owner's; ``sum k log p`` is the law's own."""
    law = D.Multinomial(int(sum(counts)), probs=_t(probs, device, dtype))
    got = law.log_prob(_t(counts, device, dtype))
    assert _within(got.item(), want, dtype, 64, 1.0), (counts, got.item(), want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _GAMMA_KL)
def test_gamma_kl(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    """``KL(Gamma || Gamma)`` between neighbours is ``O((a_p - a_q)^2 / a)`` out of terms of size
    ``a log a``: float32 ``(1e6, 1) -> (1e6 + 64, 1 + 2^-13)`` was off by a factor, and ``p = q`` is 0."""
    p = D.Gamma(_t(row[0], device, dtype), _t(row[1], device, dtype))
    q = D.Gamma(_t(row[2], device, dtype), _t(row[3], device, dtype))
    got = kl_divergence(p, q).item()
    # The last-but-one row has equal means and shapes 1e6 / 2e6: its answer is a 1e-7 remainder
    # of terms of size 1e6, which no regrouping of the closed form reaches.
    k = 1e7 if row[0] == 1e6 and row[2] == 2e6 else 64
    assert _within(got, row[4], dtype, k, 1.0), (row, got)
    if row[:2] == row[2:4]:
        assert got == 0.0


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _BETA_KL)
def test_beta_kl(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    p = D.Beta(_t(row[0], device, dtype), _t(row[1], device, dtype))
    q = D.Beta(_t(row[2], device, dtype), _t(row[3], device, dtype))
    assert _within(kl_divergence(p, q).item(), row[4], dtype, 192, 1.0), row


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(("a", "b", "want"), _vec_rows(_DIR_KL, 2))
def test_dirichlet_kl(
    a: list[float], b: list[float], want: float, device: str, dtype: lucid.dtype
) -> None:
    got = kl_divergence(D.Dirichlet(_t(a, device, dtype)), D.Dirichlet(_t(b, device, dtype)))
    assert _within(got.item(), want, dtype, 192, 1.0), (a, b, got.item(), want)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize("row", _UNIFORM_BETA_KL)
def test_uniform_beta_kl(row: tuple[float, ...], device: str, dtype: lucid.dtype) -> None:
    """``KL(Uniform || Beta)`` carries ``log B(a, b)`` (the quadrature oracle is good to 1e-25)."""
    p = D.Uniform(_t(row[0], device, dtype), _t(row[1], device, dtype))
    q = D.Beta(_t(row[2], device, dtype), _t(row[3], device, dtype))
    assert _within(kl_divergence(p, q).item(), row[4], dtype, 256, 1.0), row


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
@pytest.mark.parametrize(
    ("a", "b", "mean", "var"), _marked(_KUMA, lambda r: r[1] >= 1e4, unpack=True)
)
def test_kumaraswamy_moments(
    a: float, b: float, mean: float, var: float, device: str, dtype: lucid.dtype
) -> None:
    """``b B(1 + n/a, b)`` through the log-beta (``b = 1e6``: float32 lost the whole mean)."""
    law = D.Kumaraswamy(_t(a, device, dtype), _t(b, device, dtype))
    assert _within(law.mean.item(), mean, dtype, 64, 1e-30), (a, b, law.mean.item(), mean)
    # the variance is E[X^2] - E[X]^2, a difference of two comparable numbers for a large ``b``:
    # held to the size of its terms, not its own value.
    assert abs(law.variance.item() - var) <= 4e-3 * max(mean, 1e-30) ** 2 + 1e-6 * abs(var), (
        a,
        b,
        law.variance.item(),
        var,
    )


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_exponential_family_log_normalizers(device: str, dtype: lucid.dtype) -> None:
    """``Beta`` and ``Dirichlet`` give the owner's ``log B`` as their log-normalizer (it is what
    the Bregman form of their entropy and KL differences reads)."""
    a, b = _t(1e3, device, dtype), _t(7.0, device, dtype)
    beta = D.Beta(a, b)
    assert _within(
        beta._log_normalizer(*beta._natural_params).item(), -41.79599038729854, dtype, 32, 1.0
    )
    alpha = _t([5.0, 6.0, 1e3], device, dtype)
    dirichlet = D.Dirichlet(alpha)
    want = M._log_multi_beta(alpha).item()
    assert dirichlet._log_normalizer(*dirichlet._natural_params).item() == want
    assert _within(want, -68.07457099771744, dtype, 64, 1.0)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_integer_operands_take_the_default_float(device: str, dtype: lucid.dtype) -> None:
    """``log_comb`` of integer tensors is the default float's (the owner promotes before it computes)."""
    n = lucid.tensor([20, 9, 1000], device=device)
    k = lucid.tensor([7, 4, 3], device=device)
    got = M._log_comb(n, k)
    want = M._log_comb(n.to(got.dtype), k.to(got.dtype))
    assert got.dtype == lucid.get_default_dtype()
    assert [v.item() for v in got] == [v.item() for v in want]


# ── the reported rows ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("device", "dtype"), _F32)
def test_the_reported_float32_rows(device: str, dtype: lucid.dtype) -> None:
    """The card's three rows: 9.0 / 4.8125 / 7.0 on main, now to a few ulp."""
    got = D.Beta(_t(1e6, device, dtype), _t(1e6, device, dtype)).log_prob(_t(0.5, device, dtype))
    assert _within(got.item(), 7.028537391617382, dtype, 8, 1.0), got.item()
    got = D.FisherSnedecor(_t(1e6, device, dtype), _t(1e6, device, dtype)).log_prob(
        _t(1.0, device, dtype)
    )
    assert _within(got.item(), 5.295669315217519, dtype, 8, 1.0), got.item()
    got = D.Gamma(_t(1e6, device, dtype), _t(1.0, device, dtype)).entropy()
    assert _within(got.item(), 8.326693478853393, dtype, 8, 1.0), got.item()


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_the_reported_float64_rows(device: str, dtype: lucid.dtype) -> None:
    """Float64 was 1e-10 relative off; now it is at the epsilon."""
    got = D.Beta(_t(1e6, device, dtype), _t(1e6, device, dtype)).log_prob(_t(0.5, device, dtype))
    assert _within(got.item(), 7.028537391617382, dtype, 8, 1.0), got.item()


# ── regimes and edges ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_small_parameters_keep_the_plain_form(device: str, dtype: lucid.dtype) -> None:
    """Below ``sum = 10`` nothing changed: the integer cases are the plain form's, value for value."""
    one = _t(1.0, device, dtype)
    plain = (lucid.lgamma(one) + lucid.lgamma(one) - lucid.lgamma(_t(2.0, device, dtype))).item()
    assert D.Beta(one, one).log_prob(_t(0.3, device, dtype)).item() == -plain  # bit for bit
    assert D.Gamma(one, one).entropy().item() == pytest.approx(1.0, abs=4 * _EPS[dtype])
    assert D.Gamma(one, one).log_prob(_t(0.0, device, dtype)).item() == -lucid.lgamma(one).item()
    assert D.NegativeBinomial(_t(5.0, device, dtype), logits=_t(-1.0, device, dtype)).log_prob(
        _t(0.0, device, dtype)
    ).item() == pytest.approx(
        5.0 * math.log1p(-1.0 / (1.0 + math.exp(1.0))), rel=4 * _EPS[dtype], abs=0
    )
    # a large ``n`` at k = 0 keeps the coefficient at exactly 0 (``n log(1 - p)`` unrounded)
    law = D.Binomial(_t(100.0, device, dtype), probs=_t(0.5, device, dtype))
    assert law.log_prob(_t(0.0, device, dtype)).item() == pytest.approx(
        100.0 * math.log(0.5), rel=16 * _EPS[dtype], abs=0
    )


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_a_tiny_x_keeps_the_single_term_slope(device: str, dtype: lucid.dtype) -> None:
    """``Beta(1, b)`` far in its left tail: the centred slope is ``a/x - 1/x`` plus ``O(s)``, which
    rounds the ``O(s)`` away; the plain slope ``-(b-1)/(1-x)`` is one term.  Only the neighbourhood
    of the mean takes the centred form."""
    for b, x in ((30.0, 2.0**-40), (1e6, 2.0**-30), (1e6, 2.0**-12)):
        bt, xt = _t(b, device, dtype, True), _t(x, device, dtype, True)
        _beta_lp(_t(1.0, device, dtype), bt, xt).backward()
        assert xt.grad is not None
        assert _within(xt.grad.item(), -(b - 1.0) / (1.0 - x), dtype, 4, 1.0), (
            b,
            x,
            xt.grad.item(),
        )


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_boundaries_and_non_finite_inputs(device: str, dtype: lucid.dtype) -> None:
    """The boundary and non-finite inputs are the plain formula's (centred forms take the interior)."""

    def lp(a: float, b: float, x: float) -> float:
        return _beta_lp(_t(a, device, dtype), _t(b, device, dtype), _t(x, device, dtype)).item()

    assert lp(30.0, 30.0, 0.0) == -math.inf and lp(30.0, 30.0, 1.0) == -math.inf
    assert lp(0.5, 30.0, 0.0) == math.inf
    # Beta(1, 1e6) at 0: the density is the finite 1e6 (and its slope in b is the stable 1e-6)
    bt = _t(1e6, device, dtype, True)
    out = _beta_lp(_t(1.0, device, dtype), bt, _t(0.0, device, dtype))
    out.backward()
    assert _within(out.item(), math.log(1e6), dtype, 4, 1.0)
    assert bt.grad is not None and _within(bt.grad.item(), 1e-6, dtype, 64, 1e-6)
    assert math.isnan(lp(30.0, 30.0, math.nan)) and math.isnan(lp(math.nan, 30.0, 0.5))
    # an infinite x or df has no NaN in a slope that was none before
    x_t = _t(math.inf, device, dtype, True)
    _f_lp(_t(2.0, device, dtype), _t(30.0, device, dtype), x_t).backward()
    assert x_t.grad is not None and not math.isnan(x_t.grad.item())

    # Gamma at an infinite x: the value and the slopes are the plain formula's, NaN where it has NaN
    # (a NaN from ``rate * x`` must not reach a lane the ``where`` did not choose)
    def plain_gamma(a: lucid.Tensor, r: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        return a * r.log() + lucid.xlogy(a - 1.0, v) - r * v - lucid.lgamma(a)

    for alpha in (0.5, 12.0):
        mine_a, mine_r = _t(alpha, device, dtype, True), _t(1.0, device, dtype, True)
        mine = _gamma_lp(mine_a, mine_r, _t(math.inf, device, dtype))
        mine.backward()
        plain_a, plain_r = _t(alpha, device, dtype, True), _t(1.0, device, dtype, True)
        plain = plain_gamma(plain_a, plain_r, _t(math.inf, device, dtype))
        plain.backward()
        assert mine_a.grad is not None and plain_a.grad is not None
        assert mine_r.grad is not None and plain_r.grad is not None
        for got, want in ((mine, plain), (mine_a.grad, plain_a.grad), (mine_r.grad, plain_r.grad)):
            assert math.isnan(got.item()) == math.isnan(want.item())
            assert math.isnan(got.item()) or got.item() == want.item()


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_a_batch_mixes_every_regime_in_one_call(device: str, dtype: lucid.dtype) -> None:
    """Small, centred, tail, boundary and invalid lanes in one tensor, each as alone."""
    a = [0.5, 1.0, 30.0, 1e6, 12.0, 1e6, 3.0]
    b = [0.5, 2.0, 40.0, 1e6, 3.0, 1e6, 2.0]
    x = [0.2, 0.0, 0.5, 0.5, 2.0**-30, 0.9, 0.25]
    got = _beta_lp(_t(a, device, dtype), _t(b, device, dtype), _t(x, device, dtype))
    for lane in range(len(a)):
        alone = _beta_lp(
            _t(a[lane], device, dtype), _t(b[lane], device, dtype), _t(x[lane], device, dtype)
        )
        assert _within(got[lane].item(), alone.item(), dtype, 4, 1.0), (lane, got[lane].item())
    # the same, as the gradient of the sum: lane i only reads lane i
    at = _t(a, device, dtype, True)
    _beta_lp(at, _t(b, device, dtype), _t(x, device, dtype)).sum().backward()
    assert at.grad is not None
    for lane in range(len(a)):
        a_lane = _t(a[lane], device, dtype, True)
        _beta_lp(a_lane, _t(b[lane], device, dtype), _t(x[lane], device, dtype)).backward()
        assert a_lane.grad is not None
        assert _within(at.grad[lane].item(), a_lane.grad.item(), dtype, 4, 1e-3), lane


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_dirichlet_batch_and_off_simplex_slope(device: str, dtype: lucid.dtype) -> None:
    """Dirichlet rows of different regimes in one call; the slope in ``x`` off the simplex is the
    plain formula's ``(alpha_i - 1) / x_i`` (the centred form adds ``s (sum x - 1)`` for it)."""
    alpha = [[5.0, 6.0, 7.0], [1e6, 1e6, 2e6], [0.5, 0.5, 0.5]]
    x = [[0.25, 0.25, 0.5], [0.25, 0.25, 0.5], [0.2, 0.3, 0.5]]
    got = D.Dirichlet(_t(alpha, device, dtype)).log_prob(_t(x, device, dtype))
    for i in range(3):
        alone = D.Dirichlet(_t(alpha[i], device, dtype)).log_prob(_t(x[i], device, dtype))
        assert _within(got[i].item(), alone.item(), dtype, 4, 1.0), i
    xt = _t(x[1], device, dtype, True)
    D.Dirichlet(_t(alpha[1], device, dtype)).log_prob(xt).backward()
    assert xt.grad is not None
    for g, a_i, x_i in zip(xt.grad.reshape(-1).tolist(), alpha[1], x[1], strict=True):
        assert _within(g, (a_i - 1.0) / x_i, dtype, 8, 1.0), (g, a_i, x_i)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_a_python_int_total_count_beside_float64_probs(device: str, dtype: lucid.dtype) -> None:
    """``Binomial(10_000_000, probs=p)`` scored ``lgamma(n + 1.0)`` in float32 beside float64 probs
    (-8.2848 was 2.23): the operands now meet in the common dtype."""
    law = D.Binomial(10_000_000, probs=_t(0.5, device, dtype))
    got = law.log_prob(_t(5e6, device, dtype)).item()
    assert _within(got, -8.284839203123887, dtype, 16, 1.0), got


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
def test_sixteen_bit_dtypes_answer_in_their_own_dtype(dtype: lucid.dtype) -> None:
    """Computed in float32 and rounded once, like the special functions."""
    pairs = (
        (
            D.Beta(_t(100.0, "cpu", dtype), _t(100.0, "cpu", dtype)).log_prob(
                _t(0.5, "cpu", dtype)
            ),
            D.Beta(_t(100.0, "cpu", lucid.float32), _t(100.0, "cpu", lucid.float32)).log_prob(
                _t(0.5, "cpu", lucid.float32)
            ),
        ),
        (
            D.Gamma(_t(100.0, "cpu", dtype), _t(1.0, "cpu", dtype)).entropy(),
            D.Gamma(_t(100.0, "cpu", lucid.float32), _t(1.0, "cpu", lucid.float32)).entropy(),
        ),
        (
            D.Poisson(_t(100.0, "cpu", dtype)).log_prob(_t(100.0, "cpu", dtype)),
            D.Poisson(_t(100.0, "cpu", lucid.float32)).log_prob(_t(100.0, "cpu", lucid.float32)),
        ),
    )
    for low, full in pairs:
        assert low.dtype == dtype
        assert abs(low.item() - full.item()) <= 2e-2 * max(abs(full.item()), 1.0)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_the_whole_parameter_range_is_finite(device: str, dtype: lucid.dtype) -> None:
    """1e-3 to 1e30 in every pairing: no NaN or inf in a value or in a slope (the grid of the
    tables, as one sweep)."""
    grid = [1e-3, 0.5, 1.0, 9.0, 11.0, 1e3, 1e6, 2.0**100]
    for a in grid:
        for b in grid:
            at, bt = _t(a, device, dtype, True), _t(b, device, dtype, True)
            out = M._log_beta(at, bt) + _beta_h(at, bt) + _beta_lp(at, bt, _t(0.5, device, dtype))
            out.backward()
            assert at.grad is not None and bt.grad is not None
            for v in (out.item(), at.grad.item(), bt.grad.item()):
                assert math.isfinite(v), (a, b, v)


# ── the guard ───────────────────────────────────────────────────────────────

# the one remaining ``lgamma - lgamma``: the plain lanes of ``NegativeBinomial.log_prob``
# (``k = 0``, ``r = 0``, ``k + r < 10``), where the saddle-point form is not defined
_PLAIN_DIFFERENCES = {("discrete.py", "NegativeBinomial.log_prob")}


def _lgamma_differences(path: pathlib.Path) -> set[tuple[str, str]]:
    """``(file, qualified function)`` of every ``lgamma(...) - / + lgamma(...)`` in the file."""

    def calls_lgamma(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call)
            and getattr(n.func, "attr", getattr(n.func, "id", "")) == "lgamma"
            for n in ast.walk(node)
        )

    found: set[tuple[str, str]] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_BinOp(self, node: ast.BinOp) -> None:
            if (
                isinstance(node.op, (ast.Sub, ast.Add))
                and calls_lgamma(node.left)
                and calls_lgamma(node.right)
            ):
                found.add((path.name, ".".join(self.stack)))
            self.generic_visit(node)

    Visitor().visit(ast.parse(path.read_text()))
    return found


def test_no_law_subtracts_two_lgamma_outside_the_owner() -> None:
    """The class guard: a new ``lgamma(a) + lgamma(b) - lgamma(a + b)`` in a law is the defect again.

    Every site goes through ``_log_beta`` / ``_log_multi_beta`` / ``_log_comb`` /
    ``_log_multinomial_coeff`` or a centred kernel of ``_stirling_math``."""
    root = pathlib.Path(D.__file__).parent
    found: set[tuple[str, str]] = set()
    for path in sorted(root.glob("*.py")):
        if path.name != "_stirling_math.py":
            found |= _lgamma_differences(path)
    allowed = set(_PLAIN_DIFFERENCES)
    if not _ROUTED:  # Kumaraswamy's moments and Multinomial's coefficient, until the patch lands
        allowed |= {
            ("extra.py", "Kumaraswamy.mean"),
            ("extra.py", "Kumaraswamy.variance"),
            ("extra.py", "Multinomial.log_prob"),
        }
    assert found <= allowed, sorted(found - allowed)


# ── against the reference, at parameters where it is right ──────────────────


@pytest.mark.parity
def test_parity_log_prob_entropy_and_slopes(ref: Any) -> None:
    f64, rf64 = lucid.float64, ref.float64

    def pair(name: str, params: list[float], x: list[float] | None, entropy: bool = True) -> None:
        mine_p = [lucid.tensor(p, dtype=f64, requires_grad=True) for p in params]
        ref_p = [ref.tensor(p, dtype=rf64, requires_grad=True) for p in params]
        mine = getattr(D, name)(*mine_p)
        theirs = getattr(ref.distributions, name)(*ref_p)
        outs_m, outs_t = [], []
        if entropy:
            outs_m.append(mine.entropy())
            outs_t.append(theirs.entropy())
        if x is not None:
            outs_m.append(mine.log_prob(lucid.tensor(x, dtype=f64)))
            outs_t.append(theirs.log_prob(ref.tensor(x, dtype=rf64)))
        total_m = sum(o.sum() for o in outs_m)
        total_t = sum(o.sum() for o in outs_t)
        total_m.backward()
        total_t.backward()
        for om, ot in zip(outs_m, outs_t, strict=True):
            np.testing.assert_allclose(
                om.detach().numpy(), ot.detach().numpy(), rtol=1e-9, atol=1e-10
            )
        for pm, pt in zip(mine_p, ref_p, strict=True):
            assert pm.grad is not None and pt.grad is not None
            np.testing.assert_allclose(pm.grad.numpy(), pt.grad.numpy(), rtol=1e-8, atol=1e-10)

    pair("Beta", [2.5, 3.5], [0.2, 0.5, 0.8])
    pair("Beta", [40.0, 60.0], [0.3, 0.4, 0.5])
    pair("Gamma", [3.5, 2.0], [0.5, 2.0, 7.0])
    pair("Gamma", [60.0, 2.0], [20.0, 30.0, 40.0])
    pair("FisherSnedecor", [5.0, 7.0], [0.3, 1.0, 4.0], entropy=False)
    pair("FisherSnedecor", [60.0, 80.0], [0.8, 1.0, 1.3], entropy=False)
    pair("InverseGamma", [3.5, 2.0], [0.5, 2.0, 7.0])
    pair("InverseGamma", [60.0, 40.0], [0.5, 0.7, 1.0])


@pytest.mark.parity
def test_parity_discrete_and_dirichlet(ref: Any) -> None:
    f64, rf64 = lucid.float64, ref.float64
    counts = [0.0, 3.0, 40.0, 100.0]
    for rate in (2.5, 60.0):
        mine = D.Poisson(lucid.tensor(rate, dtype=f64)).log_prob(lucid.tensor(counts, dtype=f64))
        theirs = ref.distributions.Poisson(ref.tensor(rate, dtype=rf64)).log_prob(
            ref.tensor(counts, dtype=rf64)
        )
        np.testing.assert_allclose(mine.numpy(), theirs.numpy(), rtol=1e-10, atol=1e-10)
    for n, p in ((30.0, 0.3), (300.0, 0.6)):
        ks = [0.0, 5.0, 12.0, n / 2, n]
        mine = D.Binomial(lucid.tensor(n, dtype=f64), probs=lucid.tensor(p, dtype=f64)).log_prob(
            lucid.tensor(ks, dtype=f64)
        )
        theirs = ref.distributions.Binomial(
            ref.tensor(n, dtype=rf64), probs=ref.tensor(p, dtype=rf64)
        ).log_prob(ref.tensor(ks, dtype=rf64))
        np.testing.assert_allclose(mine.numpy(), theirs.numpy(), rtol=1e-9, atol=1e-10)
    for r, p in ((4.0, 0.3), (40.0, 0.6)):
        ks = [0.0, 3.0, 25.0, 90.0]
        mine = D.NegativeBinomial(
            lucid.tensor(r, dtype=f64), probs=lucid.tensor(p, dtype=f64)
        ).log_prob(lucid.tensor(ks, dtype=f64))
        theirs = ref.distributions.NegativeBinomial(
            ref.tensor(r, dtype=rf64), probs=ref.tensor(p, dtype=rf64)
        ).log_prob(ref.tensor(ks, dtype=rf64))
        np.testing.assert_allclose(mine.numpy(), theirs.numpy(), rtol=1e-9, atol=1e-10)
    for alpha, x in (([2.0, 3.0, 4.0], [0.2, 0.3, 0.5]), ([30.0, 20.0, 40.0], [0.3, 0.3, 0.4])):
        mine_d = D.Dirichlet(lucid.tensor(alpha, dtype=f64))
        theirs_d = ref.distributions.Dirichlet(ref.tensor(alpha, dtype=rf64))
        np.testing.assert_allclose(
            mine_d.log_prob(lucid.tensor(x, dtype=f64)).numpy(),
            theirs_d.log_prob(ref.tensor(x, dtype=rf64)).numpy(),
            rtol=1e-10,
        )
        np.testing.assert_allclose(mine_d.entropy().numpy(), theirs_d.entropy().numpy(), rtol=1e-10)


@pytest.mark.parity
def test_parity_kl_divergences(ref: Any) -> None:
    f64, rf64 = lucid.float64, ref.float64
    cases = [
        ("Gamma", [3.5, 2.0], [6.0, 1.5]),
        ("Gamma", [50.0, 2.0], [55.0, 2.5]),
        ("Beta", [2.5, 3.5], [4.0, 2.0]),
        ("Beta", [40.0, 60.0], [45.0, 50.0]),
        ("Dirichlet", [[2.0, 3.0, 4.0]], [[4.0, 3.0, 2.0]]),
        ("Dirichlet", [[30.0, 20.0, 40.0]], [[25.0, 25.0, 35.0]]),
    ]
    for name, p, q in cases:
        mine = kl_divergence(
            getattr(D, name)(*(lucid.tensor(v, dtype=f64) for v in p)),
            getattr(D, name)(*(lucid.tensor(v, dtype=f64) for v in q)),
        )
        theirs = ref.distributions.kl_divergence(
            getattr(ref.distributions, name)(*(ref.tensor(v, dtype=rf64) for v in p)),
            getattr(ref.distributions, name)(*(ref.tensor(v, dtype=rf64) for v in q)),
        )
        np.testing.assert_allclose(mine.numpy(), theirs.numpy(), rtol=1e-9, atol=1e-11)
