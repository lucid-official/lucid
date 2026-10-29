"""Bessel-function coefficients, and the ``float64`` path of ``lucid.i0`` and
the ``lucid.special`` Bessel family.

The ``float32`` path of those functions uses the Abramowitz & Stegun §9.4 /
§9.8 polynomial fits (the ``_AS_*`` tables below, shared by ``lucid.i0`` and
``lucid.special.i0e``), whose coefficients carry about eight significant
digits: good to ~1e-7, the float32 rounding level, and no further.  In
``float64`` that left errors of 1e-8 relative (``i0``) and, near a zero of
``J0`` / ``J1``, the whole value (``J0`` at its first zero returned
``-3.6e-8``).  A ``float64`` input takes the Cephes double-precision
approximations here instead:

* ``I0`` / ``I1`` (and the ``e^{-|x|}``-scaled forms) — Chebyshev series on
  ``[0, 8]`` and on ``(8, ∞)`` in ``32/x``; ``I0`` itself is its power series
  in ``x²/4`` on ``[0, 8]``, which keeps the curvature at 0.
* ``J0`` / ``J1`` / ``Y0`` / ``Y1`` — rational functions on ``[0, 5]``, with
  ``J0`` / ``J1`` written as ``(x² - r₁²)(x² - r₂²) · R(x²)`` so their first two
  zeros are exact, and the Hankel-asymptotic ``P cos - Q sin`` form beyond.
* ``K0`` / ``K1`` (and the ``e^{x}``-scaled forms) — Chebyshev series on
  ``[0, 2]`` (with the ``log(x/2) · I`` term) and on ``(2, ∞)`` in ``8/x``.

Every function runs both of its branches on an argument clamped into that
branch's interval and selects with ``where``, so neither branch sees an
argument it would overflow or divide by zero on — in the forward value or in
its gradient.  ``Y`` and ``K`` live on the positive half-line: ``-inf`` /
``+inf`` at 0 and NaN below, as the reference framework has them.

The Cephes coefficient tables are those of the Cephes Mathematical Library
(Stephen L. Moshier, 3-clause BSD licence; see THIRD_PARTY_NOTICES.md at the
repository root).
"""

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import lucid
from lucid._ops.composite._opmath import _in_float32

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_PIO4 = math.pi / 4.0
_THPIO4 = 3.0 * math.pi / 4.0
_SQ2OPI = math.sqrt(2.0 / math.pi)
_TWOOPI = 2.0 / math.pi
# ``I0(x) = Σ (x²/4)^k / (k!)²``, highest power first; 30 terms hold ``|x| <= 8``
# to float64's last digit (the 30th is 1e-29 of the sum).
_I0_SERIES: tuple[float, ...] = tuple(1.0 / math.factorial(k) ** 2 for k in range(30, -1, -1))
# Abramowitz & Stegun 9.8.1 / 9.8.2 — the float32 ``I0``: a series in
# ``(x/3.75)²`` for ``|x| <= 3.75``, and in ``3.75/|x|`` times
# ``exp(|x|)/√|x|`` beyond.  The large-argument series alone is the scaled
# ``i0e``.
_AS_I0_SMALL: tuple[float, ...] = (
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.0360768,
    0.0045813,
)
_AS_I0_LARGE: tuple[float, ...] = (
    0.39894228,
    0.01328592,
    0.00225319,
    -0.00157565,
    0.00916281,
    -0.02057706,
    0.02635537,
    -0.01647633,
    0.00392377,
)

# Squares of the first two zeros of J0 and of J1.
_J0_DR1 = 5.78318596294678452118e00
_J0_DR2 = 3.04712623436620863991e01
_J1_Z1 = 1.46819706421238932572e01
_J1_Z2 = 4.92184563216946036703e01


def _polevl(z: Tensor, coeffs: tuple[float, ...]) -> Tensor:
    """``c[0] z^n + … + c[n]`` by Horner's rule."""
    acc = lucid.full_like(z, coeffs[0])
    for c in coeffs[1:]:
        acc = acc * z + c
    return acc


def _p1evl(z: Tensor, coeffs: tuple[float, ...]) -> Tensor:
    """``z^(n+1) + c[0] z^n + … + c[n]`` — the leading coefficient is 1."""
    acc = z + coeffs[0]
    for c in coeffs[1:]:
        acc = acc * z + c
    return acc


def _chbevl(y: Tensor, coeffs: tuple[float, ...]) -> Tensor:
    """Chebyshev series ``Σ' c[k] T_k(y/2)`` by Clenshaw's recurrence."""
    b0 = lucid.full_like(y, coeffs[0])
    b1 = lucid.zeros_like(y)
    b2 = lucid.zeros_like(y)
    for c in coeffs[1:]:
        b2 = b1
        b1 = b0
        b0 = y * b1 - b2 + c
    return 0.5 * (b0 - b2)


def _below(ax: Tensor, edge: float) -> Tensor:
    """``ax`` clamped into a lower branch ``[·, edge]``."""
    return lucid.clamp(ax, max=edge)


def _above(ax: Tensor, edge: float) -> Tensor:
    """``ax`` clamped into an upper branch ``[edge, ∞)``."""
    return lucid.clamp(ax, min=edge)


def _on_half_line(x: Tensor, value: Tensor, at_zero: float) -> Tensor:
    """``value`` where ``x > 0``, ``at_zero`` at 0 and NaN below."""
    edge = lucid.where(x == 0.0, lucid.full_like(x, at_zero), lucid.full_like(x, math.nan))
    return lucid.where(x > 0.0, value, edge)


def _positive(x: Tensor, edge: float) -> Tensor:
    """``x`` clamped into ``(0, edge]``; a point outside the domain reads 1."""
    return lucid.where(x > 0.0, _below(x, edge), lucid.ones_like(x))


def _half_line_split(
    x: Tensor,
    edge: float,
    small: Callable[[Tensor], Tensor],
    large: Callable[[Tensor], Tensor],
    at_zero: float,
) -> Tensor:
    """A ``Y`` / ``K`` function: ``small`` on ``(0, edge]``, ``large`` beyond.

    Each branch runs on ``x`` clamped into its own interval, so neither
    divides by zero or takes the log of a non-positive number — the unused
    branch's ``nan`` would reach the gradient through ``where`` — and the
    domain edge is :func:`_on_half_line`'s.  Both dtype paths of all six
    functions go through here, so they share one domain rule — and one
    precision rule: a 16-bit ``x`` is evaluated in float32
    (:func:`_in_float32`), where ``Y0`` had answered NaN on ``(0, 8]``.
    """

    def split(v: Tensor) -> Tensor:
        value = lucid.where(v <= edge, small(_positive(v, edge)), large(_above(v, edge)))
        return _on_half_line(v, value, at_zero)

    return _in_float32(split, x)


# ── I0 / I1 ──────────────────────────────────────────────────────────────────


def _vanish_at_inf(x: Tensor, scaled: Callable[[Tensor], Tensor]) -> Tensor:
    """``scaled(x)`` for a function that tends to 0 at ``±inf``, 0 there.

    The function is evaluated at ``x`` clamped to the dtype's largest finite
    value, so every slope it forms stays finite: at ``inf`` itself an
    ``exp(-|x|)`` or a ``1 / sqrt(x)`` meets its own derivative as
    ``0 * inf``, a NaN in the second derivative where the answer is 0.
    The clamp has no gradient there, and the ``where`` takes the constant.
    """
    bound = lucid.finfo(x.dtype).max
    value = scaled(lucid.clip(x, -bound, bound))
    return lucid.where(lucid.isinf(x), lucid.zeros_like(x), value)


def _i0_series(x: Tensor) -> Tensor:
    """``I0(x)`` for ``|x| <= 8`` by its power series in ``x²/4``.

    A function of ``x * x``, so its curvature at 0 is the true ``1/2``;
    the Chebyshev series in ``|x|`` has none, as ``|x|`` has no second
    derivative there for the chain rule to carry.
    """
    return _polevl(x * x * 0.25, _I0_SERIES)


def _i0e(x: Tensor) -> Tensor:
    """``exp(-|x|) I0(x)``."""

    def scaled(v: Tensor) -> Tensor:
        ax = lucid.abs(v)
        small = lucid.exp(-_below(ax, 8.0)) * _i0_series(lucid.clip(v, -8.0, 8.0))
        big = _above(ax, 8.0)
        large = _chbevl(32.0 / big - 2.0, _I0E_B) / lucid.sqrt(big)
        return lucid.where(ax <= 8.0, small, large)

    return _vanish_at_inf(x, scaled)


def _i0(x: Tensor) -> Tensor:
    """``I0(x)``."""
    return lucid.exp(lucid.abs(x)) * _i0e(x)


def _i0_as_fit(input: Tensor) -> Tensor:
    ax = lucid.abs(input)
    ax_safe = _above(ax, 3.75)
    t_small = (lucid.clip(input, -3.75, 3.75) * (1.0 / 3.75)) ** 2
    val_small, t_pow = lucid.full_like(input, _AS_I0_SMALL[0]), lucid.ones_like(input)
    for c in _AS_I0_SMALL[1:]:
        t_pow = t_pow * t_small
        val_small = val_small + c * t_pow
    y_large = 3.75 / ax_safe
    val_large, y_pow = lucid.full_like(input, _AS_I0_LARGE[0]), lucid.ones_like(input)
    for c in _AS_I0_LARGE[1:]:
        y_pow = y_pow * y_large
        val_large = val_large + c * y_pow
    return lucid.where(ax <= 3.75, val_small, val_large * lucid.exp(ax_safe) / lucid.sqrt(ax_safe))


def _i1e(x: Tensor) -> Tensor:
    """``exp(-|x|) I1(x)``; odd in ``x``.

    The small branch is odd through ``x`` itself, so its slope at 0 is the
    true ``1/2``: written ``sign(x) · g(|x|)`` it would have a zero
    derivative there, as ``sign`` and ``|x|`` both do.
    """

    def scaled(v: Tensor) -> Tensor:
        ax = lucid.abs(v)
        small = _chbevl(_below(ax, 8.0) * 0.5 - 2.0, _I1E_A) * lucid.clip(v, -8.0, 8.0)
        big = _above(ax, 8.0)
        large = lucid.sign(v) * _chbevl(32.0 / big - 2.0, _I1E_B) / lucid.sqrt(big)
        return lucid.where(ax <= 8.0, small, large)

    return _vanish_at_inf(x, scaled)


def _i1(x: Tensor) -> Tensor:
    """``I1(x)``; odd in ``x``."""
    return lucid.exp(lucid.abs(x)) * _i1e(x)


# ── J0 / J1 / Y0 / Y1 ────────────────────────────────────────────────────────


def _hankel(
    big: Tensor, phase: float, tables: tuple[tuple[float, ...], ...], second_kind: bool
) -> Tensor:
    """The ``x >= 5`` form: ``√(2/πx) (P cos θ - (5/x) Q sin θ)`` for ``J``,
    ``√(2/πx) (P sin θ + (5/x) Q cos θ)`` for ``Y``, with ``θ = x - phase``."""
    pp, pq, qp, qq = tables
    z = 25.0 / (big * big)
    p = _polevl(z, pp) / _polevl(z, pq)
    q = (5.0 / big) * (_polevl(z, qp) / _p1evl(z, qq))
    theta = big - phase
    if second_kind:
        wave = p * lucid.sin(theta) + q * lucid.cos(theta)
    else:
        wave = p * lucid.cos(theta) - q * lucid.sin(theta)
    return wave * _SQ2OPI / lucid.sqrt(big)


def _j0_small(s: Tensor) -> Tensor:
    z = s * s
    return (z - _J0_DR1) * (z - _J0_DR2) * _polevl(z, _J0_RP) / _p1evl(z, _J0_RQ)


def _j1_small(x: Tensor) -> Tensor:
    """``J1`` on ``[-5, 5]``, odd through ``x`` itself (slope ``1/2`` at 0)."""
    z = x * x
    ratio = _polevl(z, _J1_RP) / _p1evl(z, _J1_RQ)
    return ratio * x * (z - _J1_Z1) * (z - _J1_Z2)


def _j0(x: Tensor) -> Tensor:
    """``J0(x)``; even in ``x``."""
    ax = lucid.abs(x)
    large = _hankel(_above(ax, 5.0), _PIO4, _J0_HANKEL, second_kind=False)
    return lucid.where(ax <= 5.0, _j0_small(_below(ax, 5.0)), large)


def _j1(x: Tensor) -> Tensor:
    """``J1(x)``; odd in ``x``."""
    ax = lucid.abs(x)
    large = lucid.sign(x) * _hankel(_above(ax, 5.0), _THPIO4, _J1_HANKEL, second_kind=False)
    return lucid.where(ax <= 5.0, _j1_small(lucid.clip(x, -5.0, 5.0)), large)


def _y0_small(s: Tensor) -> Tensor:
    z = s * s
    return _polevl(z, _Y0_YP) / _p1evl(z, _Y0_YQ) + _TWOOPI * lucid.log(s) * _j0_small(s)


def _y1_small(s: Tensor) -> Tensor:
    z = s * s
    return s * (_polevl(z, _Y1_YP) / _p1evl(z, _Y1_YQ)) + _TWOOPI * (
        _j1_small(s) * lucid.log(s) - 1.0 / s
    )


def _y0(x: Tensor) -> Tensor:
    """``Y0(x)`` on ``x > 0``; ``x <= 0`` as the module says."""
    return _half_line_split(
        x, 5.0, _y0_small, lambda big: _hankel(big, _PIO4, _J0_HANKEL, second_kind=True), -math.inf
    )


def _y1(x: Tensor) -> Tensor:
    """``Y1(x)`` on ``x > 0``; ``x <= 0`` as the module says."""
    return _half_line_split(
        x,
        5.0,
        _y1_small,
        lambda big: _hankel(big, _THPIO4, _J1_HANKEL, second_kind=True),
        -math.inf,
    )


# ── K0 / K1 ──────────────────────────────────────────────────────────────────


def _k0_small(s: Tensor) -> Tensor:
    return _chbevl(s * s - 2.0, _K0_A) - lucid.log(0.5 * s) * _i0(s)


def _k1_small(s: Tensor) -> Tensor:
    return lucid.log(0.5 * s) * _i1(s) + _chbevl(s * s - 2.0, _K1_A) / s


def _k_large_scaled(big: Tensor, coeffs: tuple[float, ...]) -> Tensor:
    """``e^x K(x)`` for ``x >= 2``."""
    return _chbevl(8.0 / big - 2.0, coeffs) / lucid.sqrt(big)


def _k0e(x: Tensor) -> Tensor:
    """``exp(x) K0(x)`` on ``x > 0``; ``x <= 0`` as the module says."""
    return _half_line_split(
        x,
        2.0,
        lambda s: _k0_small(s) * lucid.exp(s),
        lambda big: _k_large_scaled(big, _K0_B),
        math.inf,
    )


def _k0(x: Tensor) -> Tensor:
    """``K0(x)`` on ``x > 0``; ``x <= 0`` as the module says."""
    return _half_line_split(
        x, 2.0, _k0_small, lambda big: _k_large_scaled(big, _K0_B) * lucid.exp(-big), math.inf
    )


def _k1e(x: Tensor) -> Tensor:
    """``exp(x) K1(x)`` on ``x > 0``; ``x <= 0`` as the module says."""
    return _half_line_split(
        x,
        2.0,
        lambda s: _k1_small(s) * lucid.exp(s),
        lambda big: _k_large_scaled(big, _K1_B),
        math.inf,
    )


def _k1(x: Tensor) -> Tensor:
    """``K1(x)`` on ``x > 0``; ``x <= 0`` as the module says."""
    return _half_line_split(
        x, 2.0, _k1_small, lambda big: _k_large_scaled(big, _K1_B) * lucid.exp(-big), math.inf
    )


# ── Cephes coefficient tables ────────────────────────────────────────────────
#
# Each table is listed highest-order coefficient first, as Horner's rule and
# Clenshaw's recurrence consume it.  A ``_p1evl`` table omits its leading 1.

_I0E_B: tuple[float, ...] = (
    -7.233180487874754e-18,
    -4.830504485944182e-18,
    4.46562142029676e-17,
    3.461222867697461e-17,
    -2.8276239805165836e-16,
    -3.425485619677219e-16,
    1.7725601330565263e-15,
    3.8116806693526224e-15,
    -9.554846698828307e-15,
    -4.150569347287222e-14,
    1.54008621752141e-14,
    3.8527783827421426e-13,
    7.180124451383666e-13,
    -1.7941785315068062e-12,
    -1.3215811840447713e-11,
    -3.1499165279632416e-11,
    1.1889147107846439e-11,
    4.94060238822497e-10,
    3.3962320257083865e-09,
    2.266668990498178e-08,
    2.0489185894690638e-07,
    2.8913705208347567e-06,
    6.889758346916825e-05,
    0.0033691164782556943,
    0.8044904110141088,
)

_I1E_A: tuple[float, ...] = (
    2.7779141127610464e-18,
    -2.111421214358166e-17,
    1.5536319577362005e-16,
    -1.1055969477353862e-15,
    7.600684294735408e-15,
    -5.042185504727912e-14,
    3.223793365945575e-13,
    -1.9839743977649436e-12,
    1.1736186298890901e-11,
    -6.663489723502027e-11,
    3.625590281552117e-10,
    -1.8872497517228294e-09,
    9.381537386495773e-09,
    -4.445059128796328e-08,
    2.0032947535521353e-07,
    -8.568720264695455e-07,
    3.4702513081376785e-06,
    -1.3273163656039436e-05,
    4.781565107550054e-05,
    -0.00016176081582589674,
    0.0005122859561685758,
    -0.0015135724506312532,
    0.004156422944312888,
    -0.010564084894626197,
    0.024726449030626516,
    -0.05294598120809499,
    0.1026436586898471,
    -0.17641651835783406,
    0.25258718644363365,
)

_I1E_B: tuple[float, ...] = (
    7.517296310842105e-18,
    4.414348323071708e-18,
    -4.6503053684893586e-17,
    -3.209525921993424e-17,
    2.96262899764595e-16,
    3.3082023109209285e-16,
    -1.8803547755107825e-15,
    -3.8144030724370075e-15,
    1.0420276984128802e-14,
    4.272440016711951e-14,
    -2.1015418427726643e-14,
    -4.0835511110921974e-13,
    -7.198551776245908e-13,
    2.0356285441470896e-12,
    1.4125807436613782e-11,
    3.2526035830154884e-11,
    -1.8974958123505413e-11,
    -5.589743462196584e-10,
    -3.835380385964237e-09,
    -2.6314688468895196e-08,
    -2.512236237870209e-07,
    -3.882564808877691e-06,
    -0.00011058893876262371,
    -0.009761097491361469,
    0.7785762350182801,
)

_J0_PP: tuple[float, ...] = (
    0.0007969367292973471,
    0.08283523921074408,
    1.239533716464143,
    5.447250030587687,
    8.74716500199817,
    5.303240382353949,
    1.0,
)

_J0_PQ: tuple[float, ...] = (
    0.0009244088105588637,
    0.08562884743544745,
    1.2535274390105895,
    5.470977403304171,
    8.761908832370695,
    5.306052882353947,
    1.0,
)

_J0_QP: tuple[float, ...] = (
    -0.011366383889846916,
    -1.2825271867050931,
    -19.553954425773597,
    -93.20601521237683,
    -177.68116798048806,
    -147.07750515495118,
    -51.41053267665993,
    -6.050143506007285,
)

_J0_QQ: tuple[float, ...] = (
    64.3178256118178,
    856.4300259769806,
    3882.4018360540163,
    7240.467741956525,
    5930.727011873169,
    2062.0933166032783,
    242.0057402402914,
)

_J0_RP: tuple[float, ...] = (
    -4794432209.782018,
    1956174919465.5657,
    -249248344360967.72,
    9708622510473064.0,
)

_J0_RQ: tuple[float, ...] = (
    499.563147152651,
    173785.4016763747,
    48440965.83399621,
    11185553704.535683,
    2112775201154.892,
    310518229857422.56,
    3.1812195594320496e16,
    1.7108629408104315e18,
)

_J1_PP: tuple[float, ...] = (
    0.0007621256162081731,
    0.07313970569409176,
    1.1271960812968493,
    5.112079511468076,
    8.424045901417724,
    5.214515986823615,
    1.0,
)

_J1_PQ: tuple[float, ...] = (
    0.0005713231280725487,
    0.06884559087544954,
    1.105142326340617,
    5.073863861286015,
    8.399855543276042,
    5.209828486823619,
    1.0,
)

_J1_QP: tuple[float, ...] = (
    0.05108625947501766,
    4.982138729512334,
    75.82382841325453,
    366.7796093601508,
    710.8563049989261,
    597.4896124006136,
    211.68875710057213,
    25.207020585802372,
)

_J1_QQ: tuple[float, ...] = (
    74.23732770356752,
    1056.4488603826283,
    4986.410583376536,
    9562.318924047562,
    7997.041604473507,
    2826.1927851763908,
    336.0936078106983,
)

_J1_RP: tuple[float, ...] = (
    -899971225.7055594,
    452228297998.19403,
    -72749424522181.83,
    3682957328638529.0,
)

_J1_RQ: tuple[float, ...] = (
    620.8364781180543,
    256987.25675774884,
    83514679.14319493,
    22151159547.97925,
    4749141220799.914,
    784369607876235.9,
    8.952223361846274e16,
    5.322786203326801e18,
)

_Y0_YP: tuple[float, ...] = (
    15592.436785523574,
    -14663929.59039716,
    5435264770.518765,
    -982136065717.9115,
    87590639439536.7,
    -3466283033847297.0,
    4.4273326857256984e16,
    -1.8495080043698668e16,
)

_Y0_YQ: tuple[float, ...] = (
    1041.2835366425984,
    626107.330137135,
    268919633.39381415,
    86400248710.3935,
    20297961275010.555,
    3171577528429750.5,
    2.5059625617265306e17,
)

_Y1_YP: tuple[float, ...] = (
    1263204747.9017804,
    -647355876379.1603,
    114509511541823.73,
    -8127702555013251.0,
    2.024394757135949e17,
    -7.788771962659501e17,
)

_Y1_YQ: tuple[float, ...] = (
    594.3015923461282,
    235564.09294306856,
    73481194.44597217,
    18760131610.870617,
    3882312774962.3857,
    620557727146953.8,
    6.871410873553005e16,
    3.9727060811656064e18,
)

_K0_A: tuple[float, ...] = (
    1.374465435613523e-16,
    4.25981614279661e-14,
    1.0349695257633842e-11,
    1.904516377220209e-09,
    2.5347910790261494e-07,
    2.286212103119452e-05,
    0.001264615411446926,
    0.0359799365153615,
    0.3442898999246285,
    -0.5353273932339028,
)

_K0_B: tuple[float, ...] = (
    5.300433772686263e-18,
    -1.6475804301524212e-17,
    5.2103915050390274e-17,
    -1.678231096805412e-16,
    5.512055978524319e-16,
    -1.848593377343779e-15,
    6.3400764774050706e-15,
    -2.2275133269916698e-14,
    8.032890775363575e-14,
    -2.9800969231727303e-13,
    1.140340588208475e-12,
    -4.514597883373944e-12,
    1.8559491149547177e-11,
    -7.957489244477107e-11,
    3.577397281400301e-10,
    -1.69753450938906e-09,
    8.574034017414225e-09,
    -4.660489897687948e-08,
    2.766813639445015e-07,
    -1.8317555227191195e-06,
    1.39498137188765e-05,
    -0.00012849549581627802,
    0.0015698838857300533,
    -0.0314481013119645,
    2.4403030820659555,
)

_K1_A: tuple[float, ...] = (
    -7.023863479386288e-18,
    -2.427449850519366e-15,
    -6.666901694199329e-13,
    -1.4114883926335278e-10,
    -2.213387630734726e-08,
    -2.4334061415659684e-06,
    -0.0001730288957513052,
    -0.006975723859639864,
    -0.12261118082265715,
    -0.3531559607765449,
    1.5253002273389478,
)

_K1_B: tuple[float, ...] = (
    -5.756744483665017e-18,
    1.7940508731475592e-17,
    -5.689462558442859e-17,
    1.838093544366639e-16,
    -6.057047248373319e-16,
    2.038703165624334e-15,
    -7.019837090418314e-15,
    2.4771544244813043e-14,
    -8.976705182324994e-14,
    3.3484196660784293e-13,
    -1.2891739609510289e-12,
    5.13963967348173e-12,
    -2.1299678384275683e-11,
    9.218315187605006e-11,
    -4.1903547593418965e-10,
    2.015049755197033e-09,
    -1.0345762465678097e-08,
    5.7410841254500495e-08,
    -3.5019606030878126e-07,
    2.406484947837217e-06,
    -1.936197974166083e-05,
    0.00019521551847135162,
    -0.002857816859622779,
    0.10392373657681724,
    2.7206261904844427,
)

_J0_HANKEL = (_J0_PP, _J0_PQ, _J0_QP, _J0_QQ)
_J1_HANKEL = (_J1_PP, _J1_PQ, _J1_QP, _J1_QQ)
