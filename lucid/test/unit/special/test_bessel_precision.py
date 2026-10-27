"""float64 Bessel functions are double-precision accurate (LCD-110).

The defect class: a ``float64`` input ran the ``float32``-grade
Abramowitz & Stegun polynomials, whose eight-digit coefficients stop at
~1e-7.  ``i0(1.)`` was ``1.2660658480`` against ``1.2660658778``,
``bessel_j0`` at its first zero ``-3.6e-8`` against ``~1e-17``, and
``bessel_j1(3.8317)`` off by 3e-3 relative.  ``float64`` now takes the
Cephes double-precision series; ``float32`` keeps its path.

* Every function against a 50-digit table (computed with mpmath and
  written out below), over a grid that holds tiny arguments, both sides
  of every branch edge, the first two zeros of ``J0`` / ``J1`` and large
  arguments — at double-precision tolerance.
* The ``float32`` path, on CPU and Metal: its own accuracy, its dtype,
  and the same answer on both devices.
* Derivatives through the ``float64`` path, against the closed-form
  identities (``J0' = -J1``, ``I0' = I1``, ``K0' = -K1`` ...).
* The Kaiser window, whose ``float64`` form was capped by a ``float32``-grade
  denominator.
* Against the reference: values, and the edges (``0``, negative, ``±inf``,
  NaN) of every function.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.special as S
from lucid.signal.windows import kaiser
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
_EVEN = {"i0", "i0e", "j0"}
_ODD = {"i1", "i1e", "j1"}

_INF = math.inf
_NAN = math.nan

# The first two zeros of J0 and J1, as the nearest doubles.
_ZEROS = {
    "j0": (2.404825557695773, 5.520078110286311),
    "j1": (3.8317059702075125, 7.015586669815619),
}
_TRUTH: dict[str, tuple[tuple[float, float], ...]] = {
    "i0": (
        (1e-08, 1.0),
        (0.001, 1.0000002500000156),
        (0.5, 1.0634833707413236),
        (1.0, 1.2660658777520084),
        (2.0, 2.2795853023360673),
        (3.0, 4.8807925858650245),
        (4.9, 24.914779075837764),
        (5.0, 27.239871823604446),
        (5.1, 29.788855440238837),
        (7.5, 268.16131151518937),
        (8.0, 427.5641157218048),
        (8.1, 469.5006067101215),
        (12.0, 18948.925349296307),
        (30.0, 781672297823.9775),
        (100.0, 1.0737517071310738e42),
        (700.0, 1.5295933476718737e302),
    ),
    "i0e": (
        (1e-08, 0.9999999900000001),
        (0.001, 0.9990007495835156),
        (0.5, 0.6450352704491501),
        (1.0, 0.46575960759364043),
        (2.0, 0.30850832255367105),
        (3.0, 0.2430003541618254),
        (4.9, 0.1855299720819534),
        (5.0, 0.18354081260932836),
        (5.1, 0.18161510209591797),
        (7.5, 0.14831583007739552),
        (8.0, 0.14343178185685032),
        (8.1, 0.14251180948829528),
        (12.0, 0.11642622121344044),
        (30.0, 0.0731459464822373),
        (100.0, 0.03994437929909668),
        (700.0, 0.015081295651531358),
    ),
    "i1": (
        (1e-08, 5e-09),
        (0.001, 0.0005000000625000026),
        (0.5, 0.2578943053908963),
        (1.0, 0.565159103992485),
        (2.0, 1.590636854637329),
        (3.0, 3.9533702174026093),
        (4.9, 22.199348620092497),
        (5.0, 24.335642142450528),
        (5.1, 26.68043567947711),
        (7.5, 249.58436542268814),
        (8.0, 399.8731367825601),
        (8.1, 439.48430891035844),
        (12.0, 18141.348781638833),
        (30.0, 768532038938.957),
        (100.0, 1.0683693903381625e42),
        (700.0, 1.5285003902339006e302),
    ),
    "i1e": (
        (1e-08, 4.99999995e-09),
        (0.001, 0.0004995003123542213),
        (0.5, 0.1564208031848717),
        (1.0, 0.20791041534970844),
        (2.0, 0.21526928924893765),
        (3.0, 0.19682671329730086),
        (4.9, 0.16530929361992835),
        (5.0, 0.16397226694454237),
        (5.1, 0.16266385459531296),
        (7.5, 0.1380412115485542),
        (8.0, 0.13414249329269817),
        (8.1, 0.13340068832583662),
        (12.0, 0.11146429929018098),
        (30.0, 0.07191633059864755),
        (100.0, 0.03974415302513025),
        (700.0, 0.015070519444716848),
    ),
    "j0": (
        (1e-08, 1.0),
        (0.001, 0.9999997500000156),
        (0.5, 0.9384698072408129),
        (1.0, 0.7651976865579666),
        (2.0, 0.22389077914123567),
        (2.404825557695773, -6.10876525973673e-17),
        (3.0, -0.26005195490193345),
        (4.9, -0.2097383275853262),
        (5.0, -0.1775967713143383),
        (5.1, -0.14433474706050065),
        (5.520078110286311, -2.7522649432621832e-17),
        (7.5, 0.2663396578803784),
        (8.0, 0.1716508071375539),
        (8.1, 0.14751745404437766),
        (12.0, 0.047689310796833535),
        (30.0, -0.08636798358104021),
        (100.0, 0.019985850304223122),
        (700.0, -0.006288272465068767),
    ),
    "j1": (
        (1e-08, 5e-09),
        (0.001, 0.0004999999375000026),
        (0.5, 0.2422684576748739),
        (1.0, 0.4400505857449335),
        (2.0, 0.5767248077568734),
        (3.0, 0.3390589585259365),
        (3.8317059702075125, -6.149807356994906e-17),
        (4.9, -0.31469467101519066),
        (5.0, -0.32757913759146523),
        (5.1, -0.3370972020182318),
        (7.015586669815619, 2.825339409478929e-17),
        (7.5, 0.1352484275797055),
        (8.0, 0.23463634685391463),
        (8.1, 0.24760776698159287),
        (12.0, -0.2234471044906276),
        (30.0, -0.11875106261662294),
        (100.0, -0.07714535201411216),
        (700.0, 0.029489824084030333),
    ),
    "y0": (
        (1e-08, -11.80077387717953),
        (0.001, -4.471416611375923),
        (0.5, -0.44451873350670656),
        (1.0, 0.08825696421567696),
        (2.0, 0.5103756726497451),
        (2.404825557695773, 0.509924383448479),
        (3.0, 0.3768500100127904),
        (4.9, -0.2920545942440142),
        (5.0, -0.30851762524903376),
        (5.1, -0.3216024491248594),
        (5.520078110286311, -0.3389361307570227),
        (7.5, 0.11731328614820863),
        (8.0, 0.22352148938756622),
        (8.1, 0.23809132870223482),
        (12.0, -0.22523731263436145),
        (30.0, -0.11729573168666403),
        (100.0, -0.07724431336508315),
        (700.0, 0.02949430818089382),
    ),
    "y1": (
        (1e-08, -63661977.236758195),
        (0.001, -636.6221672311394),
        (0.5, -1.471472392670243),
        (1.0, -0.7812128213002887),
        (2.0, -0.10703243154093754),
        (3.0, 0.3246744247918),
        (3.8317059702075125, 0.41251739515882574),
        (4.9, 0.18124669204504856),
        (5.0, 0.14786314339122683),
        (5.1, 0.11373644197749973),
        (7.015586669815619, -0.30236208914434987),
        (7.5, -0.25912851048611624),
        (8.0, -0.1580604617312475),
        (8.1, -0.13314879595249593),
        (12.0, -0.05709921826089652),
        (30.0, 0.08442557066174723),
        (100.0, -0.020372312002759792),
        (700.0, 0.00630934142145256),
    ),
    "k0": (
        (1e-08, 18.536612259610777),
        (0.001, 7.023688800562382),
        (0.5, 0.9244190712276659),
        (1.0, 0.42102443824070834),
        (2.0, 0.11389387274953344),
        (3.0, 0.03473950438627925),
        (4.9, 0.004118936235515887),
        (5.0, 0.0036910983340425942),
        (5.1, 0.0033083102180174657),
        (7.5, 0.00024917761635611437),
        (8.0, 0.0001464707052228154),
        (8.1, 0.00013173427864935836),
        (12.0, 2.2008253973114916e-06),
        (30.0, 2.1324774964630563e-14),
        (100.0, 4.656628229175902e-45),
        (700.0, 4.669776431685377e-306),
    ),
    "k1": (
        (1e-08, 99999999.9999999),
        (0.001, 999.9962381560856),
        (0.5, 1.656441120003301),
        (1.0, 0.6019072301972346),
        (2.0, 0.13986588181652243),
        (3.0, 0.040156431128194184),
        (4.9, 0.0045211691772998365),
        (5.0, 0.004044613445452165),
        (5.1, 0.0036191814623178),
        (7.5, 0.0002652973901252895),
        (8.0, 0.00015536921180500115),
        (8.1, 0.00013964122894503082),
        (12.0, 2.290757464767188e-06),
        (30.0, 2.1677320018915495e-14),
        (100.0, 4.6798537356369095e-45),
        (700.0, 4.6731107967079664e-306),
    ),
    "k0e": (
        (1e-08, 18.536612444976903),
        (0.001, 7.030716002378251),
        (0.5, 1.5241093857739094),
        (1.0, 1.144463079806895),
        (2.0, 0.8415682150707714),
        (3.0, 0.6977615980438517),
        (4.9, 0.5531310396037262),
        (5.0, 0.547807564313519),
        (5.1, 0.5426353518989783),
        (7.5, 0.4505236991049157),
        (8.0, 0.4366230186015861),
        (8.1, 0.4339943754308567),
        (12.0, 0.3581948784890782),
        (30.0, 0.22788666561625373),
        (100.0, 0.12517562165912657),
        (700.0, 0.04736236945461357),
    ),
    "k1e": (
        (1e-08, 100000000.99999991),
        (0.001, 1000.9967345590684),
        (0.5, 2.731009708211786),
        (1.0, 1.6361534862632583),
        (2.0, 1.0334768470686886),
        (3.0, 0.8065634801287869),
        (4.9, 0.6071468127379163),
        (5.0, 0.6002738587883126),
        (5.1, 0.5936250463138127),
        (7.5, 0.4796689337910206),
        (8.0, 0.4631490928704961),
        (8.1, 0.4600435707528057),
        (12.0, 0.37283175336970986),
        (30.0, 0.2316541293777118),
        (100.0, 0.12579995047957854),
        (700.0, 0.04739618765349454),
    ),
}


def _f64(values: list[float], requires_grad: bool = False) -> lucid.Tensor:
    return lucid.tensor(values, dtype=lucid.float64, requires_grad=requires_grad)


def _close(got: np.ndarray, xs: list[float], want: list[float]) -> None:
    """Double precision: 1e-14 relative, or an absolute floor that grows with
    ``x`` — an oscillating function at ``x = 700`` carries the rounding of its
    phase ``x - π/4``, ~1e-16 · x, and at a zero only the absolute error means
    anything."""
    floor = 1e-15 * np.maximum(1.0, np.abs(np.asarray(xs)))
    np.testing.assert_array_less(
        np.abs(got - np.asarray(want)), 1e-14 * np.abs(np.asarray(want)) + floor
    )


@pytest.mark.parametrize("name", list(_FUNCS))
def test_float64_is_double_precision(name: str) -> None:
    fn, _ = _FUNCS[name]
    xs = [x for x, _ in _TRUTH[name]]
    want = [v for _, v in _TRUTH[name]]
    got = fn(_f64(xs))
    assert got.dtype == lucid.float64
    _close(got.numpy(), xs, want)
    # Symmetry carries the precision to the negative axis.
    if name in _EVEN | _ODD:
        sign = 1.0 if name in _EVEN else -1.0
        _close(fn(_f64([-x for x in xs])).numpy(), xs, [sign * v for v in want])


@pytest.mark.parametrize("name", ["j0", "j1"])
def test_zeros_are_resolved(name: str) -> None:
    fn, _ = _FUNCS[name]
    got = fn(_f64(list(_ZEROS[name]))).numpy()
    assert np.all(np.abs(got) < 1e-15)


def _devices() -> list[str]:
    return ["cpu", "metal"] if metal_available() else ["cpu"]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("name", list(_FUNCS))
def test_float32_path_is_float32_grade(name: str, device: str) -> None:
    fn, _ = _FUNCS[name]
    # The float32 range of each function: I0 / I1 overflow past ~88.
    table = [(x, v) for x, v in _TRUTH[name] if x <= 30.0]
    xs = [x for x, _ in table]
    want = np.asarray([v for _, v in table])
    got = fn(lucid.tensor(xs, device=device))
    assert got.dtype == lucid.float32
    np.testing.assert_allclose(got.to("cpu").numpy(), want, rtol=2e-6, atol=2e-6)
    if device == "metal":
        on_cpu = fn(lucid.tensor(xs)).numpy()
        np.testing.assert_allclose(got.to("cpu").numpy(), on_cpu, rtol=1e-5, atol=1e-6)


# name -> (derivative by identity, grid)
_POS = [1e-3, 0.7, 1.9, 2.0, 2.1, 4.9, 5.0, 5.1, 7.9, 8.0, 8.1, 20.0]
_ALL = _POS + [0.0, -1e-3, -0.7, -3.0, -5.1, -9.0]


def _over_x(fn: UnaryFn) -> UnaryFn:
    """``fn(x) / x`` for ``I1``, ``J1`` or ``i1e``, whose limit at 0 is 1/2."""

    def out(v: lucid.Tensor) -> lucid.Tensor:
        safe = lucid.where(v == 0.0, lucid.ones_like(v), v)
        return lucid.where(v == 0.0, lucid.full_like(v, 0.5), fn(v) / safe)

    return out


_DERIVATIVES: dict[str, tuple[UnaryFn, list[float]]] = {
    "i0": (S.i1, _ALL),
    "i0e": (lambda v: S.i1e(v) - lucid.sign(v) * S.i0e(v), _ALL),
    # I1' = I0 - I1/x: 1/2 at 0, which an odd form sign(x)·g(|x|) flattens to 0.
    "i1": (lambda v: lucid.i0(v) - _over_x(S.i1)(v), _ALL),
    "i1e": (
        lambda v: S.i0e(v) - _over_x(S.i1e)(v) - lucid.sign(v) * S.i1e(v),
        _ALL,
    ),
    "j0": (lambda v: -S.bessel_j1(v), _ALL),
    "j1": (lambda v: S.bessel_j0(v) - _over_x(S.bessel_j1)(v), _ALL),
    "y0": (lambda v: -S.bessel_y1(v), _POS),
    "y1": (lambda v: S.bessel_y0(v) - S.bessel_y1(v) / v, _POS),
    "k0": (lambda v: -S.modified_bessel_k1(v), _POS),
    "k1": (lambda v: -S.modified_bessel_k0(v) - S.modified_bessel_k1(v) / v, _POS),
    "k0e": (
        lambda v: S.scaled_modified_bessel_k0(v) - S.scaled_modified_bessel_k1(v),
        _POS,
    ),
    "k1e": (
        lambda v: (
            S.scaled_modified_bessel_k1(v)
            - S.scaled_modified_bessel_k0(v)
            - S.scaled_modified_bessel_k1(v) / v
        ),
        _POS,
    ),
}


@pytest.mark.parametrize("name", list(_DERIVATIVES))
def test_float64_gradient_matches_the_identity(name: str) -> None:
    fn, _ = _FUNCS[name]
    deriv, xs = _DERIVATIVES[name]
    x = _f64(xs, requires_grad=True)
    fn(x).sum().backward()
    want = deriv(_f64(xs)).numpy()
    np.testing.assert_allclose(x.grad.numpy(), want, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("device", _devices())
def test_float32_gradient_at_zero(device: str) -> None:
    """``J0'(0) = I0'(0) = 0`` and ``J1'(0) = I1'(0) = i1e'(0) = 1/2``.  The
    float32 ``J`` used to answer NaN there: its unused large-argument branch
    divided by ``|x|``."""
    slopes = {"j0": 0.0, "j1": 0.5, "i0": 0.0, "i1": 0.5, "i1e": 0.5}
    for name, slope in slopes.items():
        x = lucid.tensor([0.0, -0.0], device=device, requires_grad=True)
        _FUNCS[name][0](x).sum().backward()
        np.testing.assert_allclose(x.grad.to("cpu").numpy(), [slope, slope], atol=1e-6)


@pytest.mark.parametrize("beta", [0.5, 3.0, 5.0, 8.6, 14.0])
def test_kaiser_float64_peak_and_shape(beta: float) -> None:
    w = kaiser(9, beta, dtype=lucid.float64)
    assert w.dtype == lucid.float64
    # 1 to rounding, not exactly: the peak's numerator and the denominator are
    # the same ``i0(beta)`` evaluated in different-length tensors.
    assert abs(w.numpy()[4] - 1.0) <= 2.0**-52
    n = np.arange(9.0)
    want = [
        float(_mp_i0(beta * math.sqrt(1.0 - (2.0 * k / 8.0 - 1.0) ** 2)) / _mp_i0(beta)) for k in n
    ]
    np.testing.assert_allclose(w.numpy(), want, rtol=1e-14, atol=1e-16)


def _mp_i0(x: float) -> float:
    """``I0`` by its power series in double precision — exact enough here."""
    term, total, k = 1.0, 1.0, 0
    while term > 1e-18 * total:
        k += 1
        term *= (x / 2.0) ** 2 / (k * k)
        total += term
    return total


# ── against the reference ────────────────────────────────────────────────────

_EDGES = [-_INF, -2.0, -1e-6, -0.0, 0.0, 1e-6, _INF, _NAN]
# Where the reference is a few digits short: its J / Y rational functions on
# roughly [4, 20] drop the leading 1 of the denominator Cephes implies, so it
# is off by up to ~1e-5 relative there, and by 1.6e-7 at J0's second zero.
# The 50-digit table above pins Lucid's values; this only checks the two
# agree to the reference's error.
_REF_LOOSE = {"j0", "j1", "y0", "y1"}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_FUNCS))
def test_float64_matches_the_reference(name: str, ref: Any) -> None:
    fn, attr = _FUNCS[name]
    xs = [x for x, _ in _TRUTH[name]] + _EDGES
    got = fn(_f64(xs)).numpy()
    want = getattr(ref.special, attr)(ref.tensor(xs, dtype=ref.float64)).numpy()
    # Same edges: NaN where the reference has NaN, the same infinities.
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[np.isinf(want)], want[np.isinf(want)])
    finite = np.isfinite(want)
    rtol, atol = (2e-5, 5e-7) if name in _REF_LOOSE else (1e-13, 1e-15)
    np.testing.assert_allclose(got[finite], want[finite], rtol=rtol, atol=atol)


@pytest.mark.parity
@pytest.mark.parametrize("beta", [0.0, 5.0, 8.6, 40.0])
@pytest.mark.parametrize("sym", [True, False])
def test_kaiser_float64_matches_the_reference(beta: float, sym: bool, ref: Any) -> None:
    got = kaiser(64, beta, sym=sym, dtype=lucid.float64).numpy()
    want = ref.signal.windows.kaiser(64, beta=beta, sym=sym, dtype=ref.float64).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-16)
