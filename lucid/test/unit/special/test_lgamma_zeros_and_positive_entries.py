"""LCD-1049 / LCD-1058: ``lgamma`` is exactly 0 at its zeros, and the positive-domain entries.

``lucid/_ops/composite/_gamma.py`` evaluates the Lanczos approximation as a *difference* from its
own value at ``x = 1`` (``_Lanczos.difference``), so ``lgamma(1)`` and ``lgamma(2)`` are exactly 0
and the value near them is a few ulp *of the tiny value*: it was a sum of three numbers of size ~7
whose rounding (4.77e-7 in float32, 8.9e-16 in float64) was the whole answer there.  The same
evaluation serves two entries per function: the public one, which covers the whole real line, and
``_lgamma_positive`` / ``_digamma_positive`` / ``_polygamma_positive`` for a caller that guarantees
``x > 0`` (the distributions' concentrations and counts).

The oracle tables below are 60-digit values (mpmath, scratch only, pasted as literals) at inputs that
float32 holds exactly, so one table serves both dtypes.  The error unit is the dtype's epsilon,
measured against ``max(|true|, 1)``; near a zero, where the true value is tiny, it is relative to the
value itself.  Measured worst over a dense sweep before / after: float32 lgamma 15 / 3 on (0, 1e30],
24 / 13 on (-60, 0); float64 15 / 4; near a zero 1e7 / 8.  The bounds are the old worst cases or
better.

The engine-event counts are the structural guard on cost: they do not depend on the machine.
"""

import math
from collections.abc import Callable

import pytest

import lucid
from lucid._ops.composite._gamma import (
    _LANCZOS_F64,
    _LANCZOS_F64_G,
    _LANCZOS_F64_P,
    _digamma_positive,
    _Lanczos,
    _lgamma,
    _lgamma_positive,
    _polygamma_positive,
    _polygamma_value,
)
from lucid.test._fixtures.devices import device_dtype_params

_FLOATS = device_dtype_params([lucid.float32, lucid.float64])
_EPS = {lucid.float32: 2.0**-23, lucid.float64: 2.0**-52}
_HALVES = device_dtype_params([lucid.float16, lucid.bfloat16])

Unary = Callable[[lucid.Tensor], lucid.Tensor]

# (x, true value) ---------------------------------------------------------------------------------
# lgamma near its zeros
_NEAR_ZEROS: tuple[tuple[float, float], ...] = (
    (1.5, -0.12078223763524522235),
    (0.5, 0.57236494292470008707),
    (1.25, -0.098271836421813161464),
    (0.75, 0.20328095143129537148),
    (1.125, -0.060023184126039582931),
    (0.875, 0.085858707225334323502),
    (1.0625, -0.032957100293577819083),
    (0.9375, 0.039390901734582300658),
    (1.015625, -0.0088197097057330692049),
    (0.984375, 0.009221337197578781045),
    (1.00390625, -0.0022422226599611501448),
    (0.99609375, 0.0022673224879091198692),
    (1.0009765625, -0.0005629031799912046317),
    (0.9990234375, 0.00056447191185512338426),
    (1.000244140625, -0.00014087277616326635097),
    (0.999755859375, 0.00014097082187592237051),
    (1.0000152587890625, -8.8074205801979051941e-6),
    (0.9999847412109375, 8.8078035712554869805e-6),
    (1.0000009536743164, -5.5047500661488667909e-7),
    (0.9999990463256836, 5.504765026737052438e-7),
    (1.000000238418579, -1.3761889190915445568e-7),
    (0.9999997615814209, 1.3761898541283061595e-7),
    (2.5, 0.28468287047291915963),
    (1.5, -0.12078223763524522235),
    (2.25, 0.1248717148923965943),
    (1.75, -0.084401121020485555958),
    (2.125, 0.057759851530343871607),
    (1.875, -0.047672685399188299644),
    (2.0625, 0.027667521522857023497),
    (1.9375, -0.025147619402988871015),
    (2.015625, 0.006684476830232184946),
    (1.984375, -0.0065270197705603875625),
    (2.00390625, 0.0016564177556961728692),
    (1.99609375, -0.0016465768332272092231),
    (2.0009765625, 0.00041318279306425426426),
    (1.9990234375, -0.00041256773597148940171),
    (2.000244140625, 0.00010323805136409635819),
    (1.999755859375, -0.00010319961029799208618),
    (2.0000152587890625, 6.4512520681644922117e-6),
    (1.9999847412109375, -6.4511019077505914e-6),
    (2.0000009536743164, 4.0319885504430155482e-7),
    (1.9999990463256836, -4.0319826848018476345e-7),
    (2.000000238418579, 1.0079965877070313142e-7),
    (1.999999761581421, -1.0079962211044583196e-7),
)
# lgamma on the positive axis
_LGAMMA_POSITIVE: tuple[tuple[float, float], ...] = (
    (0.0009765625, 6.9309089024194618895),
    (0.0625, 2.7396316219462034186),
    (0.375, 0.86307398227064746241),
    (0.5, 0.57236494292470008707),
    (0.625, 0.36082949548894018118),
    (0.875, 0.085858707225334323502),
    (1.125, -0.060023184126039582931),
    (1.5, -0.12078223763524522235),
    (1.875, -0.047672685399188299644),
    (2.0625, 0.027667521522857023497),
    (2.5, 0.28468287047291915963),
    (3.3125, 1.0000614361194534421),
    (4.6875, 2.7184558819095180177),
    (7.75, 8.0254583963159838712),
    (12.3125, 18.269845001573754807),
    (25.5, 56.389167643719946744),
    (33.3125, 82.647356803509330153),
    (77.75, 259.47777340799031894),
    (150.5, 602.51395487058541195),
    (1234.5, 7550.5509010778948957),
    (98765.25, 1037079.7850760921421),
    (10000000.0, 151180949.36947391394),
    (1099511627776.0, 29385423763643.474007),
    (3.458764513820541e18, 1.4418704894320176141e20),
)
# digamma on the positive axis
_DIGAMMA_POSITIVE: tuple[tuple[float, float], ...] = (
    (0.0009765625, -1024.5756104293406219),
    (0.0625, -16.478853490060104367),
    (0.375, -2.7539990491451395758),
    (0.5, -1.9635100260214234794),
    (0.625, -1.4527087645765665672),
    (0.875, -0.80401707154769538232),
    (1.125, -0.3884926632958548678),
    (1.5, 0.036489973978576520559),
    (1.875, 0.33884007130944747482),
    (2.0625, 0.46232298052813092761),
    (2.5, 0.70315664064524318723),
    (3.3125, 1.0392316181273544209),
    (4.6875, 1.4344570350348912136),
    (7.75, 1.9817915626943455972),
    (12.3125, 2.4694565302453566355),
    (25.5, 3.2189424728839197665),
    (33.3125, 3.4908482339890603922),
    (77.75, 4.3470538977650382218),
    (150.5, 5.0106371459337046472),
    (1234.5, 7.1180162318279978433),
    (98765.25, 11.500496038702094556),
    (10000000.0, 16.118095600958318955),
    (1099511627776.0, 27.725887222397357629),
    (3.458764513820541e18, 42.687443122264828256),
    (1.0, -0.57721566490153286061),
    (2.0, 0.42278433509846713939),
    (1.4619140625, 0.00027276858695440814547),
)
_LGAMMA_NEGATIVE: tuple[tuple[float, float], ...] = (
    (-0.375, 1.341658748500666418),
    (-0.5, 1.2655121234846453965),
    (-1.5, 0.86004701537648101451),
    (-2.25, 0.55550154502064747059),
    (-7.75, -8.5818477635518666694),
    (-33.3125, -84.823999092931682639),
    (-150.5, -606.3831880689249422),
    (-1000.5, -5914.4377011168518766),
)


def _t(
    values: float | list[float], device: str, dtype: lucid.dtype, grad: bool = False
) -> lucid.Tensor:
    return lucid.tensor(values, dtype=dtype, device=device, requires_grad=grad)


def _units(got: float, true: float, dtype: lucid.dtype, relative: bool) -> float:
    """``|got - true|`` in epsilons of ``dtype``, of ``|true|`` or of ``max(|true|, 1)``."""
    scale = abs(true) if relative else max(abs(true), 1.0)
    return abs(got - true) / (_EPS[dtype] * scale)


def _worst(
    table: tuple[tuple[float, float], ...], values: list[float], dtype: lucid.dtype, relative: bool
) -> float:
    return max(_units(g, t, dtype, relative) for g, (_, t) in zip(values, table, strict=True))


_LGAMMA_ENTRIES = pytest.mark.parametrize(
    "entry", [lucid.lgamma, _lgamma_positive], ids=["public", "positive"]
)
_DIGAMMA_ENTRIES = pytest.mark.parametrize(
    "entry", [lucid.digamma, _digamma_positive], ids=["public", "positive"]
)


# ── exact zeros (LCD-1049) ──────────────────────────────────────────────────


@_LGAMMA_ENTRIES
@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_lgamma_is_exactly_zero_at_one_and_two(
    entry: Unary, device: str, dtype: lucid.dtype
) -> None:
    """``lgamma(1) = lgamma(2) = 0`` on every stream and dtype (it was 4.77e-7 / -8.9e-16)."""
    x = _t([1.0, 2.0, 1.5, 3.0, 0.25, 1.0, 2.0], device, dtype)
    got = entry(x).tolist()
    assert got[0] == got[1] == got[5] == got[6] == 0.0, got
    # a lone element takes the same path, and the elements around it are ordinary values
    assert entry(_t(1.0, device, dtype)).item() == 0.0
    assert entry(_t(2.0, device, dtype)).item() == 0.0
    assert _units(got[2], -0.12078223763524522, dtype, relative=False) <= 8.0
    assert _units(got[3], 0.6931471805599453, dtype, relative=False) <= 8.0


@_LGAMMA_ENTRIES
@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_lgamma_near_its_zeros_is_accurate_relative_to_its_value(
    entry: Unary, device: str, dtype: lucid.dtype
) -> None:
    """``1 +- 2^-k`` and ``2 +- 2^-k``: the error is a few ulp of the tiny value, not of 7."""
    xs = [x for x, _ in _NEAR_ZEROS]
    got = entry(_t(xs, device, dtype)).tolist()
    assert _worst(_NEAR_ZEROS, got, dtype, relative=True) <= 12.0


# ── accuracy against the oracle ─────────────────────────────────────────────


@_LGAMMA_ENTRIES
@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_lgamma_on_the_positive_axis(entry: Unary, device: str, dtype: lucid.dtype) -> None:
    """From 1e-3 to 1e18: at most 6 epsilons (the old worst was 15 in float32 and 13 in float64)."""
    xs = [x for x, _ in _LGAMMA_POSITIVE]
    got = entry(_t(xs, device, dtype)).tolist()
    assert _worst(_LGAMMA_POSITIVE, got, dtype, relative=False) <= 6.0


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_lgamma_on_the_negative_axis(device: str, dtype: lucid.dtype) -> None:
    """The reflection side: its own rounding stays, the series under it is the difference form (old worst 24)."""
    xs = [x for x, _ in _LGAMMA_NEGATIVE]
    got = lucid.lgamma(_t(xs, device, dtype)).tolist()
    assert _worst(_LGAMMA_NEGATIVE, got, dtype, relative=False) <= 16.0


@_DIGAMMA_ENTRIES
@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_digamma_on_the_positive_axis(entry: Unary, device: str, dtype: lucid.dtype) -> None:
    """Float32 runs a shift of 4 and four Bernoulli terms: no worse than the shift of 8 (5.3).

    Float64 keeps the shift of 8 and its 1e-14 truncation (25 epsilons), unchanged.
    """
    xs = [x for x, _ in _DIGAMMA_POSITIVE]
    got = entry(_t(xs, device, dtype)).tolist()
    bound = 6.0 if dtype == lucid.float32 else 32.0
    assert _worst(_DIGAMMA_POSITIVE, got, dtype, relative=False) <= bound


# ── the positive-domain entries ─────────────────────────────────────────────


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_the_two_entries_agree_bit_for_bit_from_one_half_up(
    device: str, dtype: lucid.dtype
) -> None:
    """Above 1/2 the public entry reads the same series at ``x``: the reflection is not in the value."""
    x = _t([0.5, 0.75, 1.0, 1.5, 2.0, 3.7, 12.5, 100.25, 1e6, 3e30], device, dtype)
    assert lucid.lgamma(x).tolist() == _lgamma_positive(x).tolist()
    assert lucid.digamma(x).tolist() == _digamma_positive(x).tolist()
    assert lucid.special.polygamma(1, x).tolist() == _polygamma_positive(1, x).tolist()


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_the_two_entries_agree_below_one_half(device: str, dtype: lucid.dtype) -> None:
    """Below 1/2 the public entry reflects and the positive one steps up: two forms, one value."""
    xs = [1e-3, 0.0625, 0.1, 0.25, 0.375, 0.4375, 0.49]
    x = _t(xs, device, dtype)
    for public, positive in ((lucid.lgamma, _lgamma_positive), (lucid.digamma, _digamma_positive)):
        for a, b in zip(public(x).tolist(), positive(x).tolist(), strict=True):
            assert _units(b, a, dtype, relative=False) <= 8.0, (a, b)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_the_positive_entries_are_each_others_slopes(device: str, dtype: lucid.dtype) -> None:
    """``d lgamma = digamma``, ``d digamma = trigamma``: the slope of an entry is the next order of it."""
    xs = [0.05, 0.3, 0.75, 1.0, 1.5, 2.0, 4.5, 20.0, 300.0]
    x = _t(xs, device, dtype, grad=True)
    (g,) = lucid.autograd.grad(_lgamma_positive(x).sum(), [x], create_graph=True)
    assert g.tolist() == _digamma_positive(x.detach()).tolist()
    (g2,) = lucid.autograd.grad(g.sum(), [x])
    assert g2.tolist() == _polygamma_positive(1, x.detach()).tolist()
    (h,) = lucid.autograd.grad(_digamma_positive(x).sum(), [x])
    assert h.tolist() == _polygamma_positive(1, x.detach()).tolist()
    # and the public slopes are the same numbers
    xp = _t(xs, device, dtype, grad=True)
    (gp,) = lucid.autograd.grad(lucid.lgamma(xp).sum(), [xp])
    for a, b in zip(gp.tolist(), g.tolist(), strict=True):
        assert _units(b, a, dtype, relative=False) <= 8.0, (a, b)


@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_the_positive_entries_at_the_pole_and_nan(device: str, dtype: lucid.dtype) -> None:
    """0 is the pole as for the public entries, and a NaN stays a NaN."""
    x = _t([0.0, float("nan")], device, dtype)
    lg, dg = _lgamma_positive(x).tolist(), _digamma_positive(x).tolist()
    assert lg[0] == float("inf") and lg[1] != lg[1]
    assert dg[0] == float("-inf") and dg[1] != dg[1]


@pytest.mark.parametrize(("device", "dtype"), _HALVES)
def test_the_positive_entries_widen_a_16_bit_input(device: str, dtype: lucid.dtype) -> None:
    """A 16-bit operand is evaluated in float32 and rounded back, as the public entries do."""
    x = _t([0.5, 1.0, 3.0, 10.0], device, dtype)
    for public, positive in ((lucid.lgamma, _lgamma_positive), (lucid.digamma, _digamma_positive)):
        out = positive(x)
        assert out.dtype == dtype
        assert out.tolist() == public(x).tolist()


def test_the_positive_entries_cast_an_integer_input(device: str) -> None:
    """An integer operand becomes the default float first, as for the public entries."""
    x = lucid.tensor([1, 2, 3, 10], dtype=lucid.int64, device=device)
    assert _lgamma_positive(x).tolist() == lucid.lgamma(x).tolist()
    assert _digamma_positive(x).tolist() == lucid.digamma(x).tolist()


# ── cost (engine events of one call; machine independent) ───────────────────


def _events(fn: Callable[[], object]) -> int:
    fn()
    with lucid.profiler.profile() as prof:
        fn()
    return len(list(prof.events()))


@pytest.mark.parametrize(
    ("name", "entry", "limit"),
    [
        ("lgamma", lucid.lgamma, 100),
        ("lgamma_positive", _lgamma_positive, 70),
        ("digamma", lucid.digamma, 90),
        ("digamma_positive", _digamma_positive, 40),
    ],
)
def test_the_forward_pass_stays_within_its_event_budget(
    name: str, entry: Unary, limit: int, device: str
) -> None:
    """One Lanczos sum, one series, a shift of 4 in float32: they were 143 / 112 events.

    The two Lanczos sums of the old ``lgamma`` (the direct one and the reflected one) are one now,
    run at ``x`` or at ``1 - x``; the positive entries drop the reflection, the pole test and the
    infinity guard.  A change that adds a pass shows here before it shows in a benchmark.
    """
    x = 0.1 + 5.0 * lucid.rand(256, device=device)
    assert _events(lambda: entry(x)) <= limit, name


@pytest.mark.parametrize(("public", "positive"), [(lucid.lgamma, _lgamma_positive)])
def test_forward_plus_backward_of_the_positive_entry_is_cheaper_than_the_public_one(
    public: Unary, positive: Unary, device: str
) -> None:
    """Forward + backward: the slope is the digamma of the same entry (95 events against 177)."""
    x = (0.1 + 5.0 * lucid.rand(256, device=device)).requires_grad_(True)

    def step(fn: Unary) -> int:
        return _events(lambda: fn(x).sum().backward())

    assert step(positive) <= 120
    assert step(positive) < step(public)


# ── what lucid.compile differentiates ───────────────────────────────────────


@pytest.mark.parametrize("positive", [False, True], ids=["public", "positive"])
@pytest.mark.parametrize(("device", "dtype"), _FLOATS)
def test_the_trace_forms_differentiate_to_the_slopes(
    positive: bool, device: str, dtype: lucid.dtype
) -> None:
    """A compile records the composite ``_trace_form`` and differentiates *it*, not ``backward``.

    So the plain ops of ``lgamma`` / ``digamma`` (the difference form, the ``where`` that picks
    the argument, the logs of 1 on the lanes that do not own them) must carry the slope: finite
    on every lane, and the digamma / trigamma to within the series' own error (float64's digamma
    plan is 25 epsilons, so a slope that is a derivative of it is no closer).
    """
    xs = [0.01, 0.25, 0.4999, 0.5, 0.75, 1.0, 1.25, 1.4999, 1.5, 1.75, 2.0, 2.5, 7.3, 33.0, 1e3]
    if not positive:
        xs += [-0.25, -0.5, -0.75, -1.5, -2.25, -7.25, -33.5]
    for value, slope in (
        (lambda v: _lgamma(v, positive), lucid.digamma(_t(xs, device, dtype))),
        (
            lambda v: _polygamma_value(0, v, positive),
            lucid.special.polygamma(1, _t(xs, device, dtype)),
        ),
    ):
        x = _t(xs, device, dtype, grad=True)
        (g,) = lucid.autograd.grad(value(x).sum(), [x])
        for got, want in zip(g.tolist(), slope.tolist(), strict=True):
            assert _units(got, want, dtype, relative=False) <= 32.0, (got, want)


# ── the Lanczos sets ────────────────────────────────────────────────────────


@pytest.mark.parametrize("count", [0, 1, 3, 7])
def test_a_lanczos_set_needs_an_even_number_of_terms(count: int) -> None:
    """The partial fractions are added in pairs: an odd count would drop its last term unseen."""
    with pytest.raises(ValueError, match="in pairs"):
        _Lanczos(5.0, (1.0, *[1.0 / (k + 1) for k in range(count)]))


def test_the_difference_form_is_the_lanczos_form_minus_its_value_at_one() -> None:
    """``F(z) - F(0)`` written out term by term in double precision, against the pairs' sum.

    Independent of the pairing: the plain partial fractions ``sum_k P_k / (z + k)`` of the same set.
    """
    p, t0 = _LANCZOS_F64_P, _LANCZOS_F64_G + 0.5

    def forward(z: float) -> float:
        s = p[0] + math.fsum(p[k] / (z + k) for k in range(1, len(p)))
        return (-z) + (z + 0.5) * math.log(z + t0) + math.log(s)

    zs = [-0.4, -0.1, 0.05, 0.3, 1.0, 2.5, 9.0]
    got = _LANCZOS_F64.difference(
        lucid.tensor(zs, dtype=lucid.float64), lucid.ones((), dtype=lucid.float64)
    ).tolist()
    for z, g in zip(zs, got, strict=True):
        want = forward(z) - forward(0.0)
        assert abs(g - want) <= 1e-13 * max(1.0, abs(want)), (z, g, want)
