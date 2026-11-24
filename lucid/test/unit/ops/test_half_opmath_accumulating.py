"""The accumulating ops compute a half tensor in float32 and round once (LCD-1087).

``addmm``, ``addbmm``, ``baddbmm``, ``addcmul``, ``addcdiv`` and ``lerp`` read the Python scalar of
a float16 / bfloat16 tensor as a ``float`` and keep the product, the scale and the sum in float32
until the answer is stored; the reference's kernels do the same ("op math").  Lucid composed them
from engine ops and rounded after every one: the scalar to the half type (``alpha=300.7`` on
bfloat16 was 300, ``weight=70000`` on float16 was infinity, and ``inf * 0`` is NaN), the product,
the scaled product and the sum.  ``lucid._ops.composite._opmath._in_float32_as`` now widens the
operands and narrows the answer, for all six, and:

* the answer is the float32 answer rounded once, on both devices, so it is within one half ulp of
  the reference's (of the largest term when the terms cancel);
* a gradient reaches each operand in the operand's own dtype, and is the float32 gradient rounded
  once;
* an operand set the op refuses (a float32 beside half) is still refused, not widened into an
  answer;
* autocast and ``compile`` take the same path, and so do the method and in-place spellings;
* a float32 or float64 tensor takes the engine's ops as before, bit for bit.

The one place the two may differ by more than a rounding is ``addbmm``: the reference adds the
batches one ``addmm`` at a time, rounding after each.
"""

import math
import random
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any, NamedTuple

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._ops.composite._opmath import _in_float32, _in_float32_as
from lucid.amp import autocast
from lucid.test._fixtures.devices import devices_supporting

pytestmark = pytest.mark.parity

_HALF = ("float16", "bfloat16")
_MANTISSA = {"float16": 10, "bfloat16": 7}
_MIN_EXPONENT = {"float16": -14, "bfloat16": -126}
#: The half ulps, of the largest term, two correct answers may differ by.  The reference's
#: ``addbmm`` rounds after each of its three batches, so it may be a rounding or two further.
_ULPS = {"addbmm": 2.0}
#: Scalars a half type cannot hold exactly (``0.3``, ``300.7``), one past float16's range
#: (``70000``), one far below it, and the default.
_SCALARS = (0.3, 2.5, 300.7, 70000.0, 1e-3, 1.0)
_BETAS = (1.0, 0.7)


class _Case(NamedTuple):
    """One op: its operand shapes, its call, and the terms an element of its answer adds up."""

    shapes: tuple[tuple[int, ...], ...]
    call: Callable[[Any, list[Any], float, float], Any]
    #: ``(*operands of one element, scalar, beta) -> terms`` of that element, to size the
    #: tolerance of a sum that cancels.  The matmul family is checked on positive operands, where
    #: no term is larger than the sum.
    terms: Callable[..., tuple[float, ...]] | None = None
    #: The operand indices that stay positive on signed data (a divisor).
    positive: tuple[int, ...] = ()


_CASES: dict[str, _Case] = {
    "addmm": _Case(((4, 5), (4, 6), (6, 5)), lambda lib, ts, s, b: lib.addmm(*ts, alpha=s, beta=b)),
    "baddbmm": _Case(
        ((3, 4, 5), (3, 4, 6), (3, 6, 5)), lambda lib, ts, s, b: lib.baddbmm(*ts, alpha=s, beta=b)
    ),
    "addbmm": _Case(
        ((4, 5), (3, 4, 6), (3, 6, 5)), lambda lib, ts, s, b: lib.addbmm(*ts, alpha=s, beta=b)
    ),
    "addcmul": _Case(
        ((9,), (9,), (9,)),
        lambda lib, ts, s, b: lib.addcmul(*ts, value=s),
        lambda i, a, c, s, b: (i, s * a * c),
    ),
    "addcdiv": _Case(
        ((9,), (9,), (9,)),
        lambda lib, ts, s, b: lib.addcdiv(*ts, value=s),
        lambda i, a, c, s, b: (i, s * a / c),
        positive=(2,),
    ),
    "lerp": _Case(
        ((9,), (9,)), lambda lib, ts, s, b: lib.lerp(*ts, s), lambda i, e, s, b: (i, s * (e - i))
    ),
}
_PRODUCTS = ("addmm", "baddbmm", "addbmm")


def _nested(rng: random.Random, shape: tuple[int, ...], signed: bool) -> Any:
    if not shape:
        return rng.uniform(-2.0, 2.0) if signed else rng.uniform(0.5, 1.5)
    return [_nested(rng, shape[1:], signed) for _ in range(shape[0])]


def _flat(values: Any) -> Iterator[float]:
    if isinstance(values, list):
        for value in values:
            yield from _flat(value)
    else:
        yield values


def _operands(case: _Case, seed: int, signed: bool) -> list[Any]:
    rng = random.Random(seed)
    return [
        _nested(rng, shape, signed and i not in case.positive)
        for i, shape in enumerate(case.shapes)
    ]


def _build(lib: Any, dtype: str, device: str | None, values: list[Any]) -> list[Any]:
    kwargs = {} if device is None else {"device": device}
    return [lib.tensor(v, dtype=getattr(lib, dtype), **kwargs) for v in values]


def _ulp(value: float, dtype: str) -> float:
    """The spacing of ``dtype`` at ``value`` (its subnormal spacing below the smallest normal)."""
    exponent = _MIN_EXPONENT[dtype]
    if value != 0:
        exponent = max(math.frexp(value)[1] - 1, exponent)
    return 2.0 ** (exponent - _MANTISSA[dtype])


def _same(got: list[float], want: list[float]) -> bool:
    """Equal lists, a NaN equal to a NaN."""
    return all(g == w or (math.isnan(g) and math.isnan(w)) for g, w in zip(got, want, strict=True))


def _cells(names: tuple[str, ...], scalars: tuple[float, ...] = _SCALARS) -> list[Any]:
    return [
        pytest.param(
            name, dtype, device, scalar, beta, id=f"{name}-{dtype}-{device}-{scalar}-{beta}"
        )
        for name in names
        for dtype in _HALF
        for device in devices_supporting(getattr(lucid, dtype))
        for scalar in scalars
        for beta in (_BETAS if name in _PRODUCTS else (1.0,))
    ]


# ── the reference's answer, within a rounding ────────────────────────────────────────────────


@pytest.mark.parametrize(("name", "dtype", "device", "scalar", "beta"), _cells(tuple(_CASES)))
def test_a_half_answer_is_within_a_rounding_of_the_reference(
    name: str, dtype: str, device: str, scalar: float, beta: float, ref: ModuleType
) -> None:
    case = _CASES[name]
    for signed in (False, True) if case.terms else (False,):
        values = _operands(case, 7, signed)
        theirs = _build(ref, dtype, None, values)
        got = list(
            _flat(case.call(lucid, _build(lucid, dtype, device, values), scalar, beta).tolist())
        )
        want = list(_flat(case.call(ref, theirs, scalar, beta).float().tolist()))
        # what the half type holds of each operand: the terms are summed from these
        held = [list(_flat(t.float().tolist())) for t in theirs]
        rows = zip(*held, strict=True) if case.terms else [()] * len(want)
        for g, w, row in zip(got, want, rows, strict=True):
            if not math.isfinite(w):
                assert g == w or (math.isnan(g) and math.isnan(w)), f"{signed}: {g} vs {w}"
                continue
            terms = case.terms(*row, scalar, beta) if case.terms else ()
            tolerance = _ULPS.get(name, 1.0) * _ulp(max(abs(w), *map(abs, terms), 0.0), dtype)
            assert abs(g - w) <= tolerance, f"signed={signed}: {g} vs {w}"


@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_lerp_of_float16_with_a_weight_past_the_range_is_not_nan(device: str) -> None:
    ones = lucid.ones(2, dtype=lucid.float16, device=device)
    assert lucid.lerp(ones, ones, 70000.0).tolist() == [1.0, 1.0]


@pytest.mark.parametrize("device", devices_supporting(lucid.bfloat16))
def test_addmm_of_bfloat16_reads_alpha_as_a_float(device: str) -> None:
    ones = lucid.ones((2, 2), dtype=lucid.bfloat16, device=device)
    # 3 + 300.7 * 2 = 604.4, rounded once; 300 for 300.7 and a rounded product gave 600
    assert lucid.addmm(ones * 3, ones, ones, alpha=300.7).tolist() == [[604.0, 604.0]] * 2


# ── the float32 answer rounded once, exactly ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "dtype", "device", "scalar", "beta"), _cells(tuple(_CASES), (0.3, 300.7, 70000.0))
)
def test_a_half_answer_is_the_float32_answer_rounded_once(
    name: str, dtype: str, device: str, scalar: float, beta: float
) -> None:
    case = _CASES[name]
    half = _build(lucid, dtype, device, _operands(case, 11, signed=True))
    got = case.call(lucid, half, scalar, beta)
    want = case.call(lucid, [t.to(lucid.float32) for t in half], scalar, beta)
    assert got.dtype is getattr(lucid, dtype)
    assert _same(list(_flat(got.tolist())), list(_flat(want.to(got.dtype).tolist())))


# ── gradients ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "dtype", "device"),
    [
        pytest.param(name, dtype, device, id=f"{name}-{dtype}-{device}")
        for name in _CASES
        for dtype in _HALF
        for device in devices_supporting(getattr(lucid, dtype))
    ],
)
def test_a_gradient_is_the_float32_gradient_in_the_operand_dtype(
    name: str, dtype: str, device: str, ref: ModuleType
) -> None:
    case = _CASES[name]
    values = _operands(case, 3, signed=False)
    half = [t.requires_grad_() for t in _build(lucid, dtype, device, values)]
    case.call(lucid, half, 0.7, 0.9).sum().backward()
    # the reference's float32 gradient on what the half type holds of each operand
    wide = [t.float().requires_grad_() for t in _build(ref, dtype, None, values)]
    case.call(ref, wide, 0.7, 0.9).sum().backward()
    for narrow, reference in zip(half, wide, strict=True):
        assert narrow.grad is not None
        assert narrow.grad.dtype is getattr(lucid, dtype)
        want = list(_flat(reference.grad.tolist()))
        for g, w in zip(_flat(narrow.grad.tolist()), want, strict=True):
            assert abs(g - w) <= 1.0 * _ulp(w, dtype), f"{g} vs {w}"


# ── lerp with a weight tensor ────────────────────────────────────────────────────────────────

_WEIGHTS = ("shaped like the operands", "0-d float32")


def _weight_values(kind: str) -> Any:
    """A weight of the operands' shape (some past 1, which ``lerp`` extrapolates) or a 0-d one."""
    return _nested(random.Random(23), (9,), False) if kind == _WEIGHTS[0] else 0.3


def _weight_dtype(kind: str, dtype: str) -> str:
    return dtype if kind == _WEIGHTS[0] else "float32"


@pytest.mark.parametrize("kind", _WEIGHTS)
@pytest.mark.parametrize("dtype", _HALF)
@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_lerp_with_a_weight_tensor_is_the_float32_answer_rounded_once(
    kind: str, dtype: str, device: str, ref: ModuleType
) -> None:
    rng = random.Random(29)
    start, stop = (_nested(rng, (9,), True) for _ in range(2))
    weight = _weight_values(kind)
    half = getattr(lucid, dtype)
    ours = [*_build(lucid, dtype, device, [start, stop])]
    ours.append(
        lucid.tensor(weight, dtype=getattr(lucid, _weight_dtype(kind, dtype)), device=device)
    )
    got = lucid.lerp(*ours)
    assert got.dtype is half
    # the weight is read at the dtype the three promote to, then everything is float32
    held = [t.to(lucid.float32) for t in ours[:2]] + [ours[2].to(half).to(lucid.float32)]
    assert _same(
        list(_flat(got.tolist())), list(_flat(lucid.lerp(*held).to(half).tolist()))
    ), "not the float32 answer rounded once"
    # and within a rounding of the reference's, which takes the same weight tensor
    theirs = [*_build(ref, dtype, None, [start, stop])]
    theirs.append(ref.tensor(weight, dtype=getattr(ref, _weight_dtype(kind, dtype))))
    want = list(_flat(ref.lerp(*theirs).float().tolist()))
    held_start, held_stop = (list(_flat(t.float().tolist())) for t in theirs[:2])
    held_weight = list(_flat(theirs[2].to(getattr(ref, dtype)).float().tolist()))
    held_weight *= len(want) if len(held_weight) == 1 else 1
    for g, w, i, e, k in zip(
        _flat(got.tolist()), want, held_start, held_stop, held_weight, strict=True
    ):
        scale = max(abs(w), abs(i), abs(k * (e - i)))
        assert abs(g - w) <= _ulp(scale, dtype), f"{g} vs {w}"


@pytest.mark.parametrize("kind", _WEIGHTS)
@pytest.mark.parametrize("dtype", _HALF)
@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_lerp_with_a_weight_tensor_differentiates_in_each_operands_dtype(
    kind: str, dtype: str, device: str, ref: ModuleType
) -> None:
    rng = random.Random(31)
    start, stop = (_nested(rng, (9,), True) for _ in range(2))
    weight = _weight_values(kind)
    weight_dtype = _weight_dtype(kind, dtype)
    ours = [t.requires_grad_() for t in _build(lucid, dtype, device, [start, stop])]
    ours.append(
        lucid.tensor(weight, dtype=getattr(lucid, weight_dtype), device=device).requires_grad_()
    )
    lucid.lerp(*ours).sum().backward()
    # the reference's float32 gradient at what the half type holds of each operand
    wide = [t.float().requires_grad_() for t in _build(ref, dtype, None, [start, stop])]
    held = ref.tensor(weight, dtype=getattr(ref, dtype)).float()
    wide.append(held.requires_grad_())
    ref.lerp(*wide).sum().backward()
    for narrow, reference, name in zip(ours, wide, ("input", "end", "weight"), strict=True):
        assert narrow.grad is not None, name
        assert narrow.grad.dtype is narrow.dtype, name
        assert narrow.grad.shape == narrow.shape, name
        for g, w in zip(_flat(narrow.grad.tolist()), _flat(reference.grad.tolist()), strict=True):
            assert abs(g - w) <= _ulp(w, dtype), f"{name}: {g} vs {w}"


# ── the method and in-place twins take the same path ─────────────────────────────────────────

_SPELLINGS: dict[str, tuple[Callable[[Any, list[Any]], Any], Callable[[Any, list[Any]], Any]]] = {
    "addcmul": (
        lambda x, ts: x.addcmul(*ts, value=300.7),
        lambda x, ts: x.addcmul_(*ts, value=300.7),
    ),
    "addcdiv": (
        lambda x, ts: x.addcdiv(*ts, value=300.7),
        lambda x, ts: x.addcdiv_(*ts, value=300.7),
    ),
    "lerp": (lambda x, ts: x.lerp(*ts, 0.3), lambda x, ts: x.lerp_(*ts, 0.3)),
    "addmm": (lambda x, ts: x.addmm(*ts, alpha=300.7), lambda x, ts: x.addmm_(*ts, alpha=300.7)),
    "baddbmm": (
        lambda x, ts: x.baddbmm(*ts, alpha=300.7),
        lambda x, ts: x.baddbmm_(*ts, alpha=300.7),
    ),
    "addbmm": (lambda x, ts: x.addbmm(*ts, alpha=300.7), lambda x, ts: x.addbmm_(*ts, alpha=300.7)),
}


@pytest.mark.parametrize("name", _SPELLINGS)
@pytest.mark.parametrize("dtype", _HALF)
@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_the_method_and_in_place_forms_answer_as_the_function_does(
    name: str, dtype: str, device: str
) -> None:
    case = _CASES[name]
    values = _operands(case, 5, signed=False)
    scalar = 0.3 if name == "lerp" else 300.7
    free = case.call(lucid, _build(lucid, dtype, device, values), scalar, 1.0)
    method, in_place = _SPELLINGS[name]
    first, *rest = _build(lucid, dtype, device, values)
    assert method(first, rest).tolist() == free.tolist()
    first, *rest = _build(lucid, dtype, device, values)
    in_place(first, rest)
    assert first.tolist() == free.tolist()


# ── what is refused is still refused, what is not half is not touched ────────────────────────


@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_a_half_operand_beside_a_float32_one_is_refused_not_widened(device: str) -> None:
    half = lucid.ones((2, 2), dtype=lucid.float16, device=device)
    wide = lucid.ones((2, 2), device=device)
    zero_d = lucid.tensor(1.0, device=device)
    for call in (
        lambda: lucid.addmm(wide, half, half),
        lambda: lucid.addmm(half, wide, wide),
        lambda: lucid.addmm(zero_d, half, half),
        lambda: lucid.baddbmm(wide[None], half[None], half[None]),
        lambda: lucid.addbmm(wide, half[None], half[None]),
    ):
        with pytest.raises(_C_engine.DtypeMismatch):
            call()


@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_a_float16_beside_a_bfloat16_promotes_and_is_rounded_to_neither(device: str) -> None:
    a = lucid.ones(3, dtype=lucid.float16, device=device)
    b = lucid.ones(3, dtype=lucid.bfloat16, device=device)
    assert lucid.addcmul(a, b, b, value=0.3).dtype is lucid.float32
    assert lucid.addcdiv(a, b, b, value=0.3).dtype is lucid.float32


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_float32_and_float64_take_the_engine_ops_as_before(dtype: str, device: str) -> None:
    if dtype == "float64" and device != "cpu":
        pytest.skip("Metal has no float64")
    rng = random.Random(0)

    def make(*shape: int) -> Any:
        return lucid.tensor(_nested(rng, shape, True), dtype=getattr(lucid, dtype), device=device)

    m1, m2, bias = make(4, 6), make(6, 5), make(4, 5)
    assert (
        lucid.addmm(bias, m1, m2, alpha=300.7, beta=0.7).tolist()
        == (bias * 0.7 + lucid.matmul(m1, m2) * 300.7).tolist()
    )
    a, b, c = make(9), make(9), make(9)
    assert lucid.addcmul(a, b, c, value=0.3).tolist() == (a + (b * c) * 0.3).tolist()
    assert lucid.addcdiv(a, b, c, value=0.3).tolist() == (a + (b / c) * 0.3).tolist()
    assert lucid.lerp(a, b, 0.3).tolist() == (a + (b - a) * 0.3).tolist()


def test_operands_widen_by_their_common_dtype_not_the_first_ones() -> None:
    """``_in_float32`` took the first operand's dtype for three operands or more."""
    half, wide = lucid.ones(3, dtype=lucid.float16), lucid.ones(3)
    seen: list[list[Any]] = []

    def record(*ts: Any) -> Any:
        seen.append([t.dtype for t in ts])
        return ts[0]

    assert _in_float32(record, half, wide, half).dtype is lucid.float16
    assert seen[-1] == [lucid.float16, lucid.float32, lucid.float16]
    assert _in_float32(record, half, half, half).dtype is lucid.float16
    assert seen[-1] == [lucid.float32] * 3
    assert _in_float32_as(_C_engine.F32, record, half).dtype is lucid.float16
    assert seen[-1] == [lucid.float16]


# ── autocast and compile ─────────────────────────────────────────────────────────────────────


def _autocast_dtype(device: str) -> Any:
    return lucid.float16 if device == "metal" else lucid.bfloat16


@pytest.mark.parametrize("name", _PRODUCTS)
@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_autocast_lowers_a_product_and_rounds_it_once(name: str, device: str) -> None:
    half = _autocast_dtype(device)
    case = _CASES[name]
    wide = _build(lucid, "float32", device, _operands(case, 13, signed=False))
    with autocast(device_type=device, dtype=half):
        got = case.call(lucid, wide, 300.7, 0.7)
    held = [t.to(half).to(lucid.float32) for t in wide]
    assert got.dtype is half
    assert got.tolist() == case.call(lucid, held, 300.7, 0.7).to(half).tolist()


@pytest.mark.parametrize("name", list(_CASES))
@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_autocast_does_not_narrow_the_widened_operands_again(name: str, device: str) -> None:
    """Half operands inside an autocast scope answer as they do outside it: the float32 path
    the call takes is not cast back to the autocast dtype by the engine ops it runs."""
    half = _autocast_dtype(device)
    case = _CASES[name]
    held = _build(lucid, "float32", device, _operands(case, 19, signed=True))
    narrow = [t.to(half) for t in held]
    outside = case.call(lucid, narrow, 300.7, 0.7)
    with autocast(device_type=device, dtype=half):
        inside = case.call(lucid, narrow, 300.7, 0.7)
    assert inside.dtype is outside.dtype is half
    assert _same(list(_flat(inside.tolist())), list(_flat(outside.tolist())))


@pytest.mark.parametrize("name", ["addcmul", "addcdiv"])
@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_autocast_keeps_a_float32_set_in_float32(name: str, device: str) -> None:
    case = _CASES[name]
    wide = _build(lucid, "float32", device, _operands(case, 13, signed=False))
    with autocast(device_type=device, dtype=_autocast_dtype(device)):
        assert case.call(lucid, wide, 300.7, 0.7).dtype is lucid.float32


@pytest.mark.parametrize("name", list(_CASES))
@pytest.mark.parametrize("dtype", _HALF)
def test_a_compiled_trace_answers_as_eager_does(
    name: str, dtype: str, device_gpu_only: str
) -> None:
    case = _CASES[name]
    tensors = _build(lucid, dtype, device_gpu_only, _operands(case, 17, signed=False))

    def run(*ts: Any) -> Any:
        return case.call(lucid, list(ts), 300.7, 0.7)

    compiled = lucid.compile(run)
    compiled(*tensors)
    assert compiled(*tensors).tolist() == run(*tensors).tolist()
