"""``lerp`` takes the reference's two forms by the weight, and its one-sided gradient (LCD-1101).

The reference's CPU kernel computes ``start + weight * (end - start)`` for ``|weight| < 0.5`` and
``end - (end - start) * (1 - weight)`` otherwise.  ``lerp`` used one formula for every weight,
so for ``inf`` / ``NaN`` operands it answered what the other form does not (``lerp(0, inf, 0.5)``
was ``inf``, the reference's ``NaN``), and ``lerp(1, 1e-8, 1)`` was ``0`` rather than ``1e-8``:
a weight of ``1`` must give ``end``, exactly.  The gradient is ``grad * (1 - weight)``,
``grad * weight`` and ``grad * (end - start)`` whichever form the value took.

Lucid's Metal answers as the reference's *CPU* kernel does, not its ``mps`` one (which has a
single formula and misses ``lerp(a, b, 1) == b``): every comparison here is against the reference
on the CPU, so CPU and Metal agree with each other.  The reference's CPU kernel fuses the
multiply-add, so a finite answer may be a rounding or two away from Lucid's two-step one: finite
answers are held to two ulps of the largest term they add up, ``inf`` / ``NaN`` ones and the
ends of the segment exactly.  The two zeros compare equal, as the reference's scalar tail and its
vector body disagree on the sign of a zero (``lerp(0, -0, 1)``); the sign Lucid keeps is the end's,
which a test of its own pins.
"""

import itertools
import math
import random
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest

import lucid
from lucid.test._fixtures.devices import devices_supporting

pytestmark = pytest.mark.parity

_INF, _NAN = math.inf, math.nan
_DTYPES = ("float32", "float64", "float16", "bfloat16")
_MANTISSA = {"float64": 52, "float32": 23, "float16": 10, "bfloat16": 7}
_MIN_EXPONENT = {"float64": -1022, "float32": -126, "float16": -14, "bfloat16": -126}
_OPERANDS = (0.0, -0.0, 1.5, -2.0, _INF, -_INF, _NAN)
#: Zero, below, at and above the split, the end, past it, negative on both sides of ``-0.5``, and
#: the two non-finite weights.
_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 2.5, -0.25, -0.5, -1.5, _INF, _NAN)
#: How often each pair repeats: long enough for the reference to take its vectorised kernel.
_TILE = 64
_KINDS = ("number", "tensor", "0-d tensor")
#: The gradients upstream of the answer: the identity, and two that scale and flip it.
_UPSTREAM = (1.0, 0.7, -1.3)


def _table() -> tuple[list[float], list[float]]:
    """Every pair of operands, each repeated ``_TILE`` times."""
    pairs = itertools.product(_OPERANDS, _OPERANDS)
    start, end = zip(*(p for p in pairs for _ in range(_TILE)), strict=True)
    return list(start), list(end)


def _weight(lib: Any, kind: str, value: float, like: Any, **where: Any) -> Any:
    """``value`` as the weight of ``kind``: a number, or a tensor of ``like``'s shape or 0-d."""
    if kind == "number":
        return value
    dtype = like.dtype
    if kind == "tensor":
        return lib.full(tuple(like.shape), value, dtype=dtype, **where)
    return lib.tensor(value, dtype=dtype, **where)


def _build(lib: Any, dtype: str, values: list[float], **where: Any) -> Any:
    return lib.tensor(values, dtype=getattr(lib, dtype), **where)


def _ulp(value: float, dtype: str) -> float:
    """The spacing of ``dtype`` at ``value`` (its subnormal spacing below the smallest normal)."""
    exponent = _MIN_EXPONENT[dtype]
    if value != 0:
        exponent = max(math.frexp(value)[1] - 1, exponent)
    return 2.0 ** (exponent - _MANTISSA[dtype])


def _flat(tensor: Any) -> list[float]:
    return list(tensor.float().tolist()) if tensor.ndim else [tensor.float().item()]


def _agrees(got: float, want: float) -> bool:
    """Equal, a ``NaN`` equal to a ``NaN``, and the two zeros equal."""
    return got == want or (math.isnan(got) and math.isnan(want))


def _devices(dtype: str) -> list[str]:
    return devices_supporting(getattr(lucid, dtype))


def _cells(dtypes: tuple[str, ...] = _DTYPES, kinds: tuple[str, ...] = _KINDS) -> Iterator[Any]:
    for dtype in dtypes:
        for device in _devices(dtype):
            for kind in kinds:
                yield pytest.param(dtype, device, kind, id=f"{dtype}-{device}-{kind}")


# ── the special-value table, against the reference ───────────────────────────────────────────


@pytest.mark.parametrize(("dtype", "device", "kind"), list(_cells()))
def test_the_special_value_table_is_the_references(
    dtype: str, device: str, kind: str, ref: ModuleType
) -> None:
    start, end = _table()
    ours = [_build(lucid, dtype, v, device=device) for v in (start, end)]
    theirs = [_build(ref, dtype, v) for v in (start, end)]
    for weight in _WEIGHTS:
        got = lucid.lerp(*ours, _weight(lucid, kind, weight, ours[0], device=device))
        want = ref.lerp(*theirs, _weight(ref, kind, weight, theirs[0]))
        assert got.dtype is getattr(lucid, dtype)
        assert got.device.type == device
        for i, (g, w) in enumerate(zip(_flat(got), _flat(want), strict=True)):
            assert _agrees(g, w), f"lerp({start[i]}, {end[i]}, {weight}): {g}, the reference {w}"


@pytest.mark.parametrize(("dtype", "device", "kind"), list(_cells(("float32",), _KINDS[:2])))
def test_each_weight_of_a_single_element_answers_the_references(
    dtype: str, device: str, kind: str, ref: ModuleType
) -> None:
    """The reference's scalar tail takes the same form as its vector body: one cell at a time."""
    for a, b, weight in itertools.product(_OPERANDS, _OPERANDS, _WEIGHTS):
        ours = [_build(lucid, dtype, [v], device=device) for v in (a, b)]
        theirs = [_build(ref, dtype, [v]) for v in (a, b)]
        got = lucid.lerp(*ours, _weight(lucid, kind, weight, ours[0], device=device))
        want = ref.lerp(*theirs, _weight(ref, kind, weight, theirs[0]))
        assert _agrees(got.item(), want.item()), f"lerp({a}, {b}, {weight})"


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("kind", _KINDS)
def test_an_infinite_end_is_nan_in_the_end_form_and_infinite_in_the_start_form(
    dtype: str, kind: str
) -> None:
    """The case that named the defect: ``end - (end - start) * (1 - w)`` is ``inf - inf``."""
    for device in _devices(dtype):
        start = lucid.tensor([0.0], dtype=getattr(lucid, dtype), device=device)
        end = lucid.tensor([_INF], dtype=getattr(lucid, dtype), device=device)
        for weight in (0.5, 0.75, 1.0, -0.5, -1.5):
            assert math.isnan(
                lucid.lerp(start, end, _weight(lucid, kind, weight, start, device=device)).item()
            ), f"{device}: lerp(0, inf, {weight})"
        for weight in (0.25, -0.25):  # the other form: start + weight * inf
            got = lucid.lerp(start, end, _weight(lucid, kind, weight, start, device=device)).item()
            assert got == math.copysign(_INF, weight), f"{device}: lerp(0, inf, {weight})"


# ── finite answers: within an ulp of the fused kernel, and exact at the ends ─────────────────


def _finite_operands(dtype: str) -> tuple[list[float], list[float]]:
    rng = random.Random(41)
    start = [rng.uniform(-4.0, 4.0) for _ in range(257)]
    end = [rng.uniform(-4.0, 4.0) for _ in range(257)]
    # a start that dwarfs the end (the single formula lost the end at ``weight = 1``) and the
    # reverse, kept inside what the dtype holds
    small, large = (1e-4, 7.1e3) if dtype in ("float16", "bfloat16") else (1e-8, 7.1e5)
    return start + [1.0, small, large, -3.3], end + [small, 1.0, -4.4e-3, 2.9]


@pytest.mark.parametrize(("dtype", "device", "kind"), list(_cells()))
def test_finite_answers_are_within_two_ulps_of_the_reference(
    dtype: str, device: str, kind: str, ref: ModuleType
) -> None:
    start, end = _finite_operands(dtype)
    ours = [_build(lucid, dtype, v, device=device) for v in (start, end)]
    theirs = [_build(ref, dtype, v) for v in (start, end)]
    held = [_flat(t) for t in theirs]
    # either side of the split, which a float32 weight can sit one step away from
    weights = (0.3, 0.7, 0.4999999, 0.5, 0.5000001, -0.4999999, -0.5, -0.8, 1.7, 1e-3)
    for weight in weights:
        got = _flat(lucid.lerp(*ours, _weight(lucid, kind, weight, ours[0], device=device)))
        want = _flat(ref.lerp(*theirs, _weight(ref, kind, weight, theirs[0])))
        for g, w, a, b in zip(got, want, *held, strict=True):
            # the largest term either form adds up: the unfused sum is a rounding or two off the fused one
            terms = (abs(weight * (b - a)), abs((weight - 1) * (b - a)))
            scale = max(abs(w), abs(a), abs(b), *terms)
            assert abs(g - w) <= 2 * _ulp(scale, dtype), f"lerp({a}, {b}, {weight}): {g} vs {w}"


@pytest.mark.parametrize(("dtype", "device", "kind"), list(_cells()))
def test_a_weight_of_one_gives_the_end_and_zero_gives_the_start_exactly(
    dtype: str, device: str, kind: str
) -> None:
    start, end = _finite_operands(dtype)
    a, b = (_build(lucid, dtype, v, device=device) for v in (start, end))
    for weight, want in ((1.0, b), (0.0, a)):
        got = lucid.lerp(a, b, _weight(lucid, kind, weight, a, device=device))
        assert got.tolist() == want.tolist(), f"weight {weight}"
        assert a.lerp(b, _weight(lucid, kind, weight, a, device=device)).tolist() == want.tolist()
        assert a.clone().lerp_(b, _weight(lucid, kind, weight, a, device=device)).tolist() == (
            want.tolist()
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_a_weight_of_one_gives_the_end_including_the_sign_of_a_zero(dtype: str) -> None:
    for device in _devices(dtype):
        a = lucid.tensor([0.0, -0.0, 1.5, -2.0], dtype=getattr(lucid, dtype), device=device)
        b = lucid.tensor([-0.0, 0.0, -0.0, 0.0], dtype=getattr(lucid, dtype), device=device)
        for kind in _KINDS:
            got = lucid.lerp(a, b, _weight(lucid, kind, 1.0, a, device=device)).tolist()
            assert [math.copysign(1.0, x) for x in got] == [-1.0, 1.0, -1.0, 1.0], (device, kind)


def test_the_case_the_single_formula_lost_is_the_end() -> None:
    """``1 + 1 * (1e-8 - 1)`` is ``0`` in float32: the end, ``1e-8``, is what a weight of 1 gives."""
    a, b = lucid.tensor([1.0]), lucid.tensor([1e-8])
    assert lucid.lerp(a, b, 1.0).item() == b.item() == lucid.lerp(a, b, lucid.tensor([1.0])).item()


# ── the gradient is the reference's, whichever form the value took ───────────────────────────


def _grads(lib: Any, dtype: str, kind: str, weight: float, upstream: float, **where: Any) -> Any:
    """``(d start, d end, d weight or None)`` of ``lerp`` over the operand pairs, per element."""
    pairs = list(itertools.product(_OPERANDS, _OPERANDS))
    start, end = (
        _build(lib, dtype, [p[i] for p in pairs], **where).requires_grad_() for i in (0, 1)
    )
    w = _weight(lib, kind, weight, start, **where)
    if kind != "number":
        w.requires_grad_()
    lib.lerp(start, end, w).backward(lib.full_like(start.detach(), upstream))
    return start.grad, end.grad, (w.grad if kind != "number" else None)


@pytest.mark.parametrize(("dtype", "device", "kind"), list(_cells()))
def test_the_gradient_is_the_references(
    dtype: str, device: str, kind: str, ref: ModuleType
) -> None:
    for weight, upstream in itertools.product(_WEIGHTS, _UPSTREAM):
        got = _grads(lucid, dtype, kind, weight, upstream, device=device)
        want = _grads(ref, dtype, kind, weight, upstream)
        for name, g, w in zip(("start", "end", "weight"), got, want, strict=True):
            if w is None:
                assert g is None
                continue
            assert g is not None and g.shape == w.shape, name
            assert g.dtype is getattr(lucid, dtype), name
            for x, y in zip(_flat(g), _flat(w), strict=True):
                assert _agrees(x, y), f"d{name} (weight {weight}, upstream {upstream}): {x} vs {y}"


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_the_gradient_does_not_depend_on_the_branch(device: str) -> None:
    """``d/dw`` is ``end - start`` and ``d/dstart`` is ``1 - w`` on both sides of the split."""
    a = lucid.tensor([1.0, 3.0], device=device).requires_grad_()
    b = lucid.tensor([4.0, 11.0], device=device).requires_grad_()
    w = lucid.tensor([0.25, 0.75], device=device).requires_grad_()
    lucid.lerp(a, b, w).sum().backward()
    assert a.grad is not None and b.grad is not None and w.grad is not None
    assert a.grad.tolist() == [0.75, 0.25]
    assert b.grad.tolist() == [0.25, 0.75]
    assert w.grad.tolist() == [3.0, 8.0]


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_an_infinite_difference_does_not_poison_the_weight_gradient_of_the_other_form(
    device: str,
) -> None:
    """Differentiating the ``where`` of two forms gives the form not taken a zero cotangent, and
    ``0 * inf`` is ``NaN``: ``d/dw`` of ``lerp(0, inf, 0.5)`` is ``inf``, as the reference says."""
    a = lucid.tensor([0.0], device=device).requires_grad_()
    b = lucid.tensor([_INF], device=device).requires_grad_()
    w = lucid.tensor([0.5], device=device).requires_grad_()
    lucid.lerp(a, b, w).backward(lucid.tensor([1.0], device=device))
    assert w.grad is not None and w.grad.item() == _INF
    assert a.grad is not None and a.grad.item() == 0.5


@pytest.mark.parametrize(("dtype", "device", "kind"), list(_cells(("float32", "float16"))))
def test_the_in_place_form_writes_the_same_value_and_back_propagates_the_same(
    dtype: str, device: str, kind: str
) -> None:
    start, end = _finite_operands(dtype)
    for weight in (0.3, 0.7, 1.0):
        pair = [_build(lucid, dtype, v, device=device).requires_grad_() for v in (start, end)]
        w = _weight(lucid, kind, weight, pair[0], device=device)
        if kind != "number":
            w.requires_grad_()
        out = lucid.lerp(*pair, w)
        out.backward(lucid.full_like(out.detach(), 0.7))
        mine = [t.grad for t in pair] + [w.grad if kind != "number" else None]

        base = _build(lucid, dtype, start, device=device).requires_grad_()
        other = _build(lucid, dtype, end, device=device).requires_grad_()
        w2 = _weight(lucid, kind, weight, base, device=device)
        if kind != "number":
            w2.requires_grad_()
        written = (base * 1.0).lerp_(other, w2)
        assert written.tolist() == out.tolist()
        written.backward(lucid.full_like(written.detach(), 0.7))
        theirs = [base.grad, other.grad] + [w2.grad if kind != "number" else None]
        for x, y in zip(mine, theirs, strict=True):
            assert (x is None) == (y is None)
            if x is not None and y is not None:
                assert x.tolist() == y.tolist()


@pytest.mark.parametrize(("dtype", "device", "kind"), list(_cells(("float32",), _KINDS[1:])))
def test_each_set_of_operands_that_needs_a_gradient_gets_the_references(
    dtype: str, device: str, kind: str, ref: ModuleType
) -> None:
    """The node saves only what the gradients asked for need: every non-empty subset."""
    values = ([1.0, -2.0, 0.5, 3.0], [3.0, 0.5, -1.5, 3.5])
    for needs in itertools.product((False, True), repeat=3):
        if not any(needs):
            continue
        runs = []
        for lib, where in ((lucid, {"device": device}), (ref, {})):
            a, b = (_build(lib, dtype, v, **where) for v in values)
            w = _weight(lib, kind, 0.75, a, **where)
            ops = [t.requires_grad_(flag) for t, flag in zip((a, b, w), needs, strict=True)]
            lib.lerp(*ops).backward(lib.tensor([1.0, 0.7, -1.3, 2.0], **where))
            runs.append([t.grad for t in ops])
        for flag, got, want in zip(needs, *runs, strict=True):
            assert (got is not None) == flag == (want is not None), needs
            if got is not None and want is not None:
                assert got.shape == want.shape and got.tolist() == want.tolist(), needs


@pytest.mark.parametrize(("dtype", "device", "kind"), list(_cells(("float32",), _KINDS[1:])))
def test_a_second_derivative_is_the_references(
    dtype: str, device: str, kind: str, ref: ModuleType
) -> None:
    """The node's backward is differentiable: ``create_graph`` carries through it."""
    values = ([1.0, -2.0, 0.5, 3.0], [3.0, 0.5, -1.5, 3.5])
    results = []
    for lib, where, grad in (
        (lucid, {"device": device}, lucid.autograd.grad),
        (ref, {}, ref.autograd.grad),
    ):
        ops = [_build(lib, dtype, v, **where).requires_grad_() for v in values]
        ops.append(_weight(lib, kind, 0.75, ops[0], **where).requires_grad_())
        if kind == "tensor":
            ops[2] = _build(lib, dtype, [0.25, 0.75, 0.5, -1.5], **where).requires_grad_()
        first = grad(lib.lerp(*ops).pow(2).sum(), ops, create_graph=True)
        (first[0] * first[1] + first[2].sum() * first[0].sum()).sum().backward()
        results.append([t.grad for t in ops])
    for got, want in zip(*results, strict=True):
        assert got is not None and want is not None
        for x, y in zip(_flat(got), _flat(want), strict=True):
            assert abs(x - y) <= 1e-5 * max(1.0, abs(y)), (x, y)


# ── the spellings take one path ──────────────────────────────────────────────────────────────


def _spellings(a: Any, b: Any, w: Any) -> list[Any]:
    return [lucid.lerp(a, b, w), a.lerp(b, w), a.clone().lerp_(b, w)]


@pytest.mark.parametrize(("dtype", "device", "kind"), list(_cells(("float32", "bfloat16"))))
def test_the_function_the_method_and_the_in_place_method_answer_alike(
    dtype: str, device: str, kind: str
) -> None:
    start, end = _table()
    a, b = (_build(lucid, dtype, v, device=device) for v in (start, end))
    for weight in _WEIGHTS:
        answers = [
            t.float().tolist()
            for t in _spellings(a, b, _weight(lucid, kind, weight, a, device=device))
        ]
        for other in answers[1:]:
            assert all(_agrees(x, y) for x, y in zip(answers[0], other, strict=True)), weight


# ── no host read: a tensor weight is a device-side choice ────────────────────────────────────


@pytest.mark.parametrize(
    ("dtype", "device"), [(d, v) for d in ("float32", "float16") for v in _devices(d)]
)
def test_a_tensor_weight_is_never_read_on_the_host(
    dtype: str, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Choosing the form by ``|weight| < 0.5`` is a ``where``, so there is no host sync on Metal."""
    start, end = _table()
    a, b = (_build(lucid, dtype, v, device=device) for v in (start, end))
    ws = [_weight(lucid, kind, 0.75, a, device=device) for kind in _KINDS[1:]]

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("a tensor weight was read on the host")

    for name in ("item", "tolist", "numpy", "__bool__", "__float__", "__int__", "__index__"):
        monkeypatch.setattr(lucid.Tensor, name, refuse, raising=False)
    for w in ws:
        assert lucid.lerp(a, b, w).shape == a.shape


def _number_run(weight: float) -> Callable[[Any, Any], Any]:
    def run(a: Any, b: Any) -> Any:
        return lucid.lerp(a, b, weight)

    return run


def _tensor_run(a: Any, b: Any, w: Any) -> Any:
    return lucid.lerp(a, b, w)


@pytest.mark.parametrize("dtype", ("float32", "float16"))
@pytest.mark.parametrize("kind", _KINDS)
def test_a_compiled_trace_answers_as_eager_does(
    dtype: str, kind: str, device_gpu_only: str
) -> None:
    """A tensor weight is chosen on the device: one executable answers both forms."""
    start, end = _table()
    a, b = (_build(lucid, dtype, v, device=device_gpu_only) for v in (start, end))
    weights = (0.25, 0.5, 1.0, -1.5, _INF, _NAN)
    compiled_tensor = lucid.compile(_tensor_run)
    for weight in weights:
        if kind == "number":
            run, compiled, args = _number_run(weight), lucid.compile(_number_run(weight)), (a, b)
        else:
            w = _weight(lucid, kind, weight, a, device=device_gpu_only)
            run, compiled, args = _tensor_run, compiled_tensor, (a, b, w)
        compiled(*args)
        got = compiled(*args).float().tolist()
        want = run(*args).float().tolist()
        assert all(_agrees(x, y) for x, y in zip(got, want, strict=True)), (kind, weight)


def test_a_zero_dim_cpu_weight_beside_metal_operands_answers_on_metal(
    device_gpu_only: str, ref: ModuleType
) -> None:
    values = ([0.0, 1.0, _INF, -2.0], [4.0, 5.0, 0.0, _NAN])
    a, b = (lucid.tensor(v, device=device_gpu_only) for v in values)
    for weight in (0.25, 0.75, 1.0):
        out = lucid.lerp(a, b, lucid.tensor(weight))
        want = ref.lerp(*(ref.tensor(v) for v in values), ref.tensor(weight)).tolist()
        assert out.device.type == device_gpu_only
        assert all(_agrees(x, y) for x, y in zip(out.tolist(), want, strict=True)), weight


# ── what stays as it was ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device", devices_supporting(lucid.float32))
def test_a_complex_zero_dim_weight_takes_the_form_its_modulus_picks(
    device: str, ref: ModuleType
) -> None:
    a_vals, b_vals = [1.0, -2.0, 0.5], [3.0, 0.5, -1.5]
    # |0.25 + 0.25j| < 0.5, |0.5 + 0.5j| and |1 + 0j| are not
    for weight in (0.25 + 0.25j, 0.5 + 0.5j, 1 + 0j, -0.3 + 0.2j, 2 - 1j):
        got = lucid.lerp(
            lucid.tensor(a_vals, device=device),
            lucid.tensor(b_vals, device=device),
            lucid.tensor(weight, device=device),
        )
        want = ref.lerp(ref.tensor(a_vals), ref.tensor(b_vals), ref.tensor(weight))
        assert got.dtype is lucid.complex64
        for x, y in zip(got.tolist(), want.tolist(), strict=True):
            assert abs(complex(x) - complex(y)) <= 2e-6 * max(1.0, abs(complex(y))), weight


@pytest.mark.parametrize("device", devices_supporting(lucid.float16))
def test_the_float32_answer_rounded_once_is_what_a_half_lerp_gives(device: str) -> None:
    """The half path widens the operands first (LCD-1087): the split is the float32 one."""
    rng = random.Random(7)
    start = [rng.uniform(-3.0, 3.0) for _ in range(64)]
    end = [rng.uniform(-3.0, 3.0) for _ in range(64)]
    for dtype in ("float16", "bfloat16"):
        a, b = (_build(lucid, dtype, v, device=device) for v in (start, end))
        for weight in (0.3, 0.7, 1.0, -0.8):
            wide = lucid.lerp(a.to(lucid.float32), b.to(lucid.float32), weight)
            assert lucid.lerp(a, b, weight).tolist() == wide.to(getattr(lucid, dtype)).tolist()
