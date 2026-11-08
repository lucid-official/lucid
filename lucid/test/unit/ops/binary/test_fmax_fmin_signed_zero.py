"""``fmax`` / ``fmin`` rank a signed-zero tie like the reference (LCD-929).

The rule (the reference's vectorised kernels): the max of a signed-zero tie is ``+0`` and the
min is ``-0``, whichever operand came first; a tie of two equal zeros keeps its sign.  The
composite used to keep the first operand's zero.  The NaN rule is unchanged: a NaN operand
loses to a number, and NaN comes out only when both are NaN.  A tie's gradient still goes to
the first operand alone (LCD-436).

Values are held to the rule with ``math.copysign`` (``==`` cannot tell ``-0`` from ``+0``);
the reference is consulted on 16-element rows, its vectorised path.
"""

import math
import warnings

import pytest

import lucid
import lucid.nn as nn
from lucid.compile._core.fallback import CompileFallbackWarning

_NAN = math.nan
_INF = math.inf
_DTYPES = ("float32", "float16", "bfloat16")
_OPS = ("fmax", "fmin")


def _sign(v: float) -> float:
    return math.copysign(1.0, v)


def _expected_zero(op: str, a: float, b: float) -> float:
    """The ranked zero: max is +0 unless both are -0, min is -0 unless both are +0."""
    if op == "fmax":
        return 0.0 if (_sign(a) > 0 or _sign(b) > 0) else -0.0
    return -0.0 if (_sign(a) < 0 or _sign(b) < 0) else 0.0


def _row(values: list[float], fill: float = 0.5) -> list[float]:
    """Pad to 16 elements so the reference takes its vectorised path."""
    return values + [fill] * (16 - len(values))


def _same_bits(got: list[float], want: list[float]) -> None:
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if math.isnan(w):
            assert math.isnan(g), (i, got, want)
        else:
            assert g == w and _sign(g) == _sign(w), (i, got, want)


# ── the signed-zero rule ────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize(("a", "b"), [(-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (0.0, 0.0)], ids=str)
def test_a_zero_tie_follows_the_rule(device: str, op: str, dtype: str, a: float, b: float) -> None:
    dt = getattr(lucid, dtype)
    out = getattr(lucid, op)(
        lucid.tensor([a], dtype=dt, device=device), lucid.tensor([b], dtype=dt, device=device)
    )
    got = out.to(lucid.float32).item()
    want = _expected_zero(op, a, b)
    assert got == 0.0 and _sign(got) == _sign(want), (op, a, b, got)


@pytest.mark.parametrize("op", _OPS)
def test_the_zero_rule_holds_inside_a_long_mixed_row(device: str, op: str) -> None:
    a = [-0.0, 0.0, -0.0, 0.0, 1.0, -1.0, _NAN, 2.0] * 4
    b = [0.0, -0.0, -0.0, 0.0, 1.0, -1.0, 3.0, _NAN] * 4
    got = getattr(lucid, op)(
        lucid.tensor(a, device=device), lucid.tensor(b, device=device)
    ).tolist()
    for i in range(4):
        want = _expected_zero(op, a[i], b[i])
        assert _sign(got[i]) == _sign(want), (i, got)


@pytest.mark.parametrize("op", _OPS)
def test_a_scalar_operand_ranks_too(device: str, op: str) -> None:
    x = lucid.tensor([-0.0, 0.0], device=device)
    for scalar in (0.0, -0.0):
        got = getattr(lucid, op)(x, scalar).tolist()
        want = [_expected_zero(op, v, scalar) for v in (-0.0, 0.0)]
        _same_bits(got, want)


@pytest.mark.parametrize("op", _OPS)
def test_broadcast_zeros_follow_the_rule(device: str, op: str) -> None:
    a = lucid.tensor([[-0.0], [0.0]], device=device)
    b = lucid.tensor([0.0, -0.0, -0.0], device=device)
    got = getattr(lucid, op)(a, b).tolist()
    for i, ai in enumerate((-0.0, 0.0)):
        _same_bits(got[i], [_expected_zero(op, ai, bj) for bj in (0.0, -0.0, -0.0)])


# ── the NaN and infinity rules are unchanged ────────────────────────────────────

_NAN_ROWS = [
    (_NAN, 1.0),
    (1.0, _NAN),
    (_NAN, _NAN),
    (_NAN, 0.0),
    (0.0, _NAN),
    (_NAN, -0.0),
    (-0.0, _NAN),
    (_INF, -_INF),
    (-_INF, _INF),
    (_INF, _INF),
    (-_INF, -_INF),
    (_NAN, _INF),
    (-_INF, _NAN),
]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize(("a", "b"), _NAN_ROWS, ids=str)
def test_nan_and_infinity_rows_match_the_reference(
    ref, device: str, op: str, dtype: str, a: float, b: float
) -> None:
    got = getattr(lucid, op)(
        lucid.tensor(_row([a]), dtype=getattr(lucid, dtype), device=device),
        lucid.tensor(_row([b]), dtype=getattr(lucid, dtype), device=device),
    )
    want = getattr(ref, op)(
        ref.tensor(_row([a]), dtype=getattr(ref, dtype)),
        ref.tensor(_row([b]), dtype=getattr(ref, dtype)),
    )
    _same_bits(got.to(lucid.float32).tolist(), want.float().tolist())


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("op", _OPS)
def test_zero_rows_match_the_reference_bit_for_bit(ref, device: str, op: str, dtype: str) -> None:
    a = _row([-0.0, 0.0, -0.0, 0.0, 1.0, -1.0, 0.0, -0.0])
    b = _row([0.0, -0.0, -0.0, 0.0, 1.0, -1.0, 2.0, -2.0])
    got = getattr(lucid, op)(
        lucid.tensor(a, dtype=getattr(lucid, dtype), device=device),
        lucid.tensor(b, dtype=getattr(lucid, dtype), device=device),
    )
    want = getattr(ref, op)(
        ref.tensor(a, dtype=getattr(ref, dtype)), ref.tensor(b, dtype=getattr(ref, dtype))
    )
    _same_bits(got.to(lucid.float32).tolist(), want.float().tolist())


@pytest.mark.parametrize("op", _OPS)
def test_broadcast_matches_the_reference(ref, device: str, op: str) -> None:
    a = [[-0.0], [0.0], [_NAN], [1.0]]
    b = [0.0, -0.0, _NAN, 2.0]
    got = getattr(lucid, op)(lucid.tensor(a, device=device), lucid.tensor(b, device=device))
    want = getattr(ref, op)(ref.tensor(a), ref.tensor(b))
    assert got.shape == tuple(want.shape)
    for g_row, w_row in zip(got.tolist(), want.tolist(), strict=True):
        _same_bits(g_row, w_row)


@pytest.mark.parametrize("op", _OPS)
def test_integers_are_untouched(device: str, op: str) -> None:
    a = lucid.tensor([3, -2, 0, 7], device=device)
    b = lucid.tensor([3, 5, -1, 0], device=device)
    got = getattr(lucid, op)(a, b)
    want = [max, min][op == "fmin"]
    assert got.tolist() == [want(x, y) for x, y in zip(a.tolist(), b.tolist(), strict=True)]


# ── gradients keep the first-operand tie rule ───────────────────────────────────

_GRAD_PAIRS = [
    (-0.0, 0.0),
    (0.0, -0.0),
    (-0.0, -0.0),
    (0.0, 0.0),
    (1.0, 1.0),
    (_INF, _INF),
    (_NAN, 1.0),
    (1.0, _NAN),
    (_NAN, _NAN),
    (2.0, 1.0),
    (1.0, 2.0),
]


def _grads(lib, op: str, a: list[float], b: list[float], **kw) -> list[list[float]]:
    ta = lib.tensor(a, requires_grad=True, **kw)
    tb = lib.tensor(b, requires_grad=True, **kw)
    getattr(lib, op)(ta, tb).sum().backward()
    return [ta.grad.tolist(), tb.grad.tolist()]


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize(("a", "b"), _GRAD_PAIRS, ids=str)
def test_gradient_matches_the_reference(ref, device: str, op: str, a: float, b: float) -> None:
    ra, rb = _row([a]), _row([b], fill=0.25)
    want = _grads(ref, op, ra, rb)
    got = _grads(lucid, op, ra, rb, device=device)
    for g_col, w_col in zip(got, want, strict=True):
        assert len(g_col) == len(w_col)
        for g, w in zip(g_col, w_col, strict=True):
            assert (math.isnan(w) and math.isnan(g)) or g == w, (got, want)


@pytest.mark.parametrize("op", _OPS)
def test_a_zero_tie_sends_the_whole_gradient_to_the_first_operand(device: str, op: str) -> None:
    a = lucid.tensor([-0.0, 0.0], device=device, requires_grad=True)
    b = lucid.tensor([0.0, -0.0], device=device, requires_grad=True)
    getattr(lucid, op)(a, b).sum().backward()
    assert a.grad.tolist() == [1.0, 1.0]
    assert b.grad.tolist() == [0.0, 0.0]


@pytest.mark.parametrize("op", _OPS)
def test_broadcast_gradient_collects_every_row(ref, device: str, op: str) -> None:
    a = [[0.0], [1.0], [-1.0]]
    b = [0.0, 2.0, -2.0, 0.5]
    want = _grads(ref, op, a, b)
    got = _grads(lucid, op, a, b, device=device)
    assert got == want


@pytest.mark.parametrize("op", _OPS)
def test_second_order_path_survives(device: str, op: str) -> None:
    a = lucid.tensor([-0.0, 1.0, 2.0], device=device, requires_grad=True)
    b = lucid.tensor([0.0, 2.0, 1.0], device=device, requires_grad=True)
    (g,) = lucid.autograd.grad(getattr(lucid, op)(a, b).sum(), [a], create_graph=True)
    assert g.tolist() == ([1.0, 0.0, 1.0] if op == "fmax" else [1.0, 1.0, 0.0])


# ── compiled (Metal) ────────────────────────────────────────────────────────────


class _Apply(nn.Module):
    def __init__(self, op: str) -> None:
        super().__init__()
        self.op = op

    def forward(self, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
        return getattr(lucid, self.op)(a, b)


@pytest.mark.parametrize("op", _OPS)
def test_compiled_matches_eager_on_zeros_and_nans(device_gpu_only: str, op: str) -> None:
    a = lucid.tensor(_row([-0.0, 0.0, -0.0, 0.0, _NAN, 1.0, _NAN, _INF]), device=device_gpu_only)
    b = lucid.tensor(_row([0.0, -0.0, -0.0, 0.0, 1.0, _NAN, _NAN, -_INF]), device=device_gpu_only)
    want = getattr(lucid, op)(a, b).tolist()
    compiled = lucid.compile(_Apply(op))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        with lucid.no_grad():
            got = [compiled(a, b).tolist() for _ in range(2)]
    assert not compiled.cache_info()["eager_only"]
    for g in got:
        _same_bits(g, want)
