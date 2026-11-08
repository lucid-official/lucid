"""Extremum values that depended on representation or order (LCD-879).

Two defects of one class, each held to the rule instead of to one lucky input:

* **Tie share past the exact range of a half dtype.**  ``max`` / ``min`` / ``amax`` / ``amin``
  split a slice's gradient among the positions tied for the extremum.  The tie count was taken
  in the input dtype, where bfloat16 is exact only to 256 and float16 to 2048, so 257 equal
  bfloat16 elements shared ``1 / 256`` (or ``1 / 254``) each and the gradient summed to more
  than one.  The count is now taken in float32, as the reference's ``grad / count`` is.
* **Sign of a zero extremum.**  ``max`` of a signed-zero tie is +0 and ``min`` is -0, whichever
  operand or element came first (the reference's vectorised kernels).  Lucid kept whichever
  zero its comparison met first, so the answer followed the element order — and on Metal the
  operand order.  ``amax`` / ``amin`` of the reference are themselves order dependent on short
  rows (the first zero wins below ~16 elements), so they are held to the rule, not to it.

``max(dim)`` / ``min(dim)`` return the value *at the returned index*, so their zero keeps the
sign of the first extremum, as the reference's does; they are not under the rule.
"""

import math
import random
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.nn.functional as F

_HALF = ("bfloat16", "float16")
_ALL = ("float32", "float16", "bfloat16")
_NEG, _POS = -0.0, 0.0


def _sign(v: float) -> float:
    return math.copysign(1.0, v)


# ── tie share ───────────────────────────────────────────────────────────────────


def _ones_grad(name: str, dtype: str, device: str, n: int) -> list[float]:
    x = lucid.ones(n, dtype=getattr(lucid, dtype), device=device, requires_grad=True)
    out = {
        "max": lambda: x.max(),
        "min": lambda: x.min(),
        "amax": lambda: lucid.amax(x),
        "amin": lambda: lucid.amin(x),
        "amax_dim": lambda: lucid.amax(x, dim=0),
        "amin_dim": lambda: lucid.amin(x, dim=0),
    }[name]()
    out.backward()
    assert x.grad is not None
    return x.grad.to(lucid.float32).tolist()


@pytest.mark.parametrize("name", ["max", "min", "amax", "amin", "amax_dim", "amin_dim"])
@pytest.mark.parametrize(
    ("dtype", "n"),
    [
        ("bfloat16", 256),
        ("bfloat16", 257),
        ("bfloat16", 1000),
        ("float16", 2049),
        ("float16", 5000),
    ],
)
def test_a_tie_shares_one_over_its_true_count(device: str, name: str, dtype: str, n: int) -> None:
    got = _ones_grad(name, dtype, device, n)
    # 1 / n rounded once to the dtype, never to 1 / 256 or 1 / 254
    want = lucid.tensor([1.0 / n], dtype=getattr(lucid, dtype)).to(lucid.float32).item()
    assert got == [want] * n
    assert math.isclose(sum(got), 1.0, rel_tol=2e-2 if dtype == "bfloat16" else 2e-3)


def test_only_the_tied_positions_share(device: str) -> None:
    """300 maxima among smaller values: each gets 1/300, the rest nothing."""
    data = [1.0] * 300 + [0.5] * 40
    x = lucid.tensor(data, dtype=lucid.bfloat16, device=device, requires_grad=True)
    lucid.amax(x).backward()
    assert x.grad is not None
    got = x.grad.to(lucid.float32).tolist()
    share = lucid.tensor([1.0 / 300], dtype=lucid.bfloat16).to(lucid.float32).item()
    assert got == [share] * 300 + [0.0] * 40


def test_the_share_scales_the_incoming_gradient(device: str) -> None:
    x = lucid.ones(3, 257, dtype=lucid.bfloat16, device=device, requires_grad=True)
    out = lucid.amax(x, dim=1) * lucid.tensor([1.0, 2.0, 4.0], dtype=lucid.bfloat16, device=device)
    out.sum().backward()
    assert x.grad is not None
    rows = x.grad.to(lucid.float32).tolist()
    for row, scale in zip(rows, (1.0, 2.0, 4.0), strict=True):
        assert math.isclose(sum(row), scale, rel_tol=2e-2)
        assert len(set(row)) == 1


@pytest.mark.parametrize("name", ["max", "min"])
def test_the_second_derivative_path_counts_in_float32_too(device: str, name: str) -> None:
    """``create_graph`` builds the share from ops: its tie count must not saturate either."""
    x = lucid.ones(257, dtype=lucid.bfloat16, device=device, requires_grad=True)
    (g,) = lucid.autograd.grad(getattr(x, name)(), x, create_graph=True)
    want = lucid.tensor([1.0 / 257], dtype=lucid.bfloat16).to(lucid.float32).item()
    assert g.to(lucid.float32).tolist() == [want] * 257


@pytest.mark.parametrize("name", ["max", "min"])
@pytest.mark.parametrize("dtype", _HALF)
def test_full_max_min_gradient_matches_the_reference(
    ref: ModuleType, device: str, name: str, dtype: str
) -> None:
    n = 257 if dtype == "bfloat16" else 2049
    got = _ones_grad(name, dtype, device, n)
    t = ref.ones(n, dtype=getattr(ref, dtype), requires_grad=True)
    getattr(t, name)().backward()
    assert t.grad is not None
    assert got == t.grad.float().tolist()


# ── sign of a zero extremum ─────────────────────────────────────────────────────


def _ranked(values: list[float], is_max: bool) -> float:
    """The rule: the extremum, and for a zero extremum +0 (max) / -0 (min) whenever one is present."""
    best = max(values) if is_max else min(values)
    if best != 0.0:
        return best
    want_positive = is_max
    signs = [_sign(v) for v in values if v == 0.0]
    if want_positive:
        return 0.0 if 1.0 in signs else -0.0
    return -0.0 if -1.0 in signs else 0.0


def _same(got: float, want: float) -> bool:
    return got == want and _sign(got) == _sign(want)


def _rows() -> dict[str, list[float]]:
    rng = random.Random(879)
    rows = {
        "-0,+0": [_NEG, _POS],
        "+0,-0": [_POS, _NEG],
        "-0,-0": [_NEG, _NEG],
        "+0,+0": [_POS, _POS],
        "neg,-0,+0": [-1.0, _NEG, _POS, -2.0],
        "neg,+0,-0": [-1.0, _POS, _NEG, -2.0],
        "pos,-0,+0": [1.0, _NEG, _POS, 2.0],
        "only-neg-and-negzero": [-3.0, _NEG, -1.0],
        "five": [_NEG, _NEG, _NEG, _NEG, _POS],
        "five-rev": [_POS, _NEG, _NEG, _NEG, _NEG],
    }
    for n in (7, 16, 33, 64):
        rows[f"random-{n}"] = [rng.choice([_NEG, _POS]) for _ in range(n)]
        rows[f"random-{n}-below"] = [rng.choice([_NEG, _POS, -1.0]) for _ in range(n)]
        rows[f"random-{n}-above"] = [rng.choice([_NEG, _POS, 1.0]) for _ in range(n)]
    return rows


_ROWS = _rows()


def _t(values: list[float], dtype: str, device: str) -> lucid.Tensor:
    return lucid.tensor(values, dtype=getattr(lucid, dtype), device=device)


@pytest.mark.parametrize("dtype", _ALL)
@pytest.mark.parametrize("row", list(_ROWS))
def test_reductions_rank_the_zeros(device_cpu_only: str, dtype: str, row: str) -> None:
    v = _ROWS[row]
    x = _t(v, dtype, device_cpu_only)
    for is_max, fns in (
        (True, (lambda t: t.max(), lambda t: lucid.amax(t), lambda t: lucid.aminmax(t).max)),
        (False, (lambda t: t.min(), lambda t: lucid.amin(t), lambda t: lucid.aminmax(t).min)),
    ):
        want = _ranked(v, is_max)
        for fn in fns:
            assert _same(fn(x).item(), want), (v, is_max)


@pytest.mark.parametrize("dtype", _ALL)
@pytest.mark.parametrize("row", list(_ROWS))
def test_strided_reductions_rank_the_zeros(device_cpu_only: str, dtype: str, row: str) -> None:
    """The reduced axis is not the contiguous one: columns of a 2-D tensor, and a 3-D middle axis."""
    v = _ROWS[row]
    n = len(v)
    mat = [[v[i], v[(i + 1) % n], v[(i + 2) % n]] for i in range(n)]
    x = _t([c for r in mat for c in r], dtype, device_cpu_only).reshape(n, 3)
    for is_max, fn in ((True, lucid.amax), (False, lucid.amin)):
        got = fn(x, 0).tolist()
        for c in range(3):
            assert _same(got[c], _ranked([mat[i][c] for i in range(n)], is_max)), (v, c)
        both = fn(x, (0, 1)).item()
        assert _same(both, _ranked([e for r in mat for e in r], is_max))
    cube = x.reshape(1, n, 3)
    assert _same(lucid.amax(cube, 1).tolist()[0][0], _ranked([mat[i][0] for i in range(n)], True))


@pytest.mark.parametrize("dtype", _ALL)
def test_all_negative_zero_and_all_positive_zero_keep_their_sign(device: str, dtype: str) -> None:
    for fn in (
        lambda t: t.max(),
        lambda t: t.min(),
        lambda t: lucid.amax(t),
        lambda t: lucid.amin(t),
    ):
        assert _sign(fn(_t([_NEG] * 70, dtype, device)).item()) == -1.0
        assert _sign(fn(_t([_POS] * 70, dtype, device)).item()) == 1.0


@pytest.mark.parametrize("dtype", ("float32", "float16", "bfloat16"))
def test_extremum_with_nan_inf_and_zero_is_unchanged(device: str, dtype: str) -> None:
    nan, inf = math.nan, math.inf
    assert math.isnan(_t([_NEG, nan, _POS], dtype, device).max().item())
    assert math.isnan(_t([_NEG, nan, _POS], dtype, device).min().item())
    assert _t([-inf, _NEG, _POS], dtype, device).max().item() == 0.0
    assert _t([inf, _NEG, _POS], dtype, device).min().item() == 0.0
    assert _t([inf, _NEG, _POS], dtype, device).max().item() == inf
    assert _t([-inf, _NEG, _POS], dtype, device).min().item() == -inf


def test_the_empty_and_zero_dim_cases_still_reduce(device: str) -> None:
    assert lucid.amax(lucid.tensor(3.0, device=device)).item() == 3.0
    assert _sign(lucid.amax(lucid.tensor(_NEG, device=device)).item()) == -1.0
    empty = lucid.zeros(0, 3, device=device)
    assert lucid.amax(empty, 1).shape == (0,)


@pytest.mark.parametrize("dtype", _ALL)
@pytest.mark.parametrize("row", list(_ROWS))
def test_maximum_minimum_rank_the_zeros_in_either_operand_order(
    device: str, dtype: str, row: str
) -> None:
    a = _ROWS[row]
    b = list(reversed(a))
    ta, tb = _t(a, dtype, device), _t(b, dtype, device)
    for is_max, fn in ((True, lucid.maximum), (False, lucid.minimum)):
        for x, y, vx, vy in ((ta, tb, a, b), (tb, ta, b, a)):
            got = fn(x, y).tolist()
            for g, p, q in zip(got, vx, vy, strict=True):
                assert _same(g, _ranked([p, q], is_max)), (p, q, is_max)


@pytest.mark.parametrize("dtype", _ALL)
def test_maximum_minimum_broadcast_a_scalar_zero(device: str, dtype: str) -> None:
    x = _t([_NEG, _POS, _NEG, _POS, -1.0, 1.0], dtype, device)
    zero_pos, zero_neg = _t([_POS], dtype, device), _t([_NEG], dtype, device)
    assert [_sign(v) for v in lucid.maximum(x, zero_neg).tolist()[:4]] == [-1, 1, -1, 1]
    assert [_sign(v) for v in lucid.maximum(x, zero_pos).tolist()[:4]] == [1, 1, 1, 1]
    assert [_sign(v) for v in lucid.minimum(x, zero_pos).tolist()[:4]] == [-1, 1, -1, 1]
    assert [_sign(v) for v in lucid.minimum(x, zero_neg).tolist()[:4]] == [-1, -1, -1, -1]


_CLAMPS: dict[str, tuple[Callable[[lucid.Tensor], lucid.Tensor], Callable[[float], float]]] = {
    "clamp(min=+0)": (lambda t: t.clamp(min=_POS), lambda v: _ranked([v, _POS], True)),
    "clamp(min=-0)": (lambda t: t.clamp(min=_NEG), lambda v: _ranked([v, _NEG], True)),
    "clamp(max=+0)": (lambda t: t.clamp(max=_POS), lambda v: _ranked([v, _POS], False)),
    "clamp(max=-0)": (lambda t: t.clamp(max=_NEG), lambda v: _ranked([v, _NEG], False)),
    "relu": (F.relu, lambda v: _ranked([v, _POS], True)),
}


@pytest.mark.parametrize("dtype", _ALL)
@pytest.mark.parametrize("name", list(_CLAMPS))
def test_clamp_to_zero_ranks_the_zeros(device: str, dtype: str, name: str) -> None:
    fn, rule = _CLAMPS[name]
    v = [_NEG, _POS, _NEG, _POS, -1.0, 1.0, _NEG, _POS, _POS, _NEG, _NEG, _POS, _NEG]
    got = fn(_t(v, dtype, device)).tolist()
    for g, e in zip(got, v, strict=True):
        assert _same(g, rule(e)), (name, e)


@pytest.mark.parametrize("dtype", _ALL)
def test_clamp_keeps_nan_and_the_ordinary_values(device: str, dtype: str) -> None:
    got = _t([math.nan, -2.0, 0.5, 2.0], dtype, device).clamp(min=-1.0, max=1.0).tolist()
    assert math.isnan(got[0]) and got[1:] == [-1.0, 0.5, 1.0]
    got = _t([math.nan, -2.0, 2.0], dtype, device).clamp(min=0.0).tolist()
    assert math.isnan(got[0]) and got[1:] == [0.0, 2.0]


@pytest.mark.parametrize("name", ["max", "min"])
def test_the_gradient_of_a_zero_tie_is_still_shared(device: str, name: str) -> None:
    """Ranking the value must not move the gradient: every tied zero still gets its share."""
    x = lucid.tensor([_NEG, _POS, _NEG, _POS], device=device, requires_grad=True)
    getattr(x, name)().backward()
    assert x.grad is not None and x.grad.tolist() == [0.25] * 4


@pytest.mark.parametrize("name", ["max", "min"])
def test_reference_agrees_on_the_full_reduction(
    ref: ModuleType, device_cpu_only: str, name: str
) -> None:
    """``max()`` / ``min()`` of the reference follow the rule on every length."""
    rng = random.Random(7)
    for n in (2, 3, 5, 8, 33, 100):
        for _ in range(8):
            v = [rng.choice([_NEG, _POS]) for _ in range(n)]
            got = getattr(_t(v, "float32", device_cpu_only), name)().item()
            want = getattr(ref.tensor(v), name)().item()
            assert _same(got, want), v


# MLX's reduction keeps the zero its comparison tree met first, so a Metal max / min of a
# signed-zero tie follows the element order.  The exact form needs a second pass over the input
# (+47..77% on every Metal max / amax / amin), so it is not applied.  Held by strict xfails so a
# fix shows as an XPASS.
@pytest.mark.xfail(
    strict=True,
    reason="LCD-930: Metal and compiled max/min reductions keep the backend's zero order "
    "(exact form costs one more pass, +47-77%)",
)
@pytest.mark.parametrize("dtype", _ALL)
@pytest.mark.parametrize(
    ("name", "row", "is_max"),
    [
        ("max", [_NEG, _POS], True),
        ("amax", [_NEG, _POS], True),
        ("min", [_POS, _NEG], False),
        ("amin", [_POS, _NEG], False),
    ],
)
def test_metal_reduction_zero_order_is_a_known_difference(
    device_gpu_only: str, dtype: str, name: str, row: list[float], is_max: bool
) -> None:
    x = _t(row, dtype, device_gpu_only)
    got = getattr(x, name)().item() if name in ("max", "min") else getattr(lucid, name)(x).item()
    assert _same(got, _ranked(row, is_max))
