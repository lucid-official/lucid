"""``rearrange`` / ``reduce`` / ``repeat`` own the reference einops grammar.

The engine kernels behind these used to parse patterns themselves and
honoured little of what a pattern means (LCD-189): a keyword length was
overwritten by the input's dimension instead of checked, an unused keyword
was accepted, ``...`` was a syntax error, a unit axis inside a group or a
group of groups crashed with ``bad_variant_access``, a number on the right
of ``repeat`` was a reshape error, and a malformed pattern surfaced as
whatever reshape or permute it broke.

One matrix drives everything here.  Every case states what Lucid answers
— the output shape, or the refusal class — and the order-sensitive ones
are checked element by element against numpy, so the contract holds
without the reference installed.  Under ``-m parity`` the same matrix is
replayed against the einops package itself (values, dtypes and which cases
it refuses), and the gradients against einops over the reference framework.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.einops as le
from lucid._C import engine as _C_engine

_Bad = _C_engine.InvalidArgument
_Shape = _C_engine.ShapeMismatch

# (op, pattern, input shape, reduction, keyword lengths, output shape or the
# refusal class).  ``op`` is the entry point; ``reduction`` is for reduce.
_Case = tuple[str, str, tuple[int, ...], str | None, dict[str, Any], object]

_CASES: list[_Case] = [
    # keyword lengths are checks, not overrides
    ("rearrange", "b c h -> b (c h)", (2, 3, 4), None, {}, (2, 12)),
    ("rearrange", "b c h -> b (c h)", (2, 3, 4), None, {"h": 4}, (2, 12)),
    ("rearrange", "b c h -> b (c h)", (2, 3, 4), None, {"h": 7}, _Shape),
    ("rearrange", "b c h -> b (c h)", (2, 3, 4), None, {"z": 3}, _Bad),
    ("rearrange", "(a b) c -> a b c", (6, 4), None, {"a": 2}, (2, 3, 4)),
    ("rearrange", "(a b) c -> a b c", (6, 4), None, {"a": 2, "b": 3}, (2, 3, 4)),
    ("rearrange", "(a b) c -> a b c", (6, 4), None, {"a": 2, "b": 2}, _Shape),
    ("rearrange", "(a b) c -> a b c", (6, 4), None, {"a": 4}, _Shape),
    ("rearrange", "(a b) c -> a b c", (6, 4), None, {}, _Bad),
    ("rearrange", "(a b) c -> b (c a)", (6, 4), None, {"b": 3}, (3, 8)),
    ("reduce", "b c -> b", (2, 3), "sum", {"c": 3}, (2,)),
    ("reduce", "b c -> b", (2, 3), "sum", {"c": 4}, _Shape),
    ("repeat", "h w -> h w", (2, 3), None, {"h": 3}, _Shape),
    # ...
    ("rearrange", "b ... -> b (...)", (2, 3, 4), None, {}, (2, 12)),
    ("rearrange", "b ... -> (...) b", (2, 3, 4, 5), None, {}, (60, 2)),
    ("rearrange", "... -> ...", (2, 3), None, {}, (2, 3)),
    ("rearrange", "... a -> a ...", (2, 3, 4), None, {}, (4, 2, 3)),
    ("rearrange", "a ... b -> b ... a", (2, 3, 4, 5), None, {}, (5, 3, 4, 2)),
    ("rearrange", "a ... b -> b (...) a", (2, 5), None, {}, (5, 1, 2)),
    ("rearrange", "(a ...) -> a ...", (6,), None, {}, _Bad),
    ("rearrange", "a b -> ...", (2, 3), None, {}, _Bad),
    ("rearrange", "... -> a", (2, 3), None, {}, _Bad),
    ("rearrange", "a ... ... -> a", (2, 3), None, {}, _Bad),
    ("rearrange", "a.b -> a", (2, 3), None, {}, _Bad),
    # U+2026 is the parser's own stand-in for '...', never pattern text
    ("rearrange", "a … -> … a", (2, 3), None, {}, _Bad),
    ("rearrange", "a … … -> a", (2, 3), None, {}, _Bad),
    ("rearrange", "a ... … -> a", (2, 3), None, {}, _Bad),
    ("reduce", "a b c ... -> a", (2, 3), "sum", {}, _Shape),
    ("reduce", "b ... -> b", (2, 3, 4), "sum", {}, (2,)),
    ("reduce", "... h -> h", (2, 3, 4), "mean", {}, (4,)),
    ("reduce", "b ... h -> b h", (2, 3, 4, 5), "max", {}, (2, 5)),
    ("reduce", "b ... -> ... b", (2, 3, 4), "sum", {}, (3, 4, 2)),
    ("repeat", "... -> ... c", (2, 3), None, {"c": 2}, (2, 3, 2)),
    ("repeat", "b ... -> b c ...", (2, 3, 4), None, {"c": 2}, (2, 2, 3, 4)),
    ("repeat", "b ... -> b (c ...)", (2, 3, 4), None, {"c": 2}, (2, 24)),
    # literals: 1 and () are unit axes, other numbers are unnamed axes
    ("rearrange", "b c -> (b 1) c", (2, 3), None, {}, (2, 3)),
    ("rearrange", "b c -> b 1 c", (2, 3), None, {}, (2, 1, 3)),
    ("rearrange", "b c -> b () c", (2, 3), None, {}, (2, 1, 3)),
    ("rearrange", "b 1 c -> b c", (2, 1, 3), None, {}, (2, 3)),
    ("rearrange", "b () c -> c b", (2, 1, 3), None, {}, (3, 2)),
    ("rearrange", "b 1 c -> b c", (2, 2, 3), None, {}, _Shape),
    ("rearrange", "h w -> h 3 w", (2, 3), None, {}, _Bad),
    ("rearrange", "h w -> (h 2) w", (2, 3), None, {}, _Bad),
    ("rearrange", "a 0 -> a", (2, 0), None, {}, _Bad),
    ("reduce", "(h 2) (w 2) -> h w", (4, 6), "max", {}, (2, 3)),
    ("reduce", "b (h 2) -> b h", (2, 5), "max", {}, _Shape),
    ("reduce", "b c -> b c 1", (2, 3), "sum", {}, (2, 3, 1)),
    ("reduce", "b c -> b () 1", (2, 3), "sum", {}, (2, 1, 1)),
    ("reduce", "b c -> 3 b", (2, 3), "sum", {}, _Bad),
    ("repeat", "h w -> (h 2) w", (2, 3), None, {}, (4, 3)),
    ("repeat", "h w -> h 3 w", (2, 3), None, {}, (2, 3, 3)),
    ("repeat", "h w -> (2 h) (w 3)", (2, 3), None, {}, (4, 9)),
    ("repeat", "h w -> h 2 w 3", (2, 3), None, {}, (2, 2, 3, 3)),
    ("repeat", "h 1 -> h c", (2, 1), None, {"c": 3}, (2, 3)),
    ("repeat", "h () -> (h c)", (2, 1), None, {"c": 3}, (6,)),
    ("repeat", "h -> h c 1", (2,), None, {"c": 3}, (2, 3, 1)),
    ("repeat", "(h 2) -> h c", (4,), None, {"c": 3}, _Bad),
    # groups are one level deep, as in the reference
    ("rearrange", "b c h w -> b ((c h) w)", (2, 3, 4, 5), None, {}, _Bad),
    ("rearrange", "((b c)) -> b c", (6,), None, {}, _Bad),
    ("rearrange", "a (b -> a b", (2, 3), None, {}, _Bad),
    ("rearrange", "a b) -> a b", (2, 3), None, {}, _Bad),
    # which axes may appear on which side
    ("rearrange", "a a -> a", (2, 2), None, {}, _Bad),
    ("rearrange", "a b -> a", (2, 3), None, {}, _Bad),
    ("rearrange", "a b -> a b c", (2, 3), None, {"c": 2}, _Bad),
    ("reduce", "b c -> b d", (2, 3), "sum", {}, _Bad),
    ("repeat", "h w -> h", (2, 3), None, {}, _Bad),
    ("repeat", "h w -> h w c", (2, 3), None, {}, _Bad),
    ("repeat", "h w -> h w c", (2, 3), None, {"c": 2}, (2, 3, 2)),
    ("repeat", "h w -> c h w", (2, 3), None, {"c": 0}, (0, 2, 3)),
    ("repeat", "h w -> w h c", (2, 3), None, {"c": 2}, (3, 2, 2)),
    # names, arrows, ranks
    ("rearrange", "a b c -> a b c", (2, 3), None, {}, _Shape),
    ("rearrange", "a b -> b a -> a", (2, 3), None, {}, _Bad),
    ("rearrange", "a b b a", (2, 3), None, {}, _Bad),
    ("rearrange", "a b -> a b_", (2, 3), None, {}, _Bad),
    ("rearrange", "a _b -> a _b", (2, 3), None, {}, _Bad),
    ("rearrange", "a b -> a $", (2, 3), None, {}, _Bad),
    ("rearrange", "time batch -> batch time", (2, 3), None, {}, (3, 2)),
    ("rearrange", "a1 b2 -> b2 a1", (2, 3), None, {}, (3, 2)),
    ("rearrange", "a b-> b a", (2, 3), None, {}, (3, 2)),
    # the reductions, and patterns models actually write
    ("reduce", "b c h -> b", (2, 3, 4), "sum", {}, (2,)),
    ("reduce", "b c h -> b c", (2, 3, 4), "mean", {}, (2, 3)),
    ("reduce", "b c h -> h b", (2, 3, 4), "min", {}, (4, 2)),
    ("reduce", "b c h -> b", (2, 3, 4), "prod", {}, (2,)),
    ("reduce", "b c -> b", (2, 3), "any", {}, (2,)),
    ("reduce", "b c -> c", (2, 3), "all", {}, (3,)),
    ("reduce", "b c -> ", (2, 3), "sum", {}, ()),
    ("reduce", "b c -> (b c)", (2, 3), "sum", {}, (6,)),
    ("reduce", "b c -> b c", (2, 3), "sum", {}, (2, 3)),
    ("reduce", "b c -> b", (0, 3), "sum", {}, (0,)),
    ("reduce", "b c -> b", (2, 3), "median", {}, _Bad),
    ("rearrange", "b n (h d) -> b h n d", (2, 5, 12), None, {"h": 3}, (2, 3, 5, 4)),
    ("rearrange", "b h n d -> b n (h d)", (2, 3, 5, 4), None, {}, (2, 5, 12)),
    (
        "rearrange",
        "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
        (2, 8, 8, 3),
        None,
        {"p1": 4, "p2": 4},
        (2, 4, 48),
    ),
    ("rearrange", "a b -> b a", (0, 3), None, {}, (3, 0)),
    ("repeat", "b h n d -> b (h r) n d", (2, 2, 3, 4), None, {"r": 2}, (2, 4, 3, 4)),
]


def _id(case: _Case) -> str:
    op, pattern, shape, reduction, kw, _ = case
    extra = "".join(f",{k}={v}" for k, v in kw.items())
    return f"{op}[{reduction or ''}]{pattern}{list(shape)}{extra}"


def _input(shape: tuple[int, ...], reduction: str | None) -> np.ndarray:
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    return x > 0 if reduction in ("any", "all") else x


def _call(module: Any, case: _Case, x: Any) -> Any:
    op, pattern, _, reduction, kw, _ = case
    if op == "reduce":
        return module.reduce(x, pattern, reduction, **kw)
    return getattr(module, op)(x, pattern, **kw)


@pytest.mark.parametrize("case", _CASES, ids=[_id(c) for c in _CASES])
def test_the_pattern_answers_as_the_reference_does(case: _Case, device: str) -> None:
    expected = case[-1]
    x = lucid.tensor(_input(case[2], case[3]), device=device)
    if isinstance(expected, type):
        with pytest.raises(expected):
            _call(le, case, x)
        return
    assert _call(le, case, x).shape == expected


# Refused here, accepted by the einops package: it uses U+2026 as its own
# ellipsis marker too, so a hand-written one leaks through as an ellipsis
# without the '...' bookkeeping ("a … -> … a" swaps the axes, while
# "… -> …" counts the marker as one axis).
_STRICTER_THAN_EINOPS = {"a … -> … a"}

_VALUE_CASES = [c for c in _CASES if not isinstance(c[-1], type)]
_REFUSED_CASES = [
    c for c in _CASES if isinstance(c[-1], type) and c[1] not in _STRICTER_THAN_EINOPS
]


def _split_heads(x: np.ndarray) -> np.ndarray:
    b, n, hd = x.shape
    return x.reshape(b, n, 3, hd // 3).transpose(0, 2, 1, 3)


def _merge_heads(x: np.ndarray) -> np.ndarray:
    b, h, n, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, n, h * d)


def _patchify(x: np.ndarray) -> np.ndarray:
    b, hp, wp, c = x.shape
    patches = x.reshape(b, hp // 4, 4, wp // 4, 4, c).transpose(0, 1, 3, 2, 4, 5)
    return patches.reshape(b, (hp // 4) * (wp // 4), 16 * c)


def _windows_max(x: np.ndarray) -> np.ndarray:
    h2, w2 = x.shape
    return x.reshape(h2 // 2, 2, w2 // 2, 2).max(axis=(1, 3))


# Order-sensitive rows, each against a numpy oracle: these hold on every
# run, where the einops package (and so the parity values) may be absent.
_ORACLES: list[tuple[_Case, Any]] = [
    (
        ("repeat", "h w -> (h 2) w", (2, 3), None, {}, None),
        lambda x: np.repeat(x, 2, 0),
    ),
    (
        ("repeat", "h w -> (2 h) w", (2, 3), None, {}, None),
        lambda x: np.tile(x, (2, 1)),
    ),
    (
        ("repeat", "h w -> (2 h) (w 3)", (2, 3), None, {}, None),
        lambda x: np.repeat(np.tile(x, (2, 1)), 3, 1),
    ),
    (
        ("repeat", "b h n d -> b (h r) n d", (2, 2, 3, 4), None, {"r": 2}, None),
        lambda x: np.repeat(x, 2, 1),
    ),
    (
        ("repeat", "h w -> w h c", (2, 3), None, {"c": 2}, None),
        lambda x: np.broadcast_to(x.T[..., None], (3, 2, 2)),
    ),
    (("reduce", "(h 2) (w 2) -> h w", (4, 6), "max", {}, None), _windows_max),
    (
        (
            "rearrange",
            "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
            (2, 8, 8, 3),
            None,
            {"p1": 4, "p2": 4},
            None,
        ),
        _patchify,
    ),
    (
        ("rearrange", "b n (h d) -> b h n d", (2, 5, 12), None, {"h": 3}, None),
        _split_heads,
    ),
    (("rearrange", "b h n d -> b n (h d)", (2, 3, 5, 4), None, {}, None), _merge_heads),
    (
        ("rearrange", "a b c -> c a b", (2, 3, 4), None, {}, None),
        lambda x: x.transpose(2, 0, 1),
    ),
    (
        ("rearrange", "a ... b -> b ... a", (2, 3, 4, 5), None, {}, None),
        lambda x: x.swapaxes(0, -1),
    ),
    (
        ("rearrange", "(a b) c -> b (c a)", (6, 4), None, {"b": 3}, None),
        lambda x: x.reshape(2, 3, 4).transpose(1, 2, 0).reshape(3, 8),
    ),
    (
        ("rearrange", "b ... -> (...) b", (2, 3, 4, 5), None, {}, None),
        lambda x: x.reshape(2, -1).T,
    ),
]


@pytest.mark.parametrize("case, oracle", _ORACLES, ids=[_id(c) for c, _ in _ORACLES])
def test_the_values_follow_the_pattern_s_order(case: _Case, oracle: Any, device: str) -> None:
    x = _input(case[2], case[3])
    got = _call(le, case, lucid.tensor(x, device=device)).numpy()
    np.testing.assert_array_equal(got, oracle(x))


@pytest.mark.parity
@pytest.mark.parametrize("case", _VALUE_CASES, ids=[_id(c) for c in _VALUE_CASES])
def test_values_match_the_einops_package(case: _Case, device: str) -> None:
    einops = pytest.importorskip("einops")
    x = _input(case[2], case[3])
    expected = np.asarray(_call(einops, case, x))
    got = _call(le, case, lucid.tensor(x, device=device)).numpy()
    assert got.shape == expected.shape
    assert got.dtype.kind == expected.dtype.kind
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("case", _REFUSED_CASES, ids=[_id(c) for c in _REFUSED_CASES])
def test_the_einops_package_refuses_the_same_patterns(case: _Case) -> None:
    einops = pytest.importorskip("einops")
    # ``ValueError`` is what the package raises for a missing or doubled
    # arrow (an unpacking error), before its own checks run.
    with pytest.raises((einops.EinopsError, ValueError)):
        _call(einops, case, _input(case[2], case[3]))


_GRAD_CASES: list[_Case] = [
    ("rearrange", "b n (h d) -> b h n d", (2, 5, 12), None, {"h": 3}, None),
    ("rearrange", "b ... -> b (...)", (2, 3, 4), None, {}, None),
    ("rearrange", "b 1 c -> c b", (2, 1, 3), None, {}, None),
    ("reduce", "(h 2) (w 2) -> h w", (4, 6), "max", {}, None),
    ("reduce", "b ... h -> b h", (2, 3, 4, 5), "mean", {}, None),
    ("reduce", "b c h -> h b", (2, 3, 4), "sum", {}, None),
    ("reduce", "b c h -> b", (2, 3, 4), "prod", {}, None),
    ("repeat", "h w -> (h 2) w c", (2, 3), None, {"c": 2}, None),
    ("repeat", "b ... -> b (c ...)", (2, 3, 4), None, {"c": 2}, None),
]


@pytest.mark.parity
@pytest.mark.parametrize("case", _GRAD_CASES, ids=[_id(c) for c in _GRAD_CASES])
def test_gradients_match_einops_over_the_reference(case: _Case, device: str, ref: Any) -> None:
    einops = pytest.importorskip("einops")
    x = _input(case[2], None)
    tx = ref.tensor(x, requires_grad=True)
    lx = lucid.tensor(x, device=device, requires_grad=True)
    ty = _call(einops, case, tx)
    ly = _call(le, case, lx)
    w = np.random.default_rng(1).standard_normal(tuple(ty.shape)).astype(np.float32)
    (ty * ref.tensor(w)).sum().backward()
    (ly * lucid.tensor(w, device=device)).sum().backward()
    assert lx.grad is not None
    np.testing.assert_allclose(lx.grad.numpy(), tx.grad.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "call, axis",
    [
        (lambda x: le.rearrange(x, "b c h -> b (c h)", h=7), "'h'"),
        (lambda x: le.rearrange(x, "b c h -> b (c h)", z=3), "'z'"),
        (lambda x: le.rearrange(x, "a b c -> a (b c) d"), "'d'"),
        (lambda x: le.rearrange(x, "a a c -> a c"), "'a'"),
        (lambda x: le.repeat(x, "a b c -> a b c n"), "'n'"),
        (lambda x: le.reduce(x, "a b c -> a e", "sum"), "'e'"),
    ],
    ids=["mismatch", "unused", "rhs-only", "duplicate", "unsized", "created"],
)
def test_a_refusal_names_the_axis_and_the_pattern(call: Any, axis: str) -> None:
    with pytest.raises(_C_engine.LucidError) as info:
        call(lucid.zeros(2, 3, 4))
    message = str(info.value)
    assert axis in message
    assert "einops." in message and "(2, 3, 4)" in message


def test_mean_of_an_integer_tensor_is_refused_as_the_reference_does() -> None:
    with pytest.raises(NotImplementedError, match="floating-point"):
        le.reduce(lucid.ones(2, 3, dtype=lucid.int32), "a b -> a", "mean")
    # Without an axis to reduce there is no mean to take.
    assert le.reduce(lucid.ones(2, 3, dtype=lucid.int32), "a b -> b a", "mean").shape == (
        3,
        2,
    )


@pytest.mark.parametrize("value", ["2", 2.0, None], ids=["str", "float", "none"])
def test_a_keyword_length_must_be_an_integer(value: object) -> None:
    lengths: dict[str, Any] = {"c": value}
    with pytest.raises(TypeError, match="'c'"):
        le.repeat(lucid.zeros(2), "h -> h c", **lengths)


def test_a_negative_keyword_length_is_refused() -> None:
    with pytest.raises(_Bad, match="non-negative"):
        le.repeat(lucid.zeros(2), "h -> h c", c=-1)


def test_a_repeated_call_reuses_its_plan() -> None:
    from lucid.einops import _recipe

    x = lucid.zeros(2, 3, 4)
    le.rearrange(x, "b c h -> b (c h)")
    before = _recipe._plan.cache_info().hits
    le.rearrange(x, "b c h -> b (c h)")
    assert _recipe._plan.cache_info().hits == before + 1
