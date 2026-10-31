"""The einops entry points take what the einops package takes (LCD-768).

Lists of tensors, callable reductions, ``pack`` / ``unpack`` and
``parse_shape`` are held to the package's answers: the same shapes, the same
values, the same refusals.  ``einsum`` additionally takes its operands as one
list, as the reference does.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.einops as le
from lucid._C import engine as _C_engine


def _arrays(*shapes: tuple[int, ...]) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


def _tensors(arrays: list[np.ndarray], device: str) -> list[lucid.Tensor]:
    return [lucid.tensor(a, device=device) for a in arrays]


# ── lists of tensors ─────────────────────────────────────────────────────────

_LIST_CASES = [
    ("rearrange", "b h w -> b (h w)", None, {}),
    ("rearrange", "b h w -> h (b w)", None, {}),
    ("reduce", "b h w -> h", "sum", {}),
    ("reduce", "b h w -> b", "max", {}),
    ("repeat", "b h w -> b h w c", None, {"c": 2}),
]


def _call(module: Any, case: tuple[Any, ...], x: Any) -> Any:
    name, pattern, reduction, kwargs = case
    if name == "reduce":
        return module.reduce(x, pattern, reduction, **kwargs)
    return getattr(module, name)(x, pattern, **kwargs)


@pytest.mark.parametrize("case", _LIST_CASES, ids=[f"{c[0]}:{c[1]}" for c in _LIST_CASES])
def test_a_list_of_tensors_is_stacked_on_a_new_leading_axis(
    case: tuple[Any, ...], device: str
) -> None:
    arrays = _arrays((2, 3), (2, 3), (2, 3), (2, 3))
    got = _call(le, case, _tensors(arrays, device))
    want = _call(le, case, lucid.tensor(np.stack(arrays), device=device))
    np.testing.assert_array_equal(got.numpy(), want.numpy())


@pytest.mark.parity
@pytest.mark.parametrize("case", _LIST_CASES, ids=[f"{c[0]}:{c[1]}" for c in _LIST_CASES])
def test_a_list_of_tensors_matches_the_einops_package(case: tuple[Any, ...], device: str) -> None:
    einops = pytest.importorskip("einops")
    arrays = _arrays((2, 3), (2, 3), (2, 3), (2, 3))
    got = _call(le, case, _tensors(arrays, device)).numpy()
    want = np.asarray(_call(einops, case, arrays))
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_a_list_keeps_the_autograd_graph(device: str) -> None:
    xs = [lucid.ones(2, 3, device=device).requires_grad_(True) for _ in range(3)]
    le.reduce(xs, "b h w -> h", "sum").sum().backward()
    assert all(x.grad is not None and x.grad.tolist() == [[1.0] * 3] * 2 for x in xs)


def test_an_empty_list_is_refused() -> None:
    with pytest.raises(TypeError, match="empty list"):
        le.rearrange([], "b h -> b h")


def test_a_list_of_non_tensors_is_refused() -> None:
    with pytest.raises(TypeError, match="not a Tensor"):
        le.rearrange([lucid.ones(2), 3], "b h -> b h")


def test_a_tuple_is_not_a_list_of_tensors() -> None:
    # The package takes only a list: anything else is a single tensor.
    with pytest.raises(TypeError, match="Expected Tensor"):
        le.rearrange((lucid.ones(2), lucid.ones(2)), "b h -> b h")  # type: ignore[arg-type]


# ── callable reductions ──────────────────────────────────────────────────────

# pattern, shape the callable receives (reduced axes moved last), axes it is told
_CALLABLE_PATTERNS = [
    ("a b c -> c a", (4, 2, 3), (2,)),
    ("a b c -> b", (3, 2, 4), (1, 2)),
    ("a b c -> a b c", None, None),
    ("a b c -> a c b", None, None),
    ("a b c -> 1 c", (4, 2, 3), (1, 2)),
    ("a b c -> ", (2, 3, 4), (0, 1, 2)),
]


@pytest.mark.parametrize(
    ("pattern", "moved", "axes"), _CALLABLE_PATTERNS, ids=[c[0] for c in _CALLABLE_PATTERNS]
)
def test_a_callable_is_told_the_trailing_axes_of_the_moved_tensor(
    pattern: str, moved: tuple[int, ...] | None, axes: tuple[int, ...] | None, device: str
) -> None:
    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def total(x: lucid.Tensor, reduced: tuple[int, ...]) -> lucid.Tensor:
        seen.append((tuple(x.shape), reduced))
        return lucid.sum(x, dim=list(reduced))

    x = lucid.tensor(_arrays((2, 3, 4))[0], device=device)
    got = le.reduce(x, pattern, total)
    want = le.reduce(x, pattern, "sum")
    assert (seen == [(moved, axes)]) if moved is not None else not seen
    assert got.shape == want.shape
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-5)


@pytest.mark.parity
@pytest.mark.parametrize("pattern", [c[0] for c in _CALLABLE_PATTERNS])
def test_a_callable_sees_what_the_einops_package_shows_it(pattern: str, device: str) -> None:
    einops = pytest.importorskip("einops")
    ours: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    theirs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def mine(x: lucid.Tensor, axes: tuple[int, ...]) -> lucid.Tensor:
        ours.append((tuple(x.shape), tuple(axes)))
        return lucid.sum(x, dim=list(axes))

    def package(x: Any, axes: tuple[int, ...]) -> Any:
        theirs.append((tuple(x.shape), tuple(axes)))
        return x.sum(axes)

    (array,) = _arrays((2, 3, 4))
    got = le.reduce(lucid.tensor(array, device=device), pattern, mine)
    want = einops.reduce(array, pattern, package)
    assert ours == theirs
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5)


def test_a_callable_is_not_called_when_nothing_is_reduced() -> None:
    def never(x: lucid.Tensor, axes: tuple[int, ...]) -> lucid.Tensor:
        raise AssertionError("no axis is reduced")

    out = le.reduce(lucid.ones(2, 3), "a b -> b a", never)
    assert out.shape == (3, 2)


def test_a_callable_must_return_a_tensor() -> None:
    with pytest.raises(TypeError, match="not a Tensor"):
        le.reduce(lucid.ones(2, 3), "a b -> a", lambda x, axes: 3)  # type: ignore[arg-type,return-value]


def test_a_callable_follows_the_same_pattern_rules() -> None:
    with pytest.raises(_C_engine.InvalidArgument):
        le.reduce(lucid.ones(2), "a -> a b", lambda x, axes: x)


@pytest.mark.parametrize("name", ["<callable>", "", "None", "callable"])
def test_a_string_that_is_not_a_reduction_is_still_refused(name: str) -> None:
    with pytest.raises(_C_engine.InvalidArgument, match="unknown reduction"):
        le.reduce(lucid.ones(2, 2), "a b -> a", name)


# ── parse_shape ──────────────────────────────────────────────────────────────

_PARSE_CASES = [
    ((2, 3, 5, 7), "batch _ h w"),
    ((2, 3, 5, 7), "b ... w"),
    ((2, 3, 5, 7), "... w"),
    ((2, 3), "_ _"),
    ((2, 1, 3), "b 1 c"),
    ((2, 1, 3), "b () c"),
    ((2, 4, 3), "b 4 c"),
    ((5,), "..."),
    ((5,), "a ..."),
    ((2, 3, 4), "a ... b c"),
]

_PARSE_REFUSED = [
    ((2, 3), "a"),
    ((2, 3), "a b c"),
    ((2, 3), "a a"),
    ((2, 3), "(a b)"),
    ((2, 3), "a 2"),
    ((2, 3), "1 b"),
    ((2, 3), "a ->"),
    ((2, 3), "_a b"),
    ((2, 3), "a.b"),
]


@pytest.mark.parametrize(("shape", "pattern"), _PARSE_CASES)
def test_parse_shape_matches_the_documented_reading(shape: tuple[int, ...], pattern: str) -> None:
    got = le.parse_shape(lucid.zeros(*shape), pattern)
    assert all(isinstance(v, int) for v in got.values())
    assert "_" not in got


@pytest.mark.parity
@pytest.mark.parametrize(("shape", "pattern"), _PARSE_CASES)
def test_parse_shape_matches_the_einops_package(shape: tuple[int, ...], pattern: str) -> None:
    einops = pytest.importorskip("einops")
    assert le.parse_shape(lucid.zeros(*shape), pattern) == einops.parse_shape(
        np.zeros(shape), pattern
    )


@pytest.mark.parity
@pytest.mark.parametrize(("shape", "pattern"), _PARSE_REFUSED)
def test_parse_shape_refuses_what_the_einops_package_refuses(
    shape: tuple[int, ...], pattern: str
) -> None:
    einops = pytest.importorskip("einops")
    with pytest.raises((einops.EinopsError, RuntimeError)):
        einops.parse_shape(np.zeros(shape), pattern)
    with pytest.raises((_C_engine.InvalidArgument, _C_engine.ShapeMismatch)):
        le.parse_shape(lucid.zeros(*shape), pattern)


def test_parse_shape_feeds_the_other_entry_points() -> None:
    x = lucid.zeros(2, 3, 5, 7)
    y = lucid.zeros(70)
    assert le.rearrange(y, "(c h w) -> c h w", **le.parse_shape(x, "_ _ h w")).shape == (2, 5, 7)


def test_parse_shape_names_the_pattern_and_the_shape() -> None:
    with pytest.raises(_C_engine.ShapeMismatch, match=r"parse_shape\('a b c'\).*\(2, 3\)"):
        le.parse_shape(lucid.zeros(2, 3), "a b c")


# ── pack / unpack ────────────────────────────────────────────────────────────

_PACK_CASES = [
    ("i j * k", [(2, 3, 5), (2, 3, 7, 5), (2, 3, 7, 9, 5)]),
    ("b *", [(4, 2), (4,), (4, 3, 5)]),
    ("* c", [(6,), (2, 3, 6), (9, 6)]),
    ("*", [(), (3,), (2, 2)]),
    ("a * b", [(2, 4), (2, 0, 4), (2, 3, 4)]),
]


@pytest.mark.parametrize(("pattern", "shapes"), _PACK_CASES)
def test_pack_then_unpack_returns_the_inputs(
    pattern: str, shapes: list[tuple[int, ...]], device: str
) -> None:
    ins = _tensors(_arrays(*shapes), device)
    packed, shapes_back = le.pack(ins, pattern)
    outs = le.unpack(packed, shapes_back, pattern)
    assert len(outs) == len(ins)
    for out, original in zip(outs, ins, strict=True):
        assert out.shape == original.shape
        np.testing.assert_array_equal(out.numpy(), original.numpy())


@pytest.mark.parity
@pytest.mark.parametrize(("pattern", "shapes"), _PACK_CASES)
def test_pack_and_unpack_match_the_einops_package(
    pattern: str, shapes: list[tuple[int, ...]], device: str
) -> None:
    einops = pytest.importorskip("einops")
    arrays = _arrays(*shapes)
    packed, covered = le.pack(_tensors(arrays, device), pattern)
    want, want_covered = einops.pack(arrays, pattern)
    assert covered == [tuple(s) for s in want_covered]
    assert packed.shape == want.shape
    np.testing.assert_array_equal(packed.numpy(), want)
    for got, expected in zip(
        le.unpack(packed, covered, pattern), einops.unpack(want, want_covered, pattern), strict=True
    ):
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got.numpy(), expected)


def test_unpack_infers_one_unknown_length(device: str) -> None:
    ins = _tensors(_arrays((2, 3, 5), (2, 3, 7, 5), (2, 3, 7, 9, 5)), device)
    packed, _ = le.pack(ins, "i j * k")
    outs = le.unpack(packed, [(), (7,), (-1, 9)], "i j * k")
    assert [tuple(o.shape) for o in outs] == [(2, 3, 5), (2, 3, 7, 5), (2, 3, 7, 9, 5)]


def test_unpack_infers_a_leading_unknown_length(device: str) -> None:
    ins = _tensors(_arrays((2, 3, 5), (2, 3, 7, 5), (2, 3, 7, 9, 5)), device)
    packed, _ = le.pack(ins, "i j * k")
    outs = le.unpack(packed, [(-1,), (7,), (7, 9)], "i j * k")
    assert [tuple(o.shape) for o in outs] == [(2, 3, 1, 5), (2, 3, 7, 5), (2, 3, 7, 9, 5)]


@pytest.mark.parity
def test_unpack_infers_the_unknown_length_as_the_einops_package_does(device: str) -> None:
    einops = pytest.importorskip("einops")
    arrays = _arrays((2, 3, 5), (2, 3, 7, 5), (2, 3, 7, 9, 5))
    packed, _ = le.pack(_tensors(arrays, device), "i j * k")
    for parts in ([(), (7,), (-1, 9)], [(-1,), (7,), (7, 9)], [(), (-1,), (7, 9)]):
        got = le.unpack(packed, parts, "i j * k")
        want = einops.unpack(packed.numpy(), parts, "i j * k")
        for g, w in zip(got, want, strict=True):
            assert tuple(g.shape) == w.shape
            np.testing.assert_array_equal(g.numpy(), w)


def test_pack_keeps_the_autograd_graph(device: str) -> None:
    a = lucid.ones(2, 3, device=device).requires_grad_(True)
    b = lucid.ones(2, 3, 4, device=device).requires_grad_(True)
    packed, covered = le.pack([a, b], "n *")
    first, second = le.unpack(packed, covered, "n *")
    (first.sum() + 2 * second.sum()).backward()
    assert a.grad is not None and b.grad is not None
    assert a.grad.tolist() == [[1.0] * 3] * 2
    assert b.grad.numpy().min() == 2.0


@pytest.mark.parametrize("pattern", ["i j k", "i * *", "i * i", "i _ *", "1 *", "i j-k *", "(i) *"])
def test_pack_refuses_a_bad_pattern(pattern: str) -> None:
    with pytest.raises(_C_engine.InvalidArgument):
        le.pack([lucid.ones(2, 3, 4)], pattern)
    with pytest.raises(_C_engine.InvalidArgument):
        le.unpack(lucid.ones(2, 3, 4), [(3,)], pattern)


def test_pack_refuses_too_few_axes() -> None:
    with pytest.raises(_C_engine.ShapeMismatch, match="at least 3"):
        le.pack([lucid.ones(2, 3, 4), lucid.ones(2)], "a b * c")


def test_pack_refuses_mismatched_axes_around_the_star() -> None:
    with pytest.raises(_C_engine.ShapeMismatch):
        le.pack([lucid.ones(2, 3), lucid.ones(4, 3)], "a *")


def test_pack_refuses_no_tensors_and_non_tensors() -> None:
    with pytest.raises(_C_engine.InvalidArgument, match="no tensors"):
        le.pack([], "a *")
    with pytest.raises(TypeError, match="not a Tensor"):
        le.pack([lucid.ones(2, 3), 5], "a *")  # type: ignore[list-item]


def test_unpack_refuses_the_wrong_rank() -> None:
    with pytest.raises(_C_engine.ShapeMismatch, match="names 3 axes"):
        le.unpack(lucid.ones(2, 3), [(3,)], "a * b")


@pytest.mark.parametrize(
    "parts", [[(2,), (2,)], [(5,)], [(-1,), (-1,)], [(-1, 7)], [(-1, 0)], [(-2,)], [(1, -1, -1)]]
)
def test_unpack_refuses_parts_that_do_not_add_up(parts: list[tuple[int, ...]]) -> None:
    with pytest.raises((_C_engine.InvalidArgument, _C_engine.ShapeMismatch)):
        le.unpack(lucid.ones(2, 3), parts, "a *")


# ── einsum operands as one list ──────────────────────────────────────────────


@pytest.mark.parametrize("wrap", [list, tuple])
def test_einsum_takes_its_operands_as_one_sequence(wrap: type, device: str) -> None:
    a, b = _tensors(_arrays((2, 3), (3, 4)), device)
    spread = le.einsum("ij,jk->ik", a, b)
    gathered = le.einsum("ij,jk->ik", wrap([a, b]))
    np.testing.assert_array_equal(gathered.numpy(), spread.numpy())


def test_einsum_takes_a_single_operand_in_a_list(device: str) -> None:
    (a,) = _tensors(_arrays((3, 3)), device)
    np.testing.assert_array_equal(le.einsum("ii->i", [a]).numpy(), le.einsum("ii->i", a).numpy())


def test_einsum_still_refuses_the_wrong_operand_count() -> None:
    with pytest.raises(_C_engine.InvalidArgument):
        le.einsum("ij,jk->ik", [lucid.ones(2, 3)])


@pytest.mark.parity
def test_einsum_takes_a_list_as_the_reference_does(ref: Any, device: str) -> None:
    arrays = _arrays((2, 3), (3, 4))
    want = ref.einsum("ij,jk->ik", [ref.tensor(a) for a in arrays]).numpy()
    got = le.einsum("ij,jk->ik", _tensors(arrays, device)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5)
