"""``vmap``'s argument contract: what it accepts, what it refuses, and with what.

Each table crosses the calls that used to misbehave — ``randomness='same'``
drawing per element, an unvalidated ``randomness``, a batch of no elements,
``in_dims`` / ``out_dims`` of the wrong length, ``out_dims=None``, a
``chunk_size`` of 0, a non-Tensor result or argument, the vectorised
strategy with a ``chunk_size`` — with the value they must give, and (under
``-m parity``) with what the reference framework does for the same call.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.func as func
from lucid import Tensor
from lucid.func._pytree import StructureError

Fn = Callable[..., object]

_STRATEGIES = ("auto", "isolated", "vectorized")


def _close(got: object, want: object) -> None:
    assert isinstance(got, Tensor) and isinstance(want, Tensor)
    assert tuple(got.shape) == tuple(want.shape), (tuple(got.shape), tuple(want.shape))
    if got.numel():
        assert float((got - want).abs().max().item()) < 1e-6


def _grid(device: str) -> Tensor:
    return lucid.arange(12.0, device=device).reshape(4, 3)


# ── randomness ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["bogus", "", "Same", "ERROR", "none"])
def test_an_unknown_randomness_is_refused_where_the_transform_is_built(mode: str) -> None:
    with pytest.raises(RuntimeError, match="Only allowed values for randomness are"):
        func.vmap(lambda a: a, randomness=mode)
    with pytest.raises(RuntimeError, match="Only allowed values for randomness are"):
        func.jacfwd(lambda a: a, randomness=mode)


def _draw(device: str, **kwargs: object) -> tuple[Tensor, Tensor]:
    """``vmap(x + rand(3))`` over four zero rows, and the next draw after it."""
    lucid.manual_seed(0)
    out = func.vmap(lambda x: x + lucid.rand(3, device=device), **kwargs)(  # type: ignore[arg-type]
        lucid.zeros(4, 3, device=device)
    )
    assert isinstance(out, Tensor)
    return out, lucid.rand(1, device=device)


@pytest.mark.parametrize("strategy", _STRATEGIES)
@pytest.mark.parametrize("chunk_size", [None, 1, 3, 4])
def test_same_randomness_gives_every_element_the_first_draw(
    device: str, strategy: str, chunk_size: int | None
) -> None:
    lucid.manual_seed(0)
    one_draw = lucid.rand(3, device=device)
    after_one_call = lucid.rand(1, device=device)
    out, after = _draw(device, randomness="same", strategy=strategy, chunk_size=chunk_size)
    _close(out, one_draw.reshape(1, 3).expand(4, 3))
    # the generator stands where a single call leaves it
    _close(after, after_one_call)


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_different_randomness_gives_every_element_its_own_draw(device: str, strategy: str) -> None:
    out, _ = _draw(device, randomness="different", strategy=strategy)
    rows = [tuple(out[i].tolist()) for i in range(4)]
    assert len(set(rows)) == 4


def test_different_randomness_is_the_sequence_of_plain_draws(device: str) -> None:
    lucid.manual_seed(0)
    plain = lucid.stack([lucid.rand(3, device=device) for _ in range(4)])
    out, _ = _draw(device, randomness="different")
    _close(out, plain)


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_error_randomness_refuses_a_random_op(device: str, strategy: str) -> None:
    with pytest.raises(RuntimeError, match="random op"):
        _draw(device, strategy=strategy)
    with pytest.raises(RuntimeError, match="random op"):
        func.vmap(lambda x: x + lucid.rand(3), strategy=strategy)(lucid.zeros(0, 3))


def test_the_randomness_of_the_enclosing_call_is_restored(device: str) -> None:
    from lucid._ctx import get_randomness

    before = get_randomness()
    _draw(device, randomness="same")
    assert get_randomness() == before
    with pytest.raises(RuntimeError):
        _draw(device)
    assert get_randomness() == before


# ── a batch of no elements ───────────────────────────────────────────────────

_EMPTY: dict[str, tuple[Fn, dict[str, object], object]] = {
    "reduce": (lambda a: a.sum(), {}, (0,)),
    "elementwise": (lambda a: a * 2.0, {}, (0, 3)),
    "out_dims-1": (lambda a: a * 2.0, {"out_dims": 1}, (3, 0)),
    "out_dims--1": (lambda a: a * 2.0, {"out_dims": -1}, (3, 0)),
    "tuple": (lambda a: (a.sum(), a), {}, ((0,), (0, 3))),
    "dict": (lambda a: {"s": a.sum(), "a": a}, {}, {"s": (0,), "a": (0, 3)}),
    "in_dims-1": (lambda a: a.sum(), {"in_dims": 1}, (0,)),
    "chunked": (lambda a: a * 2.0, {"chunk_size": 2}, (0, 3)),
    "grad": (func.grad(lambda a: (a**2).sum()), {}, (0, 3)),
}


def _shapes(tree: object) -> object:
    if isinstance(tree, Tensor):
        return tuple(tree.shape)
    if isinstance(tree, tuple):
        return tuple(_shapes(t) for t in tree)
    assert isinstance(tree, dict)
    return {k: _shapes(v) for k, v in tree.items()}


@pytest.mark.parametrize("strategy", _STRATEGIES)
@pytest.mark.parametrize("name", list(_EMPTY))
def test_a_batch_of_no_elements_has_the_shapes_of_one_with_an_empty_batch_axis(
    device: str, name: str, strategy: str
) -> None:
    f, kwargs, want = _EMPTY[name]
    shape = (3, 0) if kwargs.get("in_dims") == 1 else (0, 3)
    got = func.vmap(f, strategy=strategy, **kwargs)(lucid.zeros(*shape, device=device))  # type: ignore[arg-type]
    assert _shapes(got) == want


@pytest.mark.parametrize("out_dim", [3, -4, 9])
def test_an_out_dim_outside_the_output_of_a_batch_of_no_elements_is_an_index_error(
    device: str, out_dim: int
) -> None:
    with pytest.raises(IndexError, match="out of range"):
        func.vmap(lambda a: a * 2.0, out_dims=out_dim)(lucid.zeros(0, 3, device=device))


def test_a_batch_of_no_elements_keeps_the_dtype_and_the_device(device: str) -> None:
    x = lucid.zeros(0, 3, device=device, dtype=lucid.float16)
    out = func.vmap(lambda a: a + 1)(x)
    assert isinstance(out, Tensor)
    assert out.dtype == x.dtype and out.device == x.device


def test_an_unbatched_output_of_a_batch_of_no_elements_is_what_func_returned(device: str) -> None:
    w = lucid.ones(2, device=device)
    out = func.vmap(lambda a, b: (a.sum(), b * 2.0), in_dims=(0, None), out_dims=(0, None))(
        lucid.zeros(0, 3, device=device), w
    )
    assert isinstance(out, tuple)
    assert tuple(out[0].shape) == (0,)
    _close(out[1], w * 2.0)


# ── in_dims ──────────────────────────────────────────────────────────────────

_BAD_IN_DIMS: dict[str, tuple[object, int]] = {
    "one-too-many": ((0, 0), 1),
    "one-too-few": ((0,), 2),
    "none-too-few": ((None,), 2),
    "nested-length": (((0, 0),), 1),
}


@pytest.mark.parametrize("name", list(_BAD_IN_DIMS))
def test_in_dims_of_the_wrong_length_is_refused(device: str, name: str) -> None:
    in_dims, n_args = _BAD_IN_DIMS[name]
    x = _grid(device)
    with pytest.raises(StructureError, match="in_dims is not compatible"):
        func.vmap(lambda *a: a[0], in_dims=in_dims)(*([x] * n_args))  # type: ignore[arg-type]


@pytest.mark.parametrize("in_dims", ["0", [0], {"x": 0}, 1.5, ("0",), (0, [0.5])])
def test_in_dims_that_are_not_dims_are_refused(in_dims: object) -> None:
    with pytest.raises(ValueError, match="in_dims"):
        func.vmap(lambda a: a, in_dims=in_dims)  # type: ignore[arg-type]


def test_in_dims_of_the_wrong_length_is_a_value_error() -> None:
    assert issubclass(StructureError, ValueError)


@pytest.mark.parametrize("in_dim", [2, -3, 5])
def test_an_in_dim_outside_the_tensor_is_refused(device: str, in_dim: int) -> None:
    with pytest.raises(ValueError, match="expected in_dim to satisfy"):
        func.vmap(lambda a: a, in_dims=in_dim)(_grid(device))


def test_arguments_with_different_batch_sizes_are_refused(device: str) -> None:
    with pytest.raises(ValueError, match="same size in the mapped dimension"):
        func.vmap(lambda a, b: a)(_grid(device), lucid.zeros(5, 3, device=device))


@pytest.mark.parametrize(
    "scalar", [2.0, 3, "tag", None, True], ids=["float", "int", "str", "none", "bool"]
)
def test_a_non_tensor_given_an_in_dim_is_a_value_error(device: str, scalar: object) -> None:
    with pytest.raises(ValueError, match="cannot vmap over non-Tensor arguments"):
        func.vmap(lambda a, b: a, in_dims=0)(_grid(device), scalar)
    got = func.vmap(lambda a, b: a, in_dims=(0, None))(_grid(device), scalar)
    _close(got, _grid(device))


# ── out_dims ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("out_dims", [(0, 0, 0), (0,), ((0, 0), 0), {"a": 0}, [0, 0, 0]], ids=str)
def test_out_dims_that_do_not_fit_the_result_are_refused(device: str, out_dims: object) -> None:
    with pytest.raises(StructureError, match="out_dims is not compatible"):
        func.vmap(lambda a: (a, a), out_dims=out_dims)(_grid(device))  # type: ignore[arg-type]


def test_a_tuple_of_out_dims_for_a_lone_tensor_is_refused(device: str) -> None:
    with pytest.raises(StructureError, match="out_dims is not compatible"):
        func.vmap(lambda a: a, out_dims=(0, 0))(_grid(device))
    _close(func.vmap(lambda a: a, out_dims=(1,))(_grid(device)), _grid(device).T)


@pytest.mark.parametrize("out_dims", ["0", 1.5, (0, "1"), {"a": None, "b": [1.0]}])
def test_out_dims_that_are_not_dims_are_refused(out_dims: object) -> None:
    with pytest.raises(ValueError, match="`out_dims` must be an int, None or"):
        func.vmap(lambda a: a, out_dims=out_dims)  # type: ignore[arg-type]


@pytest.mark.parametrize("strategy", _STRATEGIES)
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_an_output_with_out_dim_none_is_not_batched(
    device: str, strategy: str, chunk_size: int | None
) -> None:
    x = _grid(device)
    w = lucid.full((2,), 7.0, device=device)
    got = func.vmap(
        lambda a: (a * 2.0, w * 3.0),
        out_dims=(0, None),
        strategy=strategy,
        chunk_size=chunk_size,
    )(x)
    assert isinstance(got, tuple)
    _close(got[0], x * 2.0)
    _close(got[1], w * 3.0)


def test_out_dims_none_for_every_output(device: str) -> None:
    w = lucid.full((2,), 7.0, device=device)
    got = func.vmap(lambda a: w + 1.0, out_dims=None)(_grid(device))
    _close(got, w + 1.0)


def test_an_output_with_out_dim_none_may_be_a_non_tensor(device: str) -> None:
    got = func.vmap(lambda a: (a, "label", 2.5), out_dims=(0, None, None))(_grid(device))
    assert isinstance(got, tuple)
    _close(got[0], _grid(device))
    assert got[1:] == ("label", 2.5)


@pytest.mark.parametrize("strategy", ["auto", "isolated"])
def test_an_output_with_out_dim_none_that_depends_on_the_batch_is_refused(
    device: str, strategy: str
) -> None:
    with pytest.raises(ValueError, match="depends on the batch where out_dims is None"):
        func.vmap(lambda a: a.sum(), out_dims=None, strategy=strategy)(_grid(device))
    with pytest.raises(ValueError, match="depends on the batch where out_dims is None"):
        func.vmap(lambda a: a, out_dims=None, chunk_size=2)(_grid(device))


def test_an_output_with_out_dim_none_whose_shape_depends_on_the_batch_is_refused(
    device: str,
) -> None:
    sizes = lucid.tensor([1.0, 2.0], device=device)
    with pytest.raises(ValueError, match="depends on the batch where out_dims is None"):
        func.vmap(lambda n: lucid.zeros(int(n.item())), out_dims=None)(sizes)


def test_an_output_with_out_dim_none_that_is_nan_is_unbatched(device: str) -> None:
    nan = lucid.full((2,), float("nan"), device=device)
    got = func.vmap(lambda a: nan, out_dims=None)(_grid(device))
    assert isinstance(got, Tensor) and bool(got.isnan().all().item())


@pytest.mark.parametrize(
    "value", [3.0, 3, "s", None, [1.0], {"k": 1.0}, (1, lucid.ones(2))], ids=str
)
def test_a_result_that_is_not_made_of_tensors_is_a_value_error(device: str, value: object) -> None:
    for strategy in _STRATEGIES:
        with pytest.raises(ValueError, match="func must only return Tensors"):
            func.vmap(lambda a: value, strategy=strategy)(_grid(device))
    with pytest.raises(ValueError, match="func must only return Tensors"):
        func.vmap(lambda a: value)(lucid.zeros(0, 3, device=device))


# ── chunk_size ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("chunk_size", [0, -1, -7])
def test_a_chunk_size_that_is_not_positive_is_refused(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size should be None or greater than 0"):
        func.vmap(lambda a: a, chunk_size=chunk_size)


@pytest.mark.parametrize("strategy", ["vectorized", "auto"])
@pytest.mark.parametrize("in_dim", [0, 1, -1])
@pytest.mark.parametrize("out_dim", [0, 1, -1])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7])
def test_chunked_batches_equal_the_unchunked_one(
    device: str, strategy: str, in_dim: int, out_dim: int, chunk_size: int
) -> None:
    n = 5
    shape = (n, 3) if in_dim == 0 else (3, n)
    x = lucid.arange(15.0, device=device).reshape(*shape)
    f = lambda a: a * 2.0 + 1.0  # noqa: E731
    want = func.vmap(f, in_dims=in_dim, out_dims=out_dim, strategy=strategy)(x)
    got = func.vmap(f, in_dims=in_dim, out_dims=out_dim, strategy=strategy, chunk_size=chunk_size)(
        x
    )
    _close(got, want)
    expected = (x.moveaxis(in_dim, 0) * 2.0 + 1.0).moveaxis(0, out_dim)
    _close(got, expected)


def test_a_vectorised_chunk_sees_the_batch_at_the_front(device: str) -> None:
    seen: list[tuple[int, ...]] = []

    def f(a: Tensor) -> Tensor:
        seen.append(tuple(a.shape))
        return a

    x = lucid.zeros(3, 6, device=device)
    func.vmap(f, in_dims=1, strategy="vectorized", chunk_size=4)(x)
    assert seen == [(4, 3), (2, 3)]


# ── strategy ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["Auto", "vectorised", "", "loop"])
def test_an_unknown_strategy_is_refused(strategy: str) -> None:
    with pytest.raises(ValueError, match="strategy must be"):
        func.vmap(lambda a: a, strategy=strategy)


def test_the_arguments_of_the_transform_are_readable() -> None:
    v = func.vmap(lambda a: a, randomness="same", chunk_size=2, strategy="isolated")
    assert (v.randomness, v.chunk_size, v.strategy) == ("same", 2, "isolated")  # type: ignore[attr-defined]


# ── the reference framework refuses and accepts the same calls ──────────────

_Call = Callable[[ModuleType], object]


def _fail(m: ModuleType, build: Callable[[ModuleType], object]) -> type[BaseException]:
    try:
        build(m)
    except Exception as error:  # the class is what is compared
        return type(error)
    raise AssertionError("the call did not fail")


def _x(m: ModuleType, *shape: int) -> object:
    return m.ones(*shape)


_REFUSED: dict[str, Callable[[ModuleType], object]] = {
    "randomness": lambda m: m.func.vmap(lambda a: a, randomness="bogus")(_x(m, 3, 2)),
    "in_dims-long": lambda m: m.func.vmap(lambda a: a, in_dims=(0, 0))(_x(m, 3, 2)),
    "in_dims-short": lambda m: m.func.vmap(lambda a, b: a, in_dims=(0,))(_x(m, 3, 2), _x(m, 3, 2)),
    "out_dims-long": lambda m: m.func.vmap(lambda a: (a, a), out_dims=(0, 0, 0))(_x(m, 3, 2)),
    "out_dims-lone": lambda m: m.func.vmap(lambda a: a, out_dims=(0, 0))(_x(m, 3, 2)),
    "chunk-zero": lambda m: m.func.vmap(lambda a: a, chunk_size=0)(_x(m, 3, 2)),
    "chunk-negative": lambda m: m.func.vmap(lambda a: a, chunk_size=-1)(_x(m, 3, 2)),
    "float-result": lambda m: m.func.vmap(lambda a: 3.0)(_x(m, 3, 2)),
    "none-result": lambda m: m.func.vmap(lambda a: None)(_x(m, 3, 2)),
    "float-arg": lambda m: m.func.vmap(lambda a, b: a, in_dims=0)(_x(m, 3, 2), 2.0),
    "in_dim-range": lambda m: m.func.vmap(lambda a: a, in_dims=2)(_x(m, 3, 2)),
    "batch-sizes": lambda m: m.func.vmap(lambda a, b: a)(_x(m, 3, 2), _x(m, 4, 2)),
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_REFUSED))
def test_vmap_refuses_what_the_reference_refuses_with_the_same_class(
    ref: ModuleType, name: str
) -> None:
    build = _REFUSED[name]
    ours, theirs = _fail(lucid, build), _fail(ref, build)
    assert issubclass(ours, theirs), (ours, theirs)


_ACCEPTED: dict[str, Callable[[ModuleType], object]] = {
    "empty-reduce": lambda m: m.func.vmap(lambda a: a.sum())(m.zeros(0, 3)),
    "empty-elementwise": lambda m: m.func.vmap(lambda a: a * 2.0)(m.zeros(0, 3)),
    "empty-out_dims": lambda m: m.func.vmap(lambda a: a * 2.0, out_dims=1)(m.zeros(0, 3)),
    "empty-two-args": lambda m: m.func.vmap(lambda a, b: a + b, in_dims=(0, None))(
        m.zeros(0, 3), m.ones(3)
    ),
    "out_dims-none-tuple": lambda m: m.func.vmap(lambda a: (a, m.ones(2)), out_dims=(0, None))(
        m.arange(6.0).reshape(3, 2)
    ),
    "same-rows-agree": lambda m: m.func.vmap(lambda a: a + m.rand(3), randomness="same")(
        m.zeros(4, 3)
    ).std(0),
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_ACCEPTED))
def test_vmap_gives_what_the_reference_gives(ref: ModuleType, name: str) -> None:
    got, want = _ACCEPTED[name](lucid), _ACCEPTED[name](ref)

    def listed(tree: object) -> object:
        if isinstance(tree, (tuple, list)):
            return [listed(t) for t in tree]
        shape = tuple(tree.shape)  # type: ignore[attr-defined]
        return shape, tree.tolist()  # type: ignore[attr-defined]

    assert listed(got) == listed(want)
