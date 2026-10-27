"""``fill_`` and ``zero_`` write a view whose elements overlap; other writes do not.

Two elements of an expanded view, of unfold windows that overlap, or of an
``as_strided`` view that revisits a position are one byte.  Because Lucid
computes in-place results out of place and scatters them back into the target
view on both CPU and Metal, a value-dependent write has no single value to
leave at a shared position and is refused for every self-overlapping view
(deliberately stricter than the reference, which refuses only stride-0 overlap
and sequentially compounds overlapping ``unfold`` / ``as_strided`` elements in
storage order — LCD-537).  A fill leaves the same value in every element, so
the elements that share a byte agree; the reference lets it through, and Lucid
does too on both devices (LCD-28).
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

_NO_METAL = pytest.mark.skipif(not metal_available(), reason="needs Metal")

#: name -> (base values, the overlapping view of the base, base indices it reads)
_VIEWS: dict[str, tuple[list[float], Callable[[lucid.Tensor], lucid.Tensor], set[int]]] = {
    "expand": ([1.0, 2.0, 3.0], lambda b: b.reshape(1, 3).expand(2, 3), {0, 1, 2}),
    "unfold": ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lambda b: b.unfold(0, 3, 2), {0, 1, 2, 3, 4}),
    "as_strided": (
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        lambda b: b.as_strided((3, 3), (1, 1)),
        {0, 1, 2, 3, 4},
    ),
}

_FILLS: dict[str, tuple[Callable[[lucid.Tensor], object], float]] = {
    "fill_": (lambda v: v.fill_(2.5), 2.5),
    "zero_": (lambda v: v.zero_(), 0.0),
}

_REFUSED: dict[str, Callable[[lucid.Tensor], object]] = {
    "add_": lambda v: v.add_(1.0),
    "mul_": lambda v: v.mul_(2.0),
    "clamp_": lambda v: v.clamp_(0.0, 1.5),
    "copy_": lambda v: v.copy_(lucid.ones_like(v)),
}


def _devices() -> list[object]:
    return ["cpu", pytest.param("metal", marks=[_NO_METAL])]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("fill", list(_FILLS))
@pytest.mark.parametrize("view", list(_VIEWS))
def test_a_fill_reaches_every_element_the_view_reads(view: str, fill: str, device: str) -> None:
    values, make, read = _VIEWS[view]
    write, value = _FILLS[fill]
    base = lucid.tensor(values, device=device)
    v = make(base)
    write(v)
    want = [value if i in read else x for i, x in enumerate(values)]
    assert base.to("cpu").tolist() == want
    assert all(x == value for x in v.contiguous().reshape(-1).to("cpu").tolist())


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", ["float16", "int64", "bool"])
def test_a_fill_into_an_expanded_view_keeps_the_dtype(dtype: str, device: str) -> None:
    dt = getattr(lucid, dtype)
    base = lucid.zeros(1, 3, dtype=dt, device=device)
    base.expand(4, 3).fill_(1)
    assert base.dtype == dt
    assert base.to("cpu").tolist() == [[1, 1, 1]]


@pytest.mark.parametrize("device", _devices())
def test_a_fill_into_an_expanded_view_whose_base_is_gone(device: str) -> None:
    v = lucid.zeros(1, 3, device=device).expand(2, 3)
    v.fill_(5.0)
    assert v.to("cpu").tolist() == [[5.0] * 3] * 2
    v.zero_()
    assert v.to("cpu").tolist() == [[0.0] * 3] * 2


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("op", list(_REFUSED))
@pytest.mark.parametrize("view", list(_VIEWS))
def test_a_value_dependent_write_is_refused_and_writes_nothing(
    view: str, op: str, device: str
) -> None:
    values, make, _ = _VIEWS[view]
    base = lucid.tensor(values, device=device)
    with pytest.raises(RuntimeError, match="overlap"):
        _REFUSED[op](make(base))
    assert base.to("cpu").tolist() == values


@pytest.mark.parametrize("device", _devices())
def test_a_fill_into_a_view_of_a_leaf_that_requires_grad_is_still_refused(device: str) -> None:
    w = lucid.zeros(1, 3, device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match="leaf"):
        w.expand(2, 3).fill_(1.0)
    assert w.to("cpu").tolist() == [[0.0] * 3]


@pytest.mark.parity
@pytest.mark.parametrize("fill", list(_FILLS))
@pytest.mark.parametrize("view", list(_VIEWS))
def test_the_fill_matches_the_reference(view: str, fill: str, ref: ModuleType) -> None:
    values, make, _ = _VIEWS[view]
    write, _ = _FILLS[fill]
    base = lucid.tensor(values)
    write(make(base))
    r = ref.tensor(values)
    write(make(r))
    assert base.tolist() == r.tolist()


@pytest.mark.parity
@pytest.mark.parametrize("device", _devices())
def test_overlapping_unfold_and_as_strided_write_refusal_is_a_documented_departure(
    device: str, ref: ModuleType
) -> None:
    """Pin the LCD-537 contract: Lucid refuses all self-overlapping value writes.

    The reference refuses stride-0 ``expand`` writes, but allows overlapping
    ``unfold`` and ``as_strided`` writes by compounding elements sequentially in
    storage order.  Lucid refuses all three on both CPU and Metal and leaves the
    base tensor unmodified.
    """
    r_exp = ref.arange(3.0).reshape(1, 3).expand(2, 3)
    with pytest.raises(RuntimeError):
        r_exp.add_(100.0)
    l_exp_base = lucid.arange(3.0, device=device)
    with pytest.raises(RuntimeError, match="overlap"):
        l_exp_base.reshape(1, 3).expand(2, 3).add_(100.0)
    assert l_exp_base.to("cpu").tolist() == [0.0, 1.0, 2.0]

    r_unf = ref.arange(6.0)
    r_unf.unfold(0, 2, 1).add_(100.0)
    assert r_unf.tolist() == [100.0, 201.0, 202.0, 203.0, 204.0, 105.0]

    for make in (lambda b: b.unfold(0, 2, 1), lambda b: b.as_strided((5, 2), (1, 1))):
        base = lucid.arange(6.0, device=device)
        with pytest.raises(RuntimeError, match="overlap"):
            make(base).add_(100.0)
        assert base.to("cpu").tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
