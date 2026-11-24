"""The vector products refuse a vector that is not 1-D, as the reference does (LCD-1061).

``dot`` and ``vdot`` take two 1-D tensors.  The engine's CPU kernel answered a pair of
2-D tensors (``int8 (1, 1) . (1, 1)`` was ``tensor([[1]])``) and refused a 0-d pair with
a kernel message.  The reference refuses everything but 1-D with a ``RuntimeError``:
``1D tensors expected, but got 2D and 2D tensors``.  Every spelling goes through one
adapter (:mod:`lucid._ops._products`), so ``Tensor.dot``, ``lucid.linalg.dot`` and ``vdot``
(which forwards to ``dot``) cannot drift apart.

``outer`` (and ``ger``, which forwards to it) and ``addr`` take vectors too, and said
``outer: requires 1-D inputs`` from the engine, a ``LucidError``; the reference names
the operand: ``outer: Expected 1-D argument self, but got 2-D``,
``addr: Expected 1-D argument vec1, but got 0-D``.
"""

from collections.abc import Callable
from itertools import product
from types import ModuleType
from typing import Any

import pytest

import lucid
from lucid._C import engine as _C_engine

_SPELLINGS: dict[str, Callable[[Any, Any], Any]] = {
    "Tensor.dot": lambda a, b: a.dot(b),
    "linalg.dot": lambda a, b: lucid.linalg.dot(a, b),
    "vdot": lambda a, b: lucid.vdot(a, b),
}

_REFERENCE_SPELLINGS: dict[str, Callable[[Any, Any, ModuleType], Any]] = {
    "Tensor.dot": lambda a, b, ref: a.dot(b),
    "linalg.dot": lambda a, b, ref: ref.dot(a, b),
    "vdot": lambda a, b, ref: ref.vdot(a, b),
}

_DTYPES = [lucid.int8, lucid.float32, lucid.float16]


def _ones(shape: tuple[int, ...], dtype: lucid.dtype, device: str) -> lucid.Tensor:
    return lucid.ones(shape, dtype=dtype, device=device)


@pytest.mark.parametrize("spelling", list(_SPELLINGS))
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).removeprefix("lucid."))
@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        pytest.param((1, 1), (1, 1), id="2d-2d-issue"),
        pytest.param((2, 2), (2, 2), id="2d-2d"),
        pytest.param((), (), id="0d-0d"),
        pytest.param((3,), (3, 1), id="1d-2d"),
        pytest.param((3, 1), (3,), id="2d-1d"),
        pytest.param((), (3,), id="0d-1d"),
        pytest.param((2, 2, 2), (2, 2, 2), id="3d-3d"),
    ],
)
def test_dot_refuses_every_rank_but_one(
    spelling: str,
    dtype: lucid.dtype,
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
    device: str,
) -> None:
    a, b = _ones(shape_a, dtype, device), _ones(shape_b, dtype, device)
    with pytest.raises(_C_engine.ShapeMismatch) as raised:
        _SPELLINGS[spelling](a, b)
    assert isinstance(raised.value, RuntimeError)
    assert str(raised.value) == (
        f"1D tensors expected, but got {len(shape_a)}D and {len(shape_b)}D tensors"
    )


@pytest.mark.parametrize("spelling", list(_SPELLINGS))
def test_dot_of_two_vectors_is_unchanged(spelling: str, device: str) -> None:
    a = lucid.tensor([1.0, 2.0, 3.0], device=device)
    b = lucid.tensor([4.0, 5.0, 6.0], device=device)
    assert _SPELLINGS[spelling](a, b).item() == 32.0


def test_dot_of_two_vectors_still_back_propagates() -> None:
    a = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = lucid.tensor([4.0, 5.0, 6.0])
    a.dot(b).backward()
    assert a.grad is not None
    assert a.grad.tolist() == [4.0, 5.0, 6.0]


def test_dot_still_refuses_two_dtypes_and_two_lengths() -> None:
    ints = lucid.ones(3, dtype=lucid.int8)
    with pytest.raises(_C_engine.DtypeMismatch):
        ints.dot(lucid.ones(3, dtype=lucid.int16))
    with pytest.raises(_C_engine.ShapeMismatch):
        ints.dot(lucid.ones(2, dtype=lucid.int8))


@pytest.mark.parametrize("spelling", list(_SPELLINGS))
def test_dot_refuses_a_python_number(spelling: str) -> None:
    with pytest.raises(TypeError):
        _SPELLINGS[spelling](lucid.ones(3), 2.0)


def test_vdot_of_complex_vectors_is_not_conjugated_away_by_the_rank_check() -> None:
    a = lucid.tensor([1 + 2j, 3 - 1j])
    b = lucid.tensor([2 - 1j, 1 + 1j])
    assert lucid.vdot(a, b).item() == complex(2, -1)
    with pytest.raises(_C_engine.ShapeMismatch):
        lucid.vdot(a.reshape(1, 2), b.reshape(1, 2))


@pytest.mark.parity
@pytest.mark.parametrize("spelling", list(_SPELLINGS))
def test_dot_accepts_and_refuses_the_ranks_the_reference_does(
    spelling: str, ref: ModuleType
) -> None:
    """Every rank pair 0..3: the same pairs answer, the rest refuse with the same message."""
    problems: list[str] = []
    for rank_a, rank_b in product(range(4), repeat=2):
        shape_a, shape_b = (3,) * rank_a, (3,) * rank_b
        label = f"{rank_a}D . {rank_b}D"
        operands = (_ones(shape_a, lucid.float32, "cpu"), _ones(shape_b, lucid.float32, "cpu"))
        try:
            want = _REFERENCE_SPELLINGS[spelling](
                ref.ones(shape_a, dtype=ref.float32), ref.ones(shape_b, dtype=ref.float32), ref
            )
        except RuntimeError as refusal:
            with pytest.raises(RuntimeError) as raised:
                _SPELLINGS[spelling](*operands)
            if str(raised.value) != str(refusal):
                problems.append(f"{label}: {str(raised.value)!r}, reference {str(refusal)!r}")
            continue
        got = _SPELLINGS[spelling](*operands)
        if got.tolist() != want.tolist():
            problems.append(f"{label}: {got.tolist()}, reference {want.tolist()}")
    assert not problems, "\n".join(problems)


_PRODUCTS: dict[str, Callable[[Any, Any], Any]] = {
    "Tensor.outer": lambda a, b: a.outer(b),
    "linalg.outer": lambda a, b: lucid.linalg.outer(a, b),
    "ger": lambda a, b: lucid.ger(a, b),
    "Tensor.ger": lambda a, b: a.ger(b),
    "addr": lambda a, b: lucid.addr(lucid.zeros((3, 3), dtype=a.dtype), a, b),
    "Tensor.addr": lambda a, b: lucid.zeros((3, 3), dtype=a.dtype).addr(a, b),
}

_REFERENCE_PRODUCTS: dict[str, Callable[[Any, Any, ModuleType], Any]] = {
    "Tensor.outer": lambda a, b, ref: a.outer(b),
    "linalg.outer": lambda a, b, ref: ref.outer(a, b),
    "ger": lambda a, b, ref: ref.ger(a, b),
    "Tensor.ger": lambda a, b, ref: a.ger(b),
    "addr": lambda a, b, ref: ref.addr(ref.zeros((3, 3), dtype=a.dtype), a, b),
    "Tensor.addr": lambda a, b, ref: ref.zeros((3, 3), dtype=a.dtype).addr(a, b),
}

_RANK_PAIRS = [(0, 1), (1, 0), (2, 1), (1, 2), (0, 0), (2, 2), (3, 1)]


@pytest.mark.parametrize("spelling", list(_PRODUCTS))
@pytest.mark.parametrize(("rank_a", "rank_b"), _RANK_PAIRS)
def test_outer_and_addr_refuse_a_vector_of_another_rank(
    spelling: str, rank_a: int, rank_b: int, device: str
) -> None:
    a = _ones((3,) * rank_a, lucid.float32, device)
    b = _ones((3,) * rank_b, lucid.float32, device)
    with pytest.raises(_C_engine.ShapeMismatch) as raised:
        _PRODUCTS[spelling](a, b)
    assert isinstance(raised.value, RuntimeError)
    assert "Expected 1-D argument" in str(raised.value)


@pytest.mark.parity
@pytest.mark.parametrize("spelling", list(_PRODUCTS))
@pytest.mark.parametrize(("rank_a", "rank_b"), _RANK_PAIRS)
def test_outer_and_addr_name_the_operand_as_the_reference_does(
    spelling: str, rank_a: int, rank_b: int, ref: ModuleType
) -> None:
    shape_a, shape_b = (3,) * rank_a, (3,) * rank_b
    with pytest.raises(RuntimeError) as refusal:
        _REFERENCE_PRODUCTS[spelling](
            ref.ones(shape_a, dtype=ref.float32), ref.ones(shape_b, dtype=ref.float32), ref
        )
    with pytest.raises(RuntimeError) as raised:
        _PRODUCTS[spelling](
            _ones(shape_a, lucid.float32, "cpu"), _ones(shape_b, lucid.float32, "cpu")
        )
    assert str(raised.value) == str(refusal.value)


@pytest.mark.parametrize("spelling", ["addr", "Tensor.addr"])
def test_addr_refuses_an_accumulator_that_does_not_expand_to_the_product(spelling: str) -> None:
    """A 3-D accumulator used to broadcast the product into a 3-D result."""
    vec = lucid.ones(3)
    accumulator = lucid.zeros((2, 3, 3))
    call = (
        (lambda: lucid.addr(accumulator, vec, vec))
        if spelling == "addr"
        else (lambda: accumulator.addr(vec, vec))
    )
    with pytest.raises(_C_engine.ShapeMismatch):
        call()


@pytest.mark.parametrize("accumulator_shape", [(), (3,), (1, 3), (3, 1), (3, 3)])
def test_addr_still_expands_an_accumulator_to_the_product(
    accumulator_shape: tuple[int, ...],
) -> None:
    out = lucid.addr(lucid.zeros(accumulator_shape), lucid.ones(3), lucid.ones(3))
    assert out.shape == (3, 3)
