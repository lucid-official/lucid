"""A key's index tensors are refused at the key, under the caller's name (LCD-611).

``index_put(x, (int16_index,), v)`` raised ``LucidError: index_select:
indices must be int32 or int64`` — a ``RuntimeError`` naming an op the
caller never called — and so did ``x[i]`` and ``x[i] = v``; the reference
raises ``IndexError``.  ``_indexing._normalize_key`` reads every key, so the
width is checked there, by the one set the composite index ops refuse by.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._ops.composite import indexing as _composite
from lucid._tensor import _indexing

_NARROW = ["int8", "int16"]

_PATHS: dict[str, Callable[[lucid.Tensor, lucid.Tensor], object]] = {
    "__getitem__": lambda x, i: x[i],
    "__setitem__": lambda x, i: x.__setitem__(i, 1.0),
    "index_put": lambda x, i: lucid.index_put(x, (i,), lucid.ones(3, device=x.device)),
    "index_put_": lambda x, i: x.index_put_((i,), lucid.ones(3, device=x.device)),
}


@pytest.mark.parametrize("dtype", _NARROW)
@pytest.mark.parametrize("op", sorted(_PATHS))
def test_a_narrow_index_is_an_index_error_naming_the_caller(
    device: str, dtype: str, op: str
) -> None:
    x = lucid.zeros(4, 3, device=device)
    index = lucid.tensor([0, 1], dtype=getattr(lucid, dtype), device=device)
    with pytest.raises(IndexError, match=rf"^{op}: .*int32, int64 or bool.*{dtype}"):
        _PATHS[op](x, index)
    assert x.tolist() == [[0.0] * 3] * 4


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("op", sorted(_PATHS))
def test_the_engine_widths_pass(device: str, dtype: str, op: str) -> None:
    x = lucid.zeros(4, 3, device=device)
    _PATHS[op](x, lucid.tensor([0, 1], dtype=getattr(lucid, dtype), device=device))


def test_a_zero_d_index_of_any_width_is_its_integer(device: str) -> None:
    x = lucid.arange(12.0, device=device).reshape(4, 3)
    assert x[lucid.tensor(2, dtype=lucid.int16)].tolist() == [6.0, 7.0, 8.0]
    x[lucid.tensor(0, dtype=lucid.int8)] = 0.5
    assert x[0].tolist() == [0.5] * 3


def test_one_set_of_index_widths() -> None:
    assert _composite._INDEX_DTYPES is _indexing._INDEX_DTYPES


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _NARROW)
def test_the_reference_refuses_with_an_index_error(ref: ModuleType, dtype: str) -> None:
    x = ref.zeros(4, 3)
    index = ref.tensor([0, 1], dtype=getattr(ref, dtype))
    with pytest.raises(IndexError):
        ref.index_put(x, (index,), ref.ones(3))
    with pytest.raises(IndexError):
        x[index]
