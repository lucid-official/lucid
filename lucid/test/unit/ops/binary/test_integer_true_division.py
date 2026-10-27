"""True division of integers gives floats, as the reference's does.

The engine's ``div`` keeps an integer dtype when both operands have one, so
``tensor([7, 8]) / 2`` came back ``[3, 4]``: the wrong numbers, with nothing
to say so.
"""

import pytest

import lucid


def _ints(*values: int, dtype: lucid.dtype = lucid.int64) -> lucid.Tensor:
    return lucid.tensor(list(values), dtype=dtype)


def test_an_integer_tensor_over_a_number_is_floating() -> None:
    out = _ints(7, 8) / 2
    assert out.dtype == lucid.float32
    assert out.tolist() == [3.5, 4.0]


def test_integer_tensors_divide_truly() -> None:
    assert (_ints(7, 8, dtype=lucid.int32) / _ints(2, 4, dtype=lucid.int32)).tolist() == [
        3.5,
        2.0,
    ]


def test_a_number_over_an_integer_tensor() -> None:
    assert (7 / _ints(2)).tolist() == [3.5]


def test_a_float_number_is_not_truncated() -> None:
    assert (_ints(7) / 2.5).tolist() == [2.799999952316284]


def test_bools_divide_as_floats() -> None:
    assert (lucid.tensor([True, False]) / 2).tolist() == [0.5, 0.0]


def test_the_function_and_method_forms_agree() -> None:
    assert lucid.div(_ints(7, 8), 2).tolist() == [3.5, 4.0]
    assert _ints(7, 8).div(2).tolist() == [3.5, 4.0]
    assert lucid.divide(_ints(7, 8), 2).tolist() == [3.5, 4.0]
    assert lucid.true_divide(7, _ints(2)).tolist() == [3.5]


def test_a_half_precision_operand_keeps_its_precision() -> None:
    half = lucid.tensor([2.0], dtype=lucid.float16)
    assert (_ints(7, dtype=lucid.int32) / half).dtype == lucid.float16
    assert lucid.divide(_ints(7, dtype=lucid.int32), half).dtype == lucid.float16


def test_in_place_true_division_of_an_integer_tensor_is_refused() -> None:
    a = _ints(7, 8)
    with pytest.raises(RuntimeError, match="can't be cast"):
        a /= 2
    assert a.tolist() == [7, 8]


def test_div_in_place_on_an_integer_tensor_is_refused() -> None:
    # ``div_`` floor-divided in silence, leaving [3, 4], where ``/=`` above
    # already refused.
    a = _ints(7, 8)
    with pytest.raises(RuntimeError, match="can't be cast"):
        a.div_(2)
    with pytest.raises(RuntimeError, match="can't be cast"):
        a.div_(lucid.tensor([2.0, 4.0]))
    with pytest.raises(RuntimeError, match="can't be cast"):
        lucid.tensor([True, False]).div_(2)
    assert a.dtype == lucid.int64
    assert a.tolist() == [7, 8]


def test_div_in_place_on_a_float_tensor_is_unchanged() -> None:
    a = lucid.tensor([1.0, 3.0])
    a.div_(2)
    assert a.tolist() == [0.5, 1.5]
    a.div_(_ints(1, 3))
    assert a.tolist() == [0.5, 0.5]


def test_float_division_is_unchanged() -> None:
    a = lucid.tensor([1.0, 3.0])
    assert (a / 2).tolist() == [0.5, 1.5]
    a /= 2
    assert a.tolist() == [0.5, 1.5]
