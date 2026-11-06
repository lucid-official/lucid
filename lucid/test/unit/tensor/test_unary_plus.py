"""``+tensor`` is :func:`lucid.positive`: the tensor itself, refused on booleans."""

import pytest

import lucid


@pytest.mark.parametrize("dtype", [lucid.float32, lucid.int32, lucid.complex64])
def test_unary_plus_is_the_tensor_itself(dtype: lucid.dtype) -> None:
    x = lucid.tensor([1.0, -2.0], dtype=dtype)
    assert +x is x
    assert lucid.positive(x) is x


def test_unary_plus_keeps_the_graph() -> None:
    x = lucid.tensor([1.0, -2.0], requires_grad=True)
    (+x).sum().backward()
    assert x.grad.tolist() == [1.0, 1.0]


@pytest.mark.parametrize("call", [lambda t: +t, lucid.positive], ids=["operator", "function"])
def test_a_boolean_tensor_is_refused(call) -> None:
    with pytest.raises(RuntimeError, match="boolean"):
        call(lucid.tensor([True, False]))


def test_on_metal_too() -> None:
    x = lucid.tensor([1.0, -2.0], device="metal")
    assert +x is x
