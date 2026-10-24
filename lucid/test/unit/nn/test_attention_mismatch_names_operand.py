"""An attention operand mismatch names the operand that differs (LCD-562).

The engine tested Q, K and V in one expression and always reported Q and
K, so a float64 V against float32 Q and K raised "expected float32, got
float32" — the one message that cannot help.  DETR's half-precision bug
surfaced through exactly that line.
"""

import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available


def _qkv(odd: str, **change: object) -> tuple[lucid.Tensor, ...]:
    lucid.manual_seed(0)
    q, k, v = (lucid.randn(1, 2, 4, 8) for _ in range(3))
    named = {"q": q, "k": k, "v": v}
    named[odd] = named[odd].to(**change)
    return named["q"], named["k"], named["v"]


@pytest.mark.parametrize(("odd", "pair"), [("k", "Q/K"), ("v", "Q/V")])
def test_dtype_mismatch_names_the_odd_operand(odd: str, pair: str) -> None:
    q, k, v = _qkv(odd, dtype=lucid.float64)
    with pytest.raises(TypeError) as err:
        F.scaled_dot_product_attention(q, k, v)
    message = str(err.value)
    assert pair in message, message
    assert "expected float32, got float64" in message, message


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize(("odd", "pair"), [("k", "Q/K"), ("v", "Q/V")])
def test_device_mismatch_names_the_odd_operand(odd: str, pair: str) -> None:
    q, k, v = _qkv(odd, device="metal")
    with pytest.raises(ValueError) as err:
        F.scaled_dot_product_attention(q, k, v)
    message = str(err.value)
    assert pair in message, message
    assert "expected cpu, got metal" in message, message
