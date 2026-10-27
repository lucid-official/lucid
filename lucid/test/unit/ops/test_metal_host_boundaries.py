"""Native Metal forwards must not observe a Python host scalar or array."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


@pytest.mark.parametrize("name", ["linear", "conv", "layernorm", "softmax", "attention", "fft"])
def test_native_forward_has_no_python_host_observation(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    if name == "conv":
        module = nn.Conv2d(3, 4, 3, padding=1).to("metal")
        x = lucid.randn(1, 3, 8, 8, device="metal")
        call = lambda: module(x)
    elif name in ("linear", "layernorm"):
        module = (nn.Linear(8, 4) if name == "linear" else nn.LayerNorm(8)).to("metal")
        x = lucid.randn(2, 8, device="metal")
        call = lambda: module(x)
    elif name == "attention":
        q, k, v = (lucid.randn(1, 2, 4, 8, device="metal") for _ in range(3))
        call = lambda: F.scaled_dot_product_attention(q, k, v)
    else:
        x = lucid.randn(2, 8, device="metal")
        call = lambda: lucid.fft.fft(x) if name == "fft" else F.softmax(x, dim=-1)
    with lucid.no_grad():
        expected = call().numpy().copy()

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("unexpected Python host observation in native Metal forward")

        with monkeypatch.context() as guard:
            for method in ("numpy", "item", "cpu", "tolist", "__float__", "__bool__"):
                guard.setattr(lucid.Tensor, method, forbidden)
            output = call()
        assert output.is_metal
        np.testing.assert_allclose(output.numpy(), expected, rtol=1e-6, atol=1e-6)
