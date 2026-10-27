"""A compiled call whose attention would materialise a huge score matrix runs eagerly.

The compiled graph lowers attention to matmul, softmax and matmul and so
writes every score; eager's fused kernel never does.  On an M4 Max a Wan
DiT block over 18,720 keys took 120.8 ms compiled against 87.5 ms eager
(CHA-14).  Past half a GiB of scores the call is routed to eager, once
announced, and remembered — the limit is lowered here so the test needs no
such matrix.
"""

import warnings

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile._core import attention_cost
from lucid.test._fixtures.devices import metal_available

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="compile needs metal"),
    pytest.mark.usefixtures("inference"),
]


class _Attend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(16, 16)

    def forward(self, x: lucid.Tensor, cache: lucid.Tensor) -> lucid.Tensor:
        q = self.q(x).reshape(1, -1, 2, 8).permute(0, 2, 1, 3)
        kv = cache.reshape(1, -1, 2, 8).permute(0, 2, 1, 3)
        return F.scaled_dot_product_attention(q, kv, kv).sum(dim=1)


def _inputs() -> tuple[lucid.Tensor, lucid.Tensor]:
    rng = np.random.default_rng(0)
    x = lucid.tensor(rng.standard_normal((1, 12, 16)).astype(np.float32), device="metal")
    cache = lucid.tensor(rng.standard_normal((1, 40, 16)).astype(np.float32), device="metal")
    return x, cache


def test_long_attention_runs_eagerly_and_says_so(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(attention_cost, "SCORE_LIMIT_BYTES", 64)
    monkeypatch.delenv("LUCID_COMPILE_LONG_ATTENTION", raising=False)
    model = _Attend().to("metal").eval()
    compiled = lucid.compile(model)
    x, cache = _inputs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = compiled(x, cache)
    assert any("score matrix" in str(w.message) for w in caught)
    np.testing.assert_allclose(got.numpy(), model(x, cache).numpy(), rtol=1e-5, atol=1e-5)
    assert len(compiled._eager_only) == 1  # remembered: the next call skips the trace
    compiled(x, cache)
    assert not compiled._cache


def test_the_limit_and_the_override(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    x, cache = _inputs()
    model = _Attend().to("metal").eval()
    # Below the limit: compiled as before.
    compiled = lucid.compile(model)
    compiled(x, cache)
    assert compiled._cache and not compiled._eager_only
    # Above it, but told to compile anyway.
    monkeypatch.setattr(attention_cost, "SCORE_LIMIT_BYTES", 64)
    monkeypatch.setenv("LUCID_COMPILE_LONG_ATTENTION", "compile")
    forced = lucid.compile(model)
    forced(x, cache)
    assert forced._cache and not forced._eager_only
