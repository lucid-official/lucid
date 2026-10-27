"""Keyword-passed tensor inputs compile (they used to force eager).

The MPSGraph feed list is positional, but a forward called as
``model(x, mask=mask)`` is common (attention masks, optional conditioning,
``token_type_ids=`` …).  The compiler traces those keyword tensors like
positional ones and binds each executable feed back to either an ``args``
index or a ``kwargs`` name at call time (``_CacheEntry.input_source``).

Three properties pinned here:

  1. **A keyword tensor compiles** — one cache entry, no eager fallback,
     fp32-parity vs eager.
  2. **Re-keying on a non-tensor kwarg** — a different scalar kwarg value
     is captured by the signature, so it recompiles instead of silently
     reusing the wrong executable.
  3. **Dropping a kwarg between calls falls back gracefully** — the
     per-call feed resolver run-eagers rather than feeding ``None``.
"""

import pytest

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_cache_hit,
    assert_no_eager_fallback,
    metal_tensor,
)

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


class _MaskedMLP(nn.Module):
    """``forward(x, mask=None, scale=1.0)`` — a kwarg tensor + a kwarg scalar."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(
        self, x: lucid.Tensor, mask: lucid.Tensor | None = None, scale: float = 1.0
    ) -> lucid.Tensor:
        y = self.fc(x) * scale
        if mask is not None:
            y = y + mask
        return y


def test_keyword_tensor_compiles() -> None:
    """``cm(x, mask=mask)`` builds one executable, parity-exact vs eager."""
    model = _MaskedMLP().to(COMPILE_DEVICE).eval()
    x = metal_tensor(4, 8)
    mask = metal_tensor(4, 8)

    eager = model(x, mask=mask)

    cm = lucid.compile(model)
    out = cm(x, mask=mask)
    cm(x, mask=mask)  # second call must be a cache hit, not a recompile

    assert_cache_hit(cm, expected_entries=1)
    assert_no_eager_fallback(cm)
    diff = float((eager - out).abs().max().item())
    assert diff < 1e-5, f"keyword-tensor compile parity broken: {diff:.3e}"


def test_positional_and_keyword_mix() -> None:
    """A model called purely by keyword still compiles + matches eager."""
    model = _MaskedMLP().to(COMPILE_DEVICE).eval()
    x = metal_tensor(2, 8)
    mask = metal_tensor(2, 8)

    eager = model(x=x, mask=mask)
    cm = lucid.compile(model)
    out = cm(x=x, mask=mask)

    assert_cache_hit(cm, expected_entries=1)
    diff = float((eager - out).abs().max().item())
    assert diff < 1e-5, f"all-keyword compile parity broken: {diff:.3e}"


def test_scalar_kwarg_rekeys() -> None:
    """A different non-tensor kwarg value recompiles (distinct signature)."""
    model = _MaskedMLP().to(COMPILE_DEVICE).eval()
    x = metal_tensor(4, 8)

    cm = lucid.compile(model)
    cm(x, scale=1.0)
    cm(x, scale=2.0)  # different scalar → must NOT reuse the scale=1.0 graph

    assert_cache_hit(cm, expected_entries=2)
    # both must be parity-exact against their respective eager evals
    for s in (1.0, 2.0):
        eager = model(x, scale=s)
        out = cm(x, scale=s)
        diff = float((eager - out).abs().max().item())
        assert diff < 1e-5, f"scale={s} parity broken: {diff:.3e}"


def test_dropping_kwarg_falls_back_gracefully() -> None:
    """Dropping the keyword tensor on a later call run-eagers, never crashes.

    The first call compiles a graph whose feed list includes ``mask``.  A
    follow-up call with the *same shape signature* but no ``mask`` would
    leave that feed unbound — the per-call resolver detects the missing
    Tensor and routes the whole call through eager instead of feeding a
    bogus value.
    """
    model = _MaskedMLP().to(COMPILE_DEVICE).eval()
    x = metal_tensor(4, 8)
    mask = metal_tensor(4, 8)

    cm = lucid.compile(model)
    cm(x, mask=mask)  # compiles with a mask feed

    # Same x signature, but the cached entry for THIS signature still
    # expects a mask.  Distinct signatures keep them apart in practice
    # (signature_of incorporates kwargs), so this asserts the no-mask call
    # produces the correct eager result rather than crashing.
    out = cm(x)
    eager = model(x)
    diff = float((eager - out).abs().max().item())
    assert diff < 1e-5, f"no-mask fallback parity broken: {diff:.3e}"
