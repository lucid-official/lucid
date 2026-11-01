"""Attention on an empty key length or head dimension (LCD-752).

The masked path reduced over the key axis to find all-``-inf`` rows, which
the engine refuses on an empty axis, so a causal or masked call with no
keys raised ``IndexError`` while the unmasked one returned zeros.  The
default scale ``1/sqrt(E)`` divided by zero for ``E == 0``.  The reference
returns zeros for no keys, and a uniform average of the values for no head
dimension (every score is an empty dot product).
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

#: (id, q shape, k shape, v shape, mask shape or None, is_causal)
_CASES = [
    ("causal_no_keys", (2, 3, 5, 4), (2, 3, 0, 4), (2, 3, 0, 4), None, True),
    ("mask_no_keys", (2, 3, 5, 4), (2, 3, 0, 4), (2, 3, 0, 4), (5, 0), False),
    ("causal_mask_no_keys", (2, 3, 5, 4), (2, 3, 0, 4), (2, 3, 0, 4), (5, 0), True),
    ("plain_no_keys", (2, 3, 5, 4), (2, 3, 0, 4), (2, 3, 0, 4), None, False),
    ("no_queries_causal", (2, 3, 0, 4), (2, 3, 6, 4), (2, 3, 6, 4), None, True),
    ("empty_all_causal", (2, 3, 0, 4), (2, 3, 0, 4), (2, 3, 0, 4), None, True),
    ("no_head_dim", (2, 3, 5, 0), (2, 3, 6, 0), (2, 3, 6, 4), None, False),
    ("no_head_dim_causal", (2, 3, 5, 0), (2, 3, 6, 0), (2, 3, 6, 4), None, True),
    ("no_head_dim_mask", (2, 3, 5, 0), (2, 3, 6, 0), (2, 3, 6, 4), (5, 6), False),
]
_IDS = [c[0] for c in _CASES]


def _arrays(*shapes: tuple[int, ...]) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


@pytest.mark.parity
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_sdpa_empty_extents_match_the_reference(
    ref: Any, device: str, case: tuple[Any, ...]
) -> None:
    _, qs, ks, vs, ms, causal = case
    arrays = _arrays(qs, ks, vs)
    mask = None if ms is None else _arrays(ms)[0]

    ours = [lucid.tensor(a, device=device, requires_grad=True) for a in arrays]
    out = F.scaled_dot_product_attention(
        *ours,
        attn_mask=None if mask is None else lucid.tensor(mask, device=device),
        is_causal=causal,
    )
    theirs = [ref.from_numpy(a).requires_grad_() for a in arrays]
    want = ref.nn.functional.scaled_dot_product_attention(
        *theirs, attn_mask=None if mask is None else ref.from_numpy(mask), is_causal=causal
    )
    weights = _arrays(tuple(want.shape))[0]
    (out * lucid.tensor(weights, device=device)).sum().backward()
    (want * ref.from_numpy(weights)).sum().backward()

    assert out.shape == tuple(want.shape)
    np.testing.assert_allclose(out.cpu().numpy(), want.detach().numpy(), rtol=1e-5, atol=1e-6)
    for name, mine, theirs_t in zip("qkv", ours, theirs, strict=True):
        assert mine.grad is not None, name
        assert mine.grad.shape == tuple(theirs_t.grad.shape), name
        np.testing.assert_allclose(
            mine.grad.cpu().numpy(), theirs_t.grad.numpy(), rtol=1e-5, atol=1e-6, err_msg=name
        )
