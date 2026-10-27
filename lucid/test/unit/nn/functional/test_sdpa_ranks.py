"""Attention takes the same shapes on Metal as on the CPU.

Metal's fused attention accepts (B, H, L, D) only, and a (B, L, D) query —
what MobileNet-v4's hybrid attention passes — raised "expected to be rank
4" there while the CPU answered.  Both hybrid checkpoints failed on Metal
the first time anything ran them at a batch through their published
weights (the pretrained-accuracy sweep).  The functional folds any other
rank into four axes and back; this holds the two devices to the same
answer and the same gradient at every rank it folds.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

CASES = [
    ((2, 16, 8), None, False),
    ((2, 16, 8), (2, 16, 16), False),
    ((2, 16, 8), None, True),
    ((16, 8), (16, 16), False),
    ((2, 3, 4, 16, 8), (3, 1, 16, 16), False),
]


@pytest.mark.parametrize("shape,mask_shape,causal", CASES)
def test_metal_attends_at_every_rank_as_the_cpu_does(shape, mask_shape, causal) -> None:
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal(shape).astype(np.float32) for _ in range(3))
    mask = None if mask_shape is None else rng.standard_normal(mask_shape).astype(np.float32)
    outs, grads = {}, {}
    for device in ("cpu", "metal"):
        query = lucid.tensor(q, device=device, requires_grad=True)
        out = F.scaled_dot_product_attention(
            query,
            lucid.tensor(k, device=device),
            lucid.tensor(v, device=device),
            attn_mask=None if mask is None else lucid.tensor(mask, device=device),
            is_causal=causal,
        )
        out.sum().backward()
        assert tuple(out.shape) == shape
        outs[device], grads[device] = out.numpy(), query.grad.numpy()
    np.testing.assert_allclose(outs["metal"], outs["cpu"], atol=1e-5)
    np.testing.assert_allclose(grads["metal"], grads["cpu"], atol=1e-5)
