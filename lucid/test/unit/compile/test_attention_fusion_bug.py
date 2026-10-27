"""Regression: MPSGraph's fused-attention pass miscompiles some shapes.

MPSGraph pattern-matches ``softmax(QKᵀ) @ V`` onto an internal fused-attention
kernel that silently produces WRONG output for a window of shapes — sequence
length N in [17, 24] with batch ≥ 3 (on macOS 26).  Individual ops and the
2-op sub-chains are all correct; only the full chain triggers it, so the bug is
purely in the fusion.  Eager is always correct.

The compile layer dodges it by emitting any matmul that reads a softmax output
(directly or through an inference-mode ``dropout``) in transposed form —
``a @ b == (bᵀ @ aᵀ)ᵀ`` — which breaks MPSGraph's pattern match.

These tests pin the exact bad shapes (the model-zoo coverage used batch 2 and
missed them).  ``compiled`` must match a NumPy reference, not just eager.
"""

import numpy as np

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.nn.functional as F

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

# Shapes inside (and bracketing) the bad window, at batch >= 3.
_BAD_SHAPES = [(8, 12, 17, 64), (8, 12, 20, 64), (8, 12, 24, 64), (4, 12, 18, 64)]
_OK_SHAPES = [(8, 12, 16, 64), (8, 12, 32, 64), (2, 12, 20, 64)]


def _np_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = np.einsum("bhid,bhjd->bhij", q, k)
    s = s - s.max(-1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(-1, keepdims=True)
    return np.einsum("bhij,bhjd->bhid", w, v)


class _SdpaOp(nn.Module):
    def forward(self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        return F.scaled_dot_product_attention(q, k, v, scale=1.0)


class _ManualAttn(nn.Module):  # manual q@kᵀ; softmax; @v
    def forward(self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        return F.softmax(q @ k.permute(0, 1, 3, 2), dim=-1) @ v


def _compiled_attention(shape: tuple[int, ...], *, use_sdpa_op: bool) -> float:
    """Compile attention at ``shape`` and return max abs error vs a NumPy ref."""
    b, h, n, d = shape
    rng = np.random.default_rng(n * 100 + b)
    arrs = [rng.standard_normal((b, h, n, d)).astype(np.float32) for _ in range(3)]
    ins = [lucid.tensor(a.copy(), device=COMPILE_DEVICE) for a in arrs]

    model: nn.Module = _SdpaOp() if use_sdpa_op else _ManualAttn()
    model = model.to(COMPILE_DEVICE).eval()
    cm = lucid.compile(model)
    out = cm(*ins).numpy()
    ref = _np_attention(*arrs)
    return float(np.abs(out - ref).max())


def test_manual_attention_bad_shapes_correct() -> None:
    for shape in _BAD_SHAPES + _OK_SHAPES:
        err = _compiled_attention(shape, use_sdpa_op=False)
        assert err < 1e-3, f"manual attention {shape} miscompiled: {err:.3e}"


def test_sdpa_op_bad_shapes_correct() -> None:
    for shape in _BAD_SHAPES + _OK_SHAPES:
        err = _compiled_attention(shape, use_sdpa_op=True)
        assert err < 1e-3, f"SDPA op {shape} miscompiled: {err:.3e}"


def test_vit_base_batch8_correct() -> None:
    # Real-model repro: ViT-base (12 heads, head_dim 64) on a 64×64 image gives
    # 17 tokens; at batch 8 the manual attention hit the bug.  Compile must
    # match eager.
    model = M.vit_base_16(image_size=64, num_classes=10).to(COMPILE_DEVICE).eval()
    x = lucid.tensor(
        np.random.default_rng(0).standard_normal((8, 3, 64, 64)).astype(np.float32),
        device=COMPILE_DEVICE,
    )

    def unwrap(out: object) -> lucid.Tensor:
        for attr in ("logits", "last_hidden_state"):
            v = getattr(out, attr, None)
            if v is not None and hasattr(v, "eval"):
                return v
        return out  # type: ignore[return-value]

    eager = unwrap(model(x))
    cm = lucid.compile(model)
    compiled = unwrap(cm(x))
    compiled.eval()
    assert float((eager - compiled).abs().max().item()) < 1e-4


# Run in a child: the defect is an abort inside MPSGraph, which takes the
# interpreter with it rather than raising.
_UNEQUAL_WIDTHS = """
import numpy as np
import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

# The fast path, as on hardware the capability probe clears — the only path
# where MPSGraph sees the plain ``attn @ v`` and can fuse it.
_C_engine.compile.set_attention_workaround_state(0)
rng = np.random.default_rng(0)
q, k = (rng.standard_normal((3, 8, 23, 32)).astype(np.float32) for _ in range(2))
v = rng.standard_normal((3, 8, 23, 128)).astype(np.float32)


def attend(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, scale=1.0)


dev = [lucid.tensor(a, device="{device}") for a in (q, k, v)]
out = lucid.compile(attend)(*dev).numpy()
s = np.einsum("bhid,bhjd->bhij", q, k)
w = np.exp(s - s.max(-1, keepdims=True))
ref = np.einsum("bhij,bhjd->bhid", w / w.sum(-1, keepdims=True), v)
print(float(np.abs(out - ref).max()))
"""


def test_sdpa_value_width_unlike_key_width_compiles() -> None:
    """A value head wider than the key (EfficientFormer: 128 against 32).

    On macOS 26 MPSGraph fuses that attention too and then aborts the process
    in its optimisation passes ("MLIR pass manager failed"), so the emitter
    takes the transposed form for it whatever the probe decided.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", _UNEQUAL_WIDTHS.replace("{device}", COMPILE_DEVICE)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"compiled attention with unequal widths exited {proc.returncode}: "
        f"{proc.stderr.strip().splitlines()[-1:] or proc.stdout}"
    )
    err = float(proc.stdout.strip().splitlines()[-1])
    assert err < 1e-3, f"attention with unequal widths miscompiled: {err:.3e}"
