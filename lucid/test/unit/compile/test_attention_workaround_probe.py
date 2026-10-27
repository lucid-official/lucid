"""The fused-attention workaround capability probe.

The compile emitters carry a workaround for an MPSGraph bug that miscompiles
``softmax(QKᵀ) @ V`` on some GPUs (M1 Pro / macOS 26), but the workaround (a
transposed value-matmul) costs ~+70% on the attention path, so it is gated on a
one-time runtime probe: the engine flag starts at -1 (apply, always-correct) and
the probe clears it to 0 only after a known-bad envelope verifies *this* GPU is
unaffected; on affected GPUs it sets 1.

These tests are hardware-agnostic: whatever the probe decides, compiled attention
must match a NumPy reference.  They assert the probe *runs* (flag leaves -1) and
that correctness holds for both verdicts.
"""

import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


class _Attn(nn.Module):
    def forward(self, q: lucid.Tensor, k: lucid.Tensor, v: lucid.Tensor) -> lucid.Tensor:
        return F.softmax(q @ k.permute(0, 1, 3, 2), dim=-1) @ v


def _np_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = np.einsum("bhid,bhjd->bhij", q, k)
    s = s - s.max(-1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(-1, keepdims=True)
    return np.einsum("bhij,bhjd->bhid", w, v)


def test_probe_runs_and_resolves_flag() -> None:
    # Compiling any attention graph must resolve the capability flag away from
    # the -1 "unprobed" sentinel to a concrete verdict (0 unaffected / 1 affected).
    rng = np.random.default_rng(0)
    arrs = [rng.standard_normal((8, 12, 17, 64)).astype(np.float32) for _ in range(3)]
    ins = [lucid.tensor(a.copy(), device=COMPILE_DEVICE) for a in arrs]
    lucid.compile(_Attn().to(COMPILE_DEVICE).eval())(*ins).eval()

    state = _C_engine.compile.attention_workaround_state()
    assert state in (0, 1), f"probe left flag unresolved: {state}"


def test_attention_correct_under_probe_verdict() -> None:
    # Whatever the probe decided, compiled attention across the (formerly) bad
    # window must match a NumPy reference.
    for b, n in ((8, 17), (8, 20), (8, 24), (4, 18), (3, 22)):
        rng = np.random.default_rng(b * 100 + n)
        arrs = [rng.standard_normal((b, 12, n, 64)).astype(np.float32) for _ in range(3)]
        ins = [lucid.tensor(a.copy(), device=COMPILE_DEVICE) for a in arrs]
        out = lucid.compile(_Attn().to(COMPILE_DEVICE).eval())(*ins).numpy()
        err = float(np.abs(out - _np_attention(*arrs)).max())
        assert err < 1e-3, f"attention [{b},12,{n},64] miscompiled: {err:.3e}"
