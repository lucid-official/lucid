"""``ctc_loss`` handles empty input sequences (``input_lengths == 0``) and short
sequences on CPU and Metal without out-of-bounds lattice indexing (LCD-257).

``ctc_loss_forward`` allocated ``alpha`` of size ``T_b * L`` and wrote
``alpha[0]`` and read ``alpha[(T_b - 1) * L + (L - 1)]`` unconditionally,
writing before and past the zero-length allocation when ``T_b == 0`` and
crashing the process with ``SIGSEGV``.
"""

from types import ModuleType
from typing import Literal

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

Reduction = Literal["none", "mean", "sum"]

_LENGTHS = (0, 1, 2)
_REDUCTIONS: tuple[Reduction, ...] = ("none", "mean", "sum")


def _log_probs_np(time_steps: int, batch: int, classes: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((time_steps, batch, classes)).astype(np.float32)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _assert_matches(got: np.ndarray, want: np.ndarray) -> None:
    assert got.shape == want.shape
    nan_mask = np.isnan(want)
    assert np.array_equal(np.isnan(got), nan_mask)
    inf_mask = np.isposinf(want)
    assert np.array_equal(np.isposinf(got), inf_mask)
    finite = ~(nan_mask | inf_mask)
    if np.any(finite):
        np.testing.assert_allclose(got[finite], want[finite], rtol=1e-5, atol=1e-5)


@pytest.mark.parity
@pytest.mark.parametrize("t_b", _LENGTHS)
@pytest.mark.parametrize("s_b", _LENGTHS)
@pytest.mark.parametrize("zero_infinity", [False, True])
@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_ctc_loss_zero_and_short_lengths_match_reference(
    ref: ModuleType, device: str, t_b: int, s_b: int, zero_infinity: bool, reduction: Reduction
) -> None:
    lp_np = _log_probs_np(time_steps=3, batch=1, classes=4)
    tgt_np = np.array([[1, 2]], dtype=np.int64)[:, :s_b]

    lp = lucid.tensor(lp_np, dtype=lucid.float32, device=device, requires_grad=True)
    targets = lucid.tensor(tgt_np, dtype=lucid.int64, device=device)
    il = lucid.tensor([t_b], dtype=lucid.int64, device=device)
    tl = lucid.tensor([s_b], dtype=lucid.int64, device=device)
    out = F.ctc_loss(lp, targets, il, tl, blank=0, reduction=reduction, zero_infinity=zero_infinity)
    (out.sum() if reduction == "none" else out).backward()
    assert lp.grad is not None

    ref_lp = ref.tensor(lp_np, dtype=ref.float32, requires_grad=True)
    ref_tgt = ref.tensor(tgt_np, dtype=ref.int64)
    ref_il = ref.tensor([t_b], dtype=ref.int64)
    ref_tl = ref.tensor([s_b], dtype=ref.int64)
    ref_out = ref.nn.functional.ctc_loss(
        ref_lp, ref_tgt, ref_il, ref_tl, blank=0, reduction=reduction, zero_infinity=zero_infinity
    )
    (ref_out.sum() if reduction == "none" else ref_out).backward()

    _assert_matches(out.numpy(), ref_out.detach().cpu().numpy())
    _assert_matches(lp.grad.numpy(), ref_lp.grad.detach().cpu().numpy())


@pytest.mark.parity
@pytest.mark.parametrize("zero_infinity", [False, True])
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("concatenated", [False, True])
def test_ctc_loss_mixed_zero_and_nonzero_batch_advances_target_offset(
    ref: ModuleType, device: str, zero_infinity: bool, reduction: Reduction, concatenated: bool
) -> None:
    lp_np = _log_probs_np(time_steps=4, batch=3, classes=5)
    padded_np = np.array([[1, 2], [3, 4], [2, 1]], dtype=np.int64)
    il_list = [0, 0, 4]
    tl_list = [0, 2, 2]
    tgt_np = np.array([3, 4, 2, 1], dtype=np.int64) if concatenated else padded_np

    lp = lucid.tensor(lp_np, dtype=lucid.float32, device=device, requires_grad=True)
    targets = lucid.tensor(tgt_np, dtype=lucid.int64, device=device)
    out = F.ctc_loss(
        lp, targets, il_list, tl_list, blank=0, reduction=reduction, zero_infinity=zero_infinity
    )
    (out.sum() if reduction == "none" else out).backward()
    assert lp.grad is not None

    ref_lp = ref.tensor(lp_np, dtype=ref.float32, requires_grad=True)
    ref_tgt = ref.tensor(tgt_np, dtype=ref.int64)
    ref_out = ref.nn.functional.ctc_loss(
        ref_lp, ref_tgt, il_list, tl_list, blank=0, reduction=reduction, zero_infinity=zero_infinity
    )
    (ref_out.sum() if reduction == "none" else ref_out).backward()

    _assert_matches(out.numpy(), ref_out.detach().cpu().numpy())
    _assert_matches(lp.grad.numpy(), ref_lp.grad.detach().cpu().numpy())
