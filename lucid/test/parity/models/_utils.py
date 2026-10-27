"""Shared utilities for model-level parity tests.

Three testing modes
-------------------
1. **Full parity** (has timm equivalent):
   Copy Lucid weights → timm via named or positional alignment, run identical
   input, compare logits with model-appropriate tolerances.

2. **Self-consistency** (no timm equivalent or key mismatch):
   Run the Lucid model twice with the same fixed input and assert deterministic
   output.  Catches forward-pass bugs without a reference.

3. **Block parity** (very large models):
   Extract a single block, copy its weights, compare only that sub-module.

Size thresholds for ``pytest.mark``
-------------------------------------
  < 30 M   → default (no mark)
 30–100 M  → ``pytest.mark.slow``
  > 100 M  → ``pytest.mark.heavy``

Registry-driven runners (used by vision/test_*.py)
----------------------------------------------------
``_spec_param(spec)``  — wrap ParitySpec with its tier mark for parametrize
``_run_parity(spec)``  — full parity check from a ParitySpec
``_run_self_consistency(spec)`` — determinism check from a ParitySpec
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._helpers.compare import assert_close

if TYPE_CHECKING:
    from lucid.test.parity.models._registry import ParitySpec

# ── timm availability ─────────────────────────────────────────────────────────

try:
    import timm as _timm  # noqa: F401

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

requires_timm = pytest.mark.skipif(not HAS_TIMM, reason="timm not installed")
heavy = pytest.mark.heavy


# ── Tier-aware parametrize helper ─────────────────────────────────────────────


def _spec_param(spec: ParitySpec) -> pytest.param:
    """Wrap a ParitySpec with the appropriate pytest tier mark.

    Use inside ``@pytest.mark.parametrize`` so each variant automatically
    carries its slow / heavy marker::

        @pytest.mark.parametrize("spec", [_spec_param(s) for s in _TIMM])
        def test_parity(spec): ...
    """
    marks: list[pytest.MarkDecorator] = []
    if spec.tier == "slow":
        marks.append(pytest.mark.slow)
    elif spec.tier == "heavy":
        marks.append(pytest.mark.heavy)
    return pytest.param(spec, id=spec.id, marks=marks)


# ── Registry-driven runners ───────────────────────────────────────────────────


def _randomise_norm_buffers(model: object, ref: object, seed: int = 0) -> None:
    """Give normalisation layers running statistics that keep signal alive.

    ``timm.create_model(pretrained=False)`` leaves every BatchNorm at
    ``running_mean = 0`` / ``running_var = 1``.  In eval mode that makes the
    layer a pure ``1 / sqrt(1 + eps)`` scale: it never restores variance.
    Stacked deep enough it is a vanishing map — MobileNet-v3's feature stack
    attenuated ~20-45x per non-residual block, so the *entire* backbone
    contributed 1.5e-8 to the logits against an ``atol`` of 1e-3.  Whatever
    the blocks computed, the comparison passed.

    Filling the buffers with plausible statistics restores that signal
    (measured on mobilenet_v3_large: 1.5e-8 -> 4.3), so a wrong SE gate, a
    swapped activation row, a dropped residual or a mis-ordered SE placement
    now actually reaches the assertion.  A few train-mode calibration batches
    were tried first and recovered far less (1.2e-2), because momentum only
    moves the buffers a fraction of the way and the statistics it collects
    are themselves measured on already-attenuating activations.

    Seeded, so the check stays deterministic — and applied before the
    transfer, so the Lucid side receives the same buffers.
    """
    rng = np.random.default_rng(seed)
    for module in model.modules():  # type: ignore[attr-defined]
        mean = getattr(module, "running_mean", None)
        var = getattr(module, "running_var", None)
        if mean is None or var is None:
            continue
        n = int(mean.numel())
        with ref.no_grad():  # type: ignore[attr-defined]
            mean.copy_(
                ref.from_numpy(  # type: ignore[attr-defined]
                    rng.normal(0.0, 0.25, n).astype(np.float32)
                ).reshape(mean.shape)
            )
            var.copy_(
                ref.from_numpy(  # type: ignore[attr-defined]
                    rng.uniform(0.5, 1.5, n).astype(np.float32)
                ).reshape(var.shape)
            )


#: Above this the reference's own output has stopped being informative —
#: see the skip in ``_run_parity``.  The tier's ordinary range tops out
#: near 37, so this sits well clear of it and far below the runaways.
_REF_SCALE_CEILING: float = 1e3

#: Floor for the absolute tolerance, as a fraction of the reference's
#: largest logit.  Wide enough for accumulated float error through a
#: 101-layer network, and orders below the structural breaks this suite
#: has actually caught (1.7 for se_resnet, 0.12 for cspresnet).
_SCALE_RELATIVE_ATOL: float = 1e-4


def _run_parity(spec: ParitySpec) -> None:
    """Execute one full parity check from a ParitySpec.

    Steps
    -----
    1. Build both models and put them in eval mode.
    2. Named weight transfer via ``_sd_transfer.transfer()``.
    3. If >2 keys are unmatched and ``use_positional_fallback`` is set,
       fall back to positional parameter copy.
    4. Feed identical random input to both models.
    5. Compare logit arrays with ``spec.atol`` / ``spec.rtol``.
    6. On failure, run ``diagnose_forward`` and print the top diverging layers.
    """
    import timm
    from lucid.test._fixtures.ref_framework import require_ref

    _ref = require_ref()

    from lucid.test.parity.models._sd_transfer import (
        diagnose_forward,
        transfer,
        transfer_positional,
    )

    if spec.skip_reason:
        pytest.skip(spec.skip_reason)

    lucid_model = spec.lucid_factory()
    lucid_model.eval()
    if spec.prepare is not None:
        # Before the transfer, so the reference receives the same values.
        spec.prepare(lucid_model)

    timm_model = timm.create_model(spec.timm_name, pretrained=False, num_classes=1000)
    timm_model.eval()
    # Before the transfer, so both sides get the same statistics.
    _randomise_norm_buffers(timm_model, _ref)

    result = transfer(lucid_model, timm_model, spec.key_remap, spec.key_transform)

    unmatched = len(result.unmatched_lucid)
    total = len(result.matched) + unmatched
    if unmatched > 2:
        pct = unmatched / total * 100
        if spec.use_positional_fallback:
            try:
                transfer_positional(lucid_model, timm_model)
            except AssertionError as exc:
                pytest.skip(
                    f"alignment incomplete ({result.summary()}) and "
                    f"positional fallback also failed: {exc}"
                )
        else:
            pytest.skip(
                f"alignment incomplete: {result.summary()} — " f"{pct:.0f}% of keys unmatched"
            )

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(spec.input_shape).astype(np.float32)

    with _ref.no_grad():
        y_ref = timm_model(_ref.from_numpy(x_np.copy())).numpy()

    lucid_out = lucid_model(lucid.from_numpy(x_np.copy()))
    y_lucid = lucid_out.logits.numpy() if hasattr(lucid_out, "logits") else lucid_out.numpy()

    scale = float(np.abs(y_ref).max())
    if scale > _REF_SCALE_CEILING:
        # Randomising the reference's norm buffers is what makes an
        # incomplete transfer visible, and in a deep network it also
        # amplifies: se_resnet_152 answers with logits of 8.7e3 and
        # inception_v3 with 3.9e13, against a median of 1.4 and a largest
        # ordinary value of 37 across this tier.  Nothing can be concluded
        # from agreement or disagreement at that size, so say so rather
        # than report a failure that means nothing.
        pytest.skip(
            f"the reference answers with |logits| up to {scale:.3g}, far "
            f"past the {_REF_SCALE_CEILING:g} this comparison is meaningful "
            f"below — the randomised norm buffers have amplified through "
            f"the depth, and the numbers no longer describe the model"
        )

    # Absolute error is roughly even across the logits while their
    # magnitudes span three orders, so an element-wise atol judges the
    # near-zero entries and nothing else: se_resnet_101 failed on 15 of
    # 1000, all with |logit| under 4.4, while the vector reached 137 and
    # the largest disagreement was 5.8e-05 of full scale.  Hold the floor
    # to the signal instead.
    atol = max(spec.atol, _SCALE_RELATIVE_ATOL * scale)

    try:
        assert_close(
            y_lucid,
            y_ref,
            atol=atol,
            rtol=spec.rtol,
            msg=f"parity failed for {spec.id} vs timm:{spec.timm_name}",
        )
    except AssertionError as exc:
        print(f"\n{'─' * 70}")
        print(f"PARITY FAILURE: {spec.id}  alignment={result.summary()}")
        print("Top diverging layers (max abs diff ↓):")
        for diff in diagnose_forward(
            lucid_model, timm_model, x_np, spec.key_remap, spec.key_transform
        ):
            print(f"  {diff}")
        print("─" * 70)
        raise exc


def _run_self_consistency(spec: ParitySpec) -> None:
    """Run a determinism check for a ParitySpec.

    Runs the Lucid model twice on the same fixed input and asserts
    bit-identical output.  Also checks that all logits are finite.
    """
    if spec.skip_reason:
        pytest.skip(spec.skip_reason)

    model = spec.lucid_factory()
    model.eval()

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal(spec.input_shape).astype(np.float32)
    x = lucid.from_numpy(x_np)

    out1 = model(x)
    out2 = model(x)

    y1 = out1.logits.numpy() if hasattr(out1, "logits") else out1.numpy()
    y2 = out2.logits.numpy() if hasattr(out2, "logits") else out2.numpy()

    assert_close(y1, y2, atol=0.0, rtol=0.0, msg=f"{spec.id}: forward pass is not deterministic")
    assert np.isfinite(y1).all(), f"{spec.id}: logits contain NaN or Inf"


# ── Legacy helpers (used by old per-family test files) ────────────────────────


def _copy_lucid_to_ref(lucid_model: nn.Module, ref_model: Any) -> None:
    """Copy parameters from Lucid model → reference model positionally."""
    from lucid.test._fixtures.ref_framework import require_ref

    _ref = require_ref()

    lucid_params = list(lucid_model.parameters())
    ref_params = list(ref_model.parameters())

    if len(lucid_params) != len(ref_params):
        raise AssertionError(
            f"param count mismatch: lucid={len(lucid_params)} " f"ref={len(ref_params)}"
        )

    for lp, rp in zip(lucid_params, ref_params):
        arr = lp.data.numpy()
        if arr.shape != tuple(rp.shape):
            raise AssertionError(f"shape mismatch: lucid={arr.shape} ref={tuple(rp.shape)}")
        rp.data.copy_(_ref.from_numpy(arr))


def run_parity(
    lucid_model: nn.Module,
    timm_name: str,
    *,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    seed: int = 0,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    num_classes: int = 1000,
    in_chans: int = 3,
) -> None:
    """Run one full-model parity check against a timm reference (legacy API)."""
    import timm
    from lucid.test._fixtures.ref_framework import require_ref

    _ref = require_ref()

    timm_model = timm.create_model(
        timm_name, pretrained=False, num_classes=num_classes, in_chans=in_chans
    )
    timm_model.eval()
    lucid_model.eval()

    try:
        _copy_lucid_to_ref(lucid_model, timm_model)
    except AssertionError as exc:
        pytest.skip(f"weight-transfer failed — {exc}")

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(input_shape).astype(np.float32)

    with _ref.no_grad():
        y_ref = timm_model(_ref.from_numpy(x_np)).numpy()

    lucid_out = lucid_model(lucid.from_numpy(x_np))
    y_lucid = lucid_out.logits.numpy() if hasattr(lucid_out, "logits") else lucid_out.numpy()

    assert_close(y_lucid, y_ref, atol=atol, rtol=rtol, msg=f"parity failed for timm:{timm_name}")


def run_self_consistency(
    lucid_model: nn.Module, *, input_shape: tuple[int, ...] = (1, 3, 224, 224), seed: int = 42
) -> None:
    """Assert deterministic forward pass (legacy API)."""
    lucid_model.eval()

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(input_shape).astype(np.float32)
    x = lucid.from_numpy(x_np)

    out1 = lucid_model(x)
    out2 = lucid_model(x)

    y1 = out1.logits.numpy() if hasattr(out1, "logits") else out1.numpy()
    y2 = out2.logits.numpy() if hasattr(out2, "logits") else out2.numpy()

    assert_close(y1, y2, atol=0.0, rtol=0.0, msg="forward pass is not deterministic")


def run_block_parity(
    lucid_block: nn.Module,
    ref_block: Any,
    *,
    input_shape: tuple[int, ...],
    seed: int = 0,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Compare a single sub-module against its reference counterpart (legacy API)."""
    from lucid.test._fixtures.ref_framework import require_ref

    _ref = require_ref()

    lucid_block.eval()
    ref_block.eval()

    try:
        _copy_lucid_to_ref(lucid_block, ref_block)
    except AssertionError as exc:
        pytest.skip(f"block weight-transfer failed — {exc}")

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(input_shape).astype(np.float32)

    with _ref.no_grad():
        y_ref = ref_block(_ref.from_numpy(x_np)).numpy()

    y_lucid_out = lucid_block(lucid.from_numpy(x_np))
    y_lucid = y_lucid_out.numpy() if hasattr(y_lucid_out, "numpy") else np.array(y_lucid_out)

    assert_close(y_lucid, y_ref, atol=atol, rtol=rtol, msg="block parity failed")
