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
``_run_parity(spec)``  — full parity check from a ParitySpec; returns a
                         ``ParityResult`` (reference scale, tolerance, worst error)
``_run_self_consistency(spec)`` — determinism check from a ParitySpec
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._helpers.compare import assert_close

if TYPE_CHECKING:
    from lucid.test.parity.models._registry import ParitySpec
    from lucid.test.parity.models._sd_transfer import AlignResult

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
#
# A parity test is only a test if the reference's answer is far from zero
# relative to the tolerance: an all-zeros Lucid model passes any comparison whose
# expected value is within ``atol`` of zero.  Two things made that happen here --
# a freshly initialised network carries no signal through its depth (BatchNorm at
# its default statistics rescales nothing and the stack collapses, or amplifies
# without bound), and a zero-initialised head answers exactly 0 whatever the body
# computes -- so the harness establishes the signal before it transfers anything.
# Both repairs act on the *Lucid* model, the source of the transfer: ``transfer``
# copies Lucid's parameters *and buffers* over the reference's, so anything done to
# the reference's buffers first is overwritten, which is how an earlier attempt to
# randomise them came to do nothing at all.

#: Standard deviation of the draw that replaces a parameter that is exactly zero.
#: ViT's own initialisation uses 0.02, so a woken head answers at the scale a
#: freshly initialised classifier would.
_WAKE_STD: float = 0.02
_WAKE_SEED: int = 0

#: Images drawn beside the test input when the norm statistics are measured.  The
#: test input is one of them, so no activation it produces lies outside its own
#: calibration batch (a held-out input is amplified by 1/sqrt(var) in every layer
#: whose variance came out small -- ResNeSt-101 answered at 2.8e4); the others keep a
#: layer that sees one position per channel from having a variance of zero.
_CALIBRATION_EXTRA: int = 3
_CALIBRATION_SEED: int = 1

_BATCH_NORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

#: Above this the reference's own output has stopped being informative -- float32
#: cannot resolve a tolerance against it.  Calibrated models answer within 0.1-5, so
#: this is a bug in the harness, and is reported as one rather than skipped.
_REF_SCALE_CEILING: float = 1e3

#: Floor for the absolute tolerance, as a fraction of the reference's largest logit.
#: Wide enough for accumulated float error through a 150-layer network (measured as
#: much as 7.5e-4 of the largest logit, Inception-v4; the reference's own float32 answer
#: sits 9e-4 from its float64 one at ResNeSt-101).  At this floor the signal-to-tolerance
#: ratio is 1 / (2e-3 + rtol) = 333 however large the logits are.
_SCALE_RELATIVE_ATOL: float = 2e-3

#: The least ratio of the reference's largest logit to the loosest allowance of the
#: comparison (``atol + rtol * max|logit|``).  Passing within one tolerance must mean
#: agreeing to within 1% of the answer: at 100 a model that is wrong by one tolerance
#: is wrong by 1% of the output, and below it ``allclose`` stops saying much -- at 10
#: a model answering 10% off would pass.  The loosest allowance is at most
#: ``scale * (rtol + atol / scale)``, so a spec with ``rtol=1e-3`` reaches the ratio of
#: 100 whenever ``atol`` is under about 9e-3 of the logits' scale.
_MIN_SIGNAL_TO_TOLERANCE: float = 100.0


@dataclass(frozen=True)
class ParityResult:
    """What one passing comparison measured, so a sweep can report it."""

    reference_scale: float
    atol: float
    rtol: float
    max_abs_error: float

    @property
    def tolerance(self) -> float:
        """The loosest allowance anywhere in the comparison."""
        return self.atol + self.rtol * self.reference_scale

    @property
    def signal_to_tolerance(self) -> float:
        return self.reference_scale / self.tolerance


def _wake_zero_parameters(model: nn.Module) -> int:
    """Draw every parameter that is exactly zero from ``N(0, _WAKE_STD)``.

    A tensor of zeros carries no signal through any multiplicative path: ViT's
    zero-initialised head answers 0 whatever the body computes, and a zero bias or
    norm shift is a term the comparison cannot see go missing.  Seeded, and applied to
    the Lucid model before the transfer, so both sides hold the same values.
    """
    lucid.manual_seed(_WAKE_SEED)
    woken = 0
    for parameter in model.parameters():
        if parameter.numel() and parameter.data.abs().max().item() == 0.0:
            nn.init.normal_(parameter, 0.0, _WAKE_STD)
            woken += 1
    return woken


def _calibrate_norm_statistics(model: nn.Module, x_np: np.ndarray) -> int:
    """Set every BatchNorm's running statistics to those of the activations it sees.

    A freshly built network sits at ``running_mean = 0`` / ``running_var = 1``, which in
    eval mode is a bare ``1 / sqrt(1 + eps)`` scale: it never restores the variance the
    convolutions around it remove.  Deep enough that is a vanishing map -- the
    MobileNets' logits came out at 1e-9 -- or, in a residual stack, an amplifier --
    ResNet-152 at 1.6e8.  A trained network's statistics are its activations'; one
    pass in train mode with a cumulative average (``momentum=None``) leaves exactly
    that, so every layer then hands the next an O(1) signal.  Only the norm layers
    switch to train mode, so dropout and stochastic depth stay off.
    """
    norms = [m for m in model.modules() if isinstance(m, _BATCH_NORMS) and m.track_running_stats]
    momenta = [norm.momentum for norm in norms]
    for norm in norms:
        norm.reset_running_stats()
        norm.momentum = None
        norm.train()
    rng = np.random.default_rng(_CALIBRATION_SEED)
    extra = rng.standard_normal((_CALIBRATION_EXTRA, *x_np.shape[1:])).astype(np.float32)
    try:
        with lucid.no_grad():
            model(lucid.from_numpy(np.concatenate([x_np, extra])))
    finally:
        for norm, momentum in zip(norms, momenta, strict=True):
            norm.momentum = momentum
            norm.eval()
    return len(norms)


def _build_aligned_pair(spec: ParitySpec) -> tuple[nn.Module, Any, np.ndarray, AlignResult]:
    """Build the Lucid model and its reference, live, with the weights carried across.

    Steps
    -----
    1. Build the Lucid model, put it in eval mode, run the spec's ``prepare``.
    2. Wake its zero parameters and calibrate its norm statistics on the test input.
    3. Build the reference (``spec.timm_kwargs``) and copy Lucid's parameters *and
       buffers* into it by name (``_sd_transfer.transfer``).
    4. If more than 2 keys are unmatched, fall back to positional copy when the spec
       allows it; otherwise skip, saying how much did not line up.
    """
    import timm

    from lucid.test.parity.models._sd_transfer import transfer, transfer_positional

    lucid_model = spec.lucid_factory()
    lucid_model.eval()
    if spec.prepare is not None:
        spec.prepare(lucid_model)
    x_np = np.random.default_rng(0).standard_normal(spec.input_shape).astype(np.float32)
    _wake_zero_parameters(lucid_model)
    _calibrate_norm_statistics(lucid_model, x_np)

    timm_model = timm.create_model(
        spec.timm_name, pretrained=False, num_classes=1000, **spec.timm_kwargs
    )
    timm_model.eval()
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
    return lucid_model, timm_model, x_np, result


def _require_signal(spec_id: str, scale: float, tolerance: float) -> None:
    """Fail -- never skip -- a comparison whose reference answer cannot discriminate.

    ``scale`` is the reference's largest |logit|, ``tolerance`` the loosest allowance of
    the comparison.  A skip reads green in a run and is indistinguishable from a
    comparison that passed, which is how six models went unchecked.
    """
    if scale > _REF_SCALE_CEILING:
        pytest.fail(
            f"{spec_id}: the reference answers with |logits| up to {scale:.3g}, past the "
            f"{_REF_SCALE_CEILING:g} a float32 comparison can resolve -- the harness failed "
            f"to bring the forward pass into range",
            pytrace=False,
        )
    if scale < _MIN_SIGNAL_TO_TOLERANCE * tolerance:
        pytest.fail(
            f"{spec_id}: the reference answers with |logits| up to {scale:.3g} against a "
            f"tolerance of {tolerance:.3g} -- {scale / tolerance:.3g}x, under the "
            f"{_MIN_SIGNAL_TO_TOLERANCE:g}x below which agreement within the tolerance says "
            f"little (an all-zeros model passes at 1x).  Tighten the spec's atol, or give "
            f"the harness a way to wake the forward pass.",
            pytrace=False,
        )


def _run_parity(
    spec: ParitySpec, mutate: Callable[[nn.Module], None] | None = None
) -> ParityResult:
    """Execute one full parity check from a ParitySpec.

    Steps
    -----
    1. Build the live, aligned pair (``_build_aligned_pair``).
    2. ``mutate``, when given, edits the Lucid model after the transfer -- the
       sensitivity test uses it to prove a wrong model fails this comparison.
    3. Feed identical random input to both models.
    4. Fail unless the reference's answer is ``_MIN_SIGNAL_TO_TOLERANCE`` times the
       tolerance (``_require_signal``).
    5. Compare logit arrays with ``spec.atol`` / ``spec.rtol``.
    6. On failure, run ``diagnose_forward`` and print the top diverging layers.
    """
    from lucid.test._fixtures.ref_framework import require_ref
    from lucid.test.parity.models._sd_transfer import diagnose_forward

    _ref = require_ref()

    if spec.skip_reason:
        pytest.skip(spec.skip_reason)

    lucid_model, timm_model, x_np, result = _build_aligned_pair(spec)
    if mutate is not None:
        mutate(lucid_model)

    with _ref.no_grad():
        y_ref = timm_model(_ref.from_numpy(x_np.copy())).numpy()

    lucid_out = lucid_model(lucid.from_numpy(x_np.copy()))
    y_lucid = lucid_out.logits.numpy() if hasattr(lucid_out, "logits") else lucid_out.numpy()

    scale = float(np.abs(y_ref).max())
    # Absolute error is roughly even across the logits while their magnitudes span
    # orders, so an element-wise atol judges the near-zero entries and nothing else.
    # Hold the floor to the signal instead.
    atol = max(spec.atol, _SCALE_RELATIVE_ATOL * scale)
    _require_signal(spec.id, scale, atol + spec.rtol * scale)

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
    return ParityResult(scale, atol, spec.rtol, float(np.abs(y_lucid - y_ref).max()))


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
