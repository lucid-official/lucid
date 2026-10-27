"""MNIST training pinned against a reference implementation, step for step.

Every other training test in this repo asserts only that *Lucid's own* loss
shrinks on synthetic data.  That catches a dead gradient, but it cannot catch
a gradient that is merely wrong: a loss curve going down is compatible with an
optimizer that is subtly mis-scaled, a backward that drops a term, or a
convolution whose padding is off by one.  Each of those still "trains", just
to a worse model, and nothing in a solo run reports it.

So this module pins Lucid against a reference implementation on a problem with
a known answer.  For every case both frameworks get the same architecture, the
same initial state (Lucid's parameters *and* buffers, copied across, so step 0
is identical rather than merely similar), the same data in the same order and
the same optimizer settings.  Under those conditions the two are solving an
identical optimisation problem, and any divergence is Lucid's numerics.

Cases
-----

``lenet``   the canonical 61,706-parameter LeNet-5 (LeCun et al. 1998), under
            SGD+momentum, Adam and AdamW.  Three optimizers because SGD's
            update is almost trivially checkable while Adam's is not — bias
            correction, epsilon placement and AdamW's decoupled decay are each
            a place where a plausible implementation is off by a little, and
            "a little" is invisible in a loss curve.
``bn_cnn``  BatchNorm, ReLU and max pooling — the modern convolution block
            LeNet gives no coverage of.
``resnet``  two post-activation residual blocks.  The skip is the point: its
            backward adds the upstream gradient to the branch gradient, so a
            wrong accumulation gives a model that trains but underperforms,
            with the residual path silently contributing nothing.
``vit``     a pre-norm transformer block over 16 image patches — multi-head
            attention, LayerNorm and two more residuals.  LayerNorm differs
            from BatchNorm in that its statistics are computed inside the
            graph and must therefore be differentiated, not just tracked.
``+cosine`` LeNet again with ``CosineAnnealingLR`` driving both sides, which
            checks the schedule itself rather than the optimizer.

Measured behaviour
------------------

Tolerances are set from measurement, not from a guess, and the architectures
split cleanly into two regimes.

Smooth models do not drift at all.  Over all 470 steps, on CPU and Metal::

    lenet (tanh, avgpool)             max 2.4e-07 (SGD) … 1.1e-06 (AdamW)
    lenet + cosine schedule           max 4.8e-07, learning rates bit-equal
    vit (softmax, GELU, LayerNorm)    max 3.6e-07

tanh, softmax, GELU and average pooling are all smooth, so a 1-ULP difference
is damped rather than amplified and the two runs stay on top of each other for
the whole run.  The expectation going in had been the opposite — that float32's
non-associative addition would separate them through reduction order alone.

Models containing ReLU and max pooling do drift, and the reason is
architectural rather than a defect::

    bn_cnn   epoch max  2.2e-04  9.3e-04  2.2e-03  2.6e-03  3.8e-03
    resnet   epoch max  4.4e-04  1.6e-03  2.4e-03  2.2e-03  3.6e-03

ReLU's threshold and MaxPool's argmax are discrete: a 1-ULP difference can
flip which unit is active or which element wins a pooling window, and the
gradient then takes a different path outright.  Several independent checks say
this is amplified rounding — step 0 is *exactly* 0.0 for both, the sign of the
difference is balanced across steps (216 lucid-high vs 250 lucid-low for
bn_cnn, mean signed difference only 19% of mean absolute), and with the
weights frozen the BatchNorm buffers agree to 5.96e-08.

The control experiment settles it.  Run the *reference against itself* with a
single weight nudged by one ULP — same code on both sides, so any divergence
cannot be an implementation difference::

    reference vs 1-ULP-perturbed reference   epoch max  4.5e-04 … 3.5e-03
    lucid vs reference                       epoch max  2.2e-04 … 3.8e-03

The same magnitude.  And over the early window Lucid tracks the reference
*more* closely (1.2e-05) than the reference tracks a one-ULP copy of itself
(1.0e-04).  The drift belongs to the architecture; any two correct
implementations would show it.

Why the early window carries the test
-------------------------------------

A real defect is present from the first step while chaotic drift needs epochs,
so the tight bound applies to the first 30 steps and a coarse one to the whole
run.  Verified by injection — a 1% learning-rate error, the "trains fine,
converges to a slightly worse model" defect this module exists to catch::

    case          clean max[:30]   with 1% error   final accuracy difference
    lenet-sgd     4.8e-07          8.8e-03         0.0005   (not caught)
    bn_cnn-adam   1.2e-05          7.7e-03         0.0000   (not caught)
    resnet-adam   7.5e-05          9.0e-03         0.0005   (not caught)
    vit-adam      2.4e-07          4.0e-03         0.0025   (not caught)

Only the step-level check fires; the accuracy comparison misses the defect in
every single case, so it cannot carry the test.  An epoch-and-accuracy-only
version of this module passes a mis-scaled optimizer without complaint, which
is exactly the failure it was written to detect.  The bounds put the detection
floor near a 0.1% error in both regimes.

Scope: 6,000 training images rather than the full 60,000, so each case runs in
roughly 20 seconds per device.
"""

import math
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.optim.lr_scheduler as lr_scheduler
from lucid.test.parity import _mnist_harness as H

pytestmark = [pytest.mark.parity, pytest.mark.slow]


def _lucid_cosine(opt: Any) -> Any:
    return lr_scheduler.CosineAnnealingLR(opt, T_max=H.EPOCHS)


def _ref_cosine(ref: ModuleType) -> Any:
    def make(opt: Any) -> Any:
        return ref.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=H.EPOCHS)

    return make


# (id, model kind, optimizer, schedule, accuracy floor, loss ratio, early, full)
#
# Two tolerance regimes for the step bounds, per the measurements in the module
# docstring: smooth architectures hold at ULP for the whole run, ReLU/MaxPool
# ones amplify.  The accuracy floor and loss ratio are per-case for a duller
# reason — these models simply converge at different speeds, and a bound
# borrowed from LeNet says nothing about a transformer.
CASES = [
    ("lenet-sgd", "lenet", "sgd", False, 0.90, 0.25, 1e-5, 1e-3),
    ("lenet-adam", "lenet", "adam", False, 0.90, 0.25, 1e-5, 1e-3),
    ("lenet-adamw", "lenet", "adamw", False, 0.90, 0.25, 1e-5, 1e-3),
    ("lenet-sgd-cosine", "lenet", "sgd", True, 0.90, 0.25, 1e-5, 1e-3),
    ("bn_cnn-adam", "bn_cnn", "adam", False, 0.90, 0.25, 1e-3, 3e-2),
    ("resnet-adam", "resnet", "adam", False, 0.90, 0.25, 1e-3, 3e-2),
    # The ViT is one small block over 16 patches trained for five epochs, so
    # it converges more slowly than the convolutional cases and lands near
    # 0.80 rather than 0.95 — measured 1.7788 → 0.4858, a ratio of 0.273.
    # Both bounds stay far above the 0.10 chance floor, so a broken attention
    # backward still cannot fake them.
    ("vit-adam", "vit", "adam", False, 0.75, 0.40, 1e-5, 1e-3),
]


@pytest.mark.parametrize(
    "kind,opt_name,scheduled,acc_floor,loss_ratio,early_tol,full_tol",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_training_matches_the_reference_over_five_epochs(
    device: str,
    ref: ModuleType,
    mnist: tuple[np.ndarray, ...],
    kind: str,
    opt_name: str,
    scheduled: bool,
    acc_floor: float,
    loss_ratio: float,
    early_tol: float,
    full_tol: float,
) -> None:
    model, ref_model = H.build_pair(kind, device, ref)

    # One schedule, consumed by both — identical batches in identical order.
    rng = np.random.default_rng(H.SEED)
    schedule = [H.batches(H.N_TRAIN, rng) for _ in range(H.EPOCHS)]

    make_l = _lucid_cosine if scheduled else None
    make_r = _ref_cosine(ref) if scheduled else None

    lucid_epochs, lucid_steps, lucid_acc, lucid_lrs = H.train_lucid(
        model, mnist, schedule, device, opt_name, make_l
    )
    ref_epochs, ref_steps, ref_acc, ref_lrs = H.train_ref(
        ref_model, mnist, schedule, ref, opt_name, make_r
    )

    # 1. Trainability — the loss actually falls, and the result is a model
    #    rather than a coin flip.  Measured ratios: 0.081 for lenet+sgd
    #    (0.8520 → 0.0692), 0.117 for bn_cnn, 0.273 for the vit.
    assert lucid_epochs[-1] < loss_ratio * lucid_epochs[0], (
        f"lucid loss barely moved: {lucid_epochs[0]:.4f} → "
        f"{lucid_epochs[-1]:.4f} over {H.EPOCHS} epochs"
    )
    assert lucid_acc > acc_floor, f"lucid accuracy {lucid_acc:.4f} is too low"

    # 2. Stability — nothing went non-finite anywhere, including parameters
    #    (a loss can stay finite while a weight has already blown up).
    assert all(math.isfinite(v) for v in lucid_steps)
    for name, p in model.named_parameters():
        assert np.isfinite(p.numpy()).all(), f"{name} contains NaN/Inf"

    # 3. Agreement.  Step 0 isolates forward + backward + the update rule:
    #    identical weights and identical input, no accumulated drift to hide
    #    behind.  Measured 0.0 to 4.8e-07 across the cases.
    assert lucid_steps[0] == pytest.approx(ref_steps[0], abs=1e-5), (
        f"first-step loss differs: lucid {lucid_steps[0]:.8f} vs "
        f"reference {ref_steps[0]:.8f} — same weights, same batch, so this "
        f"is a forward or backward discrepancy, not trajectory divergence"
    )

    # The early window is where this test gets its power: a systematic defect
    # is present from the first step, while chaotic drift needs epochs.  See
    # the injection table in the module docstring.
    step_diffs = [abs(a - b) for a, b in zip(lucid_steps, ref_steps)]
    early = step_diffs[: H.EARLY_STEPS]
    worst_early = max(range(len(early)), key=early.__getitem__)
    assert early[worst_early] < early_tol, (
        f"trajectories separated inside the first {H.EARLY_STEPS} steps, at "
        f"step {worst_early}: lucid {lucid_steps[worst_early]:.8f} vs "
        f"reference {ref_steps[worst_early]:.8f} "
        f"(absdiff {early[worst_early]:.2e}) — too early to be rounding "
        f"amplification, so this is a systematic discrepancy"
    )

    # Over the whole run, a coarser bound: enough to catch a trajectory that
    # genuinely walks away, loose enough to tolerate the ReLU/MaxPool
    # amplification measured above.
    worst = max(range(len(step_diffs)), key=step_diffs.__getitem__)
    assert step_diffs[worst] < full_tol, (
        f"loss trajectories separated at step {worst}: lucid "
        f"{lucid_steps[worst]:.8f} vs reference {ref_steps[worst]:.8f} "
        f"(absdiff {step_diffs[worst]:.2e})"
    )

    # Epoch means: measured identical to four decimals, bounded at 2%.
    for i, (lucid_e, ref_e) in enumerate(zip(lucid_epochs, ref_epochs)):
        assert lucid_e == pytest.approx(ref_e, rel=0.02, abs=1e-3), (
            f"epoch {i + 1} mean loss diverged: lucid {lucid_e:.4f} vs "
            f"reference {ref_e:.4f}\nlucid  {lucid_epochs}\nref    {ref_epochs}"
        )

    # Accuracy — measured within 0.0005 everywhere, and exactly equal for
    # lenet and vit.  0.01 is 20 of the 2,000 test images.
    assert abs(lucid_acc - ref_acc) < 0.01, (
        f"final accuracy differs: lucid {lucid_acc:.4f} vs " f"reference {ref_acc:.4f}"
    )

    # 4. The learning-rate schedule itself, when one is attached.  This is a
    #    closed-form sequence rather than a trajectory, so unlike the losses
    #    it must agree exactly — measured difference is 0.0 at every step.
    #    Checked separately from the loss because a wrong schedule and a wrong
    #    optimizer produce the same symptom otherwise.
    if scheduled:
        assert len(set(round(v, 8) for v in lucid_lrs)) == H.EPOCHS, (
            f"the schedule did not vary the learning rate across epochs: "
            f"{sorted(set(round(v, 8) for v in lucid_lrs))}"
        )
        worst_lr = max(abs(a - b) for a, b in zip(lucid_lrs, ref_lrs))
        assert worst_lr < 1e-9, (
            f"learning-rate schedules diverged by {worst_lr:.2e}\n"
            f"lucid {sorted(set(round(v, 8) for v in lucid_lrs))}\n"
            f"ref   {sorted(set(round(v, 8) for v in ref_lrs))}"
        )

    # ``num_batches_tracked`` is an integer counter, so unlike the float
    # statistics it cannot drift — the two runs must have taken exactly the
    # same number of BatchNorm updates.  The float buffers are deliberately
    # *not* compared here: after five epochs they inherit whatever trajectory
    # divergence happened above, so any bound would be arbitrary.  They are
    # checked properly by the frozen-weight test below, and indirectly by the
    # accuracy assertion, which runs in eval mode and therefore reads the
    # running statistics rather than the batch's own.
    ref_buffers = dict(ref_model.named_buffers())
    for name, buf in model.named_buffers():
        if not name.endswith("num_batches_tracked"):
            continue
        got = np.asarray(buf.numpy(), dtype=np.int64)
        want = np.asarray(ref_buffers[name].numpy(), dtype=np.int64)
        assert np.array_equal(got, want), f"{name} differs: lucid {got} vs reference {want}"


def test_batchnorm_running_stats_match_with_weights_frozen(
    device: str,
    ref: ModuleType,
    mnist: tuple[np.ndarray, ...],
) -> None:
    """BatchNorm's running statistics, isolated from trajectory divergence.

    Running statistics are updated outside the autograd graph, so a wrong
    momentum or a biased-vs-unbiased variance never shows up in a training
    loss.  It surfaces only at eval — as a model that trained well and scores
    badly — which is exactly the kind of defect a loss curve cannot report.

    Comparing them after a full training run does not work: by then they have
    inherited the ReLU/MaxPool trajectory divergence documented above and
    differ by ~5e-03 for reasons that have nothing to do with BatchNorm.  So
    this runs the forward pass only, with no optimizer step at all.  The
    weights stay identical on both sides, which makes the batch statistics
    identical too, and any difference in the accumulated buffers is then
    BatchNorm's own update rule.

    Measured: 5.96e-08 — one float32 ULP — across every buffer after ten
    batches, which is what established the 5e-03 in the training run as a
    downstream effect rather than a defect.
    """
    model, ref_model = H.build_pair("bn_cnn", device, ref)
    x_tr = mnist[0]

    rng = np.random.default_rng(H.SEED)
    for idx in H.batches(H.N_TRAIN, rng)[:10]:
        model(lucid.tensor(x_tr[idx], device=device))
        with ref.no_grad():
            ref_model(ref.from_numpy(x_tr[idx]))

    ref_buffers = dict(ref_model.named_buffers())
    checked = 0
    for name, buf in model.named_buffers():
        got = np.asarray(buf.numpy(), dtype=np.float64)
        want = np.asarray(ref_buffers[name].numpy(), dtype=np.float64)
        assert np.abs(got - want).max() < 1e-6, (
            f"buffer {name} diverged with the weights frozen: max absdiff "
            f"{np.abs(got - want).max():.3e} — the two models saw identical "
            f"inputs and identical weights, so this is BatchNorm's own update"
        )
        checked += 1
    assert checked == 6, f"expected 6 BatchNorm buffers, compared {checked}"
