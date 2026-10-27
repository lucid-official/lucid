"""AMP and quantization on MNIST — verified against what they promise.

The sibling module pins Lucid's training against a reference step for step.
That approach does not transfer here, and it is worth being explicit about
why: both mixed precision and post-training quantization are *lossy by
design*.  Two correct implementations legitimately disagree — they may cast a
different set of ops, pick a different loss scale, or choose different
observer statistics — so a per-step numeric comparison against the reference
would measure implementation choices rather than correctness.

What can be checked is the contract each feature actually offers:

  * **AMP** — training in half precision reaches the same quality as float32,
    without overflowing to NaN.  That is the whole promise; speed is a
    separate concern this test does not measure.
  * **Quantization** — an int8 model stays close to the float model it came
    from, and the accuracy it loses is small.

Both are also checked for *not being no-ops*, which matters more than it
sounds.  An ``autocast`` that silently failed to cast anything, or a
``convert`` that returned the float model untouched, would sail through every
quality assertion above — the numbers would be perfect because nothing
happened.  Those two checks are what make the rest meaningful.
"""

import math
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.amp as amp
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
import lucid.quantization as quant
from lucid.test.parity import _mnist_harness as H

pytestmark = [pytest.mark.parity, pytest.mark.slow]

# Shorter than the training-parity module: these compare Lucid against itself
# under two configurations, so each case is two full runs.  Three epochs is
# enough for the fp32/AMP quality gap to be meaningful.
AMP_EPOCHS = 3


def _train(
    model: Any,
    data: tuple[np.ndarray, ...],
    schedule: list[list[np.ndarray]],
    device: str,
    use_amp: bool,
) -> tuple[list[float], float, list[Any]]:
    """Train with or without autocast, reporting (losses, accuracy, dtypes)."""
    x_tr, y_tr, x_te, y_te = data
    opt = optim.Adam(model.parameters(), lr=H.ADAM_LR)
    scaler = amp.GradScaler(enabled=use_amp)

    losses: list[float] = []
    seen_dtypes: list[Any] = []
    for epoch_batches in schedule:
        for idx in epoch_batches:
            xb = lucid.tensor(x_tr[idx], device=device)
            yb = lucid.tensor(y_tr[idx], dtype=lucid.int64, device=device)

            opt.zero_grad()
            if use_amp:
                with amp.autocast(device_type=device):
                    out = H.logits(model(xb))
                    loss = F.cross_entropy(out, yb)
                seen_dtypes.append(out.dtype)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = H.logits(model(xb))
                loss = F.cross_entropy(out, yb)
                seen_dtypes.append(out.dtype)
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))

    model.eval()
    correct = 0
    with lucid.no_grad():
        for i in range(0, len(y_te), 256):
            xb = lucid.tensor(x_te[i : i + 256], device=device)
            pred = H.logits(model(xb)).numpy().argmax(axis=1)
            correct += int((pred == y_te[i : i + 256]).sum())
    model.train()
    return losses, correct / len(y_te), seen_dtypes


# ── AMP ─────────────────────────────────────────────────────────────────────


def test_autocast_actually_casts(device_gpu_only: str) -> None:
    """Anti-vacuity check for everything below.

    An ``autocast`` that quietly cast nothing would pass every quality
    assertion in this module — the numbers would match float32 perfectly
    because they *would be* float32.  So assert the dtype changes first, and
    changes back outside the block.
    """
    device = device_gpu_only
    model = nn.Linear(16, 4).to(device)
    x = lucid.tensor(np.zeros((2, 16), dtype=np.float32), device=device)

    assert model(x).dtype == lucid.float32

    with amp.autocast(device_type=device):
        assert model(x).dtype == lucid.float16, (
            "autocast did not cast — every other AMP assertion here would " "then be vacuously true"
        )

    assert model(x).dtype == lucid.float32, "autocast leaked past its block"


def test_grad_scaler_skips_the_step_on_a_non_finite_gradient(
    device_gpu_only: str,
) -> None:
    """The scaler's real job: drop the update when the gradient overflowed.

    Half precision overflows at 65,504, and a scaled loss reaches that
    regularly.  If the scaler stepped anyway the parameters would take an inf
    and the run would be lost, so the skip is what makes AMP usable at all —
    and it is invisible in a loss curve, since a skipped step just looks like
    a flat one.
    """
    device = device_gpu_only
    model = nn.Linear(4, 2).to(device)
    opt = optim.Adam(model.parameters(), lr=0.1)
    scaler = amp.GradScaler(init_scale=2.0**16)

    before = model.weight.numpy().copy()

    # The gradient is overwritten directly rather than provoked with a huge
    # input.  A large input does not reliably reach infinity — one attempt
    # produced a finite 6.55e34 gradient, and the parameters still did not
    # move, because Adam's second moment squared it to 1e60 and overflowed to
    # inf on its own.  That looks identical to a skipped step from the
    # outside, so the test would have "passed" while measuring nothing.
    x = lucid.tensor(np.ones((1, 4), dtype=np.float32), device=device)
    scaler.scale(model(x).sum()).backward()
    for p in model.parameters():
        p.grad = lucid.tensor(np.full(p.shape, np.inf, dtype=np.float32), device=device)

    scaler.step(opt)
    scaler.update()

    after = model.weight.numpy()
    assert np.isfinite(after).all(), (
        "a non-finite gradient reached the parameters — the scaler stepped "
        "when it should have skipped"
    )
    assert np.array_equal(
        before, after
    ), "the scaler applied an update derived from a non-finite gradient"
    assert scaler.get_scale() < 2.0**16, (
        "the scaler did not back off after seeing a non-finite gradient, so "
        "the next step would overflow the same way"
    )


def test_grad_scaler_keeps_the_scale_when_gradients_are_finite(
    device_gpu_only: str,
) -> None:
    """The other half of the contract, and the reason the test above needs a
    real infinity: backing off on a merely *large* gradient would shrink the
    scale until half precision underflows instead."""
    device = device_gpu_only
    model = nn.Linear(4, 2).to(device)
    opt = optim.Adam(model.parameters(), lr=0.1)
    scaler = amp.GradScaler(init_scale=2.0**16)

    x = lucid.tensor(np.ones((1, 4), dtype=np.float32), device=device)
    scaler.scale(model(x).sum()).backward()
    assert all(np.isfinite(p.grad.numpy()).all() for p in model.parameters())

    before = model.weight.numpy().copy()
    scaler.step(opt)
    scaler.update()

    assert scaler.get_scale() == 2.0**16, "the scale moved on a healthy step"
    assert not np.array_equal(
        before, model.weight.numpy()
    ), "a finite gradient was skipped — the scaler is dropping good updates"


def test_amp_training_reaches_float32_quality(
    device_gpu_only: str,
    mnist: tuple[np.ndarray, ...],
) -> None:
    """Half-precision training must land in the same place as float32.

    This is the promise users actually rely on: turn AMP on, get the same
    model.  A subtly wrong cast list — a reduction or a normalisation left in
    half where it needed float32 — degrades accuracy by a few points rather
    than failing outright, which is why the comparison is against a float32
    run of the identical setup rather than against a fixed threshold.
    """
    device = device_gpu_only
    rng = np.random.default_rng(H.SEED)
    schedule = [H.batches(H.N_TRAIN, rng) for _ in range(AMP_EPOCHS)]

    lucid.manual_seed(H.SEED)
    fp32_model = H.build_lucid_bn_cnn().to(device)
    fp32_losses, fp32_acc, fp32_dtypes = _train(fp32_model, mnist, schedule, device, use_amp=False)

    lucid.manual_seed(H.SEED)
    amp_model = H.build_lucid_bn_cnn().to(device)
    amp_losses, amp_acc, amp_dtypes = _train(amp_model, mnist, schedule, device, use_amp=True)

    assert set(fp32_dtypes) == {lucid.float32}
    assert lucid.float16 in set(
        amp_dtypes
    ), "the AMP run never produced a half-precision activation"

    assert all(math.isfinite(v) for v in amp_losses), (
        "AMP training produced a non-finite loss — the scaler failed to keep "
        "the scaled gradients in range"
    )
    for name, p in amp_model.named_parameters():
        assert np.isfinite(p.numpy()).all(), f"{name} is non-finite after AMP"

    assert amp_losses[-1] < 0.35 * amp_losses[0], (
        f"AMP training did not converge: {amp_losses[0]:.4f} → " f"{amp_losses[-1]:.4f}"
    )
    assert amp_acc > 0.90, f"AMP accuracy {amp_acc:.4f} is too low"
    assert abs(amp_acc - fp32_acc) < 0.02, (
        f"AMP cost more accuracy than rounding explains: {amp_acc:.4f} vs "
        f"float32 {fp32_acc:.4f} — a cast list that leaves a reduction in "
        f"half precision looks exactly like this"
    )


# ── quantization ────────────────────────────────────────────────────────────


def _fit_float_model(data: tuple[np.ndarray, ...], device: str) -> tuple[nn.Module, float]:
    """A briefly-trained float model to quantize.  Two epochs is enough for
    accuracy to be well above chance, which is what makes a quantization
    regression visible."""
    rng = np.random.default_rng(H.SEED)
    schedule = [H.batches(H.N_TRAIN, rng) for _ in range(2)]
    lucid.manual_seed(H.SEED)
    model = H.build_lucid_bn_cnn().to(device)
    _, acc, _ = _train(model, data, schedule, device, use_amp=False)
    return model, acc


def _quantize(model: nn.Module, data: tuple[np.ndarray, ...], device: str) -> Any:
    prepared = quant.prepare(model, quant.get_default_qconfig_mapping())
    prepared.eval()
    x_tr = data[0]
    with lucid.no_grad():  # calibration: observers record activation ranges
        for i in range(0, 512, 64):
            prepared(lucid.tensor(x_tr[i : i + 64], device=device))
    return quant.convert(prepared)


def _accuracy(model: Any, data: tuple[np.ndarray, ...], device: str) -> float:
    _, _, x_te, y_te = data
    model.eval()
    correct = 0
    with lucid.no_grad():
        for i in range(0, len(y_te), 256):
            xb = lucid.tensor(x_te[i : i + 256], device=device)
            pred = H.logits(model(xb)).numpy().argmax(axis=1)
            correct += int((pred == y_te[i : i + 256]).sum())
    return correct / len(y_te)


def test_convert_actually_quantizes(device: str, mnist: tuple[np.ndarray, ...]) -> None:
    """Anti-vacuity check: ``convert`` must really replace the float weights.

    A ``convert`` that returned the model untouched would score identically to
    float and pass every accuracy assertion below.  The observable proof is
    int8 weight buffers with their scale and zero-point, which a float model
    does not have.
    """
    lucid.manual_seed(H.SEED)
    model = H.build_lucid_bn_cnn().to(device)
    converted = _quantize(model, mnist, device)

    buffers = dict(converted.named_buffers())
    int8_weights = [n for n, b in buffers.items() if b.dtype == lucid.int8]
    assert int8_weights, (
        f"no int8 buffers after convert — the model was not quantized. "
        f"buffers present: {sorted(buffers)[:8]}"
    )
    for name in int8_weights:
        stem = name.rsplit("_int8", 1)[0]
        assert f"{stem}_scale" in buffers, f"{name} has no scale"
        assert f"{stem}_zero_point" in buffers, f"{name} has no zero point"


def test_quantized_model_stays_close_to_the_float_model(
    device: str, mnist: tuple[np.ndarray, ...]
) -> None:
    """int8 inference must track the float model it was derived from.

    Quantization is lossy, so the bar is not equality — it is that the loss is
    small and bounded.  Both halves matter: a large output error means the
    scales are wrong, while a large accuracy drop means the errors landed
    where they change the prediction.
    """
    float_model, float_acc = _fit_float_model(mnist, device)
    assert float_acc > 0.90, (
        f"the float model only reached {float_acc:.4f}; a quantization "
        f"regression would not be visible against that"
    )

    converted = _quantize(float_model, mnist, device)

    x_te = mnist[2][:256]
    float_model.eval()
    with lucid.no_grad():
        xb = lucid.tensor(x_te, device=device)
        float_out = H.logits(float_model(xb)).numpy()
        quant_out = H.logits(converted(xb)).numpy()

    assert np.isfinite(quant_out).all(), "quantized inference produced NaN/Inf"

    spread = float(np.abs(float_out).max())
    worst = float(np.abs(float_out - quant_out).max())
    assert worst < 0.15 * spread, (
        f"quantized logits are {worst:.4f} off against a logit range of "
        f"{spread:.4f} — that is a scale/zero-point problem, not rounding"
    )

    quant_acc = _accuracy(converted, mnist, device)
    assert quant_acc > float_acc - 0.03, (
        f"quantization cost {float_acc - quant_acc:.4f} accuracy "
        f"({float_acc:.4f} → {quant_acc:.4f}); int8 post-training "
        f"quantization of a model this size should cost far less"
    )
