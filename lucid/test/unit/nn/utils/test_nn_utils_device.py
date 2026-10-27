"""``nn.utils`` helpers must work on the input's device.

Found 2026-07-26 by a coverage-directed probe.  ``lucid/nn/utils`` was the
lowest-covered subsystem in the tree (27% of statements executed), and probing
it turned up **3 device bugs in 15 helpers** — a ~10x higher hit rate than the
same probe applied to better-covered surface (1 in 31).

All three are the same family as ``pdist`` / transforms / crossvit / vander: a
helper tensor created without ``device=`` and then combined with GPU data.

* ``spectral_norm`` seeded its power-iteration vectors with ``lucid.randn(...)``
  on the default device, then contracted them with the weight every forward.
* ``prune.random_unstructured`` built its mask with ``lucid.rand(...)`` and
  multiplied the weight by it.
* ``pad_packed_sequence`` built the ``unsorted_indices`` reorder tensor without
  a device and gathered from the (GPU) output with it.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn

U = nn.utils
DEVICES = ["cpu", "metal"]


def _t(*shape, device="cpu", seed=0):
    return lucid.tensor(
        np.random.default_rng(seed).standard_normal(shape).astype(np.float32), device=device
    )


@pytest.mark.parametrize("device", DEVICES)
def test_spectral_norm_runs_on_device(device):
    lucid.manual_seed(0)
    layer = U.spectral_norm(nn.Linear(4, 3).to(device))
    out = layer(_t(2, 4, device=device))
    assert str(out.device) == f"device('{device}')"
    assert out.shape == (2, 3)
    assert not np.isnan(out.numpy()).any()


@pytest.mark.parametrize("device", DEVICES)
def test_spectral_norm_backward(device):
    lucid.manual_seed(0)
    layer = U.spectral_norm(nn.Linear(4, 3).to(device))
    out = layer(_t(2, 4, device=device))
    (out * lucid.ones_like(out)).sum().backward()
    grads = [p.grad for p in layer.parameters() if p.grad is not None]
    assert grads, "spectral_norm blocked the gradient path"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prune_fn", ["random_unstructured", "l1_unstructured"])
def test_prune_runs_on_device(device, prune_fn):
    lucid.manual_seed(0)
    layer = nn.Linear(4, 3).to(device)
    getattr(U.prune, prune_fn)(layer, "weight", 0.5)
    out = layer(_t(2, 4, device=device))
    assert str(out.device) == f"device('{device}')"
    assert out.shape == (2, 3)


@pytest.mark.parametrize("device", DEVICES)
def test_pad_packed_sequence_roundtrip(device):
    data = np.random.default_rng(0).standard_normal((5, 2, 3)).astype(np.float32)
    lengths = lucid.tensor(np.array([5, 3], dtype=np.int64))
    packed = U.pack_padded_sequence(lucid.tensor(data, device=device), lengths)
    out, out_lengths = U.pad_packed_sequence(packed)
    assert str(out.device) == f"device('{device}')"
    assert out.shape == (5, 2, 3)
    # Only the valid timesteps must survive the round trip.
    assert np.abs(out.numpy()[:5, 0] - data[:5, 0]).max() < 1e-6
    assert np.abs(out.numpy()[:3, 1] - data[:3, 1]).max() < 1e-6


def test_pad_packed_sequence_agrees_across_devices():
    data = np.random.default_rng(1).standard_normal((6, 3, 4)).astype(np.float32)
    lengths = lucid.tensor(np.array([6, 4, 2], dtype=np.int64))
    outs = {}
    for device in DEVICES:
        packed = U.pack_padded_sequence(lucid.tensor(data, device=device), lengths)
        outs[device] = U.pad_packed_sequence(packed)[0].numpy()
    assert np.abs(outs["cpu"] - outs["metal"]).max() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_fuse_conv_bn_weights_needs_positive_variance(device):
    """Guards the probe's own mistake: variance is non-negative by definition.

    A negative ``running_var`` yields NaN through the ``sqrt`` — that is correct
    behaviour for invalid input, not a bug, and this pins the valid case.
    """
    weight = _t(4, 3, 3, 3, device=device)
    zeros = lucid.tensor(np.zeros(4, dtype=np.float32), device=device)
    ones = lucid.tensor(np.ones(4, dtype=np.float32), device=device)
    var = lucid.tensor(np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32), device=device)
    fused_w, fused_b = U.fuse_conv_bn_weights(weight, zeros, zeros, var, 1e-5, ones, zeros)
    assert not np.isnan(fused_w.numpy()).any()
    assert not np.isnan(fused_b.numpy()).any()
    assert str(fused_w.device) == f"device('{device}')"
