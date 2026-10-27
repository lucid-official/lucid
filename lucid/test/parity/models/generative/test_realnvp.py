"""RealNVP parity tests — multi-scale affine flow vs an inline reference build.

The reference framework ships no RealNVP, so parity runs against a
from-scratch float64 reference implementation of the paper's stages that
reads Lucid's own weights.  That pins the pieces a coupling flow can get
silently wrong — which half each mask protects, the squeeze's channel
ordering, the log-determinant of the affine scale, the logit stage's
Jacobian, and the exactness of the inverse.

Batch normalisation is disabled in these fixtures so the flow is a pure
per-sample bijection; the flow-level batch-norm stage is covered by the
unit suite instead.
"""

import math
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
from lucid.models.generative.realnvp import (
    RealNVPConfig,
    RealNVPForImageGeneration,
    RealNVPModel,
)

pytestmark = [pytest.mark.parity]


def _cfg(**overrides: object) -> RealNVPConfig:
    base: dict[str, object] = {
        "sample_size": 8,
        "in_channels": 3,
        "out_channels": 3,
        "num_scales": 2,
        "residual_blocks": 1,
        "base_dim": 4,
        "use_batch_norm": False,
    }
    base.update(overrides)
    return RealNVPConfig(**base)  # type: ignore[arg-type]


def _built(cfg: RealNVPConfig) -> tuple[RealNVPModel, np.ndarray]:
    """A RealNVP model excited off its identity init, plus a data batch."""
    lucid.manual_seed(0)
    model = RealNVPModel(cfg).eval()
    for param in model.parameters():
        if param.ndim == 4:
            lucid.nn.init.normal_(param, mean=0.0, std=0.1)
    for name, param in model.named_parameters():
        if name.endswith("rescale"):
            lucid.nn.init.constant_(param, 0.4)

    rng = np.random.default_rng(0)
    c, h, w = cfg.image_shape
    x = rng.uniform(0.05, 0.95, size=(3, c, h, w)).astype(np.float32)

    # One forward before anything reads the model.  The coupling convs are
    # weight-normalised, so ``conv.weight`` is not a parameter at all — it
    # is ``g · v/‖v‖``, cached on the module and refreshed by a forward
    # pre-hook.  Randomising ``weight_v`` above therefore leaves the cache
    # holding the value from *before* the initialisation, and the mirror
    # below reads ``layer.weight`` directly.  Lucid's own forward never
    # notices, because it runs the hook first; the mirror does not, so it
    # was building its reference out of pre-init weights and disagreeing
    # with a model that was behaving correctly.
    model.encode(lucid.tensor(x))
    return model, x


def _ref_conv(ref: ModuleType, layer: Any, x: Any) -> Any:
    weight = ref.tensor(layer.weight.numpy(), dtype=ref.float64)
    bias = ref.tensor(layer.bias.numpy(), dtype=ref.float64)
    return ref.nn.functional.conv2d(x, weight, bias, padding=1)


def _ref_coupling_net(ref: ModuleType, net: Any, x: Any) -> tuple[Any, Any]:
    """Reference pass through one coupling network → ``(log_scale, shift)``."""
    h = _ref_conv(ref, net.stem, x)
    for block in net.blocks:
        inner = ref.relu(h)
        inner = _ref_conv(ref, block.conv1, inner)
        inner = ref.relu(inner)
        h = h + _ref_conv(ref, block.conv2, inner)
    h = ref.relu(h)
    out = _ref_conv(ref, net.head, h)

    channels = x.shape[1]
    # ``rescale`` is per-channel — ``(1, C, 1, 1)``, broadcast over the
    # coupling output — not the single scalar it started as.  Reading it
    # with ``.item()`` raised "item() can only be called on a tensor with
    # one element" and took all eight parity checks with it, which is a
    # mirror that stopped mirroring rather than a defect in the model.
    rescale = ref.tensor(net.rescale.numpy(), dtype=ref.float64)
    log_scale = rescale * ref.tanh(out[:, :channels])
    return log_scale, out[:, channels:]


def _ref_encode(ref: ModuleType, model: RealNVPModel, x_np: np.ndarray) -> tuple[Any, Any]:
    """Reference forward direction — returns ``(z, log_det)``."""
    cfg = model.config
    constraint = float(cfg.data_constraint)  # type: ignore[attr-defined]
    num_scales = int(cfg.num_scales)  # type: ignore[attr-defined]
    batch = x_np.shape[0]

    x = ref.tensor(x_np, dtype=ref.float64)
    s = (x - 0.5) * constraint + 0.5
    h = ref.log(s) - ref.log(1.0 - s)
    log_det = (
        (ref.nn.functional.softplus(h) + ref.nn.functional.softplus(-h) + math.log(constraint))
        .reshape(batch, -1)
        .sum(dim=-1)
    )

    parts = []
    for idx in range(num_scales):
        last = idx == num_scales - 1
        stages = list(model.scales[idx])
        n_checker = 4 if last else 3

        for stage in stages[:n_checker]:
            h, inc = _ref_coupling(ref, stage, h)
            log_det = log_det + inc
        if last:
            parts.append(h.reshape(batch, -1))
            break

        # squeeze: (B, C, H, W) → (B, 4C, H/2, W/2)
        b, c, height, width = h.shape
        h = h.reshape(b, c, height // 2, 2, width // 2, 2)
        h = h.permute(0, 1, 3, 5, 2, 4).reshape(b, 4 * c, height // 2, width // 2)

        for stage in stages[n_checker:]:
            h, inc = _ref_coupling(ref, stage, h)
            log_det = log_det + inc

        split = h.shape[1] // 2
        parts.append(h[:, :split].reshape(batch, -1))
        h = h[:, split:]

    return ref.cat(parts, dim=-1), log_det


def _ref_coupling(ref: ModuleType, stage: Any, h: Any) -> tuple[Any, Any]:
    mask = ref.tensor(stage.mask.numpy(), dtype=ref.float64)
    log_scale, shift = _ref_coupling_net(ref, stage.net, h * mask)
    keep = 1.0 - mask
    log_scale = log_scale * keep
    shift = shift * keep
    out = h * mask + keep * (h * ref.exp(log_scale) + shift)
    return out, log_scale.reshape(h.shape[0], -1).sum(dim=-1)


def _ref_log_prob(ref: ModuleType, model: RealNVPModel, x_np: np.ndarray) -> Any:
    z, log_det = _ref_encode(ref, model, x_np)
    log_p = -0.5 * (z * z + math.log(2.0 * math.pi))
    return log_p.sum(dim=-1) + log_det


# ─────────────────────────────────────────────────────────────────────────────
# Forward direction
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("num_scales", [1, 2])
def test_latent_parity(ref: ModuleType, num_scales: int) -> None:
    """``encode`` must reproduce the reference bijection element-wise."""
    model, x_np = _built(_cfg(num_scales=num_scales))
    z_lucid, _ = model.encode(lucid.tensor(x_np))
    z_ref, _ = _ref_encode(ref, model, x_np)
    np.testing.assert_allclose(
        z_lucid.numpy(), z_ref.numpy().astype(np.float32), rtol=1e-4, atol=1e-4
    )


def test_log_det_parity(ref: ModuleType) -> None:
    """Logit stage + every affine coupling, summed per sample."""
    model, x_np = _built(_cfg())
    _z, log_det = model.encode(lucid.tensor(x_np))
    _z_ref, log_det_ref = _ref_encode(ref, model, x_np)
    np.testing.assert_allclose(
        log_det.numpy(),
        log_det_ref.numpy().astype(np.float32),
        rtol=1e-4,
        atol=1e-3,
    )


def test_log_prob_parity(ref: ModuleType) -> None:
    """Exact log-likelihood under the Gaussian prior."""
    model, x_np = _built(_cfg())
    got = model.log_prob(lucid.tensor(x_np)).numpy()
    expected = _ref_log_prob(ref, model, x_np).numpy().astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-2)


def test_squeeze_matches_reference_channel_order(ref: ModuleType) -> None:
    """Space-to-depth ordering must agree — a transposed variant would
    still round-trip but would pair the wrong channels in the mask."""
    from lucid.models.generative.realnvp._model import _squeeze

    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
    got = _squeeze(lucid.tensor(x_np)).numpy()

    x = ref.tensor(x_np, dtype=ref.float64)
    b, c, h, w = x.shape
    expected = (
        x.reshape(b, c, h // 2, 2, w // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, 4 * c, h // 2, w // 2)
    )
    np.testing.assert_allclose(got, expected.numpy().astype(np.float32), atol=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Inverse direction + training objective
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("num_scales", [1, 2])
def test_inverse_parity(ref: ModuleType, num_scales: int) -> None:
    """``decode`` must invert the *reference* latent back onto the data.

    Stronger than a Lucid-only round-trip: it proves both directions agree
    on the same bijection rather than on a shared bug.
    """
    model, x_np = _built(_cfg(num_scales=num_scales))
    z_ref, _ = _ref_encode(ref, model, x_np)
    x_back = model.decode(lucid.tensor(z_ref.numpy().astype(np.float32)))
    np.testing.assert_allclose(x_back.numpy(), x_np, rtol=1e-3, atol=1e-3)


def test_bits_per_dim_parity(ref: ModuleType) -> None:
    """The reported metric, in the paper's units."""
    cfg = _cfg()
    model, x_np = _built(cfg)
    got = model.bits_per_dim(lucid.tensor(x_np)).numpy()
    # ``+ num_bits`` is the 8-bit discretisation offset.  ``log_prob`` is a
    # density over ``[0, 1]^D``; the paper reports against the discrete
    # data, and the ``x / 256`` inside the squash is exactly that change of
    # variables — worth ``num_bits`` in bits/dim.  Omitting it here put the
    # reference a full 8 bits/dim below the model and below Table 1.
    expected = (
        -_ref_log_prob(ref, model, x_np).numpy() / (cfg.input_dim * math.log(2.0))
        + float(cfg.num_bits)
    ).astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_loss_parity(ref: ModuleType) -> None:
    """The generation head's loss is the mean bits/dim of the batch."""
    cfg = _cfg()
    model, x_np = _built(cfg)
    head = RealNVPForImageGeneration(cfg).eval()
    head.realnvp.load_state_dict(model.state_dict(), strict=True)

    out = head(lucid.tensor(x_np))
    assert out.loss is not None
    # Mean bits/dim carries the same discretisation offset as
    # ``bits_per_dim`` — the head applies it so the reported loss is on the
    # paper's scale.
    expected = float(
        (-_ref_log_prob(ref, model, x_np) / (cfg.input_dim * math.log(2.0))).mean().item()
    ) + float(cfg.num_bits)
    assert abs(float(out.loss.item()) - expected) < 1e-3
