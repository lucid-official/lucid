"""NICE parity tests — exact-likelihood flow against an inline reference build.

The reference framework ships no NICE, so parity is established against a
from-scratch reference implementation of the paper's flow (tile reorder →
four additive couplings with index reversal → diagonal scaling) that reads
Lucid's own weights.  That still pins every piece of math a flow can get
silently wrong: which half of the split is transformed, whether the roles
alternate, the log-determinant, both prior densities, and the exactness of
the inverse.

The reference side runs in float64 while Lucid runs in float32, so the
tolerances below are the float32 noise floor rather than bit-equality.
"""

import math
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
from lucid.models.generative.nice import NICEConfig, NICEForImageGeneration, NICEModel

pytestmark = [pytest.mark.parity]


def _cfg(**overrides: object) -> NICEConfig:
    base: dict[str, object] = {
        "input_dim": 16,
        "num_coupling_layers": 4,
        "num_hidden_layers": 2,
        "hidden_dim": 8,
    }
    base.update(overrides)
    return NICEConfig(**base)  # type: ignore[arg-type]


def _built(cfg: NICEConfig) -> tuple[NICEModel, np.ndarray]:
    """A NICE model with a non-trivial scaling, plus a data batch."""
    lucid.manual_seed(0)
    model = NICEModel(cfg).eval()
    # The published init keeps couplings near the identity (irange = 0.01)
    # and the scaling at exactly zero; perturb both so parity has to match
    # real numbers rather than a near-identity map.
    for param in model.parameters():
        if param.ndim == 2:
            lucid.nn.init.normal_(param, mean=0.0, std=0.3)
    lucid.nn.init.normal_(model.scaling.log_scale, mean=0.0, std=0.25)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, cfg.input_dim)).astype(np.float32)
    return model, x


def _ref_encode(ref: ModuleType, model: NICEModel, x_np: np.ndarray) -> tuple[Any, Any]:
    """Reference NICE forward direction — returns ``(h, log_det)``."""
    dim = model.input_dim
    split = dim // 2
    batch = x_np.shape[0]

    h = ref.tensor(x_np, dtype=ref.float64)
    if model.config.input_reorder == "tile":  # type: ignore[attr-defined]
        h = h.reshape(batch, dim // 2, 2).transpose(1, 2).reshape(batch, dim)

    for layer in model.couplings:
        h1, h2 = h[:, :split], h[:, split:]
        m = h1
        for linear in layer.coupling.hidden:
            weight = ref.tensor(linear.weight.numpy(), dtype=ref.float64)
            bias = ref.tensor(linear.bias.numpy(), dtype=ref.float64)
            m = ref.relu(m @ weight.T + bias)
        weight = ref.tensor(layer.coupling.proj.weight.numpy(), dtype=ref.float64)
        bias = ref.tensor(layer.coupling.proj.bias.numpy(), dtype=ref.float64)
        m = m @ weight.T + bias

        h = ref.cat([h1, h2 + m], dim=-1)
        h = ref.flip(h, [1])

    log_scale = ref.tensor(model.scaling.log_scale.numpy(), dtype=ref.float64)
    return h * ref.exp(log_scale), log_scale.sum()


def _ref_log_prob(ref: ModuleType, model: NICEModel, x_np: np.ndarray) -> Any:
    """Reference exact log-likelihood, ``log p_H(f(x)) + Σ_i s_i``."""
    h, log_det = _ref_encode(ref, model, x_np)
    if model.prior == "logistic":
        log_p = -(ref.nn.functional.softplus(h) + ref.nn.functional.softplus(-h))
    else:
        log_p = -0.5 * (h * h + math.log(2.0 * math.pi))
    return log_p.sum(dim=-1) + log_det


# ─────────────────────────────────────────────────────────────────────────────
# Forward direction
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reorder", ["tile", "none"])
def test_latent_parity(ref: ModuleType, reorder: str) -> None:
    """``encode`` must reproduce the reference bijection element-wise."""
    model, x_np = _built(_cfg(input_reorder=reorder))
    h_lucid, _ = model.encode(lucid.tensor(x_np))
    h_ref, _ = _ref_encode(ref, model, x_np)
    np.testing.assert_allclose(
        h_lucid.numpy(), h_ref.numpy().astype(np.float32), rtol=1e-4, atol=1e-4
    )


def test_log_det_parity(ref: ModuleType) -> None:
    """Additive couplings contribute nothing; the scaling contributes Σ s_i."""
    model, x_np = _built(_cfg())
    _h, log_det = model.encode(lucid.tensor(x_np))
    _h_ref, log_det_ref = _ref_encode(ref, model, x_np)
    np.testing.assert_allclose(
        log_det.numpy(),
        np.full((x_np.shape[0],), float(log_det_ref.item()), dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("prior", ["logistic", "gaussian"])
def test_log_prob_parity(ref: ModuleType, prior: str) -> None:
    """Exact log-likelihood parity for both priors the paper uses."""
    model, x_np = _built(_cfg(prior=prior))
    got = model.log_prob(lucid.tensor(x_np)).numpy()
    expected = _ref_log_prob(ref, model, x_np).numpy().astype(np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Inverse direction + training objective
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reorder", ["tile", "none"])
def test_inverse_parity(ref: ModuleType, reorder: str) -> None:
    """``decode`` must invert the *reference* latent back onto the data.

    Stronger than a Lucid-only round-trip: it proves the two directions
    agree on the same bijection rather than on a shared bug.
    """
    model, x_np = _built(_cfg(input_reorder=reorder))
    h_ref, _ = _ref_encode(ref, model, x_np)
    x_back = model.decode(lucid.tensor(h_ref.numpy().astype(np.float32)))
    np.testing.assert_allclose(x_back.numpy(), x_np, rtol=1e-3, atol=1e-3)


def test_nll_loss_parity(ref: ModuleType) -> None:
    """The generation head's loss is exactly ``-mean(log p(x))``."""
    cfg = _cfg()
    model, x_np = _built(cfg)
    head = NICEForImageGeneration(cfg).eval()
    head.nice.load_state_dict(model.state_dict(), strict=True)

    out = head(lucid.tensor(x_np))
    assert out.loss is not None
    expected = -float(_ref_log_prob(ref, model, x_np).mean().item())
    assert abs(float(out.loss.item()) - expected) < 1e-2


def test_scaling_gradient_parity(ref: ModuleType) -> None:
    """Backward through the flow matches the reference autograd gradient."""
    cfg = _cfg(prior="gaussian")
    model, x_np = _built(cfg)
    head = NICEForImageGeneration(cfg)
    head.nice.load_state_dict(model.state_dict(), strict=True)

    out = head(lucid.tensor(x_np))
    assert out.loss is not None
    out.loss.backward()
    grad = head.nice.scaling.log_scale.grad
    assert grad is not None

    log_scale = ref.tensor(model.scaling.log_scale.numpy(), dtype=ref.float64, requires_grad=True)
    dim = model.input_dim
    split = dim // 2
    batch = x_np.shape[0]
    h = ref.tensor(x_np, dtype=ref.float64)
    h = h.reshape(batch, dim // 2, 2).transpose(1, 2).reshape(batch, dim)
    for layer in model.couplings:
        h1, h2 = h[:, :split], h[:, split:]
        m = h1
        for linear in layer.coupling.hidden:
            weight = ref.tensor(linear.weight.numpy(), dtype=ref.float64)
            bias = ref.tensor(linear.bias.numpy(), dtype=ref.float64)
            m = ref.relu(m @ weight.T + bias)
        weight = ref.tensor(layer.coupling.proj.weight.numpy(), dtype=ref.float64)
        bias = ref.tensor(layer.coupling.proj.bias.numpy(), dtype=ref.float64)
        h = ref.cat([h1, h2 + m @ weight.T + bias], dim=-1)
        h = ref.flip(h, [1])
    h = h * ref.exp(log_scale)
    log_prob = (-0.5 * (h * h + math.log(2.0 * math.pi))).sum(dim=-1) + log_scale.sum()
    (-log_prob.mean()).backward()

    assert log_scale.grad is not None
    np.testing.assert_allclose(
        grad.numpy(), log_scale.grad.numpy().astype(np.float32), rtol=1e-3, atol=1e-3
    )
