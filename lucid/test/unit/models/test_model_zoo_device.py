"""Model-zoo forward parity across devices.

Found 2026-07-26 by sweeping every vision family on Metal: ``crossvit`` raised
``bad_variant_access``.  Its bicubic-resize helper built the sampling
coordinates with ``lucid.arange(...)`` and no ``device=``, then used them to
index the (Metal) feature map — the same CPU-index-into-GPU-data pattern as the
``pdist`` and transforms bugs.

After the fix the sweep reports **0 families that work on the CPU but fail on
Metal**.  These tests keep a representative slice of that sweep in CI, cheaply:
config overrides shrink the models so the check costs a forward pass, not a
real network.
"""

import numpy as np
import pytest

import lucid
import lucid.models as M
from lucid.models.generative._common._rssm import RSSMState

DEVICES = ["cpu", "metal"]


def _output_tensor(out):
    if hasattr(out, "shape"):
        return out
    for attr in ("logits", "last_hidden_state"):
        if hasattr(out, attr):
            return getattr(out, attr)
    raise AssertionError(f"no tensor in model output: {type(out)!r}")


def _forward(name, device, size=224, seed=0, **overrides):
    lucid.manual_seed(0)
    model = M.create_model(name, **overrides).to(device).eval()
    data = np.random.default_rng(seed).standard_normal((1, 3, size, size))
    x = lucid.tensor(data.astype(np.float32), device=device)
    with lucid.no_grad():
        return _output_tensor(model(x)).numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_crossvit_runs_on_device(device):
    """The regression: bicubic-resize coords were CPU-only."""
    out = _forward("crossvit_15", device)
    assert out.ndim == 2
    assert not np.isnan(out).any()


def test_crossvit_matches_across_devices():
    cpu = _forward("crossvit_15", "cpu")
    metal = _forward("crossvit_15", "metal")
    assert cpu.shape == metal.shape
    assert np.abs(cpu - metal).max() < 1e-4


# A slice across architecture styles: plain CNN, residual, depthwise,
# windowed attention, plain ViT.  Kept small so the sweep is cheap.
_REPRESENTATIVE = [
    ("resnet_18", 224),
    ("mobilenet_v2", 224),
    ("convnext_tiny", 224),
    ("swin_tiny", 224),
    ("vit_base_16", 224),
]


@pytest.mark.parametrize("name,size", _REPRESENTATIVE)
def test_representative_families_match_across_devices(name, size):
    cpu = _forward(name, "cpu", size=size)
    metal = _forward(name, "metal", size=size)
    assert cpu.shape == metal.shape, name
    assert not np.isnan(metal).any(), name
    assert np.abs(cpu - metal).max() < 1e-3, name


@pytest.mark.parametrize("name,size", _REPRESENTATIVE)
@pytest.mark.parametrize("device", DEVICES)
def test_representative_families_train_one_step(name, size, device):
    """Forward + backward + step — the gradient path must stay on-device too."""
    lucid.manual_seed(0)
    model = M.create_model(name).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
    data = np.random.default_rng(1).standard_normal((1, 3, size, size))
    x = lucid.tensor(data.astype(np.float32), device=device)
    out = _output_tensor(model(x))
    loss = (out * out).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{name}: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')", name


# ─────────────────────────────────────────────────────────────────────────────
# Continuous flows
#
# The sweep above drives every model through one ``model(x)`` call, which the
# flow families do not fit: Flow Matching's forward takes ``(sample, t)``, and
# what actually needs checking is not a forward pass but a *solve* — an ODE
# integrated on-device, with a divergence taken by autograd inside it.  That is
# the pattern most likely to leak a CPU tensor into GPU data, so it gets its
# own checks rather than being left out of the sweep.
# ─────────────────────────────────────────────────────────────────────────────


def _paired(name, **overrides):
    """The same model on both devices, weight-for-weight."""
    lucid.manual_seed(0)
    cpu = M.create_model(name, **overrides).eval()
    metal = M.create_model(name, **overrides).eval()
    metal.load_state_dict(cpu.state_dict())
    return cpu, metal.to("metal")


def _agree(cpu_out, metal_out, tol, what):
    # Guard the instrument: without this, a model that quietly stayed on
    # the CPU would make every comparison below trivially true, and the
    # check would keep passing while testing nothing.
    assert str(cpu_out.device) == "device('cpu')", what
    assert str(metal_out.device) == "device('metal')", what
    a = cpu_out.numpy()
    b = metal_out.to("cpu").numpy()
    assert a.shape == b.shape, what
    assert not np.isnan(b).any(), what
    scale = max(float(np.abs(a).max()), 1e-8)
    assert (
        np.abs(a - b).max() / scale < tol
    ), f"{what}: rel {np.abs(a - b).max() / scale}"


def test_neural_ode_matches_across_devices():
    """Encode, decode and log-density must agree wherever they are solved."""
    cpu, metal = _paired(
        "neural_ode",
        sample_size=(1, 2),
        in_channels=1,
        out_channels=1,
        hidden_dim=16,
        rtol=1e-6,
        atol=1e-8,
    )
    x = lucid.rand((4, 1, 1, 2))
    x_gpu = x.to("metal")

    z_cpu, det_cpu = cpu.encode(x)
    z_metal, det_metal = metal.encode(x_gpu)
    _agree(z_cpu, z_metal, 1e-4, "neural_ode encode")
    _agree(det_cpu, det_metal, 1e-4, "neural_ode log_det")
    _agree(cpu.decode(z_cpu), metal.decode(z_metal), 1e-4, "neural_ode decode")
    # Exact trace at two dimensions, so this one is deterministic.
    _agree(cpu.log_prob(x), metal.log_prob(x_gpu), 1e-4, "neural_ode log_prob")


def test_flow_matching_matches_across_devices():
    """Field, path, target and a fixed-noise solve.

    ``log_prob`` is left out on purpose: above two dimensions the trace is
    a Hutchinson estimate, and the two devices draw their own probes — the
    disagreement would be Monte-Carlo variance, not a device bug.
    """
    cpu, metal = _paired(
        "flow_matching_cifar",
        sample_size=8,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(4,),
        resnet_groups=8,
    )
    x1 = lucid.randn((2, 3, 8, 8))
    x0 = lucid.randn((2, 3, 8, 8))
    t = lucid.tensor([0.3, 0.7])
    g1, g0, gt = x1.to("metal"), x0.to("metal"), t.to("metal")

    _agree(
        cpu(x1, lucid.tensor(0.4)).sample,
        metal(g1, lucid.tensor(0.4, device="metal")).sample,
        1e-4,
        "flow_matching velocity",
    )
    _agree(
        cpu.path_sample(x1, x0, t), metal.path_sample(g1, g0, gt), 1e-5, "path_sample"
    )
    _agree(
        cpu.conditional_target(x1, x0, t),
        metal.conditional_target(g1, g0, gt),
        1e-5,
        "conditional_target",
    )
    _agree(
        cpu.sample(noise=x0, steps=8),
        metal.sample(noise=g0, steps=8),
        1e-4,
        "flow_matching sample",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_flow_matching_trains_one_step_on_device(device):
    """The objective is simulation-free, so no solve should appear here."""
    lucid.manual_seed(0)
    model = M.create_model(
        "flow_matching_cifar_gen",
        sample_size=8,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(4,),
        resnet_groups=8,
    ).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
    out = model(lucid.randn((2, 3, 8, 8), device=device))
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    assert model.nfe == 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "flow_matching: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


def test_rectified_flow_matches_across_devices():
    """Path, target, a fixed-noise solve, and the straightness measure.

    ``straightness`` is included because it is the only quantity here that
    both solves *and* reduces over the whole batch, so a stray CPU tensor
    inside the loop would show up as a disagreement rather than as a
    crash.
    """
    cpu, metal = _paired(
        "rectified_flow_cifar",
        sample_size=8,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        resnet_groups=8,
        init_scale=1.0,
    )
    x1 = lucid.randn((2, 3, 8, 8))
    x0 = lucid.randn((2, 3, 8, 8))
    t = lucid.tensor([0.3, 0.7])
    g1, g0, gt = x1.to("metal"), x0.to("metal"), t.to("metal")

    _agree(cpu.path_sample(x1, x0, t), metal.path_sample(g1, g0, gt), 1e-6, "path")
    _agree(
        cpu.conditional_target(x1, x0, t),
        metal.conditional_target(g1, g0, gt),
        1e-6,
        "target",
    )
    _agree(
        cpu.sample(noise=x0, steps=8), metal.sample(noise=g0, steps=8), 1e-4, "sample"
    )
    _agree(cpu.one_step(x0), metal.one_step(g0), 1e-4, "one_step")

    s_cpu = float(cpu.straightness(x0, steps=8))
    s_metal = float(metal.straightness(g0, steps=8))
    assert abs(s_cpu - s_metal) / max(abs(s_cpu), 1e-8) < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_rectified_flow_reflow_round_trips_on_device(device):
    """Generate couplings, then train on them — both halves must stay put.

    This is the one training path in the flow families that contains a
    solve, so it is where a device leak would be easiest to miss.
    """
    lucid.manual_seed(0)
    model = M.create_model(
        "rectified_flow_cifar_gen",
        sample_size=8,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        resnet_groups=8,
        init_scale=1.0,
    ).to(device)

    model.eval()
    z0, z1 = model.reflow_pairs(n_samples=2, device=device, steps=4)
    assert str(z0.device) == f"device('{device}')"
    assert str(z1.device) == f"device('{device}')"

    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
    out = model(z1, noise=z0)
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "rectified_flow: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


@pytest.mark.parametrize("device", DEVICES)
def test_rectified_flow_trains_one_step_on_device(device):
    """The objective is simulation-free, so no solve should appear here."""
    lucid.manual_seed(0)
    model = M.create_model(
        "rectified_flow_cifar_gen",
        sample_size=8,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        resnet_groups=8,
    ).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
    out = model(lucid.randn((2, 3, 8, 8), device=device))
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    assert model.nfe == 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "rectified_flow: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


def test_rectified_flow_high_resolution_path_matches_across_devices():
    """Filtered resampling and both pyramids — buffers must travel with ``.to()``."""
    cpu, metal = _paired(
        "rectified_flow_afhq_cat",
        sample_size=16,
        base_channels=8,
        channel_mult=(1, 2, 2),
        num_res_blocks=1,
        attention_resolutions=(4,),
        init_scale=1.0,
    )
    x = lucid.randn((2, 3, 16, 16))
    t = lucid.tensor(0.4)
    _agree(
        cpu(x, t).sample,
        metal(x.to("metal"), t.to("metal")).sample,
        1e-3,
        "fir + pyramid field",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_neural_ode_trains_one_step_on_device(device):
    """Here a solve *is* the forward pass, and its gradients must stay put."""
    lucid.manual_seed(0)
    model = M.create_model(
        "neural_ode_gen",
        sample_size=(1, 2),
        in_channels=1,
        out_channels=1,
        hidden_dim=16,
        rtol=1e-4,
        atol=1e-4,
    ).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)
    out = model(lucid.rand((2, 1, 1, 2), device=device))
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    assert model.nfe > 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "neural_ode: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


# ─────────────────────────────────────────────────────────────────────────────
# Discrete latents
#
# VQ-VAE is the one family whose forward *computes indices on the device and
# then indexes with them* — ``argmin`` over a distance matrix, straight into a
# codebook gather.  That is the exact shape of the bug this file was opened
# for (``crossvit`` built sampling coordinates on the CPU and indexed a Metal
# feature map with them), so it does not belong in the tolerance-based sweep
# above: the interesting output is an integer field, and a float tolerance
# cannot see a single entry landing on the wrong codebook row.
# ─────────────────────────────────────────────────────────────────────────────

_VQVAE_SMALL = {
    "sample_size": 16,
    "num_embeddings": 32,
    "embedding_dim": 8,
    "hidden_channels": 16,
    "residual_hidden_channels": 16,
}


def test_vqvae_matches_across_devices():
    """Encoder, quantiser and decoder must agree wherever they run."""
    cpu, metal = _paired("vqvae", **_VQVAE_SMALL)
    x = lucid.rand((2, 3, 16, 16))
    x_gpu = x.to("metal")

    z_cpu, z_metal = cpu.encode(x), metal.encode(x_gpu)
    _agree(z_cpu, z_metal, 1e-4, "vqvae encode")

    q_cpu, q_metal = cpu.quantize(z_cpu), metal.quantize(z_metal)
    _agree(q_cpu.quantized, q_metal.quantized, 1e-4, "vqvae quantized")
    _agree(
        cpu.decode(q_cpu.quantized),
        metal.decode(q_metal.quantized),
        1e-4,
        "vqvae decode",
    )


def test_vqvae_codebook_indices_match_across_devices():
    """The discrete field is exact or it is wrong — no tolerance applies.

    One entry picking a different codebook row changes the reconstruction
    outright, and every float comparison in this file would still pass: the
    two rows are both plausible vectors of the same magnitude.
    """
    cpu, metal = _paired("vqvae", **_VQVAE_SMALL)
    x = lucid.rand((2, 3, 16, 16))

    idx_cpu = cpu.encode_indices(x)
    idx_metal = metal.encode_indices(x.to("metal"))
    assert str(idx_cpu.device) == "device('cpu')"
    assert str(idx_metal.device) == "device('metal')"

    a = idx_cpu.numpy()
    b = idx_metal.to("cpu").numpy()
    assert a.shape == b.shape == (2, 4, 4)
    assert (a == b).all(), f"{int((a != b).sum())} of {a.size} positions disagree"


def test_vqvae_detokenises_from_device_indices():
    """A gather driven by indices that were produced on the device."""
    cpu, metal = _paired("vqvae", **_VQVAE_SMALL)
    x = lucid.rand((2, 3, 16, 16))

    out_cpu = cpu.decode_indices(cpu.encode_indices(x))
    out_metal = metal.decode_indices(metal.encode_indices(x.to("metal")))
    _agree(out_cpu, out_metal, 1e-4, "vqvae decode_indices")


@pytest.mark.parametrize("device", DEVICES)
def test_vqvae_trains_one_step_on_device(device):
    """The three-term objective, and the codebook's own gradient path.

    Asserted separately from the rest: the codebook is the one parameter the
    reconstruction term cannot reach — straight-through routes past it — so a
    device bug in the codebook term alone would leave every other parameter
    looking healthy.
    """
    lucid.manual_seed(0)
    model = M.create_model("vqvae_gen", **_VQVAE_SMALL).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-4)

    out = model(lucid.rand((2, 3, 16, 16), device=device))
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    codebook = model.vqvae.quantizer.weight
    assert codebook.grad is not None, "vqvae: the codebook received no gradient"
    assert float(abs(codebook.grad).sum()) > 0.0

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "vqvae: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


@pytest.mark.parametrize("device", DEVICES)
def test_vqvae_generate_runs_on_device(device):
    """Sampling draws integer codes and gathers with them, on-device."""
    lucid.manual_seed(0)
    model = M.create_model("vqvae_gen", **_VQVAE_SMALL).to(device).eval()
    samples = model.generate(2).samples

    assert str(samples.device) == f"device('{device}')"
    assert samples.shape == (2, 3, 16, 16)
    assert not np.isnan(samples.to("cpu").numpy()).any()


# ─────────────────────────────────────────────────────────────────────────────
# World models
#
# PlaNet does not fit the sweep above: its forward takes two sequences and
# returns neither ``logits`` nor ``last_hidden_state``.  It also cannot be
# compared trajectory-for-trajectory across devices — every step draws a
# fresh ``randn`` for the reparameterised latent, and the CPU and Metal
# kernels turn the one stream into different samples (a Metal ``rand(4)``
# advances it one block, a CPU one four), so the sequences diverge after
# step 0 by design rather than by defect.
#
# What *is* deterministic gets compared exactly: the encoder over the whole
# sequence, the decoder and the reward head from a fixed feature tensor,
# and the first dynamics step, which depends only on the zero-initialised
# state and ``actions[:, 0]``.  Between them they cover every layer.
# ─────────────────────────────────────────────────────────────────────────────

_PLANET_SMALL = {
    "action_dim": 2,
    "stoch_size": 4,
    "deter_size": 8,
    "hidden_size": 8,
    "cnn_depth": 4,
    "reward_hidden": 8,
}


def test_planet_encoder_matches_across_devices():
    """The convolutional embedding is sampling-free, so it must agree in full."""
    cpu, metal = _paired("planet", **_PLANET_SMALL)
    obs = lucid.rand((2, 3, 3, 64, 64))
    _agree(cpu.encode(obs), metal.encode(obs.to("metal")), 1e-4, "planet encode")


def test_planet_first_dynamics_step_matches_across_devices():
    """Step 0 is deterministic — nothing sampled has entered it yet."""
    cpu, metal = _paired("planet", **_PLANET_SMALL)
    obs = lucid.rand((2, 1, 3, 64, 64))
    act = lucid.rand((2, 1, 2))

    p_cpu, q_cpu = cpu.observe(obs, act)
    p_metal, q_metal = metal.observe(obs.to("metal"), act.to("metal"))

    _agree(p_cpu.deter, p_metal.deter, 1e-4, "planet deter")
    _agree(p_cpu.mean, p_metal.mean, 1e-4, "planet prior mean")
    _agree(p_cpu.std, p_metal.std, 1e-4, "planet prior std")
    _agree(q_cpu.mean, q_metal.mean, 1e-4, "planet posterior mean")
    _agree(q_cpu.std, q_metal.std, 1e-4, "planet posterior std")


def test_planet_decoder_and_reward_match_across_devices():
    """Both heads read a state; feed them the same one on each device."""
    cpu, metal = _paired("planet", **_PLANET_SMALL)
    feature = lucid.rand((2, 3, cpu.config.latent_size))
    state = RSSMState(
        deter=feature[..., : cpu.config.deter_size],
        stoch=feature[..., cpu.config.deter_size :],
        mean=feature[..., cpu.config.deter_size :],
        std=feature[..., cpu.config.deter_size :],
    )
    gpu_state = state.map(lambda t: t.to("metal"))

    _agree(cpu.decode(state), metal.decode(gpu_state), 1e-4, "planet decode")
    _agree(
        cpu.predict_reward(state),
        metal.predict_reward(gpu_state),
        1e-4,
        "planet reward",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_planet_imagines_on_device(device):
    """The longest recurrence in the zoo — a place a CPU index could leak."""
    lucid.manual_seed(0)
    model = M.create_model("planet", **_PLANET_SMALL).to(device).eval()
    actions = lucid.rand((2, 6, 2), device=device)

    imagined = model.imagine(model.rssm.initial(2, device=device), actions)
    assert str(imagined.stoch.device) == f"device('{device}')"
    assert imagined.stoch.shape == (2, 6, 4)
    assert not np.isnan(imagined.stoch.to("cpu").numpy()).any()


@pytest.mark.parametrize("device", DEVICES)
def test_planet_trains_one_step_on_device(device):
    """Forward, backward and step over the full unroll, staying on-device."""
    lucid.manual_seed(0)
    model = M.create_model("planet_world_model", **_PLANET_SMALL).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-5)

    out = model(
        lucid.rand((2, 3, 3, 64, 64), device=device),
        lucid.rand((2, 3, 2), device=device),
        rewards=lucid.rand((2, 3), device=device),
    )
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "planet: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


# ─────────────────────────────────────────────────────────────────────────────
# Dreamer
#
# Everything said about PlaNet above applies — plus one more sampling
# source.  Dreamer's imagination draws twice per step, once for the latent
# and once for the action, so an imagined trajectory diverges across devices
# even faster.  The deterministic surfaces are compared exactly; the
# behaviour pass is checked for residency and finiteness instead, which is
# what the device test is actually for.
# ─────────────────────────────────────────────────────────────────────────────

_DREAMER_SMALL = {
    "action_dim": 2,
    "stoch_size": 4,
    "deter_size": 8,
    "hidden_size": 8,
    "cnn_depth": 4,
    "reward_hidden": 8,
    "actor_hidden": 8,
    "value_hidden": 8,
    "horizon": 3,
}


def test_dreamer_value_and_actor_heads_match_across_devices():
    """Both new heads read a state; feed them the same one on each device."""
    cpu, metal = _paired("dreamer", **_DREAMER_SMALL)
    feature = lucid.rand((2, 3, cpu.config.latent_size))
    state = RSSMState(
        deter=feature[..., : cpu.config.deter_size],
        stoch=feature[..., cpu.config.deter_size :],
        mean=feature[..., cpu.config.deter_size :],
        std=feature[..., cpu.config.deter_size :],
    )
    gpu_state = state.map(lambda t: t.to("metal"))

    _agree(
        cpu.predict_value(state),
        metal.predict_value(gpu_state),
        1e-4,
        "dreamer value",
    )
    # The actor's mean is deterministic even though its sample is not.
    _agree(
        cpu.actor.distribution(state.feature)[0],
        metal.actor.distribution(gpu_state.feature)[0],
        1e-4,
        "dreamer actor mean",
    )
    _agree(
        cpu.act(state, sample=False),
        metal.act(gpu_state, sample=False),
        1e-4,
        "dreamer act(mean)",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_dreamer_imagines_under_its_own_policy_on_device(device):
    """Actor and dynamics alternate for the whole horizon, staying on-device."""
    lucid.manual_seed(0)
    model = M.create_model("dreamer", **_DREAMER_SMALL).to(device).eval()

    states, actions = model.imagine(model.rssm.initial(2, device=device), 5)
    assert str(states.stoch.device) == f"device('{device}')"
    assert str(actions.device) == f"device('{device}')"
    assert states.stoch.shape == (2, 6, 4)
    assert actions.shape == (2, 5, 2)
    assert not np.isnan(actions.to("cpu").numpy()).any()
    assert np.all(np.abs(actions.to("cpu").numpy()) <= 1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_dreamer_three_losses_backward_on_device(device):
    """Each loss must reach its own group's gradients without leaving the device."""
    lucid.manual_seed(0)
    model = M.create_model("dreamer_world_model", **_DREAMER_SMALL).to(device)
    model.train()

    out = model(
        lucid.rand((2, 3, 3, 64, 64), device=device),
        lucid.rand((2, 3, 2), device=device),
        lucid.rand((2, 3), device=device),
    )
    assert out.behavior is not None
    for loss in (out.loss, out.behavior.actor_loss, out.behavior.value_loss):
        assert str(loss.device) == f"device('{device}')"
        assert not np.isnan(loss.to("cpu").numpy()).any()

    model.zero_grad()
    out.behavior.actor_loss.backward()
    grads = [p.grad for p in model.actor_parameters() if p.grad is not None]
    assert grads, "dreamer: the actor received no gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


@pytest.mark.parametrize("device", DEVICES)
def test_dreamer_discount_head_runs_on_device(device):
    """`pcont` swaps a Python float for a tensor inside the return recursion."""
    lucid.manual_seed(0)
    model = M.create_model("dreamer_world_model", pcont=True, **_DREAMER_SMALL).to(
        device
    )
    model.train()

    out = model(
        lucid.rand((2, 4, 3, 64, 64), device=device),
        lucid.rand((2, 4, 2), device=device),
        lucid.rand((2, 4), device=device),
        lucid.ones((2, 4), device=device),
    )
    assert out.pcont_loss is not None and out.behavior is not None
    assert str(out.pcont_loss.device) == f"device('{device}')"
    assert out.behavior.imagined_pcont is not None
    assert str(out.behavior.imagined_pcont.device) == f"device('{device}')"
    assert not np.isnan(out.behavior.lambda_return.to("cpu").numpy()).any()

    model.zero_grad()
    out.pcont_loss.backward()
    grads = [
        p.grad for p in model.dreamer.pcont_head.parameters() if p.grad is not None
    ]
    assert grads, "dreamer: the discount head received no gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')"


@pytest.mark.parametrize("device", DEVICES)
def test_dreamer_rollout_stays_on_device(device):
    """The whole harness — policy belief, driver, replay — without a round trip."""
    from lucid.utils.rollout import LatentPolicy, SequenceReplay, StepResult, rollout

    class _Tiny:
        def reset(self):
            self.t = 0
            return lucid.rand((3, 64, 64), device=device)

        def step(self, action):
            self.t += 1
            return StepResult(
                lucid.rand((3, 64, 64), device=device), 1.0, False, self.t >= 4
            )

    lucid.manual_seed(0)
    model = M.create_model("dreamer", **_DREAMER_SMALL).to(device).eval()
    policy = LatentPolicy(
        model.encode, model.rssm, lambda s: model.act(s, sample=False), 2, noise=0.3
    )

    episode, total = rollout(_Tiny(), policy)
    assert len(episode) == 4
    assert str(episode.observations.device) == f"device('{device}')"
    assert str(episode.actions.device) == f"device('{device}')"
    assert np.all(np.abs(episode.actions.to("cpu").numpy()) <= 1.0)

    replay = SequenceReplay()
    replay.add(episode)
    batch = replay.sample(2, 3)
    assert str(batch.observations.device) == f"device('{device}')"
    assert not np.isnan(batch.observations.to("cpu").numpy()).any()


# ─────────────────────────────────────────────────────────────────────────────
# DreamerV2
#
# The discrete latent takes a different route through the engine than
# anything else in this family — a multinomial draw, a one-hot scatter,
# and erf/erfinv inside the policy — so none of the Dreamer coverage above
# says anything about it running on Metal.
# ─────────────────────────────────────────────────────────────────────────────

_DREAMER_V2_SMALL = {
    "action_dim": 2,
    "cnn_depth": 4,
    "stoch_size": 4,
    "discrete": 5,
    "deter_size": 16,
    "hidden_size": 16,
    "actor_hidden": 16,
    "value_hidden": 16,
    "reward_hidden": 16,
    "horizon": 4,
    "pcont": False,
}


def test_dreamer_v2_encoder_matches_across_devices():
    """Sampling-free, so it must agree exactly."""
    cpu, metal = _paired("dreamer_v2", **_DREAMER_V2_SMALL)
    obs = lucid.rand((2, 3, 3, 64, 64))
    _agree(cpu.encode(obs), metal.encode(obs.to("metal")), 1e-4, "dreamer_v2 encode")


def test_dreamer_v2_categorical_latent_is_one_hot_on_device():
    """The multinomial draw and the one-hot scatter, on the GPU."""
    lucid.manual_seed(0)
    model = M.create_model("dreamer_v2", **_DREAMER_V2_SMALL).to("metal").eval()
    _, posteriors = model.observe(
        lucid.rand((2, 3, 3, 64, 64), device="metal"),
        lucid.rand((2, 3, 2), device="metal"),
    )
    assert str(posteriors.stoch.device) == "device('metal')"
    assert str(posteriors.logits.device) == "device('metal')"
    grid = posteriors.stoch.reshape(2, 3, 4, 5).to("cpu").numpy()
    assert np.allclose(grid.sum(axis=-1), 1.0, atol=1e-5)


def test_dreamer_v2_policy_matches_across_devices():
    """``erf`` and ``erfinv`` sit inside the truncated normal's mode."""
    cpu, metal = _paired("dreamer_v2", **_DREAMER_V2_SMALL)
    feature = lucid.rand((2, 3, cpu.config.latent_size))
    state = RSSMState(
        deter=feature[..., : cpu.config.deter_size],
        stoch=feature[..., cpu.config.deter_size :],
        logits=feature[..., cpu.config.deter_size :].reshape(2, 3, 4, 5),
    )
    gpu_state = state.map(lambda t: t.to("metal"))
    _agree(
        cpu.act(state, sample=False),
        metal.act(gpu_state, sample=False),
        1e-4,
        "dreamer_v2 act(mode)",
    )
    _agree(
        cpu.actor.distribution(state.feature).entropy(),
        metal.actor.distribution(gpu_state.feature).entropy(),
        1e-4,
        "dreamer_v2 entropy",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_dreamer_v2_trains_one_step_on_device(device):
    """All three losses and the target refresh, staying on-device."""
    lucid.manual_seed(0)
    model = M.create_model("dreamer_v2_world_model", **_DREAMER_V2_SMALL).to(device)
    model.train()

    out = model(
        lucid.rand((2, 4, 3, 64, 64), device=device),
        lucid.rand((2, 4, 2), device=device),
        lucid.rand((2, 4), device=device),
    )
    assert out.behavior is not None
    for loss in (out.loss, out.behavior.actor_loss, out.behavior.value_loss):
        assert str(loss.device) == f"device('{device}')"
        assert not np.isnan(loss.to("cpu").numpy()).any()

    model.backward(out)
    model.update_slow_target()
    for group in (model.world_parameters(), model.actor_parameters()):
        grads = [p.grad for p in group if p.grad is not None]
        assert grads
        for g in grads:
            assert str(g.device) == f"device('{device}')"


@pytest.mark.parametrize("device", DEVICES)
def test_dreamer_v2_discrete_policy_on_device(device):
    """The one-hot action takes the same CPU round-trip the latent does.

    ``multinomial`` is data-dependent, so on Metal its result comes back
    on the CPU — the bug the categorical latent already shipped once.
    """
    lucid.manual_seed(0)
    config = dict(_DREAMER_V2_SMALL, action_dim=4, action_space="discrete")
    model = M.create_model("dreamer_v2_world_model", **config).to(device)
    model.train()

    out = model(
        lucid.rand((2, 4, 3, 64, 64), device=device),
        lucid.rand((2, 4, 4), device=device),
        lucid.rand((2, 4), device=device),
    )
    assert out.behavior is not None
    drawn = out.behavior.imagined_action
    assert str(drawn.device) == f"device('{device}')"
    assert np.allclose(drawn.to("cpu").numpy().sum(axis=-1), 1.0, atol=1e-4)

    model.backward(out)
    grads = [p.grad for p in model.actor_parameters() if p.grad is not None]
    assert grads
    for g in grads:
        assert str(g.device) == f"device('{device}')"


# ─────────────────────────────────────────────────────────────────────────────
# DreamerV3
#
# Three surfaces here are new code rather than a reconfiguration of the
# earlier families, and each has its own way of quietly leaving the
# accelerator.
#
# The **block recurrence** multiplies a grouped activation by a rank-3
# weight, which is the first batched matmul in this corner of the zoo.
#
# The **two-hot head** is written entirely out of comparisons and sums for
# exactly this reason — the obvious implementation reaches for a bucket
# index, and a data-dependent index is what sent DreamerV2's discrete
# action through a CPU round-trip. The test pins that it did not.
#
# The **return normaliser** sorts, which is the one operation here whose
# Metal implementation is not a straight elementwise map.
# ─────────────────────────────────────────────────────────────────────────────

_DREAMER_V3_SMALL = {
    "action_dim": 2,
    "stoch_size": 4,
    "discrete": 5,
    "deter_size": 8,
    "hidden_size": 8,
    "blocks": 4,
    "cnn_depth": 4,
    "reward_hidden": 8,
    "actor_hidden": 8,
    "value_hidden": 8,
    "num_bins": 7,
    "horizon": 3,
    "pcont": False,
}


def test_dreamer_v3_block_recurrence_matches_across_devices():
    """A rank-3 weight and a grouped reshape, compared elementwise."""
    cpu, metal = _paired("dreamer_v3_12m", **_DREAMER_V3_SMALL)
    stoch = lucid.rand((2, 20))
    action = lucid.rand((2, 2))
    deter = lucid.rand((2, 8))
    _agree(
        cpu.rssm._recurrent(stoch, action, deter),
        metal.rssm._recurrent(stoch.to("metal"), action.to("metal"), deter.to("metal")),
        1e-4,
        "dreamer_v3 block recurrence",
    )


def test_dreamer_v3_two_hot_head_matches_across_devices():
    """Prediction and loss, over four orders of magnitude of target."""
    cpu, metal = _paired("dreamer_v3_12m", **_DREAMER_V3_SMALL)
    feature = lucid.rand((2, 3, cpu.config.latent_size))
    target = lucid.tensor([[0.01, 1.0, 10000.0], [-5.0, 0.0, 250.0]])
    _agree(
        cpu.reward_head.predict(feature),
        metal.reward_head.predict(feature.to("metal")),
        1e-4,
        "dreamer_v3 two-hot predict",
    )
    _agree(
        cpu.reward_head.cross_entropy(feature, target),
        metal.reward_head.cross_entropy(feature.to("metal"), target.to("metal")),
        1e-4,
        "dreamer_v3 two-hot loss",
    )


@pytest.mark.parametrize("device", DEVICES)
def test_dreamer_v3_return_normaliser_sorts_on_device(device):
    """The percentile spread, which is the one sort in this family."""
    from lucid.models.generative.dreamer_v3._objectives import percentile

    values = lucid.tensor(
        [float(i) for i in range(101)], device=device, dtype=lucid.float32
    )
    for fraction in (5.0, 50.0, 95.0):
        got = percentile(values, fraction)
        assert str(got.device) == f"device('{device}')"
        assert abs(float(got.to("cpu").item()) - fraction) < 1e-4


@pytest.mark.parametrize("device", DEVICES)
def test_dreamer_v3_trains_one_step_on_device(device):
    """All three losses, the replay critic and the slow update, on-device."""
    lucid.manual_seed(0)
    model = M.create_model("dreamer_v3_12m_world_model", **_DREAMER_V3_SMALL).to(device)
    model.train()

    out = model(
        lucid.rand((2, 4, 3, 64, 64), device=device),
        lucid.rand((2, 4, 2), device=device),
        lucid.rand((2, 4), device=device),
    )
    assert out.behavior is not None
    assert out.behavior.replay_value_loss is not None
    for loss in (
        out.loss,
        out.dynamics_loss,
        out.representation_loss,
        out.behavior.actor_loss,
        out.behavior.value_loss,
    ):
        assert str(loss.device) == f"device('{device}')"
        assert not np.isnan(loss.to("cpu").numpy()).any()

    model.backward(out)
    model.update_slow_critic()
    for group in (model.world_parameters(), model.actor_parameters()):
        grads = [p.grad for p in group if p.grad is not None]
        assert grads
        for g in grads:
            assert str(g.device) == f"device('{device}')"


@pytest.mark.parametrize("device", DEVICES)
def test_dreamer_v3_discrete_policy_on_device(device):
    """Gumbel-max, so the one-hot draw never leaves the accelerator."""
    lucid.manual_seed(0)
    config = dict(_DREAMER_V3_SMALL, action_dim=4, action_space="discrete")
    model = M.create_model("dreamer_v3_12m_world_model", **config).to(device)
    model.train()

    out = model(
        lucid.rand((2, 4, 3, 64, 64), device=device),
        lucid.rand((2, 4, 4), device=device),
        lucid.rand((2, 4), device=device),
    )
    assert out.behavior is not None
    drawn = out.behavior.imagined_action
    assert str(drawn.device) == f"device('{device}')"
    assert np.allclose(drawn.to("cpu").numpy().sum(axis=-1), 1.0, atol=1e-4)

    model.backward(out)
    grads = [p.grad for p in model.actor_parameters() if p.grad is not None]
    assert grads
    for g in grads:
        assert str(g.device) == f"device('{device}')"


@pytest.mark.parametrize("device", DEVICES)
def test_dreamer_discrete_policy_on_device(device):
    """Dreamer v1's categorical policy, drawn straight-through on-device.

    Same hazard the v2 discrete path had: a data-dependent draw round-trips
    to the CPU on Metal. ``OneHotCategorical`` uses Gumbel-max precisely so
    it does not, and the imagined actions must come back on the device they
    were asked for.
    """
    lucid.manual_seed(0)
    config = dict(_DREAMER_SMALL, action_dim=4, action_space="discrete", pcont=False)
    model = M.create_model("dreamer_world_model", **config).to(device)
    model.train()

    out = model(
        lucid.rand((2, 4, 3, 64, 64), device=device),
        lucid.rand((2, 4, 4), device=device),
        rewards=lucid.rand((2, 4), device=device),
    )
    assert out.behavior is not None
    drawn = out.behavior.imagined_action
    assert str(drawn.device) == f"device('{device}')"
    assert np.allclose(drawn.to("cpu").numpy().sum(axis=-1), 1.0, atol=1e-4)

    model.backward(out)
    grads = [p.grad for p in model.actor_parameters() if p.grad is not None]
    assert grads
    for g in grads:
        assert str(g.device) == f"device('{device}')"


# ─────────────────────────────────────────────────────────────────────────────
# Score-SDE
#
# Three samplers, and the one worth the most attention is the ODE — it
# leaves this family entirely and runs inside ``lucid.diffeq``, so a
# device that is lost there is lost silently.  The likelihood adds a
# backward pass per solver step, which is the other place residency can
# slip.
# ─────────────────────────────────────────────────────────────────────────────

_SCORE_SDE_SMALL = {
    "sample_size": 8,
    "base_channels": 8,
    "channel_mult": (1,),
    "num_res_blocks": 1,
    "resnet_groups": 4,
    "attention_resolutions": (),
    "num_scales": 20,
}


def test_score_sde_score_matches_across_devices():
    """The network and the sigma it is divided by, compared elementwise."""
    cpu, metal = _paired("score_sde_vp", **_SCORE_SDE_SMALL)
    x = lucid.rand((2, 3, 8, 8))
    t = lucid.ones((2,)) * 0.5
    _agree(
        cpu.score(x, t),
        metal.score(x.to("metal"), t.to("metal")),
        1e-4,
        "score_sde score",
    )


@pytest.mark.parametrize("kind", ["vp", "ve", "subvp"])
def test_score_sde_marginal_prob_matches_across_devices(kind):
    """The perturbation kernel is pure arithmetic — it must agree exactly."""
    cpu, metal = _paired(f"score_sde_{kind}", **_SCORE_SDE_SMALL)
    x = lucid.rand((4, 3, 8, 8))
    t = lucid.ones((4,)) * 0.3
    mean_c, std_c = cpu.sde.marginal_prob(x, t)
    mean_m, std_m = metal.sde.marginal_prob(x.to("metal"), t.to("metal"))
    _agree(mean_c, mean_m, 1e-5, f"score_sde {kind} kernel mean")
    _agree(std_c, std_m, 1e-5, f"score_sde {kind} kernel std")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["euler", "pc", "ode"])
def test_score_sde_samples_on_device(device, method):
    """Every sampler, including the one that runs inside lucid.diffeq."""
    lucid.manual_seed(0)
    model = M.create_model("score_sde_vp_gen", **_SCORE_SDE_SMALL).to(device).eval()
    samples = model.generate(2, method=method, steps=4, device=device).samples
    assert str(samples.device) == f"device('{device}')"
    assert not np.isnan(samples.to("cpu").numpy()).any()


@pytest.mark.parametrize("device", DEVICES)
def test_score_sde_likelihood_on_device(device):
    """A backward pass per solver step — where residency most easily slips."""
    lucid.manual_seed(0)
    model = M.create_model("score_sde_vp_gen", **_SCORE_SDE_SMALL).to(device).eval()
    out = model.log_likelihood(lucid.rand((2, 3, 8, 8), device=device), steps=3)
    assert str(out.device) == f"device('{device}')"
    assert not np.isnan(out.to("cpu").numpy()).any()


@pytest.mark.parametrize("device", DEVICES)
def test_score_sde_trains_one_step_on_device(device):
    lucid.manual_seed(0)
    model = M.create_model("score_sde_vp_gen", **_SCORE_SDE_SMALL).to(device)
    model.train()
    out = model(lucid.rand((2, 3, 8, 8), device=device))
    assert str(out.loss.device) == f"device('{device}')"
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    for g in grads:
        assert str(g.device) == f"device('{device}')"


# ─────────────────────────────────────────────────────────────────────────────
# Set-prediction losses (DETR, MaskFormer)
#
# Both losses Hungarian-match on the host — the data-dependent CPU round-trip
# H3 allows — and then build per-query targets, class weights and match
# indices from the Python result.  Those tensors were created without
# ``device=``, so they sat on the CPU and the first op mixing them with Metal
# predictions raised ``DeviceMismatch`` (CHA-66): neither family could take a
# training step on Metal, while every forward-only check above passed.
# ─────────────────────────────────────────────────────────────────────────────

_DETR_SMALL = {
    "num_classes": 4,
    "backbone_layers": (1, 1, 1, 1),
    "n_head": 2,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dim_feedforward": 32,
    "num_queries": 4,
}
_MASKFORMER_SMALL = {
    "num_classes": 4,
    "backbone_layers": (1, 1, 1, 1),
    "n_head": 2,
    "num_decoder_layers": 1,
    "dim_feedforward": 32,
    "num_queries": 4,
    "fpn_out_channels": 32,
}
_SET_PREDICTION = {
    "detr": ("detr_resnet50", _DETR_SMALL),
    "maskformer": ("maskformer_resnet50", _MASKFORMER_SMALL),
}


def _set_targets(family, device):
    """Fixed targets, built on the CPU and moved, so both devices see the same."""
    if family == "detr":
        boxes = [
            [[0.25, 0.25, 0.75, 0.75], [0.1, 0.2, 0.4, 0.9]],
            [[0.5, 0.1, 0.9, 0.6], [0.0, 0.0, 0.3, 0.3]],
        ]
        labels = [[1, 3], [0, 2]]
        return [
            {
                "boxes": lucid.tensor(b).to(device),
                "labels": lucid.tensor(lb).long().to(device),
            }
            for b, lb in zip(boxes, labels)
        ]
    masks = np.random.default_rng(3).integers(0, 4, (2, 64, 64))
    return {"masks": lucid.tensor(masks.astype(np.int64)).to(device)}


def _set_images(seed=2):
    data = np.random.default_rng(seed).standard_normal((2, 3, 64, 64))
    return lucid.tensor(data.astype(np.float32))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", sorted(_SET_PREDICTION))
def test_set_prediction_family_trains_one_step_on_device(family, device):
    """Factory in, loss out, backward, step — every gradient on the device."""
    factory, overrides = _SET_PREDICTION[family]
    lucid.manual_seed(0)
    model = M.create_model(factory, **overrides).to(device)
    model.train()
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-2)
    before = {
        n: p.detach().to("cpu").numpy().copy() for n, p in model.named_parameters()
    }

    out = model(_set_images().to(device), targets=_set_targets(family, device))
    loss = out.loss
    assert loss.ndim == 0, f"{family}: loss has shape {loss.shape}"
    assert str(loss.device) == f"device('{device}')", family
    assert np.isfinite(float(loss.item())), family
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{family}: no parameter received a gradient"
    for g in grads:
        assert str(g.device) == f"device('{device}')", family
    moved = [
        n
        for n, p in model.named_parameters()
        if np.abs(p.detach().to("cpu").numpy() - before[n]).max() > 0
    ]
    assert moved, f"{family}: the step changed no parameter"


@pytest.mark.parametrize("family", sorted(_SET_PREDICTION))
def test_set_prediction_loss_matches_across_devices(family):
    """Same weights, inputs and targets: the loss and its gradients agree.

    Dropout is off — the two devices draw their own masks, and the
    disagreement would be sampling, not a device bug.  Gradients that are
    zero in exact arithmetic (a key-projection bias under softmax) come out
    as ~1e-9 noise on both sides, so the bound has an absolute floor.
    """
    factory, overrides = _SET_PREDICTION[family]
    cpu, metal = _paired(factory, **overrides, dropout=0.0)
    x = _set_images()

    cpu.train()
    metal.train()
    loss_cpu = cpu(x, targets=_set_targets(family, "cpu")).loss
    loss_metal = metal(x.to("metal"), targets=_set_targets(family, "metal")).loss
    _agree(loss_cpu, loss_metal, 1e-4, f"{family} loss")

    loss_cpu.backward()
    loss_metal.backward()
    metal_params = dict(metal.named_parameters())
    for name, p in cpu.named_parameters():
        q = metal_params[name]
        assert (p.grad is None) == (q.grad is None), f"{family} {name}"
        if p.grad is None:
            continue
        assert str(q.grad.device) == "device('metal')", f"{family} {name}"
        a = p.grad.numpy()
        b = q.grad.to("cpu").numpy()
        bound = 1e-6 + 1e-3 * float(np.abs(a).max())
        assert np.abs(a - b).max() <= bound, f"{family} {name}"


@pytest.mark.parametrize("family", sorted(_SET_PREDICTION))
def test_set_prediction_inference_matches_across_devices(family):
    factory, overrides = _SET_PREDICTION[family]
    cpu, metal = _paired(factory, **overrides)
    x = _set_images()
    with lucid.no_grad():
        out_cpu = cpu(x)
        out_metal = metal(x.to("metal"))
    _agree(out_cpu.logits, out_metal.logits, 1e-4, f"{family} logits")
    if family == "detr":
        _agree(out_cpu.pred_boxes, out_metal.pred_boxes, 1e-4, "detr boxes")


def test_detr_postprocess_stays_on_device():
    """Detections come back where the predictions are, not on the CPU."""
    factory, overrides = _SET_PREDICTION["detr"]
    cpu, metal = _paired(factory, **overrides, score_thresh=0.0)
    x = _set_images()
    with lucid.no_grad():
        res_cpu = cpu.postprocess(cpu(x), [(64, 64), (64, 64)])
        res_metal = metal.postprocess(metal(x.to("metal")), [(64, 64), (64, 64)])
    for rc, rm in zip(res_cpu, res_metal):
        for key in ("boxes", "scores", "labels"):
            assert str(rm[key].device) == "device('metal')", key
            np.testing.assert_allclose(
                rm[key].to("cpu").numpy(), rc[key].numpy(), rtol=1e-4, atol=1e-4
            )


@pytest.mark.parametrize("device", DEVICES)
def test_detr_image_without_boxes_trains_on_device(device):
    """The no-object-only branch, with the box terms' scalar zero fallbacks."""
    factory, overrides = _SET_PREDICTION["detr"]
    lucid.manual_seed(0)
    model = M.create_model(factory, **overrides).to(device)
    model.train()
    empty = {
        "boxes": lucid.zeros((0, 4), device=device),
        "labels": lucid.zeros((0,), device=device).long(),
    }
    loss = model(_set_images().to(device), targets=[empty, empty]).loss
    assert loss.ndim == 0
    assert str(loss.device) == f"device('{device}')"
    loss.backward()


@pytest.mark.parametrize("device", DEVICES)
def test_maskformer_ignore_only_batch_gives_scalar_loss_on_device(device):
    """No valid class anywhere: only the class term, and still a scalar.

    The mask-term fallback used to be a ``(1,)`` CPU zero, so this batch
    produced a ``(1,)`` loss on the CPU and a ``DeviceMismatch`` on Metal.
    """
    factory, overrides = _SET_PREDICTION["maskformer"]
    lucid.manual_seed(0)
    model = M.create_model(factory, **overrides).to(device)
    model.train()
    ignore = lucid.full((2, 64, 64), 255, device=device).long()
    loss = model(_set_images().to(device), targets={"masks": ignore}).loss
    assert loss.ndim == 0, f"loss has shape {loss.shape}"
    assert str(loss.device) == f"device('{device}')"
    loss.backward()
