"""LCD-411: the denoising families keep the zoo's output contract end to end.

DiT and MeanFlow's networks once returned a bare tensor, and Stable
Diffusion's task model once aliased ``generate`` and returned pixels — so
a caller following "every forward returns a ``ModelOutput``" got a tensor,
and a training loop written against the image-generation task got images
and no loss.  Each family is driven here through its public factory, at a
size a unit test can afford (config overrides, never an invented factory):
one optimizer step that has to move the weights, and generation on every
device.  Where a family's training loop or decoder must call more than
``forward`` — TVM's weight averages and its latent whitening — that call
is driven here too.  The last section counts host syncs per sampling loop
on Metal, which must not grow with the number of steps.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.models as M
from lucid._C import engine as _C_engine
from lucid.models.generative.ddpm import DDPMConfig, DDPMForImageGeneration
from lucid.models.generative.tvm import TVMForImageGeneration, TVMModel, TVMOutput
from lucid.test._fixtures.devices import metal_available

DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]

_TRANSFORMER = dict(sample_size=8, hidden_size=32, depth=1, num_heads=4, num_classes=4)
_SD = dict(
    sample_size=32,
    vae_block_out_channels=(8, 8),
    vae_layers_per_block=1,
    unet_block_out_channels=(8, 16),
    unet_layers_per_block=1,
    norm_num_groups=2,
    cross_attention_dim=16,
    attention_head_dim=4,
    context_length=8,
    downsample_factor=2,
)


def _build(name: str, device: str) -> lucid.nn.Module:
    overrides = _SD if name.startswith("stable_diffusion") else _TRANSFORMER
    return M.create_model(name, **overrides).to(device)


def _labels(device: str) -> lucid.Tensor:
    return lucid.tensor([0, 1], dtype=lucid.int64).to(device)


# ── the denoising networks return the shared output ────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_dit_network_returns_a_diffusion_output(device: str) -> None:
    model = _build("dit_small_2", device).eval()
    out = model(
        lucid.randn((2, 4, 8, 8), device=device),
        lucid.tensor([10.0, 500.0]).to(device),
        _labels(device),
    )
    assert isinstance(out, M.DiffusionModelOutput)
    assert out.sample.shape == (2, 8, 8, 8) and out.loss is None
    assert out[0] is out.sample


@pytest.mark.parametrize("device", DEVICES)
def test_mean_flow_network_returns_a_diffusion_output(device: str) -> None:
    model = _build("mean_flow_base_2", device).eval()
    out = model(
        lucid.randn((2, 4, 8, 8), device=device),
        lucid.tensor([0.0, 0.2]).to(device),
        lucid.tensor([1.0, 0.7]).to(device),
        _labels(device),
    )
    assert isinstance(out, M.DiffusionModelOutput)
    assert out.sample.shape == (2, 4, 8, 8) and out.loss is None


@pytest.mark.parametrize("device", DEVICES)
def test_tvm_network_returns_a_diffusion_output(device: str) -> None:
    model = _build("tvm_imagenet256", device).eval()
    out = model(
        lucid.randn((2, 4, 8, 8), device=device),
        lucid.tensor([1.0, 0.7]).to(device),
        lucid.tensor([0.0, 0.2]).to(device),
        _labels(device),
    )
    assert isinstance(out, M.DiffusionModelOutput)
    assert out.sample.shape == (2, 4, 8, 8) and out.loss is None


# ── one training step from the public factory ──────────────────────────────


def _dit_step(model: lucid.nn.Module, device: str) -> M.ModelOutput:
    return model(lucid.randn((2, 4, 8, 8), device=device), _labels(device))


def _sd_step(model: lucid.nn.Module, device: str) -> M.ModelOutput:
    return model(lucid.randn((2, 3, 32, 32), device=device), lucid.randn((2, 8, 16), device=device))


_TRAIN: dict[str, Callable[[lucid.nn.Module, str], M.ModelOutput]] = {
    "dit_small_2_gen": _dit_step,
    "mean_flow_base_2_gen": _dit_step,
    "stable_diffusion_gen": _sd_step,
    "tvm_imagenet256_gen": _dit_step,
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(_TRAIN))
def test_a_training_step_moves_the_weights(name: str, device: str) -> None:
    lucid.manual_seed(0)
    model = _build(name, device).train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = lucid.optim.SGD(model.parameters(), lr=0.1)

    out = _TRAIN[name](model, device)
    assert isinstance(out, M.ModelOutput)
    loss = out["loss"]
    assert loss.shape == () and bool(loss.isfinite().item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    moved = [
        n
        for n, p in model.named_parameters()
        if float((p.detach() - before[n]).abs().max().item()) > 0.0
    ]
    assert moved, f"{name}: no parameter changed after an optimizer step"


# ── what a training loop and a decoder call besides forward ────────────────


def _max_abs(x: lucid.Tensor) -> float:
    return float(x.abs().max().item())


@pytest.mark.parametrize("device", DEVICES)
def test_tvm_step_reports_its_terms_and_advances_both_averages(device: str) -> None:
    """Algorithm 1's step as a loop takes it: the loss is the sum of the two
    terms the output reports, and ``update_ema`` after the optimizer step
    writes ``γ·θ₀ + (1 − γ)·θ₁`` into θ* and θ** — both start at θ₀, which
    the first forward copies when it reads the target."""
    lucid.manual_seed(0)
    model = _build("tvm_imagenet256_gen", device).train()
    assert isinstance(model, TVMForImageGeneration)
    initial = {n: p.detach().clone() for n, p in model.tvm.named_parameters()}

    out = model(lucid.randn((2, 4, 8, 8), device=device), _labels(device))
    assert isinstance(out, TVMOutput)
    terms = out.terminal_loss + out.flow_matching_loss
    assert _max_abs(out.loss.detach() - terms) <= 1e-5 * max(1.0, _max_abs(terms))
    optimizer = lucid.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()
    model.update_ema()

    eval_decay = model.config.eval_ema_decay
    assert eval_decay is not None
    live = dict(model.tvm.named_parameters())
    averages = (
        ("target", model.target_network, model.config.target_ema_decay),
        ("eval", model.ema_network, eval_decay),
    )
    for key, network, decay in averages:
        moved = False
        for name, slow in network.named_parameters():
            expected = decay * initial[name] + (1.0 - decay) * live[name].detach()
            assert _max_abs(slow - expected) <= 1e-6, f"{key}: {name}"
            moved = moved or _max_abs(slow - initial[name]) > 0.0
        assert moved, f"the {key} average did not move"


@pytest.mark.parametrize("device", DEVICES)
def test_tvm_samples_round_trip_through_the_vae_latent_whitening(device: str) -> None:
    """The sampler's output is whitened latent; ``to_vae_latent`` maps it to
    the raw VAE latent with the config's per-channel statistics, and
    ``from_vae_latent`` — what training images go through — inverts it."""
    lucid.manual_seed(0)
    model = _build("tvm_imagenet256_gen", device).eval()
    assert isinstance(model, TVMForImageGeneration)
    samples = model.generate(2, labels=_labels(device)).samples

    latent = model.to_vae_latent(samples)
    mean = lucid.tensor(list(model.config.latent_mean), device=device)
    std = lucid.tensor(list(model.config.latent_std), device=device)
    expected = samples * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
    assert _max_abs(latent - expected) <= 1e-5 * max(1.0, _max_abs(expected))
    assert _max_abs(model.from_vae_latent(latent) - samples) <= 1e-5 * max(1.0, _max_abs(samples))


def _tvm_averages(model: TVMForImageGeneration) -> list[tuple[str, lucid.nn.Module]]:
    return [("target", model.target_network), ("eval", model.ema_network)]


@pytest.mark.parametrize("device", DEVICES)
def test_tvm_averages_pair_by_name_after_the_live_order_changes(device: str) -> None:
    """LCD-552: ``update_ema`` pairs each average's parameter with the live
    one of the same name.  Deleting a child and assigning it back moves it
    to the end of its registry, so the live network's parameter order then
    differs from the averages' — and the time and interval embedders have
    the same shapes, so a positional pairing would average one into the
    other without an error."""
    lucid.manual_seed(0)
    model = _build("tvm_imagenet256_gen", device).train()
    assert isinstance(model, TVMForImageGeneration)
    built_order = [name for name, _ in model.target_network.named_parameters()]
    embedder = model.tvm.time_embed
    del model.tvm.time_embed
    model.tvm.time_embed = embedder
    live_order = [name for name, _ in model.tvm.named_parameters()]
    assert live_order != built_order and sorted(live_order) == sorted(built_order)

    initial = {n: p.detach().clone() for n, p in model.tvm.named_parameters()}
    with lucid.no_grad():
        for param in model.tvm.parameters():
            param.add_(lucid.randn(tuple(param.shape), device=device))
    model.update_ema()

    live = dict(model.tvm.named_parameters())
    decays = {"target": model.config.target_ema_decay}
    decays["eval"] = model.config.eval_ema_decay or 0.0
    for key, network in _tvm_averages(model):
        for name, slow in network.named_parameters():
            mixed = decays[key] * initial[name] + (1.0 - decays[key]) * live[name]
            assert _max_abs(slow - mixed.detach()) <= 1e-5, f"{key}: {name}"


@pytest.mark.parametrize("through", ["network", "model"])
def test_tvm_averages_restart_from_weights_loaded_into_the_network(through: str) -> None:
    """LCD-552: loading weights into the live network — on ``model.tvm`` or
    on the model — restarts both averages from them.  They used to stay at
    whatever the network held before, so a fine-tune from a checkpoint read
    its first targets off the random initialisation.  An average's own
    checkpoint, loaded after the network's, is kept."""
    lucid.manual_seed(0)
    model = _build("tvm_imagenet256_gen", "cpu").train()
    assert isinstance(model, TVMForImageGeneration)
    stale = [n for n, _ in _tvm_averages(model)]
    assert stale == ["target", "eval"]
    lucid.manual_seed(1)
    donor = _build("tvm_imagenet256_gen", "cpu")
    assert isinstance(donor, TVMForImageGeneration)
    if through == "network":
        model.tvm.load_state_dict(donor.tvm.state_dict())
    else:
        model.load_state_dict(donor.state_dict())

    loaded = dict(donor.tvm.named_parameters())
    for key, network in _tvm_averages(model):
        for name, slow in network.named_parameters():
            assert _max_abs(slow - loaded[name]) == 0.0, f"{key}: {name}"

    resumed = {k: lucid.randn(tuple(v.shape)) for k, v in loaded.items()}
    model.ema_network.load_state_dict(resumed)
    for name, slow in model.ema_network.named_parameters():
        assert _max_abs(slow - resumed[name]) == 0.0, f"eval: {name}"


def test_tvm_network_copies_and_pickles_without_the_wrapper() -> None:
    """LCD-552: the network notices loads by counting them, not by holding a
    callback into the wrapper.  A hook bound to the wrapper made ``model.tvm``
    hold it, so ``copy.deepcopy(model.tvm)`` cloned the wrapper and both
    averages too (5 GB at the paper's size) and ``pickle.dumps(model.tvm)``
    was three times the size of a network built on its own."""
    import copy
    import pickle

    lucid.manual_seed(0)
    model = _build("tvm_imagenet256_gen", "cpu")
    assert isinstance(model, TVMForImageGeneration)
    averages = [network for _, network in _tvm_averages(model)]

    memo: dict[int, object] = {}
    copied = copy.deepcopy(model.tvm, memo)
    assert id(model) not in memo, "deep-copying the network copied the wrapper"
    assert not any(id(a) in memo for a in averages), "and the averages with it"
    lone = M.create_model("tvm_imagenet256", **_TRANSFORMER)
    assert len(pickle.dumps(model.tvm)) <= 1.05 * len(pickle.dumps(lone))

    # The copy still counts its own loads, and the wrapper still sees its own.
    copied.load_state_dict(copied.state_dict())
    loaded = {n: lucid.randn(tuple(p.shape)) for n, p in model.tvm.state_dict().items()}
    model.tvm.load_state_dict(loaded)
    for key, network in _tvm_averages(model):
        for name, slow in network.named_parameters():
            assert _max_abs(slow - loaded[name]) == 0.0, f"{key}: {name}"


def test_tvm_averages_copy_the_network_without_reinitialising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LCD-552: an average holds the live weights from the start, so building
    one does not rerun the Lipschitz initialisation (an SVD per linear layer)
    or draw from the generator — it used to build a whole ``TVMModel``."""
    import lucid.models.generative.tvm._model as tvm_model

    lucid.manual_seed(0)
    model = _build("tvm_imagenet256_gen", "cpu")
    assert isinstance(model, TVMForImageGeneration)

    def _refuse(weight: lucid.Tensor) -> None:
        raise AssertionError("building an average reran the spectral init")

    monkeypatch.setattr(tvm_model, "_spectral_", _refuse)
    lucid.manual_seed(5)
    untouched = lucid.rand(8)
    lucid.manual_seed(5)
    averages = _tvm_averages(model)
    assert _max_abs(lucid.rand(8) - untouched) == 0.0, "building drew numbers"

    live = dict(model.tvm.named_parameters())
    for key, network in averages:
        assert isinstance(network, TVMModel) and not network.training
        for name, slow in network.named_parameters():
            assert not slow.requires_grad, f"{key}: {name} is trainable"
            assert _max_abs(slow - live[name].detach()) == 0.0, f"{key}: {name}"
        assert _max_abs(network.pos_embed - model.tvm.pos_embed) == 0.0


# ── generation from the public factory ─────────────────────────────────────


def _generate(name: str, model: lucid.nn.Module, device: str) -> M.GenerationOutput:
    if name == "stable_diffusion_gen":
        context = lucid.randn((2, 8, 16), device=device)
        return model.generate(context, context, num_inference_steps=2)
    if name == "dit_small_2_gen":
        return model.generate(2, labels=_labels(device), steps=2)
    return model.generate(2, labels=_labels(device))


_SAMPLE_SHAPES = {
    "dit_small_2_gen": (2, 4, 8, 8),
    "mean_flow_base_2_gen": (2, 4, 8, 8),
    "stable_diffusion_gen": (2, 3, 32, 32),
    "tvm_imagenet256_gen": (2, 4, 8, 8),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(_SAMPLE_SHAPES))
def test_generation_returns_finite_samples_on_the_device(name: str, device: str) -> None:
    lucid.manual_seed(0)
    model = _build(name, device).eval()
    out = _generate(name, model, device)
    assert isinstance(out, M.GenerationOutput)
    assert out.samples.shape == _SAMPLE_SHAPES[name]
    assert str(out.samples.device) == f"device('{device}')"
    assert bool(out.samples.isfinite().all().item())


# ── host syncs per sampling loop (Metal) ───────────────────────────────────


def _syncs(run: Callable[[], M.GenerationOutput]) -> int:
    run()  # first call compiles kernels and fills caches
    before = _C_engine.host_sync_count()
    run().samples.eval()
    return _C_engine.host_sync_count() - before


def _ddpm() -> DDPMForImageGeneration:
    config = DDPMConfig(
        sample_size=8,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        resnet_groups=4,
        num_train_timesteps=20,
        attention_resolutions=(),
    )
    return DDPMForImageGeneration(config).to("metal").eval()


def _mixin_loop(steps: int) -> Callable[[], M.GenerationOutput]:
    model = _ddpm()
    scheduler = model.make_scheduler()
    return lambda: model.generate(scheduler, n_samples=1, num_inference_steps=steps)


def _sd_loop(steps: int) -> Callable[[], M.GenerationOutput]:
    model = _build("stable_diffusion_gen", "metal").eval()
    context = lucid.randn((1, 8, 16), device="metal")
    return lambda: model.generate(context, context, num_inference_steps=steps)


def _labelled_loop(name: str) -> Callable[[int], Callable[[], M.GenerationOutput]]:
    def make(steps: int) -> Callable[[], M.GenerationOutput]:
        model = _build(name, "metal").eval()
        return lambda: model.generate(1, labels=_labels("metal")[:1], steps=steps)

    return make


# dit, mean_flow and tvm embed their class labels once per denoising step;
# nn.Embedding's index check no longer reads the indices back (LCD-283).
_LOOPS = [
    pytest.param(_mixin_loop, id="diffusion_mixin"),
    pytest.param(_sd_loop, id="stable_diffusion"),
    pytest.param(_labelled_loop("dit_small_2_gen"), id="dit"),
    pytest.param(_labelled_loop("mean_flow_base_2_gen"), id="mean_flow"),
    pytest.param(_labelled_loop("tvm_imagenet256_gen"), id="tvm"),
]


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("loop", _LOOPS)
def test_a_sampling_loop_syncs_a_bounded_number_of_times(
    loop: Callable[[int], Callable[[], M.GenerationOutput]],
) -> None:
    """Timesteps live on the schedule's device (LCD-383); a loop that read
    one back per step would sync once per step."""
    short, long = _syncs(loop(2)), _syncs(loop(10))
    assert long == short <= 1, f"{short} syncs at 2 steps, {long} at 10"


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_the_sync_count_sees_a_read() -> None:
    """Guards the check above: a read on Metal is counted."""
    before = _C_engine.host_sync_count()
    lucid.ones(3, device="metal").sum().item()
    assert _C_engine.host_sync_count() > before
