"""Diffusion schedules are gathered on the device, not read on the host (LCD-372).

``extract_into_tensor`` read every timestep with ``.item()`` and indexed
the schedule in Python, and Stable Diffusion's training forward noised one
sample at a time from ``int(timestep[i].item())``.  A compiled or exported
training step replays a host-read value from the traced call, so
``lucid.compile`` ran these steps eagerly.  The schedule tables also stayed
on the CPU when the model moved, so DDPM's hybrid objective raised
``DeviceMismatch`` on Metal.

The old selection is written out inline here and compared bit for bit; the
compile checks prove no host read is left in a training forward.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.compile import _tracing
from lucid.models._utils._generative import extract_into_tensor
from lucid.models.generative import DDPMScheduler
from lucid.models.generative.stable_diffusion import (
    DDIMScheduler,
    StableDiffusionConfig,
    StableDiffusionModel,
)
from lucid.test._fixtures.devices import metal_available

DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]

STEPS = 100

DDPM_TINY = {
    "sample_size": 16,
    "base_channels": 32,
    "channel_mult": (1, 2),
    "num_res_blocks": 1,
    "resnet_groups": 16,
    "num_train_timesteps": STEPS,
    "attention_resolutions": (),
}

SD_TINY = {
    "sample_size": 32,
    "downsample_factor": 4,
    "vae_block_out_channels": (32, 64, 64),
    "unet_block_out_channels": (32, 64),
    "attention_head_dim": 32,
    "cross_attention_dim": 16,
    "context_length": 4,
}


def _old_extract(arr: Tensor, timesteps: Tensor, shape: tuple[int, ...]) -> Tensor:
    """The lookup ``extract_into_tensor`` made before LCD-372."""
    vals = [
        float(arr[int(timesteps[b].item())].item())
        for b in range(int(timesteps.shape[0]))
    ]
    out = lucid.tensor(vals, device=arr.device.type)
    while out.ndim < len(shape):
        out = out.unsqueeze(-1)
    return out


def _old_sd_noise(
    scheduler: DDIMScheduler, latent: Tensor, noise: Tensor, timestep: Tensor
) -> Tensor:
    """Stable Diffusion's per-sample noising before LCD-372."""
    rows = []
    for i in range(int(latent.shape[0])):
        alpha = float(scheduler.alphas_cumprod[int(timestep[i].item())].item())
        rows.append(alpha**0.5 * latent[i] + (1.0 - alpha) ** 0.5 * noise[i])
    return lucid.stack(rows, dim=0)


def _ddpm(device: str, *, learn_sigma: bool = True) -> nn.Module:
    lucid.manual_seed(0)
    width = 6 if learn_sigma else 3
    model = M.create_model(
        "ddpm_cifar_gen", learn_sigma=learn_sigma, out_channels=width, **DDPM_TINY
    )
    return model.to(device)


def _sd(device: str) -> StableDiffusionModel:
    lucid.manual_seed(0)
    return StableDiffusionModel(StableDiffusionConfig(**SD_TINY)).to(device)


# ── extract_into_tensor ──────────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 2, 4])
def test_extract_matches_the_host_lookup(device: str, rank: int) -> None:
    lucid.manual_seed(1)
    table = lucid.rand((STEPS,)).to(device)
    steps = lucid.tensor([0, STEPS - 1, 37, -1, -STEPS, 37]).long().to(device)
    shape = (int(steps.shape[0]), 3, 5, 7)[:rank]
    got = extract_into_tensor(table, steps, shape)
    want = _old_extract(table, steps, shape)
    assert got.shape == want.shape == (shape[0], *([1] * (rank - 1)))
    assert got.dtype == want.dtype
    assert got.tolist() == want.tolist()


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_extract_follows_the_timesteps_device() -> None:
    table = lucid.rand((STEPS,))
    steps = lucid.tensor([3, 90]).long().to("metal")
    got = extract_into_tensor(table, steps, (2, 1, 1, 1))
    assert got.device.type == "metal"
    assert got.reshape(-1).tolist() == [table[3].item(), table[90].item()]


def test_extract_refuses_out_of_range_on_cpu() -> None:
    with pytest.raises(IndexError):
        extract_into_tensor(lucid.rand((STEPS,)), lucid.tensor([STEPS]), (1, 1))


# ── Stable Diffusion: add_noise and the training forward ─────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_sd_add_noise_matches_per_sample_noising(device: str) -> None:
    scheduler = DDIMScheduler(StableDiffusionConfig())
    lucid.manual_seed(2)
    latent = lucid.randn((5, 4, 8, 8)).to(device)
    noise = lucid.randn((5, 4, 8, 8)).to(device)
    steps = lucid.tensor([0, 1, 499, 998, 999]).long().to(device)
    scheduler._apply(lambda t: t.to(device))
    got = scheduler.add_noise(latent, noise, steps)
    assert got.tolist() == _old_sd_noise(scheduler, latent, noise, steps).tolist()
    # The int form, which samplers use, is the same coefficients.
    one = scheduler.add_noise(latent[2:3], noise[2:3], 499)
    assert one.tolist() == got[2:3].tolist()


def test_sd_add_noise_refuses_an_int_outside_the_schedule() -> None:
    scheduler = DDIMScheduler(StableDiffusionConfig())
    zeros = lucid.zeros((1, 4, 8, 8))
    with pytest.raises(IndexError, match=r"\[-1000, 1000\)"):
        scheduler.add_noise(zeros, zeros, 1000)


@pytest.mark.parametrize("device", DEVICES)
def test_sd_forward_noises_as_before(device: str) -> None:
    model = _sd(device).eval()
    lucid.manual_seed(3)
    images = lucid.randn((2, 3, 32, 32)).to(device)
    context = lucid.randn((2, 4, 16)).to(device)
    steps = lucid.tensor([3, 870]).long().to(device)

    lucid.manual_seed(4)
    out = model(images, context, steps, return_loss=True)

    # Same draws in the same order: the posterior sample, then the noise.
    lucid.manual_seed(4)
    latent = model.encode_image(images)
    shape = tuple(int(s) for s in latent.shape)
    noise = lucid.randn(shape, device=latent.device.type, dtype=latent.dtype)
    want = _old_sd_noise(model.scheduler, latent, noise, steps)
    assert out.latent.tolist() == want.tolist()


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_sd_noising_tables_move_with_the_model() -> None:
    model = _sd("metal")
    assert model.scheduler._signal_scale.device.type == "metal"
    assert model.scheduler._noise_scale.device.type == "metal"
    model.to("cpu")
    assert model.scheduler._signal_scale.device.type == "cpu"


# ── DDPM: the hybrid objective's tables ──────────────────────────────────────


def test_ddpm_schedule_buffers_stay_out_of_the_checkpoint() -> None:
    plain = set(_ddpm("cpu", learn_sigma=False).state_dict())
    hybrid = _ddpm("cpu", learn_sigma=True)
    keys = set(hybrid.state_dict())
    assert not any(k.startswith("_") for k in keys)
    # Only the wider output conv differs; no schedule constant is a key.
    assert keys == plain


@pytest.mark.parametrize("device", DEVICES)
def test_ddpm_hybrid_training_step(device: str) -> None:
    model = _ddpm(device)
    scheduler = DDPMScheduler(STEPS)
    lucid.manual_seed(5)
    x0 = lucid.randn((2, 3, 16, 16)).to(device)
    eps = lucid.randn((2, 3, 16, 16)).to(device)
    steps = lucid.tensor([0, 61]).long().to(device)
    x_t = scheduler.add_noise(x0, eps, steps)
    assert x_t.device.type == device

    opt = lucid.optim.SGD(list(model.parameters()), lr=1e-3)
    out = model(x_t, steps, target=eps, x_start=x0)
    opt.zero_grad()
    out.loss.backward()
    opt.step()
    assert out.loss_vlb is not None
    assert bool(lucid.isfinite(out.loss).item())


def _ddpm_scheduler() -> DDPMScheduler:
    return DDPMScheduler(STEPS)


def _ddim_scheduler() -> DDIMScheduler:
    return DDIMScheduler(StableDiffusionConfig(num_train_timesteps=STEPS))


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize(
    "build", [_ddpm_scheduler, _ddim_scheduler], ids=["ddpm", "ddim"]
)
def test_add_noise_traces_the_same_cold_or_warm(
    build: Callable[[], DDPMScheduler | DDIMScheduler],
) -> None:
    """A CPU scheduler noising Metal samples records one graph (LCD-627).

    ``add_noise`` moves the tables to the sample's device on first use and
    keeps them there.  Inside a trace the first call recorded that move — a
    copy of every table — and a later one fed the moved tables, so a
    recorded training step depended on what the scheduler had served.  A
    standalone DDIM scheduler moved its tables only through a model's
    ``_apply``, so it copied them on every call (LCD-639).
    """
    scheduler = build()
    lucid.manual_seed(6)
    x0 = lucid.randn((2, 3, 4, 4)).to("metal")
    eps = lucid.randn((2, 3, 4, 4)).to("metal")
    steps = lucid.tensor([0, 61]).long().to("metal")

    def recorded() -> tuple[list[str], Tensor]:
        with _tracing(watch=[x0, eps, steps]) as tracer, lucid.no_grad():
            noised = scheduler.add_noise(x0, eps, steps)
        return [op.name for op in tracer.graph.ops], noised

    (cold, first), (warm, second) = recorded(), recorded()
    assert cold == warm
    assert "to_device" not in cold
    assert first.device.type == "metal"
    assert first.tolist() == second.tolist()


# ── compile: no host read left in a training forward ─────────────────────────


class _DDPMLoss(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x_t: Tensor, eps: Tensor, t: Tensor, x0: Tensor) -> Tensor:
        return self.model(x_t, t, target=eps, x_start=x0).loss


class _SDLoss(nn.Module):
    def __init__(self, model: StableDiffusionModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: Tensor, context: Tensor, t: Tensor) -> Tensor:
        loss = self.model(images, context, t, return_loss=True).loss
        assert loss is not None
        return loss


def _ddpm_batch(seed: int, steps: list[int]) -> tuple[Tensor, ...]:
    lucid.manual_seed(seed)
    x0 = lucid.randn((2, 3, 16, 16)).to("metal")
    eps = lucid.randn((2, 3, 16, 16)).to("metal")
    t = lucid.tensor(steps).long().to("metal")
    return DDPMScheduler(STEPS).add_noise(x0, eps, t), eps, t, x0


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_ddpm_hybrid_loss_compiles() -> None:
    model = _DDPMLoss(_ddpm("metal").eval())
    compiled = lucid.compile.compile(model)
    # Under grad mode a trainable model's compiled call runs eagerly (LCD-582).
    with lucid.no_grad():
        compiled(*_ddpm_batch(1, [0, 61]))
        fresh = _ddpm_batch(2, [99, 7])
        got = compiled(*fresh)
    info = compiled.cache_info()
    assert not info["eager_only"], info
    assert info["entries"] == 1, info
    want = model(*fresh)
    assert abs(float(got.item()) - float(want.item())) <= 1e-5


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_sd_training_loss_compiles() -> None:
    compiled = lucid.compile.compile(_SDLoss(_sd("metal")))
    lucid.manual_seed(6)
    images = lucid.randn((2, 3, 32, 32)).to("metal")
    context = lucid.randn((2, 4, 16)).to("metal")
    with lucid.no_grad():
        compiled(images, context, lucid.tensor([3, 870]).long().to("metal"))
        got = compiled(images, context, lucid.tensor([500, 17]).long().to("metal"))
    info = compiled.cache_info()
    assert not info["eager_only"], info
    assert info["entries"] == 1, info
    assert bool(lucid.isfinite(got).item())
