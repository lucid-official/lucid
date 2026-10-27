"""Unit tests for Rectified Flow (Liu, Gong & Liu, 2023) — straight paths.

Most of this family is shapes and plumbing, which tests catch by
construction.  Three things are not, and they are what the file is
weighted towards:

* **the objective is the straight line** — checked against Flow
  Matching's optimal-transport path at ``sigma_min = 0``, which it must
  equal exactly rather than approximately;
* **the resampling filter** — a transcription of a published algorithm,
  checked against an independent transcription of the same algorithm
  rather than against a property it happens to satisfy;
* **straightness is a real instrument** — a constant field must measure
  zero and a random one must not, or every later reflow number is
  vacuous.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.models import (
    DiffusionModelOutput,
    GenerationOutput,
    RectifiedFlowConfig,
    RectifiedFlowForImageGeneration,
    RectifiedFlowModel,
    create_model,
    is_model,
)
from lucid.models.generative.flow_matching import FlowMatchingConfig, FlowMatchingModel
from lucid.models.generative.rectified_flow._model import (
    _TIME_SCALE,
    _FIRResample,
    _VelocityField,
)
from lucid.test._fixtures.devices import metal_available


def _cfg(**overrides: object) -> RectifiedFlowConfig:
    """Small enough to solve repeatedly, wide enough to be a real U-Net."""
    base: dict[str, object] = {
        "sample_size": 8,
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 16,
        "channel_mult": (1, 2),
        "num_res_blocks": 1,
        "attention_resolutions": (4,),
        "resnet_groups": 8,
        "rtol": 1e-5,
        "atol": 1e-5,
    }
    base.update(overrides)
    return RectifiedFlowConfig(**base)  # type: ignore[arg-type]


def _worst(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float(np.abs(a.numpy() - b.numpy()).max())


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


def test_config_defaults_match_the_released_cifar_configuration() -> None:
    """The authors published their configs; these are copied, not inferred."""
    cfg = RectifiedFlowConfig()
    assert cfg.base_channels == 128
    assert cfg.channel_mult == (1, 2, 2, 2)
    assert cfg.num_res_blocks == 4
    assert cfg.attention_resolutions == (16,)
    assert cfg.dropout == 0.15
    assert cfg.fir is False
    assert cfg.progressive == "none"
    assert cfg.embedding_type == "positional"
    assert cfg.t_schedule == "uniform"
    assert cfg.time_eps == 1e-3


def test_data_dim() -> None:
    assert RectifiedFlowConfig(sample_size=32, in_channels=3).data_dim == 3072
    assert RectifiedFlowConfig(sample_size=(4, 8), in_channels=1, out_channels=1).data_dim == 32


@pytest.mark.parametrize(
    "schedule,expected",
    [("uniform", False), ("t0", True), ("t1", True), (4, True)],
)
def test_is_distillation(schedule: object, expected: bool) -> None:
    assert RectifiedFlowConfig(t_schedule=schedule).is_distillation is expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 3, "out_channels": 4},
        {"base_channels": 0},
        {"channel_mult": ()},
        {"num_res_blocks": 0},
        {"dropout": 1.0},
        {"fir_kernel": (1, 0, 1)},
        {"t_schedule": "sometimes"},
        {"t_schedule": 1},
        {"t_schedule": True},
        {"time_eps": 1.0},
        {"rtol": 0.0},
    ],
)
def test_config_rejects_bad_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RectifiedFlowConfig(**kwargs)  # type: ignore[arg-type]


def test_integer_one_is_refused_rather_than_aliased() -> None:
    """``k = 1`` and ``"t0"`` would mean the same thing; only one spelling."""
    with pytest.raises(ValueError, match="use 't0'"):
        RectifiedFlowConfig(t_schedule=1)


# ─────────────────────────────────────────────────────────────────────────────
# The path — exact, not approximate
# ─────────────────────────────────────────────────────────────────────────────


def test_path_hits_both_endpoints_exactly() -> None:
    model = RectifiedFlowModel(_cfg()).eval()
    x1 = lucid.randn((4, 3, 8, 8))
    x0 = lucid.randn((4, 3, 8, 8))
    assert _worst(model.path_sample(x1, x0, lucid.zeros((4,))), x0) == 0.0
    assert _worst(model.path_sample(x1, x0, lucid.ones((4,))), x1) == 0.0


def test_target_is_the_chord_and_does_not_depend_on_time() -> None:
    """A straight line at constant speed has one velocity, at every ``t``."""
    model = RectifiedFlowModel(_cfg()).eval()
    x1 = lucid.randn((4, 3, 8, 8))
    x0 = lucid.randn((4, 3, 8, 8))
    early = model.conditional_target(x1, x0, lucid.tensor([0.05, 0.2, 0.5, 0.8]))
    late = model.conditional_target(x1, x0, lucid.tensor([0.9, 0.9, 0.9, 0.9]))
    assert _worst(early, x1 - x0) == 0.0
    assert _worst(early, late) == 0.0


def test_is_flow_matching_optimal_transport_at_sigma_min_zero() -> None:
    """The lineage claim, and it is an identity rather than a resemblance.

    Rectified Flow's forward process *is* the previous family's
    optimal-transport path with the terminal width taken to zero.  If this
    ever stops holding exactly, one of the two derivations has drifted.
    """
    rf = RectifiedFlowModel(_cfg()).eval()
    fm = FlowMatchingModel(
        FlowMatchingConfig(
            sample_size=8,
            base_channels=16,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            resnet_groups=8,
            path="ot",
            sigma_min=0.0,
        )
    ).eval()
    x1 = lucid.randn((4, 3, 8, 8))
    x0 = lucid.randn((4, 3, 8, 8))
    t = lucid.tensor([0.0, 0.25, 0.5, 1.0])
    assert _worst(rf.path_sample(x1, x0, t), fm.path_sample(x1, x0, t)) == 0.0
    assert _worst(rf.conditional_target(x1, x0, t), fm.conditional_target(x1, x0, t)) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Time schedules
# ─────────────────────────────────────────────────────────────────────────────


def test_uniform_schedule_covers_the_interval() -> None:
    model = RectifiedFlowModel(_cfg()).eval()
    t = model.sample_times(4096).numpy()
    assert t.min() >= model.config.time_eps
    assert t.max() <= 1.0
    assert 0.4 < float(t.mean()) < 0.6


@pytest.mark.parametrize("schedule,value", [("t0", 1e-3), ("t1", 1.0)])
def test_pinned_schedules_are_constant(schedule: str, value: float) -> None:
    model = RectifiedFlowModel(_cfg(t_schedule=schedule)).eval()
    t = model.sample_times(64).numpy()
    assert np.allclose(t, value)


def test_integer_schedule_lands_on_the_euler_grid() -> None:
    """k-step distillation must draw exactly the times a k-step sampler visits."""
    steps = 4
    model = RectifiedFlowModel(_cfg(t_schedule=steps)).eval()
    t = model.sample_times(4096).numpy()
    eps = model.config.time_eps
    grid = np.array([index * (1.0 - eps) / steps + eps for index in range(steps)])
    assert np.abs(t[:, None] - grid[None, :]).min(axis=1).max() < 1e-6
    # And every point of the grid is actually reachable.
    assert len(np.unique(np.round(t, 6))) == steps


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────


def test_loss_solves_nothing() -> None:
    """Simulation-free training is the premise the whole lineage rests on."""
    model = RectifiedFlowModel(_cfg()).eval()
    loss, prediction, target = model.rectified_flow_loss(lucid.randn((2, 3, 8, 8)))
    assert loss.shape == ()
    assert prediction.shape == (2, 3, 8, 8)
    assert target.shape == (2, 3, 8, 8)
    assert model.nfe == 0


def test_loss_reaches_every_parameter() -> None:
    model = RectifiedFlowModel(_cfg())
    model.train()
    loss, _, _ = model.rectified_flow_loss(lucid.randn((2, 3, 8, 8)))
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_paired_noise_is_the_reflow_objective() -> None:
    """Reflow is not a second loss — it is this one with the pairing supplied."""
    model = RectifiedFlowModel(_cfg()).eval()
    z1 = lucid.randn((2, 3, 8, 8))
    z0 = lucid.randn((2, 3, 8, 8))
    loss, _, target = model.rectified_flow_loss(z1, z0)
    assert loss.shape == ()
    assert _worst(target, z1 - z0) == 0.0


def test_distilling_model_refuses_unpaired_data() -> None:
    """Silently training against independent draws would look fine and not be.

    A pinned ``t`` regressed against an unrelated ``x0`` teaches the field
    the mean of the data, and every shape and loss value would still look
    reasonable.  Refusing is the only way this surfaces.
    """
    model = RectifiedFlowModel(_cfg(t_schedule="t0")).eval()
    with pytest.raises(ValueError, match="pairs"):
        model.rectified_flow_loss(lucid.randn((2, 3, 8, 8)))
    model.rectified_flow_loss(lucid.randn((2, 3, 8, 8)), lucid.randn((2, 3, 8, 8)))


def test_mismatched_pairing_is_refused() -> None:
    model = RectifiedFlowModel(_cfg()).eval()
    with pytest.raises(ValueError):
        model.rectified_flow_loss(lucid.randn((2, 3, 8, 8)), lucid.randn((3, 3, 8, 8)))


# ─────────────────────────────────────────────────────────────────────────────
# Straightness — the instrument the paper's claim is read off
# ─────────────────────────────────────────────────────────────────────────────


def _constant_field(value: float) -> RectifiedFlowModel:
    """A model whose velocity is the same everywhere, so every path is straight."""
    model = RectifiedFlowModel(_cfg(init_scale=1.0)).eval()
    for _, p in model.named_parameters():
        nn.init.zeros_(p)
    nn.init.constant_(model.field.conv_out.bias, value)
    return model


def test_straightness_is_zero_for_a_straight_flow() -> None:
    model = _constant_field(0.7)
    noise = lucid.randn((4, 3, 8, 8))
    assert float(model.straightness(noise, steps=16)) < 1e-8


def test_straightness_is_positive_for_a_curved_one() -> None:
    """Guards the instrument: without this the test above proves nothing."""
    model = RectifiedFlowModel(_cfg(init_scale=1.0)).eval()
    noise = lucid.randn((4, 3, 8, 8))
    assert float(model.straightness(noise, steps=16)) > 1e-4


def test_one_step_is_exact_when_the_flow_is_straight() -> None:
    """The method's entire payoff, stated as an equality."""
    model = _constant_field(0.7)
    noise = lucid.randn((4, 3, 8, 8))
    one = model.one_step(noise)
    many = model.sample(noise=noise, steps=64)
    assert _worst(one, many) < 1e-4
    # And the displacement is the field itself, integrated over unit time.
    assert abs(float((many - noise).mean()) - 0.7) < 1e-4


def test_straightness_rejects_a_bad_budget() -> None:
    model = RectifiedFlowModel(_cfg()).eval()
    with pytest.raises(ValueError):
        model.straightness(steps=0)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling and reflow
# ─────────────────────────────────────────────────────────────────────────────


def test_fixed_budget_costs_exactly_that_many_evaluations() -> None:
    """Euler, not Runge-Kutta: one evaluation per step is the point."""
    model = RectifiedFlowModel(_cfg()).eval()
    model.sample(n_samples=2, steps=7)
    assert model.nfe == 7
    model.one_step(lucid.randn((2, 3, 8, 8)))
    assert model.nfe == 1


def test_adaptive_solve_runs_and_costs_more_than_nothing() -> None:
    model = RectifiedFlowModel(_cfg(init_scale=1.0)).eval()
    out = model.sample(n_samples=2)
    assert out.shape == (2, 3, 8, 8)
    assert model.nfe > 0


def test_sample_accepts_a_fixed_starting_point() -> None:
    model = RectifiedFlowModel(_cfg(init_scale=1.0)).eval()
    noise = lucid.randn((2, 3, 8, 8))
    assert _worst(model.sample(noise=noise, steps=4), model.sample(noise=noise, steps=4)) == 0.0


def test_reflow_pairs_returns_the_coupling_it_solved() -> None:
    """``z0`` must be the noise that produced ``z1``, not a fresh draw."""
    model = RectifiedFlowModel(_cfg(init_scale=1.0)).eval()
    noise = lucid.randn((3, 3, 8, 8))
    z0, z1 = model.reflow_pairs(noise=noise, steps=4)
    assert _worst(z0, noise) == 0.0
    assert _worst(z1, model.sample(noise=noise, steps=4)) == 0.0


def test_reflow_pairs_feed_straight_back_into_the_objective() -> None:
    model = RectifiedFlowModel(_cfg(init_scale=1.0)).eval()
    z0, z1 = model.reflow_pairs(n_samples=3, steps=4)
    loss, _, _ = model.rectified_flow_loss(z1, z0)
    assert loss.shape == ()


def test_source_need_not_be_gaussian() -> None:
    """The paper's other half: transporting between two domains.

    Nothing in the objective, the sampler or the reflow loop assumes the
    source is a prior — passing ``noise=`` is the whole mechanism, so
    domain-to-domain transfer is the same three calls.  Pinned because
    the class docstring claims it.
    """
    model = RectifiedFlowModel(_cfg(init_scale=1.0)).eval()
    source = lucid.randn((4, 3, 8, 8)) * 0.3 - 2.0
    target = lucid.randn((4, 3, 8, 8)) * 0.3 + 2.0

    loss, _, chord = model.rectified_flow_loss(target, source)
    assert loss.shape == ()
    assert _worst(chord, target - source) == 0.0

    z0, z1 = model.reflow_pairs(noise=source, steps=4)
    assert _worst(z0, source) == 0.0
    assert _worst(z1, model.sample(noise=source, steps=4)) == 0.0


@pytest.mark.parametrize("bad", [{"n_samples": 0}, {"steps": 0}])
def test_sample_rejects_bad_budgets(bad: dict[str, int]) -> None:
    model = RectifiedFlowModel(_cfg()).eval()
    with pytest.raises(ValueError):
        model.sample(**bad)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

# Built through the registered factories, since ``create_model(...)`` then
# ``.to("metal")`` is the path a user takes.
_SMALL: dict[str, object] = {
    "sample_size": 8,
    "base_channels": 16,
    "channel_mult": (1, 2),
    "num_res_blocks": 1,
    "attention_resolutions": (),
    "resnet_groups": 8,
    "init_scale": 1.0,
}
_METAL = "device('metal')"
_needs_metal = pytest.mark.skipif(not metal_available(), reason="metal unavailable")


def _on_metal(name: str) -> object:
    return create_model(name, **_SMALL).eval().to("metal")


@_needs_metal
def test_every_draw_follows_the_model_to_metal() -> None:
    """A model moved to the accelerator samples there without being told.

    Every default used to be ``"cpu"`` — and ``straightness`` passed no
    device at all — so ``.to("metal")`` followed by a default call drew its
    noise on the CPU and the first field evaluation raised a device
    mismatch.  Each entry point draws separately, so each is exercised.
    """
    model = _on_metal("rectified_flow_cifar")
    assert isinstance(model, RectifiedFlowModel)
    assert str(model.sample(n_samples=2, steps=2).device) == _METAL
    z0, z1 = model.reflow_pairs(n_samples=2, steps=2)
    assert (str(z0.device), str(z1.device)) == (_METAL, _METAL)
    assert str(model.straightness(n_samples=2, steps=2).device) == _METAL
    assert str(model.sample_times(3).device) == _METAL


@_needs_metal
def test_the_generator_follows_the_model_to_metal() -> None:
    model = _on_metal("rectified_flow_cifar_gen")
    assert isinstance(model, RectifiedFlowForImageGeneration)
    assert str(model.generate(n_samples=2, steps=1).samples.device) == _METAL
    z0, z1 = model.reflow_pairs(n_samples=2, steps=2)
    assert (str(z0.device), str(z1.device)) == (_METAL, _METAL)


@_needs_metal
def test_an_explicit_device_still_wins() -> None:
    """Following the model is the default, not an override of the caller."""
    model = _on_metal("rectified_flow_cifar")
    assert isinstance(model, RectifiedFlowModel)
    assert str(model.sample_times(3, device="cpu").device) == "device('cpu')"
    assert str(model.sample(n_samples=1, steps=1, device="metal").device) == _METAL


# ─────────────────────────────────────────────────────────────────────────────
# Filtered resampling
# ─────────────────────────────────────────────────────────────────────────────


def _reference_upfirdn2d(
    x: np.ndarray, taps: tuple[int, ...], up: int, down: int, pad: tuple[int, int]
) -> np.ndarray:
    """Independent transcription of the published resampling algorithm.

    Zero-insert, pad, convolve with the flipped kernel, decimate.  Written
    straight from the algorithm rather than from the module under test, so
    agreement means two routes reached the same numbers.
    """
    n, c, h, w = x.shape
    k = np.outer(np.asarray(taps, float), np.asarray(taps, float))
    k = k / k.sum() * (up * up)
    size = k.shape[0]

    grid = np.zeros((n, c, h * up, w * up))
    grid[:, :, ::up, ::up] = x
    lo, hi = pad
    grid = np.pad(grid, ((0, 0), (0, 0), (lo, hi), (lo, hi)))

    flipped = k[::-1, ::-1]
    oh, ow = grid.shape[2] - size + 1, grid.shape[3] - size + 1
    out = np.zeros((n, c, oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = (grid[:, :, i : i + size, j : j + size] * flipped).sum(axis=(2, 3))
    return out[:, :, ::down, ::down]


@pytest.mark.parametrize("taps", [(1, 3, 3, 1), (1, 2, 5, 1)])
@pytest.mark.parametrize("up", [True, False])
def test_fir_matches_an_independent_transcription(taps: tuple[int, ...], up: bool) -> None:
    """The asymmetric filter is what catches a missing kernel flip."""
    data = np.random.default_rng(0).standard_normal((2, 3, 8, 8)).astype(np.float32)
    module = _FIRResample(3, taps, up=up).eval()
    got = module(lucid.tensor(data)).numpy().astype(np.float64)
    overhang = len(taps) - 2
    if up:
        want = _reference_upfirdn2d(
            data.astype(np.float64),
            taps,
            2,
            1,
            ((overhang + 1) // 2 + 1, overhang // 2),
        )
    else:
        want = _reference_upfirdn2d(
            data.astype(np.float64), taps, 1, 2, ((overhang + 1) // 2, overhang // 2)
        )
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 1e-5


def test_fir_preserves_a_constant_away_from_the_border() -> None:
    """Unit DC gain; the border is attenuated because both routes zero-pad."""
    const = lucid.ones((1, 3, 16, 16)) * 2.5
    up = _FIRResample(3, (1, 3, 3, 1), up=True).eval()(const).numpy()
    down = _FIRResample(3, (1, 3, 3, 1), up=False).eval()(const).numpy()
    assert np.abs(up[:, :, 4:-4, 4:-4] - 2.5).max() < 1e-5
    assert np.abs(down[:, :, 2:-2, 2:-2] - 2.5).max() < 1e-5


def test_fir_holds_no_parameters() -> None:
    module = _FIRResample(3, (1, 3, 3, 1), up=True)
    assert not list(module.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# The velocity field
# ─────────────────────────────────────────────────────────────────────────────


def test_field_preserves_shape() -> None:
    field = _VelocityField(_cfg()).eval()
    out = field(lucid.randn((2, 3, 8, 8)), lucid.tensor([0.3, 0.7]))
    assert out.shape == (2, 3, 8, 8)


def test_field_accepts_the_scalar_time_a_solver_passes() -> None:
    field = _VelocityField(_cfg()).eval()
    batched = field(lucid.randn((2, 3, 8, 8)), lucid.tensor([0.4, 0.4]))
    assert batched.shape == (2, 3, 8, 8)
    lucid.manual_seed(0)
    x = lucid.randn((2, 3, 8, 8))
    assert _worst(field(x, lucid.tensor(0.4)), field(x, lucid.tensor([0.4, 0.4]))) == 0.0


def test_high_resolution_path_runs_and_stays_finite() -> None:
    """Filtered resampling, both pyramids and Fourier time features at once."""
    cfg = _cfg(
        sample_size=16,
        base_channels=8,
        channel_mult=(1, 2, 2),
        num_res_blocks=1,
        attention_resolutions=(4,),
        fir=True,
        progressive="output_skip",
        progressive_input="input_skip",
        embedding_type="fourier",
        init_scale=1.0,
    )
    out = _VelocityField(cfg).eval()(lucid.randn((2, 3, 16, 16)), lucid.tensor([0.5, 0.5]))
    assert out.shape == (2, 3, 16, 16)
    assert bool(lucid.isfinite(out).all())


def test_fourier_time_embedding_is_finite_at_zero() -> None:
    """The reference embeds ``log(t)``, which is singular where ``t`` starts."""
    cfg = _cfg(embedding_type="fourier", init_scale=1.0)
    field = _VelocityField(cfg).eval()
    out = field(lucid.randn((2, 3, 8, 8)), lucid.tensor([0.0, 0.0]))
    assert bool(lucid.isfinite(out).all())


def test_time_scale_is_the_reference_value() -> None:
    """999, not 1000 — the last index of a 1000-step grid."""
    assert _TIME_SCALE == 999.0


def test_field_is_not_exported() -> None:
    """The model owns its field; no caller composes one."""
    import lucid.models as models

    assert not hasattr(models, "_VelocityField")
    assert not hasattr(models, "RectifiedFlowUNet")


# ─────────────────────────────────────────────────────────────────────────────
# Likelihood
# ─────────────────────────────────────────────────────────────────────────────


def test_log_prob_and_bits_per_dim_agree() -> None:
    cfg = _cfg(
        sample_size=(1, 2),
        in_channels=1,
        out_channels=1,
        channel_mult=(1,),
        attention_resolutions=(),
    )
    model = RectifiedFlowModel(cfg).eval()
    assert model.trace_method == "exact"
    x = lucid.randn((2, 1, 1, 2))
    lp = model.log_prob(x)
    bpd = model.bits_per_dim(x)
    assert lp.shape == (2,)
    assert np.abs(bpd.numpy() - (-lp.numpy() / (2 * np.log(2.0)))).max() < 1e-5


def test_zero_field_gives_the_prior_exactly() -> None:
    """With no transport the density must be the standard normal it starts from."""
    cfg = _cfg(
        sample_size=(1, 2),
        in_channels=1,
        out_channels=1,
        channel_mult=(1,),
        attention_resolutions=(),
    )
    model = RectifiedFlowModel(cfg).eval()
    for _, p in model.named_parameters():
        nn.init.zeros_(p)
    x = lucid.randn((4, 1, 1, 2))
    flat = x.reshape(4, -1).numpy()
    want = -0.5 * (flat**2).sum(axis=-1) - np.log(2.0 * np.pi)
    assert np.abs(model.log_prob(x).numpy() - want).max() < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Model plumbing
# ─────────────────────────────────────────────────────────────────────────────


def test_forward_returns_a_velocity() -> None:
    model = RectifiedFlowModel(_cfg()).eval()
    out = model(lucid.randn((2, 3, 8, 8)), lucid.tensor(0.4))
    assert isinstance(out, DiffusionModelOutput)
    assert out.sample.shape == (2, 3, 8, 8)


def test_generator_forward_and_generate() -> None:
    model = RectifiedFlowForImageGeneration(_cfg()).eval()
    out = model(lucid.randn((2, 3, 8, 8)))
    assert isinstance(out, DiffusionModelOutput)
    assert out.loss is not None and out.loss.shape == ()
    gen = model.generate(n_samples=2, steps=1)
    assert isinstance(gen, GenerationOutput)
    assert gen.samples.shape == (2, 3, 8, 8)
    assert model.nfe == 1


def test_generator_forwards_the_pairing_through() -> None:
    model = RectifiedFlowForImageGeneration(_cfg(init_scale=1.0)).eval()
    z0, z1 = model.reflow_pairs(n_samples=2, steps=4)
    assert model(z1, noise=z0).loss is not None


def test_state_dict_holds_each_weight_once() -> None:
    """The likelihood dynamics reference the field; they must not register it.

    A shared submodule is invisible to ``parameters()`` — Lucid dedupes by
    identity there — and shows up only as a doubled ``state_dict``, so a
    parameter count would not catch it.
    """
    model = RectifiedFlowModel(_cfg()).eval()
    buffers = len(list(model.buffers()))
    assert len(model.state_dict()) == len(list(model.parameters())) + buffers


def test_high_resolution_state_dict_holds_each_weight_once() -> None:
    """The variant that actually has buffers — filter kernels and frequencies."""
    model = RectifiedFlowModel(
        _cfg(
            sample_size=16,
            base_channels=8,
            channel_mult=(1, 2, 2),
            num_res_blocks=1,
            attention_resolutions=(4,),
            fir=True,
            progressive="output_skip",
            progressive_input="input_skip",
            embedding_type="fourier",
        )
    ).eval()
    buffers = len(list(model.buffers()))
    assert buffers > 0
    assert len(model.state_dict()) == len(list(model.parameters())) + buffers


def test_rejects_wrongly_shaped_input() -> None:
    model = RectifiedFlowModel(_cfg()).eval()
    with pytest.raises(ValueError):
        model.rectified_flow_loss(lucid.randn((2, 3, 16, 16)))


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_FACTORIES = [
    "rectified_flow_cifar",
    "rectified_flow_bedroom",
    "rectified_flow_church",
    "rectified_flow_celeba_hq",
    "rectified_flow_afhq_cat",
]


@pytest.mark.parametrize("name", _FACTORIES + [f"{n}_gen" for n in _FACTORIES])
def test_factories_are_registered(name: str) -> None:
    assert is_model(name)


def test_cifar_factory_builds_the_published_configuration() -> None:
    model = create_model("rectified_flow_cifar")
    assert model.config.sample_size == 32
    assert model.config.num_res_blocks == 4
    assert model.config.dropout == 0.15
    assert model.config.fir is False


def test_high_resolution_factories_carry_the_pyramid() -> None:
    for name in ("rectified_flow_bedroom", "rectified_flow_celeba_hq"):
        cfg = create_model(name).config
        assert cfg.sample_size == 256
        assert cfg.channel_mult == (1, 1, 2, 2, 2, 2, 2)
        assert cfg.fir is True
        assert cfg.progressive == "output_skip"
        assert cfg.progressive_input == "input_skip"
        assert cfg.embedding_type == "fourier"


def test_overrides_reach_the_config() -> None:
    model = create_model("rectified_flow_cifar", sample_size=8, base_channels=16, t_schedule="t1")
    assert model.config.sample_size == 8
    assert model.t_schedule == "t1"
