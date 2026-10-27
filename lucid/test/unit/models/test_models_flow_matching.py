"""Unit tests for Flow Matching (Lipman et al., 2023) — simulation-free CNF."""

import math

import pytest

import lucid
import lucid.nn as _nn
from lucid.models import (
    DiffusionModelOutput,
    FlowMatchingConfig,
    FlowMatchingForImageGeneration,
    FlowMatchingModel,
    GenerationOutput,
    create_model,
    is_model,
)
from lucid.models.generative.flow_matching._model import _path_coefficients, _VelocityField

_PATHS = ("ot", "diffusion")


def _cfg(**overrides: object) -> FlowMatchingConfig:
    """Small enough to solve repeatedly, wide enough to be a real U-Net."""
    base: dict[str, object] = {
        "sample_size": 8,
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 16,
        "channel_mult": (1, 2),
        "num_res_blocks": 1,
        "attention_resolutions": (4,),
        "num_head_channels": 8,
        "resnet_groups": 8,
        "rtol": 1e-5,
        "atol": 1e-5,
    }
    base.update(overrides)
    return FlowMatchingConfig(**base)  # type: ignore[arg-type]


def _flat(tensor: lucid.Tensor) -> list[float]:
    out: list[float] = []

    def walk(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        else:
            out.append(float(value))  # type: ignore[arg-type]

    walk(tensor.tolist())
    return out


def _worst(tensor: lucid.Tensor) -> float:
    return max(abs(v) for v in _flat(tensor))


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestFlowMatchingConfig:
    def test_defaults(self) -> None:
        cfg = FlowMatchingConfig()
        assert cfg.path == "ot"
        assert cfg.base_channels == 128
        assert cfg.num_res_blocks == 2
        assert cfg.num_head_channels == 64
        assert cfg.attention_resolutions == (32, 16, 8)
        assert cfg.solver == "dopri5"
        assert cfg.model_type == "flow_matching"

    def test_data_dim_and_trace_default(self) -> None:
        assert FlowMatchingConfig().data_dim == 3072
        assert FlowMatchingConfig().resolved_trace_method == "hutchinson"
        tiny = FlowMatchingConfig(sample_size=(1, 2), in_channels=1, out_channels=1)
        assert tiny.data_dim == 2
        assert tiny.resolved_trace_method == "exact"

    def test_velocity_lives_in_the_sample_space(self) -> None:
        with pytest.raises(ValueError, match="in_channels must equal out_channels"):
            FlowMatchingConfig(in_channels=3, out_channels=1)

    @pytest.mark.parametrize(
        "field",
        [
            {"base_channels": 0},
            {"channel_mult": ()},
            {"channel_mult": (1, 0)},
            {"num_res_blocks": 0},
            {"num_head_channels": 0},
            {"dropout": 1.0},
            {"sigma_min": 1.0},
            {"beta_min": 30.0},
            {"rtol": 0.0},
            {"exact_trace_max_dim": 0},
        ],
    )
    def test_rejects_impossible_settings(self, field: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            FlowMatchingConfig(**field)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Probability paths — Theorem 3
# ─────────────────────────────────────────────────────────────────────────────


class TestProbabilityPaths:
    @pytest.mark.parametrize("path", _PATHS)
    def test_derivatives_match_finite_differences(self, path: str) -> None:
        """The schedules' derivatives are the only hand-derived maths here.

        Nothing downstream would notice if they were wrong — the loss
        would still be a plausible number, and every shape would agree.
        """
        step = 1e-5
        kwargs = {"sigma_min": 1e-4, "beta_min": 0.1, "beta_max": 20.0}
        for value in (0.05, 0.25, 0.5, 0.75, 0.95):
            here = lucid.tensor([value], dtype=lucid.float64)
            _, _, a_dot, sigma_dot = _path_coefficients(path, here, **kwargs)  # type: ignore[arg-type]
            plus = lucid.tensor([value + step], dtype=lucid.float64)
            minus = lucid.tensor([value - step], dtype=lucid.float64)
            a_p, s_p, _, _ = _path_coefficients(path, plus, **kwargs)  # type: ignore[arg-type]
            a_m, s_m, _, _ = _path_coefficients(path, minus, **kwargs)  # type: ignore[arg-type]
            assert abs(_flat(a_dot)[0] - (_flat(a_p)[0] - _flat(a_m)[0]) / (2 * step)) < 1e-7
            assert abs(_flat(sigma_dot)[0] - (_flat(s_p)[0] - _flat(s_m)[0]) / (2 * step)) < 1e-7

    def test_optimal_transport_boundary_conditions_are_exact(self) -> None:
        kwargs = {"sigma_min": 1e-4, "beta_min": 0.1, "beta_max": 20.0}
        a0, s0, _, _ = _path_coefficients(
            "ot", lucid.tensor([0.0], dtype=lucid.float64), **kwargs  # type: ignore[arg-type]
        )
        a1, s1, _, _ = _path_coefficients(
            "ot", lucid.tensor([1.0], dtype=lucid.float64), **kwargs  # type: ignore[arg-type]
        )
        assert _flat(a0)[0] == 0.0 and _flat(s0)[0] == 1.0
        assert abs(_flat(a1)[0] - 1.0) < 1e-12
        assert abs(_flat(s1)[0] - 1e-4) < 1e-12

    def test_diffusion_reaches_the_noise_end_only_approximately(self) -> None:
        """A documented difference between the two paths, not a defect.

        The variance-preserving process converges on the prior in the
        limit; at ``t = 0`` a little of the data still shows through.
        """
        a0, s0, _, _ = _path_coefficients(
            "diffusion",
            lucid.tensor([0.0], dtype=lucid.float64),
            sigma_min=1e-4,
            beta_min=0.1,
            beta_max=20.0,
        )
        assert 0.0 < _flat(a0)[0] < 1e-2
        assert abs(_flat(s0)[0] - 1.0) < 1e-4

    @pytest.mark.parametrize("path", _PATHS)
    def test_reduced_target_equals_literal_theorem_3(self, path: str) -> None:
        r"""``sigma' x0 + a' x1`` is Theorem 3 without dividing by sigma.

        The literal expression divides by :math:`\sigma_t`, which goes to
        zero at ``t = 1`` on the diffusion path; the reduced one is the
        same quantity everywhere and finite there.
        """
        model = FlowMatchingModel(
            _cfg(
                sample_size=(1, 2),
                in_channels=1,
                out_channels=1,
                path=path,
                channel_mult=(1,),
                attention_resolutions=(),
                resnet_groups=4,
            )
        ).eval()
        x1 = lucid.randn((5, 1, 1, 2))
        x0 = lucid.randn((5, 1, 1, 2))
        t = lucid.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

        reduced = model.conditional_target(x1, x0, t)
        x_t = model.path_sample(x1, x0, t)
        a, sigma, a_dot, sigma_dot = model.coefficients(t.reshape(-1, 1, 1, 1))
        literal = (sigma_dot / sigma) * (x_t - a * x1) + a_dot * x1
        assert _worst(reduced - literal) < 1e-5

    def test_optimal_transport_target_is_the_published_closed_form(self) -> None:
        """Paper eq. (23), which the general derivation has to reproduce."""
        model = FlowMatchingModel(
            _cfg(
                sample_size=(1, 2),
                in_channels=1,
                out_channels=1,
                path="ot",
                channel_mult=(1,),
                attention_resolutions=(),
                resnet_groups=4,
            )
        ).eval()
        x1 = lucid.randn((6, 1, 1, 2))
        x0 = lucid.randn((6, 1, 1, 2))
        published = x1 - (1.0 - 1e-4) * x0
        for value in (0.0, 0.25, 0.5, 0.75, 1.0):
            t = lucid.tensor([value] * 6)
            assert _worst(model.conditional_target(x1, x0, t) - published) < 1e-6

    def test_straight_paths_are_cheaper_to_follow(self) -> None:
        """The reason the paper prefers optimal transport.

        Measured on the analytic conditional fields, so this is a property
        of the paths and not of any trained network.  Stopped short of
        ``t = 1``: the diffusion path's field carries a ``1/sigma_t``
        factor that diverges exactly there, which is most of why it is
        expensive, and integrating into a singularity makes for a slow
        test rather than a clearer one.
        """
        counts: dict[str, int] = {}
        for path in _PATHS:
            model = FlowMatchingModel(
                _cfg(
                    sample_size=(1, 2),
                    in_channels=1,
                    out_channels=1,
                    path=path,
                    channel_mult=(1,),
                    attention_resolutions=(),
                    resnet_groups=4,
                )
            ).eval()
            lucid.manual_seed(1)
            x1 = lucid.randn((8, 1, 1, 2)) * 2.0
            x0 = lucid.randn((8, 1, 1, 2))
            calls = 0

            def field(t: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
                nonlocal calls
                calls += 1
                image = x.reshape(8, 1, 1, 2)
                a, sigma, a_dot, sigma_dot = model.coefficients(t.reshape(1, 1, 1, 1))
                return ((sigma_dot / sigma) * (image - a * x1) + a_dot * x1).reshape(8, -1)

            lucid.diffeq.odeint(
                field, x0.reshape(8, -1), [0.0, 0.9], rtol=1e-5, atol=1e-5, return_trajectory=False
            )
            counts[path] = calls

        assert counts["ot"] < counts["diffusion"]

    @pytest.mark.parametrize("path", _PATHS)
    def test_conditional_field_transports_noise_onto_the_data(self, path: str) -> None:
        """Integrating u_t(.|x1) from t=0 must land on psi_1(x0).

        Exercises the sign of the field, the time direction, and the
        solver in one statement with a known answer.
        """
        model = FlowMatchingModel(
            _cfg(
                sample_size=(1, 2),
                in_channels=1,
                out_channels=1,
                path=path,
                channel_mult=(1,),
                attention_resolutions=(),
                resnet_groups=4,
            )
        ).eval()
        x1 = lucid.randn((4, 1, 1, 2))
        x0 = lucid.randn((4, 1, 1, 2))

        def field(t: lucid.Tensor, x: lucid.Tensor) -> lucid.Tensor:
            image = x.reshape(4, 1, 1, 2)
            a, sigma, a_dot, sigma_dot = model.coefficients(t.reshape(1, 1, 1, 1))
            return ((sigma_dot / sigma) * (image - a * x1) + a_dot * x1).reshape(4, -1)

        arrived = lucid.diffeq.odeint(
            field, x0.reshape(4, -1), [0.0, 1.0], rtol=1e-9, atol=1e-11, return_trajectory=False
        )
        a1, s1, _, _ = model.coefficients(lucid.tensor(1.0))
        expected = (a1 * x1 + s1 * x0).reshape(4, -1)
        assert _worst(arrived - expected) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Velocity field
# ─────────────────────────────────────────────────────────────────────────────


class TestVelocityField:
    def test_output_matches_the_input_space(self) -> None:
        unet = _VelocityField(_cfg()).eval()
        x = lucid.randn((2, 3, 8, 8))
        assert unet(x, lucid.tensor([0.3, 0.7])).shape == (2, 3, 8, 8)

    def test_scalar_time_is_broadcast(self) -> None:
        """The solver hands the field a 0-D time, not one per sample."""
        unet = _VelocityField(_cfg()).eval()
        x = lucid.randn((3, 3, 8, 8))
        scalar = unet(x, lucid.tensor(0.4))
        per_sample = unet(x, lucid.tensor([0.4, 0.4, 0.4]))
        assert _worst(scalar - per_sample) < 1e-6

    def test_time_actually_changes_the_field(self) -> None:
        # The reference zero-initialises ``conv_out`` *and* the last conv of
        # every residual branch, so a freshly built field is dormant: it
        # predicts exactly zero, and — because ``t`` enters only through those
        # residual branches — the time embedding cannot reach the output at
        # all.  Both are the intended initial state, so comparing raw outputs
        # at init would compare 0 against 0 and prove nothing.  Wake the
        # residual branches and the output projection, as the first optimiser
        # step does, then check that t still reaches the prediction.
        unet = _VelocityField(_cfg()).eval()
        for name, mod in unet.named_modules():
            if isinstance(mod, _nn.Conv2d) and (
                name.endswith("conv2") or name.endswith("conv_out")
            ):
                _nn.init.normal_(mod.weight, mean=0.0, std=0.05)
        x = lucid.randn((2, 3, 8, 8))
        assert _worst(unet(x, lucid.tensor(0.1)) - unet(x, lucid.tensor(0.9))) > 1e-4

    def test_predicted_velocity_is_zero_at_initialisation(self) -> None:
        """The zero-initialised output projection is a documented contract."""
        unet = _VelocityField(_cfg()).eval()
        x = lucid.randn((2, 3, 8, 8))
        assert _worst(unet(x, lucid.tensor(0.5))) == 0.0

    def test_matches_the_checkpoint_validated_u_net_exactly(self) -> None:
        """The wiring is borrowed, so pin it to the thing it was borrowed from.

        A U-Net is a few hundred lines of index arithmetic — skip-connection
        order, the downsample's asymmetric padding, the attention reshape,
        where the time embedding is injected.  Every one of those can be
        wrong and still give a network that trains happily and reports a
        plausible loss; shapes do not catch it.

        ``_DDPMUNet`` loads converted official checkpoints, so its wiring is
        validated against published weights.  Configured equivalently, this
        field must *be* that function — same parameter names, same shapes,
        bit-identical output — which transfers that validation here and
        fails loudly if either drifts.
        """
        from lucid.models.generative.ddpm import DDPMConfig
        from lucid.models.generative.ddpm._model import _DDPMUNet

        common: dict[str, object] = {
            "sample_size": 16,
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 16,
            "channel_mult": (1, 2, 2),
            "num_res_blocks": 2,
            "attention_resolutions": (8, 4),
            "dropout": 0.0,
            "resnet_groups": 8,
        }
        reference = _DDPMUNet(DDPMConfig(**common, num_heads=1)).eval()  # type: ignore[arg-type]
        # More head channels than any stage is wide reproduces the
        # reference's fixed single head.
        field = _VelocityField(
            FlowMatchingConfig(**common, num_head_channels=10_000)  # type: ignore[arg-type]
        ).eval()

        assert sorted(field.state_dict()) == sorted(reference.state_dict())
        field.load_state_dict(reference.state_dict())

        x = lucid.randn((2, 3, 16, 16))
        # Discrete step 400 and continuous t = 0.4 meet at the same
        # embedding input, since the field scales time by 1000.
        assert (
            _worst(reference(x, lucid.tensor([400, 400])) - field(x, lucid.tensor([0.4, 0.4])))
            == 0.0
        )

    def test_attention_head_count_follows_the_width(self) -> None:
        from lucid.models.generative.flow_matching._model import _AttentionBlock

        assert _AttentionBlock(64, head_channels=16, groups=8).num_heads == 4
        assert _AttentionBlock(128, head_channels=16, groups=8).num_heads == 8
        # Never zero, and never a count the channels do not divide by.
        assert _AttentionBlock(8, head_channels=64, groups=8).num_heads == 1


# ─────────────────────────────────────────────────────────────────────────────
# The model
# ─────────────────────────────────────────────────────────────────────────────


class TestFlowMatchingModel:
    def test_training_solves_nothing(self) -> None:
        """The method's whole claim, directly observable."""
        model = FlowMatchingModel(_cfg())
        loss, prediction, target = model.flow_matching_loss(lucid.randn((2, 3, 8, 8)))
        assert loss.shape == ()
        assert prediction.shape == (2, 3, 8, 8)
        assert target.shape == (2, 3, 8, 8)
        assert model.nfe == 0

    def test_path_sample_hits_both_endpoints(self) -> None:
        model = FlowMatchingModel(_cfg()).eval()
        x1 = lucid.randn((3, 3, 8, 8))
        x0 = lucid.randn((3, 3, 8, 8))
        at_zero = model.path_sample(x1, x0, lucid.tensor([0.0] * 3))
        at_one = model.path_sample(x1, x0, lucid.tensor([1.0] * 3))
        assert _worst(at_zero - x0) < 1e-6
        assert _worst(at_one - x1) < 1e-3  # sigma_min * x0 remains

    def test_sampling_shapes(self) -> None:
        model = FlowMatchingModel(_cfg()).eval()
        assert model.sample(n_samples=2).shape == (2, 3, 8, 8)
        assert model.nfe > 0
        assert model.sample(n_samples=2, steps=4).shape == (2, 3, 8, 8)

    def test_fixed_step_sampling_costs_exactly_the_budget(self) -> None:
        # ``steps`` is a *budget*, so the cost has to be a fixed multiple of
        # it: one field evaluation per step for Euler, two for midpoint, four
        # for RK4 — and no retries in any case.
        for solver, per_step in (("euler", 1), ("midpoint", 2), ("rk4", 4)):
            model = FlowMatchingModel(_cfg(solver=solver)).eval()
            model.sample(n_samples=2, steps=5)
            assert model.nfe == 5 * per_step, solver

    def test_fixed_step_sampling_refuses_to_go_adaptive(self) -> None:
        # ``solver`` defaults to an adaptive method, which on a fixed grid
        # would re-choose its own step sizes and stop ``steps`` being a
        # budget at all.  The fixed-step path falls back to Euler instead —
        # one evaluation per step, as the reference's sampler defaults to.
        model = FlowMatchingModel(_cfg(solver="dopri5")).eval()
        model.sample(n_samples=2, steps=5)
        assert model.nfe == 5

    def test_sampling_is_deterministic_given_the_noise(self) -> None:
        model = FlowMatchingModel(_cfg()).eval()
        noise = lucid.randn((2, 3, 8, 8))
        first = model.sample(noise=noise, steps=4)
        second = model.sample(noise=noise, steps=4)
        assert _worst(first - second) == 0.0

    def test_a_zero_field_leaves_the_prior_untouched(self) -> None:
        r"""With :math:`v \equiv 0` the flow is the identity.

        Then :math:`\log p_1(x)` must be exactly the standard normal's
        log-density — which pins the time direction, the prior's
        normalising constant, and the fact that a vanishing divergence
        contributes nothing.
        """
        model = FlowMatchingModel(_cfg()).eval()
        with lucid.no_grad():
            model.field.conv_out.weight *= 0.0
            if model.field.conv_out.bias is not None:
                model.field.conv_out.bias *= 0.0

        x = lucid.randn((2, 3, 8, 8))
        dim = 3 * 8 * 8
        expected = [
            -0.5 * sum(v * v for v in row) - 0.5 * dim * math.log(2.0 * math.pi)
            for row in [_flat(x)[i * dim : (i + 1) * dim] for i in range(2)]
        ]
        got = _flat(model.log_prob(x))
        assert max(abs(a - b) for a, b in zip(expected, got)) < 1e-2

    def test_bits_per_dim_is_the_scaled_negative_likelihood(self) -> None:
        model = FlowMatchingModel(_cfg()).eval()
        x = lucid.randn((2, 3, 8, 8))
        lucid.manual_seed(0)
        log_prob = _flat(model.log_prob(x))
        lucid.manual_seed(0)
        bits = _flat(model.bits_per_dim(x))
        scale = 3 * 8 * 8 * math.log(2.0)
        assert max(abs(-a / scale - b) for a, b in zip(log_prob, bits)) < 1e-4

    def test_rejects_wrong_shapes(self) -> None:
        model = FlowMatchingModel(_cfg()).eval()
        with pytest.raises(ValueError, match=r"expects \(B, 3, 8, 8\)"):
            model.flow_matching_loss(lucid.randn((2, 1, 8, 8)))

    def test_forward_returns_the_velocity(self) -> None:
        model = FlowMatchingModel(_cfg()).eval()
        out = model(lucid.randn((2, 3, 8, 8)), lucid.tensor(0.5))
        assert isinstance(out, DiffusionModelOutput)
        assert out.sample.shape == (2, 3, 8, 8)
        assert out.loss is None


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper
# ─────────────────────────────────────────────────────────────────────────────


class TestFlowMatchingForImageGeneration:
    def test_forward_carries_the_objective(self) -> None:
        model = FlowMatchingForImageGeneration(_cfg())
        out = model(lucid.randn((2, 3, 8, 8)))
        assert isinstance(out, DiffusionModelOutput)
        assert out.loss is not None and out.loss.shape == ()
        assert out.sample.shape == (2, 3, 8, 8)

    def test_generate_returns_samples(self) -> None:
        model = FlowMatchingForImageGeneration(_cfg()).eval()
        out = model.generate(n_samples=3, steps=4)
        assert isinstance(out, GenerationOutput)
        assert out.samples.shape == (3, 3, 8, 8)

    def test_generate_rejects_impossible_budgets(self) -> None:
        model = FlowMatchingForImageGeneration(_cfg()).eval()
        with pytest.raises(ValueError, match="n_samples"):
            model.generate(n_samples=0)
        with pytest.raises(ValueError, match="steps"):
            model.generate(n_samples=1, steps=0)

    def test_gradients_reach_every_parameter(self) -> None:
        model = FlowMatchingForImageGeneration(_cfg())
        out = model(lucid.randn((2, 3, 8, 8)))
        assert out.loss is not None
        out.loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} received no gradient"

    def test_training_reduces_the_objective(self) -> None:
        """A regression that never converges is not a regression.

        Each step draws its own time and noise, so two steps' losses differ
        by more than twelve steps of training move them, and comparing them
        was a coin flip on the seed.  The first step's draws are replayed
        after training instead: the same sample of the objective, before and
        after.
        """
        import lucid.optim as optim

        lucid.manual_seed(0)
        model = FlowMatchingForImageGeneration(_cfg())
        opt = optim.Adam(model.parameters(), lr=3e-3)
        target = lucid.randn((4, 3, 8, 8))

        draws = lucid.get_rng_state()
        first = 0.0
        for step in range(12):
            opt.zero_grad()
            out = model(target)
            assert out.loss is not None
            out.loss.backward()
            opt.step()
            first = float(out.loss.tolist()) if step == 0 else first
        lucid.set_rng_state(draws)
        with lucid.no_grad():
            after = model(target).loss
        assert after is not None
        assert float(after.tolist()) < first


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestFlowMatchingRegistry:
    @pytest.mark.parametrize(
        "name",
        [
            "flow_matching_cifar",
            "flow_matching_imagenet32",
            "flow_matching_imagenet64",
            "flow_matching_imagenet128",
            "flow_matching_cifar_gen",
            "flow_matching_imagenet32_gen",
            "flow_matching_imagenet64_gen",
            "flow_matching_imagenet128_gen",
        ],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_paper_sample_sizes(self) -> None:
        assert create_model("flow_matching_cifar").config.sample_size == 32
        assert create_model("flow_matching_imagenet64").config.sample_size == 64
        assert create_model("flow_matching_imagenet128").config.sample_size == 128

    def test_create_model_applies_overrides(self) -> None:
        model = create_model(
            "flow_matching_cifar",
            sample_size=8,
            base_channels=16,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            resnet_groups=8,
        )
        assert isinstance(model, FlowMatchingModel)
        assert model.input_dim == 3 * 8 * 8
        assert model.path == "ot"
