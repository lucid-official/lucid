"""Unit tests for Neural ODE (Chen et al., 2018) — continuous normalizing flow."""

import math

import pytest

import lucid
from lucid.models import (
    GenerationOutput,
    NeuralODEConfig,
    NeuralODEForImageGeneration,
    NeuralODEModel,
    NormalizingFlowOutput,
    create_model,
    is_model,
)
from lucid.models._utils._generative import flow_prior_log_prob


def _cfg(**overrides: object) -> NeuralODEConfig:
    """The paper's own setting: two-dimensional, so the trace stays exact."""
    base: dict[str, object] = {
        "sample_size": (1, 2),
        "in_channels": 1,
        "out_channels": 1,
        "hidden_dim": 16,
        "num_blocks": 2,
        "rtol": 1e-7,
        "atol": 1e-9,
    }
    base.update(overrides)
    return NeuralODEConfig(**base)  # type: ignore[arg-type]


def _excited(**overrides: object) -> NeuralODEModel:
    """Build a model whose field actually moves the state.

    A freshly initialised field barely transports anything, so a wrong
    log-determinant would still look approximately right; scaling the
    weights up gives the tests something to be wrong about.  The seed is
    set before construction so the initialisation — and therefore how far
    the flow pushes probability mass — is the same on every run.
    """
    lucid.manual_seed(0)
    model = NeuralODEModel(_cfg(**overrides)).eval()
    with lucid.no_grad():
        for param in model.parameters():
            param += param * 4.0
    return model


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


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestNeuralODEConfig:
    def test_defaults(self) -> None:
        cfg = NeuralODEConfig()
        assert cfg.hidden_dim == 64  # the paper's M
        assert cfg.num_blocks == 2
        assert cfg.solver == "dopri5"
        assert cfg.prior == "gaussian"
        assert cfg.use_adjoint is True
        assert cfg.trace_noise == "rademacher"
        assert cfg.model_type == "neural_ode"

    def test_data_dim_flattens_the_sample(self) -> None:
        assert NeuralODEConfig(sample_size=32, in_channels=3).data_dim == 3072
        assert NeuralODEConfig(sample_size=(1, 2), in_channels=1, out_channels=1).data_dim == 2

    def test_trace_method_follows_the_dimension(self) -> None:
        # Exact where the paper's method is affordable, estimated above it.
        small = NeuralODEConfig(sample_size=(1, 2), in_channels=1, out_channels=1)
        assert small.resolved_trace_method == "exact"
        assert NeuralODEConfig(sample_size=32, in_channels=3).resolved_trace_method == (
            "hutchinson"
        )
        # An explicit choice always wins over the dimension.
        forced = NeuralODEConfig(sample_size=32, in_channels=3, trace_method="exact")
        assert forced.resolved_trace_method == "exact"

    def test_bijection_requires_matching_channels(self) -> None:
        with pytest.raises(ValueError, match="bijection"):
            NeuralODEConfig(in_channels=3, out_channels=1)

    @pytest.mark.parametrize(
        "field",
        [
            {"hidden_dim": 0},
            {"num_blocks": 0},
            {"rtol": 0.0},
            {"atol": -1e-5},
            {"exact_trace_max_dim": 0},
        ],
    )
    def test_rejects_impossible_settings(self, field: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            NeuralODEConfig(**field)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# The flow itself
# ─────────────────────────────────────────────────────────────────────────────


class TestNeuralODEModel:
    def test_encode_decode_shapes(self) -> None:
        model = NeuralODEModel(_cfg()).eval()
        z, log_det = model.encode(lucid.rand((3, 1, 1, 2)))
        assert z.shape == (3, 2)
        assert log_det.shape == (3,)
        assert model.decode(z).shape == (3, 1, 1, 2)

    def test_round_trip_is_the_field_run_backwards(self) -> None:
        # No hand-written inverse exists to disagree with the forward map:
        # decode is the same vector field integrated from t=1 to t=0.
        model = _excited()
        x = lucid.rand((4, 1, 1, 2)) * 2 - 1
        z, _ = model.encode(x)
        assert max(abs(v) for v in _flat(model.decode(z) - x)) < 1e-4

    def test_log_prob_is_prior_plus_log_determinant(self) -> None:
        model = _excited()
        x = lucid.rand((4, 1, 1, 2))
        z, log_det = model.encode(x)
        manual = _flat(flow_prior_log_prob("gaussian", z).sum(dim=-1) + log_det)
        direct = _flat(model.log_prob(x))
        assert max(abs(a - b) for a, b in zip(manual, direct)) < 1e-5

    def test_density_integrates_to_one(self) -> None:
        r"""The check that a sign error in the divergence cannot survive.

        :math:`\int p(x)\,dx` is exactly 1 for any correctly implemented
        continuous flow, whatever its weights — and lands nowhere near 1
        if the trace is accumulated with the wrong sign or dropped.
        """
        model = _excited()
        # Wide enough that the mass left outside is smaller than the
        # tolerance: an excited flow transports probability well past the
        # prior's own support.
        radius, count = 10.0, 121
        step = 2 * radius / (count - 1)
        axis = [-radius + i * step for i in range(count)]
        total = 0.0
        for index, y in enumerate(axis):
            row = [
                math.exp(v) for v in _flat(model.log_prob(lucid.tensor([[[[x, y]]] for x in axis])))
            ]
            strip = (sum(row) - 0.5 * (row[0] + row[-1])) * step
            edge = 0.5 if index in (0, count - 1) else 1.0
            total += edge * strip * step
        assert abs(total - 1.0) < 5e-3

    def test_nfe_counts_the_solver_s_work(self) -> None:
        # Depth is not a hyper-parameter here — it is whatever accuracy the
        # caller asked for, so a tighter tolerance can only cost more.
        loose = _excited(rtol=1e-4, atol=1e-4)
        tight = NeuralODEModel(_cfg(rtol=1e-10, atol=1e-12)).eval()
        tight.load_state_dict(loose.state_dict())

        x = lucid.rand((2, 1, 1, 2))
        loose.encode(x)
        tight.encode(x)
        assert loose.nfe > 0
        assert tight.nfe > loose.nfe

    def test_hutchinson_is_unbiased(self) -> None:
        exact = _excited()
        estimated = NeuralODEModel(_cfg(trace_method="hutchinson")).eval()
        estimated.load_state_dict(exact.state_dict())

        x = lucid.rand((2, 1, 1, 2)) * 2 - 1
        reference = _flat(exact.encode(x)[1])
        draws = 200
        total = [0.0] * len(reference)
        for _ in range(draws):
            total = [a + b for a, b in zip(total, _flat(estimated.encode(x)[1]))]
        mean = [v / draws for v in total]
        # Loose on purpose: this is a Monte-Carlo mean, and the point is
        # that it converges on the exact value rather than some other one.
        assert max(abs(a - b) for a, b in zip(reference, mean)) < 0.05

    def test_adjoint_and_direct_gradients_agree(self) -> None:
        """Constant memory must not cost correctness.

        The adjoint reconstructs the gradient by integrating backwards;
        the direct path differentiates the solver's own arithmetic.  They
        are different computations of the same derivative.
        """
        adjoint = _excited()
        direct = NeuralODEModel(_cfg(use_adjoint=False)).eval()
        direct.load_state_dict(adjoint.state_dict())

        x = lucid.rand((2, 1, 1, 2)) * 2 - 1
        for model in (adjoint, direct):
            model.log_prob(x).sum().backward()

        scale = 0.0
        worst = 0.0
        for left, right in zip(adjoint.parameters(), direct.parameters()):
            assert left.grad is not None and right.grad is not None
            a, b = _flat(left.grad), _flat(right.grad)
            scale = max(scale, max(abs(v) for v in b))
            worst = max(worst, max(abs(p - q) for p, q in zip(a, b)))
        assert worst <= 1e-4 * max(scale, 1.0)

    def test_rejects_wrong_shapes(self) -> None:
        model = NeuralODEModel(_cfg()).eval()
        with pytest.raises(ValueError, match=r"expects \(B, 1, 1, 2\)"):
            model.encode(lucid.rand((2, 3, 4, 4)))
        with pytest.raises(ValueError, match=r"expects \(B, 2\)"):
            model.decode(lucid.rand((2, 5)))

    def test_forward_returns_a_flow_output(self) -> None:
        model = NeuralODEModel(_cfg()).eval()
        out = model(lucid.rand((2, 1, 1, 2)))
        assert isinstance(out, NormalizingFlowOutput)
        assert out.latent.shape == (2, 2)
        assert out.log_det_jacobian.shape == (2,)
        assert out.log_prob.shape == (2,)
        assert out.loss is None


# ─────────────────────────────────────────────────────────────────────────────
# Task wrapper
# ─────────────────────────────────────────────────────────────────────────────


class TestNeuralODEForImageGeneration:
    def test_loss_is_bits_per_dim(self) -> None:
        model = NeuralODEForImageGeneration(_cfg()).eval()
        x = lucid.rand((3, 1, 1, 2))
        out = model(x)
        assert isinstance(out, NormalizingFlowOutput)
        assert out.loss is not None and out.loss.shape == ()
        expected = -sum(_flat(model.neural_ode.log_prob(x))) / (3 * 2 * math.log(2.0))
        assert abs(float(out.loss.tolist()) - expected) < 1e-4

    def test_generate_returns_samples(self) -> None:
        model = NeuralODEForImageGeneration(_cfg()).eval()
        out = model.generate(n_samples=3)
        assert isinstance(out, GenerationOutput)
        assert out.samples.shape == (3, 1, 1, 2)

    def test_generate_rejects_non_positive_temperature(self) -> None:
        model = NeuralODEForImageGeneration(_cfg()).eval()
        with pytest.raises(ValueError, match="temperature"):
            model.generate(n_samples=1, temperature=0.0)

    def test_gradients_reach_every_parameter(self) -> None:
        model = NeuralODEForImageGeneration(_cfg(rtol=1e-4, atol=1e-4))
        out = model(lucid.rand((2, 1, 1, 2)))
        assert out.loss is not None
        out.loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} received no gradient"
            assert max(abs(v) for v in _flat(param.grad)) > 0.0, f"{name} grad is zero"


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestNeuralODERegistry:
    @pytest.mark.parametrize("name", ["neural_ode", "neural_ode_gen"])
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_create_model_applies_overrides(self) -> None:
        model = create_model("neural_ode", sample_size=(1, 2), in_channels=1, out_channels=1)
        assert isinstance(model, NeuralODEModel)
        assert model.input_dim == 2
        assert model.trace_method == "exact"

    def test_default_shape_switches_to_the_estimator(self) -> None:
        model = create_model("neural_ode_gen")
        assert isinstance(model, NeuralODEForImageGeneration)
        assert model.neural_ode.input_dim == 3072
        assert model.neural_ode.trace_method == "hutchinson"
