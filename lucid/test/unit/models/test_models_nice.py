"""Unit tests for NICE (Dinh et al., 2014) — first flow family in the zoo."""

import math

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.models import (
    GenerationOutput,
    NICEConfig,
    NICEForImageGeneration,
    NICEModel,
    NormalizingFlowOutput,
    create_model,
    is_model,
)


def _tiny_cfg(**overrides: object) -> NICEConfig:
    """16-dimensional flow — small enough for exact Jacobians."""
    base: dict[str, object] = {
        "input_dim": 16,
        "num_coupling_layers": 4,
        "num_hidden_layers": 2,
        "hidden_dim": 8,
    }
    base.update(overrides)
    return NICEConfig(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestNICEConfig:
    def test_defaults_match_paper_mnist_column(self) -> None:
        cfg = NICEConfig()
        assert cfg.input_dim == 784
        assert cfg.num_coupling_layers == 4
        assert cfg.num_hidden_layers == 5
        assert cfg.hidden_dim == 1_000
        assert cfg.prior == "logistic"
        assert cfg.act_fn == "relu"
        assert cfg.input_reorder == "tile"
        assert cfg.init_range == 0.01
        assert cfg.model_type == "nice"

    def test_split_is_half_the_dimension(self) -> None:
        cfg = NICEConfig(input_dim=3_072)
        assert cfg.split == 1_536

    def test_spatial_fields_are_pinned_to_the_vector_encoding(self) -> None:
        """The base tier's image fields carry no invented shape."""
        cfg = NICEConfig(input_dim=3_072)
        assert cfg.in_channels == 1 and cfg.out_channels == 1
        assert cfg.sample_size == (1, 3_072)

    def test_odd_input_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be even"):
            NICEConfig(input_dim=9)

    def test_non_positive_input_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="input_dim must be positive"):
            NICEConfig(input_dim=0)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("num_coupling_layers", 0, "num_coupling_layers"),
            ("num_hidden_layers", 0, "num_hidden_layers"),
            ("hidden_dim", 0, "hidden_dim"),
            ("init_range", 0.0, "init_range"),
        ],
    )
    def test_non_positive_knobs_rejected(self, field: str, value: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _tiny_cfg(**{field: value})


# ─────────────────────────────────────────────────────────────────────────────
# The bijection
# ─────────────────────────────────────────────────────────────────────────────


class TestNICEModel:
    def test_encode_shapes(self) -> None:
        cfg = _tiny_cfg()
        m = NICEModel(cfg).eval()
        h, log_det = m.encode(lucid.rand((3, 16)))
        assert tuple(h.shape) == (3, cfg.input_dim)
        assert tuple(log_det.shape) == (3,)

    def test_forward_output_contract(self) -> None:
        m = NICEModel(_tiny_cfg()).eval()
        out = m(lucid.rand((2, 16)))
        assert isinstance(out, NormalizingFlowOutput)
        assert tuple(out.latent.shape) == (2, 16)
        assert tuple(out.log_prob.shape) == (2,)
        assert out.loss is None  # bare trunk never fills the loss slot

    @pytest.mark.parametrize("reorder", ["tile", "none"])
    def test_decode_is_exact_inverse(self, reorder: str) -> None:
        m = NICEModel(_tiny_cfg(input_reorder=reorder)).eval()
        # Perturb the scaling away from identity so the inverse has work
        # to do in every stage.
        lucid.manual_seed(0)
        x = lucid.rand((4, 16))
        h, _ = m.encode(x)
        x_rt = m.decode(h)
        maxdiff = float((x_rt - x).abs().max().item())
        assert maxdiff < 1e-5, f"round-trip drifted by {maxdiff}"

    def test_inverse_exact_with_non_identity_scaling(self) -> None:
        m = NICEModel(_tiny_cfg()).eval()
        nn.init.normal_(m.scaling.log_scale, mean=0.0, std=0.3)
        x = lucid.rand((2, 16))
        h, _ = m.encode(x)
        maxdiff = float((m.decode(h) - x).abs().max().item())
        assert maxdiff < 1e-4, f"round-trip drifted by {maxdiff}"

    def test_couplings_are_volume_preserving(self) -> None:
        """log|det| comes from the diagonal scaling alone (s = 0 → 0)."""
        m = NICEModel(_tiny_cfg()).eval()
        _h, log_det = m.encode(lucid.rand((2, 16)))
        assert abs(float(log_det[0].item())) < 1e-6

    def test_log_det_equals_sum_of_log_scale(self) -> None:
        m = NICEModel(_tiny_cfg()).eval()
        nn.init.constant_(m.scaling.log_scale, 0.25)
        _h, log_det = m.encode(lucid.rand((3, 16)))
        expected = 0.25 * 16
        for b in range(3):
            assert abs(float(log_det[b].item()) - expected) < 1e-4

    def test_log_det_matches_autograd_jacobian(self) -> None:
        """The reported log-determinant must be the true one.

        Builds the full 16x16 Jacobian of ``x → f(x)`` by reverse-mode
        differentiation and compares ``slogdet`` against what ``encode``
        claims.
        """
        m = NICEModel(_tiny_cfg()).eval()
        nn.init.normal_(m.scaling.log_scale, mean=0.0, std=0.4)

        x = lucid.rand((1, 16), requires_grad=True)

        def flow(inp: lucid.Tensor) -> lucid.Tensor:
            h, _ = m.encode(inp)
            return h.reshape(16)

        jac = lucid.autograd.jacobian(flow, x)
        assert isinstance(jac, lucid.Tensor)
        _sign, logabsdet = linalg.slogdet(jac.reshape(16, 16))

        _h, log_det = m.encode(x)
        assert abs(float(logabsdet.item()) - float(log_det[0].item())) < 1e-3

    def test_all_dimensions_interact_after_four_couplings(self) -> None:
        """Four alternating couplings make the Jacobian fully dense.

        The paper's motivation for stacking (≥ 3 layers) — with a single
        coupling half the rows stay one-hot.
        """

        def density(num_layers: int) -> float:
            m = NICEModel(_tiny_cfg(num_coupling_layers=num_layers)).eval()
            # Push weights away from the near-identity init (irange = 0.01)
            # so influence actually shows up numerically.
            for p in m.parameters():
                if p.ndim == 2:
                    nn.init.normal_(p, mean=0.0, std=0.5)
            x = lucid.rand((1, 16), requires_grad=True)

            def flow(inp: lucid.Tensor) -> lucid.Tensor:
                h, _ = m.encode(inp)
                return h.reshape(16)

            jac = lucid.autograd.jacobian(flow, x)
            assert isinstance(jac, lucid.Tensor)
            nonzero = (jac.reshape(16, 16).abs() > 1e-8).to(lucid.float32)
            return float(nonzero.mean().item())

        assert density(1) < 0.8
        assert density(4) > 0.99

    def test_log_prob_decomposition(self) -> None:
        from lucid.models._utils import flow_prior_log_prob

        m = NICEModel(_tiny_cfg()).eval()
        x = lucid.rand((2, 16))
        h, log_det = m.encode(x)
        manual = flow_prior_log_prob("logistic", h).sum(dim=-1) + log_det
        got = m.log_prob(x)
        assert float((got - manual).abs().max().item()) < 1e-5

    def test_gaussian_prior_log_prob(self) -> None:
        m = NICEModel(_tiny_cfg(prior="gaussian")).eval()
        assert m.prior == "gaussian"
        x = lucid.rand((2, 16))
        h, log_det = m.encode(x)
        expected = (-0.5 * (h * h + math.log(2.0 * math.pi))).sum(dim=-1) + log_det
        assert float((m.log_prob(x) - expected).abs().max().item()) < 1e-5

    def test_tile_matches_the_official_permutation(self) -> None:
        """The reshape/swap view chain must equal the released ``mode='tile'``
        permutation: even indices first, odd indices second."""
        from lucid.models.generative.nice._model import _tile, _untile

        dim = 16
        perm = list(range(0, dim, 2)) + list(range(1, dim, 2))
        x = lucid.arange(0, dim, dtype=lucid.float32).reshape(1, dim)
        assert _tile(x).tolist()[0] == [float(p) for p in perm]
        assert _untile(_tile(x)).tolist()[0] == [float(i) for i in range(dim)]

    def test_rejects_wrong_input_shape(self) -> None:
        m = NICEModel(_tiny_cfg()).eval()
        with pytest.raises(ValueError, match=r"expects \(B, 16\) samples"):
            m(lucid.rand((2, 8)))
        with pytest.raises(ValueError, match="expects"):
            m(lucid.rand((2, 1, 4, 4)))

    def test_rejects_wrong_latent_shape(self) -> None:
        m = NICEModel(_tiny_cfg()).eval()
        with pytest.raises(ValueError, match=r"expects \(B, 16\) latents"):
            m.decode(lucid.rand((2, 8)))


# ─────────────────────────────────────────────────────────────────────────────
# Generation head
# ─────────────────────────────────────────────────────────────────────────────


class TestNICEForImageGeneration:
    def test_loss_is_negative_mean_log_likelihood(self) -> None:
        m = NICEForImageGeneration(_tiny_cfg()).eval()
        x = lucid.rand((4, 16))
        out = m(x)
        assert out.loss is not None
        expected = -float(out.log_prob.mean().item())
        assert abs(float(out.loss.item()) - expected) < 1e-5

    def test_generate_shape(self) -> None:
        m = NICEForImageGeneration(_tiny_cfg()).eval()
        out = m.generate(n_samples=3)
        assert isinstance(out, GenerationOutput)
        assert tuple(out.samples.shape) == (3, 16)

    @pytest.mark.parametrize("prior", ["logistic", "gaussian"])
    def test_generate_finite_for_both_priors(self, prior: str) -> None:
        m = NICEForImageGeneration(_tiny_cfg(prior=prior)).eval()
        s = m.generate(n_samples=8).samples
        assert bool(s.isfinite().all().item())

    def test_backward_reaches_every_parameter(self) -> None:
        """NLL training path: one step must produce grads everywhere."""
        m = NICEForImageGeneration(_tiny_cfg())
        out = m(lucid.rand((2, 16)))
        assert out.loss is not None
        out.loss.backward()
        missing = [name for name, p in m.named_parameters() if p.grad is None]
        assert not missing, f"no gradient for {missing}"

    def test_scaling_gradient_matches_closed_form(self) -> None:
        r"""Closed-form check that the log-determinant term is in the loss.

        With a Gaussian prior and :math:`h = \exp(s) \odot u`,
        :math:`\log p = \sum_i [-\tfrac{1}{2}(h_i^2 + \log 2\pi) + s_i]`, so
        :math:`\partial(-\log p)/\partial s_i = h_i^2 - 1`.  Averaged over
        the batch that is ``mean_b(h²) - 1`` — wrong wiring of either the
        prior or the scaling breaks it.
        """
        m = NICEForImageGeneration(_tiny_cfg(prior="gaussian"))
        nn.init.normal_(m.nice.scaling.log_scale, mean=0.0, std=0.2)
        x = lucid.rand((5, 16))
        out = m(x)
        assert out.loss is not None
        out.loss.backward()

        grad = m.nice.scaling.log_scale.grad
        assert grad is not None
        expected = (out.latent * out.latent).mean(dim=0) - 1.0
        maxdiff = float((grad - expected).abs().max().item())
        assert maxdiff < 1e-4, f"scaling gradient off by {maxdiff}"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: does it actually learn a density?
# ─────────────────────────────────────────────────────────────────────────────


class TestNICETraining:
    def test_maximum_likelihood_training_fits_a_toy_density(self) -> None:
        """Train on an alternating-column pattern and check both directions.

        A flow can pass every shape / inverse test and still not be a
        density if the objective is wired backwards.  This pins the whole
        loop: NLL must fall, and prior samples pushed back through
        ``f^{-1}`` must reproduce the data's column structure.
        """
        import lucid.optim as optim

        lucid.manual_seed(0)
        cfg = _tiny_cfg(hidden_dim=32)
        model = NICEForImageGeneration(cfg)
        opt = optim.Adam(model.parameters(), lr=1e-3)

        pattern = lucid.tensor([[0.9, 0.1, 0.9, 0.1] * 4])

        def batch(n: int = 32) -> lucid.Tensor:
            return pattern.expand(n, 16) + lucid.randn((n, 16)) * 0.05

        first: float | None = None
        for step in range(200):
            out = model(batch())
            assert out.loss is not None
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            if step == 0:
                first = float(out.loss.item())

        final_out = model(batch(128))
        assert final_out.loss is not None
        assert first is not None
        last = float(final_out.loss.item())
        assert last < first - 1.0, f"NLL did not improve: {first:.3f} → {last:.3f}"

        samples = model.eval().generate(n_samples=128).samples
        cols = [float(samples[:, c::4].mean().item()) for c in range(4)]
        assert cols[0] > cols[1] and cols[2] > cols[3], f"structure not learnt: {cols}"


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestNICERegistry:
    @pytest.mark.parametrize(
        "name",
        [
            "nice_mnist",
            "nice_tfd",
            "nice_svhn",
            "nice_cifar",
            "nice_mnist_gen",
            "nice_tfd_gen",
            "nice_svhn_gen",
            "nice_cifar_gen",
        ],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("nice_mnist", (784, 5, 1_000, "logistic")),
            ("nice_tfd", (2_304, 4, 5_000, "gaussian")),
            ("nice_svhn", (3_072, 4, 2_000, "logistic")),
            ("nice_cifar", (3_072, 4, 2_000, "logistic")),
        ],
    )
    def test_factory_configs_match_paper_figure_3(
        self, name: str, expected: tuple[int, int, int, str]
    ) -> None:
        from lucid.models import AutoConfig

        cfg = AutoConfig.from_pretrained(name)
        assert isinstance(cfg, NICEConfig)
        dim, hidden_layers, hidden_dim, prior = expected
        assert cfg.input_dim == dim
        assert cfg.num_coupling_layers == 4
        assert cfg.num_hidden_layers == hidden_layers
        assert cfg.hidden_dim == hidden_dim
        assert cfg.prior == prior

    def test_factory_with_override(self) -> None:
        m = create_model(
            "nice_mnist",
            input_dim=16,
            num_coupling_layers=2,
            num_hidden_layers=1,
            hidden_dim=8,
        )
        assert isinstance(m, NICEModel)
        out = m.eval()(lucid.rand((1, 16)))
        assert tuple(out.latent.shape) == (1, 16)

    def test_auto_image_generation_dispatch(self) -> None:
        from lucid.models import AutoModelForImageGeneration
        from lucid.models._registry import _registry_lookup

        entry = _registry_lookup("nice_mnist_gen", task=AutoModelForImageGeneration._task)
        assert entry.model_class is NICEForImageGeneration
