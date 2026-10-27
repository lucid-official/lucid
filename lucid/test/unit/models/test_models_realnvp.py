"""Unit tests for RealNVP (Dinh et al., 2016) — multi-scale affine flow."""

import math

import pytest

import lucid
from lucid._tensor.tensor import Tensor
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.models import (
    GenerationOutput,
    NormalizingFlowOutput,
    RealNVPConfig,
    RealNVPForImageGeneration,
    RealNVPModel,
    create_model,
    is_model,
)
from lucid.models.generative.realnvp._model import _squeeze, _unsqueeze


def _tiny_cfg(**overrides: object) -> RealNVPConfig:
    """8x8 RGB, two scales — small enough to invert and differentiate."""
    base: dict[str, object] = {
        "sample_size": 8,
        "in_channels": 3,
        "out_channels": 3,
        "num_scales": 2,
        "residual_blocks": 1,
        "base_dim": 4,
    }
    base.update(overrides)
    return RealNVPConfig(**base)  # type: ignore[arg-type]


def _excite(model: RealNVPModel) -> None:
    """Push the flow off its identity initialisation.

    Every coupling starts as the identity (zero head, zero rescaling), so a
    freshly built model would satisfy any invertibility test trivially.
    """
    lucid.manual_seed(0)
    for param in model.parameters():
        if param.ndim == 4:
            nn.init.normal_(param, mean=0.0, std=0.1)
    for name, param in model.named_parameters():
        if name.endswith("rescale"):
            nn.init.constant_(param, 0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestRealNVPConfig:
    def test_defaults_match_paper_imagenet32(self) -> None:
        cfg = RealNVPConfig()
        assert cfg.sample_size == 32
        assert cfg.in_channels == 3 and cfg.out_channels == 3
        assert cfg.num_scales == 4
        assert cfg.residual_blocks == 4
        assert cfg.base_dim == 32
        assert cfg.prior == "gaussian"
        assert cfg.act_fn == "relu"
        assert cfg.use_batch_norm is True
        assert cfg.data_constraint == 0.9
        assert cfg.model_type == "realnvp"

    def test_geometry_properties(self) -> None:
        cfg = RealNVPConfig(sample_size=64, num_scales=5, residual_blocks=2)
        assert cfg.image_shape == (3, 64, 64)
        assert cfg.input_dim == 3 * 64 * 64
        # Squeeze multiplies channels by 4, the factor-out halves them.
        assert cfg.scale_channels == (3, 6, 12, 24, 48)

    def test_channel_count_must_be_preserved(self) -> None:
        with pytest.raises(ValueError, match="bijection"):
            RealNVPConfig(in_channels=3, out_channels=1)

    def test_sample_size_must_survive_every_squeeze(self) -> None:
        # 12 is not divisible by 2 ** (4 - 1) = 8.
        with pytest.raises(ValueError, match="divisible"):
            RealNVPConfig(sample_size=12, num_scales=4)

    def test_odd_final_scale_rejected(self) -> None:
        # 24 / 2 ** 3 = 3 — an odd final extent has no checkerboard.
        with pytest.raises(ValueError, match="odd"):
            RealNVPConfig(sample_size=24, num_scales=4)

    def test_data_constraint_range(self) -> None:
        with pytest.raises(ValueError, match="data_constraint"):
            RealNVPConfig(data_constraint=1.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("num_scales", 0), ("residual_blocks", 0), ("base_dim", 0)],
    )
    def test_non_positive_knobs_rejected(self, field: str, value: int) -> None:
        with pytest.raises(ValueError, match=field):
            _tiny_cfg(**{field: value})


# ─────────────────────────────────────────────────────────────────────────────
# Volume-preserving reshapes
# ─────────────────────────────────────────────────────────────────────────────


class TestSqueeze:
    def test_shape_and_exact_inverse(self) -> None:
        x = lucid.arange(0, 2 * 3 * 4 * 4, dtype=lucid.float32).reshape(2, 3, 4, 4)
        sq = _squeeze(x)
        assert tuple(sq.shape) == (2, 12, 2, 2)
        assert float((_unsqueeze(sq) - x).abs().max().item()) == 0.0

    def test_preserves_every_value(self) -> None:
        x = lucid.rand((1, 2, 4, 4))
        assert abs(float(_squeeze(x).sum().item()) - float(x.sum().item())) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# The bijection
# ─────────────────────────────────────────────────────────────────────────────


class TestRealNVPModel:
    def test_encode_shapes(self) -> None:
        cfg = _tiny_cfg()
        m = RealNVPModel(cfg).eval()
        z, log_det = m.encode(lucid.rand((3, 3, 8, 8)))
        assert tuple(z.shape) == (3, cfg.input_dim)
        assert tuple(log_det.shape) == (3,)

    def test_forward_output_contract(self) -> None:
        m = RealNVPModel(_tiny_cfg()).eval()
        out = m(lucid.rand((2, 3, 8, 8)))
        assert isinstance(out, NormalizingFlowOutput)
        assert tuple(out.latent.shape) == (2, 192)
        assert tuple(out.log_prob.shape) == (2,)
        assert out.loss is None

    @pytest.mark.parametrize("use_batch_norm", [True, False])
    def test_decode_is_exact_inverse(self, use_batch_norm: bool) -> None:
        m = RealNVPModel(_tiny_cfg(use_batch_norm=use_batch_norm)).eval()
        _excite(m)
        x = lucid.rand((3, 3, 8, 8))
        z, _ = m.encode(x)
        maxdiff = float((m.decode(z) - x).abs().max().item())
        assert maxdiff < 1e-4, f"round-trip drifted by {maxdiff}"

    def test_decode_is_exact_at_three_scales(self) -> None:
        """Factor-out bookkeeping has to survive more than one split."""
        m = RealNVPModel(_tiny_cfg(num_scales=3, sample_size=16)).eval()
        _excite(m)
        x = lucid.rand((2, 3, 16, 16))
        z, _ = m.encode(x)
        assert float((m.decode(z) - x).abs().max().item()) < 1e-4

    def test_log_det_matches_autograd_jacobian(self) -> None:
        """The reported log-determinant must be the true one.

        Differentiates the whole flow — logit stage, couplings, flow
        batch-norm, squeeze and factor-out — and compares ``slogdet`` of
        the exact Jacobian against what ``encode`` claims.
        """
        cfg = _tiny_cfg(sample_size=4)
        m = RealNVPModel(cfg).eval()
        _excite(m)
        dim = cfg.input_dim
        x = lucid.rand((1, 3, 4, 4), requires_grad=True)

        def flow(inp: lucid.Tensor) -> lucid.Tensor:
            z, _ = m.encode(inp)
            return z.reshape(dim)

        jac = lucid.autograd.jacobian(flow, x)
        assert isinstance(jac, lucid.Tensor)
        _sign, logabsdet = linalg.slogdet(jac.reshape(dim, dim))

        _z, log_det = m.encode(x)
        diff = abs(float(logabsdet.item()) - float(log_det[0].item()))
        assert diff < 1e-2, f"log-det off by {diff}"

    def test_identity_at_initialisation(self) -> None:
        """Zero head + zero rescaling ⇒ the couplings start as identities.

        Only the logit stage should move the data, which makes the first
        optimisation steps well-conditioned.
        """
        cfg = _tiny_cfg(use_batch_norm=False)
        m = RealNVPModel(cfg).eval()
        x = lucid.rand((2, 3, 8, 8))
        z, _ = m.encode(x)
        expected, _ = m.logit(x)
        # z is the flattened factor-out stack; compare against the logit
        # output routed through the same squeeze / split path.
        assert tuple(z.shape) == (2, 192)
        assert abs(float(z.abs().sum().item()) - float(expected.abs().sum().item())) < 1e-3

    def test_log_prob_decomposition(self) -> None:
        from lucid.models._utils import flow_prior_log_prob

        m = RealNVPModel(_tiny_cfg()).eval()
        x = lucid.rand((2, 3, 8, 8))
        z, log_det = m.encode(x)
        manual = flow_prior_log_prob("gaussian", z).sum(dim=-1) + log_det
        assert float((m.log_prob(x) - manual).abs().max().item()) < 1e-3

    def test_bits_per_dim_matches_log_prob(self) -> None:
        cfg = _tiny_cfg()
        m = RealNVPModel(cfg).eval()
        x = lucid.rand((2, 3, 8, 8))
        # ``log_prob`` is the density over the model's own [0, 1] input space;
        # the reported metric adds the 8-bit discretisation term so it lands
        # on the paper's Table 1 scale.
        expected = -m.log_prob(x) / (cfg.input_dim * math.log(2.0)) + cfg.num_bits
        assert float((m.bits_per_dim(x) - expected).abs().max().item()) < 1e-6

    def test_bits_per_dim_carries_the_discretisation_offset(self) -> None:
        cfg = _tiny_cfg()
        m = RealNVPModel(cfg).eval()
        x = lucid.rand((2, 3, 8, 8))
        uncorrected = -m.log_prob(x) / (cfg.input_dim * math.log(2.0))
        offset = float((m.bits_per_dim(x) - uncorrected).mean().item())
        assert abs(offset - cfg.num_bits) < 1e-6

    def test_logit_stage_inverts(self) -> None:
        m = RealNVPModel(_tiny_cfg()).eval()
        x = lucid.rand((2, 3, 8, 8))
        y, _ = m.logit(x)
        assert float((m.logit.inverse(y) - x).abs().max().item()) < 1e-5

    def test_rejects_wrong_input_shape(self) -> None:
        m = RealNVPModel(_tiny_cfg()).eval()
        with pytest.raises(ValueError, match=r"expects \(B, 3, 8, 8\)"):
            m(lucid.rand((2, 1, 8, 8)))

    def test_rejects_wrong_latent_shape(self) -> None:
        m = RealNVPModel(_tiny_cfg()).eval()
        with pytest.raises(ValueError, match=r"expects \(B, 192\) latents"):
            m.decode(lucid.rand((2, 64)))


# ─────────────────────────────────────────────────────────────────────────────
# Generation head
# ─────────────────────────────────────────────────────────────────────────────


class TestRealNVPForImageGeneration:
    def test_loss_is_mean_bits_per_dim(self) -> None:
        cfg = _tiny_cfg()
        m = RealNVPForImageGeneration(cfg).eval()
        x = lucid.rand((4, 3, 8, 8))
        out = m(x)
        assert out.loss is not None
        expected = float(
            (-out.log_prob / (cfg.input_dim * math.log(2.0)) + cfg.num_bits).mean().item()
        )
        assert abs(float(out.loss.item()) - expected) < 1e-6

    def test_generate_shape_and_range(self) -> None:
        m = RealNVPForImageGeneration(_tiny_cfg()).eval()
        out = m.generate(n_samples=3)
        assert isinstance(out, GenerationOutput)
        assert tuple(out.samples.shape) == (3, 3, 8, 8)
        # The logit squash is not saturating: samples land within
        # ±(1 - c) / 2c of [0, 1].
        slack = (1.0 - 0.9) / (2.0 * 0.9)
        assert float(out.samples.min().item()) > -slack - 1e-3
        assert float(out.samples.max().item()) < 1.0 + slack + 1e-3

    def test_generate_rejects_non_positive_temperature(self) -> None:
        m = RealNVPForImageGeneration(_tiny_cfg()).eval()
        with pytest.raises(ValueError, match="temperature"):
            m.generate(n_samples=1, temperature=0.0)

    def test_backward_reaches_every_parameter(self) -> None:
        m = RealNVPForImageGeneration(_tiny_cfg())
        out = m(lucid.rand((2, 3, 8, 8)))
        assert out.loss is not None
        out.loss.backward()
        missing = [name for name, p in m.named_parameters() if p.grad is None]
        assert not missing, f"no gradient for {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


class TestRealNVPTraining:
    def test_bits_per_dim_decreases_on_a_toy_density(self) -> None:
        """Maximum-likelihood training must actually improve the metric."""
        import lucid.optim as optim

        lucid.manual_seed(0)
        model = RealNVPForImageGeneration(_tiny_cfg(sample_size=4, base_dim=8))
        opt = optim.Adam(model.parameters(), lr=1e-3)

        def batch(n: int = 16) -> lucid.Tensor:
            # Left half bright, right half dark — structure a flow can find.
            base = lucid.tensor([[[[0.8, 0.8, 0.2, 0.2]] * 4] * 3])
            return (base.expand(n, 3, 4, 4) + lucid.randn((n, 3, 4, 4)) * 0.02).clip(0.01, 0.99)

        first: float | None = None
        for step in range(60):
            out = model(batch())
            assert out.loss is not None
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            if step == 0:
                first = float(out.loss.item())

        final = model(batch(32))
        assert final.loss is not None and first is not None
        assert (
            float(final.loss.item()) < first
        ), f"bits/dim did not improve: {first} → {float(final.loss.item())}"


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestRealNVPRegistry:
    @pytest.mark.parametrize(
        "name",
        [
            "realnvp_cifar",
            "realnvp_imagenet32",
            "realnvp_imagenet64",
            "realnvp_lsun",
            "realnvp_celeba",
            "realnvp_cifar_gen",
            "realnvp_imagenet32_gen",
            "realnvp_imagenet64_gen",
            "realnvp_lsun_gen",
            "realnvp_celeba_gen",
        ],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            # (sample_size, num_scales, residual_blocks, base_dim)
            ("realnvp_cifar", (32, 2, 8, 64)),
            ("realnvp_imagenet32", (32, 4, 4, 32)),
            ("realnvp_imagenet64", (64, 5, 2, 32)),
            ("realnvp_lsun", (64, 5, 2, 32)),
            ("realnvp_celeba", (64, 5, 2, 32)),
        ],
    )
    def test_factory_configs_match_paper_section_4_1(
        self, name: str, expected: tuple[int, int, int, int]
    ) -> None:
        from lucid.models import AutoConfig

        cfg = AutoConfig.from_pretrained(name)
        assert isinstance(cfg, RealNVPConfig)
        size, scales, blocks, dim = expected
        assert cfg.sample_size == size
        assert cfg.num_scales == scales
        assert cfg.residual_blocks == blocks
        assert cfg.base_dim == dim
        assert cfg.prior == "gaussian"

    def test_factory_with_override(self) -> None:
        m = create_model(
            "realnvp_cifar",
            sample_size=8,
            num_scales=2,
            residual_blocks=1,
            base_dim=4,
        )
        assert isinstance(m, RealNVPModel)
        out = m.eval()(lucid.rand((1, 3, 8, 8)))
        assert tuple(out.latent.shape) == (1, 192)

    def test_auto_image_generation_dispatch(self) -> None:
        from lucid.models import AutoModelForImageGeneration
        from lucid.models._registry import _registry_lookup

        entry = _registry_lookup("realnvp_cifar_gen", task=AutoModelForImageGeneration._task)
        assert entry.model_class is RealNVPForImageGeneration


class TestFlowBatchNormVariant:
    """§3.7 / Appendix E's running-average batch norm.

    On the *first* training batch the running estimates are still their
    initial ``(0, 1)``, so the paper's variant leaves the input essentially
    untouched while ordinary batch norm centres it — which makes the two
    modes trivially distinguishable without appealing to a converged model.
    """

    @staticmethod
    def _batch() -> Tensor:
        # per-channel means 11 and 22
        return lucid.tensor([[[[10.0, 12.0]], [[20.0, 24.0]]]])

    def test_ordinary_mode_centres_the_batch(self) -> None:
        from lucid.models.generative.realnvp._model import _FlowBatchNorm

        bn = _FlowBatchNorm(2)
        bn.train()
        y, _ = bn(self._batch())
        means = y.mean(dim=(0, 2, 3)).tolist()
        assert all(abs(v) < 1e-5 for v in means)

    def test_paper_variant_uses_the_running_estimates(self) -> None:
        from lucid.models.generative.realnvp._model import _FlowBatchNorm

        bn = _FlowBatchNorm(2, use_running_stats=True)
        bn.train()
        x = self._batch()
        y, _ = bn(x)
        # running stats are (0, 1) on the first batch, so y ~= x
        assert float((y - x).abs().max().item()) < 1e-2

    def test_both_modes_still_track_the_running_estimates(self) -> None:
        """The variants differ in what they *use*, not in what they track."""
        from lucid.models.generative.realnvp._model import _FlowBatchNorm

        for running in (False, True):
            bn = _FlowBatchNorm(2, use_running_stats=running)
            bn.train()
            bn(self._batch())
            got = [round(v, 4) for v in bn.running_mean.reshape(-1).tolist()]
            assert got == [1.1, 2.2]
