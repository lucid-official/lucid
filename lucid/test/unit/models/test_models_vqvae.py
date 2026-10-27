"""Unit tests for VQ-VAE (van den Oord et al., 2017).

The gradient tests are the load-bearing ones.  Quantisation is a hard
``argmin``, so the only thing making this architecture trainable is the
straight-through estimator, and the only thing training the codebook is
the separate codebook term.  Both are invisible in a shape check and
would break silently — a model that still runs, still returns the right
shapes, and never learns.
"""

import pytest

import lucid
from lucid.models._utils._generative import generative_activation
from lucid.models import (
    GenerationOutput,
    VQVAEConfig,
    VQVAEForImageGeneration,
    VQVAEModel,
    create_model,
    is_model,
)
from lucid.models.generative.vqvae import VQVAEOutput


def _tiny_cfg(**overrides: object) -> VQVAEConfig:
    base: dict[str, object] = {
        "sample_size": 16,
        "in_channels": 3,
        "out_channels": 3,
        "num_embeddings": 32,
        "embedding_dim": 8,
        "hidden_channels": 16,
        "residual_hidden_channels": 16,
    }
    base.update(overrides)
    return VQVAEConfig(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestVQVAEConfig:
    def test_defaults_match_paper(self) -> None:
        cfg = VQVAEConfig()
        assert cfg.num_embeddings == 512
        assert cfg.embedding_dim == 256
        assert cfg.hidden_channels == 256
        assert cfg.num_downsample_layers == 2
        assert cfg.num_residual_layers == 2
        assert cfg.commitment_cost == 0.25
        assert cfg.act_fn == "relu"
        assert cfg.recon_loss == "mse"
        assert cfg.model_type == "vqvae"

    def test_latent_grid_size(self) -> None:
        assert VQVAEConfig(sample_size=32).latent_grid_size == (8, 8)
        assert VQVAEConfig(sample_size=128).latent_grid_size == (32, 32)
        assert VQVAEConfig(sample_size=(32, 64)).latent_grid_size == (8, 16)

    def test_sample_size_must_divide(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            VQVAEConfig(sample_size=30)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("num_embeddings", 0),
            ("embedding_dim", 0),
            ("hidden_channels", 0),
            ("num_downsample_layers", 0),
            ("residual_hidden_channels", 0),
            ("commitment_cost", -0.1),
        ],
    )
    def test_rejects_invalid_fields(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field):
            VQVAEConfig(**{field: value})  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        cfg = VQVAEConfig()
        with pytest.raises(Exception):
            cfg.num_embeddings = 4  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Forward shapes
# ─────────────────────────────────────────────────────────────────────────────


class TestVQVAEForward:
    def test_model_forward_shapes(self) -> None:
        model = VQVAEModel(_tiny_cfg()).eval()
        out = model(lucid.randn((2, 3, 16, 16)))

        assert isinstance(out, VQVAEOutput)
        assert out.sample.shape == (2, 3, 16, 16)
        assert out.latent.shape == (2, 8, 4, 4)
        assert out.indices.shape == (2, 4, 4)
        assert out.perplexity.shape == ()

    def test_bare_model_reports_no_total_loss(self) -> None:
        # The trunk exposes the two codebook terms but does not build an
        # objective — that is the task wrapper's job.
        out = VQVAEModel(_tiny_cfg()).eval()(lucid.randn((1, 3, 16, 16)))
        assert out.loss is None
        assert out.recon_loss is None
        assert out.codebook_loss is not None
        assert out.commitment_loss is not None

    def test_indices_are_valid_codebook_entries(self) -> None:
        cfg = _tiny_cfg()
        out = VQVAEModel(cfg).eval()(lucid.randn((2, 3, 16, 16)))
        assert out.indices.dtype == lucid.int64
        assert int(out.indices.min().item()) >= 0
        assert int(out.indices.max().item()) < cfg.num_embeddings

    def test_perplexity_within_codebook_bounds(self) -> None:
        cfg = _tiny_cfg()
        out = VQVAEModel(cfg).eval()(lucid.randn((4, 3, 16, 16)))
        perplexity = float(out.perplexity.item())
        assert 1.0 <= perplexity <= float(cfg.num_embeddings) + 1e-4

    def test_head_forward_populates_every_loss(self) -> None:
        model = VQVAEForImageGeneration(_tiny_cfg()).eval()
        out = model(lucid.randn((2, 3, 16, 16)))

        for term in (out.loss, out.recon_loss, out.codebook_loss, out.commitment_loss):
            assert term is not None
            assert term.shape == ()

    def test_total_loss_is_the_weighted_sum(self) -> None:
        cfg = _tiny_cfg(commitment_cost=0.25)
        out = VQVAEForImageGeneration(cfg).eval()(lucid.randn((2, 3, 16, 16)))

        assert out.loss is not None
        assert out.recon_loss is not None
        assert out.codebook_loss is not None
        assert out.commitment_loss is not None
        expected = (
            float(out.recon_loss.item())
            + float(out.codebook_loss.item())
            + 0.25 * float(out.commitment_loss.item())
        )
        assert abs(float(out.loss.item()) - expected) < 1e-4

    def test_non_square_and_deeper_stacks(self) -> None:
        cfg = _tiny_cfg(sample_size=(16, 32), num_downsample_layers=3)
        out = VQVAEModel(cfg).eval()(lucid.randn((1, 3, 16, 32)))
        assert out.sample.shape == (1, 3, 16, 32)
        assert out.indices.shape == (1, 2, 4)

    def test_zero_residual_layers(self) -> None:
        out = VQVAEModel(_tiny_cfg(num_residual_layers=0)).eval()(lucid.randn((1, 3, 16, 16)))
        assert out.sample.shape == (1, 3, 16, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Quantisation semantics
# ─────────────────────────────────────────────────────────────────────────────


class TestQuantisation:
    def test_straight_through_is_value_identical_to_hard_quantisation(self) -> None:
        # The estimator may only change the *backward* pass.  If it altered
        # the forward value, the decoder would be reading something that is
        # not a codebook entry.
        model = VQVAEModel(_tiny_cfg()).eval()
        z_e = model.encode(lucid.randn((2, 3, 16, 16)))
        q = model.quantize(z_e)
        # ``lookup`` returns trailing-axis layout; ``quantize`` re-permutes
        # to image layout, so compare in the same one.
        hard = model.quantizer.lookup(q.indices).permute(0, 3, 1, 2)

        assert float((q.quantized - hard).abs().max().item()) < 1e-6

    def test_quantize_preserves_image_layout(self) -> None:
        cfg = _tiny_cfg()
        model = VQVAEModel(cfg).eval()
        q = model.quantize(model.encode(lucid.randn((2, 3, 16, 16))))

        assert q.quantized.shape == (2, cfg.embedding_dim, 4, 4)
        assert q.indices.shape == (2, 4, 4)

    def test_assignment_is_nearest_neighbour(self) -> None:
        # Feed the codebook's own entries back in: each must select itself.
        cfg = _tiny_cfg()
        model = VQVAEModel(cfg).eval()
        codes = model.quantizer.weight  # (K, D)
        # Lay the first 4 entries out as a 2x2 field in trailing-axis layout.
        field = codes[:4].reshape(1, 2, 2, cfg.embedding_dim)
        picked = model.quantizer.assign(field).reshape(-1)

        assert [int(picked[i].item()) for i in range(4)] == [0, 1, 2, 3]
        assert cfg.num_embeddings >= 4

    def test_tokeniser_round_trip_matches_forward(self) -> None:
        model = VQVAEModel(_tiny_cfg()).eval()
        x = lucid.randn((2, 3, 16, 16))

        direct = model(x).sample
        round_trip = model.decode_indices(model.encode_indices(x))
        assert float((direct - round_trip).abs().max().item()) < 1e-5

    def test_encode_indices_shape(self) -> None:
        model = VQVAEModel(_tiny_cfg()).eval()
        assert model.encode_indices(lucid.randn((3, 3, 16, 16))).shape == (3, 4, 4)
        assert model.latent_grid_size == (4, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient routing — the part that makes the architecture trainable
# ─────────────────────────────────────────────────────────────────────────────


class TestGradientRouting:
    def test_reconstruction_alone_reaches_the_encoder(self) -> None:
        # argmin has zero gradient everywhere, so without the straight-through
        # estimator the encoder would never receive a reconstruction signal.
        model = VQVAEModel(_tiny_cfg())
        x = lucid.randn((2, 3, 16, 16))
        ((model(x).sample - x) ** 2).mean().backward()

        grad = model.encoder.down_blocks[0].weight.grad
        assert grad is not None
        assert float(abs(grad).sum().item()) > 0.0

    def test_reconstruction_alone_does_not_reach_the_codebook(self) -> None:
        # This is why the codebook term exists at all: the straight-through
        # path routes the decoder's gradient *past* the codebook.
        model = VQVAEModel(_tiny_cfg())
        x = lucid.randn((2, 3, 16, 16))
        ((model(x).sample - x) ** 2).mean().backward()

        assert model.quantizer.weight.grad is None

    def test_codebook_term_trains_the_codebook_only(self) -> None:
        model = VQVAEModel(_tiny_cfg())
        out = model(lucid.randn((2, 3, 16, 16)))
        assert out.codebook_loss is not None
        out.codebook_loss.backward()

        codebook_grad = model.quantizer.weight.grad
        assert codebook_grad is not None
        assert float(abs(codebook_grad).sum().item()) > 0.0
        # sg[z_e] detaches the encoder side of the codebook term.
        assert model.encoder.down_blocks[0].weight.grad is None

    def test_commitment_term_trains_the_encoder_only(self) -> None:
        model = VQVAEModel(_tiny_cfg())
        out = model(lucid.randn((2, 3, 16, 16)))
        assert out.commitment_loss is not None
        out.commitment_loss.backward()

        encoder_grad = model.encoder.down_blocks[0].weight.grad
        assert encoder_grad is not None
        assert float(abs(encoder_grad).sum().item()) > 0.0
        # sg[e] detaches the codebook side of the commitment term.
        assert model.quantizer.weight.grad is None

    def test_full_objective_reaches_every_parameter_group(self) -> None:
        model = VQVAEForImageGeneration(_tiny_cfg())
        out = model(lucid.randn((2, 3, 16, 16)))
        assert out.loss is not None
        out.loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} received no gradient"


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction likelihoods and sampling
# ─────────────────────────────────────────────────────────────────────────────


class TestLikelihoodsAndSampling:
    def test_bce_output_lives_in_the_input_space(self) -> None:
        model = VQVAEForImageGeneration(_tiny_cfg(recon_loss="bce")).eval()
        out = model(lucid.rand((2, 3, 16, 16)))

        assert float(out.sample.min().item()) >= 0.0
        assert float(out.sample.max().item()) <= 1.0

    def test_mse_output_is_unsquashed(self) -> None:
        model = VQVAEForImageGeneration(_tiny_cfg(recon_loss="mse")).eval()
        out = model(lucid.randn((2, 3, 16, 16)))
        assert out.sample.shape == (2, 3, 16, 16)

    def test_generate_samples_the_uniform_codebook_prior(self) -> None:
        model = VQVAEForImageGeneration(_tiny_cfg()).eval()
        gen = model.generate(3)

        assert isinstance(gen, GenerationOutput)
        assert gen.samples.shape == (3, 3, 16, 16)

    def test_generate_under_bce_is_squashed_like_forward(self) -> None:
        model = VQVAEForImageGeneration(_tiny_cfg(recon_loss="bce")).eval()
        samples = model.generate(2).samples

        assert float(samples.min().item()) >= 0.0
        assert float(samples.max().item()) <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    @pytest.mark.parametrize("name", ["vqvae", "vqvae_gen"])
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_create_model_defaults(self) -> None:
        model = create_model("vqvae")
        assert isinstance(model, VQVAEModel)
        assert model.config.num_embeddings == 512
        assert model.latent_grid_size == (8, 8)

    def test_create_model_head(self) -> None:
        model = create_model("vqvae_gen")
        assert isinstance(model, VQVAEForImageGeneration)

    def test_config_override_through_registry(self) -> None:
        model = create_model("vqvae", num_embeddings=64, embedding_dim=32)
        assert model.config.num_embeddings == 64
        assert model.num_embeddings == 64
        assert model.config.embedding_dim == 32

    @pytest.mark.parametrize("name", ["vqvae", "vqvae_gen"])
    def test_pretrained_is_refused_rather_than_faked(self, name: str) -> None:
        # Silently returning random weights under ``pretrained=True`` is a
        # worse failure than raising: it surfaces as poor accuracy, not an
        # error.
        with pytest.raises(NotImplementedError, match="No pretrained weights"):
            create_model(name, pretrained=True)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder composition
#
# Shape checks cannot see an activation applied twice, and at the default
# ReLU neither can the numbers — it is idempotent.  These pin the exact
# composition instead, which is what caught the encoder activating once in
# its downsample loop and again at the head of the pre-activation residual
# block that follows.
# ─────────────────────────────────────────────────────────────────────────────


class TestEncoderComposition:
    @pytest.mark.parametrize("act", ["relu", "silu", "gelu"])
    def test_no_residuals_applies_the_activation_once(self, act: str) -> None:
        cfg = _tiny_cfg(num_downsample_layers=1, num_residual_layers=0, act_fn=act)
        enc = VQVAEModel(cfg).eval().encoder
        x = lucid.randn((2, 3, 16, 16))

        got = enc(x)
        want = enc.proj(generative_activation(act, enc.down_blocks[0](x)))
        assert float((got - want).abs().max().item()) < 1e-5

    @pytest.mark.parametrize("act", ["relu", "silu", "gelu"])
    def test_residual_block_supplies_the_boundary_activation(self, act: str) -> None:
        cfg = _tiny_cfg(num_downsample_layers=1, num_residual_layers=1, act_fn=act)
        enc = VQVAEModel(cfg).eval().encoder
        x = lucid.randn((2, 3, 16, 16))

        # The block opens with an activation of its own, so nothing may
        # activate between the convolution and it.
        got = enc(x)
        inner = enc.residuals[0](enc.down_blocks[0](x))
        want = enc.proj(generative_activation(act, inner))
        assert float((got - want).abs().max().item()) < 1e-5

    def test_decoder_does_not_activate_before_its_residual_stack(self) -> None:
        cfg = _tiny_cfg(num_downsample_layers=1, num_residual_layers=1, act_fn="silu")
        dec = VQVAEModel(cfg).eval().decoder
        z = lucid.randn((2, cfg.embedding_dim, 8, 8))

        got = dec(z)
        h = generative_activation("silu", dec.residuals[0](dec.lift(z)))
        want = dec.up_blocks[0](h)
        assert float((got - want).abs().max().item()) < 1e-5
