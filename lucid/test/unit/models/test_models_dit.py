"""DiT — the claims the paper makes, asserted rather than assumed.

The paper's two findings are architectural, and both are checkable here
without training anything: adaLN-Zero starts every block as the identity,
and shrinking the patch buys tokens without buying parameters.  A shape
test sees neither.

Each test names the mis-wiring it catches.
"""

import math

import pytest

import lucid
from lucid.models.generative.dit import (
    DiTConfig,
    DiTForImageGeneration,
    DiTModel,
)


def _tiny(**overrides: object) -> DiTConfig:
    """A model small enough to instantiate per test."""
    base = dict(
        sample_size=8,
        patch_size=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
    )
    base.update(overrides)
    return DiTConfig(**base)  # type: ignore[arg-type]


def _params(model: lucid.nn.Module) -> int:
    return sum(math.prod(p.shape) for p in model.parameters())


class TestBackbone:
    def test_adaln_zero_starts_at_the_identity(self) -> None:
        """The whole point of the "Zero": an untrained model predicts zero.

        The paper adopts adaLN-Zero over plain adaLN on this alone —
        "initialization is also important" — so a non-zero start means a
        projection was missed and the variant is adaLN in disguise.
        """
        model = DiTModel(_tiny(conditioning="adaln_zero")).eval()
        out = model(
            lucid.randn((2, 4, 8, 8)),
            lucid.tensor([10.0, 500.0]),
            lucid.tensor([1, 3], dtype=lucid.int64),
        )
        assert float(out.sample.abs().max().item()) == 0.0

    def test_the_gate_is_what_separates_adaln_from_adaln_zero(self) -> None:
        """Only the gated block is the identity at initialisation.

        The model's *output* is zero either way — the paper zero-inits
        the final linear layer for every variant — so the distinction has
        to be read at the block: adaLN-Zero leaves the token stream
        untouched, plain adaLN transforms it.  A build that gave both
        designs the gate, or zeroed both, would pass every shape test and
        make the paper's comparison unreproducible.
        """
        from lucid.models.generative._common._transformers import DiTBlock

        tokens = lucid.randn((2, 9, 32))
        cond = lucid.randn((2, 32))

        gated = DiTBlock(32, 4, gated=True).eval()
        gated.zero_conditioning()
        moved = float((gated(tokens, cond) - tokens).abs().max().item())
        assert moved == 0.0, "a zeroed gated block is not the identity"

        ungated = DiTBlock(32, 4, gated=False).eval()
        ungated.zero_conditioning()
        moved = float((ungated(tokens, cond) - tokens).abs().max().item())
        assert moved > 0.0, "an ungated block cannot be zeroed into the identity"

    def test_the_ungated_block_regresses_two_fewer_vectors(self) -> None:
        """Four modulation vectors instead of six.

        Plain adaLN regresses a shift and a scale per sub-layer;
        adaLN-Zero adds the gate.  Counted here because a `gated` flag
        that changed only the arithmetic and not the projection would
        leave two vectors' worth of parameters trained on nothing.
        """
        from lucid.models.generative._common._transformers import DiTBlock

        gated = _params(DiTBlock(32, 4, gated=True))
        ungated = _params(DiTBlock(32, 4, gated=False))
        assert gated - ungated == 2 * 32 * 32 + 2 * 32

    def test_the_timestep_sinusoid_is_256_wide_not_the_model_width(self) -> None:
        """``frequency_embedding_size`` is fixed, and that is load-bearing.

        DiT's reference implementation expands the timestep to 256
        regardless of how wide the model is.  Reusing ``hidden_size``
        instead — which is what this family did until the published
        weights were ported — adds ``(hidden_size - 256) * hidden_size``
        parameters to every variant and makes the checkpoints refuse to
        load.  At XL/2 that was 1,032,192 parameters and a shape error.
        """
        config = _tiny(hidden_size=64)
        model = DiTModel(config)
        assert config.frequency_embedding_size == 256
        first = model.time_mlp[0]
        assert tuple(first.weight.shape) == (64, 256)

    def test_the_frequency_ladder_divides_by_half(self) -> None:
        """ADM's convention, which DiT inherits — not DDPM's.

        The two differ in the denominator: ``half`` here against
        ``half - 1`` for DDPM and for the ``diffusers`` export of DiT's
        own checkpoints.  It moves the embedding by as much as 0.8, and
        nothing about a forward pass or a load would report it, so the
        ladder is pinned against its closed form.
        """
        import math

        from lucid.models.generative._common._transformers import (
            timestep_embedding,
        )

        got = timestep_embedding(lucid.tensor([300.0]), 8)
        half = 4
        freqs = [math.exp(-math.log(10000.0) * k / half) for k in range(half)]
        want = [math.cos(300.0 * f) for f in freqs]
        want += [math.sin(300.0 * f) for f in freqs]
        for index, value in enumerate(want):
            assert abs(float(got[0, index].item()) - value) < 1e-5, f"slot {index}"

    def test_the_position_table_leads_with_the_fast_axis(self) -> None:
        """Which half comes first is not a free choice.

        The reference builds its halves from ``meshgrid(w, h)``, so the
        leading half is the one that changes between adjacent tokens.
        Swapping them leaves every shape and every parameter count intact
        and quietly corrupts any ported checkpoint — this asserts the
        order directly, without needing the reference to hand.
        """
        from lucid.models.generative._common._transformers import (
            sincos_position_embedding,
        )

        side, dim = 4, 16
        table = sincos_position_embedding(dim, side)
        first, second = table[0, 0], table[0, 1]
        moved = float((first[: dim // 2] - second[: dim // 2]).abs().max().item())
        assert moved > 0.0, "the leading half must change between adjacent tokens"
        held = float((first[dim // 2 :] - second[dim // 2 :]).abs().max().item())
        assert held == 0.0, "the trailing half must be constant along a row"

    @pytest.mark.parametrize("mode", ["adaln_zero", "adaln", "cross_attention", "in_context"])
    def test_every_conditioning_design_builds_and_runs(self, mode: str) -> None:
        """All four of Section 3.2's designs are constructible.

        The comparison between them is the paper's main architectural
        result; a family that shipped only the winner could not reproduce
        the experiment that chose it.
        """
        model = DiTModel(_tiny(conditioning=mode)).eval()
        out = model(
            lucid.randn((1, 4, 8, 8)),
            lucid.tensor([100.0]),
            lucid.tensor([2], dtype=lucid.int64),
        )
        assert out.sample.shape == (1, 8, 8, 8)

    def test_in_context_conditioning_strips_its_extra_tokens(self) -> None:
        """The two conditioning tokens must not reach the decoder.

        In-context appends them to the sequence and the paper removes
        them "after the final block".  Leaving them in would decode two
        extra patches, so the output would not fit the latent — caught
        here by the shape the model must still produce.
        """
        model = DiTModel(_tiny(conditioning="in_context")).eval()
        out = model(lucid.randn((1, 4, 8, 8)), lucid.tensor([50.0]))
        assert out.sample.shape == (1, 8, 8, 8)

    def test_the_decoder_emits_two_channels_per_input_channel(self) -> None:
        """A noise *and* a diagonal covariance, per ADM's parameterisation."""
        model = DiTModel(_tiny()).eval()
        out = model(lucid.randn((1, 4, 8, 8)), lucid.tensor([1.0]))
        assert out.sample.shape[1] == 8 == 2 * 4

    def test_a_fixed_variance_model_emits_one(self) -> None:
        """With ``learn_sigma`` off the decoder is back to ``C`` channels."""
        model = DiTModel(_tiny(learn_sigma=False, out_channels=4)).eval()
        out = model(lucid.randn((1, 4, 8, 8)), lucid.tensor([1.0]))
        assert out.sample.shape[1] == 4


class TestScaling:
    def test_shrinking_the_patch_buys_tokens_not_parameters(self) -> None:
        """The paper's central claim, as an assertion.

        "As model size is held constant and patch size is decreased, the
        transformer's total parameters are effectively unchanged
        (actually, total parameters slightly *decrease*), and only Gflops
        are increased."  Sixteen times the tokens must not cost a
        materially larger model — if it does, the patch is being applied
        somewhere it should not be.
        """
        small_patch = DiTModel(_tiny(patch_size=2))
        large_patch = DiTModel(_tiny(patch_size=4))
        assert small_patch.config.num_patches == 4 * large_patch.config.num_patches
        ratio = _params(small_patch) / _params(large_patch)
        assert 0.9 < ratio < 1.1, f"parameters moved by {ratio:.2f}x with the patch"

    def test_the_paper_table_shapes(self) -> None:
        """Table 1, recorded so a config edit cannot quietly rewrite it."""
        from lucid.models import dit_base_2, dit_large_2, dit_small_2, dit_xlarge_2

        assert (dit_small_2().config.depth, dit_small_2().config.hidden_size) == (
            12,
            384,
        )
        assert (dit_base_2().config.depth, dit_base_2().config.hidden_size) == (12, 768)
        assert (dit_large_2().config.depth, dit_large_2().config.hidden_size) == (
            24,
            1024,
        )
        assert (dit_xlarge_2().config.depth, dit_xlarge_2().config.hidden_size) == (
            28,
            1152,
        )


class TestObjective:
    def test_the_loss_reaches_the_parameters(self) -> None:
        model = DiTForImageGeneration(_tiny())
        out = model(lucid.randn((2, 4, 8, 8)), lucid.tensor([1, 3], dtype=lucid.int64))
        out.loss.backward()
        assert sum(1 for p in model.parameters() if p.grad is not None) > 0

    def test_the_covariance_is_reported_beside_the_noise(self) -> None:
        """Both halves of the decoder's output reach the caller.

        The simple objective trains only the noise, so a build that threw
        the covariance away would still train and still report a falling
        loss — and would leave ADM's full objective unreachable.
        """
        model = DiTForImageGeneration(_tiny())
        out = model(lucid.randn((2, 4, 8, 8)))
        assert out.noise_pred.shape == (2, 4, 8, 8)
        assert out.variance_pred is not None
        assert out.variance_pred.shape == (2, 4, 8, 8)

    def test_a_fixed_variance_model_reports_none(self) -> None:
        model = DiTForImageGeneration(_tiny(learn_sigma=False, out_channels=4))
        out = model(lucid.randn((2, 4, 8, 8)))
        assert out.variance_pred is None


class TestSampling:
    def test_sampling_returns_a_latent(self) -> None:
        model = DiTForImageGeneration(_tiny()).eval()
        assert model.generate(2, steps=3).samples.shape == (2, 4, 8, 8)

    def test_more_steps_is_a_different_trajectory(self) -> None:
        """The step count must actually change the schedule walked.

        A sampler that ignored ``steps`` — looping a fixed number of
        times, or reusing one timestep — would return the right shape
        from both calls.
        """
        model = DiTForImageGeneration(_tiny()).eval()
        start = lucid.randn((2, 4, 8, 8))
        few = model.generate(2, steps=2, noise=start).samples
        many = model.generate(2, steps=8, noise=start).samples
        assert float((few - many).abs().max().item()) > 0.0

    def test_eta_reaches_the_protocol_the_paper_reports(self) -> None:
        """The paper's FID is DDPM's sampler, not DDIM's.

        DiT follows ADM and reports FID over 250 *DDPM* steps.  The
        default here is deterministic — this family takes no seed, so a
        sampler that drew on every call could not be reproduced — which
        means the paper's own protocol has to stay reachable.  A build
        that hard-coded ``eta = 0`` would sample fine and quietly report
        numbers that do not compare to the paper's.
        """
        model = DiTForImageGeneration(_tiny()).eval()
        start = lucid.randn((2, 4, 8, 8))

        once = model.generate(2, steps=4, noise=start).samples
        twice = model.generate(2, steps=4, noise=start).samples
        assert float((once - twice).abs().max().item()) == 0.0, "eta=0 must be exact"

        stochastic = model.generate(2, steps=4, eta=1.0, noise=start).samples
        assert float((once - stochastic).abs().max().item()) > 0.0
        again = model.generate(2, steps=4, eta=1.0, noise=start).samples
        assert float((stochastic - again).abs().max().item()) > 0.0, "eta=1 is a draw"
        assert bool(stochastic.isfinite().all().item())

    def test_eta_outside_the_unit_interval_is_refused(self) -> None:
        model = DiTForImageGeneration(_tiny())
        with pytest.raises(ValueError, match="eta interpolates"):
            model.generate(1, steps=1, eta=1.5)

    def test_a_non_positive_step_count_is_refused(self) -> None:
        model = DiTForImageGeneration(_tiny())
        with pytest.raises(ValueError, match="steps must be positive"):
            model.generate(1, steps=0)


class TestConfig:
    def test_a_partial_patch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="divisible by patch_size"):
            DiTConfig(sample_size=8, patch_size=3)

    def test_the_covariance_channels_must_agree_with_learn_sigma(self) -> None:
        """``out_channels`` and ``learn_sigma`` describe the same decision.

        Changing one without the other is the mistake this catches, and
        it would otherwise surface as a reshape error inside the decoder.
        """
        with pytest.raises(ValueError, match="out_channels must be"):
            DiTConfig(in_channels=4, out_channels=4, learn_sigma=True)
        with pytest.raises(ValueError, match="out_channels must be"):
            DiTConfig(in_channels=4, out_channels=8, learn_sigma=False)

    def test_the_diffusion_defaults_are_adm_s(self) -> None:
        """A linear schedule over 1000 steps from 1e-4 to 2e-2.

        The paper retains ADM's hyperparameters wholesale and says so; the
        numbers are recorded here rather than left to the base class,
        because a change there would silently move this family too.
        """
        config = DiTConfig()
        assert config.num_train_timesteps == 1000
        assert config.beta_schedule == "linear"
        assert (config.beta_start, config.beta_end) == (1e-4, 2e-2)
