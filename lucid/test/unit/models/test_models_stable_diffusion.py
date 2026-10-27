"""Unit tests for Stable Diffusion (Rombach et al., 2022).

Latent diffusion is four things bolted together, and three of the joints
fail silently — the model trains, the loss falls, and the samples are
wrong. Each gets a test that a plausible mis-wiring fails, and a
companion that proves the test can fail.

**Cross-attention has a direction.** Q from the image, K/V from the
conditioning. Swap them and the shapes still line up, the loss still
falls, and the prompt is ignored.

**The latent is rescaled.** The first stage emits latents an order of
magnitude wider than the diffusion process assumes. Omit the factor and
the forward process's noise is negligible against the signal.

**``scaled_linear`` is not linear.** The schedule interpolates in
sqrt(beta). A plain ramp between the same endpoints plots almost
identically and denoises to mush.
"""

import math

import pytest

import lucid
import lucid.models as M
from lucid.nn._shadow import shadow_alloc
from lucid.models.generative.stable_diffusion import (
    AutoencoderKL,
    DDIMScheduler,
    DiagonalGaussian,
    PNDMScheduler,
    StableDiffusionConfig,
    StableDiffusionForImageGeneration,
    StableDiffusionModel,
    StableDiffusionWeights,
    UNet2DConditionModel,
)

_TINY = dict(
    sample_size=32,
    downsample_factor=4,
    vae_block_out_channels=(32, 64, 64),
    vae_layers_per_block=1,
    unet_block_out_channels=(32, 64),
    unet_layers_per_block=1,
    attention_head_dim=4,
    cross_attention_dim=16,
    context_length=4,
    norm_num_groups=32,
)


# Recorded from the reference PNDM implementation at skip_prk_steps=True,
# stepping a ones latent with a constant 0.5 epsilon for ten steps, in
# float64.  A constant epsilon makes the number a property of the sampler
# alone — no weights, no network, nothing else that could drift.
_REFERENCE_PNDM_MEAN = 4.267492704686


def _tiny(**overrides: object) -> StableDiffusionConfig:
    merged = dict(_TINY)
    merged.update(overrides)
    return StableDiffusionConfig(**merged)  # type: ignore[arg-type]


class TestConfig:
    def test_the_factor_must_match_the_autoencoder_depth(self) -> None:
        """``f`` is a consequence of the stride-2 count, not a free field."""
        with pytest.raises(ValueError, match="does not match"):
            _tiny(vae_block_out_channels=(32, 64, 64, 64))

    def test_a_latent_the_unet_cannot_halve_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot survive"):
            _tiny(sample_size=4, unet_block_out_channels=(32, 64, 64))

    def test_widths_must_divide_the_norm_groups(self) -> None:
        with pytest.raises(ValueError, match="norm_num_groups"):
            _tiny(unet_block_out_channels=(32, 65))

    def test_an_unknown_schedule_is_refused(self) -> None:
        with pytest.raises(ValueError, match="beta_schedule"):
            _tiny(beta_schedule="cosine")

    def test_attention_head_dim_is_a_count_not_a_dimension(self) -> None:
        """The released field name lies, and reading it the other way is
        free in parameters and wrong in activations.

        At 320 wide with the released value of 8 the reference builds
        eight heads of forty channels. Building forty heads of eight has
        identical shapes, identical parameter counts, and a relative
        error of 0.36 against the reference — which is how this was
        found, and why the count is asserted here rather than trusted.
        """
        from lucid.models.generative.stable_diffusion._unet import (
            _SpatialTransformer,
        )

        config = StableDiffusionConfig()
        assert config.attention_head_dim == 8
        tower = _SpatialTransformer(320, config.attention_head_dim, 768, 32)
        assert tower.blocks[0].attn1.num_heads == 8

    def test_the_defaults_are_the_released_configuration(self) -> None:
        """Read from the published unet/vae/scheduler configs, not memory."""
        config = StableDiffusionConfig()
        assert config.vae_block_out_channels == (128, 256, 512, 512)
        assert config.unet_block_out_channels == (320, 640, 1280, 1280)
        assert (config.latent_channels, config.downsample_factor) == (4, 8)
        assert (config.cross_attention_dim, config.context_length) == (768, 77)
        assert (config.beta_start, config.beta_end) == (0.00085, 0.012)
        assert config.beta_schedule == "scaled_linear"
        assert config.latent_size == 64


class TestTheLatentIsSpatial:
    """The reason this family cannot reuse ``lucid.models.VAEModel``."""

    def test_the_latent_keeps_height_and_width(self) -> None:
        config = _tiny()
        vae = AutoencoderKL(config).eval()
        out = vae(lucid.randn((2, 3, 32, 32)))
        assert out.latent.shape == (2, 4, 8, 8)
        assert out.reconstruction.shape == (2, 3, 32, 32)

    def test_the_existing_vae_flattens_instead(self) -> None:
        """Guards the test above — it would be trivially true of any
        autoencoder if the zoo's other one were already spatial."""
        from lucid.models.generative.vae import VAEConfig

        assert isinstance(VAEConfig().latent_dim, int), (
            "the zoo's VAE grew a spatial latent, so this family's "
            "separate first stage may no longer be justified"
        )


class TestThePosterior:
    def test_a_standard_normal_has_zero_divergence(self) -> None:
        zeros = lucid.zeros((2, 4, 8, 8))
        assert float(DiagonalGaussian(zeros, zeros).kl().item()) == pytest.approx(0.0)

    def test_the_divergence_grows_with_the_mean(self) -> None:
        zeros = lucid.zeros((2, 4, 8, 8))
        shifted = DiagonalGaussian(lucid.ones((2, 4, 8, 8)), zeros)
        assert float(shifted.kl().item()) > 0.0

    def test_the_mode_is_deterministic_and_the_sample_is_not(self) -> None:
        """Two call sites want different things; a tuple would hide that."""
        lucid.manual_seed(0)
        vae = AutoencoderKL(_tiny()).eval()
        x = lucid.randn((1, 3, 32, 32))
        assert (
            float((vae(x, sample=False).latent - vae(x, sample=False).latent).abs().max().item())
            == 0.0
        )
        assert (
            float((vae(x, sample=True).latent - vae(x, sample=True).latent).abs().max().item())
            > 0.0
        )


class TestCrossAttentionHasADirection:
    """The joint that fails silently.

    Q comes from the image and K/V from the conditioning. The reversed
    wiring produces a model of identical shape whose output does not
    depend on the prompt — so the test is that changing the conditioning
    changes the output, and the guard is that changing an *unrelated*
    tensor of the same shape does not.
    """

    def test_the_conditioning_reaches_the_output(self) -> None:
        lucid.manual_seed(0)
        unet = UNet2DConditionModel(_tiny()).eval()
        latent = lucid.randn((1, 4, 8, 8))
        step = lucid.tensor([10.0])
        first = unet(latent, step, lucid.randn((1, 4, 16)))
        second = unet(latent, step, lucid.randn((1, 4, 16)))
        assert float((first - second).abs().max().item()) > 1e-5, (
            "the output is unchanged by the conditioning — cross-attention "
            "is not wired, or its queries and keys are reversed"
        )

    def test_the_timestep_reaches_the_output(self) -> None:
        """A separate path; a model can carry one and drop the other."""
        lucid.manual_seed(0)
        unet = UNet2DConditionModel(_tiny()).eval()
        latent = lucid.randn((1, 4, 8, 8))
        context = lucid.randn((1, 4, 16))
        early = unet(latent, lucid.tensor([5.0]), context)
        late = unet(latent, lucid.tensor([900.0]), context)
        assert float((early - late).abs().max().item()) > 1e-5

    def test_a_context_of_the_wrong_width_is_refused(self) -> None:
        unet = UNet2DConditionModel(_tiny()).eval()
        with pytest.raises(ValueError, match="cross_attention_dim"):
            unet(lucid.randn((1, 4, 8, 8)), lucid.tensor([1.0]), lucid.randn((1, 4, 99)))

    def test_the_conditioning_may_be_any_length(self) -> None:
        """Cross-attention does not constrain the sequence length, and a
        model that reshaped instead of attending would fail here."""
        unet = UNet2DConditionModel(_tiny()).eval()
        for length in (1, 4, 13):
            out = unet(
                lucid.randn((1, 4, 8, 8)),
                lucid.tensor([1.0]),
                lucid.randn((1, length, 16)),
            )
            assert out.shape == (1, 4, 8, 8)


class TestTheSchedule:
    def test_scaled_linear_is_not_linear(self) -> None:
        """Same endpoints, different curve — and the released models use
        the squared one."""
        scaled = DDIMScheduler(StableDiffusionConfig())
        linear = DDIMScheduler(StableDiffusionConfig(beta_schedule="linear"))
        assert float((scaled.betas - linear.betas).abs().max().item()) > 1e-4

    def test_the_endpoints_agree(self) -> None:
        """Guards the test above: the two differ in the middle, not at the
        ends, which is exactly why a plot does not catch the mistake."""
        scaled = DDIMScheduler(StableDiffusionConfig())
        linear = DDIMScheduler(StableDiffusionConfig(beta_schedule="linear"))
        for index in (0, -1):
            assert float(scaled.betas[index].item()) == pytest.approx(
                float(linear.betas[index].item()), abs=1e-6
            )

    def test_alphas_decrease_to_almost_nothing(self) -> None:
        scheduler = DDIMScheduler(StableDiffusionConfig())
        assert float(scheduler.alphas_cumprod[0].item()) > 0.99
        assert float(scheduler.alphas_cumprod[-1].item()) < 0.01

    def test_timesteps_descend_and_are_counted(self) -> None:
        scheduler = DDIMScheduler(StableDiffusionConfig())
        steps = scheduler.timesteps(50)
        assert len(steps) == 50
        assert steps == sorted(steps, reverse=True)

    def test_the_trajectory_carries_the_released_offset(self) -> None:
        """``steps_offset`` shifts every visited time by one.

        The per-step arithmetic is identical with or without it, so a
        trajectory built on the wrong times takes correct steps to a
        different image. Measured against the reference: the offset is
        the difference between a relative error of 7.4e-01 and 5.2e-07
        over ten steps.
        """
        steps = DDIMScheduler(StableDiffusionConfig()).timesteps(10)
        assert steps[:3] == [901, 801, 701]
        assert steps[-1] == 1

    def test_dropping_the_offset_would_change_the_times(self) -> None:
        """Guards the test above."""
        plain = DDIMScheduler(StableDiffusionConfig(steps_offset=0))
        assert plain.timesteps(10)[:3] == [900, 800, 700]

    def test_the_final_step_bootstraps_from_alpha_zero(self) -> None:
        """``set_alpha_to_one`` is false in the release, so the step past
        the end uses ``alphas_cumprod[0]`` — near one, and not one."""
        config = StableDiffusionConfig()
        assert config.set_alpha_to_one is False
        scheduler = DDIMScheduler(config)
        latent = lucid.zeros((1, 4, 8, 8))
        noise = lucid.ones((1, 4, 8, 8))
        released = scheduler.step(noise, 1, -1, latent)
        one = DDIMScheduler(StableDiffusionConfig(set_alpha_to_one=True)).step(noise, 1, -1, latent)
        assert float((released - one).abs().max().item()) > 1e-6

    def test_more_steps_than_the_schedule_is_refused(self) -> None:
        with pytest.raises(ValueError, match="num_inference_steps"):
            DDIMScheduler(StableDiffusionConfig()).timesteps(2000)

    def test_eta_zero_is_deterministic(self) -> None:
        """The property a reproducible seed rests on."""
        lucid.manual_seed(0)
        scheduler = DDIMScheduler(StableDiffusionConfig())
        latent, noise = lucid.randn((1, 4, 8, 8)), lucid.randn((1, 4, 8, 8))
        first = scheduler.step(noise, 999, 979, latent)
        second = scheduler.step(noise, 999, 979, latent)
        assert float((first - second).abs().max().item()) == 0.0

    def test_eta_one_is_not(self) -> None:
        """Guards the test above — a scheduler that never adds noise would
        pass it while silently ignoring eta."""
        lucid.manual_seed(0)
        scheduler = DDIMScheduler(StableDiffusionConfig())
        latent, noise = lucid.randn((1, 4, 8, 8)), lucid.randn((1, 4, 8, 8))
        first = scheduler.step(noise, 999, 979, latent, eta=1.0)
        second = scheduler.step(noise, 999, 979, latent, eta=1.0)
        assert float((first - second).abs().max().item()) > 0.0

    def test_eta_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="eta"):
            DDIMScheduler(StableDiffusionConfig()).step(
                lucid.zeros((1, 4, 8, 8)), 999, 979, lucid.zeros((1, 4, 8, 8)), eta=2.0
            )


class TestGuidance:
    def test_scale_one_is_the_conditional_prediction(self) -> None:
        r"""The identity :math:`\epsilon_\varnothing + 1\cdot(\epsilon_c -
        \epsilon_\varnothing) = \epsilon_c`.

        Worth pinning because a guidance implementation that ignores its
        scale still produces images — this is the only cheap check that
        the scale is read at all.
        """
        lucid.manual_seed(0)
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        context = lucid.randn((1, 4, 16))
        start = lucid.zeros((1, 4, 8, 8))
        guided = model.generate(
            context, lucid.randn((1, 4, 16)), 3, guidance_scale=1.0, latent=start
        ).samples
        plain = model.generate(context, None, 3, latent=start).samples
        assert float((guided - plain).abs().max().item()) < 1e-5

    def test_a_larger_scale_changes_the_result(self) -> None:
        """Guards the test above — the scale has to be read, not just accepted.

        The threshold is deliberately far above the noise floor. An
        implementation that accepts the scale and multiplies by one
        instead leaves a residue of about 1e-5 here, which is rounding,
        not guidance; the genuine separation is about 2. A bound tucked
        just over the rounding passes the moment anything perturbs the
        arithmetic — batching the two conditionings into one forward is
        enough to do it — so it is set two orders above instead.

        The second comparison keeps both samples on the guided path, so
        it is not distinguishing a branch that was taken from one that
        was skipped.
        """
        lucid.manual_seed(0)
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        context, uncond = lucid.randn((1, 4, 16)), lucid.randn((1, 4, 16))
        start = lucid.zeros((1, 4, 8, 8))
        one = model.generate(context, uncond, 3, guidance_scale=1.0, latent=start).samples
        two = model.generate(context, uncond, 3, guidance_scale=2.0, latent=start).samples
        seven = model.generate(context, uncond, 3, guidance_scale=7.5, latent=start).samples
        assert float((one - seven).abs().max().item()) > 1e-2
        assert float((two - seven).abs().max().item()) > 1e-2

    def test_the_conditional_half_is_the_prompt(self) -> None:
        r"""Which half of the stacked batch is which.

        Guidance packs the two conditionings into one batch and slices
        the result apart, so the halves can be swapped without any shape
        objecting. The swap yields
        :math:`\epsilon_c + s(\epsilon_\varnothing - \epsilon_c)` — a
        reflection of the intended point, which still denoises to an
        image and still moves further as :math:`s` grows. What separates
        them is the limit: as :math:`s \to 1` the correct form collapses
        onto the *conditional* prediction and the swap onto the
        unconditional one. Exactly one skips the guided branch, so this
        probes just beside it.
        """
        lucid.manual_seed(0)
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        context, uncond = lucid.randn((1, 4, 16)), lucid.randn((1, 4, 16))
        start = lucid.zeros((1, 4, 8, 8))

        guided = model.generate(context, uncond, 3, guidance_scale=1.0 + 1e-4, latent=start).samples
        conditional = model.generate(context, None, 3, latent=start).samples
        unconditional = model.generate(uncond, None, 3, latent=start).samples

        to_conditional = float((guided - conditional).abs().max().item())
        to_unconditional = float((guided - unconditional).abs().max().item())

        # The guard: if the tiny model happened to predict the same thing
        # from both sequences, everything below is trivially satisfied.
        assert to_unconditional > 1e-3, "the two conditionings are not distinguishable"
        assert to_conditional < 1e-3, f"guided is {to_conditional} from conditional"

    def test_a_batch_is_guided_row_by_row(self) -> None:
        """The split lands between the two conditionings, not inside a row.

        The halves are ``[:batch]`` and ``[batch:]``; a split hardcoded
        at one, or off by a row, still returns the right shape and still
        produces an image — it just guides each row with a neighbour's
        unconditional sequence. Only a batch above one can see it, so
        two rows carrying different prompts are compared against
        generating each of them alone.
        """
        lucid.manual_seed(0)
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        context = lucid.randn((2, 4, 16))
        uncond = lucid.randn((2, 4, 16))
        start = lucid.randn((2, 4, 8, 8))

        together = model.generate(context, uncond, 3, guidance_scale=7.5, latent=start).samples
        for row in range(2):
            alone = model.generate(
                context[row : row + 1],
                uncond[row : row + 1],
                3,
                guidance_scale=7.5,
                latent=start[row : row + 1],
            ).samples
            gap = float((together[row : row + 1] - alone).abs().max().item())
            assert gap < 1e-4, f"row {row} differs by {gap}"

    def test_a_negative_scale_is_refused(self) -> None:
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        with pytest.raises(ValueError, match="guidance_scale"):
            model.generate(lucid.randn((1, 4, 16)), guidance_scale=-1.0)

    def test_a_mismatched_unconditional_is_refused(self) -> None:
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        with pytest.raises(ValueError, match="uncond_context"):
            model.generate(lucid.randn((1, 4, 16)), lucid.randn((1, 9, 16)))


class TestTheAssembledModel:
    def test_a_training_step_produces_a_loss_with_gradient(self) -> None:
        lucid.manual_seed(0)
        model = StableDiffusionModel(_tiny())
        out = model(lucid.randn((2, 3, 32, 32)), lucid.randn((2, 4, 16)), return_loss=True)
        assert out.noise_pred.shape == (2, 4, 8, 8)
        assert out.loss is not None
        out.loss.backward()
        total = sum(
            float(p.grad.abs().sum().item()) for p in model.parameters() if p.grad is not None
        )
        assert total > 0.0

    def test_the_latent_is_rescaled_on_the_way_in_and_out(self) -> None:
        """Encode-then-decode must round-trip through the same factor.

        The scale is invisible in shapes and fatal if applied once: the
        diffusion process assumes unit-ish variance, and the first stage
        emits something far wider.
        """
        lucid.manual_seed(0)
        model = StableDiffusionModel(_tiny()).eval()
        images = lucid.randn((1, 3, 32, 32))
        scaled = model.encode_image(images, sample=False)
        raw = model.vae.encode(images).mode()
        ratio = float((scaled / raw).abs().mean().item())
        assert ratio == pytest.approx(
            0.18215, rel=1e-4
        ), f"the latent scale is {ratio}, not the released 0.18215"

    def test_generation_returns_an_image(self) -> None:
        lucid.manual_seed(0)
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        out = model.generate(lucid.randn((2, 4, 16)), num_inference_steps=2)
        assert isinstance(out, M.GenerationOutput)
        assert out.samples.shape == (2, 3, 32, 32)

    def test_the_task_forward_is_a_training_step(self) -> None:
        """Calling the generation model trains it, as in every other
        image-generation family; sampling is :meth:`generate` alone.

        It once aliased ``generate`` and returned pixels, so a training
        loop written against the family contract got images and no loss.
        """
        lucid.manual_seed(0)
        model = StableDiffusionForImageGeneration(_tiny())
        before = model.stable_diffusion.unet.conv_in.weight.detach().clone()
        out = model(lucid.randn((2, 3, 32, 32)), lucid.randn((2, 4, 16)))
        assert isinstance(out, M.ModelOutput)
        assert out.noise_pred.shape == (2, 4, 8, 8)
        assert out.loss is not None and out.loss.shape == ()
        optimizer = lucid.optim.SGD(model.parameters(), lr=1.0)
        out.loss.backward()
        optimizer.step()
        after = model.stable_diffusion.unet.conv_in.weight
        assert float((after - before).abs().max().item()) > 0.0

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    def test_it_runs_on_both_devices(self, device: str) -> None:
        """Two families this session shipped a CPU-only index tensor; this
        is the check that would have caught either."""
        lucid.manual_seed(0)
        model = StableDiffusionModel(_tiny())
        model = model.metal() if device == "metal" else model
        out = model(
            lucid.randn((1, 3, 32, 32), device=device),
            lucid.randn((1, 4, 16), device=device),
            return_loss=True,
        )
        assert str(out.noise_pred.device) == f"device('{device}')"
        assert out.loss is not None


class TestTheSamplingDefaultsAreTheReleases:
    """Three numbers that decide what an unqualified `generate()` produces.

    The config's fields are covered by ``test_generative_provenance``,
    which fails the suite for any field without a recorded source. These
    are not config fields — they are :meth:`generate`'s own defaults —
    so nothing was holding them to the released pipeline's, and a caller
    who writes ``model.generate(prompt_embedding)`` gets whatever they
    happen to be.

    They are the released ``StableDiffusionPipeline.__call__``'s:
    fifty steps, guidance 7.5, and eta 0 for the deterministic DDIM
    trajectory that makes a seed reproduce an image.
    """

    @staticmethod
    def _defaults(fn: object) -> dict[str, object]:
        import inspect

        return {
            name: parameter.default for name, parameter in inspect.signature(fn).parameters.items()
        }

    def test_generate_matches_the_released_pipeline(self) -> None:
        defaults = self._defaults(StableDiffusionForImageGeneration.generate)
        assert defaults["num_inference_steps"] == 50
        assert defaults["guidance_scale"] == 7.5
        assert defaults["eta"] == 0.0

    def test_the_sampler_default_is_pndm(self) -> None:
        """The release's `model_index.json` names PNDMScheduler.

        The paper samples with DDIM, so this one is a choice between two
        defensible answers and the released pipeline is the tiebreak.
        """
        defaults = self._defaults(StableDiffusionForImageGeneration.generate)
        assert defaults["sampler"] == "pndm"

    def test_ddim_defaults_to_the_deterministic_trajectory(self) -> None:
        """eta = 0 is what makes a seed reproduce an image."""
        assert self._defaults(DDIMScheduler.step)["eta"] == 0.0


class TestVariants:
    """Structure and registration, built without weights.

    Every assertion below is about shapes, names and counts, none of
    which needs storage — so the models are built under
    :func:`~lucid.nn._shadow.shadow_alloc`. At full size this family is
    943M parameters, about three gigabytes a copy, and this file used to
    build six of them to count things that ``shadow_alloc`` counts
    exactly. It was the largest single contributor to a suite that
    cannot finish in one process.
    """

    def test_v1_matches_clips_text_width(self) -> None:
        """The conditioning contract, checked against the actual tower."""
        with shadow_alloc():
            sd = M.stable_diffusion().config
            clip = M.clip_vit_large_14().config
        assert sd.cross_attention_dim == clip.text_width == 768
        assert sd.context_length == clip.context_length == 77

    def test_only_the_citable_architecture_is_registered(self) -> None:
        """v1.1-v1.5 are weight tags, not variants, and v2's published
        configuration could not be read from the primary source — so one
        architecture is registered rather than five or two."""
        names = M.list_models(family="stable_diffusion")
        assert sorted(names) == [
            "stable_diffusion",
            "stable_diffusion_gen",
        ]

    def test_v1_declares_weights(self) -> None:
        """The registry entry, checked offline against the model it loads
        into — a stale ``num_params`` is invisible until a user
        downloads three and a half gigabytes."""
        entry = StableDiffusionWeights.DEFAULT.value
        with shadow_alloc():
            model = M.stable_diffusion()
            total = sum(math.prod(tuple(int(s) for s in p.shape)) for p in model.parameters())
        assert entry.meta["num_params"] == total
        assert entry.url.endswith("CompVis_LAION/model.safetensors")
        assert len(entry.sha256) == 64

    def test_the_sampler_takes_the_same_weights(self) -> None:
        """``stable_diffusion_gen(pretrained=True)`` has somewhere to load.

        It is the factory that produces images, so it is the one a
        caller most wants weighted — and it loads the published entry
        into the model it holds. Its docstring said the opposite for one
        release, which is the sort of claim only a test reaches: the
        code was right and nobody reading the documentation would have
        tried it.
        """
        with shadow_alloc():
            sampler = M.stable_diffusion_gen()
            inner = {name for name, _ in sampler.stable_diffusion.named_parameters()}
            direct = {name for name, _ in M.stable_diffusion().named_parameters()}
            assert inner == direct

            entry = StableDiffusionWeights.DEFAULT.value
            total = sum(
                math.prod(tuple(int(s) for s in p.shape))
                for p in sampler.stable_diffusion.parameters()
            )
        assert entry.meta["num_params"] == total


class TestTimestepEmbedding:
    """The U-Net's step features come from the shared ``timestep_embedding``."""

    def test_it_orders_cosine_before_sine(self) -> None:
        """``flip_sin_to_cos`` in the released configuration.

        The two orders train identically from scratch and are mutually
        unreadable for a checkpoint, so the released one is built and
        pinned here rather than left to whichever the transformer paper
        used.
        """
        from lucid.models.generative._common._transformers import (
            timestep_embedding,
        )

        emb = timestep_embedding(lucid.tensor([0.0]), 8)
        # At t = 0 every argument is 0, so cos gives 1 and sin gives 0.
        assert [round(v, 6) for v in emb[0, :4].tolist()] == [1.0, 1.0, 1.0, 1.0]
        assert [round(v, 6) for v in emb[0, 4:].tolist()] == [0.0, 0.0, 0.0, 0.0]

    def test_an_odd_width_is_refused(self) -> None:
        """The first U-Net width is the embedding's, split into two halves."""
        from lucid.models.generative.stable_diffusion._config import (
            StableDiffusionConfig,
        )

        with pytest.raises(ValueError, match="even"):
            StableDiffusionConfig(
                unet_block_out_channels=(7, 14),
                norm_num_groups=1,
                attention_head_dim=7,
            )

    def test_distinct_timesteps_get_distinct_embeddings(self) -> None:
        from lucid.models.generative._common._transformers import (
            timestep_embedding,
        )

        emb = timestep_embedding(lucid.tensor([0.0, 1.0, 500.0]), 16)
        assert float((emb[0] - emb[1]).abs().max().item()) > 1e-6
        assert float((emb[1] - emb[2]).abs().max().item()) > 1e-6
        assert math.isfinite(float(emb.abs().max().item()))


class TestItMatchesTheReleasedCheckpoint:
    """Parameter counts against the published archives, tensor for tensor.

    This is the check that found three architecture defects at once, and
    none of them was visible in a forward pass:

    * the feed-forward was ``Linear -> GELU -> Linear`` where the
      released blocks are GEGLU — 49,536,640 parameters short across
      sixteen transformer blocks;
    * the attention carried a bias on q/k/v, which the release does not,
      and lost the one on the output projection, which it does —
      24,960 either way;
    * the autoencoder's attention had a projection after
      ``MultiheadAttention``'s own output projection — 525,312 surplus
      across its two blocks.

    Every one of them trains, denoises and samples. The count is the
    only cheap thing that disagrees.
    """

    @staticmethod
    def _count(module: object) -> int:
        return sum(
            math.prod(tuple(int(s) for s in p.shape))
            for p in module.parameters()  # type: ignore[attr-defined]
        )

    def test_the_unet_has_the_published_size(self) -> None:
        with shadow_alloc():
            assert self._count(M.stable_diffusion().unet) == 859_520_964

    def test_the_autoencoder_has_the_published_size(self) -> None:
        with shadow_alloc():
            assert self._count(M.stable_diffusion().vae) == 83_653_863

    def test_the_feed_forward_is_gated(self) -> None:
        """GEGLU projects to twice the inner width and gates with half.

        A plain ``Linear`` of the same inner width is the wrong size by a
        factor the count above catches, but this states the reason so a
        future edit does not "simplify" it back.
        """
        from lucid.models.generative.stable_diffusion._unet import _GEGLU

        layer = _GEGLU(8, 16)
        assert tuple(layer.proj.weight.shape) == (32, 8)
        assert layer(lucid.randn((2, 8))).shape == (2, 16)

    def test_attention_has_no_input_bias_and_one_output_bias(self) -> None:
        """The released layout: ``to_q.weight`` with no ``to_q.bias``,
        and ``to_out.0.bias`` present."""
        from lucid.models.generative.stable_diffusion._unet import _TransformerBlock

        block = _TransformerBlock(32, 4, 16)
        names = {name for name, _ in block.named_parameters()}
        assert "attn1.in_proj_bias" not in names
        assert "attn1.out_proj_bias" not in names
        assert "attn1_out_bias" in names and "attn2_out_bias" in names


class TestTheNormalisationEpsilonIsTheReleasedOne:
    """The released stages do not share one epsilon, and the difference shows.

    Three values are in play and only one of them is the framework
    default. The U-Net's residual blocks and its output norm run at
    ``norm_eps`` = 1e-5; the norm *inside* every spatial transformer
    runs at 1e-6; and the whole autoencoder runs at 1e-6.

    Taking the default everywhere costs 1.1e-04 on the autoencoder's
    decoder against the same weights — the size of a float32
    accumulation over a network this deep, which is exactly why it
    survived a conversion that was checked to "round-off". A count
    cannot see it, a shape cannot see it, and a parity threshold set
    from the wrong model agrees with itself.
    """

    @staticmethod
    def _epsilons(module: object) -> set[float]:
        return {
            float(m.eps)
            for m in module.modules()  # type: ignore[attr-defined]
            if isinstance(m, lucid.nn.GroupNorm)
        }

    def test_every_autoencoder_norm_runs_at_one_e_minus_six(self) -> None:
        vae = AutoencoderKL(_tiny())
        assert self._epsilons(vae) == {1e-6}

    def test_the_unet_transformer_norm_is_tighter_than_its_resnets(self) -> None:
        from lucid.models.generative.stable_diffusion._unet import (
            _ResBlock,
            _SpatialTransformer,
        )

        assert self._epsilons(_SpatialTransformer(32, 4, 16, 32)) == {1e-6}
        assert self._epsilons(_ResBlock(32, 32, 16, 32)) == {1e-5}

    def test_the_unet_carries_both_values(self) -> None:
        """Not one or the other — the released U-Net has both."""
        unet = UNet2DConditionModel(_tiny())
        assert self._epsilons(unet) == {1e-5, 1e-6}

    def test_the_two_epsilons_are_distinguishable(self) -> None:
        """The guard that keeps the three above from being vacuous.

        If 1e-5 and 1e-6 produced the same activations there would be
        nothing to pin. They do not: the same weights and the same input
        under the two values disagree by more than float32 round-off.
        """
        vae = AutoencoderKL(_tiny()).eval()
        image = lucid.randn((1, 3, 32, 32))
        reference = vae.encode(image).mode()

        for norm in vae.modules():
            if isinstance(norm, lucid.nn.GroupNorm):
                norm.eps = 1e-5
        shifted = vae.encode(image).mode()

        scale = float(lucid.abs(reference).max().item())
        error = float(lucid.abs(shifted - reference).max().item()) / scale
        assert error > 1e-7


class TestPNDMIsWhatTheReleaseShips:
    """The paper samples with DDIM; the released pipeline's
    ``model_index.json`` names ``PNDMScheduler``.  Both are here, and both
    were checked against the reference implementation — these tests pin
    the parts where PNDM is not simply "DDIM with better coefficients"."""

    def test_the_trajectory_repeats_its_second_step(self) -> None:
        """The multistep rule needs a history it does not have on the first
        call, so the opener evaluates one interval twice.  The reference
        builds this by slicing the *ascending* array, which puts the repeat
        second in descending order — not second to last."""
        steps = PNDMScheduler(StableDiffusionConfig()).timesteps(10)
        assert steps == [901, 801, 801, 701, 601, 501, 401, 301, 201, 101, 1]

    def test_one_more_step_runs_than_was_asked_for(self) -> None:
        """A caller asking for 50 gets 51 evaluations.  Worth stating: it
        is the difference between a benchmark that matches the reference's
        cost and one that looks 2 % faster for free."""
        for count in (10, 20, 50):
            assert len(PNDMScheduler(StableDiffusionConfig()).timesteps(count)) == (count + 1)

    def test_the_repeat_starts_from_the_same_sample_twice(self) -> None:
        """Both halves of the opener step from the latent the first one
        began at.  Feeding the second the output of the first advances the
        trajectory by an extra interval, which no shape check can see."""
        scheduler = PNDMScheduler(StableDiffusionConfig())
        steps = scheduler.timesteps(10)
        start = lucid.randn((1, 4, 8, 8))

        scheduler.step(lucid.ones_like(start), steps[0], start)
        assert scheduler.cur_sample is not None
        assert float((scheduler.cur_sample - start).abs().max().item()) == 0.0

    def test_the_second_step_consumes_the_saved_sample(self) -> None:
        scheduler = PNDMScheduler(StableDiffusionConfig())
        steps = scheduler.timesteps(10)
        latent = lucid.randn((1, 4, 8, 8))

        first = scheduler.step(lucid.ones_like(latent), steps[0], latent)
        scheduler.step(lucid.ones_like(latent), steps[1], first)
        assert scheduler.cur_sample is None

    def test_asking_for_a_trajectory_clears_the_history(self) -> None:
        """The derivatives are state, and the second sample must not open
        with the first one's.  They are the right shape and the wrong
        numbers, so nothing downstream would notice."""
        scheduler = PNDMScheduler(StableDiffusionConfig())
        steps = scheduler.timesteps(10)
        latent = lucid.randn((1, 4, 8, 8))

        for step in steps[:4]:
            latent = scheduler.step(lucid.ones_like(latent), step, latent)
        assert scheduler.ets and scheduler.counter == 4

        scheduler.timesteps(10)
        assert scheduler.ets == [] and scheduler.counter == 0

    def test_a_reused_scheduler_reproduces_itself(self) -> None:
        """The guard above, stated as the property that matters: two
        samples run back to back through one scheduler agree."""
        scheduler = PNDMScheduler(StableDiffusionConfig())

        def sample() -> float:
            latent = lucid.ones((1, 4, 8, 8))
            for step in scheduler.timesteps(10):
                latent = scheduler.step(lucid.ones_like(latent) * 0.5, step, latent)
            return float(latent.mean().item())

        assert sample() == pytest.approx(sample(), rel=1e-6)

    def test_stepping_out_of_order_is_refused(self) -> None:
        """The order of the correction is chosen by position in the
        trajectory, not by the timestep, so a caller that skips or repeats
        one gets a plausible tensor computed by the wrong rule."""
        scheduler = PNDMScheduler(StableDiffusionConfig())
        steps = scheduler.timesteps(10)
        latent = lucid.randn((1, 4, 8, 8))

        with pytest.raises(RuntimeError, match="position"):
            scheduler.step(lucid.ones_like(latent), steps[3], latent)

    def test_running_past_the_end_is_refused(self) -> None:
        scheduler = PNDMScheduler(StableDiffusionConfig())
        steps = scheduler.timesteps(10)
        latent = lucid.ones((1, 4, 8, 8))
        for step in steps:
            latent = scheduler.step(lucid.ones_like(latent) * 0.5, step, latent)

        with pytest.raises(RuntimeError, match="all of them have been taken"):
            scheduler.step(lucid.ones_like(latent), steps[-1], latent)

    def test_step_needs_the_timestep_list_first(self) -> None:
        """PNDM's stride cannot be recovered from a timestep, because the
        trajectory repeats one."""
        scheduler = PNDMScheduler(StableDiffusionConfig())
        latent = lucid.randn((1, 4, 8, 8))

        with pytest.raises(RuntimeError, match="timesteps"):
            scheduler.step(lucid.ones_like(latent), 901, latent)

    def test_the_history_stays_bounded(self) -> None:
        """Fourth order needs four derivatives and keeps no more — an
        unbounded list would hold every activation of a 50-step sample."""
        scheduler = PNDMScheduler(StableDiffusionConfig())
        steps = scheduler.timesteps(10)
        latent = lucid.randn((1, 4, 8, 8))

        for step in steps:
            latent = scheduler.step(lucid.ones_like(latent), step, latent)
        assert len(scheduler.ets) <= 4

    def test_it_reproduces_the_reference_trajectory(self) -> None:
        """Recorded from the reference implementation at
        ``skip_prk_steps=True``, on a constant epsilon so the numbers depend
        on the sampler and nothing else.  This is the test that caught the
        opener stepping from the wrong sample: it was off by 8.7e-01 while
        every shape and every timestep was already correct."""
        scheduler = PNDMScheduler(StableDiffusionConfig())
        steps = scheduler.timesteps(10)
        latent = lucid.ones((1, 4, 8, 8))
        for step in steps:
            latent = scheduler.step(lucid.ones_like(latent) * 0.5, step, latent)

        assert float(latent.mean().item()) == pytest.approx(_REFERENCE_PNDM_MEAN, rel=1e-5)

    def test_generation_defaults_to_pndm(self) -> None:
        """The default has to be the released pipeline's, or prompts tuned
        against published samples land somewhere else."""
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        context = lucid.randn((1, 4, 16))

        latent = lucid.randn((1, 4, 8, 8))
        default = model.generate(
            context, num_inference_steps=3, guidance_scale=1.0, latent=latent
        ).samples
        pndm = model.generate(
            context,
            num_inference_steps=3,
            guidance_scale=1.0,
            latent=latent,
            sampler="pndm",
        ).samples
        assert float((default - pndm).abs().max().item()) == 0.0

    def test_ddim_remains_reachable(self) -> None:
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        context = lucid.randn((1, 4, 16))

        latent = lucid.randn((1, 4, 8, 8))
        pndm = model.generate(
            context,
            num_inference_steps=3,
            guidance_scale=1.0,
            latent=latent,
            sampler="pndm",
        ).samples
        ddim = model.generate(
            context,
            num_inference_steps=3,
            guidance_scale=1.0,
            latent=latent,
            sampler="ddim",
        ).samples
        assert pndm.shape == ddim.shape
        assert float((pndm - ddim).abs().max().item()) > 0.0

    def test_an_unknown_sampler_is_rejected(self) -> None:
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        context = lucid.randn((1, 4, 16))

        with pytest.raises(ValueError, match="pndm"):
            model.generate(context, num_inference_steps=2, sampler="dpm")

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    def test_sampling_runs_on_both_devices(self, device: str) -> None:
        """PNDM carries its derivative history across steps in a Python
        list, which makes those tensors the ones most likely to be left
        on the wrong device.  A single forward pass cannot show it — the
        history is empty on the first step."""
        lucid.manual_seed(0)
        model = StableDiffusionForImageGeneration(_tiny()).eval()
        model = model.metal() if device == "metal" else model

        image = model.generate(
            lucid.randn((1, 4, 16), device=device),
            num_inference_steps=3,
            guidance_scale=1.0,
            latent=lucid.randn((1, 4, 8, 8), device=device),
        ).samples
        assert str(image.device) == f"device('{device}')"
        assert image.shape == (1, 3, 32, 32)
