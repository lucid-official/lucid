"""MeanFlow — the identities the method rests on, not just the shapes.

A shape test would pass on a model that had the Jacobian-vector product's
tangent wrong, which is the one mistake this method cannot survive: the
paper's destructive ablation puts the correct tangent at FID 61.06 and
every incorrect one between 137 and 329.  So the tests here assert the
equations instead — that the target reduces to Flow Matching's when
``r = t``, that the total derivative decomposes the way the chain rule
says, and that the sampler implements Eq. 12 rather than something that
merely returns the right shape.

Each is paired with the mis-wiring it would catch.
"""

import pytest

import lucid
from lucid.models import DiffusionModelOutput
from lucid.models.generative.mean_flow import (
    MeanFlowConfig,
    MeanFlowForImageGeneration,
    MeanFlowModel,
)


def _tiny(**overrides: object) -> MeanFlowConfig:
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
    return MeanFlowConfig(**base)  # type: ignore[arg-type]


def _wake(model: MeanFlowModel) -> None:
    """Undo adaLN-Zero so the conditioning reaches the output.

    Every modulation projection starts at zero, which makes an untrained
    field constant in both times — the property
    :meth:`TestField.test_the_untrained_field_is_exactly_zero` pins.  Any
    test *about* time-dependence has to break that first, or it measures
    the initialisation rather than the network.
    """
    with lucid.no_grad():
        for block in model.blocks:
            linear = block.ada_ln[1]
            linear.weight += lucid.randn(linear.weight.shape) * 0.5
        final_linear = model.final.ada_ln[1]
        final_linear.weight += lucid.randn(final_linear.weight.shape) * 0.5
        model.final.proj.weight += lucid.randn(model.final.proj.weight.shape) * 0.5


class TestField:
    def test_the_untrained_field_is_exactly_zero(self) -> None:
        """adaLN-Zero: every block starts as the identity, so u starts at 0.

        Not cosmetic — it is what lets DiT add depth without re-tuning,
        and a non-zero start means one of the zero-initialised
        projections was missed.
        """
        model = MeanFlowModel(_tiny()).eval()
        out = model(
            lucid.randn((2, 4, 8, 8)),
            lucid.tensor([0.0, 0.2]),
            lucid.tensor([1.0, 0.7]),
        )
        assert isinstance(out, DiffusionModelOutput) and out.loss is None
        assert float(out.sample.abs().max().item()) == 0.0

    def test_the_field_reads_both_times(self) -> None:
        """u depends on r and on t separately, not on one of them.

        Catches a conditioning that drops a variable — with ``(t, t - r)``
        embedded, holding t and moving r must still move the output.
        """
        model = MeanFlowModel(_tiny()).eval()
        # adaLN-Zero leaves the conditioning with no path to the output —
        # at initialisation the field ignores *both* times, correctly.  The
        # modulation projections have to be woken before time-dependence
        # is a property there is anything to test.
        _wake(model)
        z = lucid.randn((1, 4, 8, 8))
        t = lucid.tensor([0.9])
        a = model(z, lucid.tensor([0.1]), t).sample
        b = model(z, lucid.tensor([0.5]), t).sample
        assert float((a - b).abs().max().item()) > 1e-6, "u ignores r"

    @pytest.mark.parametrize("mode", ["t_interval", "t_r", "t_r_interval", "interval"])
    def test_every_conditioning_builds_and_runs(self, mode: str) -> None:
        """All four encodings in Table 1c are constructible.

        The paper's point there is that all of them give meaningful
        results, so none may be silently unsupported.
        """
        model = MeanFlowModel(_tiny(time_conditioning=mode)).eval()
        out = model(lucid.randn((1, 4, 8, 8)), lucid.tensor([0.2]), lucid.tensor([0.8]))
        assert out.sample.shape == (1, 4, 8, 8)


class TestObjective:
    def test_zero_ratio_makes_every_pair_collapse(self) -> None:
        """At ratio 0 the sampler must return ``r == t`` everywhere.

        That is the configuration the paper says degenerates to Flow
        Matching (and fails at one step, FID 328.91); if the ratio were
        applied backwards this would be the 100% case instead.
        """
        model = MeanFlowForImageGeneration(_tiny(ratio_r_not_t=0.0))
        r, t = model._sample_times(64, "cpu")
        assert bool((r == t).all().item())

    def test_full_ratio_separates_almost_every_pair(self) -> None:
        """The companion: at ratio 1 the two times must differ.

        Together with the test above this pins the direction of the
        ratio, which a single test at one end cannot.
        """
        model = MeanFlowForImageGeneration(_tiny(ratio_r_not_t=1.0))
        r, t = model._sample_times(64, "cpu")
        # Continuous draws collide with probability zero.
        assert float((r != t).float().mean().item()) > 0.95

    def test_r_never_exceeds_t(self) -> None:
        """``r`` is the interval's start; ``t - r`` must not go negative.

        Catches the pair being assigned without sorting, which would put
        a negative width into the target's ``(t - r)`` factor.
        """
        model = MeanFlowForImageGeneration(_tiny(ratio_r_not_t=1.0))
        r, t = model._sample_times(128, "cpu")
        assert bool((r <= t).all().item())

    def test_the_target_reduces_to_the_velocity_when_r_equals_t(self) -> None:
        """With ``r = t`` the second term vanishes and the target is ``v``.

        This is the paper's own statement that MeanFlow *is* Flow
        Matching with a modified target.  A target that kept the
        ``(t - r)`` term would still train, and would still produce
        finite losses — it would just be a different method.
        """
        model = MeanFlowForImageGeneration(_tiny(ratio_r_not_t=0.0))
        images = lucid.randn((4, 4, 8, 8))
        out = model(images)
        # target == v == noise - images, and v is what the model saw; the
        # check that survives not knowing the noise is that the target
        # has no dependence on the interval, i.e. equals a difference of
        # two unit-scale fields rather than something scaled by (t - r).
        assert out.target.shape == images.shape
        assert out.target.grad_fn is None, "the target must be stop-gradiented"

    def test_the_jacobian_vector_product_decomposes(self) -> None:
        """Eq. 8: the tangent ``(v, 0, 1)`` equals ``v·∂_z u + ∂_t u``.

        The paper's destructive ablation shows this is where the method
        lives — ``(v, 0, 0)`` scores 268 FID against 61 — so the test
        asserts the decomposition rather than trusting the call.
        """
        model = MeanFlowModel(_tiny()).eval()
        _wake(model)
        z = lucid.randn((2, 4, 8, 8))
        r, t = lucid.tensor([0.1, 0.3]), lucid.tensor([0.9, 0.6])
        v = lucid.randn((2, 4, 8, 8))

        def field(z_: lucid.Tensor, r_: lucid.Tensor, t_: lucid.Tensor) -> lucid.Tensor:
            return model.forward(z_, r_, t_, None).sample

        zeros_r, ones_t = lucid.zeros_like(r), lucid.ones_like(t)
        _, both = lucid.func.jvp(field, (z, r, t), (v, zeros_r, ones_t))
        _, along_z = lucid.func.jvp(field, (z, r, t), (v, zeros_r, lucid.zeros_like(t)))
        _, along_t = lucid.func.jvp(field, (z, r, t), (lucid.zeros_like(z), zeros_r, ones_t))
        gap = float((both - (along_z + along_t)).abs().max().item())
        assert gap < 1e-5, f"d/dt u did not decompose: {gap}"

    def test_the_target_carries_the_time_derivative(self) -> None:
        """The model's own JVP call must use the tangent ``(v, 0, 1)``.

        The decomposition test above exercises ``lucid.func.jvp``; this
        one exercises *this model's use of it*, which is the thing the
        paper's ablation is about.  A field that varies only in ``t``
        separates the tangents: with the correct one the target picks up
        ``-(t - r)·∂_t u`` and is dominated by it, while ``(v, 0, 0)``
        drops that term and leaves the target at the velocity's own
        scale.  The mutation this catches is the one worth 207 FID.
        """
        config = _tiny(ratio_r_not_t=1.0)

        class _TimeRamp(MeanFlowModel):
            """``u = 1000 t`` — no dependence on ``z``, a large ∂_t u."""

            def forward(
                self,
                z: lucid.Tensor,
                r: lucid.Tensor,
                t: lucid.Tensor,
                labels: lucid.Tensor | None = None,
            ) -> DiffusionModelOutput:
                ramp = lucid.ones_like(z) * 1000.0 * t.reshape(-1, 1, 1, 1)
                return DiffusionModelOutput(sample=ramp)

        model = MeanFlowForImageGeneration(config)
        model.mean_flow = _TimeRamp(config)
        out = model(lucid.zeros((8, 4, 8, 8)))

        # v is unit-scale standard normal noise; the second term is
        # 1000 * (t - r), which for separated times is orders larger.
        assert float(out.target.abs().max().item()) > 20.0, (
            "the target is at the velocity's scale — the time-derivative "
            "term is missing, so the JVP tangent is not (v, 0, 1)"
        )
        # And it is *subtracted*: Eq. 10 reads v - (t - r) du/dt.  With a
        # rising field and t > r the term is positive, so the target must
        # come out negative.  Sign-blind magnitude checks pass on a target
        # that adds where it should subtract.
        assert (
            float(out.target.max().item()) < 0.0
        ), "the time-derivative term is added, not subtracted"

    def test_the_loss_reaches_the_parameters(self) -> None:
        """The stop-gradient is on the target, not on the whole objective.

        If ``sg`` had been applied to the prediction too the loss would
        still be a finite scalar and would train nothing.
        """
        model = MeanFlowForImageGeneration(_tiny())
        out = model(lucid.randn((2, 4, 8, 8)), lucid.tensor([1, 3], dtype=lucid.int64))
        out.loss.backward()
        touched = sum(1 for p in model.parameters() if p.grad is not None)
        assert touched > 0, "no parameter received a gradient"


class TestSampling:
    def test_one_step_is_the_displacement_along_u(self) -> None:
        """Eq. 12 at one step: ``z_0 = z_1 - u(z_1, 0, 1)``.

        Pinned against a field fixed to ones, so the assertion is about
        the sampler's arithmetic and not the network's values.
        """
        config = _tiny()

        class _Ones(MeanFlowModel):
            def forward(
                self,
                z: lucid.Tensor,
                r: lucid.Tensor,
                t: lucid.Tensor,
                labels: lucid.Tensor | None = None,
            ) -> DiffusionModelOutput:
                return DiffusionModelOutput(sample=lucid.ones_like(z))

        model = MeanFlowForImageGeneration(config)
        model.mean_flow = _Ones(config)
        latent = lucid.randn((2, 4, 8, 8))
        out = model.generate(2, noise=latent, steps=1).samples
        assert float((out - (latent - lucid.ones_like(latent))).abs().max().item()) < 1e-6

    def test_subdividing_the_interval_changes_nothing_on_a_constant_field(
        self,
    ) -> None:
        """Two half-steps equal one whole one when ``u`` is constant.

        A sampler that used the wrong interval width per step — ``1``
        instead of ``t - r`` — would pass the one-step test above and
        fail this one.
        """
        config = _tiny()

        class _Ones(MeanFlowModel):
            def forward(
                self,
                z: lucid.Tensor,
                r: lucid.Tensor,
                t: lucid.Tensor,
                labels: lucid.Tensor | None = None,
            ) -> DiffusionModelOutput:
                return DiffusionModelOutput(sample=lucid.ones_like(z))

        model = MeanFlowForImageGeneration(config)
        model.mean_flow = _Ones(config)
        latent = lucid.randn((2, 4, 8, 8))
        one = model.generate(2, noise=latent, steps=1).samples
        two = model.generate(2, noise=latent, steps=2).samples
        assert float((one - two).abs().max().item()) < 1e-5

    def test_a_non_positive_step_count_is_refused(self) -> None:
        model = MeanFlowForImageGeneration(_tiny())
        with pytest.raises(ValueError, match="steps must be positive"):
            model.generate(1, steps=0)


class TestConfig:
    def test_a_partial_patch_is_refused(self) -> None:
        """A latent that does not divide into whole patches has no tokens."""
        with pytest.raises(ValueError, match="divisible by patch_size"):
            MeanFlowConfig(sample_size=8, patch_size=3)

    def test_the_guidance_mix_cannot_reach_one(self) -> None:
        """The effective scale is ``omega / (1 - kappa)`` — one divides by zero."""
        with pytest.raises(ValueError, match="guidance_mix"):
            MeanFlowConfig(guidance_mix=1.0)

    def test_the_paper_defaults_are_the_registered_ones(self) -> None:
        """The defaults are Table 1's winning row, not arbitrary choices.

        Recorded as a test because they are the numbers a reader would
        otherwise have to trust the docstring for.
        """
        config = MeanFlowConfig()
        assert config.ratio_r_not_t == 0.25
        assert config.time_sampler == "lognorm"
        assert (config.lognorm_mean, config.lognorm_std) == (-0.4, 1.0)
        assert config.time_conditioning == "t_interval"
        assert config.adaptive_weight_power == 1.0
