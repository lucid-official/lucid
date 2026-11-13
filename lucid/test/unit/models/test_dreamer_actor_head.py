"""The policy and reward heads DreamerV3 shares with its predecessor, and where they part.

``Actor`` is one class serving DreamerV2 and DreamerV3, and the two released
heads are not the same head.  DreamerV2 reads a box action as a *truncated*
Gaussian around ``tanh(raw)`` with ``std = 2 * sigmoid(raw / 2) + min_std``.
DreamerV3's release (``embodied/jax/heads.py`` ``bounded_normal``) reads it as
an *untruncated* ``Normal(tanh(raw_mean), std)`` with
``std = (max_std - min_std) * sigmoid(raw_std + 2) + min_std``, starts the
policy's output layer at ``outscale 0.01`` (``dreamerv3/configs.yaml``), and
starts the reward and value heads' output layers at zero (paper, "Critic
learning").  Each is a constructor argument of ``Actor``, so each is tested
here on the class that owns it, and the two families are pinned from both
sides: V3 gets the release, V2 keeps its own.

Every property is checked against a value computed *here*, from the
formula, rather than read back from the code under test; the formulas'
sources are quoted in ``test_generative_provenance``.  Device-sensitive
properties run on the CPU and on Metal.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest

import lucid
import lucid.distributions as D
from lucid.models.generative._common._actor import Actor, ContinuousPolicy
from lucid.models.generative._common._dists import Normal, OneHotCategorical, TruncatedNormal
from lucid.test._default_device import default_device, metal_available

_LOG_2PI_E = math.log(2.0 * math.pi * math.e)


@pytest.fixture(params=["cpu", "metal"])
def device(request: pytest.FixtureRequest) -> Iterator[str]:
    """Run the test with this device as the default, so models and inputs land on it."""
    if request.param == "metal" and not metal_available():
        pytest.skip("needs Metal")
    with default_device(request.param):
        yield request.param


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _values(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


@dataclass(frozen=True)
class _Spec:
    """An ``Actor`` over an 8-wide latent, spelt so a test overrides only what it is about."""

    action_dim: int = 3
    min_std: float = 0.1
    max_std: float = 1.0
    discrete: bool = False
    continuous_dist: ContinuousPolicy = "truncated_normal"
    outscale: float | None = None

    def build(self) -> Actor:
        return Actor(
            8,
            16,
            2,
            self.action_dim,
            "silu",
            self.min_std,
            self.discrete,
            continuous_dist=self.continuous_dist,
            max_std=self.max_std,
            outscale=self.outscale,
        )


def _actor(**overrides: object) -> Actor:
    return replace(_Spec(), **overrides).build()


def _pin(actor: Actor, mean: list[float], std: list[float]) -> None:
    """Make the raw output a constant — the trunk's weights zeroed, the bias the answer."""
    out = actor.head.out
    with lucid.no_grad():
        out.weight[:] = lucid.zeros_like(out.weight)
        out.bias[:] = lucid.tensor(mean + std, device=out.bias.device.type)


def _same_seed(seed: int, **overrides: object) -> Actor:
    lucid.manual_seed(seed)
    return _actor(**overrides)


def _feature(batch: int = 5, width: int = 8) -> lucid.Tensor:
    return lucid.randn((batch, 2, width))


class TestNormal:
    """The untruncated Gaussian, against closed forms and against ``lucid.distributions``."""

    def test_log_prob_is_the_closed_form(self, device: str) -> None:
        loc, scale, x = [0.3, -0.7, 0.0], [0.5, 1.0, 2.0], [0.1, -2.0, 3.0]
        got = _values(
            Normal(lucid.tensor(loc, device=device), lucid.tensor(scale, device=device)).log_prob(
                lucid.tensor(x, device=device)
            )
        )
        want = [
            -0.5 * ((xi - m) / s) ** 2 - math.log(s) - 0.5 * math.log(2.0 * math.pi)
            for xi, m, s in zip(x, loc, scale, strict=True)
        ]
        assert got == pytest.approx(want, abs=1e-5)

    def test_entropy_is_the_closed_form(self, device: str) -> None:
        scale = [0.1, 0.5, 1.0, 2.1]
        got = _values(
            Normal(lucid.zeros((4,), device=device), lucid.tensor(scale, device=device)).entropy()
        )
        want = [0.5 * _LOG_2PI_E + math.log(s) for s in scale]
        assert got == pytest.approx(want, abs=1e-5)
        # 0.5 * log(2 * pi * e * sigma^2): the form the released code writes.
        assert want == pytest.approx(
            [0.5 * math.log(2.0 * math.pi * math.e * s * s) for s in scale], abs=1e-12
        )

    def test_it_agrees_with_the_public_normal(self, device: str) -> None:
        loc = lucid.randn((4, 3), device=device)
        scale = lucid.rand((4, 3), device=device) + 0.2
        x = lucid.randn((4, 3), device=device)
        ours, theirs = Normal(loc, scale), D.Normal(loc, scale)
        assert _values(ours.log_prob(x)) == pytest.approx(_values(theirs.log_prob(x)), abs=1e-5)
        assert _values(ours.entropy()) == pytest.approx(_values(theirs.entropy()), abs=1e-5)

    def test_mean_and_mode_are_loc(self, device: str) -> None:
        loc = lucid.tensor([0.25, -0.5], device=device)
        dist = Normal(loc, lucid.ones((2,), device=device))
        assert _values(dist.mean) == _values(dist.mode) == pytest.approx([0.25, -0.5])

    def test_a_draw_is_not_clipped_and_has_the_right_moments(self, device: str) -> None:
        lucid.manual_seed(0)
        loc, scale = 0.3, 0.5
        draws = _values(
            Normal(
                lucid.full((20000,), loc, device=device), lucid.full((20000,), scale, device=device)
            ).rsample()
        )
        mean = sum(draws) / len(draws)
        spread = math.sqrt(sum((d - mean) ** 2 for d in draws) / len(draws))
        assert mean == pytest.approx(loc, abs=0.02)
        assert spread == pytest.approx(scale, abs=0.02)
        assert max(draws) > 1.0 and min(draws) < -0.2  # nothing cut it off at the box

    def test_the_draw_carries_a_gradient_to_both_parameters(self, device: str) -> None:
        loc = lucid.zeros((6,), device=device, requires_grad=True)
        scale = lucid.ones((6,), device=device, requires_grad=True)
        Normal(loc, scale).rsample().sum().backward()
        assert _values(loc.grad) == [1.0] * 6
        assert loc.grad is not None and scale.grad is not None
        assert any(v != 0.0 for v in _values(scale.grad))


class TestBoundedNormalHead:
    """DreamerV3's released continuous head, as an ``Actor`` setting."""

    def _bounded(self, **overrides: object) -> Actor:
        return _actor(continuous_dist="bounded_normal", **overrides)

    def test_the_distribution_is_an_untruncated_normal(self, device: str) -> None:
        policy = self._bounded().distribution(_feature())
        assert type(policy) is Normal and not isinstance(policy, TruncatedNormal)

    def test_the_mean_is_tanh_of_the_raw_output(self, device: str) -> None:
        actor = self._bounded()
        feature = _feature()
        raw = actor.head(feature)[..., :3]
        want = [math.tanh(v) for v in _values(raw)]
        assert _values(actor.distribution(feature).mean) == pytest.approx(want, abs=1e-5)

    def test_the_scale_is_the_released_formula_across_the_range(self, device: str) -> None:
        """``(max - min) * sigmoid(raw + 2) + min``, swept over raw outputs from -20 to 20."""
        points = [-20.0 + i for i in range(41)]
        actor = self._bounded(action_dim=len(points), max_std=1.0)
        _pin(actor, [0.0] * len(points), points)
        got = _values(actor.distribution(_feature()).scale[0, 0])
        want = [0.9 * _sigmoid(x + 2.0) + 0.1 for x in points]
        assert got == pytest.approx(want, abs=1e-5)
        assert got == sorted(got)  # monotone in the raw output

    def test_the_scale_stays_inside_its_bounds_and_reaches_both(self, device: str) -> None:
        actor = self._bounded(action_dim=2, min_std=0.1, max_std=0.7)
        _pin(actor, [0.0, 0.0], [-80.0, 80.0])  # past where float32 saturates the sigmoid
        low, high = _values(actor.distribution(_feature()).scale[0, 0])
        assert low == pytest.approx(0.1, abs=1e-6) and low >= 0.1
        assert high == pytest.approx(0.7, abs=1e-6) and high <= 0.7 + 1e-6

    def test_a_zero_output_starts_the_scale_near_its_ceiling(self, device: str) -> None:
        actor = self._bounded(outscale=0.0)
        policy = actor.distribution(_feature())
        assert _values(policy.scale) == pytest.approx([0.9 * _sigmoid(2.0) + 0.1] * 30, abs=1e-6)
        assert _values(policy.mean) == [0.0] * 30

    def test_samples_leave_the_box_where_a_truncated_policy_could_not(self, device: str) -> None:
        lucid.manual_seed(0)
        feature = lucid.zeros((4000, 8))
        bounded = self._bounded(action_dim=1)
        _pin(bounded, [0.0], [80.0])  # std = max_std = 1.0, mean = 0
        draws = _values(bounded(feature))
        outside = sum(abs(d) > 1.0 for d in draws) / len(draws)
        assert outside == pytest.approx(0.3173, abs=0.03)  # P(|N(0, 1)| > 1)
        assert sum(draws) / len(draws) == pytest.approx(0.0, abs=0.06)

        truncated = _actor(action_dim=1)
        _pin(truncated, [0.0], [80.0])  # its widest: std = 2.1
        assert all(-1.0 < d < 1.0 for d in _values(truncated(feature)))

    def test_the_mode_stays_inside_the_box(self, device: str) -> None:
        actor = self._bounded()
        with lucid.no_grad():
            actor.head.out.bias[:] = lucid.full_like(actor.head.out.bias, 40.0)
        assert max(abs(v) for v in _values(actor(_feature(), sample=False))) <= 1.0

    def test_log_prob_is_the_summed_normal_density(self, device: str) -> None:
        actor = self._bounded()
        feature = _feature()
        action = actor(feature)
        policy = actor.distribution(feature)
        loc, scale, x = (_values(t) for t in (policy.mean, policy.scale, action))
        per_dim = [
            -0.5 * ((xi - m) / s) ** 2 - math.log(s) - 0.5 * math.log(2.0 * math.pi)
            for xi, m, s in zip(x, loc, scale, strict=True)
        ]
        want = [sum(per_dim[i : i + 3]) for i in range(0, len(per_dim), 3)]
        assert _values(actor.log_prob(feature, action)) == pytest.approx(want, abs=1e-4)

    def test_entropy_is_at_most_that_of_the_widest_policy(self, device: str) -> None:
        actor = self._bounded(action_dim=2, max_std=1.0)
        _pin(actor, [0.0, 0.0], [80.0, 80.0])
        widest = 2 * 0.5 * _LOG_2PI_E  # two dims at std 1: the ceiling of the entropy bonus
        assert _values(actor.entropy(_feature())) == pytest.approx([widest] * 10, abs=1e-4)
        assert max(_values(self._bounded(action_dim=2).entropy(_feature()))) < widest + 1e-6

    def test_the_policy_gradient_reaches_the_output_layer(self, device: str) -> None:
        actor = self._bounded(outscale=0.01)
        feature = _feature()
        actor.log_prob(feature, actor(feature).detach()).sum().backward()
        assert any(v != 0.0 for v in _values(actor.head.out.weight.grad))

    def test_the_discrete_head_is_untouched_by_the_continuous_choice(self, device: str) -> None:
        policy = self._bounded(discrete=True).distribution(_feature())
        assert isinstance(policy, OneHotCategorical)


class TestTruncatedHeadIsTheDefault:
    """What ``Actor`` does when nothing is asked of it — DreamerV2's head."""

    def test_the_distribution_is_truncated(self, device: str) -> None:
        assert isinstance(_actor().distribution(_feature()), TruncatedNormal)

    def test_the_scale_is_dreamer_v2s_formula(self, device: str) -> None:
        """``2 * sigmoid(raw / 2) + min_std`` — up to 2.1, where V3's tops out at 1."""
        points = [-20.0 + i for i in range(41)]
        actor = _actor(action_dim=len(points))
        _pin(actor, [0.0] * len(points), points)
        got = _values(actor.distribution(_feature()).scale[0, 0])
        assert got == pytest.approx([2.0 * _sigmoid(x / 2.0) + 0.1 for x in points], abs=1e-5)
        assert max(got) > 2.09

    def test_max_std_is_ignored_here_even_when_it_would_be_inconsistent(self, device: str) -> None:
        """The way ``unimix`` is ignored under a continuous action space."""
        assert isinstance(
            _actor(max_std=0.01, min_std=0.1).distribution(_feature()), TruncatedNormal
        )

    def test_no_outscale_means_the_layers_default_initialisation(self, device: str) -> None:
        actor = _same_seed(3)
        assert any(v != 0.0 for v in _values(actor.head.out.bias))
        assert max(abs(v) for v in _values(actor.head.out.weight)) > 0.05


class TestOutscale:
    """``outscale``: the released layers scale the kernel's initial draw and start the bias at zero."""

    def test_it_scales_the_kernel_and_zeroes_the_bias(self, device: str) -> None:
        plain = _same_seed(7)
        scaled = _same_seed(7, outscale=0.01)
        assert _values(scaled.head.out.weight) == pytest.approx(
            [0.01 * v for v in _values(plain.head.out.weight)], rel=1e-5, abs=1e-9
        )
        assert _values(scaled.head.out.bias) == [0.0] * 6
        assert any(v != 0.0 for v in _values(plain.head.out.bias))

    def test_nothing_else_is_initialised_differently(self, device: str) -> None:
        """Scaling consumes no random numbers, so the trunk is the same trunk."""
        plain = _same_seed(7)
        scaled = _same_seed(7, outscale=0.01)
        for a, b in zip(
            plain.head.layers.parameters(), scaled.head.layers.parameters(), strict=True
        ):
            assert _values(a) == _values(b)

    def test_zero_gives_an_output_that_is_exactly_zero(self, device: str) -> None:
        actor = _same_seed(7, outscale=0.0)
        assert all(v == 0.0 for v in _values(actor.head.out.weight))
        assert all(v == 0.0 for v in _values(actor.head.out.bias))
        assert _values(actor.head(_feature())) == [0.0] * 60

    def test_the_default_scale_makes_a_near_uniform_discrete_policy(self, device: str) -> None:
        actor = _same_seed(7, action_dim=5, discrete=True, outscale=0.01)
        probs = _values(actor.distribution(_feature()).probs)
        assert max(abs(p - 0.2) for p in probs) < 0.02

    def test_it_applies_to_the_scale_row_as_much_as_the_location_row(self, device: str) -> None:
        plain = _same_seed(11, action_dim=4)
        scaled = _same_seed(11, action_dim=4, outscale=0.01)
        out_plain, out_scaled = (_values(a.head.out.weight) for a in (plain, scaled))
        half = len(out_plain) // 2  # rows [0, 4) are the location, [4, 8) the scale
        for lo, hi in ((0, half), (half, len(out_plain))):
            ratio = [s / p for s, p in zip(out_scaled[lo:hi], out_plain[lo:hi], strict=True) if p]
            assert ratio == pytest.approx([0.01] * len(ratio), rel=1e-4)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"continuous_dist": "squashed"},
            {"continuous_dist": "bounded_normal", "min_std": 0.5, "max_std": 0.1},
            {"outscale": -0.01},
        ],
    )
    def test_nonsense_is_rejected_at_construction(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            _actor(**overrides)
