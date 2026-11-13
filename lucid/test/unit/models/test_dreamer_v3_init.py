"""How DreamerV3's dense layers start, against the release they are meant to match.

The released implementation draws every dense kernel from a truncated normal
scaled by its fan-in and starts every bias at zero (``winit: trunc_normal_in``
in ``dreamerv3/configs.yaml``; ``Initializer`` in ``embodied/jax/nets.py``).
The earlier Dreamers specify nothing, so the initialiser is opt-in per block
(``dense_init``) and DreamerV3's config turns it on.

What these tests refuse to do is read the answer back from the code under
test.  The distribution is pinned by quantities worked out here: the draw's
standard deviation is ``1/sqrt(fan_in)``; it stops at ``2 * 1.1368 /
sqrt(fan_in)`` where a Gaussian of that spread would not; and the share of
draws inside one *underlying* standard deviation is the truncated normal's
``(2*Phi(1) - 1) / (2*Phi(2) - 1)``, which a plain Gaussian, a uniform, and a
truncation at some other point would each miss.

The class-wide guard is :class:`TestEveryDenseLayerOfTheModel`: it walks the
built model, so a dense layer added later without the knob starts with a
uniform bias and fails there rather than quietly keeping the default.
"""

import math
from collections.abc import Iterator
from dataclasses import replace

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.models import (
    DreamerV2Config,
    DreamerV2Model,
    DreamerV3Config,
    DreamerV3ForWorldModeling,
    DreamerV3Model,
    create_model,
)
from lucid.models.generative._common._actor import Actor
from lucid.models.generative._common._init import fan_in_trunc_normal_, initialise_linear_layers_
from lucid.models.generative._common._pixel_nets import DenseHead, PixelDecoder
from lucid.models.generative._common._rssm import RSSM, BlockLinear
from lucid.models.generative.dreamer_v3._heads import _TwoHotHead
from lucid.test._default_device import default_device, metal_available

# ``embodied/jax/nets.py:170`` and the truncation at ``:169``.
_RESCALE = 1.1368
_TRUNCATION = 2.0


@pytest.fixture(params=["cpu", "metal"])
def device(request: pytest.FixtureRequest) -> Iterator[str]:
    """Run the test with this device as the default, so layers land on it."""
    if request.param == "metal" and not metal_available():
        pytest.skip("needs Metal")
    with default_device(request.param):
        yield request.param


def _values(t: lucid.Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _moments(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


# The share of a normal truncated at +-2 sigma that lies within +-1 sigma.
_SHARE_INSIDE_ONE_SIGMA = (2 * _phi(1.0) - 1) / (2 * _phi(_TRUNCATION) - 1)


class TestFanInTruncNormal:
    """``fan_in_trunc_normal_`` is ``trunc_normal_in``: the distribution, not just its scale."""

    @pytest.mark.parametrize("fan_in", [16, 64, 300])
    def test_the_draws_have_standard_deviation_one_over_root_fan_in(
        self, device: str, fan_in: int
    ) -> None:
        lucid.manual_seed(0)
        w = fan_in_trunc_normal_(lucid.empty(400, fan_in), fan_in)
        mean, std = _moments(_values(w))
        n = 400 * fan_in
        assert abs(mean) < 5.0 / math.sqrt(fan_in) / math.sqrt(n)
        assert std * math.sqrt(fan_in) == pytest.approx(1.0, rel=0.03)

    def test_the_draws_are_cut_where_the_release_cuts_them(self, device: str) -> None:
        """A Gaussian of this spread would leave the bound 2.3% of the time; this never does."""
        lucid.manual_seed(1)
        fan_in = 64
        bound = _TRUNCATION * _RESCALE / math.sqrt(fan_in)
        values = _values(fan_in_trunc_normal_(lucid.empty(256, fan_in), fan_in))
        assert max(abs(v) for v in values) <= bound * (1.0 + 1e-5)
        assert max(abs(v) for v in values) > 0.97 * bound  # and it reaches the cut

    def test_the_share_inside_one_underlying_sigma_is_the_truncated_normals(
        self, device: str
    ) -> None:
        lucid.manual_seed(2)
        fan_in = 64
        sigma = _RESCALE / math.sqrt(fan_in)
        values = _values(fan_in_trunc_normal_(lucid.empty(512, fan_in), fan_in))
        share = sum(abs(v) < sigma for v in values) / len(values)
        assert share == pytest.approx(_SHARE_INSIDE_ONE_SIGMA, abs=0.01)

    def test_outscale_scales_the_draw_and_consumes_the_same_random_numbers(
        self, device: str
    ) -> None:
        lucid.manual_seed(3)
        plain = _values(fan_in_trunc_normal_(lucid.empty(8, 8), 8))
        lucid.manual_seed(3)
        scaled = _values(fan_in_trunc_normal_(lucid.empty(8, 8), 8, outscale=0.01))
        assert scaled == pytest.approx([0.01 * v for v in plain], rel=1e-5, abs=1e-9)

    def test_outscale_zero_is_exactly_zero(self, device: str) -> None:
        w = fan_in_trunc_normal_(lucid.ones((4, 4)), 4, outscale=0.0)
        assert all(v == 0.0 for v in _values(w))

    @pytest.mark.parametrize("kwargs", [{"fan_in": 0}, {"fan_in": 4, "outscale": -0.1}])
    def test_nonsense_is_refused(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            fan_in_trunc_normal_(lucid.empty(4, 4), **kwargs)


class TestInitialiseLinearLayers:
    def test_default_leaves_the_layer_as_its_constructor_drew_it(self) -> None:
        lucid.manual_seed(4)
        untouched = nn.Linear(16, 8)
        lucid.manual_seed(4)
        layer = nn.Linear(16, 8)
        initialise_linear_layers_(layer, "default")
        assert _values(layer.weight) == _values(untouched.weight)
        assert _values(layer.bias) == _values(untouched.bias)
        assert any(v != 0.0 for v in _values(layer.bias))  # the default's bias is not zero

    def test_the_release_init_uses_in_features_and_zeroes_the_bias(self, device: str) -> None:
        lucid.manual_seed(5)
        layer = nn.Linear(48, 96)
        initialise_linear_layers_(layer, "fan_in_trunc_normal")
        assert all(v == 0.0 for v in _values(layer.bias))
        _, std = _moments(_values(layer.weight))
        assert std * math.sqrt(48) == pytest.approx(1.0, rel=0.05)  # in, not out (96)

    def test_a_layer_without_a_bias_is_fine(self) -> None:
        layer = nn.Linear(8, 4, bias=False)
        initialise_linear_layers_(layer, "fan_in_trunc_normal")
        assert layer.bias is None

    def test_an_unknown_name_is_refused_even_by_the_no_op(self) -> None:
        with pytest.raises(ValueError, match="dense_init"):
            initialise_linear_layers_(nn.Linear(4, 4), "glorot")  # type: ignore[arg-type]


class TestBlockLinear:
    """The release's fan for a block kernel is the *whole* input width, not the block's."""

    def test_the_fan_is_the_whole_input_width(self, device: str) -> None:
        """``nets.py:196-197`` multiplies the within-block width back by the block count."""
        lucid.manual_seed(6)
        layer = BlockLinear(256, 128, blocks=8, dense_init="fan_in_trunc_normal")
        _, std = _moments(_values(layer.weight))
        assert std * math.sqrt(256) == pytest.approx(1.0, rel=0.05)
        assert std * math.sqrt(256 // 8) < 0.5  # not the within-block fan

    def test_the_bias_is_zero(self, device: str) -> None:
        layer = BlockLinear(32, 16, blocks=4, dense_init="fan_in_trunc_normal")
        assert all(v == 0.0 for v in _values(layer.bias))

    def test_default_is_still_uniform_on_the_within_block_fan(self) -> None:
        layer = BlockLinear(32, 16, blocks=4)
        bound = 1.0 / math.sqrt(32 // 4)
        assert max(abs(v) for v in _values(layer.weight)) <= bound
        assert any(v != 0.0 for v in _values(layer.bias))

    def test_reset_parameters_returns_to_the_declared_distribution(self) -> None:
        layer = BlockLinear(32, 16, blocks=4, dense_init="fan_in_trunc_normal")
        with lucid.no_grad():
            layer.bias[:] = lucid.ones_like(layer.bias)
        layer.reset_parameters()
        assert all(v == 0.0 for v in _values(layer.bias))

    def test_an_unknown_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dense_init"):
            BlockLinear(8, 8, blocks=2, dense_init="glorot")  # type: ignore[arg-type]


class TestSharedBlocksKeepTheirDefault:
    """DreamerV2 and PlaNet share these blocks and do not opt in — their draws must not move."""

    def test_a_dense_head_by_default_draws_exactly_what_nn_linear_draws(self) -> None:
        lucid.manual_seed(7)
        head = DenseHead(8, 16, 2, out_features=3)
        lucid.manual_seed(7)
        reference = [nn.Linear(8, 16), nn.Linear(16, 16), nn.Linear(16, 3)]
        built = [*head.layers, head.out]
        for ours, theirs in zip(built, reference, strict=True):
            assert _values(ours.weight) == _values(theirs.weight)
            assert _values(ours.bias) == _values(theirs.bias)

    def test_a_dense_head_can_opt_in(self) -> None:
        head = DenseHead(8, 16, 2, out_features=3, dense_init="fan_in_trunc_normal")
        assert all(v == 0.0 for layer in (*head.layers, head.out) for v in _values(layer.bias))

    def test_the_decoder_lift_is_dense_the_deconvolutions_are_not(self) -> None:
        decoder = PixelDecoder(16, 3, 4, "silu", dense_init="fan_in_trunc_normal")
        assert all(v == 0.0 for v in _values(decoder.lift.bias))
        assert any(v != 0.0 for v in _values(decoder.deconvs[0].bias))

    def test_the_gru_recurrence_has_nothing_in_the_release_to_match(self) -> None:
        with pytest.raises(ValueError, match="blocks"):
            RSSM(4, 8, 8, 2, 6, discrete=3, dense_init="fan_in_trunc_normal")

    def test_the_rssm_by_default_keeps_the_uniform_biases(self) -> None:
        rssm = RSSM(4, 8, 8, 2, 6, discrete=3, blocks=2)
        assert any(v != 0.0 for v in _values(rssm.prior_head[0].bias))

    def test_the_actor_scales_what_dense_init_drew(self) -> None:
        """``outscale`` composes on top: the draw is the same, the kernel x0.01, the bias zero."""
        kwargs: dict[str, object] = {"continuous_dist": "bounded_normal"}
        lucid.manual_seed(8)
        unit = Actor(8, 16, 2, 3, "silu", 0.1, dense_init="fan_in_trunc_normal", **kwargs)
        lucid.manual_seed(8)
        scaled = Actor(
            8, 16, 2, 3, "silu", 0.1, outscale=0.01, dense_init="fan_in_trunc_normal", **kwargs
        )
        assert _values(scaled.head.out.weight) == pytest.approx(
            [0.01 * v for v in _values(unit.head.out.weight)], rel=1e-5, abs=1e-9
        )
        assert all(v == 0.0 for v in _values(scaled.head.out.bias))

    def test_zero_init_still_wins_on_the_output_of_a_two_hot_head(self) -> None:
        head = _TwoHotHead(8, 16, 2, num_bins=5, zero_init=True, dense_init="fan_in_trunc_normal")
        assert all(v == 0.0 for v in _values(head.trunk.out.weight))
        assert any(v != 0.0 for v in _values(head.trunk.layers[0].weight))


_V3 = DreamerV3Config(
    action_dim=2,
    cnn_depth=4,
    stoch_size=8,
    discrete=8,
    deter_size=64,
    hidden_size=64,
    actor_hidden=64,
    value_hidden=64,
    reward_hidden=64,
    num_bins=7,
    horizon=4,
    blocks=8,
    pcont=True,
)

# The layers whose start the release overrides on top of ``trunc_normal_in`` —
# ``rewhead.outscale`` and ``value.outscale`` 0.0, ``policy.outscale`` 0.01 —
# and what each multiplies the draw by.  Everything else is 1.
_OUTSCALE = {
    "actor.head.out": 0.01,
    "reward_head.trunk.out": 0.0,
    "value_head.trunk.out": 0.0,
    "slow_value_head.trunk.out": 0.0,
}


def _dense_layers(model: nn.Module) -> list[tuple[str, nn.Module, int]]:
    """``(name, layer, fan_in)`` for every dense layer, with the release's fan for each kind."""
    layers: list[tuple[str, nn.Module, int]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module, module.in_features))
        elif isinstance(module, BlockLinear):
            layers.append((name, module, module.blocks * module.in_block))
    return layers


@pytest.fixture(scope="module")
def model() -> DreamerV3Model:
    lucid.manual_seed(0)
    return DreamerV3Model(_V3)


class TestEveryDenseLayerOfTheModel:
    """The class-wide guard: walk the built model, so a layer left out of the knob is caught."""

    def test_the_walk_finds_the_layers_it_should(self, model: DreamerV3Model) -> None:
        names = {name.removeprefix("dreamer_v3.") for name, _, _ in _dense_layers(model)}
        assert {
            "decoder.lift",
            "rssm.deter_in",
            "rssm.stoch_in",
            "rssm.action_in",
            "rssm.block_hidden",
            "rssm.block_gate",
            "rssm.prior_head.0",
            "rssm.posterior_head.1",
            "actor.head.layers.0",
            "actor.head.out",
            "reward_head.trunk.layers.0",
            "value_head.trunk.out",
            "slow_value_head.trunk.layers.2",
            "pcont_head.layers.0",
            "pcont_head.out",
        } <= names

    def test_every_bias_is_zero(self, model: DreamerV3Model) -> None:
        for name, layer, _ in _dense_layers(model):
            assert all(v == 0.0 for v in _values(layer.bias)), name

    def test_every_kernel_stays_inside_its_layers_truncation(self, model: DreamerV3Model) -> None:
        for name, layer, fan_in in _dense_layers(model):
            outscale = _OUTSCALE.get(name, 1.0)
            bound = outscale * _TRUNCATION * _RESCALE / math.sqrt(fan_in)
            assert max(abs(v) for v in _values(layer.weight)) <= bound * (1.0 + 1e-5), name

    def test_the_kernels_pooled_over_the_unit_layers_are_the_truncated_normal(
        self, model: DreamerV3Model
    ) -> None:
        """Scaled by each layer's own fan, the draws are one distribution — the one above."""
        pooled: list[float] = []
        for name, layer, fan_in in _dense_layers(model):
            if name.removeprefix("dreamer_v3.") not in _OUTSCALE:
                pooled += [v * math.sqrt(fan_in) for v in _values(layer.weight)]
        _, std = _moments(pooled)
        assert std == pytest.approx(1.0, rel=0.02)
        share = sum(abs(v) < _RESCALE for v in pooled) / len(pooled)
        assert share == pytest.approx(_SHARE_INSIDE_ONE_SIGMA, abs=0.01)

    def test_the_output_layers_keep_the_release_s_own_scales_on_top(
        self, model: DreamerV3Model
    ) -> None:
        for head in ("reward_head", "value_head", "slow_value_head"):
            assert all(v == 0.0 for v in _values(getattr(model, head).trunk.out.weight)), head
        out = model.actor.head.out
        assert max(abs(v) for v in _values(out.weight)) <= 0.01 * _TRUNCATION * _RESCALE / 8
        assert any(v != 0.0 for v in _values(out.weight))  # scaled, not zeroed


class TestTheKnob:
    def test_release_init_is_the_default(self) -> None:
        assert DreamerV3Config(action_dim=2).dense_init == "fan_in_trunc_normal"

    def test_an_unknown_initialiser_is_refused_at_the_config(self) -> None:
        with pytest.raises(ValueError, match="dense_init"):
            DreamerV3Config(action_dim=2, dense_init="glorot")  # type: ignore[arg-type]

    def test_default_keeps_the_layers_own_start_with_the_output_rules_on_top(self) -> None:
        lucid.manual_seed(0)
        model = DreamerV3Model(replace(_V3, dense_init="default"))
        assert any(v != 0.0 for v in _values(model.rssm.prior_head[0].bias))
        assert any(v != 0.0 for v in _values(model.reward_head.trunk.layers[0].bias))
        assert all(v == 0.0 for v in _values(model.reward_head.trunk.out.weight))

    def test_checkpoints_are_interchangeable(self) -> None:
        """Only values differ: same keys, shapes, dtypes, parameter count."""
        lucid.manual_seed(0)
        old = DreamerV3Model(replace(_V3, dense_init="default"))
        lucid.manual_seed(0)
        new = DreamerV3Model(_V3)
        old_state, new_state = old.state_dict(), new.state_dict()
        assert list(old_state) == list(new_state)
        for key, tensor in old_state.items():
            assert tensor.shape == new_state[key].shape and tensor.dtype == new_state[key].dtype
        assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())

    def test_dreamer_v2_is_not_touched(self) -> None:
        """V2 never passes the knob, so its dense layers keep ``nn.Linear``'s uniform bias."""
        lucid.manual_seed(0)
        model = DreamerV2Model(
            DreamerV2Config(
                action_dim=2,
                cnn_depth=4,
                stoch_size=4,
                discrete=5,
                deter_size=16,
                hidden_size=16,
                actor_hidden=16,
                value_hidden=16,
                reward_hidden=16,
                horizon=4,
                pcont=False,
            )
        )
        assert any(v != 0.0 for v in _values(model.rssm.prior_head[0].bias))
        assert any(v != 0.0 for v in _values(model.actor.head.layers[0].bias))


class TestTheReleaseInitialisedModelWorks:
    """Reachable end to end from the public factory: a training step, then inference."""

    def test_the_public_factory_trains_and_acts(self, device: str) -> None:
        lucid.manual_seed(0)
        model = create_model(
            "dreamer_v3_12m_world_model",
            action_dim=2,
            cnn_depth=4,
            stoch_size=4,
            discrete=5,
            deter_size=16,
            hidden_size=16,
            actor_hidden=16,
            value_hidden=16,
            reward_hidden=16,
            num_bins=7,
            horizon=4,
            blocks=4,
            pcont=False,
        )
        assert isinstance(model, DreamerV3ForWorldModeling)
        assert model.config.dense_init == "fan_in_trunc_normal"
        optimisers = [
            optim.Adam(model.world_parameters(), lr=1e-3),
            optim.Adam(model.value_parameters(), lr=1e-3),
            optim.Adam(model.actor_parameters(), lr=1e-3),
        ]
        batch = (lucid.randn((2, 4, 3, 64, 64)), lucid.randn((2, 4, 2)), lucid.randn((2, 4)))
        # The reconstruction loss reaches the decoder's lift whatever the KL's free bits clip.
        before = _values(model.dreamer_v3.decoder.lift.weight)
        output = model(*batch)
        assert math.isfinite(float(output.loss.item()))
        model.backward(output)
        for opt in optimisers:
            opt.step()
        assert _values(model.dreamer_v3.decoder.lift.weight) != before  # it received a gradient
        assert math.isfinite(float(model(*batch).loss.item()))

        model.eval()
        inner = model.dreamer_v3
        _, posteriors = inner.observe(batch[0], batch[1])
        action = inner.act(posteriors.map(lambda t: t[:, -1]), sample=False)
        assert action.shape == (2, 2)
        assert all(math.isfinite(v) for v in _values(action))
