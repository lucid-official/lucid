"""EVA redistributes the rank budget over the layers (LCD-977).

``initialize_lora_eva_weights`` gives each layer the rank its activations call for, so a layer's
adapter is rebuilt at a size the layer already holds an adapter of.  ``LoraLayer.update_layer``
refuses a name the layer holds, and that was the only way in: a redistribution that moved any
layer off ``r`` raised ``Adapter with name 'default' already exists.``  Each test here gives the
two layers inputs with unlike spectra, so the ranks that come back differ from ``r`` — and from
each other.

A layer EVA gives rank 0 (LCD-979) loses the EVA adapter and nothing else: the other adapters it
holds, their parameters and their configs stay, and only a layer left holding no adapter at all
goes back to the module it wrapped.  EVA giving every layer rank 0 (LCD-982) is refused before any
layer changes: the adapter would hold no layer, and what is saved of it could not be loaded.

``rank_pattern`` / ``alpha_pattern`` match a name by its dotted tail, so the entry of ``q_proj`` is
the one read for ``block.q_proj`` as well (LCD-982): a layer at the default is listed when an entry
for another name would be read for it, and a saved adapter comes back at every layer's own rank.
"""

from copy import deepcopy
import itertools
from pathlib import Path
from typing import NamedTuple

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.peft import (
    EvaConfig,
    LoraConfig,
    PeftModel,
    get_eva_state_dict,
    get_layer_status,
    get_peft_model,
    initialize_lora_eva_weights,
)
from lucid.peft._targeting import _pattern_value, _selects
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners.lora import Linear, LoraLayer
from lucid.peft.tuners.lora._eva import _recorded, _select_layers
from lucid.test.unit.peft._tuner_recipes import perturb_adapter, same, span

_D = 8
_R = 2
_ALPHA = 4


class _TwoStreams(nn.Module):
    """``q_proj`` reads the first half of the features and ``k_proj`` the second."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(_D, _D)
        self.k_proj = nn.Linear(_D, _D)
        self.head = nn.Linear(_D, 2)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.q_proj(x[:, :_D]) + self.k_proj(x[:, _D:]))


def _base(device: str) -> _TwoStreams:
    lucid.manual_seed(3)
    return _TwoStreams().to(device)


def _calibration(device: str) -> list[lucid.Tensor]:
    """Two batches: ``q_proj`` sees one direction and little else, ``k_proj`` a graded spread."""
    lucid.manual_seed(11)
    direction = lucid.randn(1, _D, device=device)
    spread = lucid.tensor([4.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3], device=device)
    batches = []
    for _ in range(2):
        q = lucid.randn(32, 1, device=device) @ direction + 0.01 * lucid.randn(
            32, _D, device=device
        )
        k = lucid.randn(32, _D, device=device) * spread
        batches.append(lucid.cat([q, k], dim=1))
    return batches


def _config(*, use_dora: bool = False) -> LoraConfig:
    return LoraConfig(
        r=_R,
        lora_alpha=_ALPHA,
        target_modules=["q_proj", "k_proj"],
        init_lora_weights="eva",
        eva_config=EvaConfig(rho=4, tau=0.5),
        use_dora=use_dora,
    )


def _state(base: nn.Module, config: LoraConfig, device: str) -> dict[str, lucid.Tensor]:
    return get_eva_state_dict(
        base,
        dataloader=_calibration(device),
        peft_config=config,
        forward_fn=lambda m, x: m(x),
        prepare_model_inputs_fn=None,
        prepare_layer_inputs_fn=None,
    )


def _layers(model: PeftModel) -> dict[str, LoraLayer]:
    return {
        key.removeprefix("base_model.model."): layer
        for key, layer in _tuner_layers(model)
        if isinstance(layer, LoraLayer)
    }


@pytest.fixture
def redistributed(device: str) -> tuple[PeftModel, dict[str, lucid.Tensor]]:
    """A model EVA-initialised at ranks other than ``r``, and the singular vectors it was given."""
    base, config = _base(device), _config()
    state = _state(base, config, device)
    ranks = {name: int(w.shape[0]) for name, w in state.items()}
    assert ranks["q_proj"] < _R < ranks["k_proj"], ranks  # the premise of every test here
    model = get_peft_model(base, config)
    initialize_lora_eva_weights(model, eva_state_dict=state)
    return model, state


def test_the_adapter_is_rebuilt_at_the_ranks_the_data_gave(
    redistributed: tuple[PeftModel, dict[str, lucid.Tensor]],
) -> None:
    model, state = redistributed
    for name, layer in _layers(model).items():
        w, rank = state[name], int(state[name].shape[0])
        assert layer.r["default"] == rank
        assert layer.lora_A["default"].weight.shape == (rank, _D)
        assert layer.lora_B["default"].weight.shape == (_D, rank)
        assert same(layer.lora_A["default"].weight, w)
        assert same(
            layer.lora_B["default"].weight, lucid.zeros_like(layer.lora_B["default"].weight)
        )
        # ``adjust_scaling_factors``: alpha grows with the rank, so the scale is the one r gave.
        assert layer.lora_alpha["default"] == _ALPHA * rank / _R
        assert layer.scaling["default"] == _ALPHA / _R


def test_the_config_records_what_the_data_gave(
    redistributed: tuple[PeftModel, dict[str, lucid.Tensor]],
) -> None:
    model, state = redistributed
    config = model.peft_config["default"]
    ranks = {name: int(w.shape[0]) for name, w in state.items()}
    assert config.rank_pattern == ranks
    assert config.alpha_pattern == {name: _ALPHA * rank / _R for name, rank in ranks.items()}
    assert sorted(config.target_modules or []) == ["k_proj", "q_proj"]


def test_the_forward_is_the_base_at_the_start_and_the_projection_after_it(
    redistributed: tuple[PeftModel, dict[str, lucid.Tensor]], device: str
) -> None:
    model, state = redistributed
    x = _calibration(device)[0]
    layers = _layers(model)
    head = model.get_base_model().head
    with lucid.no_grad():
        start = model(x)
        assert span(start, head(_streams(layers, x, state, update=False))) < 1e-6
        lucid.manual_seed(5)
        for layer in layers.values():
            weight = layer.lora_B["default"].weight
            weight.copy_(0.3 * lucid.randn(*weight.shape, device=weight.device))
        out = model(x)
        assert span(out, start) > 1e-2  # the update is not nothing
        assert span(out, head(_streams(layers, x, state, update=True))) < 1e-4


def _streams(
    layers: dict[str, LoraLayer], x: lucid.Tensor, state: dict[str, lucid.Tensor], *, update: bool
) -> lucid.Tensor:
    """``q_proj(x_q) + k_proj(x_k)`` from the base layers; with ``update``, plus ``s B A x`` with ``A`` the vectors EVA gave."""
    q, k = layers["q_proj"], layers["k_proj"]
    total = q.get_base_layer()(x[:, :_D]) + k.get_base_layer()(x[:, _D:])
    if not update:
        return total
    for name, layer, half in (("q_proj", q, x[:, :_D]), ("k_proj", k, x[:, _D:])):
        low = (half @ state[name].mT) @ layer.lora_B["default"].weight.mT
        total = total + layer.scaling["default"] * low
    return total


def test_the_redistributed_ranks_survive_save_and_load(
    redistributed: tuple[PeftModel, dict[str, lucid.Tensor]], device: str, tmp_path: Path
) -> None:
    model, state = redistributed
    perturb_adapter(model, "default", seed=5)
    model.save_pretrained(tmp_path)

    saved = LoraConfig.from_pretrained(tmp_path)
    assert saved.rank_pattern == model.peft_config["default"].rank_pattern
    assert saved.alpha_pattern == model.peft_config["default"].alpha_pattern

    loaded = PeftModel.from_pretrained(_base(device), tmp_path)
    x = _calibration(device)[0]
    for name, layer in _layers(loaded).items():
        assert layer.r["default"] == int(state[name].shape[0])
        assert layer.lora_alpha["default"] == _layers(model)[name].lora_alpha["default"]
    with lucid.no_grad():
        assert span(loaded(x), model(x)) < 1e-5


def test_a_dora_adapter_is_rebuilt_with_the_magnitude_of_the_weight(device: str) -> None:
    base, config = _base(device), _config(use_dora=True)
    state = _state(base, config, device)
    model = get_peft_model(base, config)
    x = _calibration(device)[0]
    with lucid.no_grad():
        before = model(x)
    initialize_lora_eva_weights(model, eva_state_dict=state)
    for name, layer in _layers(model).items():
        assert layer.r["default"] == int(state[name].shape[0])
        assert layer.use_dora["default"]
        norm = linalg.norm(layer.get_base_layer().weight, dim=1)
        assert span(layer.lora_magnitude_vector["default"].weight, norm) < 1e-6
    with lucid.no_grad():
        assert span(model(x), before) < 1e-5  # B is zero: the start is the base


def test_an_adapter_added_to_a_merged_layer_is_rebuilt_from_the_weights_before_the_merge(
    device: str,
) -> None:
    base = _base(device)
    other = LoraConfig(r=3, lora_alpha=6, target_modules=["q_proj", "k_proj"])
    model = get_peft_model(base, other, adapter_name="other")
    perturb_adapter(model, "other", seed=5)
    model.base_model.merge_adapter()
    config = _config(use_dora=True)
    model.add_adapter("eva", config)
    layers = _layers(model)
    assert all(layer.merged_adapters == ["other"] for layer in layers.values())
    merged = {
        name: layer.get_base_layer().weight.detach().clone() for name, layer in layers.items()
    }

    state = _state(_base(device), config, device)  # the premise: ranks that are not r
    assert {int(w.shape[0]) for w in state.values()} != {_R}
    initialize_lora_eva_weights(model, eva_state_dict=state, adapter_name="eva")

    for name, layer in layers.items():
        assert layer.r["eva"] == int(state[name].shape[0])
        assert layer.merged_adapters == ["other"]
        assert same(layer.get_base_layer().weight, merged[name])  # folded again, as it was
        # The magnitude is the norm of the weight without "other" in it.
        layer.unmerge()
        norm = linalg.norm(layer.get_base_layer().weight, dim=1)
        assert span(layer.lora_magnitude_vector["eva"].weight, norm) < 1e-6


def test_an_adapter_that_is_not_in_use_stays_frozen(device: str) -> None:
    base, config = _base(device), _config()
    model = get_peft_model(base, LoraConfig(r=_R, target_modules=["q_proj", "k_proj"]))
    model.add_adapter("eva", config)  # frozen and inactive, as an added adapter starts
    state = _state(_base(device), config, device)
    initialize_lora_eva_weights(model, eva_state_dict=state, adapter_name="eva")
    for name, layer in _layers(model).items():
        assert layer.r["eva"] == int(state[name].shape[0])
        assert layer.r["default"] == _R
        assert not any(p.requires_grad for p in layer._adapter_parameters("eva"))
        assert all(p.requires_grad for p in layer._adapter_parameters("default"))


def test_a_merged_adapter_is_not_initialised_and_the_model_is_left_as_it_was(device: str) -> None:
    base, config = _base(device), _config()
    state = _state(base, config, device)
    model = get_peft_model(base, config)
    model.base_model.merge_adapter()
    ranks = {name: layer.r["default"] for name, layer in _layers(model).items()}
    with pytest.raises(ValueError, match="merged"):
        initialize_lora_eva_weights(model, eva_state_dict=state)
    assert {name: layer.r["default"] for name, layer in _layers(model).items()} == ranks


class TestResizeAdapter:
    """What the layer keeps of an adapter it starts over."""

    @pytest.fixture
    def layer(self, device: str) -> Linear:
        lucid.manual_seed(3)
        base = nn.Linear(_D, 6, device=device)
        return Linear(base, "a", r=2, lora_alpha=4, lora_dropout=0.1, use_dora=True, lora_bias=True)

    def test_it_keeps_what_the_adapter_is_besides_its_size(self, layer: Linear) -> None:
        dropout = layer.lora_dropout["a"]
        layer._resize_adapter("a", 5, 10, use_rslora=True)
        assert layer.lora_dropout["a"] is dropout
        assert layer.use_dora["a"] and layer.lora_bias["a"]
        assert layer.r["a"] == 5 and layer.lora_alpha["a"] == 10
        assert layer.scaling["a"] == pytest.approx(10 / 5**0.5)
        assert layer.lora_A["a"].weight.shape == (5, _D)
        assert layer.lora_B["a"].weight.shape == (6, 5)
        assert layer.lora_B["a"].bias.shape == (6,)
        assert same(layer.lora_B["a"].weight, lucid.zeros_like(layer.lora_B["a"].weight))
        assert layer.lora_magnitude_vector["a"].weight.shape == (6,)

    def test_it_refuses_what_it_cannot_resize(self, layer: Linear) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            layer._resize_adapter("missing", 3, 3)
        with pytest.raises(ValueError, match="positive"):
            layer._resize_adapter("a", 0, 3)
        layer.merge()
        with pytest.raises(ValueError, match="merged"):
            layer._resize_adapter("a", 3, 3)
        assert layer.r["a"] == 2


# ── a layer EVA gives no rank (LCD-979) ──────────────────────────────────────


class _Nested(nn.Module):
    """``block.q_proj`` ends in ``q_proj``, the name of the layer beside it."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(_D, _D)
        self.block = nn.Module()
        self.block.q_proj = nn.Linear(_D, _D)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.q_proj(x) + self.block.q_proj(x)


class _Beside(NamedTuple):
    """A plain LoRA adapter ``default`` and an ``eva`` one on both layers; EVA gives ``q_proj`` no rank."""

    model: PeftModel
    state: dict[str, lucid.Tensor]
    #: ``default``'s parameters as they were, and what it was configured as.
    held: dict[str, lucid.Tensor]
    default_config: LoraConfig
    #: What each adapter gives for ``x`` before EVA ran, with that adapter in use.
    before: dict[str, lucid.Tensor]
    x: lucid.Tensor


def _no_rank(state: dict[str, lucid.Tensor], device: str) -> dict[str, lucid.Tensor]:
    """``state`` with ``q_proj`` given no direction at all: EVA's answer for a layer that needs no adapter."""
    return {**state, "q_proj": lucid.zeros(0, _D, device=device)}


def _beside_plain_lora(device: str, in_use: str) -> _Beside:
    base, config = _base(device), _config()
    model = get_peft_model(
        base, LoraConfig(r=_R, lora_alpha=_ALPHA, target_modules=["q_proj", "k_proj"])
    )
    perturb_adapter(model, "default", seed=5)  # an update that is not nothing, so a lost one shows
    model.add_adapter("eva", config)
    held = {n: p.detach().clone() for n, p in model.named_parameters() if ".default." in n}
    x = _calibration(device)[0]
    before: dict[str, lucid.Tensor] = {}
    with lucid.no_grad():
        for name in ("default", "eva"):
            model.set_adapter(name)
            before[name] = model(x)
    model.set_adapter(in_use)
    state = _no_rank(_state(_base(device), config, device), device)
    return _Beside(model, state, held, model.peft_config["default"], before, x)


@pytest.fixture(params=["default", "eva"])
def beside(request: pytest.FixtureRequest, device: str) -> _Beside:
    """``_beside_plain_lora`` with each of its two adapters in use, EVA already run."""
    assert isinstance(request.param, str)
    made = _beside_plain_lora(device, request.param)
    initialize_lora_eva_weights(made.model, eva_state_dict=made.state, adapter_name="eva")
    return made


def test_a_layer_without_a_rank_keeps_the_adapters_beside_eva(beside: _Beside) -> None:
    layers = _layers(beside.model)
    assert layers["q_proj"]._adapter_names() == ["default"]
    assert layers["q_proj"]._adapter_parameters("eva") == []
    assert sorted(layers["k_proj"]._adapter_names()) == ["default", "eva"]
    assert layers["k_proj"].r["eva"] == int(beside.state["k_proj"].shape[0])
    assert layers["q_proj"].r["default"] == _R
    # What an optimizer of the other adapter holds is still what the layer computes with.
    params = dict(beside.model.named_parameters())
    assert beside.held.keys() <= params.keys()
    for name, before in beside.held.items():
        assert same(params[name], before)


def test_the_other_adapters_forward_is_what_it_was(beside: _Beside) -> None:
    model = beside.model
    with lucid.no_grad():
        for name in ("default", "eva"):
            model.set_adapter(name)
            assert span(model(beside.x), beside.before[name]) < 1e-5
    assert span(beside.before["default"], beside.before["eva"]) > 1e-2  # a lost update shows


def test_the_adapter_in_use_stays_in_use_on_every_layer(beside: _Beside) -> None:
    in_use = beside.model.active_adapters
    assert in_use in (["default"], ["eva"])
    for status in get_layer_status(beside.model):
        assert status.active_adapters == in_use
    # Switching reaches the layer that holds nothing of ``eva`` as it does any other.
    q_proj = _layers(beside.model)["q_proj"]
    beside.model.set_adapter("eva")
    assert q_proj.active_adapters == ["eva"]
    assert all(not p.requires_grad for p in q_proj._adapter_parameters("default"))


def test_the_config_leaves_out_the_layer_without_a_rank_and_the_others_alone(
    beside: _Beside,
) -> None:
    eva = beside.model.peft_config["eva"]
    k_rank = int(beside.state["k_proj"].shape[0])
    assert eva.target_modules == ["k_proj"]
    assert eva.rank_pattern == {"k_proj": k_rank}
    assert eva.alpha_pattern == {"k_proj": _ALPHA * k_rank / _R}
    assert beside.model.peft_config["default"] == beside.default_config


def test_the_adapters_survive_save_and_load(beside: _Beside, device: str, tmp_path: Path) -> None:
    perturb_adapter(beside.model, "eva", seed=7)  # eva's update on k_proj is not nothing either
    beside.model.save_pretrained(tmp_path)
    assert LoraConfig.from_pretrained(tmp_path / "eva").target_modules == {"k_proj"}

    loaded = PeftModel.from_pretrained(_base(device), tmp_path)
    loaded.load_adapter(tmp_path / "eva", adapter_name="eva")
    layers = _layers(loaded)
    assert layers["q_proj"]._adapter_names() == ["default"]  # not an eva layer, as it was saved
    assert sorted(layers["k_proj"]._adapter_names()) == ["default", "eva"]
    with lucid.no_grad():
        loaded.set_adapter("default")
        assert span(loaded(beside.x), beside.before["default"]) < 1e-5  # q_proj's update was saved
        loaded.set_adapter("eva")
        beside.model.set_adapter("eva")
        assert span(loaded(beside.x), beside.model(beside.x)) < 1e-5


class _Held(NamedTuple):
    """Everything about a model that EVA changing it would show in: its tree, its parameters, its configs."""

    modules: list[tuple[str, type[nn.Module]]]
    parameters: dict[str, lucid.Tensor]
    values: dict[str, lucid.Tensor]
    trainable: dict[str, bool]
    adapters: dict[str, list[str]]
    configs: dict[str, LoraConfig]


def _held(model: PeftModel) -> _Held:
    params = dict(model.named_parameters())
    return _Held(
        [(name, type(module)) for name, module in model.named_modules()],
        params,
        {name: p.detach().clone() for name, p in params.items()},
        {name: p.requires_grad for name, p in params.items()},
        {name: layer._adapter_names() for name, layer in _layers(model).items()},
        {name: deepcopy(config) for name, config in model.peft_config.items()},
    )


def _assert_untouched(model: PeftModel, before: _Held) -> None:
    after = _held(model)
    assert after.modules == before.modules
    assert after.adapters == before.adapters
    assert after.trainable == before.trainable
    assert after.configs == before.configs
    assert after.parameters.keys() == before.parameters.keys()
    for name, param in after.parameters.items():
        assert param is before.parameters[name]  # not rebuilt
        assert same(param, before.values[name])


def test_when_no_layer_gets_a_rank_it_is_refused_and_the_model_is_left_whole(device: str) -> None:
    made = _beside_plain_lora(device, "default")
    state = {name: lucid.zeros(0, _D, device=device) for name in ("q_proj", "k_proj")}
    before = _held(made.model)
    with pytest.raises(ValueError, match="every layer got rank 0"):
        initialize_lora_eva_weights(made.model, eva_state_dict=state, adapter_name="eva")
    _assert_untouched(made.model, before)
    with lucid.no_grad():
        assert span(made.model(made.x), made.before["default"]) < 1e-5


def test_when_no_layer_gets_a_rank_the_adapter_that_holds_only_eva_is_not_unwrapped(
    device: str,
) -> None:
    base, config = _base(device), _config()
    model = get_peft_model(base, config)
    state = {name: lucid.zeros(0, _D, device=device) for name in ("q_proj", "k_proj")}
    before = _held(model)
    with pytest.raises(ValueError, match="every layer got rank 0.*rho=4"):
        initialize_lora_eva_weights(model, eva_state_dict=state)
    _assert_untouched(model, before)
    assert set(_layers(model)) == {"q_proj", "k_proj"}


def test_a_layer_the_state_dict_does_not_name_still_counts_as_one_that_keeps_a_rank(
    device: str,
) -> None:
    base, config = _base(device), _config()
    model = get_peft_model(base, config)
    state = {"q_proj": lucid.zeros(0, _D, device=device)}  # nothing said of k_proj: it keeps r
    with pytest.warns(UserWarning, match="not found in the eva state_dict"):
        initialize_lora_eva_weights(model, eva_state_dict=state)
    assert set(_layers(model)) == {"k_proj"}
    assert _layers(model)["k_proj"].r["default"] == _R


def test_a_layer_that_holds_only_eva_goes_back_to_the_module_it_wrapped(device: str) -> None:
    base, config = _base(device), _config()
    state = _no_rank(_state(base, config, device), device)
    model = get_peft_model(base, config)
    weight = _layers(model)["q_proj"].get_base_layer().weight
    initialize_lora_eva_weights(model, eva_state_dict=state)

    k_rank = int(state["k_proj"].shape[0])
    plain = model.get_base_model().q_proj
    assert type(plain) is nn.Linear and plain.weight is weight
    assert set(_layers(model)) == {"k_proj"}
    recorded = model.peft_config["default"]
    assert recorded.target_modules == ["k_proj"]
    assert recorded.rank_pattern == {"k_proj": k_rank}
    x = _calibration(device)[0]
    with lucid.no_grad():
        assert span(model(x), _base(device)(x)) < 1e-5  # the start is the base


def test_a_layer_without_a_rank_does_not_come_back_on_a_name_that_ends_in_anothers(
    device: str, tmp_path: Path
) -> None:
    lucid.manual_seed(3)
    config = LoraConfig(
        r=_R,
        lora_alpha=_ALPHA,
        target_modules=["q_proj", "block.q_proj"],
        init_lora_weights="eva",
        eva_config=EvaConfig(rho=4, tau=0.5),
    )
    model = get_peft_model(_Nested().to(device), config)
    state = {
        "q_proj": lucid.randn(_R, _D, device=device),
        "block.q_proj": lucid.zeros(0, _D, device=device),
    }
    initialize_lora_eva_weights(model, eva_state_dict=state)
    assert type(model.get_base_model().block.q_proj) is nn.Linear
    model.save_pretrained(tmp_path)

    lucid.manual_seed(3)
    loaded = PeftModel.from_pretrained(_Nested().to(device), tmp_path)
    assert isinstance(loaded.get_base_model().q_proj, LoraLayer)
    assert type(loaded.get_base_model().block.q_proj) is nn.Linear


@pytest.mark.parametrize("excluded", [None, ["skip"], {"skip"}, "skip|nothing"])
def test_selecting_the_kept_layers_keeps_what_the_config_already_excluded(
    excluded: str | list[str] | set[str] | None,
) -> None:
    config = LoraConfig(target_modules=["q_proj", "block.q_proj", "skip"], exclude_modules=excluded)
    _select_layers(config, ["q_proj", "skip"], ["block.q_proj"])
    assert _selects(config, "q_proj")
    assert not _selects(config, "block.q_proj")
    assert _selects(config, "skip") == (excluded is None)


def test_the_exclusion_is_left_alone_when_no_name_selects_a_dropped_layer() -> None:
    config = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj"], exclude_modules=["head"])
    _select_layers(config, ["q_proj", "k_proj"], ["v_proj"])
    assert config.target_modules == ["q_proj", "k_proj"]
    assert config.exclude_modules == {"head"}


# ── a name that ends in another's (LCD-982) ──────────────────────────────────


def _nested_eva(
    device: str, ranks: dict[str, int], *, alpha_pattern: dict[str, float] | None = None
) -> PeftModel:
    """``_Nested`` EVA-initialised at ``ranks`` (a layer's name -> its rank), every adapter weight then moved off its start."""
    lucid.manual_seed(3)
    config = LoraConfig(
        r=_R,
        lora_alpha=_ALPHA,
        target_modules=["q_proj", "block.q_proj"],
        alpha_pattern=alpha_pattern or {},
        init_lora_weights="eva",
        eva_config=EvaConfig(rho=4, tau=0.5),
    )
    model = get_peft_model(_Nested().to(device), config)
    state = {name: lucid.randn(rank, _D, device=device) for name, rank in ranks.items()}
    initialize_lora_eva_weights(model, eva_state_dict=state)
    perturb_adapter(model, "default", seed=5)
    return model


def _reloaded(model: PeftModel, device: str, tmp_path: Path) -> PeftModel:
    model.save_pretrained(tmp_path)
    lucid.manual_seed(3)
    return PeftModel.from_pretrained(_Nested().to(device), tmp_path)


def _scales(model: PeftModel) -> dict[str, tuple[int, float]]:
    return {
        name: (layer.r["default"], layer.lora_alpha["default"])
        for name, layer in _layers(model).items()
    }


@pytest.mark.parametrize(
    "ranks",
    [
        {"q_proj": 3, "block.q_proj": _R},  # the shorter name differs: its entry reaches the other
        {"q_proj": 1, "block.q_proj": _R},
        {"q_proj": _R, "block.q_proj": 3},  # the longer name differs: nothing reaches the other
        {"q_proj": 3, "block.q_proj": 1},
    ],
    ids=["shorter-above-r", "shorter-below-r", "longer-differs", "both-differ"],
)
def test_every_layer_is_reloaded_at_the_rank_and_scale_eva_gave_it(
    device: str, tmp_path: Path, ranks: dict[str, int]
) -> None:
    model = _nested_eva(device, ranks)
    loaded = _reloaded(model, device, tmp_path)

    assert {n: rank for n, (rank, _) in _scales(loaded).items()} == ranks
    assert _scales(loaded) == _scales(model)  # adjust_scaling_factors: alpha grew with the rank
    x = lucid.randn(4, _D, device=device)
    with lucid.no_grad():
        assert span(loaded(x), model(x)) < 1e-5


def test_a_layer_at_the_default_is_listed_when_another_entry_would_be_read_for_it(
    device: str,
) -> None:
    config = _nested_eva(device, {"q_proj": 3, "block.q_proj": _R}).peft_config["default"]
    assert config.rank_pattern == {"q_proj": 3, "block.q_proj": _R}
    assert config.alpha_pattern == {"q_proj": _ALPHA * 3 / _R, "block.q_proj": float(_ALPHA)}


def test_a_layer_is_not_listed_when_no_entry_would_be_read_for_it(device: str) -> None:
    config = _nested_eva(device, {"q_proj": _R, "block.q_proj": 3}).peft_config["default"]
    assert config.rank_pattern == {"block.q_proj": 3}
    assert config.alpha_pattern == {"block.q_proj": _ALPHA * 3 / _R}


def test_a_scale_that_differs_for_one_name_is_not_read_for_a_name_that_ends_in_it(
    device: str, tmp_path: Path
) -> None:
    # Same ranks everywhere, so ``rank_pattern`` stays empty and only ``alpha_pattern`` can shadow.
    ranks = {"q_proj": _R, "block.q_proj": _R}
    model = _nested_eva(device, ranks, alpha_pattern={"q_proj": 8.0, "block.q_proj": float(_ALPHA)})
    config = model.peft_config["default"]
    assert config.rank_pattern == {}
    assert config.alpha_pattern == {"q_proj": 8.0, "block.q_proj": float(_ALPHA)}

    loaded = _reloaded(model, device, tmp_path)
    assert _scales(loaded) == {"q_proj": (_R, 8.0), "block.q_proj": (_R, float(_ALPHA))}
    x = lucid.randn(4, _D, device=device)
    with lucid.no_grad():
        assert span(loaded(x), model(x)) < 1e-5


def test_the_config_reads_back_what_each_layer_holds(device: str) -> None:
    model = _nested_eva(device, {"q_proj": 3, "block.q_proj": _R})
    config = model.peft_config["default"]
    for name, layer in _layers(model).items():
        rank = _pattern_value(config.rank_pattern, name, config.r)
        alpha = _pattern_value(config.alpha_pattern, name, config.lora_alpha)
        assert (rank, alpha) == (layer.r["default"], layer.lora_alpha["default"])


_NAMES = ("a", "b.a", "c.b.a", "b", "c.b")


def test_recorded_reads_back_every_value_whichever_names_end_in_which() -> None:
    """Every assignment of the default or one of two other values to five names that end in one another."""
    default = 2
    for assigned in itertools.product((default, 3, 5), repeat=len(_NAMES)):
        values = dict(zip(_NAMES, assigned, strict=True))
        pattern = _recorded(values, default)
        for name, value in values.items():
            assert _pattern_value(pattern, name, default) == value, (values, pattern, name)
        assert list(pattern) == [name for name in values if name in pattern]  # in layer order


def test_recorded_lists_what_differs_and_what_would_be_misread_and_nothing_else() -> None:
    assert _recorded({"a": 2, "b.a": 2, "b": 2}, 2) == {}
    assert _recorded({"a": 3, "b": 2, "c.d": 2}, 2) == {"a": 3}
    assert _recorded({"a": 3, "b.a": 2}, 2) == {"a": 3, "b.a": 2}
    assert _recorded({"a": 3, "b.a": 3}, 2) == {"a": 3, "b.a": 3}
    # ``c.b.a`` is read through ``a``'s entry too, whether or not ``b.a`` is listed for itself.
    assert _recorded({"a": 3, "b.a": 2, "c.b.a": 2}, 2) == {"a": 3, "b.a": 2, "c.b.a": 2}
