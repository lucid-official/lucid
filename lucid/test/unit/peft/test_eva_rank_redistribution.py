"""EVA redistributes the rank budget over the layers (LCD-977).

``initialize_lora_eva_weights`` gives each layer the rank its activations call for, so a layer's
adapter is rebuilt at a size the layer already holds an adapter of.  ``LoraLayer.update_layer``
refuses a name the layer holds, and that was the only way in: a redistribution that moved any
layer off ``r`` raised ``Adapter with name 'default' already exists.``  Each test here gives the
two layers inputs with unlike spectra, so the ranks that come back differ from ``r`` — and from
each other.
"""

from pathlib import Path

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.peft import (
    EvaConfig,
    LoraConfig,
    PeftModel,
    get_eva_state_dict,
    get_peft_model,
    initialize_lora_eva_weights,
)
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners.lora import Linear, LoraLayer
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
