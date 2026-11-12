"""The reference's tuner-base members ``inject_adapter`` and ``get_model_config`` (LCD-968c), and X-LoRA's answers to the merge family.

``inject_adapter(model, adapter_name, ...)`` is the reference's spelling of what ``add_adapter``
does: the adapter's config is not an argument but is read from ``peft_config[adapter_name]``.  Here
it is the same path as ``add_adapter`` (``_BaseTuner._inject``), so a method added later is held to
it without a row:

* every registered method that adds a second adapter gets, from ``inject_adapter``, the model
  ``add_adapter`` gives — parameters, flags, config, names the tuner recorded, which adapters are
  active — and, where it merges, on a model that holds merged adapters too (LCD-975's window);
* what has no meaning here is refused by name and leaves the model, and the config the caller set,
  as they were; so is what cannot be injected (no config set, another model, a name already in the
  model, another method's config, no target).

``get_model_config`` reads the config of the wrapped model as a dict, with the reference's stand-in
for a model that has none; every host answers it alike.

X-LoRA mixes its experts per token, so it has nothing to inject, merge or convert: the members are
there with the reference's signatures and each says so.  Its merge departs from the reference on
purpose (which folds the experts in at weight 1 and drops the mixture).
"""

import dataclasses
import inspect
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import (
    IA3Config,
    LoraConfig,
    LoraModel,
    PeftType,
    TrainableTokensConfig,
    TrainableTokensModel,
    UniLoraConfig,
    UniLoraModel,
    XLoraModel,
)
from lucid.peft.tuners._base import _TUNERS
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners._tuner import _BaseTuner
from lucid.test.unit.peft._tuner_recipes import RECIPES, adapted, perturb_adapter, same

_F32 = lucid.float32

#: Every registered method that puts layers into the model and can hold a second adapter beside the first.
_INJECTORS = sorted(
    (t for t, r in RECIPES.items() if r.adds and issubclass(_TUNERS[t], _BaseTuner)), key=str
)
_MERGING = [t for t in _INJECTORS if RECIPES[t].mergeable]

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are trying to modify a model with PEFT for a second time:UserWarning",
    "ignore:Adapter cannot be set when the model is merged:UserWarning",
    "ignore:Unmerge result can be inaccurate:UserWarning",
)


def _net() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))


def _lora() -> LoraModel:
    """A LoRA tuner on layer ``0``, with a second config staged as ``two`` aimed at layer ``2``."""
    host = LoraModel(_net(), LoraConfig(r=2, target_modules=["0"]), "default")
    host.peft_config["two"] = LoraConfig(r=2, target_modules=["2"])
    return host


def _facts(host: _BaseTuner) -> dict[str, object]:
    """Everything an add puts into a model or a tuner, in a form that compares."""
    model = host.model
    return {
        "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "grad": {n: p.requires_grad for n, p in model.named_parameters(remove_duplicate=False)},
        "configs": dict(host.peft_config),
        "active": host.active_adapters,
        "modules": list(host.targeted_module_names),
        "parameters": list(host.targeted_parameter_names),
        "layers": [(key, type(layer).__name__) for key, layer in _tuner_layers(model)],
    }


def _same_facts(a: dict[str, object], b: dict[str, object]) -> None:
    for name in a:
        if name == "state":
            assert isinstance(a[name], dict) and isinstance(b[name], dict)
            assert a[name].keys() == b[name].keys()
            assert all(same(a[name][k], b[name][k]) for k in a[name])
        else:
            assert a[name] == b[name], name


@pytest.mark.parametrize("peft_type", _INJECTORS, ids=lambda t: t.name)
def test_inject_adapter_gives_the_model_add_adapter_gives(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    recipe = RECIPES[peft_type]
    facts = []
    for staged in (False, True):
        host = adapted(recipe, _F32, device, tmp_path)[1].base_model
        assert isinstance(host, _BaseTuner)
        second = recipe.second(_F32, device, tmp_path)
        lucid.manual_seed(7)
        if staged:
            host.peft_config["b"] = second
            host.inject_adapter(host.model, "b")
        else:
            host.add_adapter("b", second)
        facts.append(_facts(host))
    _same_facts(*facts)


@pytest.mark.parametrize("peft_type", _MERGING, ids=lambda t: t.name)
def test_inject_adapter_on_a_merged_model_leaves_it_merged_and_gives_what_add_adapter_gives(
    peft_type: PeftType, device: str, tmp_path: Path
) -> None:
    recipe = RECIPES[peft_type]
    facts, folded = [], []
    for staged in (False, True):
        peft = adapted(recipe, _F32, device, tmp_path)[1]
        host = peft.base_model
        assert isinstance(host, _BaseTuner)
        perturb_adapter(peft, "default", seed=5)
        host.merge_adapter()
        before = {k: v.detach().clone() for k, v in host.model.state_dict().items()}
        second = recipe.second(_F32, device, tmp_path)
        lucid.manual_seed(7)
        if staged:
            host.peft_config["b"] = second
            host.inject_adapter(host.model, "b")
        else:
            host.add_adapter("b", second)
        # Merged as they were, to the bit: a base weight is what it was before the add.
        after = host.model.state_dict()
        assert all(same(after[k], v) for k, v in before.items())
        assert all(layer.merged for _, layer in _tuner_layers(host.model))
        facts.append(_facts(host))
        folded.append([list(layer.merged_adapters) for _, layer in _tuner_layers(host.model)])
    assert folded[0] == folded[1]
    _same_facts(*facts)


class TestInjectAdapter:
    def test_the_adapter_is_beside_the_held_one_frozen_and_inactive(self) -> None:
        host = _lora()
        host.inject_adapter(host.model, "two")
        assert host.active_adapters == ["default"]
        assert host.targeted_module_names == ["0", "2"]
        assert host.peft_config["two"].target_modules == {"2"}
        second = host.model[2]
        assert second.lora_A["two"].weight.requires_grad is False
        assert host.model[0].lora_A["default"].weight.requires_grad is True

    def test_the_config_kept_is_the_stored_one_not_the_object_set(self) -> None:
        host = _lora()
        staged = host.peft_config["two"]
        host.inject_adapter(host.model, "two")
        assert host.peft_config["two"] is not staged
        assert host.peft_config["two"] == staged

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("low_cpu_mem_usage", True),
            ("autocast_adapter_dtype", False),
            ("state_dict", {"0.lora_A.two.weight": lucid.zeros(2, 4)}),
        ],
    )
    def test_what_has_no_meaning_here_is_refused_and_nothing_changes(
        self, name: str, value: object
    ) -> None:
        host = _lora()
        staged = host.peft_config["two"]
        before = _facts(host)
        with pytest.raises(NotImplementedError, match=name):
            host.inject_adapter(host.model, "two", **{name: value})
        assert host.peft_config["two"] is staged
        _same_facts(before, _facts(host))

    def test_the_defaults_are_accepted(self) -> None:
        host = _lora()
        host.inject_adapter(
            host.model, "two", autocast_adapter_dtype=True, low_cpu_mem_usage=False, state_dict=None
        )
        assert "two" in host.model[2].lora_A

    def test_a_config_must_be_set(self) -> None:
        host = _lora()
        before = _facts(host)
        with pytest.raises(ValueError, match=r"peft_config\['other'\]"):
            host.inject_adapter(host.model, "other")
        _same_facts(before, _facts(host))

    def test_it_adapts_the_model_the_tuner_holds(self) -> None:
        host = _lora()
        staged = host.peft_config["two"]
        before = _facts(host)
        with pytest.raises(ValueError, match="this tuner holds"):
            host.inject_adapter(_net(), "two")
        assert host.peft_config["two"] is staged
        _same_facts(before, _facts(host))

    def test_an_adapter_the_model_holds_is_not_injected_again(self) -> None:
        host = _lora()
        held = host.peft_config["default"]
        before = _facts(host)
        with pytest.raises(ValueError, match="already exists"):
            host.inject_adapter(host.model, "default")
        assert host.peft_config["default"] is held
        _same_facts(before, _facts(host))

    def test_another_methods_config_is_refused(self) -> None:
        host = _lora()
        host.peft_config["two"] = IA3Config(target_modules=["2"], feedforward_modules=["2"])
        before = _facts(host)
        with pytest.raises(ValueError, match="different peft types"):
            host.inject_adapter(host.model, "two")
        _same_facts(before, _facts(host))

    def test_a_config_that_targets_nothing_leaves_the_model_and_the_config_as_they_were(
        self,
    ) -> None:
        host = _lora()
        host.peft_config["two"] = LoraConfig(r=2, target_modules=["no_such_module"])
        staged = host.peft_config["two"]
        before = _facts(host)
        with pytest.raises(ValueError, match="no_such_module"):
            host.inject_adapter(host.model, "two")
        assert host.peft_config["two"] is staged
        _same_facts(before, _facts(host))

    def test_a_tuner_class_that_overrides_it_inherits_the_rest_of_the_path(self) -> None:
        # UniLoRA's reference override draws its shared vector before injecting and assigns the
        # layers' indices after; here both are steps of the injection itself.
        net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
        host = UniLoraModel(
            net, UniLoraConfig(r=2, theta_d_length=16, target_modules=["0"]), "default"
        )
        host.peft_config["two"] = UniLoraConfig(r=2, theta_d_length=16, target_modules=["2"])
        host.inject_adapter(net, "two")
        assert set(host.unilora_theta_d) == {"default", "two"}
        assert "two" in net[2].unilora_indices_A and "two" in net[2].unilora_indices_B
        assert type(host).inject_adapter is _BaseTuner.inject_adapter


class TestTrainableTokensInjectAdapter:
    """The reference's override takes ``**kwargs`` where its base takes ``state_dict``."""

    @staticmethod
    def _host() -> TrainableTokensModel:
        net = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 10, bias=False))
        host = TrainableTokensModel(
            net, TrainableTokensConfig(token_indices=[1], target_modules=["0"]), "default"
        )
        host.peft_config["two"] = TrainableTokensConfig(token_indices=[2], target_modules=["0"])
        return host

    def test_it_takes_the_references_parameters(self) -> None:
        names = list(inspect.signature(TrainableTokensModel.inject_adapter).parameters)
        assert names == [
            *("self", "model", "adapter_name", "autocast_adapter_dtype"),
            *("low_cpu_mem_usage", "kwargs"),
        ]

    def test_it_injects_as_the_base_does(self) -> None:
        host = self._host()
        host.inject_adapter(host.model, "two", state_dict=None)
        assert set(host.model[0].trainable_tokens_delta) == {"default", "two"}

    def test_state_dict_is_refused_by_name(self) -> None:
        host = self._host()
        with pytest.raises(NotImplementedError, match="state_dict"):
            host.inject_adapter(host.model, "two", state_dict={"x": lucid.zeros(1)})
        assert "two" not in host.model[0].trainable_tokens_delta

    def test_any_other_keyword_is_a_type_error(self) -> None:
        host = self._host()
        with pytest.raises(TypeError, match="no_such_option"):
            host.inject_adapter(host.model, "two", no_such_option=1)

    def test_a_table_another_module_shares_is_refused_and_the_model_left_alone(self) -> None:
        net = create_model(
            "gpt2_small_lm",
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=8,
        )
        assert net.lm_head.weight is net.transformer.wte.weight
        host = TrainableTokensModel(
            net,
            TrainableTokensConfig(token_indices=[1], target_modules=["transformer.wpe"]),
            "default",
        )
        host.peft_config["two"] = TrainableTokensConfig(
            token_indices=[2], target_modules=["transformer.wte"]
        )
        with pytest.raises(NotImplementedError, match="lm_head"):
            host.inject_adapter(net, "two")
        assert type(net.transformer.wte) is nn.Embedding


@dataclasses.dataclass
class _Config:
    model_type: str = "toy"
    width: int = 3


class _WithToDict:
    model_type = "toy"

    def to_dict(self) -> dict[str, object]:
        return {"model_type": "toy", "width": 3}


class TestGetModelConfig:
    def test_a_model_without_a_config_has_the_references_stand_in(self) -> None:
        assert LoraModel.get_model_config(nn.Linear(2, 2)) == {"model_type": "custom"}

    def test_a_config_that_is_none_is_no_config(self) -> None:
        net = nn.Linear(2, 2)
        net.config = None
        assert LoraModel.get_model_config(net) == {"model_type": "custom"}

    def test_the_zoos_config_is_read_through_its_to_dict(self) -> None:
        net = create_model(
            "gpt2_small_lm",
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=8,
        )
        read = LoraModel.get_model_config(net)
        assert read == net.config.to_dict() and read["model_type"] == net.config.model_type

    def test_an_object_with_to_dict_is_converted_by_it(self) -> None:
        net = nn.Linear(2, 2)
        net.config = _WithToDict()
        assert LoraModel.get_model_config(net) == {"model_type": "toy", "width": 3}

    def test_a_dataclass_is_converted_field_by_field(self) -> None:
        net = nn.Linear(2, 2)
        net.config = _Config(width=5)
        assert LoraModel.get_model_config(net) == {"model_type": "toy", "width": 5}

    def test_a_dict_is_copied_so_the_model_keeps_its_own(self) -> None:
        net = nn.Linear(2, 2)
        held = {"model_type": "toy"}
        net.config = held
        read = LoraModel.get_model_config(net)
        assert read == held and read is not held
        read["model_type"] = "changed"
        assert held == {"model_type": "toy"}

    def test_a_config_that_is_not_a_config_is_a_type_error(self) -> None:
        net = nn.Linear(2, 2)
        net.config = 3
        with pytest.raises(TypeError, match="int"):
            LoraModel.get_model_config(net)

    @pytest.mark.parametrize("peft_type", sorted(_TUNERS, key=str), ids=lambda t: t.name)
    def test_every_host_answers_alike_on_the_class_and_on_an_instance(
        self, peft_type: PeftType, tmp_path: Path
    ) -> None:
        host = _TUNERS[peft_type]
        assert isinstance(inspect.getattr_static(host, "get_model_config"), staticmethod)
        net = nn.Linear(2, 2)
        net.config = _Config(width=7)
        assert host.get_model_config(net) == {"model_type": "toy", "width": 7}
        recipe = RECIPES[peft_type]
        instance = adapted(recipe, _F32, "cpu", tmp_path)[1].base_model
        assert instance.get_model_config(net) == {"model_type": "toy", "width": 7}


class TestXLoraMergeFamily:
    """Nothing of X-LoRA's merges, converts or injects: the members are the reference's, and say why."""

    @staticmethod
    def _host(tmp_path: Path) -> XLoraModel:
        host = adapted(RECIPES[PeftType.XLORA], _F32, "cpu", tmp_path)[1].base_model
        assert isinstance(host, XLoraModel)
        return host

    def test_merge_adapter_is_refused_for_every_way_of_asking(self, tmp_path: Path) -> None:
        host = self._host(tmp_path)
        for arguments in ({}, {"adapter_names": ["0"]}, {"safe_merge": True}):
            with pytest.raises(NotImplementedError, match="per token"):
                host.merge_adapter(**arguments)

    def test_a_refused_merge_changes_nothing(self, tmp_path: Path) -> None:
        host = self._host(tmp_path)
        before = {k: v.detach().clone() for k, v in host.model.state_dict().items()}
        with pytest.raises(NotImplementedError):
            host.merge_adapter()
        assert not any(layer.merged for _, layer in _tuner_layers(host.model))
        after = host.model.state_dict()
        assert after.keys() == before.keys() and all(same(after[k], v) for k, v in before.items())

    def test_unmerge_adapter_has_nothing_to_take_out(self, tmp_path: Path) -> None:
        host = self._host(tmp_path)
        before = {k: v.detach().clone() for k, v in host.model.state_dict().items()}
        assert host.unmerge_adapter() is None
        after = host.model.state_dict()
        assert after.keys() == before.keys() and all(same(after[k], v) for k, v in before.items())

    @pytest.mark.parametrize("name", ["default", "no_such_adapter"])
    def test_the_adapter_does_not_convert_to_lora(self, name: str, tmp_path: Path) -> None:
        host = self._host(tmp_path)
        assert host.supports_lora_conversion() is False
        assert host.supports_lora_conversion(name) is False

    def test_the_experts_alone_would_convert_which_is_why_this_answer_is_not_the_references(
        self, tmp_path: Path
    ) -> None:
        # The reference reads the answer off the experts' LoRA layers, which are all convertible.
        host = self._host(tmp_path)
        layers = [layer for _, layer in _tuner_layers(host.model)]
        assert layers and all(layer.supports_lora_conversion("0") for layer in layers)
        assert host.supports_lora_conversion() is False

    def test_inject_adapter_is_refused_and_the_model_left_as_it_was(self, tmp_path: Path) -> None:
        host = self._host(tmp_path)
        names = [n for n, _ in host.model.named_modules()]
        with pytest.raises(NotImplementedError, match="config.adapters"):
            host.inject_adapter(host.model, "other")
        assert names == [n for n, _ in host.model.named_modules()]

    def test_the_signatures_are_the_references(self) -> None:
        def names(member: str) -> list[str]:
            return list(inspect.signature(getattr(XLoraModel, member)).parameters)

        assert names("merge_adapter") == ["self", "adapter_names", "safe_merge"]
        assert names("unmerge_adapter") == ["self"]
        assert names("supports_lora_conversion") == ["self", "adapter_name"]
        assert names("inject_adapter") == [
            *("self", "model", "adapter_name", "autocast_adapter_dtype"),
            *("low_cpu_mem_usage", "state_dict"),
        ]
