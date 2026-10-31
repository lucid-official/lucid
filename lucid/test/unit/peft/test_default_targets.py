"""A config that names no ``target_modules`` takes the default for the model's ``model_type``.

The table is the reference's (restricted to the zoo's model types) plus the
zoo's own spelling of the same projections, and a row for every model type
the zoo registers (``neural_ode``, whose weights are raw parameters, has
``DEFAULT_LORA_TARGET_PARAMETERS`` instead).  That each row names modules the
zoo's classes really have is held per family in
``lucid/test/unit/models/test_peft_zoo.py``; here is the rule — where the
default applies, what is stored, what a model with no row is told.
"""

import types
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import LoraConfig, LoraModel, get_peft_model, inject_adapter_in_model
from lucid.peft._target_defaults import DEFAULT_LORA_TARGET_MODULES


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(8, 8)
        self.key = nn.Linear(8, 8)
        self.value = nn.Linear(8, 8)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.value(x) + self.query(x) + self.key(x)


class _Encoder(nn.Module):
    """A model that carries a ``config`` with a ``model_type``, as the zoo's do."""

    def __init__(self, model_type: object = "bert") -> None:
        super().__init__()
        self.config = types.SimpleNamespace(model_type=model_type)
        self.layers = nn.ModuleList([_Attention(), _Attention()])

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _add_adapter(model: nn.Module) -> None:
    get_peft_model(model, LoraConfig(r=2, target_modules=["0"])).add_adapter(
        "second", LoraConfig(r=2)
    )


#: Every way in, each of which must apply the default the same way.
_ENTRY_POINTS: dict[str, Callable[[nn.Module], object]] = {
    "get_peft_model": lambda model: get_peft_model(model, LoraConfig(r=2)),
    "inject_adapter_in_model": lambda model: inject_adapter_in_model(LoraConfig(r=2), model),
    "LoraModel": lambda model: LoraModel(model, LoraConfig(r=2)),
    "add_adapter": _add_adapter,
}


class TestTheTable:
    def test_it_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            DEFAULT_LORA_TARGET_MODULES["bert"] = frozenset({"key"})  # type: ignore[index]
        assert isinstance(DEFAULT_LORA_TARGET_MODULES["bert"], frozenset)

    def test_the_reference_s_rows_are_the_reference_s(self) -> None:
        assert DEFAULT_LORA_TARGET_MODULES["gpt2"] == {"c_attn"}
        assert DEFAULT_LORA_TARGET_MODULES["bert"] == {"query", "value"}
        assert DEFAULT_LORA_TARGET_MODULES["t5"] == {"q", "v"}

    def test_every_row_names_something(self) -> None:
        for names in DEFAULT_LORA_TARGET_MODULES.values():
            assert names and all(isinstance(n, str) and n for n in names)


class TestTheDefaultApplies:
    def test_get_peft_model(self) -> None:
        config = LoraConfig(r=2)
        peft = get_peft_model(_Encoder(), config)
        assert sorted(peft.base_model.targeted_module_names) == [
            "layers.0.query",
            "layers.0.value",
            "layers.1.query",
            "layers.1.value",
        ]
        assert peft.peft_config["default"].target_modules == {"query", "value"}
        assert config.target_modules is None  # the config passed in is left alone

    def test_inject_adapter_in_model(self) -> None:
        model = inject_adapter_in_model(LoraConfig(r=2), _Encoder())
        assert type(model.get_submodule("layers.0.query")).__name__ == "Linear"
        assert hasattr(model.get_submodule("layers.0.query"), "lora_A")
        assert not hasattr(model.get_submodule("layers.0.key"), "lora_A")

    def test_lora_model(self) -> None:
        lora = LoraModel(_Encoder(), LoraConfig(r=2))
        assert lora.peft_config["default"].target_modules == {"query", "value"}

    def test_a_second_adapter_takes_it_too(self) -> None:
        peft = get_peft_model(_Encoder(), LoraConfig(r=2, target_modules=["key"]))
        peft.add_adapter("second", LoraConfig(r=2))
        assert peft.peft_config["second"].target_modules == {"query", "value"}
        assert hasattr(peft.get_base_model().get_submodule("layers.0.query"), "lora_A")

    def test_an_explicit_choice_is_never_replaced(self) -> None:
        peft = get_peft_model(_Encoder(), LoraConfig(r=2, target_modules=["key"]))
        assert sorted(peft.base_model.targeted_module_names) == ["layers.0.key", "layers.1.key"]

    def test_a_default_at_initialisation_changes_nothing(self, device: str) -> None:
        lucid.manual_seed(0)
        model = _Encoder().to(device)
        x = lucid.randn(3, 8, device=device)
        expected = model(x)
        peft = get_peft_model(model, LoraConfig(r=2))
        assert bool((peft(x) == expected).all().item())

    def test_the_other_filters_still_narrow_it(self) -> None:
        config = LoraConfig(r=2, exclude_modules=["layers.1.query"])
        peft = get_peft_model(_Encoder(), config)
        assert "layers.1.query" not in peft.base_model.targeted_module_names
        assert "layers.0.query" in peft.base_model.targeted_module_names

    def test_a_default_that_matches_nothing_is_the_usual_not_found(self) -> None:
        model = _Encoder("gpt2")  # c_attn: not in this model
        with pytest.raises(ValueError, match=r"Target modules \['c_attn'\] not found"):
            get_peft_model(model, LoraConfig(r=2))
        assert all(p.requires_grad for p in model.parameters())


class TestNoDefault:
    @pytest.mark.parametrize("model_type", ["not_a_family", "", None, 3])
    def test_a_model_type_with_no_row_is_told_what_to_do(self, model_type: object) -> None:
        model = _Encoder(model_type)
        with pytest.raises(ValueError, match="Please specify `target_modules` in `peft_config`"):
            get_peft_model(model, LoraConfig(r=2))
        assert all(p.requires_grad for p in model.parameters())
        assert not any(hasattr(m, "lora_A") for m in model.modules())

    def test_the_message_names_the_type_and_the_types_with_a_default(self) -> None:
        with pytest.raises(ValueError, match="model type 'not_a_family'") as caught:
            get_peft_model(_Encoder("not_a_family"), LoraConfig(r=2))
        for model_type in DEFAULT_LORA_TARGET_MODULES:
            assert model_type in str(caught.value)

    @pytest.mark.parametrize("adapt", _ENTRY_POINTS.values(), ids=_ENTRY_POINTS.keys())
    def test_a_bare_module_has_no_type_whichever_way_it_is_adapted(
        self, adapt: Callable[[nn.Module], object]
    ) -> None:
        with pytest.raises(ValueError, match="model type 'custom'"):
            adapt(nn.Sequential(nn.Linear(4, 4)))


class _Planar(nn.Module):
    """A model whose weight is a raw parameter of a plain module, not a layer's."""

    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(model_type="planar")
        self.field = nn.Module()
        self.field.w = nn.Parameter(lucid.randn(6, 8))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x @ self.field.w.T


class TestTheParameterDefault:
    """A type whose default is raw parameters adapts those, and is not told to name modules."""

    @pytest.fixture(autouse=True)
    def _row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(LoraModel, "_default_parameters", {"planar": frozenset({"field.w"})})

    def test_the_parameters_are_adapted(self) -> None:
        peft = get_peft_model(_Planar(), LoraConfig(r=2))
        assert peft.base_model.targeted_parameter_names == ["field.w"]
        assert peft.base_model.targeted_module_names == []
        assert peft.peft_config["default"].target_parameters == ["field.w"]

    def test_an_explicit_choice_is_never_replaced(self) -> None:
        peft = get_peft_model(_Encoder("planar"), LoraConfig(r=2, target_modules=["key"]))
        assert peft.base_model.targeted_parameter_names == []
        assert sorted(peft.base_model.targeted_module_names) == ["layers.0.key", "layers.1.key"]

    def test_a_type_with_a_module_row_only_is_unchanged(self) -> None:
        peft = get_peft_model(_Encoder("bert"), LoraConfig(r=2))
        assert peft.base_model.targeted_parameter_names == []
        assert peft.peft_config["default"].target_modules == {"query", "value"}
