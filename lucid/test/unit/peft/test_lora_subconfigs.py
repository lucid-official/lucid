"""The option dataclasses of LoRA's extensions, and the ``LoraConfig`` fields that hold them.

``ArrowConfig``, ``BdLoraConfig``, ``KasaConfig``, ``MontecloraConfig`` and ``VeloraConfig`` carry the
options of extensions this library does not implement.  They are the reference's dataclasses (fields,
order and defaults are in ``test_reference_signatures``).  Held here: what each one validates, how a file's dict
becomes the dataclass, and that a ``LoraConfig`` which sets one is read and then refused.
"""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import lucid.peft as peft
from lucid.peft import (
    ArrowConfig,
    BdLoraConfig,
    EvaConfig,
    KasaConfig,
    LoraConfig,
    LoraGAConfig,
    MontecloraConfig,
    PeftConfig,
    VeloraConfig,
)
from lucid.peft.tuners.lora import LoraRuntimeConfig
from lucid.peft.tuners.lora._subconfigs import _subconfig

_SNAPSHOT = json.loads((Path(__file__).parent / "_reference_signatures.json").read_text())

#: LoraConfig field -> the dataclass it holds.
_FIELDS: dict[str, type] = {
    "arrow_config": ArrowConfig,
    "kasa_config": KasaConfig,
    "use_bdlora": BdLoraConfig,
    "monteclora_config": MontecloraConfig,
    "velora_config": VeloraConfig,
}


def test_the_names_are_exported() -> None:
    for name in ("ArrowConfig", "BdLoraConfig", "KasaConfig", "MontecloraConfig", "VeloraConfig"):
        assert name in peft.__all__


@pytest.mark.parametrize("name", [cls.__name__ for cls in _FIELDS.values()])
def test_a_default_has_the_reference_fields(name: str) -> None:
    config = getattr(peft, name)()
    reference = {row[0] for row in _SNAPSHOT["fields"][name]}
    assert reference <= set(vars(config))


@pytest.mark.parametrize(
    ("config_class", "fields"),
    [
        (VeloraConfig, {"num_groups": 0}),
        (VeloraConfig, {"scale": 0.0}),
        (VeloraConfig, {"init_type": "zeros"}),
        (ArrowConfig, {"top_k": 0}),
        (ArrowConfig, {"router_temperature": 0.0}),
        (BdLoraConfig, {"target_modules_bd_a": ["q"], "target_modules_bd_b": ["q", "o"]}),
        (KasaConfig, {"beta": -1.0}),
        (KasaConfig, {"gamma": -1.0}),
        (MontecloraConfig, {"num_samples": 0}),
        (MontecloraConfig, {"dirichlet_prior": 0.0}),
        (MontecloraConfig, {"buffer_size": 0}),
    ],
)
def test_a_value_the_reference_refuses_is_refused(
    config_class: Callable[..., object], fields: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        config_class(**fields)


def test_arrows_adapter_names_are_filled_in_not_passed() -> None:
    assert ArrowConfig().task_adapter_names is None and ArrowConfig().gks_adapter_names is None
    build: Callable[..., ArrowConfig] = ArrowConfig
    with pytest.raises(TypeError):
        build(task_adapter_names=["a"])


class TestSubconfigCheck:
    def test_none_an_instance_and_a_dict_are_taken(self) -> None:
        check = _subconfig(KasaConfig)
        kasa = KasaConfig(beta=0.5)
        assert check("kasa_config", None) is None
        assert check("kasa_config", kasa) is kasa
        assert check("kasa_config", {"beta": 0.5}) == kasa

    def test_a_dict_may_hold_the_fields_a_library_fills_in(self) -> None:
        """A file records ``task_adapter_names``; it is not a constructor argument."""
        config = _subconfig(ArrowConfig)(
            "arrow_config",
            {"top_k": 2, "task_adapter_names": ["a", "b"], "gks_adapter_names": None},
        )
        assert config == ArrowConfig(top_k=2)

    def test_another_key_is_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            _subconfig(KasaConfig)("kasa_config", {"alpha": 1.0})

    @pytest.mark.parametrize("bad", [3, "kasa", [1.0], ArrowConfig()])
    def test_anything_else_is_a_type_error(self, bad: object) -> None:
        with pytest.raises(TypeError, match="kasa_config"):
            _subconfig(KasaConfig)("kasa_config", bad)

    def test_it_reads_the_older_options_the_same_way(self) -> None:
        assert _subconfig(EvaConfig)("eva_config", {"rho": 1.5}) == EvaConfig(rho=1.5)
        assert _subconfig(LoraGAConfig)("lora_ga_config", {"stable_gamma": 8}) == LoraGAConfig(
            stable_gamma=8
        )


class TestLoraConfigFields:
    def test_every_extension_field_defaults_to_none(self) -> None:
        config = LoraConfig()
        assert all(getattr(config, name) is None for name in _FIELDS)
        assert all(config.to_dict()[name] is None for name in _FIELDS)

    @pytest.mark.parametrize("field", list(_FIELDS))
    def test_setting_one_is_refused_after_it_is_read(self, field: str) -> None:
        """The options are validated, then the extension is refused: nothing is silently ignored."""
        with pytest.raises(NotImplementedError, match=field):
            LoraConfig(**{field: _FIELDS[field]()})

    @pytest.mark.parametrize("field", list(_FIELDS))
    def test_a_dict_is_read_as_the_dataclass_before_it_is_refused(self, field: str) -> None:
        with pytest.raises(NotImplementedError, match=field):
            LoraConfig.from_dict({"peft_type": "LORA", field: dataclasses.asdict(_FIELDS[field]())})

    @pytest.mark.parametrize("field", list(_FIELDS))
    def test_an_invalid_option_is_reported_as_one(self, field: str) -> None:
        """A bad value fails as the bad value it is, not as the unsupported extension."""
        bad = {
            "arrow_config": {"top_k": 0},
            "kasa_config": {"beta": -1.0},
            "use_bdlora": {"target_modules_bd_a": ["a"], "target_modules_bd_b": ["a"]},
            "monteclora_config": {"num_samples": 0},
            "velora_config": {"num_groups": 0},
        }[field]
        with pytest.raises(ValueError):
            LoraConfig(**{field: bad})

    @pytest.mark.parametrize("field", list(_FIELDS))
    def test_a_value_of_another_type_is_a_type_error(self, field: str) -> None:
        with pytest.raises(TypeError, match=field):
            LoraConfig(**{field: 3})

    def test_the_default_file_still_round_trips(self, tmp_path: Path) -> None:
        config = LoraConfig(r=4, target_modules=["q"])
        config.save_pretrained(tmp_path)
        assert PeftConfig.from_pretrained(tmp_path) == config

    def test_the_runtime_config_is_the_exported_class(self) -> None:
        assert type(LoraConfig().runtime_config) is LoraRuntimeConfig
        with pytest.raises(NotImplementedError, match="ephemeral_gpu_offload"):
            LoraRuntimeConfig(ephemeral_gpu_offload=True)
