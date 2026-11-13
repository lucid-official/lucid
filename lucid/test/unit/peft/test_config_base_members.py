"""The config base's three reference members: ``check_kwargs``, ``from_peft_type``, ``is_adaption_prompt``.

Every config class inherits them from ``PeftConfig``, so the guards walk the registry
(``_CONFIG_CLASSES``) rather than a list: a method added later is held to them the day it is
defined.  The ``parity`` tests compare the results with the reference adapter library's.
"""

import json
import warnings
from pathlib import Path

import pytest

from lucid.peft import (
    AdaptionPromptConfig,
    LoraConfig,
    OFTConfig,
    PeftConfig,
    PeftType,
    PromptTuningConfig,
)
from lucid.peft._config import _CONFIG_CLASSES
from lucid.peft.tuners.oft._config import _predates
from lucid.test.unit.peft._parity import adapter_library

_SNAPSHOT = json.loads((Path(__file__).parent / "_reference_signatures.json").read_text())

#: The two configs that cannot be built with no arguments, and what each needs.
_NEEDS: dict[PeftType, dict[str, object]] = {
    PeftType.ADALORA: {"total_step": 100},
    PeftType.CPT: {"task_type": "CAUSAL_LM"},
}

_ALL = sorted(_CONFIG_CLASSES.values(), key=lambda c: c.__name__)

#: The exported configs whose ``is_adaption_prompt`` the reference defines itself (the base's is
#: False); read off the snapshot so the set is the reference's, not this file's.
_MARKED = {
    qual.partition(".")[0]
    for qual, entry in _SNAPSHOT["signatures"].items()
    if qual.endswith(".is_adaption_prompt") and not str(entry["defined_in"]).endswith("Mixin")
}


def _make(config_class: type[PeftConfig]) -> PeftConfig:
    return config_class(**_NEEDS.get(config_class._peft_type or PeftType.LORA, {}))


def _ids(config_class: type[PeftConfig]) -> str:
    return config_class.__name__


class TestRegistryGuard:
    def test_the_registry_is_walked_in_full(self) -> None:
        assert {c._peft_type for c in _ALL} == set(PeftType)

    @pytest.mark.parametrize("config_class", _ALL, ids=_ids)
    def test_from_peft_type_round_trips_a_config(self, config_class: type[PeftConfig]) -> None:
        config = _make(config_class)
        again = config_class.from_peft_type(**config.to_dict())
        assert type(again) is config_class
        assert again == config

    @pytest.mark.parametrize("config_class", _ALL, ids=_ids)
    def test_the_base_dispatches_on_peft_type_to_every_class(
        self, config_class: type[PeftConfig]
    ) -> None:
        config = _make(config_class)
        built = PeftConfig.from_peft_type(**config.to_dict())
        assert type(built) is config_class and built == config

    @pytest.mark.parametrize("config_class", _ALL, ids=_ids)
    def test_a_named_type_wins_over_the_calling_class(self, config_class: type[PeftConfig]) -> None:
        config = _make(config_class)
        assert LoraConfig.from_peft_type(**config.to_dict()) == config

    @pytest.mark.parametrize("config_class", _ALL, ids=_ids)
    def test_check_kwargs_hands_a_plain_dict_back(self, config_class: type[PeftConfig]) -> None:
        if config_class is OFTConfig:
            pytest.skip("OFT checks its kwargs; see TestOFTCheckKwargs")
        fields = {"r": 4, "target_modules": ["q"], "peft_type": "LORA"}
        assert config_class.check_kwargs(**fields) == fields
        assert config_class.check_kwargs() == {}

    @pytest.mark.parametrize("config_class", _ALL, ids=_ids)
    def test_is_adaption_prompt_is_true_where_the_reference_says(
        self, config_class: type[PeftConfig]
    ) -> None:
        assert _make(config_class).is_adaption_prompt is (config_class.__name__ in _MARKED)

    def test_the_marked_set_is_the_adaption_prompt(self) -> None:
        assert _MARKED == {"AdaptionPromptConfig"}


class TestFromPeftType:
    def test_a_peft_type_is_a_member_or_its_string(self) -> None:
        by_member = PeftConfig.from_peft_type(peft_type=PeftType.LORA, target_modules=["q"])
        by_string = PeftConfig.from_peft_type(peft_type="LORA", target_modules=["q"])
        assert by_member == by_string == LoraConfig(target_modules=["q"])

    def test_without_a_type_the_calling_class_is_built(self) -> None:
        config = OFTConfig.from_peft_type(oft_block_size=8, target_modules=["q"])
        assert type(config) is OFTConfig and config.oft_block_size == 8

    @pytest.mark.parametrize("named", ["NOT_A_METHOD", "lora", "", None, 3])
    def test_an_unknown_peft_type_is_a_key_error(self, named: object) -> None:
        with pytest.raises(KeyError, match="is not a method of lucid.peft"):
            PeftConfig.from_peft_type(peft_type=named)

    def test_the_base_with_no_type_names_what_is_missing(self) -> None:
        with pytest.raises(TypeError, match=r"missing required keys: \{'peft_type'\}"):
            PeftConfig.from_peft_type(r=4)
        with pytest.raises(TypeError, match=r"missing required keys: \{'peft_type'\}"):
            PeftConfig.from_peft_type()

    def test_a_key_that_is_off_is_dropped_with_a_warning_when_unknown(self) -> None:
        with pytest.warns(UserWarning, match=r"Unexpected keyword arguments \['future_flag'\]"):
            config = PeftConfig.from_peft_type(
                peft_type="LORA", target_modules=["q"], future_flag=None
            )
        assert config == LoraConfig(target_modules=["q"])

    def test_a_reference_only_key_that_is_off_is_dropped_quietly(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config = PeftConfig.from_peft_type(
                peft_type="LORA", target_modules=["q"], loftq_config={}, peft_version="0.21.2"
            )
        assert config == LoraConfig(target_modules=["q"])

    def test_a_key_that_is_on_is_refused_not_dropped(self) -> None:
        with pytest.raises(NotImplementedError, match="future_flag"):
            PeftConfig.from_peft_type(peft_type="LORA", target_modules=["q"], future_flag=True)


@pytest.mark.parametrize(
    ("version", "older"),
    [
        ("0.0.0", True),
        ("0.17.1", True),
        ("0.17.9.post3", True),
        ("0.18", False),
        ("0.18.0", False),
        ("0.18.0.0", False),
        ("0.18.0rc1", True),
        ("0.18.0-rc1", True),
        ("0.18.0a2", True),
        ("0.18.0.dev0", True),
        ("0.18.0rc1.dev0", True),
        ("0.18.0.post1", False),
        ("0.18.0.post1.dev0", False),
        ("0.18.1.dev0", False),
        ("0.19.0", False),
        ("1.0", False),
        ("v0.18.0", False),
        ("0.17.0+local.1", True),
        ("0.18.0+local.1", False),
    ],
)
def test_a_version_is_ordered_as_pep_440_orders_it(version: str, older: bool) -> None:
    assert _predates(version, (0, 18, 0)) is older


@pytest.mark.parametrize("text", ["", "abc", "0.18.0.", "0..1", "0.18.0x", "1.0.0.foo"])
def test_a_text_that_is_no_version_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="Invalid version"):
        _predates(text, (0, 18, 0))


class TestOFTCheckKwargs:
    def test_a_file_without_a_block_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="OFT has been updated since version 0.14.0"):
            OFTConfig.check_kwargs(r=8, module_dropout=0.0)

    def test_the_kwargs_come_back_when_they_pass(self) -> None:
        fields = {"oft_block_size": 32, "peft_version": "0.18.0", "use_cayley_neumann": True}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert OFTConfig.check_kwargs(**fields) == fields

    @pytest.mark.parametrize(
        "version", ["0.0.0", "0.17.1", "0.18.0.dev0", "0.18.0rc1", "0.17.1.dev0@abc123"]
    )
    def test_the_old_parameterisation_warns(self, version: str) -> None:
        fields: dict[str, object] = {
            "oft_block_size": 32,
            "use_cayley_neumann": True,
            "peft_version": version,
        }
        with pytest.warns(UserWarning, match="Cayley-Neumann parameterization"):
            assert OFTConfig.check_kwargs(**fields) == fields

    @pytest.mark.parametrize("recorded", [False, True])
    def test_a_file_with_no_version_does_not_warn(self, recorded: bool) -> None:
        """The reference reads a missing version as 0.0.0; this library writes none, so it cannot."""
        fields: dict[str, object] = {"oft_block_size": 32, "use_cayley_neumann": True}
        if recorded:
            fields["peft_version"] = None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert OFTConfig.check_kwargs(**fields) == fields

    @pytest.mark.parametrize("version", ["0.18.0", "0.18.1.dev0@abc123", "0.21.2", "0.18.0.post1"])
    def test_a_file_from_the_new_form_is_quiet(self, version: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            OFTConfig.check_kwargs(oft_block_size=32, use_cayley_neumann=True, peft_version=version)

    def test_the_old_form_is_not_warned_about_when_it_is_off(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            OFTConfig.check_kwargs(oft_block_size=32, use_cayley_neumann=False)
            OFTConfig.check_kwargs(oft_block_size=32)

    def test_a_version_that_is_no_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Invalid version"):
            OFTConfig.check_kwargs(oft_block_size=32, use_cayley_neumann=True, peft_version="x")
        with pytest.raises(TypeError, match="peft_version"):
            OFTConfig.check_kwargs(oft_block_size=32, use_cayley_neumann=True, peft_version=18)

    def test_the_default_config_of_this_library_is_not_refused(self) -> None:
        """``to_dict`` always carries ``oft_block_size``, so only a legacy file fails the check."""
        assert "oft_block_size" in OFTConfig(target_modules=["q"]).to_dict()


class TestAdaptionPromptMark:
    def test_the_base_is_not_an_adaption_prompt(self) -> None:
        assert LoraConfig(target_modules=["q"]).is_adaption_prompt is False

    def test_the_adaption_prompt_is(self) -> None:
        config = AdaptionPromptConfig(adapter_len=4, adapter_layers=2)
        assert config.is_adaption_prompt is True
        assert config.is_prompt_learning is False


# -- against the reference adapter library ------------------------------------

#: ``(Lucid class, kwargs both libraries take)``: a sample covering a plain tuner, the OFT override,
#: the adaption prompt, and a prompt-learning method.
_SAMPLE: list[tuple[type[PeftConfig], dict[str, object]]] = [
    (LoraConfig, {"r": 4, "target_modules": ["q"]}),
    (OFTConfig, {"oft_block_size": 4, "target_modules": ["q"]}),
    (AdaptionPromptConfig, {"adapter_len": 4, "adapter_layers": 2, "task_type": "CAUSAL_LM"}),
]


def _their_class(library: object, name: str) -> type:
    cls = getattr(library, name)
    assert isinstance(cls, type)
    return cls


@pytest.mark.parity
@pytest.mark.parametrize(
    ("config_class", "options"), _SAMPLE, ids=lambda v: getattr(v, "__name__", "")
)
def test_is_adaption_prompt_agrees_with_the_reference(
    config_class: type[PeftConfig], options: dict[str, object]
) -> None:
    library = adapter_library()
    theirs = _their_class(library, config_class.__name__)(**options)
    assert config_class(**options).is_adaption_prompt is theirs.is_adaption_prompt


@pytest.mark.parity
def test_is_adaption_prompt_is_false_on_a_prompt_learning_config_in_both() -> None:
    library = adapter_library()
    theirs = library.PromptTuningConfig(num_virtual_tokens=2)
    mine = PromptTuningConfig(num_virtual_tokens=2)
    assert mine.is_adaption_prompt is theirs.is_adaption_prompt is False
    assert mine.is_prompt_learning is theirs.is_prompt_learning is True


@pytest.mark.parity
@pytest.mark.parametrize(
    ("config_class", "options"), _SAMPLE, ids=lambda v: getattr(v, "__name__", "")
)
def test_from_peft_type_builds_what_the_reference_builds(
    config_class: type[PeftConfig], options: dict[str, object], tmp_path: Path
) -> None:
    library = adapter_library()
    name = config_class.__name__
    theirs_class = _their_class(library, name)
    theirs_class(**options).save_pretrained(tmp_path)
    raw = PeftConfig.from_json_file(str(tmp_path / "adapter_config.json"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mine = PeftConfig.from_peft_type(**raw)
        theirs = _their_class(library, "PeftConfig").from_peft_type(**raw)
    assert type(mine).__name__ == type(theirs).__name__ == name
    ours, reference = mine.to_dict(), theirs.to_dict()
    for key in set(ours) & set(reference):
        expected = reference[key]
        expected = sorted(expected) if isinstance(expected, set) else expected
        got = ours[key]
        assert str(got) == str(expected) or got == expected, key


@pytest.mark.parity
def test_an_unknown_or_missing_peft_type_fails_as_the_reference_does() -> None:
    library = adapter_library()
    theirs = _their_class(library, "PeftConfig")
    with pytest.raises(KeyError):
        theirs.from_peft_type(peft_type="NOT_A_METHOD")
    with pytest.raises(KeyError):
        PeftConfig.from_peft_type(peft_type="NOT_A_METHOD")
    with pytest.raises(TypeError, match="missing required keys"):
        theirs.from_peft_type(r=4)
    with pytest.raises(TypeError, match="missing required keys"):
        PeftConfig.from_peft_type(r=4)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("config_class", "options"), _SAMPLE, ids=lambda v: getattr(v, "__name__", "")
)
def test_check_kwargs_returns_what_the_reference_returns(
    config_class: type[PeftConfig], options: dict[str, object]
) -> None:
    library = adapter_library()
    fields = {**options, "peft_version": "0.21.2", "use_cayley_neumann": False}
    theirs_class = _their_class(library, config_class.__name__)
    assert config_class.check_kwargs(**fields) == theirs_class.check_kwargs(**fields) == fields


@pytest.mark.parity
@pytest.mark.parametrize(
    "version",
    ["0.0.0", "0.17.1", "0.18.0", "0.18.0.dev0", "0.18.1.dev0@abc123", "0.18.0.post1", "0.21.2"],
)
def test_oft_check_kwargs_warns_as_the_reference_does(version: str) -> None:
    library = adapter_library()
    fields: dict[str, object] = {
        "oft_block_size": 4,
        "use_cayley_neumann": True,
        "peft_version": version,
    }
    with warnings.catch_warnings(record=True) as theirs:
        warnings.simplefilter("always")
        library.OFTConfig.check_kwargs(**fields)
    with warnings.catch_warnings(record=True) as mine:
        warnings.simplefilter("always")
        OFTConfig.check_kwargs(**fields)
    assert len(mine) == len(theirs)
    assert [w.category for w in mine] == [w.category for w in theirs]


@pytest.mark.parity
def test_oft_check_kwargs_is_quiet_on_a_missing_version_where_the_reference_warns() -> None:
    """The one deliberate difference: this library writes no ``peft_version``, so it cannot warn on its absence."""
    library = adapter_library()
    fields: dict[str, object] = {"oft_block_size": 4, "use_cayley_neumann": True}
    with warnings.catch_warnings(record=True) as theirs:
        warnings.simplefilter("always")
        library.OFTConfig.check_kwargs(**fields)
    with warnings.catch_warnings(record=True) as mine:
        warnings.simplefilter("always")
        OFTConfig.check_kwargs(**fields)
    assert len(theirs) == 1
    assert mine == []


@pytest.mark.parity
def test_oft_check_kwargs_refuses_what_the_reference_refuses() -> None:
    library = adapter_library()
    with pytest.raises(ValueError):
        library.OFTConfig.check_kwargs(r=8)
    with pytest.raises(ValueError):
        OFTConfig.check_kwargs(r=8)
