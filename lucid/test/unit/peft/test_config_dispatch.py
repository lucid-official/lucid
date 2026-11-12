"""``from_dict``, ``from_pretrained`` and ``get_peft_config`` read the method a file names, and a file is checked first.

Whichever config class they are called on, the file's ``peft_type`` picks the class (the
reference's ``from_peft_type``), through the one registry; ``from_pretrained`` hands the file's
keys to that class's ``check_kwargs`` before any key is dropped for being unknown.  The guards walk
the registry (``_CONFIG_CLASSES``), so a method added later is held to them the day it is defined.
``get_peft_config`` is a call to ``PeftConfig.from_peft_type``, with no lookup of its own.
"""

import json
import warnings
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    LoraConfig,
    OFTConfig,
    PeftConfig,
    PeftModel,
    PeftType,
    get_peft_config,
    get_peft_model,
)
from lucid.peft import _config as config_module
from lucid.peft._config import _CONFIG_CLASSES
from lucid.test.unit.peft._parity import adapter_library

#: The two configs that cannot be built with no arguments, and what each needs.
_NEEDS: dict[PeftType, dict[str, object]] = {
    PeftType.ADALORA: {"total_step": 100},
    PeftType.CPT: {"task_type": "CAUSAL_LM"},
}

_ALL = sorted(_CONFIG_CLASSES.values(), key=lambda c: c.__name__)

#: Classes the guards call the readers on: the base, the one implemented method, and another.
_CALLERS: list[type[PeftConfig]] = [PeftConfig, LoraConfig, OFTConfig]

_CAYLEY = "Cayley-Neumann parameterization"


def _make(config_class: type[PeftConfig]) -> PeftConfig:
    return config_class(**_NEEDS.get(config_class._peft_type or PeftType.LORA, {}))


def _ids(value: object) -> str:
    return getattr(value, "__name__", str(value))


def _write(directory: Path, payload: dict[str, object]) -> Path:
    """An ``adapter_config.json`` holding ``payload``, in ``directory``."""
    (directory / "adapter_config.json").write_text(json.dumps(payload))
    return directory


def _oft_file(directory: Path, **changes: object) -> Path:
    """The file an OFT config saves, with ``changes`` laid over it (a ``None`` value removes the key)."""
    OFTConfig(oft_block_size=4, target_modules=["q"]).save_pretrained(directory)
    stored = json.loads((directory / "adapter_config.json").read_text())
    stored.update(changes)
    return _write(directory, {k: v for k, v in stored.items() if v is not None})


def _cayley_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if _CAYLEY in str(w.message)]


class TestFromDict:
    @pytest.mark.parametrize("caller", _CALLERS, ids=_ids)
    @pytest.mark.parametrize("config_class", _ALL, ids=_ids)
    def test_a_dict_of_another_method_is_that_methods_config(
        self, caller: type[PeftConfig], config_class: type[PeftConfig]
    ) -> None:
        config = _make(config_class)
        built = caller.from_dict(config.to_dict())
        assert type(built) is config_class
        assert built == config

    def test_the_fields_are_those_of_the_class_the_dict_names(self) -> None:
        built = LoraConfig.from_dict(
            {"peft_type": "OFT", "oft_block_size": 8, "r": 0, "target_modules": ["q"]}
        )
        assert isinstance(built, OFTConfig)
        assert built.oft_block_size == 8

    def test_a_member_of_the_enum_names_a_method_too(self) -> None:
        built = LoraConfig.from_dict({"peft_type": PeftType.OFT, "target_modules": ["q"]})
        assert type(built) is OFTConfig

    @pytest.mark.parametrize(
        ("caller", "expected"),
        [(PeftConfig, LoraConfig), (LoraConfig, LoraConfig), (OFTConfig, OFTConfig)],
        ids=_ids,
    )
    def test_a_dict_that_names_no_method_is_read_as_the_calling_class(
        self, caller: type[PeftConfig], expected: type[PeftConfig]
    ) -> None:
        assert type(caller.from_dict({"target_modules": ["q"]})) is expected

    @pytest.mark.parametrize("caller", _CALLERS, ids=_ids)
    @pytest.mark.parametrize("named", ["NOT_A_METHOD", "lora", "", 3])
    def test_a_method_this_library_lacks_is_a_key_error(
        self, caller: type[PeftConfig], named: object
    ) -> None:
        with pytest.raises(KeyError, match="is not a method of lucid.peft"):
            caller.from_dict({"peft_type": named})

    def test_a_feature_this_library_lacks_is_still_refused_across_methods(self) -> None:
        with pytest.raises(NotImplementedError, match="use_qalora"):
            OFTConfig.from_dict({"peft_type": "LORA", "target_modules": ["q"], "use_qalora": True})


class TestFromPretrainedDispatch:
    @pytest.mark.parametrize("caller", _CALLERS, ids=_ids)
    @pytest.mark.parametrize("config_class", _ALL, ids=_ids)
    def test_a_file_of_another_method_is_that_methods_config(
        self, caller: type[PeftConfig], config_class: type[PeftConfig], tmp_path: Path
    ) -> None:
        config = _make(config_class)
        config.save_pretrained(tmp_path)
        loaded = caller.from_pretrained(tmp_path)
        assert type(loaded) is config_class
        assert loaded == config

    @pytest.mark.parametrize(
        ("caller", "payload", "expected"),
        [
            (PeftConfig, {"target_modules": ["q"]}, LoraConfig),
            (LoraConfig, {"target_modules": ["q"]}, LoraConfig),
            (OFTConfig, {"oft_block_size": 4, "target_modules": ["q"]}, OFTConfig),
        ],
        ids=lambda v: _ids(v) if isinstance(v, type) else "",
    )
    def test_a_file_that_names_no_method_is_read_as_the_calling_class(
        self,
        caller: type[PeftConfig],
        payload: dict[str, object],
        expected: type[PeftConfig],
        tmp_path: Path,
    ) -> None:
        _write(tmp_path, payload)
        assert type(caller.from_pretrained(tmp_path)) is expected

    @pytest.mark.parametrize("caller", _CALLERS, ids=_ids)
    def test_a_method_this_library_lacks_is_a_key_error(
        self, caller: type[PeftConfig], tmp_path: Path
    ) -> None:
        _oft_file(tmp_path, peft_type="NOT_A_METHOD")
        with pytest.raises(KeyError, match="NOT_A_METHOD"):
            caller.from_pretrained(tmp_path)


class TestCheckKwargsIsCalled:
    """``from_pretrained`` runs the file's class's ``check_kwargs`` on the file's keys, then reads what it returns."""

    @pytest.fixture
    def seen(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
        """The kwargs each ``PeftConfig.check_kwargs`` call receives (every class inherits the base's)."""
        calls: list[dict[str, object]] = []

        def spy(cls: type[PeftConfig], **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return kwargs

        monkeypatch.setattr(PeftConfig, "check_kwargs", classmethod(spy))
        return calls

    @pytest.mark.parametrize("caller", _CALLERS, ids=_ids)
    def test_it_is_called_once_with_the_file_s_keys(
        self, caller: type[PeftConfig], seen: list[dict[str, object]], tmp_path: Path
    ) -> None:
        LoraConfig(r=2, target_modules=["q"]).save_pretrained(tmp_path)
        stored = json.loads((tmp_path / "adapter_config.json").read_text())
        caller.from_pretrained(tmp_path)
        assert seen == [stored]

    def test_it_sees_a_key_this_library_has_no_field_for(
        self, seen: list[dict[str, object]], tmp_path: Path
    ) -> None:
        _oft_file(tmp_path, future_flag=False)
        with pytest.warns(UserWarning, match="Unexpected keyword arguments"):
            PeftConfig.from_pretrained(tmp_path)
        assert [call["future_flag"] for call in seen] == [False]

    def test_it_is_called_on_the_class_the_file_names(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        named: list[type[PeftConfig]] = []

        def spy(cls: type[PeftConfig], **kwargs: object) -> dict[str, object]:
            named.append(cls)
            return kwargs

        monkeypatch.setattr(PeftConfig, "check_kwargs", classmethod(spy))
        _oft_file(tmp_path)
        for caller in _CALLERS:
            caller.from_pretrained(tmp_path)
        assert named == [OFTConfig] * len(_CALLERS)

    def test_the_config_is_built_from_what_it_returns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def widen(cls: type[PeftConfig], **kwargs: object) -> dict[str, object]:
            return {**kwargs, "r": 9}

        monkeypatch.setattr(LoraConfig, "check_kwargs", classmethod(widen))
        LoraConfig(r=2, target_modules=["q"]).save_pretrained(tmp_path)
        loaded = PeftConfig.from_pretrained(tmp_path)
        assert isinstance(loaded, LoraConfig)
        assert loaded.r == 9

    def test_a_refusal_from_it_is_not_swallowed(self, tmp_path: Path) -> None:
        _oft_file(tmp_path, oft_block_size=None)
        for caller in _CALLERS:
            with pytest.raises(ValueError, match="OFT has been updated since version 0.14.0"):
                caller.from_pretrained(tmp_path)

    def test_from_dict_does_not_call_it(self, seen: list[dict[str, object]]) -> None:
        LoraConfig.from_dict({"peft_type": "LORA", "target_modules": ["q"]})
        PeftConfig.from_peft_type(peft_type="LORA", target_modules=["q"])
        assert seen == []


class TestOFTFilesAreChecked:
    @pytest.mark.parametrize("caller", [PeftConfig, OFTConfig], ids=_ids)
    def test_a_file_this_library_wrote_loads_silently(
        self, caller: type[PeftConfig], tmp_path: Path
    ) -> None:
        config = OFTConfig(oft_block_size=4, target_modules=["q"])
        config.save_pretrained(tmp_path)
        stored = json.loads((tmp_path / "adapter_config.json").read_text())
        assert "peft_version" not in stored
        assert stored["use_cayley_neumann"] is True
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert caller.from_pretrained(tmp_path) == config

    @pytest.mark.parametrize("caller", _CALLERS, ids=_ids)
    @pytest.mark.parametrize("version", ["0.0.0", "0.17.1", "0.18.0.dev0", "0.17.1.dev0@abc123"])
    def test_a_file_with_an_old_version_warns_once(
        self, caller: type[PeftConfig], version: str, tmp_path: Path
    ) -> None:
        _oft_file(tmp_path, peft_version=version)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = caller.from_pretrained(tmp_path)
        assert type(loaded) is OFTConfig
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)
        assert _CAYLEY in str(caught[0].message)

    @pytest.mark.parametrize("version", ["0.18.0", "0.18.1.dev0@abc123", "0.21.2"])
    def test_a_file_with_a_new_version_is_quiet(self, version: str, tmp_path: Path) -> None:
        _oft_file(tmp_path, peft_version=version)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert PeftConfig.from_pretrained(tmp_path).peft_version == version

    def test_an_old_version_with_the_form_off_is_quiet(self, tmp_path: Path) -> None:
        _oft_file(tmp_path, peft_version="0.17.0", use_cayley_neumann=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PeftConfig.from_pretrained(tmp_path)

    def test_a_version_that_is_no_version_is_refused(self, tmp_path: Path) -> None:
        _oft_file(tmp_path, peft_version="not-a-version")
        with pytest.raises(ValueError, match="Invalid version"):
            PeftConfig.from_pretrained(tmp_path)

    def test_the_old_form_is_warned_about_before_an_unknown_key_is_dropped(
        self, tmp_path: Path
    ) -> None:
        """The reference's order: its check sees the whole file, then the unknown key is dropped with its own warning."""
        _oft_file(tmp_path, peft_version="0.17.0", other_flag=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = PeftConfig.from_pretrained(tmp_path)
        assert type(loaded) is OFTConfig
        messages = [str(w.message) for w in caught]
        assert len(messages) == 2
        assert _CAYLEY in messages[0]
        assert "Unexpected keyword arguments ['other_flag']" in messages[1]

    def test_a_key_that_turns_a_missing_feature_on_is_refused_not_dropped(
        self, tmp_path: Path
    ) -> None:
        """Where the reference drops any unknown key with a warning; dropping an "on" key changes the adapter."""
        _oft_file(tmp_path, future_flag=True)
        with pytest.raises(NotImplementedError, match="future_flag"):
            PeftConfig.from_pretrained(tmp_path)


def _net() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))


class TestModelLoadingChecksTheFile:
    """The loaders (``PeftModel.from_pretrained``) read the config through ``PeftConfig.from_pretrained``."""

    def _saved(self, directory: Path, **changes: object) -> Path:
        peft = get_peft_model(_net(), OFTConfig(oft_block_size=4, target_modules=["0"]))
        peft.save_pretrained(str(directory))
        stored = json.loads((directory / "adapter_config.json").read_text())
        stored.update(changes)
        return _write(directory, stored)

    def test_an_adapter_this_library_saved_loads_silently(self, tmp_path: Path) -> None:
        self._saved(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = PeftModel.from_pretrained(_net(), str(tmp_path))
        assert type(loaded.peft_config["default"]) is OFTConfig

    def test_an_adapter_of_the_old_form_warns_once(self, tmp_path: Path) -> None:
        self._saved(tmp_path, peft_version="0.17.0")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PeftModel.from_pretrained(_net(), str(tmp_path))
        assert len(_cayley_warnings(caught)) == 1

    def test_an_adapter_of_the_old_oft_is_refused(self, tmp_path: Path) -> None:
        self._saved(tmp_path)
        stored = json.loads((tmp_path / "adapter_config.json").read_text())
        del stored["oft_block_size"]
        _write(tmp_path, stored)
        with pytest.raises(ValueError, match="OFT has been updated since version 0.14.0"):
            PeftModel.from_pretrained(_net(), str(tmp_path))


class TestGetPeftConfig:
    """``get_peft_config`` is ``PeftConfig.from_peft_type`` of the dict, with no lookup or error of its own."""

    def test_it_calls_from_peft_type_with_the_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = LoraConfig(target_modules=["q"])
        calls: list[dict[str, object]] = []

        def spy(cls: type[PeftConfig], **kwargs: object) -> PeftConfig:
            calls.append(kwargs)
            return sentinel

        monkeypatch.setattr(PeftConfig, "from_peft_type", classmethod(spy))
        given: dict[str, object] = {"peft_type": "LORA", "r": 4}
        assert get_peft_config(given) is sentinel
        assert calls == [given]

    def test_an_unknown_method_is_the_one_registry_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Marker(Exception):
            pass

        def refuse(peft_type: object) -> type[PeftConfig]:
            raise Marker(peft_type)

        monkeypatch.setattr(config_module, "_config_class_named", refuse)
        with pytest.raises(Marker):
            get_peft_config({"peft_type": "LORA"})

    @pytest.mark.parametrize("named", ["NOT_A_METHOD", "lora", "", None, 3])
    def test_an_unknown_method_reads_as_it_does_everywhere(self, named: object) -> None:
        with pytest.raises(KeyError) as raised:
            get_peft_config({"peft_type": named})
        with pytest.raises(KeyError) as other:
            PeftConfig.from_peft_type(peft_type=named)
        assert str(raised.value) == str(other.value)
        assert "is not a method of lucid.peft; the methods are LORA, " in str(raised.value)

    def test_a_dict_of_another_method_is_that_methods_config(self) -> None:
        config = OFTConfig(oft_block_size=8, target_modules=["q"])
        assert get_peft_config(config.to_dict()) == config

    def test_a_dict_without_a_method_is_a_key_error(self) -> None:
        with pytest.raises(KeyError, match="peft_type"):
            get_peft_config({"r": 4})


# -- against the reference adapter library ------------------------------------


def _reference_oft_file(library: ModuleType, directory: Path, **changes: object) -> Path:
    """The file the reference's OFT config saves, with ``changes`` laid over it (``None`` removes a key)."""
    library.OFTConfig(oft_block_size=4, target_modules=["q"]).save_pretrained(str(directory))
    stored = json.loads((directory / "adapter_config.json").read_text())
    stored.update(changes)
    return _write(directory, {k: v for k, v in stored.items() if v is not None})


def _neumann(caught: list[warnings.WarningMessage]) -> list[str]:
    """The messages about the Cayley-Neumann form (the two libraries spell its name alike but for case)."""
    return [str(w.message) for w in caught if "neumann" in str(w.message).lower()]


@pytest.mark.parity
def test_a_file_of_another_method_is_that_methods_config_in_both(tmp_path: Path) -> None:
    library = adapter_library()
    _reference_oft_file(library, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = library.LoraConfig.from_pretrained(str(tmp_path))
        mine = LoraConfig.from_pretrained(tmp_path)
    assert type(mine).__name__ == type(theirs).__name__ == "OFTConfig"
    assert isinstance(mine, OFTConfig)
    assert mine.oft_block_size == theirs.oft_block_size == 4


@pytest.mark.parity
@pytest.mark.parametrize("version", ["0.17.0", "0.18.0", "0.21.2"])
def test_an_oft_file_is_warned_about_as_the_reference_does(version: str, tmp_path: Path) -> None:
    library = adapter_library()
    _reference_oft_file(library, tmp_path, peft_version=version)
    with warnings.catch_warnings(record=True) as theirs:
        warnings.simplefilter("always")
        library.OFTConfig.from_pretrained(str(tmp_path))
    with warnings.catch_warnings(record=True) as mine:
        warnings.simplefilter("always")
        OFTConfig.from_pretrained(tmp_path)
    assert len(_neumann(mine)) == len(_neumann(theirs)) == (1 if version == "0.17.0" else 0)


@pytest.mark.parity
def test_an_oft_file_with_no_block_size_is_refused_as_the_reference_refuses(tmp_path: Path) -> None:
    library = adapter_library()
    _reference_oft_file(library, tmp_path, oft_block_size=None)
    with pytest.raises(ValueError, match="OFT has been updated"):
        library.OFTConfig.from_pretrained(str(tmp_path))
    with pytest.raises(ValueError, match="OFT has been updated"):
        OFTConfig.from_pretrained(tmp_path)


@pytest.mark.parity
def test_an_unknown_method_in_a_file_is_a_key_error_in_both(tmp_path: Path) -> None:
    library = adapter_library()
    _reference_oft_file(library, tmp_path, peft_type="NOT_A_METHOD")
    with pytest.raises(KeyError):
        library.PeftConfig.from_pretrained(str(tmp_path))
    with pytest.raises(KeyError):
        PeftConfig.from_pretrained(tmp_path)
