"""The method that has a config and will never have an adapter: registered, refused loudly, and the B4 end gate.

Twenty ``PeftType`` members were added after the library's first twenty-six, one package each
(``lucid.peft.tuners.<method>``) with the reference's config.  Every one but ShadowPEFT has its adapter
now; that one needs the base model's decoder loop and a second backbone beside it, so its tuner is a
``_RefusedTuner``: it is in ``_REFUSED_TUNERS``, not in ``_TUNERS`` (which every guard that loops over
the working tuners reads), and building an adapter with it raises ``NotImplementedError`` giving the
reason.  What is held here:

* every method of the reference has a config class and a tuner class, working or refused;
* building an adapter with the refused method raises at every entry point, before a model is touched
  or a file is read;
* a refusal cannot be declared without its reason;
* the B4 end gate: every method but the refused one has a working tuner.
"""

import json
import re
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

import lucid.nn as nn
from lucid.peft import (
    PEFT_TYPE_TO_CONFIG_MAPPING,
    PEFT_TYPE_TO_TUNER_MAPPING,
    PeftConfig,
    PeftModel,
    PeftType,
    get_peft_config,
    get_peft_model,
    inject_adapter_in_model,
)
from lucid.peft._config import _CONFIG_CLASSES
from lucid.peft.tuners._base import _REFUSED_TUNERS, _TUNERS, _RefusedTuner, _tuner_for

_SNAPSHOT = json.loads((Path(__file__).parent / "_reference_signatures.json").read_text())

#: Methods of this library the reference no longer has.
_LUCID_ONLY = {"BONE"}

_REFUSED = list(_REFUSED_TUNERS.items())
_IDS = [peft_type.value for peft_type, _ in _REFUSED]


def _net() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


# ── the registries ───────────────────────────────────────────────────────────


def test_every_reference_method_is_a_peft_type() -> None:
    """The enum is the reference's members, plus the one method this library keeps for old files."""
    reference = {name for name, _ in _SNAPSHOT["enum_members"]["PeftType"]}
    assert {member.value for member in PeftType} == reference | _LUCID_ONLY


def test_each_method_has_exactly_one_tuner_entry() -> None:
    """Working and refused are a partition of the methods."""
    assert not set(_TUNERS) & set(_REFUSED_TUNERS)
    assert set(_TUNERS) | set(_REFUSED_TUNERS) == set(PeftType)


def test_the_mappings_have_a_class_for_every_method() -> None:
    assert set(PEFT_TYPE_TO_CONFIG_MAPPING) == set(PeftType)
    assert set(PEFT_TYPE_TO_TUNER_MAPPING) == set(PeftType)
    for peft_type, tuner in PEFT_TYPE_TO_TUNER_MAPPING.items():
        assert tuner._config_class is PEFT_TYPE_TO_CONFIG_MAPPING[peft_type], peft_type


def test_only_shadow_is_refused() -> None:
    """SHADOW needs the base model's decoder loop and a second backbone: refused for good."""
    assert set(_REFUSED_TUNERS) == {PeftType.SHADOW}


@pytest.mark.parametrize(("peft_type", "tuner"), _REFUSED, ids=_IDS)
def test_a_refused_tuner_pairs_its_method_with_the_config_class(
    peft_type: PeftType, tuner: type[_RefusedTuner]
) -> None:
    assert tuner._peft_type == peft_type
    assert tuner._config_class is _CONFIG_CLASSES[peft_type]
    assert peft_type not in _TUNERS


# ── declaring one ────────────────────────────────────────────────────────────


@pytest.fixture
def restore_registry() -> Iterator[None]:
    refused = dict(_REFUSED_TUNERS)
    yield
    _REFUSED_TUNERS.clear()
    _REFUSED_TUNERS.update(refused)


@pytest.mark.usefixtures("restore_registry")
class TestDeclaring:
    def test_a_refused_tuner_without_a_reason_is_a_type_error(self) -> None:
        before = dict(_REFUSED_TUNERS)
        with pytest.raises(TypeError, match="reason"):
            types.new_class("_NoReason", (_RefusedTuner,), {"peft_type": PeftType.SHADOW})
        assert _REFUSED_TUNERS == before

    def test_a_refusal_carries_its_reason(self) -> None:
        class _Why(_RefusedTuner, peft_type=PeftType.SHADOW, reason="it needs a thing"):
            pass

        assert _REFUSED_TUNERS[PeftType.SHADOW] is _Why
        assert "it needs a thing" in _Why._refusal()


# ── building one ─────────────────────────────────────────────────────────────


def _config(peft_type: PeftType) -> PeftConfig:
    return _CONFIG_CLASSES[peft_type]()


@pytest.mark.parametrize(("peft_type", "tuner"), _REFUSED, ids=_IDS)
class TestBuilding:
    def test_get_peft_model_raises_and_leaves_the_model_alone(
        self, peft_type: PeftType, tuner: type[_RefusedTuner]
    ) -> None:
        net = _net()
        before = {name: (type(m), list(m._modules)) for name, m in net.named_modules()}
        with pytest.raises(NotImplementedError, match=re.escape(tuner._refusal())):
            get_peft_model(net, _config(peft_type))
        assert {name: (type(m), list(m._modules)) for name, m in net.named_modules()} == before

    def test_the_class_itself_raises(self, peft_type: PeftType, tuner: type[_RefusedTuner]) -> None:
        with pytest.raises(NotImplementedError, match=re.escape(tuner._refusal())):
            tuner(_net(), _config(peft_type), "default")

    def test_inject_adapter_in_model_raises(
        self, peft_type: PeftType, tuner: type[_RefusedTuner]
    ) -> None:
        with pytest.raises(NotImplementedError, match=re.escape(tuner._refusal())):
            inject_adapter_in_model(_config(peft_type), _net())

    def test_the_tuner_lookup_raises(self, peft_type: PeftType, tuner: type[_RefusedTuner]) -> None:
        with pytest.raises(NotImplementedError, match=re.escape(tuner._refusal())):
            _tuner_for(_config(peft_type))

    def test_a_saved_config_loads_but_its_adapter_does_not(
        self, peft_type: PeftType, tuner: type[_RefusedTuner], tmp_path: Path
    ) -> None:
        """The file reads as a config; building the adapter is refused before any weights are read."""
        config = _config(peft_type)
        config.save_pretrained(tmp_path)
        assert PeftConfig.from_pretrained(tmp_path) == config
        assert get_peft_config(config.to_dict()) == config
        with pytest.raises(NotImplementedError, match=re.escape(tuner._refusal())):
            PeftModel.from_pretrained(_net(), tmp_path)

    def test_the_refusal_names_the_method_and_its_config(
        self, peft_type: PeftType, tuner: type[_RefusedTuner]
    ) -> None:
        message = tuner._refusal()
        assert str(peft_type) in message and tuner._config_class.__name__ in message


# ── the B4 end gate ──────────────────────────────────────────────────────────


def test_phase_b4_leaves_no_pending_tuner() -> None:
    """The end gate of phase B4: every method but the refused one has a working tuner."""
    assert set(PeftType) - set(_TUNERS) == {PeftType.SHADOW}
