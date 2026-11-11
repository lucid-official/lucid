"""The methods that have a config and no adapter: registered, refused loudly, and counted.

Twenty ``PeftType`` members were added after the library's first twenty-six, one package each
(``lucid.peft.tuners.<method>``) with the reference's config and a stand-in tuner.  A stand-in is
``_PendingTuner`` (the tuner is still to be written, and an issue tracks it) or ``_RefusedTuner``
(it will not be: SHADOW).  Neither is in ``_TUNERS``, which every guard that loops over the working
tuners reads, so none of them meets a stand-in.  What is held here:

* every method of the reference has a config class and a tuner class, working or not;
* building an adapter with a stand-in raises ``NotImplementedError`` naming the issue (or the reason)
  at every entry point, before a model is touched or a file is read;
* a stand-in cannot be declared without its issue or reason;
* the B4 end gate: no method is pending (flips when the last tuner lands).
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
from lucid.peft.tuners._base import (
    _PENDING_TUNERS,
    _REFUSED_TUNERS,
    _TUNERS,
    _PendingTuner,
    _RefusedTuner,
    _tuner_for,
    _WithheldTuner,
)
from lucid.test.audit._builders import _EXAMPLE_CONFIGS

_SNAPSHOT = json.loads((Path(__file__).parent / "_reference_signatures.json").read_text())

#: Methods of this library the reference no longer has.
_LUCID_ONLY = {"BONE"}

_WITHHELD: list[tuple[PeftType, type[_WithheldTuner]]] = [
    *_PENDING_TUNERS.items(),
    *_REFUSED_TUNERS.items(),
]
_IDS = [peft_type.value for peft_type, _ in _WITHHELD]


def _net() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


# ── the registries ───────────────────────────────────────────────────────────


def test_every_reference_method_is_a_peft_type() -> None:
    """The enum is the reference's members, plus the one method this library keeps for old files."""
    reference = {name for name, _ in _SNAPSHOT["enum_members"]["PeftType"]}
    assert {member.value for member in PeftType} == reference | _LUCID_ONLY


def test_each_method_has_exactly_one_tuner_entry() -> None:
    """Working, pending and refused are a partition of the methods."""
    tables = [set(_TUNERS), set(_PENDING_TUNERS), set(_REFUSED_TUNERS)]
    assert sum(len(t) for t in tables) == len(set().union(*tables))
    assert set().union(*tables) == set(PeftType)


def test_the_mappings_have_a_class_for_every_method() -> None:
    assert set(PEFT_TYPE_TO_CONFIG_MAPPING) == set(PeftType)
    assert set(PEFT_TYPE_TO_TUNER_MAPPING) == set(PeftType)
    for peft_type, tuner in PEFT_TYPE_TO_TUNER_MAPPING.items():
        assert tuner._config_class is PEFT_TYPE_TO_CONFIG_MAPPING[peft_type], peft_type


def test_only_shadow_is_refused() -> None:
    """SHADOW needs the base model's decoder loop and a second backbone: refused for good."""
    assert set(_REFUSED_TUNERS) == {PeftType.SHADOW}
    assert PeftType.SHADOW not in _PENDING_TUNERS


@pytest.mark.parametrize(("peft_type", "tuner"), _WITHHELD, ids=_IDS)
def test_a_stand_in_pairs_its_method_with_the_config_class(
    peft_type: PeftType, tuner: type[_WithheldTuner]
) -> None:
    assert tuner._peft_type == peft_type
    assert tuner._config_class is _CONFIG_CLASSES[peft_type]
    assert peft_type not in _TUNERS


@pytest.mark.parametrize(
    ("peft_type", "tuner"), list(_PENDING_TUNERS.items()), ids=[t.value for t in _PENDING_TUNERS]
)
def test_a_pending_tuner_names_the_issue_that_tracks_it(
    peft_type: PeftType, tuner: type[_PendingTuner]
) -> None:
    assert re.fullmatch(r"LCD-\d+", tuner._issue), tuner._issue
    assert tuner._issue in tuner._refusal()


# ── declaring one ────────────────────────────────────────────────────────────


@pytest.fixture
def restore_registries() -> Iterator[None]:
    pending, refused = dict(_PENDING_TUNERS), dict(_REFUSED_TUNERS)
    yield
    _PENDING_TUNERS.clear()
    _PENDING_TUNERS.update(pending)
    _REFUSED_TUNERS.clear()
    _REFUSED_TUNERS.update(refused)


@pytest.mark.usefixtures("restore_registries")
class TestDeclaring:
    def test_a_pending_tuner_without_an_issue_is_a_type_error(self) -> None:
        before = dict(_PENDING_TUNERS)
        with pytest.raises(TypeError, match="issue"):
            types.new_class("_NoIssue", (_PendingTuner,), {"peft_type": PeftType.SHADOW})
        assert _PENDING_TUNERS == before

    def test_a_refused_tuner_without_a_reason_is_a_type_error(self) -> None:
        before = dict(_REFUSED_TUNERS)
        with pytest.raises(TypeError, match="reason"):
            types.new_class("_NoReason", (_RefusedTuner,), {"peft_type": PeftType.SHADOW})
        assert _REFUSED_TUNERS == before

    def test_a_declaration_registers_under_its_own_table(self) -> None:
        # Whichever method is still pending: a method card turns its own into a
        # working tuner, so no one name stays pending through phase B4.
        peft_type = next(iter(_PENDING_TUNERS), None)
        if peft_type is None:
            pytest.skip("no method is pending any more")

        class _Again(_PendingTuner, peft_type=peft_type, issue="LCD-1"):
            pass

        assert _PENDING_TUNERS[peft_type] is _Again
        assert peft_type not in _TUNERS and peft_type not in _REFUSED_TUNERS
        assert "LCD-1" in _Again._refusal()

    def test_a_refusal_carries_its_reason(self) -> None:
        class _Why(_RefusedTuner, peft_type=PeftType.SHADOW, reason="it needs a thing"):
            pass

        assert _REFUSED_TUNERS[PeftType.SHADOW] is _Why
        assert "it needs a thing" in _Why._refusal()


# ── building one ─────────────────────────────────────────────────────────────


def _config(peft_type: PeftType) -> PeftConfig:
    return _CONFIG_CLASSES[peft_type]()


@pytest.mark.parametrize(("peft_type", "tuner"), _WITHHELD, ids=_IDS)
class TestBuilding:
    def test_get_peft_model_raises_and_leaves_the_model_alone(
        self, peft_type: PeftType, tuner: type[_WithheldTuner]
    ) -> None:
        net = _net()
        before = {name: (type(m), list(m._modules)) for name, m in net.named_modules()}
        with pytest.raises(NotImplementedError, match=re.escape(tuner._refusal())):
            get_peft_model(net, _config(peft_type))
        assert {name: (type(m), list(m._modules)) for name, m in net.named_modules()} == before

    def test_the_class_itself_raises(self, peft_type: PeftType, tuner: type[_WithheldTuner]) -> None:
        with pytest.raises(NotImplementedError, match=re.escape(tuner._refusal())):
            tuner(_net(), _config(peft_type), "default")

    def test_inject_adapter_in_model_raises(
        self, peft_type: PeftType, tuner: type[_WithheldTuner]
    ) -> None:
        with pytest.raises(NotImplementedError, match=re.escape(tuner._refusal())):
            inject_adapter_in_model(_config(peft_type), _net())

    def test_the_tuner_lookup_raises(self, peft_type: PeftType, tuner: type[_WithheldTuner]) -> None:
        with pytest.raises(NotImplementedError, match=re.escape(tuner._refusal())):
            _tuner_for(_config(peft_type))

    def test_a_saved_config_loads_but_its_adapter_does_not(
        self, peft_type: PeftType, tuner: type[_WithheldTuner], tmp_path: Path
    ) -> None:
        """The file reads as a config; building the adapter is refused before any weights are read."""
        config = _config(peft_type)
        config.save_pretrained(tmp_path)
        assert PeftConfig.from_pretrained(tmp_path) == config
        assert get_peft_config(config.to_dict()) == config
        with pytest.raises(NotImplementedError, match=re.escape(tuner._refusal())):
            PeftModel.from_pretrained(_net(), tmp_path)

    def test_the_refusal_names_the_method_and_its_config(
        self, peft_type: PeftType, tuner: type[_WithheldTuner]
    ) -> None:
        message = tuner._refusal()
        assert str(peft_type) in message and tuner._config_class.__name__ in message


# ── the audit's example configs ──────────────────────────────────────────────

#: A prompt host (a key-value cache with no module targets) and a refused method are not built from
#: the audit's two-linear model.
_NO_EXAMPLE = {PeftType.CARTRIDGE, PeftType.SHADOW}


@pytest.mark.parametrize(
    "peft_type", [t for t, _ in _WITHHELD if t not in _NO_EXAMPLE], ids=lambda t: t.value
)
def test_a_module_targeting_stand_in_has_its_audit_example_ready(peft_type: PeftType) -> None:
    """The audit builds a tuner from ``_EXAMPLE_CONFIGS`` once it is in ``_TUNERS``; no card edits that table."""
    config = _EXAMPLE_CONFIGS[peft_type]()
    assert type(config) is _CONFIG_CLASSES[peft_type]
    assert getattr(config, "target_modules") == {"0", "2"}


# ── the B4 end gate ──────────────────────────────────────────────────────────


@pytest.mark.xfail(
    strict=True,
    reason="LCD-828: the last pending tuner (AdaMSS) lands; delete this marker when it does",
)
def test_phase_b4_leaves_no_pending_tuner() -> None:
    """The end gate of phase B4: every method but the refused one has a tuner.

    Fails while any method is still a stand-in.  When the last one lands it passes, which a strict
    ``xfail`` reports as a failure — the reminder to remove the marker.
    """
    assert not _PENDING_TUNERS, sorted(str(t) for t in _PENDING_TUNERS)
