"""A tuner that refuses its model leaves the model as it found it (every registered host).

Constructing a tuner is one transaction: whatever raises, wherever it raises — a layout that
finds no head to hook after ``modules_to_save`` wrapped the others, a layer that cannot be built
for the second target after the first was initialised in place, the last step of the
constructor — the model keeps its modules, its parameters (the same objects, the same values,
the same ``requires_grad``), its attributes and its hooks.  The hosts come from the registry,
so a method added later is held to it without a row here.
"""

import dataclasses

import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import LoraConfig, PeftConfig, PeftType, TaskType
from lucid.peft.tuners._base import _TUNERS, _AdapterHost
from lucid.peft.tuners._prompt._encoder import _Injection
from lucid.peft.tuners._prompt._tuner import _PromptLearningTuner
from lucid.peft.tuners.lora._inits import _Init
from lucid.test.audit._builders import BUILDERS

_HOSTS = sorted(_TUNERS.values(), key=lambda host: host.__name__)
_PROMPTS = [host for host in _HOSTS if issubclass(host, _PromptLearningTuner)]
_IN_FRONT = [host for host in _PROMPTS if host._encoder_class.injection == _Injection.EMBEDDINGS]
_FACTORISING = sorted(s.spelling for s in _Init._strategies.values() if s.factorises)


class _Refused(Exception):
    """What the forced refusal raises."""


def _refuse(self: object) -> None:
    raise _Refused


def _fresh(host: type[_AdapterHost]) -> tuple[nn.Module, PeftConfig]:
    """The model ``host``'s audit builder adapts, taken back out, and the config it stored."""
    built = BUILDERS[host]()
    config = next(iter(built.peft_config.values()))
    return built.unload(), config


def _state(model: nn.Module) -> dict[str, object]:
    """Everything a refused construction must not change: a comparable snapshot of ``model``."""
    return {
        "state_dict": list(model.state_dict()),
        "modules": {
            name: (id(module), type(module), len(module._forward_pre_hooks))
            for name, module in model.named_modules(remove_duplicate=False)
        },
        "parameters": {
            name: (id(param), param.requires_grad, param.clone())
            for name, param in model.named_parameters(remove_duplicate=False)
        },
        "attributes": sorted(model.__dict__),
    }


def _assert_unchanged(before: dict[str, object], after: dict[str, object]) -> None:
    assert after["state_dict"] == before["state_dict"]
    assert after["modules"] == before["modules"]
    assert after["attributes"] == before["attributes"]
    kept, now = before["parameters"], after["parameters"]
    assert isinstance(kept, dict) and isinstance(now, dict)
    assert list(now) == list(kept)
    for name, (ident, flag, value) in kept.items():
        assert now[name][:2] == (ident, flag), name
        assert bool((now[name][2] == value).all().item()), f"{name} was rewritten"


def _with_a_module_trained_in_full(config: PeftConfig, name: str) -> PeftConfig:
    """``config`` that also trains the module ``name`` in full, which every host plans before it mutates."""
    return dataclasses.replace(config, modules_to_save=[name])


def _lm() -> nn.Module:
    return create_model(
        "gpt2_small_lm",
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
        tie_word_embeddings=False,
    )


def test_the_registry_has_every_family_this_guard_is_for() -> None:
    assert len(_HOSTS) >= 10
    assert {PeftType.LORA, PeftType.PROMPT_TUNING} <= set(_TUNERS)
    assert _IN_FRONT and _FACTORISING == ["olora", "pissa"]


@pytest.mark.parametrize("host", _HOSTS, ids=lambda host: host.__name__)
def test_a_failure_in_the_last_step_leaves_the_model_as_it_was(
    host: type[_AdapterHost], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, config = _fresh(host)
    module_to_save = "lm_head" if host in _PROMPTS else "2"
    config = _with_a_module_trained_in_full(config, module_to_save)
    before = _state(model)
    monkeypatch.setattr(host, "_mark_only_adapters_as_trainable", _refuse)
    with pytest.raises(_Refused):
        host(model, config)
    _assert_unchanged(before, _state(model))
    monkeypatch.undo()
    assert isinstance(host(model, config), host), "the model is fit to be adapted again"


@pytest.mark.parametrize("host", _PROMPTS, ids=lambda host: host.__name__)
def test_a_prompt_on_a_model_without_an_encoder_and_decoder_is_refused_whole(
    host: type[_AdapterHost],
) -> None:
    _, config = _fresh(host)
    model = _lm()
    config = dataclasses.replace(
        config, task_type=TaskType.SEQ_2_SEQ_LM, modules_to_save=["lm_head"]
    )
    before = _state(model)
    with pytest.raises(NotImplementedError, match="get_encoder"):
        host(model, config)
    _assert_unchanged(before, _state(model))


@pytest.mark.parametrize("host", _IN_FRONT, ids=lambda host: host.__name__)
def test_a_token_level_prompt_without_a_head_to_cut_is_refused_whole(
    host: type[_AdapterHost],
) -> None:
    _, config = _fresh(host)
    model = _lm()
    config = dataclasses.replace(config, task_type=TaskType.TOKEN_CLS, modules_to_save=["lm_head"])
    before = _state(model)
    with pytest.raises(NotImplementedError, match="classifier / score"):
        host(model, config)
    _assert_unchanged(before, _state(model))


@pytest.mark.parametrize("init", _FACTORISING)
def test_an_initialisation_that_rewrites_the_base_is_undone_when_a_later_layer_is_refused(
    init: str,
) -> None:
    lucid.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 2))
    before = _state(net)
    config = LoraConfig(r=4, target_modules=["0", "2"], init_lora_weights=init)
    with pytest.raises(ValueError, match="r <= min"):
        _TUNERS[PeftType.LORA](net, config)
    _assert_unchanged(before, _state(net))


@pytest.mark.parametrize("init", _FACTORISING)
def test_an_initialisation_that_rewrites_the_base_is_undone_when_the_constructor_fails_late(
    init: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    lucid.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 2))
    before = _state(net)
    host = _TUNERS[PeftType.LORA]
    monkeypatch.setattr(host, "_mark_only_adapters_as_trainable", _refuse)
    with pytest.raises(_Refused):
        host(net, LoraConfig(r=2, target_modules=["0", "2"], init_lora_weights=init))
    _assert_unchanged(before, _state(net))
