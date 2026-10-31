"""A compiled prompt-tuned model follows every adapter switch (CPU and Metal).

A prompt is applied by forward pre-hooks that read the tuner's ``disable_adapters`` and
``active_adapter``; ``lucid.compile`` keys a call on the public scalar attributes of every module
(LCD-703 did the same for the LoRA layer).  A model with a forward hook installed is run eagerly by
``lucid.compile`` today, so the prompt never reaches an executable and the compiled call is the
eager one; these tests pin what must hold either way: the key moves with every switch (so a future
traced prompt cannot replay the state it left), and the compiled call equals the eager one after
each.  Once the hooks are gone (``unload``) the model compiles to an executable of its own.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile._core import signature
from lucid.models import create_model
from lucid.peft import PeftModel, PromptTuningConfig, get_peft_model
from lucid.test.unit.peft.test_prompt_learning import _METHODS, _TEXT, _ids
from lucid.test.unit.peft.test_prompt_token_level import _ROWS, _scores
from lucid.test._fixtures.devices import metal_available

_BATCH, _LENGTH = 2, 5
_PROMPTS = ["prompt", "p-tuning-mlp", "prefix"]


def _classifier(device: str) -> nn.Module:
    lucid.manual_seed(0)
    return create_model("bert_base_cls", num_labels=3, **_TEXT).eval().to(device)


def _close(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool(lucid.allclose(a, b, atol=1e-5))


def _adapted(method: str, device: str) -> PeftModel:
    return get_peft_model(_classifier(device), _METHODS[method]("SEQ_CLS"))


class TestTheSwitchesAreInTheKey:
    @pytest.mark.parametrize("method", _PROMPTS)
    def test_they_are_public_scalar_attributes(self, method: str) -> None:
        tuner = _adapted(method, "cpu").base_model
        assert vars(tuner)["disable_adapters"] is False
        assert vars(tuner)["active_adapter"] == "default"
        assert "_enabled" not in vars(tuner)

    @pytest.mark.parametrize("method", _PROMPTS)
    def test_every_switch_moves_the_key_and_undoing_it_restores_it(self, method: str) -> None:
        peft = _adapted(method, "cpu")
        peft.add_adapter("other", _METHODS[method]("SEQ_CLS"))
        tuner = peft.base_model
        first = signature._module_state(peft)
        tuner.disable_adapter_layers()
        disabled = signature._module_state(peft)
        tuner.enable_adapter_layers()
        assert signature._module_state(peft) == first
        peft.set_adapter("other")
        other = signature._module_state(peft)
        peft.set_adapter("default")
        assert signature._module_state(peft) == first
        assert len({first, disabled, other}) == 3

    def test_the_disable_context_round_trips_the_key(self) -> None:
        peft = _adapted("prompt", "cpu")
        before = signature._module_state(peft)
        with peft.disable_adapter():
            assert vars(peft.base_model)["disable_adapters"] is True
            assert signature._module_state(peft) != before
        assert vars(peft.base_model)["disable_adapters"] is False
        assert signature._module_state(peft) == before

    def test_deleting_the_active_adapter_moves_the_key(self) -> None:
        peft = _adapted("prompt", "cpu")
        peft.add_adapter("other", _METHODS["prompt"]("SEQ_CLS"))
        before = signature._module_state(peft)
        with pytest.warns(UserWarning, match="was active"):
            peft.delete_adapter("default")
        assert vars(peft.base_model)["active_adapter"] == "other"
        assert signature._module_state(peft) != before


def _step_disable(peft: PeftModel) -> None:
    peft.base_model.disable_adapter_layers()


def _step_enable(peft: PeftModel) -> None:
    peft.base_model.enable_adapter_layers()


def _step_other(peft: PeftModel) -> None:
    peft.set_adapter("other")


def _step_default(peft: PeftModel) -> None:
    peft.set_adapter("default")


_STEPS: list[tuple[str, Callable[[PeftModel], None]]] = [
    ("disable", _step_disable),
    ("enable", _step_enable),
    ("other adapter", _step_other),
    ("disable the other", _step_disable),
    ("enable the other", _step_enable),
    ("default again", _step_default),
]


class TestACompiledModelFollows:
    @pytest.mark.parametrize("method", _PROMPTS)
    def test_every_switch(self, method: str, device: str) -> None:
        peft = _adapted(method, device)
        peft.add_adapter("other", _METHODS[method]("SEQ_CLS"))
        ids = _ids(device, (_BATCH, _LENGTH))
        compiled = lucid.compile(peft)
        states: dict[str, lucid.Tensor] = {}
        with lucid.no_grad():
            for name, step in [("start", lambda p: None), *_STEPS]:
                step(peft)
                eager = peft(ids).logits
                assert _close(compiled(ids).logits, eager), name
                states[name] = eager
        # The switches changed what the model computes, so agreement above is not trivial.
        assert not _close(states["start"], states["disable"])
        assert not _close(states["start"], states["other adapter"])
        assert _close(states["start"], states["default again"])

    @pytest.mark.parametrize("name", ["bert-token", "roformer-qa"])
    def test_a_token_level_head(self, name: str, device: str) -> None:
        row = _ROWS[name]
        peft = get_peft_model(row.build(device), _METHODS["prompt"](row.task))
        ids = _ids(device, (_BATCH, _LENGTH))
        compiled = lucid.compile(peft)
        with lucid.no_grad():
            for step in (lambda: None, peft.base_model.disable_adapter_layers):
                step()
                for got, want in zip(
                    _scores(row, compiled(ids)), _scores(row, peft(ids)), strict=True
                ):
                    assert _close(got, want)

    def test_a_prompt_that_changes_between_calls(self, device: str) -> None:
        peft = _adapted("prompt", device)
        ids = _ids(device, (_BATCH, _LENGTH))
        compiled = lucid.compile(peft)
        with lucid.no_grad():
            before = compiled(ids).logits
            for p in peft.base_model.prompt_encoder.parameters():
                p.copy_(p + 3.0 * lucid.randn(*p.shape, device=device))
            assert _close(compiled(ids).logits, peft(ids).logits)
            assert not _close(compiled(ids).logits, before)

    def test_a_new_prompt_length_under_the_same_name(self, device: str) -> None:
        peft = _adapted("prompt", device)
        ids = _ids(device, (_BATCH, _LENGTH))
        compiled = lucid.compile(peft)
        longer = PromptTuningConfig(num_virtual_tokens=5, task_type="SEQ_CLS")
        with lucid.no_grad():
            compiled(ids)
            peft.add_adapter("longer", longer)
            peft.set_adapter("longer")
            peft.delete_adapter("default")
            peft.add_adapter("default", longer)
            peft.set_adapter("default")
            assert _close(compiled(ids).logits, peft(ids).logits)


@pytest.mark.skipif(not metal_available(), reason="compiled executables run on Metal")
class TestOnceTheHooksAreGone:
    def test_an_unloaded_model_compiles_to_an_executable_of_its_own(self) -> None:
        peft = _adapted("prompt", "metal")
        ids = _ids("metal", (_BATCH, _LENGTH))
        compiled = lucid.compile(peft)
        _C_engine.compile.session_cache_clear()
        with lucid.no_grad():
            compiled(ids)
            assert _C_engine.compile.session_cache_size() == 0  # hooks installed: run eagerly
            model = peft.unload()
            base = lucid.compile(model)
            assert _close(base(ids).logits, model(ids).logits)
            assert _C_engine.compile.session_cache_size() > 0
