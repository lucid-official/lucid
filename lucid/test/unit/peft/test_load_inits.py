"""Loading an adapter writes the file's tensors onto the base it is given; it never re-runs a mutating init.

PiSSA and OLoRA rewrite the base weight when they build.  ``from_pretrained``
and ``load_adapter`` build the adapter only to overwrite its factors, so they
build it with ``_Init.on_load()`` — the strategy that leaves the base alone —
while the config the model keeps stays the file's (``init_lora_weights`` and
all), so a later save or conversion still knows how the adapter began.
"""

from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import LoraConfig, PeftModel, get_peft_model
from lucid.peft.tuners.lora._inits import _Default, _Gaussian, _Init, _Olora, _Pissa, _Random

_FACTORISING = ["pissa", "pissa_niter_2", "olora"]


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(8, 10)
        self.down = nn.Linear(10, 8)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.down(self.up(x).relu())


def _base(device: str) -> nn.Module:
    lucid.manual_seed(0)
    return _Net().to(device)


def _config(init: bool | str) -> LoraConfig:
    return LoraConfig(r=2, lora_alpha=4, target_modules=["up", "down"], init_lora_weights=init)


def _train(model: nn.Module, x: lucid.Tensor, steps: int = 6) -> None:
    optimizer = lucid.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
    for _ in range(steps):
        loss = (model(x) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _gap(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def _rewritten_base(peft: PeftModel, device: str) -> nn.Module:
    """A fresh model holding the base weights ``peft`` trained against (PiSSA / OLoRA wrote the residual)."""
    model = _Net().to(device)
    with lucid.no_grad():
        for name in ("up", "down"):
            source = getattr(peft.get_base_model(), name).base_layer
            target = getattr(model, name)
            target.weight.copy_(source.weight)
            target.bias.copy_(source.bias)
    return model


def _saved_base(model: nn.Module) -> dict[str, lucid.Tensor]:
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}


class TestOnLoadHook:
    @pytest.mark.parametrize("init", _FACTORISING)
    def test_a_factorising_strategy_loads_as_one_that_leaves_the_base_alone(
        self, init: str
    ) -> None:
        strategy = _Init.of(init)
        assert strategy.factorises
        loading = strategy.on_load()
        assert isinstance(loading, _Random) and not loading.factorises

    @pytest.mark.parametrize("init", [True, False, "gaussian"])
    def test_any_other_strategy_loads_as_itself(self, init: bool | str) -> None:
        strategy = _Init.of(init)
        assert strategy.on_load() is strategy
        assert isinstance(strategy, (_Default, _Random, _Gaussian))

    def test_the_factorising_strategies_share_the_hook(self) -> None:
        assert issubclass(_Pissa, _Init) and issubclass(_Olora, _Init)
        assert _Pissa.on_load is _Olora.on_load


class TestConfigStaging:
    @pytest.mark.parametrize("init", _FACTORISING)
    def test_the_staged_copy_does_not_factorise_and_the_file_config_is_untouched(
        self, init: str
    ) -> None:
        config = _config(init)
        staged = config._staged()
        assert staged.init_lora_weights is False
        assert config.init_lora_weights == init

    def test_a_config_with_nothing_to_stage_is_copied_equal(self) -> None:
        config = _config("gaussian")
        assert config._staged() == config


class TestLoadingLeavesTheBaseAlone:
    @pytest.mark.parametrize("init", _FACTORISING)
    def test_a_trained_adapter_reloads_onto_the_base_it_was_saved_beside(
        self, device: str, tmp_path: Path, init: str
    ) -> None:
        peft = get_peft_model(_base(device), _config(init))
        x = lucid.randn(12, 8, device=device)
        _train(peft, x)
        peft.save_pretrained(tmp_path / "adapter")
        rewritten = _rewritten_base(peft, device)
        kept = _saved_base(rewritten)
        loaded = PeftModel.from_pretrained(rewritten, tmp_path / "adapter")
        assert _gap(loaded(x), peft(x)) < 1e-5
        for name in ("up", "down"):
            layer = getattr(loaded.get_base_model(), name)
            assert _gap(layer.base_layer.weight, kept[f"{name}.weight"]) == 0.0, name

    @pytest.mark.parametrize("init", _FACTORISING)
    def test_loading_onto_a_pristine_base_does_not_mutate_it(
        self, device: str, tmp_path: Path, init: str
    ) -> None:
        peft = get_peft_model(_base(device), _config(init))
        peft.save_pretrained(tmp_path / "adapter")
        pristine = _base(device)
        before = _saved_base(pristine)
        loaded = PeftModel.from_pretrained(pristine, tmp_path / "adapter")
        for name in ("up", "down"):
            layer = getattr(loaded.get_base_model(), name)
            assert _gap(layer.base_layer.weight, before[f"{name}.weight"]) == 0.0, name

    @pytest.mark.parametrize("init", _FACTORISING)
    def test_the_model_keeps_the_files_init_after_loading(
        self, device: str, tmp_path: Path, init: str
    ) -> None:
        get_peft_model(_base(device), _config(init)).save_pretrained(tmp_path / "adapter")
        loaded = PeftModel.from_pretrained(_base(device), tmp_path / "adapter")
        stored = loaded.peft_config["default"]
        assert isinstance(stored, LoraConfig) and stored.init_lora_weights == init
        loaded.save_pretrained(tmp_path / "again")
        again = PeftModel.from_pretrained(_base(device), tmp_path / "again")
        assert again.peft_config["default"].init_lora_weights == init  # type: ignore[attr-defined]

    @pytest.mark.parametrize("init", _FACTORISING)
    def test_load_adapter_does_not_mutate_the_base_either(
        self, device: str, tmp_path: Path, init: str
    ) -> None:
        get_peft_model(_base(device), _config(init)).save_pretrained(tmp_path / "adapter")
        host = get_peft_model(_base(device), _config(True))
        before = {
            name: getattr(host.get_base_model(), name).base_layer.weight.clone()
            for name in ("up", "down")
        }
        host.load_adapter(tmp_path / "adapter", "loaded")
        for name, weight in before.items():
            layer = getattr(host.get_base_model(), name)
            assert _gap(layer.base_layer.weight, weight) == 0.0, name
        stored = host.peft_config["loaded"]
        assert isinstance(stored, LoraConfig) and stored.init_lora_weights == init

    def test_a_failed_load_still_restores_the_base(self, device: str, tmp_path: Path) -> None:
        get_peft_model(_base(device), _config("pissa")).save_pretrained(tmp_path / "adapter")
        wrong = nn.Sequential(nn.Linear(8, 10), nn.Linear(10, 8))
        with pytest.raises(ValueError):
            PeftModel.from_pretrained(wrong, tmp_path / "adapter")
        assert not hasattr(wrong, "peft_config")
