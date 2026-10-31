"""``save_pretrained(path_initial_model_for_weight_conversion=...)``: PiSSA / OLoRA adapters as plain LoRA.

PiSSA and OLoRA write the residual into the base weight, so the adapter they
train is relative to a base that is not the original.  Saved as it is, it
loads wrong onto the pristine base; the conversion subtracts the initial
product (:math:`\\Delta W = s(BA - B_0A_0)`) and stores it as a plain LoRA of
twice the rank, which does.
"""

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    IA3Config,
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft.tuners.lora._inits import _Init, _Pissa
from lucid.serialization import save_safetensors

_FACTORISING = ["pissa", "pissa_niter_2", "olora"]


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(8, 10)
        self.down = nn.Linear(10, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.down(self.up(x).relu()))


def _pristine(device: str) -> nn.Module:
    lucid.manual_seed(0)
    return _Net().to(device)


def _config(init: bool | str = "pissa", **fields: object) -> LoraConfig:
    options: dict[str, object] = {
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["up", "down"],
        "init_lora_weights": init,
    } | fields
    return LoraConfig(**options)  # type: ignore[arg-type]


def _train(model: nn.Module, x: lucid.Tensor, steps: int = 6) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = lucid.optim.SGD(trainable, lr=0.05)
    for _ in range(steps):
        loss = (model(x) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _gap(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def _converted(
    device: str, tmp_path: Path, config: LoraConfig
) -> tuple[PeftModel, lucid.Tensor, Path, nn.Module]:
    """``(trained in-place model, its inputs, the converted directory, a pristine base)``."""
    pristine = _pristine(device)
    peft = get_peft_model(_pristine(device), config)
    x = lucid.randn(12, 8, device=device)
    peft.save_pretrained(tmp_path / "init")
    _train(peft, x)
    peft.save_pretrained(
        tmp_path / "out", path_initial_model_for_weight_conversion=tmp_path / "init"
    )
    return peft, x, tmp_path / "out", pristine


class TestConversionRoundTrip:
    @pytest.mark.parametrize("init", _FACTORISING)
    def test_loads_onto_the_pristine_base_as_the_trained_model(
        self, device: str, tmp_path: Path, init: str
    ) -> None:
        peft, x, out, pristine = _converted(device, tmp_path, _config(init))
        loaded = PeftModel.from_pretrained(pristine, out)
        assert _gap(loaded(x), peft(x)) < 1e-4

    @pytest.mark.parametrize("init", _FACTORISING)
    def test_the_base_the_adapter_loads_onto_is_not_rewritten(
        self, device: str, tmp_path: Path, init: str
    ) -> None:
        _, _, out, pristine = _converted(device, tmp_path, _config(init))
        restored = PeftModel.from_pretrained(pristine, out).unload().state_dict()
        for name, tensor in _pristine(device).state_dict().items():
            assert _gap(restored[name], tensor) == 0.0, name

    @pytest.mark.parametrize("init", _FACTORISING)
    def test_without_the_conversion_a_plain_loader_gets_the_wrong_model(
        self, device: str, tmp_path: Path, init: str
    ) -> None:
        """The defect: the adapter saved as it is, read as a plain LoRA, is not the trained model.

        (``from_pretrained`` with the saved config would redo the initialisation on the
        pristine base, which repairs the exact variants by accident; a loader that
        applies A and B to the base as they are does not.)
        """
        peft = get_peft_model(_pristine(device), _config(init))
        x = lucid.randn(12, 8, device=device)
        _train(peft, x)
        peft.save_pretrained(tmp_path)
        plain = dataclasses.replace(PeftConfig.from_pretrained(tmp_path), init_lora_weights=True)
        wrong = PeftModel.from_pretrained(_pristine(device), tmp_path, config=plain)
        assert _gap(wrong(x), peft(x)) > 1e-2

    @pytest.mark.parametrize("init", _FACTORISING)
    def test_the_saved_config_is_a_plain_lora_of_twice_the_rank(
        self, device: str, tmp_path: Path, init: str
    ) -> None:
        peft, _, out, _ = _converted(device, tmp_path, _config(init))
        saved = json.loads((out / "adapter_config.json").read_text())
        assert (saved["r"], saved["lora_alpha"], saved["init_lora_weights"]) == (4, 8, True)
        state = get_peft_model_state_dict(peft)
        stored = PeftConfig.from_pretrained(out)
        assert isinstance(stored, LoraConfig) and stored.r == 4
        loaded = PeftModel.from_pretrained(_pristine(device), out)
        for name, tensor in get_peft_model_state_dict(loaded).items():
            axis = 0 if "lora_A" in name else 1
            assert tensor.shape[axis] == 2 * state[name].shape[axis], name

    def test_the_trained_model_and_the_initial_directory_are_left_alone(
        self, device: str, tmp_path: Path
    ) -> None:
        peft = get_peft_model(_pristine(device), _config())
        peft.save_pretrained(tmp_path / "init")
        _train(peft, lucid.randn(12, 8, device=device))

        def digest(directory: Path) -> dict[str, str]:
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}

        before, config_before = digest(tmp_path / "init"), copy.deepcopy(peft.peft_config)
        state_before = {k: v.clone() for k, v in get_peft_model_state_dict(peft).items()}
        peft.save_pretrained(
            tmp_path / "out", path_initial_model_for_weight_conversion=tmp_path / "init"
        )
        assert digest(tmp_path / "init") == before
        assert peft.peft_config == config_before
        for name, tensor in get_peft_model_state_dict(peft).items():
            assert _gap(tensor, state_before[name]) == 0.0

    def test_a_trained_head_and_trained_biases_ride_along(
        self, device: str, tmp_path: Path
    ) -> None:
        config = _config(modules_to_save=["head"], bias="all")
        peft, x, out, pristine = _converted(device, tmp_path, config)
        loaded = PeftModel.from_pretrained(pristine, out)
        assert _gap(loaded(x), peft(x)) < 1e-4

    def test_rslora_scales_alpha_by_the_square_root_of_two(
        self, device: str, tmp_path: Path
    ) -> None:
        peft, x, out, pristine = _converted(device, tmp_path, _config(use_rslora=True))
        stored = PeftConfig.from_pretrained(out)
        assert isinstance(stored, LoraConfig)
        assert stored.r == 4 and abs(stored.lora_alpha - 4 * 2**0.5) < 1e-12
        assert _gap(PeftModel.from_pretrained(pristine, out)(x), peft(x)) < 1e-4

    def test_rank_and_alpha_patterns_double_with_the_rank(
        self, device: str, tmp_path: Path
    ) -> None:
        config = _config(rank_pattern={"down": 3}, alpha_pattern={"up": 6})
        peft, x, out, pristine = _converted(device, tmp_path, config)
        stored = PeftConfig.from_pretrained(out)
        assert isinstance(stored, LoraConfig)
        assert (stored.rank_pattern, stored.alpha_pattern) == ({"down": 6}, {"up": 12})
        assert _gap(PeftModel.from_pretrained(pristine, out)(x), peft(x)) < 1e-4

    def test_the_converted_adapter_sits_beside_another_one(
        self, device: str, tmp_path: Path
    ) -> None:
        """Nothing about the base changed, so a second adapter can be added to it."""
        peft, x, out, pristine = _converted(device, tmp_path, _config())
        loaded = PeftModel.from_pretrained(pristine, out)
        loaded.add_adapter("extra", _config(False, r=3))
        loaded.set_adapter("default")
        assert _gap(loaded(x), peft(x)) < 1e-4


class TestMultipleAdapters:
    def _two(self, device: str, tmp_path: Path) -> tuple[PeftModel, lucid.Tensor, nn.Module]:
        pristine = _pristine(device)
        peft = get_peft_model(_pristine(device), _config(False), adapter_name="plain")
        peft.add_adapter("pissa", _config("pissa", r=3, lora_alpha=6))
        peft.save_pretrained(tmp_path / "init", selected_adapters=["pissa"])
        peft.set_adapter("pissa")
        x = lucid.randn(12, 8, device=device)
        _train(peft, x)
        return peft, x, pristine

    def test_it_converts_the_one_selected_and_leaves_the_others(
        self, device: str, tmp_path: Path
    ) -> None:
        peft, x, pristine = self._two(device, tmp_path)
        peft.save_pretrained(
            tmp_path / "out",
            selected_adapters=["pissa"],
            path_initial_model_for_weight_conversion=tmp_path / "init" / "pissa",
        )
        loaded = PeftModel.from_pretrained(pristine, tmp_path / "out" / "pissa", "pissa")
        assert _gap(loaded(x), peft(x)) < 1e-4
        assert loaded.peft_config["pissa"].r == 6

    def test_several_adapters_at_once_are_refused_and_nothing_is_written(
        self, device: str, tmp_path: Path
    ) -> None:
        peft, _, _ = self._two(device, tmp_path)
        with pytest.raises(ValueError, match="selected_adapters"):
            peft.save_pretrained(
                tmp_path / "out",
                path_initial_model_for_weight_conversion=tmp_path / "init" / "pissa",
            )
        assert not (tmp_path / "out").exists()


class TestRefusals:
    def _initial(self, device: str, tmp_path: Path, config: LoraConfig) -> PeftModel:
        peft = get_peft_model(_pristine(device), config)
        peft.save_pretrained(tmp_path / "init")
        _train(peft, lucid.randn(12, 8, device=device))
        return peft

    def _convert(self, peft: PeftModel, tmp_path: Path) -> None:
        peft.save_pretrained(
            tmp_path / "out", path_initial_model_for_weight_conversion=tmp_path / "init"
        )

    def test_dora_has_no_plain_lora_form(self, device: str, tmp_path: Path) -> None:
        peft = self._initial(device, tmp_path, _config(use_dora=True))
        with pytest.raises(ValueError, match="DoRA"):
            self._convert(peft, tmp_path)
        assert not (tmp_path / "out").exists()

    @pytest.mark.filterwarnings("ignore:Using Rank-Stabilized LoRA")
    def test_rslora_with_a_pattern_is_refused(self, device: str, tmp_path: Path) -> None:
        config = _config(use_rslora=True, rank_pattern={"down": 3})
        peft = self._initial(device, tmp_path, config)
        with pytest.raises(ValueError, match="rank_pattern"):
            self._convert(peft, tmp_path)

    def test_another_method_has_no_conversion(self, device: str, tmp_path: Path) -> None:
        peft = get_peft_model(
            _pristine(device), IA3Config(target_modules=["up"], feedforward_modules=["up"])
        )
        peft.save_pretrained(tmp_path / "init")
        with pytest.raises(ValueError, match="not supported for"):
            self._convert(peft, tmp_path)

    def test_an_initial_adapter_of_another_scaling_is_refused(
        self, device: str, tmp_path: Path
    ) -> None:
        peft = self._initial(device, tmp_path, _config())
        other = get_peft_model(_pristine(device), _config(lora_alpha=16))
        other.save_pretrained(tmp_path / "init")
        with pytest.raises(ValueError, match="lora_alpha"):
            self._convert(peft, tmp_path)

    def test_an_initial_adapter_for_other_layers_is_refused(
        self, device: str, tmp_path: Path
    ) -> None:
        peft = self._initial(device, tmp_path, _config())
        other = get_peft_model(_pristine(device), _config(target_modules=["up", "head"]))
        other.save_pretrained(tmp_path / "init")
        with pytest.raises(ValueError, match="different layers"):
            self._convert(peft, tmp_path)

    def test_an_initial_adapter_of_other_shapes_is_refused(
        self, device: str, tmp_path: Path
    ) -> None:
        peft = self._initial(device, tmp_path, _config())
        state = get_peft_model_state_dict(peft)
        wrong = {
            name: lucid.zeros(tensor.shape[0] + 1, *tensor.shape[1:], device=device)
            for name, tensor in state.items()
            if "lora_A" in name
        }
        wrong |= {name: tensor for name, tensor in state.items() if "lora_B" in name}
        save_safetensors(wrong, str(tmp_path / "init" / "adapter_model.safetensors"))
        with pytest.raises(ValueError, match="shape"):
            self._convert(peft, tmp_path)

    def test_a_missing_initial_directory_is_a_file_error(self, device: str, tmp_path: Path) -> None:
        peft = self._initial(device, tmp_path, _config())
        with pytest.raises(FileNotFoundError):
            peft.save_pretrained(
                tmp_path / "out", path_initial_model_for_weight_conversion=tmp_path / "nowhere"
            )
        assert not (tmp_path / "out").exists()


class TestInitsThatLeftTheBaseAlone:
    def test_a_zero_b_start_is_a_plain_lora_already_so_nothing_changes(
        self, device: str, tmp_path: Path
    ) -> None:
        peft, x, out, pristine = _converted(device, tmp_path, _config(True))
        stored = PeftConfig.from_pretrained(out)
        assert isinstance(stored, LoraConfig) and stored.r == 2
        assert _gap(PeftModel.from_pretrained(pristine, out)(x), peft(x)) < 1e-4

    @pytest.mark.parametrize("init", ["gaussian", False])
    def test_other_starts_warn_and_are_saved_as_they_are(
        self, device: str, tmp_path: Path, init: bool | str
    ) -> None:
        pristine = _pristine(device)
        peft = get_peft_model(_pristine(device), _config(init))
        peft.save_pretrained(tmp_path / "init")
        x = lucid.randn(12, 8, device=device)
        _train(peft, x)
        with pytest.warns(UserWarning, match="saved as it is"):
            peft.save_pretrained(
                tmp_path / "out", path_initial_model_for_weight_conversion=tmp_path / "init"
            )
        stored = PeftConfig.from_pretrained(tmp_path / "out")
        assert isinstance(stored, LoraConfig) and stored.r == 2
        assert _gap(PeftModel.from_pretrained(pristine, tmp_path / "out")(x), peft(x)) < 1e-4

    def test_only_a_factorising_strategy_converts(self) -> None:
        converting = {s for s in _Init._strategies.values() if s.factorises}
        assert {s.spelling for s in converting} == {"pissa", "olora"}
        assert isinstance(_Init.of("pissa_niter_3"), _Pissa)
