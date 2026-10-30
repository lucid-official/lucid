"""Adapter state and files: the reference library's keys, a lossless round trip, typed refusals.

The expected key lists are written by hand from the reference's format:
``base_model.model.`` + the module path + ``lora_A.weight`` /
``lora_B.weight`` (adapter name taken out), a ``modules_to_save`` module
under its own path, and the safetensors header tagged ``format: pt``.
"""

import json
import struct
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from lucid.serialization import save_safetensors


class _Attention(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.c_proj(self.c_attn(x)[..., : x.shape[-1]])


class _Layer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.attn = _Attention(d)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x + self.attn(x)


class _Trunk(nn.Module):
    def __init__(self, d: int, depth: int) -> None:
        super().__init__()
        self.h = nn.ModuleList([_Layer(d) for _ in range(depth)])

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        for layer in self.h:
            x = layer(x)
        return x


class _GPTShaped(nn.Module):
    """GPT-2's module names: ``transformer.h.N.attn.c_attn`` and a ``score`` head."""

    def __init__(self, d: int = 8, depth: int = 2) -> None:
        super().__init__()
        self.transformer = _Trunk(d, depth)
        self.score = nn.Linear(d, 2, bias=False)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.score(self.transformer(x))


def _model(device: str = "cpu") -> _GPTShaped:
    lucid.manual_seed(0)
    return _GPTShaped().to(device)


def _config(**kwargs: object) -> LoraConfig:
    fields: dict[str, object] = {
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["c_attn"],
        "init_lora_weights": False,
    } | kwargs
    return LoraConfig(**fields)  # type: ignore[arg-type]


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _header(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    (length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + length])
    assert isinstance(header, dict)
    return header


#: The reference's on-disk keys for the model and config above.
DISK_KEYS = [
    "base_model.model.score.weight",
    "base_model.model.transformer.h.0.attn.c_attn.lora_A.weight",
    "base_model.model.transformer.h.0.attn.c_attn.lora_B.weight",
    "base_model.model.transformer.h.1.attn.c_attn.lora_A.weight",
    "base_model.model.transformer.h.1.attn.c_attn.lora_B.weight",
]


class TestStateDict:
    def test_keys_are_the_reference_s(self) -> None:
        peft = get_peft_model(_model(), _config(modules_to_save=["score"]))
        state = get_peft_model_state_dict(peft)
        assert sorted(state) == DISK_KEYS
        assert tuple(state[DISK_KEYS[1]].shape) == (4, 8)
        assert tuple(state[DISK_KEYS[2]].shape) == (24, 4)

    def test_an_injected_model_has_no_prefix(self) -> None:
        model = inject_adapter_in_model(_config(), _model())
        assert sorted(get_peft_model_state_dict(model)) == [
            k.removeprefix("base_model.model.") for k in DISK_KEYS[1:]
        ]

    @pytest.mark.parametrize(
        ("bias", "extra"),
        [
            (
                "lora_only",
                [
                    "base_model.model.transformer.h.0.attn.c_attn.base_layer.bias",
                    "base_model.model.transformer.h.1.attn.c_attn.base_layer.bias",
                ],
            ),
            (
                "all",
                [
                    "base_model.model.transformer.h.0.attn.c_attn.base_layer.bias",
                    "base_model.model.transformer.h.0.attn.c_proj.bias",
                    "base_model.model.transformer.h.1.attn.c_attn.base_layer.bias",
                    "base_model.model.transformer.h.1.attn.c_proj.bias",
                ],
            ),
        ],
    )
    def test_trained_biases_are_saved(self, bias: str, extra: list[str]) -> None:
        peft = get_peft_model(_model(), _config(bias=bias))
        assert sorted(get_peft_model_state_dict(peft)) == sorted(DISK_KEYS[1:] + extra)

    def test_values_can_come_from_a_given_state_dict(self) -> None:
        peft = get_peft_model(_model(), _config())
        full = {k: lucid.zeros_like(v) for k, v in peft.state_dict().items()}
        state = get_peft_model_state_dict(peft, state_dict=full)
        assert all(float(v.abs().sum().item()) == 0 for v in state.values())
        partial = {k: v for k, v in full.items() if "lora_B" not in k}
        with pytest.raises(KeyError, match="lora_B"):
            get_peft_model_state_dict(peft, state_dict=partial)

    def test_a_model_without_an_adapter_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no LoRA adapter named 'default'"):
            get_peft_model_state_dict(_model())
        with pytest.raises(ValueError, match="no LoRA adapter named 'other'"):
            get_peft_model_state_dict(get_peft_model(_model(), _config()), adapter_name="other")

    def test_set_round_trips(self, device: str) -> None:
        source = get_peft_model(_model(device), _config(modules_to_save=["score"]))
        lucid.manual_seed(1)
        target = get_peft_model(_GPTShaped().to(device), _config(modules_to_save=["score"]))
        state = get_peft_model_state_dict(source)
        set_peft_model_state_dict(target, state)
        for key, value in get_peft_model_state_dict(target).items():
            assert _same(value, state[key]), key

    def test_set_refuses_what_does_not_fit_and_changes_nothing(self) -> None:
        peft = get_peft_model(_model(), _config())
        before = get_peft_model_state_dict(peft)
        good = {k: lucid.zeros_like(v) for k, v in before.items()}
        unexpected = {**good, "base_model.model.nope.lora_A.weight": lucid.zeros(4, 8)}
        with pytest.raises(ValueError, match="no such tensor"):
            set_peft_model_state_dict(peft, unexpected)
        wrong = {**good, DISK_KEYS[1]: lucid.zeros(2, 8)}
        with pytest.raises(ValueError, match="shape"):
            set_peft_model_state_dict(peft, wrong)
        for key, value in get_peft_model_state_dict(peft).items():
            assert _same(value, before[key])
        with pytest.raises(TypeError, match="not a Tensor"):
            set_peft_model_state_dict(peft, {DISK_KEYS[1]: [0.0]})  # type: ignore[dict-item]

    def test_set_warns_about_missing_adapter_keys(self) -> None:
        peft = get_peft_model(_model(), _config())
        state = get_peft_model_state_dict(peft)
        del state[DISK_KEYS[4]]
        with pytest.warns(UserWarning, match="missing adapter keys"):
            set_peft_model_state_dict(peft, state)


class TestFiles:
    def test_the_directory_is_the_reference_layout(self, tmp_path: Path) -> None:
        peft = get_peft_model(_model(), _config(modules_to_save=["score"], task_type="SEQ_CLS"))
        peft.save_pretrained(tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]
        header = _header(tmp_path / "adapter_model.safetensors")
        assert header.pop("__metadata__") == {"format": "pt"}
        assert sorted(header) == DISK_KEYS
        entry = header[DISK_KEYS[1]]
        assert isinstance(entry, dict) and entry["dtype"] == "F32" and entry["shape"] == [4, 8]
        config = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
        assert list(config) == sorted(config)
        assert config["peft_type"] == "LORA" and config["task_type"] == "SEQ_CLS"
        assert config["inference_mode"] is True and config["target_modules"] == ["c_attn"]
        assert config["modules_to_save"] == ["score"]
        assert peft.peft_config["default"].inference_mode is False

    def test_save_then_load_reproduces_the_adapter(self, device: str, tmp_path: Path) -> None:
        trained = get_peft_model(_model(device), _config(modules_to_save=["score"]))
        x = lucid.randn(3, 8, device=device)
        expected = trained(x)
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_model(device), tmp_path)
        assert _same(loaded(x), expected)
        assert not loaded.training
        assert not any(p.requires_grad for p in loaded.parameters())
        for key, value in get_peft_model_state_dict(loaded).items():
            assert value.device == expected.device, key

    def test_is_trainable_keeps_the_adapter_trainable(self, tmp_path: Path) -> None:
        get_peft_model(_model(), _config()).save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_model(), tmp_path, is_trainable=True)
        assert loaded.training
        assert loaded.get_nb_trainable_parameters()[0] == 2 * (4 * 8 + 24 * 4)
        assert loaded.peft_config["default"].inference_mode is False

    def test_a_given_config_overrides_the_file(self, tmp_path: Path) -> None:
        get_peft_model(_model(), _config()).save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_model(), tmp_path, config=_config(lora_alpha=16))
        assert loaded.peft_config["default"].lora_alpha == 16
        with pytest.raises(TypeError, match="LoraConfig"):
            PeftModel.from_pretrained(_model(), tmp_path, config={"r": 4})  # type: ignore[arg-type]

    def test_a_reference_written_adapter_loads(self, tmp_path: Path) -> None:
        source = get_peft_model(_model(), _config())
        state = get_peft_model_state_dict(source)
        save_safetensors(
            state, str(tmp_path / "adapter_model.safetensors"), metadata={"format": "pt"}
        )
        (tmp_path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "gpt2",
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": 4,
                    "lora_alpha": 8,
                    "lora_dropout": 0.1,
                    "target_modules": ["c_attn"],
                    "fan_in_fan_out": True,
                    "use_dora": False,
                    "lora_bias": False,
                    "loftq_config": {},
                    "megatron_core": "megatron.core",
                    "peft_version": "0.17.1",
                },
                indent=2,
            )
        )
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            loaded = PeftModel.from_pretrained(_model(), tmp_path)
        x = lucid.randn(2, 8)
        assert _same(loaded(x), source(x))

    def test_weights_that_do_not_fit_undo_the_injection(self, tmp_path: Path) -> None:
        get_peft_model(_model(), _config(r=2)).save_pretrained(tmp_path)
        model = _model()
        with pytest.raises(ValueError, match="shape"):
            PeftModel.from_pretrained(model, tmp_path, config=_config(r=4))
        assert type(model.transformer.h[0].attn.c_attn) is nn.Linear
        assert all(p.requires_grad for p in model.parameters())

    def test_missing_files_are_named(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            PeftModel.from_pretrained(_model(), tmp_path)
        _config().save_pretrained(tmp_path)
        with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
            PeftModel.from_pretrained(_model(), tmp_path)
        (tmp_path / "adapter_model.bin").write_bytes(b"\x80")
        with pytest.raises(NotImplementedError, match="pickle"):
            PeftModel.from_pretrained(_model(), tmp_path)

    def test_pickle_output_is_refused(self, tmp_path: Path) -> None:
        peft = get_peft_model(_model(), _config())
        with pytest.raises(NotImplementedError, match="safetensors"):
            peft.save_pretrained(tmp_path, safe_serialization=False)
        assert not any(tmp_path.iterdir())

    def test_a_named_adapter_saves_into_its_own_directory(self, tmp_path: Path) -> None:
        peft = get_peft_model(_model(), _config(), adapter_name="style")
        peft.save_pretrained(tmp_path)
        assert [p.name for p in tmp_path.iterdir()] == ["style"]
        assert sorted(p.name for p in (tmp_path / "style").iterdir()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]
        loaded = PeftModel.from_pretrained(_model(), tmp_path / "style", adapter_name="style")
        assert sorted(get_peft_model_state_dict(loaded, adapter_name="style")) == DISK_KEYS[1:]

    @pytest.mark.parametrize("finish", ["merge_and_unload", "unload"])
    def test_nothing_is_saved_once_the_adapter_is_gone(self, finish: str, tmp_path: Path) -> None:
        peft = get_peft_model(_model(), _config())
        getattr(peft, finish)()
        with pytest.raises(RuntimeError, match="merged or unloaded"):
            peft.save_pretrained(tmp_path / "out")
        assert not (tmp_path / "out").exists()
        with pytest.raises(RuntimeError, match="merged or unloaded"):
            get_peft_model_state_dict(peft)

    def test_any_failure_while_loading_undoes_the_injection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        get_peft_model(_model(), _config()).save_pretrained(tmp_path)

        def interrupted(*args: object, **kwargs: object) -> object:
            raise KeyboardInterrupt

        monkeypatch.setattr("lucid.peft._peft_model.set_peft_model_state_dict", interrupted)
        model = _model()
        with pytest.raises(KeyboardInterrupt):
            PeftModel.from_pretrained(model, tmp_path)
        assert type(model.transformer.h[0].attn.c_attn) is nn.Linear
        assert all(p.requires_grad for p in model.parameters())
