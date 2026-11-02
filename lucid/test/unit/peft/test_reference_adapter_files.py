"""An adapter directory written by the reference library loads into the zoo's text models, whole.

A user moving over brings ``adapter_config.json`` and ``adapter_model.safetensors`` from the
reference library.  The text models whose modules are named alike on both sides (GPT-2, BERT)
need no converter, so this is the proof that nothing else stands in the way: the config carries
every field of the reference's ``LoraConfig`` (release 0.21.2, ``_reference_signatures.json``),
a GPT-2 file says ``fan_in_fan_out`` as the reference's ``Conv1D`` projections need it, a BERT
classification file holds its head beside the factors.  Each expected output is computed by hand —
the twin model with ``W + s B A`` folded into the weights the test names — and a file saved back
is the same keys, so the adapter can go back to the reference.

The architectures with a converter (T5, ViT, RoFormer) are held in ``test_convert.py``; the
reference's decoder families (LLaMA, Mistral …) have no zoo counterpart.
"""

from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import PeftModel, get_peft_model_state_dict
from lucid.test.unit.peft._reference_files import (
    MODEL_PREFIX,
    lora_tensors,
    read_keys,
    reference_lora_config,
    write_adapter,
)

# The reference writes its (empty) runtime configuration into the file and ignores it on load, with this warning.
pytestmark = pytest.mark.filterwarnings("ignore:The configuration file contains a `runtime_config`")

_TEXT: dict[str, object] = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 32,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}
_DIM = 32
_RANK = 4
_ALPHA = 8
_SCALE = _ALPHA / _RANK


def _model(factory: str, **overrides: object) -> nn.Module:
    lucid.manual_seed(0)
    return create_model(factory, **{**_TEXT, **overrides}).eval()


def _factors(seed: int, out: int, inn: int) -> tuple[lucid.Tensor, lucid.Tensor]:
    """``(lora_A, lora_B)`` as the reference stores them: ``(r, in)`` and ``(out, r)``."""
    lucid.manual_seed(seed)
    return lucid.randn(_RANK, inn) * 0.3, lucid.randn(out, _RANK) * 0.3


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool(lucid.allclose(a, b, atol=1e-5, rtol=1e-4))


def _folded(model: nn.Module, deltas: dict[str, lucid.Tensor]) -> nn.Module:
    """``model`` with each named weight moved by its delta."""
    state = model.state_dict()
    for key, delta in deltas.items():
        state[key] = state[key] + delta
    model.load_state_dict(state)
    return model.eval()


def _gpt2_file(directory: Path) -> tuple[dict[str, lucid.Tensor], dict[str, lucid.Tensor]]:
    """A causal-LM adapter on ``c_attn`` and ``mlp.c_proj``; also the weight delta of each."""
    shapes = {"attn.c_attn": (3 * _DIM, _DIM), "mlp.c_proj": (_DIM, 2 * _DIM)}
    tensors: dict[str, lucid.Tensor] = {}
    deltas: dict[str, lucid.Tensor] = {}
    for i in range(2):
        for j, (part, (out, inn)) in enumerate(shapes.items()):
            module = f"transformer.h.{i}.{part}"
            a, b = _factors(10 * i + j, out, inn)
            tensors.update(lora_tensors(module, a, b))
            deltas[f"{module}.weight"] = _SCALE * (b @ a)
    config = reference_lora_config(
        "GPT2LMHeadModel",
        "transformers.models.gpt2.modeling_gpt2",
        ["c_attn", "mlp.c_proj"],
        r=_RANK,
        lora_alpha=_ALPHA,
        fan_in_fan_out=True,
        task_type="CAUSAL_LM",
    )
    write_adapter(directory, config, tensors)
    return tensors, deltas


def test_gpt2_causal_lm_adapter_loads_and_matches_the_hand_folded_model(tmp_path: Path) -> None:
    tensors, deltas = _gpt2_file(tmp_path / "ref")
    base = _model("gpt2_small_lm")
    expected = _folded(_model("gpt2_small_lm"), deltas)
    # The reference's flag describes its Conv1D (in, out) weights; the zoo's are nn.Linear.
    with pytest.warns(UserWarning, match="fan_in_fan_out"):
        loaded = PeftModel.from_pretrained(base, tmp_path / "ref")
    config = loaded.peft_config["default"]
    assert config.target_modules == {"c_attn", "mlp.c_proj"}
    assert config.fan_in_fan_out is False
    assert config.task_type == "CAUSAL_LM"
    ours = get_peft_model_state_dict(loaded)
    assert set(ours) == set(tensors)
    for key, tensor in tensors.items():
        assert _same(ours[key], tensor), key
    lucid.manual_seed(3)
    ids = lucid.randint(0, 64, (3, 10))
    with lucid.no_grad():
        assert _same(loaded(ids).logits, expected(ids).logits)


def test_gpt2_adapter_saved_back_is_the_reference_layout(tmp_path: Path) -> None:
    tensors, _ = _gpt2_file(tmp_path / "ref")
    with pytest.warns(UserWarning, match="fan_in_fan_out"):
        loaded = PeftModel.from_pretrained(_model("gpt2_small_lm"), tmp_path / "ref")
    loaded.save_pretrained(tmp_path / "back")
    assert read_keys(tmp_path / "back") == set(tensors)
    again = PeftModel.from_pretrained(_model("gpt2_small_lm"), tmp_path / "back")
    ids = lucid.randint(0, 64, (2, 8))
    with lucid.no_grad():
        assert _same(again(ids).logits, loaded(ids).logits)


def _bert_file(
    directory: Path,
) -> tuple[dict[str, lucid.Tensor], dict[str, lucid.Tensor], lucid.Tensor, lucid.Tensor]:
    """A sequence-classification adapter on ``query`` and ``value``, with its trained head."""
    tensors: dict[str, lucid.Tensor] = {}
    deltas: dict[str, lucid.Tensor] = {}
    for i in range(2):
        for j, part in enumerate(("query", "value")):
            module = f"bert.encoder.layer.{i}.attention.self.{part}"
            a, b = _factors(20 + 10 * i + j, _DIM, _DIM)
            tensors.update(lora_tensors(module, a, b))
            deltas[f"{module}.weight"] = _SCALE * (b @ a)
    lucid.manual_seed(90)
    head_weight, head_bias = lucid.randn(3, _DIM) * 0.3, lucid.randn(3) * 0.3
    tensors[f"{MODEL_PREFIX}classifier.weight"] = head_weight
    tensors[f"{MODEL_PREFIX}classifier.bias"] = head_bias
    config = reference_lora_config(
        "BertForSequenceClassification",
        "transformers.models.bert.modeling_bert",
        ["query", "value"],
        r=_RANK,
        lora_alpha=_ALPHA,
        modules_to_save=["classifier"],
        task_type="SEQ_CLS",
    )
    write_adapter(directory, config, tensors)
    return tensors, deltas, head_weight, head_bias


def test_bert_classification_adapter_loads_with_its_head(tmp_path: Path) -> None:
    tensors, deltas, head_weight, head_bias = _bert_file(tmp_path / "ref")
    base = _model("bert_base_cls", num_labels=3)
    twin = _folded(_model("bert_base_cls", num_labels=3), deltas)
    state = twin.state_dict()
    state["classifier.weight"], state["classifier.bias"] = head_weight, head_bias
    twin.load_state_dict(state)
    loaded = PeftModel.from_pretrained(base, tmp_path / "ref")
    config = loaded.peft_config["default"]
    assert config.target_modules == {"query", "value"}
    assert config.modules_to_save == ["classifier"]
    ours = get_peft_model_state_dict(loaded)
    assert set(ours) == set(tensors)
    for key, tensor in tensors.items():
        assert _same(ours[key], tensor), key
    lucid.manual_seed(4)
    ids = lucid.randint(0, 64, (3, 10))
    with lucid.no_grad():
        assert _same(loaded(ids).logits, twin(ids).logits)


def test_bert_adapter_saved_back_is_the_reference_layout(tmp_path: Path) -> None:
    tensors, *_ = _bert_file(tmp_path / "ref")
    loaded = PeftModel.from_pretrained(_model("bert_base_cls", num_labels=3), tmp_path / "ref")
    loaded.save_pretrained(tmp_path / "back")
    assert read_keys(tmp_path / "back") == set(tensors)


def test_a_file_for_another_architecture_is_refused_not_misplaced(tmp_path: Path) -> None:
    """GPT-2's keys onto a BERT: nothing silently lands on a module that merely shares a suffix."""
    _gpt2_file(tmp_path / "ref")
    with pytest.raises(ValueError, match="not found"), pytest.warns(UserWarning):
        PeftModel.from_pretrained(_model("bert_base_cls", num_labels=3), tmp_path / "ref")


_VIT: dict[str, object] = {
    "image_size": 16,
    "patch_size": 8,
    "dim": _DIM,
    "depth": 2,
    "num_heads": 2,
    "num_classes": 5,
    "dropout": 0.0,
    "attention_dropout": 0.0,
}


def _vit() -> nn.Module:
    lucid.manual_seed(0)
    model = create_model("vit_base_16_cls", **_VIT)
    nn.init.normal_(model.head.weight, std=0.2)  # the zoo's head starts at zero
    return model.eval()


def _vit_release5_file(directory: Path) -> dict[str, lucid.Tensor]:
    """A ViT adapter on ``q_proj`` / ``v_proj`` / ``o_proj`` as release 5 of the reference names them.

    Returns the weight delta each adapted zoo weight should see.
    """
    tensors: dict[str, lucid.Tensor] = {}
    deltas: dict[str, lucid.Tensor] = {}
    for i in range(2):
        fused = lucid.zeros(3 * _DIM, _DIM)
        for j, part in enumerate(("q_proj", "v_proj", "o_proj")):
            module = f"vit.layers.{i}.attention.{part}"
            a, b = _factors(40 + 10 * i + j, _DIM, _DIM)
            tensors.update(lora_tensors(module, a, b))
            delta = _SCALE * (b @ a)
            if part == "o_proj":
                deltas[f"blocks.{i}.attn.proj.weight"] = delta
                continue
            start = {"q_proj": 0, "v_proj": 2}[part] * _DIM
            fused[start : start + _DIM] = delta
        deltas[f"blocks.{i}.attn.qkv.weight"] = fused
    config = reference_lora_config(
        "ViTForImageClassification",
        "transformers.models.vit.modeling_vit",
        ["q_proj", "v_proj", "o_proj"],
        r=_RANK,
        lora_alpha=_ALPHA,
    )
    write_adapter(directory, config, tensors)
    return deltas


def test_vit_adapter_in_the_release_5_layout_fuses_into_qkv(tmp_path: Path) -> None:
    """Release 5 of the reference renamed ViT's modules (``vit.layers.N.attention.q_proj``)."""
    twin = _folded(_vit(), _vit_release5_file(tmp_path / "ref"))
    loaded = PeftModel.from_pretrained(_vit(), tmp_path / "ref")
    adapted = loaded.peft_config["default"].target_modules
    assert adapted == {f"blocks.{i}.attn.{p}" for i in range(2) for p in ("qkv", "proj")}
    lucid.manual_seed(5)
    pixels = lucid.randn(2, 3, 16, 16)
    with lucid.no_grad():
        assert _same(loaded(pixels).logits, twin(pixels).logits)


def test_vit_adapter_in_the_release_5_layout_saves_back_fused(tmp_path: Path) -> None:
    _vit_release5_file(tmp_path / "ref")
    loaded = PeftModel.from_pretrained(_vit(), tmp_path / "ref")
    loaded.save_pretrained(tmp_path / "back")
    keys = read_keys(tmp_path / "back")
    assert f"{MODEL_PREFIX}blocks.0.attn.qkv.lora_A.weight" in keys
    assert f"{MODEL_PREFIX}blocks.1.attn.proj.lora_B.weight" in keys
    again = PeftModel.from_pretrained(_vit(), tmp_path / "back").eval()
    lucid.manual_seed(6)
    pixels = lucid.randn(2, 3, 16, 16)
    with lucid.no_grad():
        assert _same(again(pixels).logits, loaded(pixels).logits)
