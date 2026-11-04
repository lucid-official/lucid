"""A LoRA adapter file written for the reference library's model gives the same outputs here.

The unit tests (``lucid/test/unit/peft/test_reference_adapter_files.py``, ``test_convert.py``) prove
the file's tensors land in the right zoo modules and that the output equals the zoo model with
``W + s B A`` folded in.  This closes the loop against the reference model itself: a tiny model of the
reference library and the zoo's counterpart are given the same weights (a key map per architecture,
the transposed ``Conv1D`` projections of GPT-2, the fused ``qkv`` of the zoo's ViT), the adapter's delta
is folded into the *reference* model's weights, the file is loaded into the zoo model with
``PeftModel.from_pretrained``, and the two forward passes must agree.

The architectures here are the ones with an adapter converter or a layout the file crosses
(GPT-2, BERT, T5, ViT).  Run with ``-m parity``; needs the reference framework and its model library.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import PeftModel
from lucid.peft._convert import t5_reference_name
from lucid.test.unit.peft._reference_files import (
    MODEL_PREFIX,
    lora_tensors,
    reference_lora_config,
    write_adapter,
)

pytestmark = [
    pytest.mark.parity,
    pytest.mark.filterwarnings("ignore:The configuration file contains a `runtime_config`"),
    pytest.mark.filterwarnings("ignore:fan_in_fan_out is set to True"),
]

_RANK = 4
_ALPHA = 8
_SCALE = _ALPHA / _RANK
_ATOL = 2e-4

type State = dict[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class _Target:
    """One adapted module: its path in the reference model and the shape of its weight, ``(out, in)``."""

    module: str
    out: int
    inn: int


@dataclasses.dataclass(frozen=True)
class _Arch:
    """What one architecture needs to cross: its models, weight map, adapter and forward."""

    name: str
    reference_model: Callable[[Any], Any]
    zoo_model: Callable[[], nn.Module]
    transfer: Callable[[State], State]
    #: Both read the reference model's state, for a layout that depends on its release.
    targets: Callable[[State], list[_Target]]
    config: Callable[[State], dict[str, object]]
    forward: Callable[[Any, Callable[[np.ndarray], Any]], np.ndarray]
    conv1d: bool = False
    #: Head tensors the file holds beside the factors, by reference key (no ``base_model.model.``).
    head: Callable[[], dict[str, np.ndarray]] = dict


def _noise(seed: int, *shape: int) -> np.ndarray:
    return (np.random.RandomState(seed).randn(*shape) * 0.3).astype(np.float32)


def _factors(seed: int, target: _Target) -> tuple[np.ndarray, np.ndarray]:
    return _noise(seed, _RANK, target.inn), _noise(seed + 1, target.out, _RANK)


def _tensor(array: np.ndarray) -> lucid.Tensor:
    return lucid.tensor(array)


# ── GPT-2 ────────────────────────────────────────────────────────────────────

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
_IDS = np.random.RandomState(7).randint(2, 64, (3, 9)).astype(np.int64)
_CONV1D = (".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight")


def _gpt2_reference(transformers: Any) -> Any:
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=64,
            n_embd=_DIM,
            n_layer=2,
            n_head=2,
            n_positions=32,
            n_inner=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=0,
            eos_token_id=0,
        )
    )


def _gpt2_transfer(state: State) -> State:
    return {k: (v.T if k.endswith(_CONV1D) else v) for k, v in state.items()}


def _run_ids(ref: ModuleType, ids: np.ndarray) -> Callable[[Any], Any]:
    def run(model: Any) -> Any:
        if isinstance(model, nn.Module):
            return model(lucid.tensor(ids))
        return model(ref.from_numpy(ids))

    return run


# ── BERT ─────────────────────────────────────────────────────────────────────

_BERT_HEAD = 3


def _bert_reference(transformers: Any) -> Any:
    return transformers.BertForSequenceClassification(
        transformers.BertConfig(
            vocab_size=64,
            hidden_size=_DIM,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=32,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            num_labels=_BERT_HEAD,
        )
    )


def _identity(state: State) -> State:
    return dict(state)


def _bert_head() -> dict[str, np.ndarray]:
    return {
        "classifier.weight": _noise(90, _BERT_HEAD, _DIM),
        "classifier.bias": _noise(91, _BERT_HEAD),
    }


# ── T5 ───────────────────────────────────────────────────────────────────────

_T5_DEC = np.random.RandomState(8).randint(2, 64, (3, 5)).astype(np.int64)


def _t5_reference(transformers: Any) -> Any:
    return transformers.T5ForConditionalGeneration(
        transformers.T5Config(
            vocab_size=64,
            d_model=_DIM,
            d_kv=16,
            d_ff=64,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=2,
            dropout_rate=0.0,
            feed_forward_proj="relu",
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
    )


def _t5_transfer(state: State) -> State:
    out: State = {}
    for key, value in state.items():
        if key.endswith(".embed_tokens.weight"):  # aliases of the shared table
            continue
        mapped = t5_reference_name(key, trunked=True)
        if mapped is None:
            raise KeyError(key)
        out[mapped] = value
    return out


# ── ViT ──────────────────────────────────────────────────────────────────────

_PIXELS = np.random.RandomState(9).randn(2, 3, 16, 16).astype(np.float32)


def _vit_reference(transformers: Any) -> Any:
    return transformers.ViTForImageClassification(
        transformers.ViTConfig(
            image_size=16,
            patch_size=8,
            hidden_size=_DIM,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=4 * _DIM,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            layer_norm_eps=1e-6,
            num_labels=5,
        )
    )


def _release5(state: State) -> bool:
    """Whether the reference's ViT is the release-5 layout (``vit.layers.N.attention.q_proj``)."""
    return "vit.layers.0.attention.q_proj.weight" in state


def _vit_projections(state: State) -> tuple[str, str]:
    return ("q_proj", "v_proj") if _release5(state) else ("query", "value")


def _vit_block(state: State, i: int) -> str:
    """The reference's path of block ``i``'s attention."""
    if _release5(state):
        return f"vit.layers.{i}.attention"
    return f"vit.encoder.layer.{i}.attention.attention"


def _vit_transfer(state: State) -> State:
    out: State = {
        "cls_token": state["vit.embeddings.cls_token"],
        "pos_embed": state["vit.embeddings.position_embeddings"],
        "patch_embed.proj.weight": state["vit.embeddings.patch_embeddings.projection.weight"],
        "patch_embed.proj.bias": state["vit.embeddings.patch_embeddings.projection.bias"],
        "norm.weight": state["vit.layernorm.weight"],
        "norm.bias": state["vit.layernorm.bias"],
        "head.weight": state["classifier.weight"],
        "head.bias": state["classifier.bias"],
    }
    if _release5(state):
        layer_of, roles = "vit.layers.{i}", ("q_proj", "k_proj", "v_proj")
        renames = {
            "layernorm_before": "norm1",
            "attention.o_proj": "attn.proj",
            "layernorm_after": "norm2",
            "mlp.fc1": "mlp.fc1",
            "mlp.fc2": "mlp.fc2",
        }
    else:
        layer_of, roles = "vit.encoder.layer.{i}", ("query", "key", "value")
        renames = {
            "layernorm_before": "norm1",
            "attention.output.dense": "attn.proj",
            "layernorm_after": "norm2",
            "intermediate.dense": "mlp.fc1",
            "output.dense": "mlp.fc2",
        }
    for i in range(2):
        layer = layer_of.format(i=i)
        for part in ("weight", "bias"):
            fused = [state[f"{_vit_block(state, i)}.{p}.{part}"] for p in roles]
            out[f"blocks.{i}.attn.qkv.{part}"] = np.concatenate(fused, axis=0)
            for old, new in renames.items():
                out[f"blocks.{i}.{new}.{part}"] = state[f"{layer}.{old}.{part}"]
    return out


def _vit_forward(ref: ModuleType) -> Callable[[Any], Any]:
    def run(model: Any) -> Any:
        if isinstance(model, nn.Module):
            return model(lucid.tensor(_PIXELS))
        return model(ref.from_numpy(_PIXELS))

    return run


def _t5_forward(ref: ModuleType) -> Callable[[Any], Any]:
    def run(model: Any) -> Any:
        if isinstance(model, nn.Module):
            return model(lucid.tensor(_IDS), decoder_input_ids=lucid.tensor(_T5_DEC))
        return model(input_ids=ref.from_numpy(_IDS), decoder_input_ids=ref.from_numpy(_T5_DEC))

    return run


def _arch(name: str, ref: ModuleType) -> _Arch:
    gpt2_targets = [
        _Target(f"transformer.h.{i}.{part}", out, inn)
        for i in range(2)
        for part, out, inn in (
            ("attn.c_attn", 3 * _DIM, _DIM),
            ("attn.c_proj", _DIM, _DIM),
            ("mlp.c_proj", _DIM, 2 * _DIM),
        )
    ]
    if name == "gpt2":
        return _Arch(
            name,
            _gpt2_reference,
            lambda: create_model("gpt2_small_lm", **_TEXT),
            _gpt2_transfer,
            lambda _state: gpt2_targets,
            lambda _state: {
                "class_name": "GPT2LMHeadModel",
                "library": "transformers.models.gpt2.modeling_gpt2",
                "targets": ["c_attn", "c_proj"],
                "fan_in_fan_out": True,
                "task_type": "CAUSAL_LM",
            },
            _run_ids(ref, _IDS),
            conv1d=True,
        )
    if name == "bert":
        return _Arch(
            name,
            _bert_reference,
            lambda: create_model("bert_base_cls", num_labels=_BERT_HEAD, **_TEXT),
            _identity,
            lambda _state: [
                _Target(f"bert.encoder.layer.{i}.attention.self.{p}", _DIM, _DIM)
                for i in range(2)
                for p in ("query", "value")
            ],
            lambda _state: {
                "class_name": "BertForSequenceClassification",
                "library": "transformers.models.bert.modeling_bert",
                "targets": ["query", "value"],
                "modules_to_save": ["classifier"],
                "task_type": "SEQ_CLS",
            },
            _run_ids(ref, _IDS),
            head=_bert_head,
        )
    if name == "t5":
        return _Arch(
            name,
            _t5_reference,
            lambda: create_model("t5_small_seq2seq", **_TEXT, num_decoder_layers=2, head_dim=16),
            _t5_transfer,
            lambda _state: [
                _Target(f"{stack}.block.{i}.layer.{layer}.{attention}.{p}", _DIM, _DIM)
                for stack, layer, attention in (
                    ("encoder", 0, "SelfAttention"),
                    ("decoder", 0, "SelfAttention"),
                    ("decoder", 1, "EncDecAttention"),
                )
                for i in range(2)
                for p in ("q", "v")
            ],
            lambda _state: {
                "class_name": "T5ForConditionalGeneration",
                "library": "transformers.models.t5.modeling_t5",
                "targets": ["q", "v"],
                "task_type": "SEQ_2_SEQ_LM",
            },
            _t5_forward(ref),
        )
    return _Arch(
        name,
        _vit_reference,
        lambda: create_model(
            "vit_base_16_cls",
            image_size=16,
            patch_size=8,
            dim=_DIM,
            depth=2,
            num_heads=2,
            num_classes=5,
            dropout=0.0,
            attention_dropout=0.0,
        ),
        _vit_transfer,
        lambda state: [
            _Target(f"{_vit_block(state, i)}.{p}", _DIM, _DIM)
            for i in range(2)
            for p in _vit_projections(state)
        ],
        lambda state: {
            "class_name": "ViTForImageClassification",
            "library": "transformers.models.vit.modeling_vit",
            "targets": list(_vit_projections(state)),
            "modules_to_save": ["classifier"],
            "task_type": None,
        },
        _vit_forward(ref),
        head=lambda: {"classifier.weight": _noise(95, 5, _DIM), "classifier.bias": _noise(96, 5)},
    )


def _outputs(run: Callable[[Any], Any], model: Any) -> np.ndarray:
    output = run(model)
    logits = output.logits
    return np.asarray(logits.numpy())


@pytest.mark.parametrize("name", ["gpt2", "bert", "t5", "vit"])
def test_the_file_gives_the_reference_model_s_outputs(
    ref: ModuleType, name: str, tmp_path: Path
) -> None:
    transformers = pytest.importorskip("transformers")
    arch = _arch(name, ref)
    run = arch.forward
    ref.manual_seed(0)
    source = arch.reference_model(transformers).eval()
    zoo = arch.zoo_model().eval()
    state = {k: v.detach().cpu().numpy() for k, v in source.state_dict().items()}
    zoo.load_state_dict({k: _tensor(v) for k, v in arch.transfer(state).items()})

    with ref.no_grad(), lucid.no_grad():
        np.testing.assert_allclose(
            _outputs(run, zoo),
            _outputs(run, source),
            atol=_ATOL,
            rtol=1e-4,
            err_msg="the base models differ",
        )
        plain = _outputs(run, source)

    tensors: dict[str, lucid.Tensor] = {}
    with ref.no_grad():
        for n, target in enumerate(arch.targets(state)):
            a, b = _factors(100 + 2 * n, target)
            tensors.update(lora_tensors(target.module, _tensor(a), _tensor(b)))
            delta = _SCALE * (b @ a)
            weight = source.get_parameter(f"{target.module}.weight")
            weight.add_(ref.from_numpy((delta.T if arch.conv1d else delta).copy()))
        for key, value in arch.head().items():
            tensors[f"{MODEL_PREFIX}{key}"] = _tensor(value)
            source.get_parameter(key).copy_(ref.from_numpy(value))
    config = reference_lora_config(r=_RANK, lora_alpha=_ALPHA, **arch.config(state))
    write_adapter(tmp_path / "adapter", config, tensors)

    loaded = PeftModel.from_pretrained(zoo, tmp_path / "adapter").eval()
    with ref.no_grad(), lucid.no_grad():
        wanted = _outputs(run, source)
        got = _outputs(run, loaded)
    assert (
        np.abs(wanted - plain).max() > 1e-3
    ), "the adapter changed nothing; the check proves nothing"
    np.testing.assert_allclose(
        got,
        wanted,
        atol=_ATOL,
        rtol=1e-4,
        err_msg="the loaded adapter differs from the folded reference",
    )
