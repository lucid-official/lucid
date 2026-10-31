"""Adapters written for the reference library's models load into the zoo's (LCD-718).

Each fixture is spelled the way the reference library writes its file — its module
names, its factor shapes, one tensor per projection — and loaded into a tiny zoo model
built with the registered factory and a config override (H11).  The expected output is
computed by hand from the file's tensors: the model with ``W + s B A`` folded into the
weights the test names itself, not through any of the converter's own maps.
"""

import json
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import IA3Config, LoraConfig, PeftModel, get_peft_model_state_dict
from lucid.peft._convert import (
    AdapterConverter,
    AdapterFile,
    ConversionContext,
    UnmappedKey,
    ViTConverter,
    convert_adapter,
    split_key,
    t5_reference_name,
)
from lucid.serialization import load_safetensors, save_safetensors
from tools import convert_adapters

_PREFIX = "base_model.model."
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
_T5: dict[str, object] = {**_TEXT, "num_decoder_layers": 2, "head_dim": 16}
_VIT: dict[str, object] = {
    "image_size": 16,
    "patch_size": 8,
    "dim": 32,
    "depth": 2,
    "num_heads": 2,
    "num_classes": 5,
    "dropout": 0.0,
    "attention_dropout": 0.0,
}
_DIM = 32


def _base(factory: str, overrides: dict[str, object]) -> nn.Module:
    lucid.manual_seed(0)
    model = create_model(factory, **overrides)
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        nn.init.normal_(head.weight, std=0.2)  # the zoo's ViT head starts at zero
    return model.eval()


def _factors(seed: int, out: int, inn: int, r: int) -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(seed)
    return lucid.randn(r, inn) * 0.3, lucid.randn(out, r) * 0.3


def _randn(seed: int, *shape: int) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randn(*shape) * 0.3


def _write(directory: Path, config: dict[str, object], tensors: dict[str, lucid.Tensor]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        tensors, str(directory / "adapter_model.safetensors"), metadata={"format": "pt"}
    )
    (directory / "adapter_config.json").write_text(json.dumps(config))
    return directory


def _reference_config(
    class_name: str, library: str, targets: list[str], **fields: object
) -> dict[str, object]:
    """An ``adapter_config.json`` as the reference library writes a LoRA adapter's."""
    config: dict[str, object] = {
        "alpha_pattern": {},
        "auto_mapping": {"base_model_class": class_name, "parent_library": library},
        "base_model_name_or_path": "reference/base",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "peft_version": "0.17.1",
        "r": 4,
        "rank_pattern": {},
        "revision": None,
        "target_modules": targets,
        "task_type": None,
        "use_dora": False,
        "use_rslora": False,
    }
    config.update(fields)
    return config


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool(lucid.allclose(a, b, atol=1e-5, rtol=1e-4))


def _with_weights(model: nn.Module, edits: dict[str, lucid.Tensor]) -> nn.Module:
    """``model`` with the named state-dict entries replaced."""
    state = model.state_dict()
    for key, value in edits.items():
        assert key in state, key
        state[key] = value
    model.load_state_dict(state)
    return model.eval()


# ── T5: a rename ─────────────────────────────────────────────────────────────

_T5_LIBRARY = "transformers.models.t5.modeling_t5"
#: reference module → zoo module, written by hand for the tiny two-block T5.
_T5_MODULES = {
    **{
        f"encoder.block.{i}.layer.0.SelfAttention.{p}": f"t5.encoder.blocks.{i}.self_attention.{p}"
        for i in range(2)
        for p in ("q", "v")
    },
    **{
        f"decoder.block.{i}.layer.0.SelfAttention.{p}": f"t5.decoder.blocks.{i}.self_attention.{p}"
        for i in range(2)
        for p in ("q", "v")
    },
    **{
        f"decoder.block.{i}.layer.1.EncDecAttention.{p}": f"t5.decoder.blocks.{i}.cross_attention.{p}"
        for i in range(2)
        for p in ("q", "v")
    },
}


def _t5_adapter(directory: Path, r: int = 4, alpha: int = 8) -> dict[str, tuple]:
    tensors: dict[str, lucid.Tensor] = {}
    factors = {}
    for n, module in enumerate(_T5_MODULES):
        a, b = _factors(100 + n, _DIM, _DIM, r)
        tensors[f"{_PREFIX}{module}.lora_A.weight"] = a
        tensors[f"{_PREFIX}{module}.lora_B.weight"] = b
        factors[module] = (a, b)
    config = _reference_config(
        "T5ForConditionalGeneration",
        _T5_LIBRARY,
        ["q", "v"],
        r=r,
        lora_alpha=alpha,
        task_type="SEQ_2_SEQ_LM",
    )
    _write(directory, config, tensors)
    return factors


def _t5_batch() -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(1)
    return lucid.randint(2, 64, (2, 9)), lucid.randint(2, 64, (2, 6))


def test_t5_adapter_loads_by_role_names_and_matches_the_hand_folded_model(tmp_path: Path) -> None:
    factors = _t5_adapter(tmp_path, r=4, alpha=8)
    base = _base("t5_small_seq2seq", _T5)
    scale = 8 / 4
    folded = _with_weights(
        _base("t5_small_seq2seq", _T5),
        {
            f"{zoo}.weight": base.state_dict()[f"{zoo}.weight"]
            + scale * (factors[ref][1] @ factors[ref][0])
            for ref, zoo in _T5_MODULES.items()
        },
    )
    loaded = PeftModel.from_pretrained(base, tmp_path)
    keys = set(get_peft_model_state_dict(loaded))
    assert keys == {
        f"{_PREFIX}{zoo}.{table}.weight"
        for zoo in _T5_MODULES.values()
        for table in ("lora_A", "lora_B")
    }
    src, dec = _t5_batch()
    with lucid.no_grad():
        assert _same(loaded(src, dec).logits, folded(src, dec).logits)
        assert not _same(folded(src, dec).logits, _base("t5_small_seq2seq", _T5)(src, dec).logits)
    config = loaded.peft_config["default"]
    assert sorted(config.target_modules) == sorted(_T5_MODULES.values())


def test_t5_trunk_only_adapter_has_no_prefix(tmp_path: Path) -> None:
    a, b = _factors(5, _DIM, _DIM, 4)
    tensors = {
        f"{_PREFIX}encoder.block.0.layer.0.SelfAttention.q.lora_A.weight": a,
        f"{_PREFIX}encoder.block.0.layer.0.SelfAttention.q.lora_B.weight": b,
    }
    _write(tmp_path, _reference_config("T5Model", _T5_LIBRARY, ["q"]), tensors)
    loaded = PeftModel.from_pretrained(_base("t5_small", _T5), tmp_path)
    assert set(get_peft_model_state_dict(loaded)) == {
        f"{_PREFIX}encoder.blocks.0.self_attention.q.lora_A.weight",
        f"{_PREFIX}encoder.blocks.0.self_attention.q.lora_B.weight",
    }


def test_t5_reference_names_cover_the_released_layout() -> None:
    cases = {
        "shared.weight": "shared.weight",
        "encoder.final_layer_norm.weight": "encoder.final_layer_norm.weight",
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": (
            "encoder.relative_attention_bias.weight"
        ),
        "decoder.block.3.layer.1.EncDecAttention.k.weight": (
            "decoder.blocks.3.cross_attention.k.weight"
        ),
        "decoder.block.3.layer.2.DenseReluDense.wo.weight": (
            "decoder.blocks.3.feed_forward.wo.weight"
        ),
        "decoder.block.1.layer.1.layer_norm.weight": "decoder.blocks.1.cross_attention_norm.weight",
        "encoder.block.4.layer.1.layer_norm.weight": "encoder.blocks.4.feed_forward_norm.weight",
    }
    for source, expected in cases.items():
        assert t5_reference_name(source) == expected
        assert t5_reference_name(source, trunked=True) == f"t5.{expected}"
    assert t5_reference_name("lm_head.weight", trunked=True) == "lm_head.weight"
    # a sub-layer the stack lacks, and a tied alias, have no zoo name
    assert t5_reference_name("encoder.block.0.layer.2.DenseReluDense.wi.weight") is None
    assert t5_reference_name("encoder.embed_tokens.weight") is None


def test_t5_head_and_patterns_are_renamed_with_the_modules(tmp_path: Path) -> None:
    a, b = _factors(6, _DIM, _DIM, 4)
    head = _randn(7, 64, _DIM)
    module = "encoder.block.1.layer.0.SelfAttention.q"
    tensors = {
        f"{_PREFIX}{module}.lora_A.weight": a,
        f"{_PREFIX}{module}.lora_B.weight": b,
        f"{_PREFIX}lm_head.weight": head,
    }
    config = _reference_config(
        "T5ForConditionalGeneration",
        _T5_LIBRARY,
        ["q"],
        modules_to_save=["lm_head"],
        task_type="SEQ_2_SEQ_LM",
        rank_pattern={"SelfAttention.q": 4},
        alpha_pattern={"encoder.block.1.layer.0.SelfAttention.q": 16},
        exclude_modules=["wo"],
    )
    converted = convert_adapter(
        AdapterFile(LoraConfig.from_dict(config), {k: v for k, v in tensors.items()}),
        ConversionContext(model_type="t5", reference_class="T5ForConditionalGeneration"),
        True,
    )
    zoo = "t5.encoder.blocks.1.self_attention.q"
    assert set(converted.tensors) == {
        f"{_PREFIX}{zoo}.lora_A.weight",
        f"{_PREFIX}{zoo}.lora_B.weight",
        f"{_PREFIX}lm_head.weight",
    }
    out = converted.config
    assert out.target_modules == {zoo} and out.exclude_modules is None
    assert out.modules_to_save == ["lm_head"]
    assert out.rank_pattern == {zoo: 4} and out.alpha_pattern == {zoo: 16}


# ── ViT: three projections become one ────────────────────────────────────────

_VIT_LIBRARY = "transformers.models.vit.modeling_vit"
_PROJECTIONS = ("query", "key", "value")


def _ref_vit(layer: int, projection: str) -> str:
    return f"vit.encoder.layer.{layer}.attention.attention.{projection}"


class _Part:
    """One reference projection's adapter: its factors and, optionally, a bias or a magnitude."""

    def __init__(
        self, seed: int, r: int, bias: bool = False, magnitude: lucid.Tensor | None = None
    ) -> None:
        self.a, self.b = _factors(seed, _DIM, _DIM, r)
        self.r = r
        self.bias = _randn(seed + 500, _DIM) if bias else None
        self.magnitude = magnitude


def _vit_tensors(layers: dict[tuple[int, str], _Part]) -> dict[str, lucid.Tensor]:
    out: dict[str, lucid.Tensor] = {}
    for (layer, projection), part in layers.items():
        module = f"{_PREFIX}{_ref_vit(layer, projection)}"
        out[f"{module}.lora_A.weight"] = part.a
        out[f"{module}.lora_B.weight"] = part.b
        if part.bias is not None:
            out[f"{module}.lora_B.bias"] = part.bias
        if part.magnitude is not None:
            out[f"{module}.lora_magnitude_vector.weight"] = part.magnitude
    return out


def _vit_batch() -> lucid.Tensor:
    lucid.manual_seed(2)
    return lucid.randn(3, 3, 16, 16)


def _vit_expected(
    parts: dict[tuple[int, str], _Part], scales: dict[tuple[int, str], float], dora: bool = False
) -> nn.Module:
    """The zoo ViT with every projection's update folded into its rows of ``qkv`` by hand."""
    folded = _base("vit_base_16_cls", _VIT)
    state = folded.state_dict()
    edits: dict[str, lucid.Tensor] = {}
    for (layer, projection), part in parts.items():
        row = _PROJECTIONS.index(projection)
        key = f"blocks.{layer}.attn.qkv.weight"
        weight = edits.get(key, state[key])
        rows = weight[row * _DIM : (row + 1) * _DIM]
        delta = scales[(layer, projection)] * (part.b @ part.a)
        if dora:
            assert part.magnitude is not None
            total = rows + delta
            norm = lucid.sqrt((total * total).sum(dim=1, keepdim=True))
            new = part.magnitude.reshape(-1, 1) * total / norm
        else:
            new = rows + delta
        edits[key] = lucid.cat([weight[: row * _DIM], new, weight[(row + 1) * _DIM :]], dim=0)
        if part.bias is not None:
            bias_key = f"blocks.{layer}.attn.qkv.bias"
            bias = edits.get(bias_key, state[bias_key])
            shifted = bias[row * _DIM : (row + 1) * _DIM] + scales[(layer, projection)] * part.bias
            edits[bias_key] = lucid.cat(
                [bias[: row * _DIM], shifted, bias[(row + 1) * _DIM :]], dim=0
            )
    return _with_weights(folded, edits)


def _vit_load(tmp_path: Path, parts: dict[tuple[int, str], _Part], **fields: object) -> PeftModel:
    config = _reference_config(
        "ViTForImageClassification", _VIT_LIBRARY, sorted({p for _, p in parts}), **fields
    )
    _write(tmp_path, config, _vit_tensors(parts))
    return PeftModel.from_pretrained(_base("vit_base_16_cls", _VIT), tmp_path)


def _assert_vit_matches(loaded: PeftModel, expected: nn.Module) -> None:
    images = _vit_batch()
    with lucid.no_grad():
        assert _same(loaded(images).logits, expected(images).logits)
        # the adapter is not a no-op, so the agreement above says something
        assert not _same(expected(images).logits, _base("vit_base_16_cls", _VIT)(images).logits)


def test_vit_query_and_value_fuse_into_qkv_with_zeros_for_the_key(tmp_path: Path) -> None:
    parts = {
        (i, p): _Part(10 * i + n, r=4) for i in range(2) for n, p in enumerate(("query", "value"))
    }
    loaded = _vit_load(tmp_path, parts, r=4, lora_alpha=8)
    expected = _vit_expected(parts, {k: 8 / 4 for k in parts})
    _assert_vit_matches(loaded, expected)
    state = get_peft_model_state_dict(loaded)
    a = state[f"{_PREFIX}blocks.0.attn.qkv.lora_A.weight"]
    b = state[f"{_PREFIX}blocks.0.attn.qkv.lora_B.weight"]
    assert tuple(a.shape) == (12, _DIM) and tuple(b.shape) == (3 * _DIM, 12)
    # stacked A; block-diagonal B; nothing for the key's rows and columns
    assert _same(a[:4], parts[(0, "query")].a) and _same(a[8:], parts[(0, "value")].a)
    assert float(lucid.abs(a[4:8]).sum().item()) == 0.0
    assert float(lucid.abs(b[_DIM : 2 * _DIM]).sum().item()) == 0.0
    assert float(lucid.abs(b[:_DIM, 4:]).sum().item()) == 0.0
    config = loaded.peft_config["default"]
    assert config.target_modules == {"blocks.0.attn.qkv", "blocks.1.attn.qkv"}
    assert config.rank_pattern["blocks.0.attn.qkv"] == 12
    # alpha / r stays what the file said: 2
    scale = config.alpha_pattern["blocks.0.attn.qkv"] / config.rank_pattern["blocks.0.attn.qkv"]
    assert scale == pytest.approx(2.0)


def test_vit_three_projections_with_their_own_ranks_and_alphas(tmp_path: Path) -> None:
    ranks = {"query": 4, "key": 2, "value": 3}
    alphas = {"query": 8, "key": 12, "value": 3}
    parts = {(0, p): _Part(30 + n, r=ranks[p]) for n, p in enumerate(_PROJECTIONS)}
    pattern = {_ref_vit(0, p): alphas[p] for p in _PROJECTIONS}
    loaded = _vit_load(
        tmp_path,
        parts,
        r=4,
        lora_alpha=8,
        alpha_pattern=pattern,
        rank_pattern={_ref_vit(0, "key"): 2, _ref_vit(0, "value"): 3},
    )
    scales = {(0, p): alphas[p] / ranks[p] for p in _PROJECTIONS}
    _assert_vit_matches(loaded, _vit_expected(parts, scales))
    config = loaded.peft_config["default"]
    assert config.rank_pattern["blocks.0.attn.qkv"] == 12  # 3 * max rank


def test_vit_rank_stabilised_scaling_survives_fusing(tmp_path: Path) -> None:
    parts = {(0, p): _Part(40 + n, r=4) for n, p in enumerate(("query", "value"))}
    loaded = _vit_load(tmp_path, parts, r=4, lora_alpha=6, use_rslora=True)
    scale = 6 / 4**0.5
    _assert_vit_matches(loaded, _vit_expected(parts, {k: scale for k in parts}))


def test_vit_lora_bias_is_stacked_and_scaled(tmp_path: Path) -> None:
    parts = {
        (0, "query"): _Part(50, r=4, bias=True),
        (0, "value"): _Part(51, r=4, bias=True),
        (0, "key"): _Part(52, r=4),
    }
    loaded = _vit_load(tmp_path, parts, r=4, lora_alpha=8, lora_bias=True)
    _assert_vit_matches(loaded, _vit_expected(parts, {k: 2.0 for k in parts}))


def _norms(model: nn.Module, layer: int, row: int) -> lucid.Tensor:
    weight = model.state_dict()[f"blocks.{layer}.attn.qkv.weight"][row * _DIM : (row + 1) * _DIM]
    return lucid.sqrt((weight * weight).sum(dim=1))


def test_vit_dora_magnitudes_stack_and_an_absent_projection_keeps_its_row_norm(
    tmp_path: Path,
) -> None:
    base = _base("vit_base_16_cls", _VIT)
    parts = {
        (0, p): _Part(60 + n, r=4, magnitude=_norms(base, 0, _PROJECTIONS.index(p)) * 1.1)
        for n, p in enumerate(("query", "value"))
    }
    loaded = _vit_load(tmp_path, parts, r=4, lora_alpha=8, use_dora=True)
    _assert_vit_matches(loaded, _vit_expected(parts, {k: 2.0 for k in parts}, dora=True))
    key = f"{_PREFIX}blocks.0.attn.qkv.lora_magnitude_vector.weight"
    magnitude = get_peft_model_state_dict(loaded)[key]
    assert _same(magnitude[_DIM : 2 * _DIM], _norms(base, 0, 1))


def test_vit_dora_on_files_alone_needs_the_model_for_an_absent_projection(tmp_path: Path) -> None:
    parts = {(0, "query"): _Part(70, r=4, magnitude=lucid.ones(_DIM))}
    config = _reference_config("ViTForImageClassification", _VIT_LIBRARY, ["query"], use_dora=True)
    source = _write(tmp_path / "src", config, _vit_tensors(parts))
    with pytest.raises(NotImplementedError, match="weight norm"):
        convert_adapters.convert_directory(source, tmp_path / "dst", model_type="vit")
    assert convert_adapters.main([str(source), str(tmp_path / "dst")]) == 2
    assert not (tmp_path / "dst" / "adapter_model.safetensors").exists()


def test_vit_dora_with_all_three_projections_converts_on_files_alone(tmp_path: Path) -> None:
    parts = {
        (0, p): _Part(80 + n, r=2, magnitude=lucid.ones(_DIM)) for n, p in enumerate(_PROJECTIONS)
    }
    config = _reference_config(
        "ViTForImageClassification", _VIT_LIBRARY, list(_PROJECTIONS), use_dora=True, r=2
    )
    source = _write(tmp_path / "src", config, _vit_tensors(parts))
    converted = convert_adapters.convert_directory(source, tmp_path / "dst", model_type="vit")
    magnitude = converted.tensors[f"{_PREFIX}blocks.0.attn.qkv.lora_magnitude_vector.weight"]
    assert tuple(magnitude.shape) == (3 * _DIM,)


def test_vit_other_modules_and_the_head_are_renamed(tmp_path: Path) -> None:
    a1, b1 = _factors(90, 128, _DIM, 4)
    a2, b2 = _factors(91, _DIM, _DIM, 4)
    head_w, head_b = _randn(92, 5, _DIM), _randn(93, 5)
    tensors = {
        f"{_PREFIX}vit.encoder.layer.0.intermediate.dense.lora_A.weight": a1,
        f"{_PREFIX}vit.encoder.layer.0.intermediate.dense.lora_B.weight": b1,
        f"{_PREFIX}vit.encoder.layer.1.attention.output.dense.lora_A.weight": a2,
        f"{_PREFIX}vit.encoder.layer.1.attention.output.dense.lora_B.weight": b2,
        f"{_PREFIX}classifier.weight": head_w,
        f"{_PREFIX}classifier.bias": head_b,
    }
    config = _reference_config(
        "ViTForImageClassification", _VIT_LIBRARY, ["dense"], modules_to_save=["classifier"]
    )
    _write(tmp_path, config, tensors)
    loaded = PeftModel.from_pretrained(_base("vit_base_16_cls", _VIT), tmp_path)
    state = get_peft_model_state_dict(loaded)
    assert set(state) == {
        f"{_PREFIX}blocks.0.mlp.fc1.lora_A.weight",
        f"{_PREFIX}blocks.0.mlp.fc1.lora_B.weight",
        f"{_PREFIX}blocks.1.attn.proj.lora_A.weight",
        f"{_PREFIX}blocks.1.attn.proj.lora_B.weight",
        f"{_PREFIX}head.weight",
        f"{_PREFIX}head.bias",
    }
    assert _same(state[f"{_PREFIX}head.weight"], head_w)
    base = _base("vit_base_16_cls", _VIT)
    expected = _with_weights(
        base,
        {
            "blocks.0.mlp.fc1.weight": base.state_dict()["blocks.0.mlp.fc1.weight"]
            + 2.0 * (b1 @ a1),
            "blocks.1.attn.proj.weight": base.state_dict()["blocks.1.attn.proj.weight"]
            + 2.0 * (b2 @ a2),
            "head.weight": head_w,
            "head.bias": head_b,
        },
    )
    _assert_vit_matches(loaded, expected)


def test_other_methods_name_themselves_when_asked_to_fuse() -> None:
    config = IA3Config.from_dict(
        {"peft_type": "IA3", "target_modules": ["query", "value"], "feedforward_modules": []}
    )
    tensors = {f"{_PREFIX}{_ref_vit(0, 'query')}.ia3_l": lucid.ones(_DIM, 1)}
    with pytest.raises(NotImplementedError, match="IA3"):
        ViTConverter(ConversionContext(model_type="vit")).convert(AdapterFile(config, tensors))


def test_other_methods_still_get_their_modules_renamed() -> None:
    config = IA3Config.from_dict(
        {
            "peft_type": "IA3",
            "target_modules": ["dense", "vit.encoder.layer.0.output.dense"],
            "feedforward_modules": ["vit.encoder.layer.0.output.dense"],
        }
    )
    tensors = {
        f"{_PREFIX}vit.encoder.layer.0.output.dense.ia3_l": lucid.ones(1, 128),
        f"{_PREFIX}vit.encoder.layer.0.attention.output.dense.ia3_l": lucid.ones(_DIM, 1),
    }
    converted = convert_adapter(
        AdapterFile(config, tensors), ConversionContext(model_type="vit"), True
    )
    assert set(converted.tensors) == {
        f"{_PREFIX}blocks.0.mlp.fc2.ia3_l",
        f"{_PREFIX}blocks.0.attn.proj.ia3_l",
    }
    assert sorted(converted.config.target_modules) == ["blocks.0.attn.proj", "blocks.0.mlp.fc2"]
    assert converted.config.feedforward_modules == {"blocks.0.mlp.fc2"}


# ── RoFormer: the encoder is shared, the head differs ────────────────────────

_ROFORMER_LIBRARY = "transformers.models.roformer.modeling_roformer"


def _roformer_encoder(seed: int) -> tuple[dict[str, lucid.Tensor], dict[str, tuple]]:
    tensors: dict[str, lucid.Tensor] = {}
    factors = {}
    for i in range(2):
        for n, p in enumerate(("query", "value")):
            module = f"roformer.encoder.layer.{i}.attention.self.{p}"
            a, b = _factors(seed + 10 * i + n, _DIM, _DIM, 4)
            tensors[f"{_PREFIX}{module}.lora_A.weight"] = a
            tensors[f"{_PREFIX}{module}.lora_B.weight"] = b
            factors[module] = (a, b)
    return tensors, factors


def _roformer_config(**fields: object) -> dict[str, object]:
    return _reference_config(
        "RoFormerForSequenceClassification",
        _ROFORMER_LIBRARY,
        ["query", "value"],
        task_type="SEQ_CLS",
        **fields,
    )


def _roformer_batch() -> lucid.Tensor:
    lucid.manual_seed(3)
    return lucid.randint(0, 64, (3, 10))


def test_roformer_encoder_adapter_is_read_as_it_is(tmp_path: Path) -> None:
    tensors, factors = _roformer_encoder(110)
    _write(tmp_path, _roformer_config(), tensors)
    base = _base("roformer_cls", _TEXT)
    expected = _with_weights(
        _base("roformer_cls", _TEXT),
        {
            f"{module}.weight": base.state_dict()[f"{module}.weight"] + 2.0 * (b @ a)
            for module, (a, b) in factors.items()
        },
    )
    with pytest.warns(UserWarning, match="missing adapter keys"):
        loaded = PeftModel.from_pretrained(base, tmp_path)
    ids = _roformer_batch()
    with lucid.no_grad():
        assert _same(loaded(ids).logits, expected(ids).logits)


def test_roformer_output_projection_is_the_zoo_classifier(tmp_path: Path) -> None:
    tensors, factors = _roformer_encoder(120)
    out_w, out_b = _randn(121, 2, _DIM), _randn(122, 2)
    tensors[f"{_PREFIX}classifier.out_proj.weight"] = out_w
    tensors[f"{_PREFIX}classifier.out_proj.bias"] = out_b
    converted = convert_adapter(
        AdapterFile(
            LoraConfig.from_dict(_roformer_config(modules_to_save=["classifier"])), tensors
        ),
        ConversionContext(model_type="roformer"),
        True,
    )
    assert _same(converted.tensors[f"{_PREFIX}classifier.weight"], out_w)
    assert f"{_PREFIX}classifier.out_proj.weight" not in converted.tensors


def test_roformer_trained_dense_head_is_refused_until_the_head_is_nested(tmp_path: Path) -> None:
    tensors, _ = _roformer_encoder(130)
    tensors[f"{_PREFIX}classifier.dense.weight"] = _randn(131, _DIM, _DIM)
    tensors[f"{_PREFIX}classifier.dense.bias"] = _randn(132, _DIM)
    tensors[f"{_PREFIX}classifier.out_proj.weight"] = _randn(133, 2, _DIM)
    _write(tmp_path, _roformer_config(modules_to_save=["classifier"]), tensors)
    base = _base("roformer_cls", _TEXT)
    before = {k: v.clone() for k, v in base.state_dict().items()}
    with pytest.raises(NotImplementedError, match="LCD-709"):
        PeftModel.from_pretrained(base, tmp_path)
    assert set(base.state_dict()) == set(before)  # nothing was injected


@pytest.mark.xfail(
    strict=True, reason="LCD-709: roformer_cls keeps its head's dense outside `classifier`"
)
def test_roformer_sequence_classification_adapter_with_its_head_loads(tmp_path: Path) -> None:
    tensors, _ = _roformer_encoder(140)
    tensors[f"{_PREFIX}classifier.dense.weight"] = _randn(141, _DIM, _DIM)
    tensors[f"{_PREFIX}classifier.dense.bias"] = _randn(142, _DIM)
    tensors[f"{_PREFIX}classifier.out_proj.weight"] = _randn(143, 2, _DIM)
    tensors[f"{_PREFIX}classifier.out_proj.bias"] = _randn(144, 2)
    _write(tmp_path, _roformer_config(modules_to_save=["classifier"]), tensors)
    PeftModel.from_pretrained(_base("roformer_cls", _TEXT), tmp_path)


def test_roformer_nested_head_converts_by_identity() -> None:
    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = nn.Linear(_DIM, _DIM)
            self.out_proj = nn.Linear(_DIM, 2)

    class Nested(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = Head()

    tensors, _ = _roformer_encoder(150)
    tensors[f"{_PREFIX}classifier.dense.weight"] = _randn(151, _DIM, _DIM)
    tensors[f"{_PREFIX}classifier.out_proj.weight"] = _randn(152, 2, _DIM)
    adapter = AdapterFile(
        LoraConfig.from_dict(_roformer_config(modules_to_save=["classifier"])), tensors
    )
    context = ConversionContext(model_type="roformer", model=Nested())
    assert convert_adapter(adapter, context, "auto") is adapter
    assert set(convert_adapter(adapter, context, True).tensors) == set(tensors)


# ── GPT-2: score becomes classifier ──────────────────────────────────────────

_GPT2_LIBRARY = "transformers.models.gpt2.modeling_gpt2"


def test_gpt2_sequence_classification_score_is_the_zoo_classifier(tmp_path: Path) -> None:
    tensors: dict[str, lucid.Tensor] = {}
    factors = {}
    for i in range(2):
        module = f"transformer.h.{i}.attn.c_attn"
        a, b = _factors(160 + i, 3 * _DIM, _DIM, 4)
        tensors[f"{_PREFIX}{module}.lora_A.weight"] = a
        tensors[f"{_PREFIX}{module}.lora_B.weight"] = b
        factors[module] = (a, b)
    score = _randn(165, 2, _DIM)
    tensors[f"{_PREFIX}score.weight"] = score
    config = _reference_config(
        "GPT2ForSequenceClassification",
        _GPT2_LIBRARY,
        ["c_attn"],
        modules_to_save=["score"],
        task_type="SEQ_CLS",
    )
    _write(tmp_path, config, tensors)
    base = _base("gpt2_small_cls", _TEXT)
    expected = _with_weights(
        _base("gpt2_small_cls", _TEXT),
        {
            **{
                f"{module}.weight": base.state_dict()[f"{module}.weight"] + 2.0 * (b @ a)
                for module, (a, b) in factors.items()
            },
            "classifier.weight": score,
        },
    )
    loaded = PeftModel.from_pretrained(base, tmp_path)
    assert _same(get_peft_model_state_dict(loaded)[f"{_PREFIX}classifier.weight"], score)
    assert loaded.peft_config["default"].modules_to_save == ["classifier"]
    lucid.manual_seed(4)
    ids = lucid.randint(0, 64, (3, 10))
    with lucid.no_grad():
        assert _same(loaded(ids).logits, expected(ids).logits)


def test_gpt2_adapter_without_a_head_is_not_converted(tmp_path: Path) -> None:
    a, b = _factors(170, 3 * _DIM, _DIM, 4)
    tensors = {
        f"{_PREFIX}transformer.h.0.attn.c_attn.lora_A.weight": a,
        f"{_PREFIX}transformer.h.0.attn.c_attn.lora_B.weight": b,
    }
    adapter = AdapterFile(
        LoraConfig.from_dict(_reference_config("GPT2LMHeadModel", _GPT2_LIBRARY, ["c_attn"])),
        tensors,
    )
    context = ConversionContext(model_type="gpt2")
    assert convert_adapter(adapter, context, "auto") is adapter
    assert convert_adapter(adapter, context, True).config.target_modules == {"c_attn"}


# ── the entry points ─────────────────────────────────────────────────────────


def test_cli_round_trip_loads_like_the_reference_file(tmp_path: Path) -> None:
    _t5_adapter(tmp_path / "ref")
    status = convert_adapters.main([str(tmp_path / "ref"), str(tmp_path / "out")])
    assert status == 0
    written = load_safetensors(str(tmp_path / "out" / "adapter_model.safetensors"))
    assert all(key.startswith(f"{_PREFIX}t5.") for key in written)
    config = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert "t5.encoder.blocks.0.self_attention.q" in config["target_modules"]
    src, dec = _t5_batch()
    direct = PeftModel.from_pretrained(_base("t5_small_seq2seq", _T5), tmp_path / "ref")
    plain = PeftModel.from_pretrained(
        _base("t5_small_seq2seq", _T5), tmp_path / "out", convert=False
    )
    with lucid.no_grad():
        assert _same(direct(src, dec).logits, plain(src, dec).logits)


def test_cli_picks_the_converter_from_the_reference_class_or_the_flag(tmp_path: Path) -> None:
    tensors = {
        f"{_PREFIX}encoder.block.0.layer.0.SelfAttention.q.lora_A.weight": lucid.ones(4, _DIM),
        f"{_PREFIX}encoder.block.0.layer.0.SelfAttention.q.lora_B.weight": lucid.ones(_DIM, 4),
    }
    config = _reference_config("T5Model", _T5_LIBRARY, ["q"])
    del config["auto_mapping"]
    source = _write(tmp_path / "src", config, tensors)
    assert convert_adapters.main([str(source), str(tmp_path / "a")]) == 2  # nothing says which
    assert convert_adapters.main([str(source), str(tmp_path / "b"), "--model-type", "t5"]) == 2
    flags = ["--reference-class", "T5Model"]
    assert convert_adapters.main([str(source), str(tmp_path / "c"), *flags]) == 0
    written = load_safetensors(str(tmp_path / "c" / "adapter_model.safetensors"))
    assert f"{_PREFIX}encoder.blocks.0.self_attention.q.lora_A.weight" in written


def test_cli_refuses_to_convert_in_place(tmp_path: Path) -> None:
    _t5_adapter(tmp_path)
    assert convert_adapters.main([str(tmp_path), str(tmp_path)]) == 2


def test_auto_leaves_an_adapter_the_zoo_wrote_alone(tmp_path: Path) -> None:
    _t5_adapter(tmp_path / "ref")
    first = PeftModel.from_pretrained(_base("t5_small_seq2seq", _T5), tmp_path / "ref")
    first.save_pretrained(tmp_path / "zoo")
    again = PeftModel.from_pretrained(_base("t5_small_seq2seq", _T5), tmp_path / "zoo")
    assert set(get_peft_model_state_dict(again)) == set(get_peft_model_state_dict(first))
    with pytest.raises(UnmappedKey, match="not in the reference layout"):
        PeftModel.from_pretrained(_base("t5_small_seq2seq", _T5), tmp_path / "zoo", convert=True)


def test_convert_false_reads_the_reference_file_as_it_is(tmp_path: Path) -> None:
    _t5_adapter(tmp_path)
    with pytest.raises(ValueError, match="do not fit this model"):
        PeftModel.from_pretrained(_base("t5_small_seq2seq", _T5), tmp_path, convert=False)


def test_convert_true_refuses_a_model_type_without_a_converter(tmp_path: Path) -> None:
    a, b = _factors(180, 4, 4, 2)
    tensors = {f"{_PREFIX}0.lora_A.weight": a[:, :4], f"{_PREFIX}0.lora_B.weight": b}
    config = _reference_config("Sequential", "x.y", ["0"], r=2)
    _write(tmp_path, config, tensors)
    with pytest.raises(ValueError, match="no adapter converter"):
        PeftModel.from_pretrained(nn.Sequential(nn.Linear(4, 4)), tmp_path, convert=True)
    loaded = PeftModel.from_pretrained(nn.Sequential(nn.Linear(4, 4)), tmp_path)  # "auto": as is
    assert sorted(get_peft_model_state_dict(loaded)) == sorted(tensors)


def test_convert_must_be_a_flag_or_auto(tmp_path: Path) -> None:
    _t5_adapter(tmp_path)
    with pytest.raises(ValueError, match="True, False or 'auto'"):
        PeftModel.from_pretrained(
            _base("t5_small_seq2seq", _T5), tmp_path, convert="yes"  # type: ignore[arg-type]  # the refusal under test
        )


def test_load_adapter_converts_too(tmp_path: Path) -> None:
    _t5_adapter(tmp_path / "ref")
    first = PeftModel.from_pretrained(_base("t5_small_seq2seq", _T5), tmp_path / "ref")
    other = PeftModel.from_pretrained(_base("t5_small_seq2seq", _T5), tmp_path / "ref")
    other.load_adapter(tmp_path / "ref", "second")
    assert set(other.peft_config) == {"default", "second"}
    assert set(get_peft_model_state_dict(other, adapter_name="second")) == set(
        get_peft_model_state_dict(first)
    )
    other.load_adapter(tmp_path / "ref", "second")  # an adapter it holds: weights replaced


def test_the_registry_serves_each_model_type() -> None:
    assert AdapterConverter.registered() == ["gpt2", "roformer", "t5", "vit"]
    assert AdapterConverter.lookup(reference_class="ViTModel") is ViTConverter
    assert AdapterConverter.lookup(reference_class="NoSuchModel") is None
    assert AdapterConverter.lookup(model_type="bert") is None


def test_split_key_ends_the_module_before_the_first_adapter_table() -> None:
    assert split_key("a.b.lora_A.weight") == ("a.b", ".lora_A.weight")
    assert split_key("a.b.lora_B.bias") == ("a.b", ".lora_B.bias")
    assert split_key("a.ia3_l") == ("a", ".ia3_l")
    assert split_key("classifier.weight") == ("classifier", ".weight")
    assert split_key("lora_A.weight") == ("lora_A", ".weight")  # a module's own name is not a table


# ── refusals and edges ───────────────────────────────────────────────────────


def test_t5_needs_to_know_whether_the_trunk_is_nested() -> None:
    adapter = AdapterFile(LoraConfig.from_dict(_reference_config("X", "y", ["q"])), {})
    with pytest.raises(ValueError, match="nests its trunk"):
        convert_adapter(adapter, ConversionContext(model_type="t5"), True)


def test_two_reference_modules_cannot_land_on_one_zoo_module() -> None:
    config = LoraConfig.from_dict(
        _reference_config("GPT2ForSequenceClassification", "y", ["c_attn"])
    )
    tensors = {
        f"{_PREFIX}score.weight": lucid.ones(2, 3),
        f"{_PREFIX}classifier.weight": lucid.ones(2, 3),
    }
    with pytest.raises(ValueError, match="two adapter tensors"):
        convert_adapter(AdapterFile(config, tensors), ConversionContext(model_type="gpt2"), True)


def test_a_pattern_that_gives_one_module_two_values_is_refused() -> None:
    module = "encoder.block.0.layer.0.SelfAttention.q"
    config = LoraConfig.from_dict(
        _reference_config("T5Model", "y", ["q"], rank_pattern={"q": 4, module: 8})
    )
    tensors = {
        f"{_PREFIX}{module}.lora_A.weight": lucid.ones(4, 3),
        f"{_PREFIX}{module}.lora_B.weight": lucid.ones(3, 4),
    }
    context = ConversionContext(model_type="t5", reference_class="T5Model")
    with pytest.raises(ValueError, match="two different values"):
        convert_adapter(AdapterFile(config, tensors), context, True)


def _vit_file(tensors: dict[str, lucid.Tensor], **fields: object) -> AdapterFile:
    config = _reference_config("ViTModel", _VIT_LIBRARY, list(_PROJECTIONS), **fields)
    return AdapterFile(LoraConfig.from_dict(config), tensors)


def test_vit_projection_without_its_factor_pair_is_refused() -> None:
    module = f"{_PREFIX}{_ref_vit(0, 'query')}"
    adapter = _vit_file({f"{module}.lora_A.weight": lucid.ones(4, _DIM)})
    with pytest.raises(ValueError, match="no lora_A / lora_B pair"):
        ViTConverter(ConversionContext(model_type="vit")).convert(adapter)


def test_vit_projections_of_different_widths_are_refused() -> None:
    q, v = f"{_PREFIX}{_ref_vit(0, 'query')}", f"{_PREFIX}{_ref_vit(0, 'value')}"
    adapter = _vit_file(
        {
            f"{q}.lora_A.weight": lucid.ones(4, _DIM),
            f"{q}.lora_B.weight": lucid.ones(_DIM, 4),
            f"{v}.lora_A.weight": lucid.ones(4, _DIM),
            f"{v}.lora_B.weight": lucid.ones(_DIM + 1, 4),
        }
    )
    with pytest.raises(ValueError, match="does not match its sibling"):
        ViTConverter(ConversionContext(model_type="vit")).convert(adapter)


def test_vit_projection_bias_of_the_base_cannot_be_fused() -> None:
    module = f"{_PREFIX}{_ref_vit(0, 'query')}"
    adapter = _vit_file({f"{module}.bias": lucid.ones(_DIM)}, bias="all")
    with pytest.raises(NotImplementedError, match="can be fused"):
        ViTConverter(ConversionContext(model_type="vit")).convert(adapter)


def test_vit_projections_with_unequal_scales_fold_the_ratio_into_the_bias_too(
    tmp_path: Path,
) -> None:
    ranks = {"query": 4, "key": 4, "value": 4}
    alphas = {"query": 8, "key": 16, "value": 4}
    parts = {(0, p): _Part(200 + n, r=4, bias=True) for n, p in enumerate(_PROJECTIONS)}
    loaded = _vit_load(
        tmp_path,
        parts,
        r=4,
        lora_alpha=8,
        lora_bias=True,
        alpha_pattern={_ref_vit(0, p): alphas[p] for p in _PROJECTIONS},
    )
    scales = {(0, p): alphas[p] / ranks[p] for p in _PROJECTIONS}
    _assert_vit_matches(loaded, _vit_expected(parts, scales))


def test_vit_dora_into_a_model_that_already_wears_an_adapter_reads_the_base_layer(
    tmp_path: Path,
) -> None:
    first = {(0, p): _Part(210 + n, r=4) for n, p in enumerate(("query", "value"))}
    loaded = _vit_load(tmp_path / "first", first, r=4, lora_alpha=8)
    base = _base("vit_base_16_cls", _VIT)
    second = {(0, "query"): _Part(220, r=2, magnitude=_norms(base, 0, 0))}
    config = _reference_config(
        "ViTForImageClassification", _VIT_LIBRARY, ["query"], r=2, use_dora=True
    )
    _write(tmp_path / "second", config, _vit_tensors(second))
    loaded.load_adapter(tmp_path / "second", "dora")
    key = f"{_PREFIX}blocks.0.attn.qkv.lora_magnitude_vector.weight"
    magnitude = get_peft_model_state_dict(loaded, adapter_name="dora")[key]
    assert _same(magnitude[_DIM : 2 * _DIM], _norms(base, 0, 1))
    assert _same(magnitude[2 * _DIM :], _norms(base, 0, 2))


def test_vit_dora_needs_the_fused_layer_to_be_a_linear() -> None:
    class Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Identity()

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = Attn()

    class Odd(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([Block()])

    module = f"{_PREFIX}{_ref_vit(0, 'query')}"
    adapter = _vit_file(
        {
            f"{module}.lora_A.weight": lucid.ones(2, _DIM),
            f"{module}.lora_B.weight": lucid.ones(_DIM, 2),
            f"{module}.lora_magnitude_vector.weight": lucid.ones(_DIM),
        },
        use_dora=True,
    )
    with pytest.raises(TypeError, match="not the zoo's fused nn.Linear"):
        ViTConverter(ConversionContext(model_type="vit", model=Odd())).convert(adapter)


def test_the_weight_tool_and_the_adapter_converter_share_one_t5_map() -> None:
    from tools.convert_weights.t5 import t5_key

    assert t5_key("decoder.block.2.layer.1.EncDecAttention.v.weight") == (
        "decoder.blocks.2.cross_attention.v.weight"
    )
    assert t5_key("encoder.embed_tokens.weight") is None
    with pytest.raises(KeyError):
        t5_key("encoder.block.0.layer.5.SelfAttention.q.weight")
