"""LoRA on the model zoo, end to end: gpt2, gpt, bert, roformer, t5, vit, pvt, swin and transformer.

Every model is a real registered factory with a small config override (H11),
adapted through the public ``lucid.peft`` surface the way a user does it, on
cpu and metal.  What is proved, per family:

* **targets** — the names a config must use are the zoo's own (recorded in
  the ``_FAMILIES`` table; ``"all-linear"`` resolves to the listed set and
  leaves the output head and any tied weight alone);
* **wiring** — each adapted layer is on the forward path: silencing its
  update alone changes the model's output;
* **training** — three optimizer steps lower the loss, move only the adapter
  (and the ``modules_to_save`` copy), and leave every base weight
  bit-identical;
* **inference** — ``disable_adapter()`` returns the pre-injection outputs,
  and ``generate`` through the wrapper returns the pre-injection tokens
  with the adapter off;
* **deployment** — ``merge_and_unload`` gives the adapted outputs from a
  plain zoo model whose ``save_pretrained`` / ``from_pretrained`` round trip
  keeps them, and an adapter saved with ``save_pretrained`` reloads onto a
  fresh base through ``PeftModel.from_pretrained``;
* **tasks** — ``get_peft_model`` returns the wrapper of the family's task
  (``PeftModelFor...``), computing the base model's outputs at initialisation,
  and ``AutoPeftModel*`` rebuilds base and adapter from the adapter's directory;
* **defaults** — a config that names no ``target_modules`` takes the default
  for the family's ``model_type`` (``DEFAULT_LORA_TARGET_MODULES``), which is
  exactly the modules the table above lists for the family.

Below the deep rows sits the coverage matrix (``_peft_rows.ROWS``): one tiny
real factory for every ``model_type`` the zoo registers, each adapted with the
type's default targets, trained one backward, saved and reloaded, and merged.
A type LoRA cannot adapt yet is a strict ``xfail`` naming the missing piece,
so the matrix is the progress meter for how much of the zoo ``lucid.peft``
reaches; and the default of every type is checked against every factory of it,
built without allocating.

Two rows reach the layer types beyond ``nn.Linear``: vit's patch embedding is
an ``nn.Conv2d`` (the convolution LoRA), and the transformer family's
``self_attn`` and token table are an ``nn.MultiheadAttention`` and an
``nn.Embedding``.  The families whose attention is ``nn.MultiheadAttention``
(transformer, clip, dit, detr, stable_diffusion, and mask2former's decoder
cross-attention) hold their projections as raw parameters; the transformer
family stands for them.  Those rows adapt by explicit name only, since
``"all-linear"`` selects ``nn.Linear`` alone.
"""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, TypedDict

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.models import create_model
from lucid.peft import (
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForSequenceClassification,
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.models._registry import _REGISTRY
from lucid.peft._target_defaults import (
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_LORA_TARGET_PARAMETERS,
)
from lucid.peft.tuners import lora
from lucid.serialization import load_safetensors, save_safetensors
from lucid.test.unit.models._peft_matrix import (
    Trial,
    ZooRow,
    default_problems,
    owned_by_adapter as _owned_by_adapter,
    zoo_model_types,
)
from lucid.test.unit.models._peft_rows import ROWS

_DEPTH = 2
_STEPS = 3
_LR = 3e-2


class _Overrides(TypedDict, total=False):
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_decoder_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    head_dim: int
    hidden_dropout: float
    attention_dropout: float
    image_size: int
    patch_size: int
    dim: int
    depth: int
    num_heads: int | tuple[int, ...]
    num_classes: int
    dropout: float
    embed_dim: int
    embed_dims: tuple[int, ...]
    depths: tuple[int, ...]
    mlp_ratios: tuple[float, ...]
    window_size: int


_TEXT = _Overrides(
    vocab_size=64,
    hidden_size=32,
    num_hidden_layers=_DEPTH,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=32,
    hidden_dropout=0.0,
    attention_dropout=0.0,
)


_VIT = _Overrides(
    image_size=16,
    patch_size=8,
    dim=32,
    depth=_DEPTH,
    num_heads=2,
    num_classes=5,
    dropout=0.0,
    attention_dropout=0.0,
)


class _Batch(NamedTuple):
    inputs: tuple[lucid.Tensor, ...]
    labels: lucid.Tensor


def _tokens(device: str, low: int, shape: tuple[int, ...], seed: int) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randint(low, 64, shape, device=device)


def _lm_batch(device: str) -> _Batch:
    ids = _tokens(device, 0, (2, 12), seed=1)
    return _Batch((ids,), ids)


def _cls_batch(device: str) -> _Batch:
    return _Batch((_tokens(device, 0, (3, 10), seed=1),), _tokens(device, 0, (3,), seed=2) % 2)


def _seq2seq_batch(device: str) -> _Batch:
    source = _tokens(device, 2, (2, 9), seed=1)
    target = _tokens(device, 2, (2, 6), seed=2)
    start = lucid.zeros((2, 1), dtype=lucid.int64, device=device)
    return _Batch((source, lucid.cat([start, target[:, :-1]], dim=1)), target)


def _images(size: int) -> Callable[[str], _Batch]:
    def batch(device: str) -> _Batch:
        lucid.manual_seed(1)
        images = lucid.randn(3, 3, size, size, device=device)
        return _Batch((images,), _tokens(device, 0, (3,), seed=2) % 5)

    return batch


_image_batch = _images(16)


def _names(*patterns: str) -> frozenset[str]:
    """Every module name the ``{i}`` patterns spell out, for each layer."""
    return frozenset(pattern.format(i=i) for pattern in patterns for i in range(_DEPTH))


class _Family(NamedTuple):
    name: str
    factory: str
    overrides: _Overrides
    batch: Callable[[str], _Batch]
    task_type: str
    #: What a user writes in ``target_modules`` — the reference library's
    #: conventional names where the zoo's modules carry them.
    targets: list[str]
    #: Modules the user asks ``modules_to_save`` for, beyond the task's head.
    saved: list[str]
    #: Every module name ``targets`` selects.
    adapted: frozenset[str]
    #: Every module name ``"all-linear"`` stands for.
    all_linear: frozenset[str]
    #: Modules trained in full: the task head plus ``saved``.
    trained_in_full: frozenset[str]
    #: Adapted modules the model's forward never calls.
    off_path: frozenset[str] = frozenset()
    #: The ways of choosing the modules that the family has.
    modes: tuple[str, ...] = ("explicit", "all-linear")
    #: The module type of the adapted names ending in each suffix; any other is an ``nn.Linear``.
    kinds: tuple[tuple[str, type[nn.Module]], ...] = ()

    def kind_of(self, name: str) -> type[nn.Module]:
        return next((kind for suffix, kind in self.kinds if name.endswith(suffix)), nn.Linear)

    @property
    def deepest_mode(self) -> str:
        """The mode that adapts the most: what the end-to-end tests run."""
        return self.modes[-1]


_BERT = "bert.encoder.layer.{i}"
_ROFORMER = "roformer.encoder.layer.{i}"
_T5_ENC = "t5.encoder.blocks.{i}"
_T5_DEC = "t5.decoder.blocks.{i}"


def _encoder_linears(layer: str) -> frozenset[str]:
    """The BERT-style layer's Linears beyond ``query`` / ``value``."""
    return _names(
        f"{layer}.attention.self.key",
        f"{layer}.attention.output.dense",
        f"{layer}.intermediate.dense",
        f"{layer}.output.dense",
    )


_GPT = _names("transformer.h.{i}.attn.c_attn")
_GPT_MORE = _names(
    "transformer.h.{i}.attn.c_proj", "transformer.h.{i}.mlp.c_fc", "transformer.h.{i}.mlp.c_proj"
)
_BERT_QV = _names(f"{_BERT}.attention.self.query", f"{_BERT}.attention.self.value")
_ROFORMER_QV = _names(f"{_ROFORMER}.attention.self.query", f"{_ROFORMER}.attention.self.value")
_T5_QV = _names(
    f"{_T5_ENC}.self_attention.q",
    f"{_T5_ENC}.self_attention.v",
    f"{_T5_DEC}.self_attention.q",
    f"{_T5_DEC}.self_attention.v",
    f"{_T5_DEC}.cross_attention.q",
    f"{_T5_DEC}.cross_attention.v",
)
_T5_MORE = _names(
    f"{_T5_ENC}.self_attention.k",
    f"{_T5_ENC}.self_attention.o",
    f"{_T5_ENC}.feed_forward.wi",
    f"{_T5_ENC}.feed_forward.wo",
    f"{_T5_DEC}.self_attention.k",
    f"{_T5_DEC}.self_attention.o",
    f"{_T5_DEC}.cross_attention.k",
    f"{_T5_DEC}.cross_attention.o",
    f"{_T5_DEC}.feed_forward.wi",
    f"{_T5_DEC}.feed_forward.wo",
)

_FAMILIES = [
    _Family(
        name="gpt2",
        factory="gpt2_small_lm",
        overrides=_TEXT,
        batch=_lm_batch,
        task_type="CAUSAL_LM",
        targets=["c_attn"],
        saved=[],
        adapted=_GPT,
        all_linear=_GPT | _GPT_MORE,
        trained_in_full=frozenset(),
    ),
    _Family(
        name="gpt",
        factory="gpt_lm",
        overrides=_TEXT,
        batch=_lm_batch,
        task_type="CAUSAL_LM",
        targets=["c_attn"],
        saved=[],
        adapted=_GPT,
        all_linear=_GPT | _GPT_MORE,
        trained_in_full=frozenset(),
    ),
    _Family(
        name="bert",
        factory="bert_base_cls",
        overrides=_TEXT,
        batch=_cls_batch,
        task_type="SEQ_CLS",
        targets=["query", "value"],
        saved=[],
        adapted=_BERT_QV,
        all_linear=_BERT_QV | _encoder_linears(_BERT) | {"bert.pooler.dense"},
        trained_in_full=frozenset({"classifier"}),
    ),
    # The zoo's RoFormer head is ``dense`` -> activation -> ``classifier``, and
    # ``get_peft_model`` adds ``classifier`` only.  ``dense`` cannot be asked
    # for through ``modules_to_save``: a name matches by suffix, so "dense"
    # would wrap every BERT-style ``....dense`` of the encoder too.  The
    # classifier reads the raw CLS state, so ``pooler.dense`` is never called.
    _Family(
        name="roformer",
        factory="roformer_cls",
        overrides=_TEXT,
        batch=_cls_batch,
        task_type="SEQ_CLS",
        targets=["query", "value"],
        saved=[],
        adapted=_ROFORMER_QV,
        all_linear=_ROFORMER_QV | _encoder_linears(_ROFORMER) | {"roformer.pooler.dense", "dense"},
        trained_in_full=frozenset({"classifier"}),
        off_path=frozenset({"roformer.pooler.dense"}),
    ),
    _Family(
        name="t5",
        factory="t5_small_seq2seq",
        overrides=_Overrides(**_TEXT, num_decoder_layers=_DEPTH, head_dim=16),
        batch=_seq2seq_batch,
        task_type="SEQ_2_SEQ_LM",
        targets=["q", "v"],
        saved=[],
        adapted=_T5_QV,
        all_linear=_T5_QV | _T5_MORE,
        trained_in_full=frozenset(),
    ),
    # The reference's ViT has ``query`` / ``value``; the zoo's fuses them into ``qkv``, so
    # one adapter covers all three projections.  Image classification has no
    # task type, and the head is named ``head``.
    _Family(
        name="vit",
        factory="vit_base_16_cls",
        overrides=_VIT,
        batch=_image_batch,
        task_type="",
        targets=["qkv"],
        saved=["head"],
        adapted=_names("blocks.{i}.attn.qkv"),
        all_linear=_names(
            "blocks.{i}.attn.qkv",
            "blocks.{i}.attn.proj",
            "blocks.{i}.mlp.fc1",
            "blocks.{i}.mlp.fc2",
        ),
        trained_in_full=frozenset({"head"}),
    ),
    # The patch embedding is the family's one convolution: the name is given in full, since
    # ``proj`` alone also ends the attention's output ``nn.Linear``.
    _Family(
        name="vit-patch",
        factory="vit_base_16_cls",
        overrides=_VIT,
        batch=_image_batch,
        task_type="",
        targets=["patch_embed.proj"],
        saved=["head"],
        adapted=frozenset({"patch_embed.proj"}),
        all_linear=frozenset(),
        trained_in_full=frozenset({"head"}),
        modes=("explicit",),
        kinds=(("patch_embed.proj", nn.Conv2d),),
    ),
    # The family that stands for the ones built on ``nn.MultiheadAttention``: its encoder's
    # ``self_attn``, whose two projections are adapted together.  The default also names the
    # decoder's ``multihead_attn``, which this encoder-only model does not have.
    _Family(
        name="transformer",
        factory="transformer_base_cls",
        overrides=_TEXT,
        batch=_cls_batch,
        task_type="SEQ_CLS",
        targets=["self_attn", "multihead_attn"],
        saved=[],
        adapted=_names("transformer.transformer.encoder.layers.{i}.self_attn"),
        all_linear=frozenset(),
        trained_in_full=frozenset({"classifier"}),
        modes=("explicit",),
        kinds=(("self_attn", nn.MultiheadAttention),),
    ),
    # A token table that nothing else shares (the transformer's is tied to its decoder's, so
    # it trains but cannot merge).  BERT's has a ``padding_idx``, which the lookup keeps.
    _Family(
        name="bert-embedding",
        factory="bert_base_cls",
        overrides=_TEXT,
        batch=_cls_batch,
        task_type="SEQ_CLS",
        targets=["word_embeddings"],
        saved=[],
        adapted=frozenset({"bert.embeddings.word_embeddings"}),
        all_linear=frozenset(),
        trained_in_full=frozenset({"classifier"}),
        modes=("explicit",),
        kinds=(("word_embeddings", nn.Embedding),),
    ),
]


def _in_stages(count: int, *patterns: str) -> frozenset[str]:
    """Every module name the ``{i}`` patterns spell out, for each of ``count`` stages."""
    return frozenset(pattern.format(i=i) for pattern in patterns for i in range(count))


_PVT_STAGES = 4
_PVT_QKV = _in_stages(_PVT_STAGES, "stages.{i}.blocks.0.attn.q", "stages.{i}.blocks.0.attn.kv")
_SWIN_STAGES = 2
_SWIN_QKV = _in_stages(_SWIN_STAGES, "stages.{i}.blocks.0.attn.qkv")

# PVT-v2 and Swin are vision transformers with a ``classifier`` head, no task type, and
# their own spelling of the projections: ``q`` + ``kv`` (PVT) and a fused ``qkv`` (Swin).
_FAMILIES += [
    _Family(
        name="pvt",
        factory="pvt_v2_b0_cls",
        overrides=_Overrides(
            embed_dims=(8, 16, 32, 64),
            depths=(1, 1, 1, 1),
            num_heads=(1, 2, 2, 4),
            mlp_ratios=(2.0, 2.0, 2.0, 2.0),
            num_classes=5,
        ),
        batch=_images(64),
        task_type="",
        targets=["q", "kv"],
        saved=["classifier"],
        adapted=_PVT_QKV,
        all_linear=_PVT_QKV
        | _in_stages(
            _PVT_STAGES,
            "stages.{i}.blocks.0.attn.proj",
            "stages.{i}.blocks.0.mlp.fc1",
            "stages.{i}.blocks.0.mlp.fc2",
        ),
        trained_in_full=frozenset({"classifier"}),
    ),
    _Family(
        name="swin",
        factory="swin_tiny_cls",
        overrides=_Overrides(
            image_size=32,
            patch_size=4,
            embed_dim=8,
            depths=(1, 1),
            num_heads=(1, 2),
            window_size=4,
            num_classes=5,
        ),
        batch=_images(32),
        task_type="",
        targets=["qkv"],
        saved=["classifier"],
        adapted=_SWIN_QKV,
        all_linear=_SWIN_QKV
        | _in_stages(
            _SWIN_STAGES,
            "stages.{i}.blocks.0.attn.proj",
            "stages.{i}.blocks.0.mlp.0",
            "stages.{i}.blocks.0.mlp.3",
        )
        | {"stages.0.downsample.proj"},
        trained_in_full=frozenset({"classifier"}),
    ),
]

#: An ``adapter_config.json`` as the reference library writes it for a GPT-2
#: LoRA (``Conv1D`` projections, hence ``fan_in_fan_out``), r=4 for the tiny model.
_REFERENCE_GPT2_ADAPTER_CONFIG = {
    "alpha_pattern": {},
    "auto_mapping": {
        "base_model_class": "GPT2LMHeadModel",
        "parent_library": "transformers.models.gpt2.modeling_gpt2",
    },
    "base_model_name_or_path": "gpt2",
    "bias": "none",
    "fan_in_fan_out": True,
    "inference_mode": True,
    "init_lora_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "modules_to_save": None,
    "peft_type": "LORA",
    "peft_version": "0.17.1",
    "r": 4,
    "rank_pattern": {},
    "revision": None,
    "target_modules": ["c_attn"],
    "task_type": "CAUSAL_LM",
    "use_dora": False,
    "use_rslora": False,
}

_GENERATING = [f for f in _FAMILIES if f.name in ("gpt2", "gpt", "t5")]
_ALL_LINEAR_FAMILIES = [f for f in _FAMILIES if "all-linear" in f.modes]
_FAMILY_MODES = [(f, mode) for f in _FAMILIES for mode in f.modes]


def _family(name: str) -> _Family:
    return next(f for f in _FAMILIES if f.name == name)


def _base(family: _Family, device: str) -> nn.Module:
    lucid.manual_seed(0)
    model = create_model(family.factory, **family.overrides)
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        # The zoo's ViT zero-initialises its classifier, which holds every
        # logit at zero whatever the layers below it do.
        nn.init.normal_(head.weight, std=0.2)
    return model.to(device)


def _config(family: _Family, mode: str, init_lora_weights: bool = True) -> LoraConfig:
    return LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=family.targets if mode == "explicit" else "all-linear",
        modules_to_save=family.saved or None,
        task_type=family.task_type or None,
        init_lora_weights=init_lora_weights,
    )


def _peft(family: _Family, mode: str, device: str, init_lora_weights: bool = True) -> PeftModel:
    return get_peft_model(_base(family, device), _config(family, mode, init_lora_weights))


def _logits(model: nn.Module, batch: _Batch) -> lucid.Tensor:
    return model(*batch.inputs).logits


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _adapter_leaves(kind: type[nn.Module]) -> tuple[str, ...]:
    """The adapter file's keys under one adapted module, as the reference writes them."""
    if kind is nn.Embedding:
        return ("lora_embedding_A", "lora_embedding_B")
    leaves = ("lora_A.weight", "lora_B.weight")
    if kind is nn.MultiheadAttention:
        return (*leaves, "base_layer.out_proj.lora_A.weight", "base_layer.out_proj.lora_B.weight")
    return leaves


def _silenced(layer: nn.Module) -> lucid.Tensor:
    """The tensor whose zeroing removes the layer's update: ``B``, or ``A`` for an embedding."""
    if isinstance(layer, lora.Embedding):
        return layer.lora_embedding_A["default"]
    assert isinstance(layer, (lora.Linear, lora.Conv2d, lora.MultiheadAttention))
    return layer.lora_B["default"].weight


def _train(peft: PeftModel, batch: _Batch, lr: float = _LR) -> list[float]:
    peft.train()
    opt = optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(_STEPS):
        opt.zero_grad()
        loss = peft(*batch.inputs, labels=batch.labels).loss
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _trained(family: _Family, device: str) -> tuple[PeftModel, _Batch]:
    """An eval-mode adapter with every LoRA matrix non-zero, trained a few steps."""
    peft = _peft(family, family.deepest_mode, device, init_lora_weights=False)
    batch = family.batch(device)
    _train(peft, batch)
    return peft.eval(), batch


_BY_FAMILY = pytest.mark.parametrize("family", _FAMILIES, ids=lambda f: f.name)
#: The rows named for a model type that has a default; the others reach other layer types.
_DEFAULT_ROWS = [f for f in _FAMILIES if f.name in DEFAULT_LORA_TARGET_MODULES]
_BY_DEFAULT_ROW = pytest.mark.parametrize("family", _DEFAULT_ROWS, ids=lambda f: f.name)
_BY_FAMILY_MODE = pytest.mark.parametrize(
    ("family", "mode"), _FAMILY_MODES, ids=[f"{f.name}-{m}" for f, m in _FAMILY_MODES]
)


# ── targets ──────────────────────────────────────────────────────────────────


@_BY_FAMILY
def test_the_zoo_s_module_names_select_what_the_table_says(family: _Family) -> None:
    peft = get_peft_model(_base(family, "cpu"), _config(family, "explicit"))
    assert set(peft.base_model.targeted_module_names) == family.adapted
    saved = set(peft.peft_config["default"].modules_to_save or [])
    assert saved == family.trained_in_full


@pytest.mark.parametrize("family", _ALL_LINEAR_FAMILIES, ids=lambda f: f.name)
def test_all_linear_leaves_out_the_head_and_the_tied_weights(family: _Family) -> None:
    peft = get_peft_model(_base(family, "cpu"), _config(family, "all-linear"))
    assert set(peft.base_model.targeted_module_names) == family.all_linear
    # The stored spelling is exactly the names adapted: a Linear head trained
    # in full (``modules_to_save``) is left out of it as it is left unadapted.
    stored = peft.peft_config["default"].target_modules
    assert stored == family.all_linear


def test_vit_has_no_query_and_value_modules_to_name() -> None:
    family = _family("vit")
    model = _base(family, "cpu")
    with pytest.raises(ValueError, match="not found"):
        get_peft_model(model, LoraConfig(r=4, target_modules=["query", "value"]))
    assert all(p.requires_grad for p in model.parameters())


# ── wiring ───────────────────────────────────────────────────────────────────


@_BY_FAMILY_MODE
def test_every_adapted_layer_is_on_the_forward_path(
    family: _Family, mode: str, device: str
) -> None:
    peft = _peft(family, mode, device, init_lora_weights=False).eval()
    batch = family.batch(device)
    with lucid.no_grad():
        live = _logits(peft, batch)
        for name in peft.base_model.targeted_module_names:
            b = _silenced(peft.get_base_model().get_submodule(name))
            kept = b.clone()
            b.zero_()
            silenced = _logits(peft, batch)
            b.copy_(kept)
            if name in family.off_path:
                assert _same(silenced, live), f"{name}: expected to be off the forward path"
            else:
                assert not _same(silenced, live), f"{name}: its update does not reach the output"
        assert _same(_logits(peft, batch), live)


# ── training ─────────────────────────────────────────────────────────────────


@_BY_FAMILY_MODE
def test_three_steps_lower_the_loss_and_move_only_the_adapter(
    family: _Family, mode: str, device: str
) -> None:
    peft = _peft(family, mode, device)
    batch = family.batch(device)
    before = {name: p.clone() for name, p in peft.named_parameters()}
    losses = _train(peft, batch)
    assert losses[-1] < losses[0], losses
    for name, param in peft.named_parameters():
        if _owned_by_adapter(name):
            assert param.requires_grad, name
        else:
            assert not param.requires_grad and param.grad is None, name
            assert _same(param, before[name]), f"{name}: a base weight changed"
    moved = {n for n, p in peft.named_parameters() if not _same(p, before[n])}
    assert moved and all(_owned_by_adapter(n) for n in moved)
    for full in family.trained_in_full:
        assert any(f"{full}.modules_to_save." in n for n in moved), f"{full} did not train"


# ── inference ────────────────────────────────────────────────────────────────


@_BY_FAMILY
def test_disable_adapter_gives_the_pre_injection_outputs(family: _Family, device: str) -> None:
    model = _base(family, device).eval()
    batch = family.batch(device)
    with lucid.no_grad():
        plain = _logits(model, batch)
        peft = get_peft_model(model, _config(family, family.deepest_mode, init_lora_weights=False))
        adapted = _logits(peft, batch)
        with peft.disable_adapter():
            off = _logits(peft, batch)
        back = _logits(peft, batch)
    assert not _same(adapted, plain)
    assert _same(off, plain)
    assert _same(back, adapted)


def _generate(model: nn.Module, family: _Family, batch: _Batch) -> lucid.Tensor:
    if family.name == "t5":
        return model.generate(batch.inputs[0], max_length=8)
    return model.generate(batch.inputs[0][:, :4], max_new_tokens=6)


@pytest.mark.parametrize("family", _GENERATING, ids=lambda f: f.name)
def test_generate_through_the_wrapper_follows_the_adapter(family: _Family, device: str) -> None:
    model = _base(family, device).eval()
    batch = family.batch(device)
    before = _generate(model, family, batch)
    # A large random update (B is not zero-initialised), so greedy decoding
    # must pick other tokens somewhere.
    config = LoraConfig(
        r=4,
        lora_alpha=256,
        target_modules=family.targets,
        task_type=family.task_type,
        init_lora_weights=False,
    )
    peft = get_peft_model(model, config).eval()
    adapted = _generate(peft, family, batch)
    with peft.disable_adapter():
        off = _generate(peft, family, batch)
    assert not _same(adapted, before)
    assert _same(off, before)


def _random_update(family: _Family, rank: int, alpha: float) -> LoraConfig:
    """A config whose ``B`` is not zeroed, so the adapter changes what the model computes."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=family.targets,
        task_type=family.task_type,
        init_lora_weights=False,
    )


@pytest.mark.parametrize("family", _GENERATING, ids=lambda f: f.name)
def test_two_adapters_train_one_switch_and_generate(family: _Family, device: str) -> None:
    model = _base(family, device).eval()
    batch = family.batch(device)
    plain = _generate(model, family, batch)
    peft = get_peft_model(model, _random_update(family, 4, 256)).eval()
    first = _generate(peft, family, batch)
    with lucid.no_grad():
        first_logits = _logits(peft, batch)

    # Adding an adapter computes and trains nothing: the first still answers.
    peft.add_adapter("second", _random_update(family, 2, 64))
    assert peft.active_adapter == "default" and _same(_generate(peft, family, batch), first)
    before = {name: p.clone() for name, p in peft.named_parameters()}
    peft.set_adapter("second")
    losses = _train(peft, batch)
    assert losses[-1] < losses[0], losses

    # Only the second adapter moved; the first and the base are bit-identical.
    for name, param in peft.named_parameters():
        if ".second." in name:
            continue
        assert not param.requires_grad and _same(param, before[name]), name
    moved = [n for n, p in peft.named_parameters() if ".second." in n and not _same(p, before[n])]
    assert moved
    peft.eval()
    with lucid.no_grad():
        second_logits = _logits(peft, batch)
    second = _generate(peft, family, batch)
    assert not _same(second_logits, first_logits)

    # Switching back and forth gives each adapter's own answers, bit for bit.
    peft.set_adapter("default")
    assert _same(_generate(peft, family, batch), first)
    with lucid.no_grad():
        assert _same(_logits(peft, batch), first_logits)
    with peft.disable_adapter():
        assert _same(_generate(peft, family, batch), plain)
    peft.set_adapter("second")
    assert _same(_generate(peft, family, batch), second)


# ── deployment ───────────────────────────────────────────────────────────────


@_BY_FAMILY
@pytest.mark.parametrize("safe_serialization", [False, True])
def test_merge_and_unload_then_zoo_save_and_load(
    family: _Family, safe_serialization: bool, device: str, tmp_path: Path
) -> None:
    peft, batch = _trained(family, device)
    keys = set(_base(family, "cpu").state_dict())
    targets = list(peft.base_model.targeted_module_names)
    with lucid.no_grad():
        adapted = _logits(peft, batch)
        merged = peft.merge_and_unload()
        assert set(merged.state_dict()) == keys
        assert all(type(merged.get_submodule(n)) is family.kind_of(n) for n in targets)
        assert lucid.allclose(_logits(merged, batch), adapted, atol=1e-4)
        merged.save_pretrained(str(tmp_path), safe_serialization=safe_serialization)
        reloaded = type(merged).from_pretrained(str(tmp_path)).to(device).eval()
        assert _same(_logits(reloaded, batch), _logits(merged, batch))


@_BY_FAMILY_MODE
def test_the_adapter_file_holds_the_adapter_under_the_zoo_s_names(
    family: _Family, mode: str, tmp_path: Path
) -> None:
    peft = _peft(family, mode, "cpu")
    peft.save_pretrained(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    stored = set(load_safetensors(str(tmp_path / "adapter_model.safetensors")))
    layers = {
        f"base_model.model.{name}.{leaf}"
        for name in peft.base_model.targeted_module_names
        for leaf in _adapter_leaves(family.kind_of(name))
    }
    # A module trained in full keeps its own name on disk, and the adapter's
    # name is in no key.
    heads = {key.rpartition(".")[0].removeprefix("base_model.model.") for key in stored - layers}
    assert layers <= stored and not any("default" in key for key in stored)
    assert heads == family.trained_in_full
    assert stored == set(get_peft_model_state_dict(peft))


@_BY_FAMILY_MODE
def test_a_saved_adapter_loads_onto_a_fresh_base(
    family: _Family, mode: str, device: str, tmp_path: Path
) -> None:
    peft = _peft(family, mode, device, init_lora_weights=False)
    batch = family.batch(device)
    _train(peft, batch)
    peft.eval().save_pretrained(tmp_path)
    with lucid.no_grad():
        trained = _logits(peft, batch)
    fresh = PeftModel.from_pretrained(_base(family, device), tmp_path)
    assert not any(p.requires_grad for p in fresh.parameters())
    with lucid.no_grad():
        assert _same(_logits(fresh, batch), trained)
    resumed = PeftModel.from_pretrained(_base(family, device), tmp_path, is_trainable=True)
    assert resumed.get_nb_trainable_parameters() == peft.get_nb_trainable_parameters()


def test_a_gpt2_adapter_written_by_the_reference_loads_by_name_and_shape(tmp_path: Path) -> None:
    """Keys and shapes spelled from the reference library's GPT-2 layout, not from Lucid's.

    Its ``c_attn`` is a ``Conv1D`` whose weight is stored transposed, which only
    the base weight shows: the adapter's ``lora_A`` is ``(r, n_embd)`` and
    ``lora_B`` ``(3 * n_embd, r)`` either way, and ``fan_in_fan_out`` is
    the one config field that differs.
    """
    r, width = 4, _TEXT["hidden_size"]
    lucid.manual_seed(7)
    written = {}
    for i in range(_DEPTH):
        prefix = f"base_model.model.transformer.h.{i}.attn.c_attn"
        written[f"{prefix}.lora_A.weight"] = lucid.randn(r, width)
        written[f"{prefix}.lora_B.weight"] = lucid.randn(3 * width, r)
    save_safetensors(
        written, str(tmp_path / "adapter_model.safetensors"), metadata={"format": "pt"}
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps(_REFERENCE_GPT2_ADAPTER_CONFIG))
    with pytest.warns(UserWarning, match="fan_in_fan_out"):
        loaded = PeftModel.from_pretrained(_base(_family("gpt2"), "cpu"), tmp_path)
    assert loaded.peft_config["default"].fan_in_fan_out is False
    assert set(get_peft_model_state_dict(loaded)) == set(written)
    for key, value in get_peft_model_state_dict(loaded).items():
        assert _same(value, written[key]), key
    batch = _family("gpt2").batch("cpu")
    with lucid.no_grad():
        assert not _same(_logits(loaded, batch), _logits(_base(_family("gpt2"), "cpu"), batch))


# ── tasks ────────────────────────────────────────────────────────────────────

#: The wrapper ``get_peft_model`` returns for each family, by the family's task type; a
#: family with no task type (the vision models) gets the plain ``PeftModel``.
_WRAPPER: dict[str, type[PeftModel]] = {
    "gpt2": PeftModelForCausalLM,
    "gpt": PeftModelForCausalLM,
    "bert": PeftModelForSequenceClassification,
    "roformer": PeftModelForSequenceClassification,
    "t5": PeftModelForSeq2SeqLM,
    "vit": PeftModel,
    "pvt": PeftModel,
    "swin": PeftModel,
    "vit-patch": PeftModel,
    "transformer": PeftModelForSequenceClassification,
    "bert-embedding": PeftModelForSequenceClassification,
}

#: The class that rebuilds base and adapter from the adapter's directory.
_AUTO = {
    "gpt2": AutoPeftModelForCausalLM,
    "gpt": AutoPeftModelForCausalLM,
    "bert": AutoPeftModelForSequenceClassification,
    "roformer": AutoPeftModelForSequenceClassification,
    "t5": AutoPeftModelForSeq2SeqLM,
    "vit": AutoPeftModel,
    "pvt": AutoPeftModel,
    "swin": AutoPeftModel,
    "vit-patch": AutoPeftModel,
    "transformer": AutoPeftModelForSequenceClassification,
    "bert-embedding": AutoPeftModelForSequenceClassification,
}


@_BY_FAMILY_MODE
def test_get_peft_model_returns_the_wrapper_of_the_familys_task(family: _Family, mode: str) -> None:
    assert type(_peft(family, mode, "cpu")) is _WRAPPER[family.name]


@_BY_FAMILY
def test_the_wrapper_computes_what_the_base_computed_at_initialisation(
    family: _Family, device: str
) -> None:
    model = _base(family, device).eval()
    batch = family.batch(device)
    with lucid.no_grad():
        plain = _logits(model, batch)
        peft = get_peft_model(model, _config(family, "explicit")).eval()
        assert _same(_logits(peft, batch), plain)


@pytest.mark.parametrize("family", _GENERATING, ids=lambda f: f.name)
def test_the_wrapper_generates_what_the_base_generated_at_initialisation(
    family: _Family, device: str
) -> None:
    model = _base(family, device).eval()
    batch = family.batch(device)
    before = _generate(model, family, batch)
    peft = get_peft_model(model, _config(family, "explicit")).eval()
    assert isinstance(peft, (PeftModelForCausalLM, PeftModelForSeq2SeqLM))
    assert _same(_generate(peft, family, batch), before)


def _auto_adapter(family: _Family, device: str, root: Path) -> tuple[Path, lucid.Tensor, _Batch]:
    """A trained adapter in ``root/adapter`` whose config names the base saved in ``root/base``."""
    base = _base(family, "cpu")
    base.save_pretrained(str(root / "base"))
    config = dataclasses.replace(
        _config(family, family.deepest_mode, init_lora_weights=False),
        base_model_name_or_path=str(root / "base"),
    )
    peft = get_peft_model(base.to(device), config)
    batch = family.batch(device)
    _train(peft, batch)
    peft.eval().save_pretrained(root / "adapter")
    with lucid.no_grad():
        return root / "adapter", _logits(peft, batch), batch


@_BY_FAMILY
def test_the_adapters_directory_rebuilds_base_and_adapter(
    family: _Family, device: str, tmp_path: Path
) -> None:
    directory, trained, batch = _auto_adapter(family, device, tmp_path)
    loaded = _AUTO[family.name].from_pretrained(directory)
    assert type(loaded) is _WRAPPER[family.name]
    assert type(loaded.get_base_model()) is type(_base(family, "cpu"))
    assert not any(p.requires_grad for p in loaded.parameters())
    loaded.to(device)
    with lucid.no_grad():
        assert _same(_logits(loaded, batch), trained)


@pytest.mark.parametrize("family", _GENERATING, ids=lambda f: f.name)
def test_the_rebuilt_model_generates_what_the_trained_one_did(
    family: _Family, device: str, tmp_path: Path
) -> None:
    directory, _, batch = _auto_adapter(family, device, tmp_path)
    trained = PeftModel.from_pretrained(_base(family, device), directory).eval()
    loaded = _AUTO[family.name].from_pretrained(directory).to(device).eval()
    assert _same(_generate(loaded, family, batch), _generate(trained, family, batch))


# ── defaults ─────────────────────────────────────────────────────────────────


def test_the_deep_rows_are_the_tables_and_every_default_is_a_matrix_row() -> None:
    assert {f.name for f in _DEFAULT_ROWS} <= set(DEFAULT_LORA_TARGET_MODULES)
    for family in _DEFAULT_ROWS:
        assert DEFAULT_LORA_TARGET_MODULES[family.name] == set(family.targets), family.name


@_BY_DEFAULT_ROW
def test_no_target_modules_takes_the_default_of_the_model_type(
    family: _Family, device: str
) -> None:
    model = _base(family, device).eval()
    assert model.config.model_type == family.name
    batch = family.batch(device)
    with lucid.no_grad():
        plain = _logits(model, batch)
    config = LoraConfig(r=4, lora_alpha=8, modules_to_save=family.saved or None)
    peft = get_peft_model(model, config).eval()
    assert set(peft.base_model.targeted_module_names) == family.adapted
    assert peft.peft_config["default"].target_modules == set(family.targets)
    with lucid.no_grad():
        assert _same(_logits(peft, batch), plain)


@_BY_DEFAULT_ROW
def test_the_default_modules_are_the_ones_the_zoo_trains_on(family: _Family, device: str) -> None:
    """The default is live: an update to each module it picks reaches the output."""
    config = LoraConfig(
        r=4, lora_alpha=8, modules_to_save=family.saved or None, init_lora_weights=False
    )
    peft = get_peft_model(_base(family, device), config).eval()
    batch = family.batch(device)
    with lucid.no_grad():
        live = _logits(peft, batch)
        for name in peft.base_model.targeted_module_names:
            b = peft.get_base_model().get_submodule(name).lora_B["default"].weight
            kept = b.clone()
            b.zero_()
            silenced = _logits(peft, batch)
            b.copy_(kept)
            assert _same(silenced, live) == (name in family.off_path), name


class _Unregistered(nn.Module):
    """A module with no ``config``: the reference's ``"custom"`` model type, which has no default."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


def test_a_model_type_the_table_lacks_is_told_to_name_its_targets() -> None:
    model = _Unregistered()
    with pytest.raises(ValueError, match="no default for model type 'custom'"):
        get_peft_model(model, LoraConfig(r=4))
    assert not any(".lora_" in n for n, _ in model.named_modules())


# ── attention, embedding and linear layers in one model ──────────────────────


def _transformer_classifier() -> nn.Module:
    lucid.manual_seed(0)
    return create_model("transformer_base_cls", **_TEXT)


def test_an_attention_has_no_linear_projections_to_name() -> None:
    """Its projections are parameters of ``self_attn``, which is what a config names."""
    model = _transformer_classifier()
    with pytest.raises(ValueError, match="not found"):
        get_peft_model(model, LoraConfig(r=4, target_modules=["out_proj", "q_proj", "in_proj"]))
    assert all(p.requires_grad for p in model.parameters())
    assert not any(".lora_" in n for n, _ in model.named_modules())


def test_three_layer_types_adapt_in_one_model_and_train_together(device: str) -> None:
    model = _transformer_classifier().to(device)
    config = LoraConfig(
        r=4, target_modules=["self_attn", "src_tok_emb", "linear1"], task_type="SEQ_CLS"
    )
    peft = get_peft_model(model, config)
    kinds = {type(model.get_submodule(n)) for n in peft.base_model.targeted_module_names}
    assert kinds == {lora.MultiheadAttention, lora.Embedding, lora.Linear}
    ids = _tokens(device, 0, (3, 10), seed=1)
    labels = _tokens(device, 0, (3,), seed=2) % 2
    losses = _train(peft, _Batch((ids,), labels))
    assert losses[-1] < losses[0], losses


def test_all_linear_on_the_transformer_family_reaches_the_feed_forward_only(device: str) -> None:
    model = _transformer_classifier().to(device)
    peft = get_peft_model(model, LoraConfig(r=4, target_modules="all-linear", task_type="SEQ_CLS"))
    assert set(peft.base_model.targeted_module_names) == _names(
        "transformer.transformer.encoder.layers.{i}.linear1",
        "transformer.transformer.encoder.layers.{i}.linear2",
    )
    ids = _tokens(device, 0, (3, 10), seed=1)
    labels = _tokens(device, 0, (3,), seed=2) % 2
    losses = _train(peft, _Batch((ids,), labels))
    assert losses[-1] < losses[0], losses


# ── a tied embedding and a head trained in full ──────────────────────────────


def _tied_head_peft(device: str, init_lora_weights: bool = True) -> PeftModel:
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        modules_to_save=["lm_head"],
        task_type="CAUSAL_LM",
        init_lora_weights=init_lora_weights,
    )
    return get_peft_model(_base(_family("gpt2"), device), config)


def test_a_lm_head_trained_in_full_leaves_the_tied_embedding_alone(device: str) -> None:
    peft = _tied_head_peft(device)
    model = peft.get_base_model()
    assert model.lm_head.original_module.weight is model.transformer.wte.weight
    table = model.transformer.wte.weight.clone()
    batch = _family("gpt2").batch(device)
    losses = _train(peft, batch)
    assert losses[-1] < losses[0], losses
    assert _same(model.transformer.wte.weight, table)
    trained = model.lm_head.modules_to_save["default"].weight
    assert trained is not model.transformer.wte.weight and not _same(trained, table)


def test_a_tied_head_trained_in_full_reloads_as_an_adapter(device: str, tmp_path: Path) -> None:
    peft = _tied_head_peft(device, init_lora_weights=False)
    batch = _family("gpt2").batch(device)
    _train(peft, batch)
    peft.eval()
    peft.save_pretrained(tmp_path)
    fresh = PeftModel.from_pretrained(_base(_family("gpt2"), device), tmp_path)
    with lucid.no_grad():
        assert _same(_logits(fresh, batch), _logits(peft, batch))


def test_merging_a_head_trained_apart_from_its_tie_is_refused(device: str) -> None:
    """LCD-708: the merged model would hold two tensors under names the zoo re-ties.

    ``from_pretrained`` re-ties ``lm_head`` to ``transformer.wte`` from the
    config, so one of the two trained tensors would silently overwrite the
    other.  The adapter route (save / ``PeftModel.from_pretrained``) is the
    supported one, held above.
    """
    with pytest.warns(UserWarning, match="merge_and_unload\\(\\) will refuse"):
        peft = _tied_head_peft(device, init_lora_weights=False)
    batch = _family("gpt2").batch(device)
    _train(peft, batch)
    peft.eval()
    with lucid.no_grad():
        before = _logits(peft, batch)
        with pytest.raises(RuntimeError, match="transformer.wte.weight"):
            peft.merge_and_unload()
        assert _same(_logits(peft, batch), before)


# ── the coverage matrix: every model type the zoo registers ──────────────────

_MATRIX_ROWS = [pytest.param(row, id=name, marks=row.marks()) for name, row in ROWS.items()]

#: Factories that cannot be built under ``shadow_alloc``, and why.
_UNSHADOWED = {
    "ddpm_imagenet64_gen": (
        "learn_sigma reads the beta schedule on the host at construction (tolist of a "
        "phantom tensor)"
    )
}
_FACTORIES = [
    pytest.param(
        name,
        id=name,
        marks=(
            [pytest.mark.xfail(strict=True, reason=_UNSHADOWED[name])]
            if name in _UNSHADOWED
            else []
        ),
    )
    for name in sorted(_REGISTRY)
]


def test_every_zoo_model_type_is_a_matrix_row() -> None:
    assert set(ROWS) == zoo_model_types()
    assert all(row.model_type == name for name, row in ROWS.items())


def test_the_defaults_are_the_matrix_rows() -> None:
    """A type the table lacks is a type ``LoraConfig()`` refuses; every type has one."""
    defaults = set(DEFAULT_LORA_TARGET_MODULES) | set(DEFAULT_LORA_TARGET_PARAMETERS)
    assert defaults == set(ROWS)


@pytest.mark.parametrize("row", _MATRIX_ROWS)
def test_a_zoo_model_type_adapts_trains_saves_and_merges(row: ZooRow) -> None:
    Trial(row).run()


@pytest.mark.parametrize("factory", _FACTORIES)
def test_a_default_reaches_adaptable_layers_in_every_factory_of_its_type(factory: str) -> None:
    assert default_problems(factory) == []


def test_tvm_averages_follow_an_adapter_injected_after_they_were_built() -> None:
    """LCD-759: the EMA copies mirror the live tree, so injecting later restarts them from it."""
    row = ROWS["tvm"]
    model = row.build()
    args, kwargs = row.inputs()
    before = model.target_network
    assert not any("lora_" in name for name, _ in before.named_parameters())
    peft = get_peft_model(model, LoraConfig(r=2, init_lora_weights=False))
    peft.train()
    assert any("lora_" in name for name, _ in model.target_network.named_parameters())
    assert model.target_network is not before
    model.update_ema()  # pairs the copies' parameters with the live ones by name
    assert not any(p.requires_grad for p in model.target_network.parameters())
    peft(*args, **kwargs)
