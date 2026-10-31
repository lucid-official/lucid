"""(IA)^3, LoHa and LoKr on the model zoo, end to end (tiny configs, CPU and Metal).

Each tuner on GPT-2 (``nn.Linear`` projections), ViT (``nn.Linear`` and
a ``nn.Conv2d`` patch embedding) and — for the (IA)^3 defaults — BERT and T5:
the adapted model starts as the base bit for bit, a few optimizer steps move
the adapter and nothing else and lower the loss, an adapter file reloads to
the same logits, merging gives the same logits, and the model-type defaults
select the modules the reference's tables name, spelled as the zoo spells them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.models import create_model
from lucid.peft import (
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    PeftConfig,
    PeftModel,
    PeftModelForCausalLM,
    get_peft_model,
)
from lucid.peft.tuners._base import _tuner_layers

_DEPTH = 2
_TEXT: dict[str, object] = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": _DEPTH,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 32,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}
_VIT: dict[str, object] = {
    "image_size": 16,
    "patch_size": 8,
    "dim": 32,
    "depth": _DEPTH,
    "num_heads": 2,
    "num_classes": 5,
    "dropout": 0.0,
    "attention_dropout": 0.0,
}


class _Batch(NamedTuple):
    inputs: tuple[lucid.Tensor, ...]
    labels: lucid.Tensor


def _tokens(device: str, shape: tuple[int, ...], seed: int, low: int = 0) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randint(low, 64, shape, device=device)


def _lm_batch(device: str) -> _Batch:
    ids = _tokens(device, (2, 12), seed=1)
    return _Batch((ids,), ids)


def _image_batch(device: str) -> _Batch:
    lucid.manual_seed(1)
    return _Batch((lucid.randn(3, 3, 16, 16, device=device),), _tokens(device, (3,), 2) % 5)


@dataclass(frozen=True)
class _Row:
    name: str
    factory: str
    overrides: dict[str, object]
    batch: Callable[[str], _Batch]
    task_type: str | None
    saved: list[str] | None
    #: The ``target_modules`` (and, for (IA)^3, ``feedforward_modules``) each tuner is asked for.
    targets: dict[str, tuple[list[str], list[str]]]


_ROWS = [
    _Row(
        "gpt2",
        "gpt2_small_lm",
        _TEXT,
        _lm_batch,
        "CAUSAL_LM",
        None,
        {
            "ia3": (["c_attn", "mlp.c_proj"], ["mlp.c_proj"]),
            "loha": (["c_attn"], []),
            "lokr": (["c_attn"], []),
        },
    ),
    _Row(
        "vit",
        "vit_base_16_cls",
        _VIT,
        _image_batch,
        None,
        ["head"],
        {"ia3": (["qkv", "fc2"], ["fc2"]), "loha": (["qkv"], []), "lokr": (["qkv"], [])},
    ),
    # The patch embedding is the family's one convolution: given in full, since ``proj``
    # alone also ends the attention's output ``nn.Linear``.
    _Row(
        "vit-patch",
        "vit_base_16_cls",
        _VIT,
        _image_batch,
        None,
        ["head"],
        {
            "ia3": (["patch_embed.proj"], []),
            "loha": (["patch_embed.proj"], []),
            "lokr": (["patch_embed.proj"], []),
        },
    ),
]
_TUNERS = ("ia3", "loha", "lokr")
_CASES = [(t, row) for t in _TUNERS for row in _ROWS]


def _config(tuner: str, row: _Row, random: bool = False) -> PeftConfig:
    targets, feedforward = row.targets[tuner]
    common: dict[str, object] = {
        "target_modules": targets,
        "modules_to_save": row.saved,
        "task_type": row.task_type,
    }
    if tuner == "ia3":
        return IA3Config(feedforward_modules=feedforward, init_ia3_weights=not random, **common)  # type: ignore[arg-type]
    if tuner == "loha":
        return LoHaConfig(r=4, alpha=8, init_weights=not random, **common)  # type: ignore[arg-type]
    return LoKrConfig(r=4, alpha=8, init_weights=not random, **common)  # type: ignore[arg-type]


def _base(row: _Row, device: str) -> nn.Module:
    lucid.manual_seed(0)
    model = create_model(row.factory, **row.overrides)
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        # The zoo's ViT zero-initialises its classifier, which holds every logit at zero.
        nn.init.normal_(head.weight, std=0.2)
    return model.to(device)


def _logits(model: nn.Module, batch: _Batch) -> lucid.Tensor:
    return model(*batch.inputs).logits


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _train(peft: PeftModel, batch: _Batch, steps: int = 3, lr: float = 3e-2) -> list[float]:
    peft.train()
    opt = optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = peft(*batch.inputs, labels=batch.labels).loss
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


_BY_CASE = pytest.mark.parametrize(
    ("tuner", "row"), _CASES, ids=[f"{t}-{r.name}" for t, r in _CASES]
)


@_BY_CASE
def test_the_adapted_model_starts_as_the_base(tuner: str, row: _Row, device: str) -> None:
    base = _base(row, device)
    batch = row.batch(device)
    expected = _logits(base, batch)
    peft = get_peft_model(base, _config(tuner, row))
    assert _same(_logits(peft.eval(), batch), expected)


@_BY_CASE
def test_training_moves_the_adapter_alone_and_lowers_the_loss(
    tuner: str, row: _Row, device: str
) -> None:
    peft = get_peft_model(_base(row, device), _config(tuner, row))
    batch = row.batch(device)
    frozen = {n: p.clone() for n, p in peft.named_parameters() if not p.requires_grad}
    initial = {n: p.clone() for n, p in peft.named_parameters() if p.requires_grad}
    assert frozen and initial
    losses = _train(peft, batch)
    assert losses[-1] < losses[0], losses
    for name, p in peft.named_parameters():
        if name in frozen:
            assert _same(p, frozen[name]), name
    assert any(not _same(p, initial[n]) for n, p in peft.named_parameters() if n in initial)


@_BY_CASE
def test_a_saved_adapter_reloads_to_the_same_logits(
    tuner: str, row: _Row, device: str, tmp_path: Path
) -> None:
    peft = get_peft_model(_base(row, device), _config(tuner, row, random=True))
    batch = row.batch(device)
    _train(peft, batch)
    peft.eval()
    with lucid.no_grad():
        expected = _logits(peft, batch)
    peft.save_pretrained(tmp_path)
    loaded = PeftModel.from_pretrained(_base(row, device), tmp_path)
    assert type(loaded.base_model) is type(peft.base_model)
    with lucid.no_grad():
        assert _same(_logits(loaded, batch), expected)


@_BY_CASE
def test_merging_gives_the_same_logits(tuner: str, row: _Row, device: str) -> None:
    peft = get_peft_model(_base(row, device), _config(tuner, row, random=True))
    batch = row.batch(device)
    _train(peft, batch)
    peft.eval()
    with lucid.no_grad():
        expected = _logits(peft, batch)
        merged = peft.merge_and_unload()
        assert not any(True for _ in _tuner_layers(merged))
        assert bool(lucid.allclose(_logits(merged, batch), expected, atol=1e-3))


def test_the_task_type_gives_the_reference_s_wrapper(device: str) -> None:
    row = _ROWS[0]
    peft = get_peft_model(_base(row, device), _config("ia3", row))
    assert type(peft) is PeftModelForCausalLM
    assert isinstance(get_peft_model(_base(_ROWS[1], device), _config("loha", _ROWS[1])), PeftModel)


# ── (IA)^3's per-model-type defaults ─────────────────────────────────────────

_GPT2_IA3 = {
    f"transformer.h.{i}.{tail}" for i in range(_DEPTH) for tail in ("attn.c_attn", "mlp.c_proj")
}
_BERT_IA3 = {
    f"bert.encoder.layer.{i}.{tail}"
    for i in range(_DEPTH)
    for tail in (
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "output.dense",
    )
}
_T5_IA3 = {
    f"t5.encoder.blocks.{i}.{tail}"
    for i in range(_DEPTH)
    for tail in ("self_attention.k", "self_attention.v", "feed_forward.wo")
} | {
    f"t5.decoder.blocks.{i}.{tail}"
    for i in range(_DEPTH)
    for tail in (
        "self_attention.k",
        "self_attention.v",
        "cross_attention.k",
        "cross_attention.v",
        "feed_forward.wo",
    )
}


class _Default(NamedTuple):
    factory: str
    overrides: dict[str, object]
    adapted: set[str]
    feedforward_suffix: tuple[str, ...]


_DEFAULTS = {
    "gpt2": _Default("gpt2_small_lm", _TEXT, _GPT2_IA3, ("mlp.c_proj",)),
    "bert": _Default("bert_base_cls", _TEXT, _BERT_IA3, ("output.dense",)),
    "t5": _Default(
        "t5_small_seq2seq",
        {**_TEXT, "num_decoder_layers": _DEPTH, "head_dim": 16},
        _T5_IA3,
        ("wo",),
    ),
}


@pytest.mark.parametrize("family", sorted(_DEFAULTS))
def test_ia3_with_no_names_adapts_what_the_reference_s_table_says(family: str) -> None:
    spec = _DEFAULTS[family]
    lucid.manual_seed(0)
    model = create_model(spec.factory, **spec.overrides)
    peft = get_peft_model(model, IA3Config())
    layers = dict(_tuner_layers(model))
    assert set(layers) == spec.adapted
    for name, layer in layers.items():
        assert layer.is_feedforward == name.endswith(spec.feedforward_suffix), name  # type: ignore[attr-defined]
    config = peft.peft_config["default"]
    assert isinstance(config, IA3Config)
    assert config.feedforward_modules == set(spec.feedforward_suffix)
    assert config.target_modules is not None and config.feedforward_modules <= config.target_modules


@pytest.mark.parametrize("family", sorted(_DEFAULTS))
def test_ia3_defaults_start_as_the_base(family: str, device: str) -> None:
    spec = _DEFAULTS[family]
    lucid.manual_seed(0)
    model = create_model(spec.factory, **spec.overrides).to(device).eval()
    if family == "t5":
        inputs = (_tokens(device, (2, 9), 1, 2), _tokens(device, (2, 6), 2, 2))
    else:
        inputs = (_tokens(device, (2, 12), 1, 2),)
    with lucid.no_grad():
        expected = model(*inputs).logits
        peft = get_peft_model(model, IA3Config())
        assert _same(peft.eval()(*inputs).logits, expected)
