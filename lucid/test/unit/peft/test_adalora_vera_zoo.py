"""AdaLoRA and VeRA on the model zoo, end to end (tiny configs, CPU and Metal).

Each tuner on GPT-2 (``nn.Linear`` projections) and ViT (``nn.Linear`` and, for AdaLoRA, the
``nn.Conv2d`` patch embedding): the adapted model starts as the base bit for bit, a few optimizer
steps move the adapter and nothing else and lower the loss, an adapter file reloads to the same
logits, merging gives the same logits.  AdaLoRA also runs its rank allocator through a real
training loop, and the default targets of both are checked against the zoo's module names.
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
from lucid.peft import PeftModel, PeftModelForCausalLM, get_peft_model
from lucid.peft.tuners._base import _tuner_layers
from lucid.peft.tuners.adalora import AdaLoraConfig, AdaLoraLayer, AdaLoraModel
from lucid.peft.tuners.vera import VeraConfig, VeraLayer

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
    targets: dict[str, list[str]]


_ROWS = [
    _Row(
        "gpt2",
        "gpt2_small_lm",
        _TEXT,
        _lm_batch,
        "CAUSAL_LM",
        None,
        {"adalora": ["c_attn"], "vera": ["c_attn"]},
    ),
    _Row(
        "vit",
        "vit_base_16_cls",
        _VIT,
        _image_batch,
        None,
        ["head"],
        {"adalora": ["qkv"], "vera": ["qkv"]},
    ),
    # The family's one convolution: only AdaLoRA has a layer for it.
    _Row(
        "vit-patch",
        "vit_base_16_cls",
        _VIT,
        _image_batch,
        None,
        ["head"],
        {"adalora": ["patch_embed.proj"]},
    ),
]
_CASES = [(t, row) for row in _ROWS for t in ("adalora", "vera") if t in row.targets]


def _config(tuner: str, row: _Row, random: bool = False) -> AdaLoraConfig | VeraConfig:
    common: dict[str, object] = {
        "target_modules": row.targets[tuner],
        "modules_to_save": row.saved,
        "task_type": row.task_type,
    }
    if tuner == "adalora":
        return AdaLoraConfig(
            init_r=4,
            target_r=2,
            lora_alpha=8,
            total_step=8,
            tinit=1,
            tfinal=2,
            init_lora_weights=not random,
            **common,  # type: ignore[arg-type]
        )
    return VeraConfig(r=8, init_weights=not random, **common)  # type: ignore[arg-type]


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
    tuner = peft.base_model
    opt = optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=lr)
    losses = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = tuner(*batch.inputs, labels=batch.labels).loss
        loss.backward()
        opt.step()
        if isinstance(tuner, AdaLoraModel):
            tuner.update_and_allocate(step)
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
        assert bool(lucid.allclose(_logits(loaded, batch), expected, atol=1e-5))


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
    assert type(get_peft_model(_base(row, device), _config("adalora", row))) is PeftModelForCausalLM
    assert isinstance(get_peft_model(_base(_ROWS[1], device), _config("vera", _ROWS[1])), PeftModel)


# ── AdaLoRA's allocation through a real loop ─────────────────────────────────


def test_the_allocator_reaches_the_target_rank_after_the_final_step(device: str) -> None:
    row = _ROWS[0]
    peft = get_peft_model(_base(row, device), _config("adalora", row, random=True))
    tuner = peft.base_model
    assert isinstance(tuner, AdaLoraModel)
    batch = row.batch(device)
    layers = [layer for _, layer in _tuner_layers(tuner.model) if isinstance(layer, AdaLoraLayer)]
    assert len(layers) == _DEPTH and tuner.rankallocator.init_bgt == 4 * _DEPTH
    _train(peft, batch, steps=7)  # total_step 8, tfinal 2: the budget is final at step 6
    config = tuner.peft_config["default"]
    assert isinstance(config, AdaLoraConfig) and config.rank_pattern is not None
    assert sum(sum(flags) for flags in config.rank_pattern.values()) <= 2 * _DEPTH
    live = sum(int((layer.lora_E["default"] != 0).sum().item()) for layer in layers)
    assert live <= 2 * _DEPTH


def test_the_penalty_lifts_the_loss_above_the_models_own(device: str) -> None:
    row = _ROWS[0]
    peft = get_peft_model(_base(row, device), _config("adalora", row, random=True))
    batch = row.batch(device)
    penalised = peft.base_model(*batch.inputs, labels=batch.labels).loss
    plain = peft.get_base_model()(*batch.inputs, labels=batch.labels).loss
    assert float(penalised.item()) > float(plain.item())


# ── the default targets ──────────────────────────────────────────────────────

_GPT2 = {f"transformer.h.{i}.attn.c_attn" for i in range(_DEPTH)}
_BERT = {
    f"bert.encoder.layer.{i}.attention.self.{n}" for i in range(_DEPTH) for n in ("query", "value")
}
_T5 = {f"t5.encoder.blocks.{i}.self_attention.{n}" for i in range(_DEPTH) for n in ("q", "v")} | {
    f"t5.decoder.blocks.{i}.{a}.{n}"
    for i in range(_DEPTH)
    for a in ("self_attention", "cross_attention")
    for n in ("q", "v")
}


class _Default(NamedTuple):
    factory: str
    overrides: dict[str, object]
    adapted: set[str]


_DEFAULTS = {
    "gpt2": _Default("gpt2_small_lm", _TEXT, _GPT2),
    "bert": _Default("bert_base_cls", _TEXT, _BERT),
    "t5": _Default(
        "t5_small_seq2seq", {**_TEXT, "num_decoder_layers": _DEPTH, "head_dim": 16}, _T5
    ),
}


@pytest.mark.parametrize("family", sorted(_DEFAULTS))
def test_vera_with_no_names_adapts_what_the_reference_s_table_says(family: str) -> None:
    spec = _DEFAULTS[family]
    lucid.manual_seed(0)
    model = create_model(spec.factory, **spec.overrides)
    get_peft_model(model, VeraConfig(r=4))
    layers = dict(_tuner_layers(model))
    assert set(layers) == spec.adapted
    assert all(isinstance(layer, VeraLayer) for layer in layers.values())


@pytest.mark.parametrize("family", ["gpt2", "bert", "t5"])
def test_adalora_with_no_names_takes_the_lora_table(family: str) -> None:
    spec = _DEFAULTS[family]
    lucid.manual_seed(0)
    model = create_model(spec.factory, **spec.overrides)
    get_peft_model(model, AdaLoraConfig(total_step=10, init_r=2, target_r=1))
    assert set(dict(_tuner_layers(model))) == spec.adapted


@pytest.mark.parametrize("family", sorted(_DEFAULTS))
def test_vera_defaults_start_as_the_base(family: str, device: str) -> None:
    spec = _DEFAULTS[family]
    lucid.manual_seed(0)
    model = create_model(spec.factory, **spec.overrides).to(device).eval()
    if family == "t5":
        inputs = (_tokens(device, (2, 9), 1, 2), _tokens(device, (2, 6), 2, 2))
    else:
        inputs = (_tokens(device, (2, 12), 1, 2),)
    with lucid.no_grad():
        expected = model(*inputs).logits
        peft = get_peft_model(model, VeraConfig(r=4))
        assert _same(peft.eval()(*inputs).logits, expected)
