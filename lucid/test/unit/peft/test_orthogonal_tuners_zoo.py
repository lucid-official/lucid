"""OFT, BOFT, HRA, FourierFT and LN tuning on the model zoo, end to end (tiny configs, CPU and Metal).

Each tuner on BERT (``nn.Linear`` projections, ``LayerNorm``), GPT-2 (``nn.Linear``
projections, ``LayerNorm``) and ViT (``nn.Linear`` and a ``nn.Conv2d`` patch embedding).
Per row: the adapted model starts as the base (to rounding for the products of
reflections and butterflies), a few optimizer steps move the adapter and nothing else
and lower the loss, an adapter file reloads to the same logits, and merging gives the
same logits from the plain model.
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
    BOFTConfig,
    FourierFTConfig,
    HRAConfig,
    LNTuningConfig,
    OFTConfig,
    PeftConfig,
    PeftModel,
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


def _tokens(device: str, shape: tuple[int, ...], seed: int, high: int = 64) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randint(0, high, shape, device=device)


def _lm_batch(device: str) -> _Batch:
    ids = _tokens(device, (2, 12), seed=1)
    return _Batch((ids,), ids)


def _classes_batch(device: str) -> _Batch:
    return _Batch((_tokens(device, (4, 12), seed=1),), _tokens(device, (4,), 2, 3))


def _image_batch(device: str) -> _Batch:
    lucid.manual_seed(1)
    return _Batch((lucid.randn(3, 3, 16, 16, device=device),), _tokens(device, (3,), 2, 5))


@dataclass(frozen=True)
class _Row:
    name: str
    factory: str
    overrides: dict[str, object]
    batch: Callable[[str], _Batch]
    task_type: str | None
    saved: list[str] | None
    #: The ``target_modules`` each tuner is asked for on this row.
    targets: dict[str, list[str]]


def _on(*names: str, norms: list[str]) -> dict[str, list[str]]:
    """``names`` for the four methods that adapt weights, ``norms`` for LN tuning."""
    return {t: list(names) for t in ("oft", "boft", "hra", "fourierft")} | {"ln_tuning": norms}


_ROWS = [
    _Row(
        "bert",
        "bert_base_cls",
        {**_TEXT, "num_labels": 3},
        _classes_batch,
        "SEQ_CLS",
        ["classifier"],
        _on("query", "value", norms=["LayerNorm"]),
    ),
    _Row(
        "gpt2",
        "gpt2_small_lm",
        _TEXT,
        _lm_batch,
        "CAUSAL_LM",
        None,
        _on("c_attn", norms=["ln_1", "ln_2", "ln_f"]),
    ),
    _Row(
        "vit",
        "vit_base_16_cls",
        _VIT,
        _image_batch,
        None,
        ["head"],
        _on("qkv", norms=["norm1", "norm2", "norm"]),
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
        {t: ["patch_embed.proj"] for t in ("oft", "boft", "hra")},
    ),
]
_TUNERS = ("oft", "boft", "hra", "fourierft", "ln_tuning")
_CASES = [(t, row) for row in _ROWS for t in _TUNERS if t in row.targets]
_BY_CASE = pytest.mark.parametrize(
    ("tuner", "row"), _CASES, ids=[f"{t}-{r.name}" for t, r in _CASES]
)
#: A product of reflections or of butterfly factors is the identity up to rounding only.
_ROUNDED_START = ("boft", "hra")


def _config(tuner: str, row: _Row, random: bool = False) -> PeftConfig:
    common: dict[str, object] = {
        "target_modules": row.targets[tuner],
        "modules_to_save": row.saved,
        "task_type": row.task_type,
    }
    if tuner == "oft":
        return OFTConfig(oft_block_size=4, init_weights=not random, **common)  # type: ignore[arg-type]
    if tuner == "boft":
        return BOFTConfig(boft_block_size=4, init_weights=not random, **common)  # type: ignore[arg-type]
    if tuner == "hra":
        return HRAConfig(r=4, init_weights=not random, **common)  # type: ignore[arg-type]
    if tuner == "fourierft":
        return FourierFTConfig(n_frequency=40, scaling=5.0, init_weights=not random, **common)  # type: ignore[arg-type]
    return LNTuningConfig(**common)  # type: ignore[arg-type]


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


@_BY_CASE
def test_the_adapted_model_starts_as_the_base(tuner: str, row: _Row, device: str) -> None:
    base = _base(row, device).eval()
    batch = row.batch(device)
    with lucid.no_grad():
        expected = _logits(base, batch)
        got = _logits(get_peft_model(base, _config(tuner, row)).eval(), batch)
    if tuner in _ROUNDED_START:
        assert bool(lucid.allclose(got, expected, atol=1e-5))
    else:
        assert _same(got, expected)


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
