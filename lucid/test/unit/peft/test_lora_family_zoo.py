"""VB-LoRA, RandLoRA, MiSS, Bone, SHiRA, C3A and trainable tokens on the model zoo, end to end (tiny configs, CPU and Metal).

Each method on GPT-2 (causal LM; its ``c_attn`` projection), BERT (sequence classification; its
``query`` and ``value``) and ViT (its ``qkv``; the classifier head is trained beside the adapter):
the adapted model starts as the base bit for bit (VB-LoRA's factors are non-zero from the first
step, so it does not), a few optimizer steps move the adapter and nothing else and lower
the loss, an adapter file reloads to the same logits, and merging gives the same logits.
Trainable tokens run on the untied GPT-2 and BERT embedding tables.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import (
    BoneConfig,
    C3AConfig,
    MissConfig,
    PeftConfig,
    PeftModel,
    RandLoraConfig,
    ShiraConfig,
    TrainableTokensConfig,
    VBLoRAConfig,
    get_peft_model,
)
from lucid.peft.tuners._base import _tuner_layers
from lucid.test.unit.peft.test_new_tuners_zoo import (
    _TEXT,
    _VIT,
    _Batch,
    _image_batch,
    _lm_batch,
    _same,
    _tokens,
    _train,
)


def _cls_batch(device: str) -> _Batch:
    return _Batch((_tokens(device, (2, 12), seed=1),), lucid.tensor([0, 1], device=device))


@dataclass(frozen=True)
class _Row:
    name: str
    factory: str
    overrides: dict[str, object]
    batch: Callable[[str], _Batch]
    task_type: str | None
    saved: list[str] | None
    #: The module the attention projections are called here.
    target: list[str]


_ROWS = [
    _Row("gpt2", "gpt2_small_lm", _TEXT, _lm_batch, "CAUSAL_LM", None, ["c_attn"]),
    _Row("bert", "bert_base_cls", _TEXT, _cls_batch, "SEQ_CLS", None, ["query", "value"]),
    _Row("vit", "vit_base_16_cls", _VIT, _image_batch, None, ["head"], ["qkv"]),
]
_METHODS = ("miss", "bone", "randlora", "vblora", "shira", "c3a")
_CASES = [(m, row) for m in _METHODS for row in _ROWS]


def _config(method: str, row: _Row, random: bool = False) -> PeftConfig:
    common: dict[str, object] = {
        "target_modules": row.target,
        "modules_to_save": row.saved,
        "task_type": row.task_type,
    }
    # Every width of the three rows' projections is a multiple of 16 (32 and 96).
    makers: dict[str, Callable[[], PeftConfig]] = {
        "miss": lambda: MissConfig(r=4, init_weights=not random, **common),  # type: ignore[arg-type]
        "bone": lambda: BoneConfig(r=4, init_weights=not random, **common),  # type: ignore[arg-type]
        "randlora": lambda: RandLoraConfig(r=8, randlora_alpha=32, init_weights=not random, **common),  # type: ignore[arg-type]
        "vblora": lambda: VBLoRAConfig(r=4, num_vectors=16, vector_length=16, topk=2, init_vector_bank_bound=0.2, **common),  # type: ignore[arg-type]
        "shira": lambda: ShiraConfig(r=2, random_seed=5, init_weights=not random, **common),  # type: ignore[arg-type]
        "c3a": lambda: C3AConfig(block_size=16, init_weights=not random, **common),  # type: ignore[arg-type]
    }
    return makers[method]()


def _base(row: _Row, device: str) -> nn.Module:
    lucid.manual_seed(0)
    model = create_model(row.factory, **row.overrides)
    for name in ("head", "classifier"):
        head = getattr(model, name, None)
        if isinstance(head, nn.Linear):
            # A zero-initialised head holds every logit at zero.
            nn.init.normal_(head.weight, std=0.2)
    return model.to(device)


def _logits(model: nn.Module, batch: _Batch) -> lucid.Tensor:
    return model(*batch.inputs).logits  # type: ignore[no-any-return]


_BY_CASE = pytest.mark.parametrize(
    ("method", "row"), _CASES, ids=[f"{m}-{r.name}" for m, r in _CASES]
)


@_BY_CASE
def test_the_adapted_model_starts_as_the_base(method: str, row: _Row, device: str) -> None:
    base = _base(row, device)
    batch = row.batch(device)
    expected = _logits(base, batch)
    peft = get_peft_model(base, _config(method, row))
    out = _logits(peft.eval(), batch)
    if method == "vblora":
        assert not _same(out, expected)  # both factors are non-zero from the first step
    else:
        assert _same(out, expected)


@_BY_CASE
def test_training_moves_the_adapter_alone_and_lowers_the_loss(
    method: str, row: _Row, device: str
) -> None:
    peft = get_peft_model(_base(row, device), _config(method, row))
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
    method: str, row: _Row, device: str, tmp_path: Path
) -> None:
    peft = get_peft_model(_base(row, device), _config(method, row, random=True))
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
def test_merging_gives_the_same_logits(method: str, row: _Row, device: str) -> None:
    peft = get_peft_model(_base(row, device), _config(method, row, random=True))
    batch = row.batch(device)
    _train(peft, batch)
    peft.eval()
    with lucid.no_grad():
        expected = _logits(peft, batch)
        merged = peft.merge_and_unload()
        assert not any(True for _ in _tuner_layers(merged))
        assert bool(lucid.allclose(_logits(merged, batch), expected, atol=1e-3))


# ── trainable tokens ─────────────────────────────────────────────────────────

_TOKEN_ROWS = [
    ("gpt2", "gpt2_small_lm", {**_TEXT, "tie_word_embeddings": False}, "transformer.wte"),
    ("bert", "bert_base_cls", _TEXT, "bert.embeddings.word_embeddings"),
]


@pytest.mark.parametrize(
    ("name", "factory", "overrides", "table"), _TOKEN_ROWS, ids=lambda v: str(v)[:12]
)
def test_trainable_tokens_train_a_few_rows_of_the_table(
    name: str, factory: str, overrides: dict[str, object], table: str, device: str
) -> None:
    del name
    lucid.manual_seed(0)
    base = create_model(factory, **overrides).to(device)
    batch = _lm_batch(device) if factory.startswith("gpt2") else _cls_batch(device)
    expected = _logits(base, batch)
    config = TrainableTokensConfig(token_indices=[3, 4, 5], target_modules=[table])
    peft = get_peft_model(base, config)
    assert _same(_logits(peft.eval(), batch), expected)
    layer = dict(_tuner_layers(peft))[f"base_model.model.{table}"]
    rows = layer.trainable_tokens_delta["default"]
    assert tuple(rows.shape) == (3, int(overrides["hidden_size"]))  # type: ignore[call-overload]
    trained = [n for n, p in peft.named_parameters() if p.requires_grad]
    assert all("trainable_tokens_delta" in n for n in trained)
    before = rows.clone()
    peft.train()
    opt = lucid.optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=3e-2)
    for _ in range(2):
        opt.zero_grad()
        out = peft(*batch.inputs, labels=batch.labels)
        out.loss.backward()
        opt.step()
    assert not _same(rows, before)
    peft.eval()
    with lucid.no_grad():
        trained_logits = _logits(peft, batch)
        merged = peft.merge_and_unload()
        assert bool(lucid.allclose(_logits(merged, batch), trained_logits, atol=1e-4))
