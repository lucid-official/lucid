"""Text heads pick their pooled token with tensor ops, not host reads (LCD-363).

``gpt_cls``, ``gpt2_small_cls`` and ``roformer_multiple_choice`` pool the
last position the padding mask keeps; the GPT and GPT-2 double-heads
models pool the position ``mc_token_ids`` names.  All of them used to read
the mask or the ids with ``.tolist()`` / ``.item()`` and index in Python,
which a compiled or exported graph replays from the traced call — so a
fresh mask got the old answer, and ``lucid.compile`` ran them eagerly.

Here the old selection is written out inline and compared with the
model's logits for every mask shape that selection had to handle, on
every device; the compile check proves no host read is left.
"""

import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.models.text.gpt import GPTConfig, GPTDoubleHeadsModel
from lucid.models.text.gpt2 import GPT2Config, GPT2DoubleHeadsModel
from lucid.test._fixtures.devices import metal_available

DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]

TINY = {
    "num_hidden_layers": 1,
    "hidden_size": 32,
    "num_attention_heads": 2,
    "intermediate_size": 64,
}

HEADS = ["gpt_cls", "gpt2_small_cls", "roformer_multiple_choice"]

L = 8

#: Every mask shape the old selection had a rule for, one row each.
MASK_ROWS = {
    "left_padding": [0, 0, 0, 1, 1, 1, 1, 1],
    "right_padding": [1, 1, 1, 1, 1, 0, 0, 0],
    "all_ones": [1] * L,
    "single_first": [1] + [0] * (L - 1),
    "single_last": [0] * (L - 1) + [1],
    "single_middle": [0, 0, 0, 1, 0, 0, 0, 0],
    "all_zeros": [0] * L,
}


def _old_last_index(mask: Tensor) -> list[int]:
    """The selection the heads made before LCD-363, read on the host."""
    out = []
    for row in mask.float().tolist():
        kept = [i for i, v in enumerate(row) if v > 0.5]
        out.append(kept[-1] if kept else 0)
    return out


def _old_pool(hidden: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return hidden[:, -1, :]
    idx = _old_last_index(mask)
    return lucid.stack([hidden[b, idx[b], :] for b in range(len(idx))], dim=0)


def _random_mask(rows: int, seed: int) -> Tensor:
    """Random left- or right-padded rows with at least one kept token."""
    lucid.manual_seed(seed)
    lengths = lucid.randint(1, L + 1, (rows,)).tolist()
    sides = lucid.randint(0, 2, (rows,)).tolist()
    out = []
    for n, left in zip(lengths, sides, strict=True):
        row = [1] * n + [0] * (L - n)
        out.append(row[::-1] if left else row)
    return lucid.tensor(out).long()


def _masks() -> dict[str, Tensor]:
    fixed = lucid.tensor(list(MASK_ROWS.values())).long()
    return {"fixed": fixed, "random": _random_mask(8, seed=3)}


def _model(name: str, device: str) -> nn.Module:
    lucid.manual_seed(0)
    return M.create_model(name, **TINY).eval().to(device)


def _ids_and_mask(name: str, mask: Tensor, device: str) -> tuple[Tensor, Tensor]:
    rows = int(mask.shape[0])
    lucid.manual_seed(1)
    if name.startswith("roformer"):
        # (N, C, L): split the rows across two choices.
        pad = rows % 2
        if pad:
            mask = lucid.cat([mask, mask[:1]], dim=0)
            rows += 1
        ids = lucid.randint(0, 100, (rows // 2, 2, L))
        mask = mask.reshape(rows // 2, 2, L)
    else:
        ids = lucid.randint(0, 100, (rows, L))
    return ids.to(device), mask.to(device)


def _old_logits(model: nn.Module, name: str, ids: Tensor, mask: Tensor | None) -> Tensor:
    if name == "roformer_multiple_choice":
        N, C = int(ids.shape[0]), int(ids.shape[1])
        flat_mask = None if mask is None else mask.reshape(N * C, L)
        hidden = model.roformer(ids.reshape(N * C, L), attention_mask=flat_mask)
        pooled = _old_pool(hidden.last_hidden_state, flat_mask)
        return model.classifier(model.dropout(pooled)).reshape(N, C)
    hidden = model.transformer(ids, attention_mask=mask).last_hidden_state
    return model.classifier(model.dropout(_old_pool(hidden, mask)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("masks", ["fixed", "random"])
@pytest.mark.parametrize("name", HEADS)
def test_pooled_token_matches_the_host_read_selection(name: str, masks: str, device: str) -> None:
    model = _model(name, device)
    ids, mask = _ids_and_mask(name, _masks()[masks], device)
    new = model(ids, mask).logits
    old = _old_logits(model, name, ids, mask)
    assert tuple(new.shape) == tuple(old.shape)
    assert float((new - old).abs().max().item()) == 0.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", HEADS)
def test_no_mask_still_pools_the_last_position(name: str, device: str) -> None:
    model = _model(name, device)
    ids, _mask = _ids_and_mask(name, _masks()["fixed"], device)
    new = model(ids).logits
    old = _old_logits(model, name, ids, None)
    assert float((new - old).abs().max().item()) == 0.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("model_cls", "config_cls"),
    [(GPTDoubleHeadsModel, GPTConfig), (GPT2DoubleHeadsModel, GPT2Config)],
    ids=["gpt", "gpt2"],
)
def test_double_heads_pool_the_named_token(model_cls: type, config_cls: type, device: str) -> None:
    lucid.manual_seed(0)
    model = model_cls(config_cls(vocab_size=100, **TINY)).eval().to(device)
    ids = lucid.randint(0, 100, (2, 3, L)).to(device)
    mc = lucid.tensor([[0, 7, 3], [5, 1, 7]]).long().to(device)
    new = model(ids, mc_token_ids=mc).mc_logits

    hidden = model.transformer(ids.reshape(6, L)).last_hidden_state.reshape(2, 3, L, -1)
    rows = [
        hidden[n, c, t : t + 1, :] for n, row in enumerate(mc.tolist()) for c, t in enumerate(row)
    ]
    pooled = lucid.cat(rows, dim=0).reshape(2, 3, -1)
    old = model.mc_head.summary(pooled).reshape(2, 3)
    assert float((new - old).abs().max().item()) == 0.0


@pytest.mark.parametrize("name", HEADS)
def test_a_training_step_reaches_the_trunk(name: str) -> None:
    """The gather keeps the pooled token on the graph: one optimiser step
    with a padded mask moves the trunk's embedding, not only the head."""
    model = M.create_model(name, **TINY).train()
    ids, mask = _ids_and_mask(name, _masks()["fixed"], "cpu")
    labels = lucid.zeros(int(ids.shape[0])).long()
    table = next(m for m in model.modules() if isinstance(m, nn.Embedding)).weight
    before = table.detach().clone()
    opt = lucid.optim.SGD(model.parameters(), lr=0.5)
    out = model(ids, mask, labels=labels)
    opt.zero_grad()
    out.loss.backward()
    opt.step()
    assert float((table.detach() - before).abs().max().item()) > 0.0


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("name", HEADS)
def test_compile_does_not_fall_back_to_eager(name: str) -> None:
    model = _model(name, "metal")
    compiled = lucid.compile.compile(model)
    ids, mask_a = _ids_and_mask(name, _masks()["fixed"], "metal")
    _ids, mask_b = _ids_and_mask(name, _masks()["random"], "metal")
    mask_b = mask_b[: int(mask_a.shape[0])]
    # Under grad mode a trainable model's compiled call runs eagerly (LCD-582).
    with lucid.no_grad():
        compiled(ids, mask_a)
        got = compiled(ids, mask_b)
    info = compiled.cache_info()
    assert not info["eager_only"], info
    assert info["entries"] == 1, info
    want = model(ids, mask_b).logits
    assert float((got.logits - want).abs().max().item()) <= 1e-5


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize(
    ("model_cls", "config_cls"),
    [(GPTDoubleHeadsModel, GPTConfig), (GPT2DoubleHeadsModel, GPT2Config)],
    ids=["gpt", "gpt2"],
)
def test_double_heads_compile_without_eager_fallback(model_cls: type, config_cls: type) -> None:
    lucid.manual_seed(0)
    model = model_cls(config_cls(vocab_size=100, **TINY)).eval().to("metal")
    compiled = lucid.compile.compile(model)
    ids = lucid.randint(0, 100, (2, 3, L)).to("metal")
    first = lucid.tensor([[0, 7, 3], [5, 1, 7]]).long().to("metal")
    fresh = lucid.tensor([[7, 2, 0], [1, 6, 4]]).long().to("metal")
    with lucid.no_grad():
        compiled(ids, first)
        got = compiled(ids, fresh)
    info = compiled.cache_info()
    assert not info["eager_only"], info
    want = model(ids, mc_token_ids=fresh).mc_logits
    assert float((got.mc_logits - want).abs().max().item()) <= 1e-5
