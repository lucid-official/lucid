"""LoRA on a zoo GPT-2: trains, generates, merges, and saves under GPT-2's adapter key names.

Built with the registered ``gpt2_small_lm`` factory and a small config
override (H11), so the path is the one a user takes.
"""

from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.models import create_model
from lucid.peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict

_TINY = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 32,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}


def _gpt2(device: str) -> nn.Module:
    lucid.manual_seed(0)
    return create_model("gpt2_small_lm", **_TINY).to(device)  # type: ignore[arg-type]


def _ids(device: str) -> lucid.Tensor:
    lucid.manual_seed(1)
    return lucid.randint(0, 64, (2, 12), device=device)


def _peft(device: str, **kwargs: object) -> PeftModel:
    fields: dict[str, object] = {
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["c_attn"],
        "task_type": "CAUSAL_LM",
    } | kwargs
    model = _gpt2(device)
    return get_peft_model(model, LoraConfig(**fields))  # type: ignore[arg-type]


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def test_three_sgd_steps_lower_the_loss_and_move_only_the_adapter(device: str) -> None:
    peft = _peft(device)
    peft.train()
    ids = _ids(device)
    before = {name: p.clone() for name, p in peft.named_parameters()}
    opt = optim.SGD([p for p in peft.parameters() if p.requires_grad], lr=0.5)
    losses = []
    for _ in range(3):
        opt.zero_grad()
        loss = peft(ids, labels=ids).loss
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0], losses
    for name, param in peft.named_parameters():
        moved = not _same(param, before[name])
        assert moved == (".lora_" in name), name


def test_generate_runs_through_the_wrapper(device: str) -> None:
    peft = _peft(device, init_lora_weights=False).eval()
    prompt = _ids(device)[:, :4]
    with lucid.no_grad():
        out = peft.generate(prompt, max_new_tokens=3)
        base = peft.get_base_model()
        with peft.disable_adapter():
            plain = peft.generate(prompt, max_new_tokens=3)
    assert tuple(out.shape) == (2, 7) and tuple(plain.shape) == (2, 7)
    assert _same(out[:, :4], prompt) and base.config.vocab_size == 64


def test_all_linear_leaves_the_tied_head(device: str) -> None:
    peft = _peft(device, target_modules="all-linear")
    names = set(peft.base_model.targeted_module_names)
    assert "lm_head" not in names
    assert names == {
        f"transformer.h.{i}.{part}"
        for i in range(2)
        for part in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    }


def test_merge_and_unload_matches_the_adapted_logits(device: str) -> None:
    peft = _peft(device, init_lora_weights=False).eval()
    ids = _ids(device)
    with lucid.no_grad():
        adapted = peft(ids).logits
        merged = peft.merge_and_unload()
        assert bool(lucid.allclose(merged(ids).logits, adapted, atol=1e-4))


def test_the_adapter_file_has_gpt2_s_key_names(tmp_path: Path) -> None:
    peft = _peft("cpu")
    assert sorted(get_peft_model_state_dict(peft)) == [
        f"base_model.model.transformer.h.{i}.attn.c_attn.lora_{m}.weight"
        for i in range(2)
        for m in ("A", "B")
    ]
    peft.save_pretrained(tmp_path)
    loaded = PeftModel.from_pretrained(_gpt2("cpu"), tmp_path)
    ids = _ids("cpu")
    with lucid.no_grad():
        assert _same(loaded(ids).logits, peft.eval()(ids).logits)


@pytest.mark.parametrize("target", [["lm_head"]])
def test_merging_into_the_tied_head_is_refused(target: list[str]) -> None:
    peft = _peft("cpu", target_modules=target)
    with pytest.raises(RuntimeError, match="transformer.wte.weight"):
        peft.merge_and_unload()
