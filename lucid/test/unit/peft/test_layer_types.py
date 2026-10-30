"""The contract every LoRA layer type keeps, on CPU and Metal.

One table of models, one per layer type (``nn.Embedding``, the three
convolutions, ``nn.MultiheadAttention``, a raw ``nn.Parameter``), and the
promises the layers share, asserted for each: a zero ``B`` is the base bit
for bit, only the adapter trains, a merged model equals the unmerged one,
unloading gives the original modules back, several adapters switch exactly,
and the adapter file carries the reference's key names.  A layer type that
breaks one of them is the failure, wherever its own maths is right.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
from lucid.compile._core import signature
from lucid.peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners import lora
from lucid.serialization import load_safetensors, save_safetensors
from lucid.test._fixtures.devices import metal_available


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-4) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.head = nn.Linear(8, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.head(out)


class _Gate(nn.Module):
    """Reads a matrix and a stack of three, as a mixture of experts does."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.randn(5, 6))
        self.v = nn.Parameter(lucid.randn(3, 5, 6))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x @ self.w.mT + (x @ self.v.mT).sum(0)


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _Gate()
        self.head = nn.Linear(5, 2)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.gate(x))


class _Case(NamedTuple):
    name: str
    build: Callable[[], nn.Module]
    batch: Callable[[], lucid.Tensor]
    #: What the config adapts.
    config: dict[str, object]
    #: The adapter file's keys for the config, spelled as the reference writes them.
    keys: frozenset[str]
    #: The module types the adapter leaves behind, by the module's name.
    plain: dict[str, type[nn.Module]]


def _ids() -> lucid.Tensor:
    lucid.manual_seed(1)
    return lucid.randint(0, 12, (2, 5))


def _images(channels: int, *size: int) -> Callable[[], lucid.Tensor]:
    def batch() -> lucid.Tensor:
        lucid.manual_seed(1)
        return lucid.randn(2, channels, *size)

    return batch


def _seq(
    kind: Callable[..., nn.Module], channels: tuple[int, int, int, int], **first: object
) -> Callable[[], nn.Module]:
    """``kind`` three times with ReLUs between: a 3-wide kernel, another that strides, a 1-wide one."""

    def build() -> nn.Module:
        lucid.manual_seed(0)
        c0, c1, c2, c3 = channels
        return nn.Sequential(
            kind(c0, c1, 3, padding=1),
            nn.ReLU(),
            kind(c1, c2, 3, padding=1, **first),
            nn.ReLU(),
            kind(c2, c3, 1),
        )

    return build


def _file_keys(prefix: str, *leaves: str) -> frozenset[str]:
    return frozenset(f"base_model.model.{prefix}.{leaf}" for leaf in leaves)


_LORA = ("lora_A.weight", "lora_B.weight")

_CASES = [
    _Case(
        "embedding",
        lambda: (lucid.manual_seed(0), nn.Sequential(nn.Embedding(12, 6), nn.Linear(6, 3)))[1],
        _ids,
        {"target_modules": ["0"]},
        _file_keys("0", "lora_embedding_A", "lora_embedding_B"),
        {"0": nn.Embedding},
    ),
    _Case(
        "conv1d",
        _seq(nn.Conv1d, (3, 4, 4, 2)),
        _images(3, 9),
        {"target_modules": ["0", "2", "4"]},
        _file_keys("0", *_LORA) | _file_keys("2", *_LORA) | _file_keys("4", *_LORA),
        {"0": nn.Conv1d, "2": nn.Conv1d, "4": nn.Conv1d},
    ),
    _Case(
        "conv2d",
        _seq(nn.Conv2d, (3, 4, 4, 2), stride=2),
        _images(3, 9, 9),
        {"target_modules": ["0", "2", "4"]},
        _file_keys("0", *_LORA) | _file_keys("2", *_LORA) | _file_keys("4", *_LORA),
        {"0": nn.Conv2d, "2": nn.Conv2d, "4": nn.Conv2d},
    ),
    _Case(
        "conv3d",
        _seq(nn.Conv3d, (2, 4, 4, 2)),
        _images(2, 4, 4, 4),
        {"target_modules": ["0", "2", "4"]},
        _file_keys("0", *_LORA) | _file_keys("2", *_LORA) | _file_keys("4", *_LORA),
        {"0": nn.Conv3d, "2": nn.Conv3d, "4": nn.Conv3d},
    ),
    _Case(
        "attention",
        lambda: (lucid.manual_seed(0), _Attention())[1],
        _images(5, 8),
        {"target_modules": ["attn"]},
        _file_keys("attn", *_LORA) | _file_keys("attn.base_layer.out_proj", *_LORA),
        {"attn": nn.MultiheadAttention},
    ),
    _Case(
        "parameters",
        lambda: (lucid.manual_seed(0), _Experts())[1],
        _images(6),
        {"target_parameters": ["gate.w", "gate.v"]},
        _file_keys("gate", *_LORA) | _file_keys("gate.base_layer", *_LORA),
        {"gate": _Gate},
    ),
]

_BY_CASE = pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)


def _config(case: _Case, **kwargs: object) -> LoraConfig:
    options: dict[str, object] = {"r": 3, "lora_alpha": 6} | case.config | kwargs
    return LoraConfig(**options)  # type: ignore[arg-type]


def _peft(case: _Case, device: str, **kwargs: object) -> tuple[PeftModel, lucid.Tensor]:
    model = case.build().to(device)
    return get_peft_model(model, _config(case, **kwargs)), case.batch().to(device)


def _of(name: str, adapter: str) -> bool:
    """Whether the state-dict key ``name`` is a tensor of ``adapter`` (a table entry's key ends in it)."""
    return f".{adapter}." in name or name.endswith(f".{adapter}")


def _adapter_owned(name: str) -> bool:
    return _of(name, "default") or _of(name, "other")


def _out(model: nn.Module, x: lucid.Tensor) -> lucid.Tensor:
    out = model(x)
    assert isinstance(out, lucid.Tensor)
    return out


@_BY_CASE
def test_a_zero_b_is_the_base_bit_for_bit(case: _Case, device: str) -> None:
    model = case.build().to(device)
    x = case.batch().to(device)
    expected = _out(model, x)
    peft = get_peft_model(model, _config(case))
    assert _same(_out(peft, x), expected)


@_BY_CASE
def test_the_adapter_is_on_the_forward_path_and_alone_trains(case: _Case, device: str) -> None:
    peft, x = _peft(case, device, init_lora_weights=False)
    before = {n: p.clone() for n, p in peft.named_parameters()}
    peft.train()
    _out(peft, x).sum().backward()
    for name, param in peft.named_parameters():
        if _adapter_owned(name):
            assert param.requires_grad and param.grad is not None, name
            assert param.grad.abs().max().item() > 0, f"{name}: no gradient reaches it"
        else:
            assert not param.requires_grad and param.grad is None, name
            assert _same(param, before[name]), name


@_BY_CASE
def test_a_merged_model_equals_the_unmerged_one(case: _Case, device: str) -> None:
    peft, x = _peft(case, device, init_lora_weights=False)
    with lucid.no_grad():
        adapted = _out(peft, x)
        original = {n: p.clone() for n, p in peft.named_parameters() if not _adapter_owned(n)}
        peft.base_model.merge_adapter()
        assert any(layer.merged for layer in peft.modules() if isinstance(layer, lora.LoraLayer))
        assert _close(_out(peft, x), adapted)
        assert not all(
            _close(p, original[n], atol=0) for n, p in peft.named_parameters() if n in original
        )
        peft.base_model.unmerge_adapter()
        assert _close(_out(peft, x), adapted)
        for name, param in peft.named_parameters():
            if name in original:
                assert _close(param, original[name], atol=1e-5), name


def _identities(model: nn.Module) -> dict[str, int]:
    return {name: id(p) for name, p in model.named_parameters()}


@_BY_CASE
def test_merge_and_unload_gives_the_original_modules_back(case: _Case, device: str) -> None:
    model = case.build().to(device)
    x = case.batch().to(device)
    keys = set(model.state_dict())
    ids = _identities(model)
    peft = get_peft_model(model, _config(case, init_lora_weights=False))
    with lucid.no_grad():
        adapted = _out(peft, x)
        merged = peft.merge_and_unload()
        assert merged is model and set(merged.state_dict()) == keys
        assert _identities(merged) == ids
        for name, kind in case.plain.items():
            assert type(merged.get_submodule(name)) is kind, name
        assert _close(_out(merged, x), adapted)
        assert all(p.requires_grad for p in merged.parameters())


@_BY_CASE
def test_unload_restores_the_model_untouched(case: _Case, device: str) -> None:
    model = case.build().to(device)
    x = case.batch().to(device)
    expected = _out(model, x)
    weights = {n: p.clone() for n, p in model.named_parameters()}
    peft = get_peft_model(model, _config(case, init_lora_weights=False))
    peft.base_model.merge_adapter()
    restored = peft.unload()
    assert restored is model
    with lucid.no_grad():
        assert _close(_out(restored, x), expected, atol=1e-5)
        for name, param in restored.named_parameters():
            assert _close(param, weights[name], atol=1e-5), name
    for name, kind in case.plain.items():
        assert type(restored.get_submodule(name)) is kind, name


@_BY_CASE
def test_two_adapters_switch_exactly(case: _Case, device: str) -> None:
    peft, x = _peft(case, device, init_lora_weights=False)
    with lucid.no_grad():
        first = _out(peft, x)
        with peft.disable_adapter():
            base = _out(peft, x)
        assert not _same(first, base)
        lucid.manual_seed(5)
        peft.add_adapter("other", _config(case, r=2, lora_alpha=8, init_lora_weights=False))
        assert _same(_out(peft, x), first)  # added, not active
        peft.set_adapter("other")
        second = _out(peft, x)
        assert not _same(second, first) and not _same(second, base)
        peft.set_adapter("default")
        assert _same(_out(peft, x), first)
        peft.set_adapter("other")
        assert _same(_out(peft, x), second)
        # Both at once is neither alone.
        peft.base_model.set_adapter(["default", "other"])
        assert not _close(_out(peft, x), first) and not _close(_out(peft, x), second)
        peft.set_adapter("other")
        peft.delete_adapter("default")
        assert _same(_out(peft, x), second)
        assert not any(_of(n, "default") for n in peft.state_dict())


@_BY_CASE
def test_set_adapter_trains_exactly_the_chosen_one(case: _Case, device: str) -> None:
    peft, x = _peft(case, device, init_lora_weights=False)
    peft.add_adapter("other", _config(case, r=2, init_lora_weights=False))
    for name, param in peft.named_parameters():
        assert param.requires_grad == _of(name, "default"), name
    peft.set_adapter("other")
    for name, param in peft.named_parameters():
        assert param.requires_grad == _of(name, "other"), name
    peft.train()
    _out(peft, x).sum().backward()
    assert all(
        p.grad is None for n, p in peft.named_parameters() if _of(n, "default")
    ), "the idle adapter received a gradient"


@_BY_CASE
def test_the_adapter_file_uses_the_reference_s_keys(
    case: _Case, device: str, tmp_path: Path
) -> None:
    peft, x = _peft(case, device, init_lora_weights=False)
    assert set(get_peft_model_state_dict(peft)) == case.keys
    peft.eval().save_pretrained(tmp_path)
    assert set(load_safetensors(str(tmp_path / "adapter_model.safetensors"))) == case.keys
    fresh = PeftModel.from_pretrained(case.build().to(device), tmp_path)
    with lucid.no_grad():
        assert _same(_out(fresh, x), _out(peft, x))


@_BY_CASE
def test_the_compile_key_sees_the_active_adapter(case: _Case) -> None:
    peft, _ = _peft(case, "cpu", init_lora_weights=False)
    peft.add_adapter("other", _config(case, r=2, init_lora_weights=False))
    peft.set_adapter("default")
    first = signature._module_state(peft)
    peft.set_adapter("other")
    second = signature._module_state(peft)
    assert first != second
    peft.set_adapter("default")
    assert signature._module_state(peft) == first
    peft.base_model.disable_adapter_layers()
    assert signature._module_state(peft) not in (first, second)


# ── an adapter file in the reference's layout, written by hand ───────────────────────


def _write_adapter(path: Path, config: dict[str, object], tensors: dict[str, lucid.Tensor]) -> None:
    save_safetensors(tensors, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
    (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", **config}))


def _reference_config(**fields: object) -> dict[str, object]:
    return {
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "init_lora_weights": True,
        "use_rslora": False,
        "use_dora": False,
        "lora_bias": False,
        "task_type": None,
        "inference_mode": True,
        "target_parameters": None,
        **fields,
    }


def test_an_embedding_adapter_written_by_the_reference_loads(device: str, tmp_path: Path) -> None:
    lucid.manual_seed(3)
    a, b = lucid.randn(2, 12), lucid.randn(6, 2)  # (r, num_embeddings), (embedding_dim, r)
    _write_adapter(
        tmp_path,
        _reference_config(target_modules=["0"]),
        {"base_model.model.0.lora_embedding_A": a, "base_model.model.0.lora_embedding_B": b},
    )
    model = _CASES[0].build().to(device)
    ids = _ids().to(device)
    plain = _out(model, ids)
    loaded = PeftModel.from_pretrained(model, tmp_path)
    # y = head(E[x] + (alpha / r) (B A)^T [x]), the reference's formula from its own tensors.
    table = (b @ a).mT.to(device) * (4 / 2)
    base = loaded.get_base_model()
    expected = base[1](base[0].base_layer(ids) + table[ids])
    with lucid.no_grad():
        assert _close(_out(loaded, ids), expected, atol=1e-5)
        assert not _close(_out(loaded, ids), plain)


def test_a_conv_adapter_written_by_the_reference_loads(device: str, tmp_path: Path) -> None:
    lucid.manual_seed(3)
    a, b = lucid.randn(2, 3, 3, 3), lucid.randn(4, 2, 1, 1)
    _write_adapter(
        tmp_path,
        _reference_config(target_modules=["0"]),
        {"base_model.model.0.lora_A.weight": a, "base_model.model.0.lora_B.weight": b},
    )
    conv = nn.Conv2d(3, 4, 3, padding=1)
    loaded = PeftModel.from_pretrained(nn.Sequential(conv.to(device)), tmp_path)
    x = lucid.randn(2, 3, 5, 5).to(device)
    # Two convolutions in a row, scaled alpha / r, from the reference's own tensors.
    import lucid.nn.functional as F

    expected = conv(x) + F.conv2d(F.conv2d(x, a.to(device), padding=1), b.to(device)) * 2.0
    with lucid.no_grad():
        assert _close(_out(loaded, x), expected, atol=1e-4)


def test_an_attention_adapter_written_by_the_reference_loads(device: str, tmp_path: Path) -> None:
    lucid.manual_seed(3)
    e = 8
    tensors = {
        "base_model.model.attn.lora_A.weight": lucid.randn(2, e),
        "base_model.model.attn.lora_B.weight": lucid.randn(3 * e, 2),
        "base_model.model.attn.base_layer.out_proj.lora_A.weight": lucid.randn(2, e),
        "base_model.model.attn.base_layer.out_proj.lora_B.weight": lucid.randn(e, 2),
    }
    _write_adapter(tmp_path, _reference_config(target_modules=["attn"]), tensors)
    model = _Attention().to(device)
    loaded = PeftModel.from_pretrained(model, tmp_path)
    # The reference's maths for this layer: the weights are shifted by alpha / r (B A).
    mha = loaded.get_base_model().attn.base_layer
    shifted = nn.MultiheadAttention(e, 2, batch_first=True).to(device)
    shifted.load_state_dict(mha.state_dict())
    t = {k: v.to(device) for k, v in tensors.items()}
    with lucid.no_grad():
        shifted.in_proj_weight.add_(
            (t["base_model.model.attn.lora_B.weight"] @ t["base_model.model.attn.lora_A.weight"])
            * 2.0
        )
        shifted.out_proj_weight.add_(
            (
                t["base_model.model.attn.base_layer.out_proj.lora_B.weight"]
                @ t["base_model.model.attn.base_layer.out_proj.lora_A.weight"]
            )
            * 2.0
        )
    x = lucid.randn(2, 5, e).to(device)
    with lucid.no_grad():
        expected = model.head(shifted(x, x, x, need_weights=False)[0])
        assert _close(_out(loaded, x), expected, atol=1e-5)


def test_a_parameter_adapter_written_by_the_reference_loads(device: str, tmp_path: Path) -> None:
    lucid.manual_seed(3)
    tensors = {  # the wrapper of the module's last parameter is the outermost, as the reference nests them
        "base_model.model.gate.lora_A.weight": lucid.randn(2 * 3, 6),  # v: (experts * r, in)
        "base_model.model.gate.lora_B.weight": lucid.randn(5, 2 * 3),  # v: (out, r * experts)
        "base_model.model.gate.base_layer.lora_A.weight": lucid.randn(2, 6),  # w: (r, in)
        "base_model.model.gate.base_layer.lora_B.weight": lucid.randn(5, 2),  # w: (out, r)
    }
    _write_adapter(tmp_path, _reference_config(target_parameters=["gate.w", "gate.v"]), tensors)
    model = _Experts().to(device)
    x = lucid.randn(4, 6).to(device)
    loaded = PeftModel.from_pretrained(model, tmp_path)
    t = {k: v.to(device) for k, v in tensors.items()}
    gate = model.gate
    # ``v`` is (experts, out, in); the reference reads A as (experts, r, in) and B as (out, r, experts).
    a = t["base_model.model.gate.lora_A.weight"].reshape(3, 2, 6)
    b = t["base_model.model.gate.lora_B.weight"].reshape(5, 2, 3)
    delta_v = lucid.stack([b[:, :, e] @ a[e] for e in range(3)]) * 2.0
    delta_w = (
        t["base_model.model.gate.base_layer.lora_B.weight"]
        @ t["base_model.model.gate.base_layer.lora_A.weight"]
    ) * 2.0
    with lucid.no_grad():
        w, v = gate.w, gate.v
        expected = model.head(x @ (w + delta_w).mT + (x @ (v + delta_v).mT).sum(0))
        assert _close(_out(loaded, x), expected, atol=1e-4)
    assert loaded.base_model.targeted_parameter_names == ["gate.w", "gate.v"]


# ── compiled forwards follow the adapter ──────────────────────────────────────────────


metal_only = pytest.mark.skipif(not metal_available(), reason="compiled executables run on Metal")


@metal_only
@pytest.mark.parametrize("name", [c.name for c in _CASES])
def test_a_compiled_forward_follows_set_adapter(name: str) -> None:
    case = next(c for c in _CASES if c.name == name)
    peft, x = _peft(case, "metal", init_lora_weights=False)
    peft.add_adapter("other", _config(case, r=2, lora_alpha=8, init_lora_weights=False))
    compiled = lucid.compile(peft)
    with lucid.no_grad():
        seen = []
        for adapter in ("default", "other", "default", "other"):
            peft.set_adapter(adapter)
            eager = _out(peft, x)
            seen.append(eager)
            assert _close(compiled(x), eager, atol=1e-5), adapter
        assert not _close(seen[0], seen[1])
        peft.base_model.merge_adapter()
        assert _close(compiled(x), seen[1], atol=1e-4)
        peft.base_model.unmerge_adapter()
        with peft.disable_adapter():
            assert not _close(compiled(x), seen[1])
        assert _close(compiled(x), seen[1], atol=1e-5)
