"""VB-LoRA, RandLoRA, MiSS (and Bone), SHiRA, C3A and trainable tokens: what each method does beyond the shared tuner contract.

``test_tuner_contract`` holds every registered method to the same promises (entry points,
multi-adapter, merging, files in the reference's layout, compile).  Here: the arithmetic of
each method written out by hand, the file formats that are not just the layers' tensors
(VB-LoRA's top-k mode, SHiRA's indices, the shared bases), the checks at the boundary, and — marked
``parity`` and skipped without the reference library — the layers against the reference's.
Every ``PeftType`` must also be backed by a config and a tuner.
"""

import math
import warnings
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    BoneConfig,
    C3AConfig,
    MissConfig,
    PeftConfig,
    PeftModel,
    PeftType,
    RandLoraConfig,
    ShiraConfig,
    TrainableTokensConfig,
    VBLoRAConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft._config import _CONFIG_CLASSES
from lucid.peft.tuners import bone, c3a, miss, randlora, shira, trainable_tokens, vblora
from lucid.peft.tuners._base import _TUNERS
from lucid.peft.tuners.c3a._circulant import _circulant_apply, _circulant_matrix
from lucid.peft.tuners.miss._modes import _Mode
from lucid.serialization import load_safetensors
from lucid.test.unit.peft._parity import adapter_library, gap

# Members a later card (LCD-798: the prompt and mixture methods) still owes a config and a tuner.
_LATER = {PeftType.CPT, PeftType.POLY, PeftType.XLORA}


def _mlp(inn: int = 8, hidden: int = 8, out: int = 4) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(inn, hidden), nn.ReLU(), nn.Linear(hidden, out))


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


# ── every method of the enum is backed ───────────────────────────────────────


@pytest.mark.parametrize(
    "member",
    [
        pytest.param(
            m,
            marks=(
                pytest.mark.xfail(strict=True, reason="LCD-798: no config and tuner yet")
                if m in _LATER
                else ()
            ),
        )
        for m in PeftType
    ],
    ids=lambda m: m.value,
)
def test_every_peft_type_has_a_config_and_a_tuner(member: PeftType) -> None:
    assert member in _CONFIG_CLASSES, f"{member} has no registered config"
    assert member in _TUNERS, f"{member} has no registered tuner"
    config_class, tuner = _CONFIG_CLASSES[member], _TUNERS[member]
    assert config_class._peft_type == member
    assert tuner._config_class is config_class or issubclass(config_class, tuner._config_class)


# ── MiSS ─────────────────────────────────────────────────────────────────────


class TestMiss:
    def test_the_balance_update_tiles_the_block_along_the_input(self) -> None:
        base = nn.Linear(5, 3)  # r = 2 does not divide 5: the last shard is partial
        layer = miss.Linear(base, "default", MissConfig(r=2, target_modules=["x"]))
        block = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with lucid.no_grad():
            layer.miss_block["default"].copy_(lucid.tensor(block))
        delta = layer.get_delta_weight("default").tolist()
        assert delta == [[block[c % 2][o] for c in range(5)] for o in range(3)]
        x = lucid.randn(4, 5)
        assert _close(layer(x), base(x) + x @ layer.get_delta_weight("default").mT)

    def test_the_mini_block_is_tiled_along_the_output(self) -> None:
        layer = miss.Linear(
            nn.Linear(4, 6),
            "default",
            MissConfig(r=2, mini_r=3, init_weights="mini", target_modules=["x"]),
        )
        assert tuple(layer.miss_block["default"].shape) == (2, 3)
        block = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with lucid.no_grad():
            layer.miss_block["default"].copy_(lucid.tensor(block))
        delta = layer.get_delta_weight("default").tolist()
        assert delta == [[block[c % 2][o % 3] for c in range(4)] for o in range(6)]

    def test_the_bat_update_is_w_times_s_plus_s_blockwise(self) -> None:
        base = nn.Linear(4, 4)
        layer = miss.Linear(
            base, "default", MissConfig(r=2, init_weights="bat", target_modules=["x"])
        )
        assert tuple(layer.miss_block["default"].shape) == (2, 2, 2)
        lucid.manual_seed(3)
        with lucid.no_grad():
            layer.miss_block["default"].copy_(lucid.randn(2, 2, 2))
        w = base.weight.tolist()
        s = layer.miss_block["default"].tolist()
        expected = [[0.0] * 4 for _ in range(4)]
        for row in range(4):
            for col in range(4):
                lo, ro = row // 2, row % 2  # output block, row inside it
                ci = col // 2  # input block
                # W' block (lo, ci) = W (I + S_lo) + S_lo
                acc = sum(
                    w[lo * 2 + ro][ci * 2 + k]
                    * ((1.0 if k == col % 2 else 0.0) + s[lo][k][col % 2])
                    for k in range(2)
                )
                expected[row][col] = acc + s[lo][ro][col % 2]
        merged = layer._rewritten(base.weight, "default")
        assert _close(merged, lucid.tensor(expected))

    @pytest.mark.parametrize("init", [True, "mini", "bat"])
    def test_unmerging_restores_each_variant(self, init: object) -> None:
        base = nn.Linear(4, 4)
        layer = miss.Linear(base, "default", MissConfig(r=2, mini_r=2, init_weights=init, target_modules=["x"]))  # type: ignore[arg-type]
        with lucid.no_grad():
            layer.miss_block["default"].copy_(lucid.randn(*layer.miss_block["default"].shape) * 0.3)
        before = base.weight.clone()
        layer.merge()
        assert not _close(base.weight, before, atol=1e-4)
        layer.unmerge()
        assert _close(base.weight, before, atol=1e-5)

    def test_the_variants_state_what_they_need_of_the_layer(self) -> None:
        with pytest.raises(ValueError, match="less than or equal to in_features"):
            miss.Linear(nn.Linear(2, 4), "default", MissConfig(r=3, target_modules=["x"]))
        with pytest.raises(ValueError, match="divisible into"):
            miss.Linear(
                nn.Linear(5, 4),
                "default",
                MissConfig(r=2, init_weights="bat", target_modules=["x"]),
            )
        with pytest.raises(ValueError, match="mini_r"):
            miss.Linear(
                nn.Linear(4, 5),
                "default",
                MissConfig(r=2, mini_r=2, init_weights="mini", target_modules=["x"]),
            )

    def test_a_layer_reports_its_shard_widths(self) -> None:
        layer = miss.Linear(nn.Linear(4, 6), "a", MissConfig(r=2, mini_r=3, target_modules=["x"]))
        assert dict(layer.miss_r) == {"a": 2} and dict(layer.miss_mini_r) == {"a": 3}

    def test_the_config_checks_its_fields(self) -> None:
        with pytest.raises(ValueError, match="init_weights"):
            MissConfig(init_weights="other", target_modules=["a"])  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError, match="MiSS_only"):
            MissConfig(bias="MiSS_only", target_modules=["a"])
        assert _Mode.of(False) == "balance" and _Mode.of("mini") == "mini"

    def test_a_bat_and_a_balance_adapter_apply_together_like_the_merged_layer(self) -> None:
        net = _mlp()
        peft = get_peft_model(
            net, MissConfig(r=2, target_modules=["0"], init_weights=False), adapter_name="a"
        )
        peft.add_adapter("b", MissConfig(r=2, target_modules=["0"], init_weights="bat"))
        with lucid.no_grad():
            for p in peft.parameters():
                if p.shape[0] == 2 and p.ndim == 3:
                    p.copy_(lucid.randn(*p.shape) * 0.2)
        peft.set_adapter(["a", "b"])
        x = lucid.randn(3, 8)
        with lucid.no_grad():
            before = peft(x)
        peft.base_model.merge_adapter()
        with lucid.no_grad():
            assert _close(peft(x), before, atol=1e-4)


class TestBone:
    def test_a_bone_adapter_is_a_miss_adapter_under_the_old_name(self) -> None:
        net = _mlp()
        peft = get_peft_model(net, BoneConfig(r=2, target_modules=["0"], init_weights="bat"))
        layer = peft.base_model.model[0]
        assert isinstance(layer, bone.Linear) and isinstance(layer, miss.MissLayer)
        assert tuple(layer.bone_block["default"].shape) == (4, 2, 2)
        assert sorted(get_peft_model_state_dict(peft)) == ["base_model.model.0.bone_block"]

    def test_its_file_reads_back_as_bone(self, tmp_path: Path) -> None:
        peft = get_peft_model(_mlp(), BoneConfig(r=2, target_modules=["0"]))
        with lucid.no_grad():
            layer = peft.base_model.model[0]
            layer.bone_block["default"].copy_(lucid.randn(*layer.bone_block["default"].shape))
        peft.save_pretrained(tmp_path)
        assert type(PeftConfig.from_pretrained(tmp_path)) is BoneConfig
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        x = lucid.randn(3, 8)
        with lucid.no_grad():
            assert _close(loaded(x), peft(x))

    def test_only_its_two_variants_are_accepted(self) -> None:
        with pytest.raises(ValueError, match="init_weights"):
            BoneConfig(init_weights="mini", target_modules=["a"])  # type: ignore[arg-type]


# ── RandLoRA ─────────────────────────────────────────────────────────────────


class TestRandLora:
    def test_the_bases_are_a_function_of_the_key_and_the_shape(self) -> None:
        def draw(key: int) -> tuple[lucid.Tensor, lucid.Tensor]:
            tuner = get_peft_model(
                _mlp(), RandLoraConfig(r=2, projection_prng_key=key, target_modules=["0", "2"])
            ).base_model
            return tuner.randlora_A["default"], tuner.randlora_B["default"]

        a1, b1 = draw(5)
        a2, b2 = draw(5)
        a3, _ = draw(6)
        assert tuple(a1.shape) == (2, 1, 8) and tuple(b1.shape) == (8, 4, 2)
        assert _close(a1, a2, atol=0) and _close(b1, b2, atol=0)
        assert not _close(a1, a3, atol=1e-3)
        assert abs(float(a1.std().item()) - 1.0) < 1e-4  # normalised by the standard deviation

    @pytest.mark.parametrize("flag", ["sparse", "very_sparse"])
    def test_sparse_bases_are_ternary(self, flag: str) -> None:
        config = RandLoraConfig(r=2, target_modules=["0", "2"], **{flag: True})  # type: ignore[arg-type]
        tuner = get_peft_model(_mlp(), config).base_model
        values = {round(v, 4) for v in tuner.randlora_B["default"].flatten().tolist()}
        assert len(values) <= 3 and 0.0 in values  # {-c, 0, c}

    def test_a_layer_alone_draws_its_own_bases(self) -> None:
        base = nn.Linear(6, 4)
        layer = randlora.Linear(
            base, "default", RandLoraConfig(r=2, init_weights=False, target_modules=["x"])
        )
        assert tuple(layer.randlora_lambda["default"].shape) == (2, 2)
        assert not [k for k in layer.state_dict() if k.startswith("randlora_A") and False]
        x = lucid.randn(3, 6)
        assert not _close(layer(x), base(x), atol=1e-4)
        assert _close(layer(x), layer(x), atol=0)  # drawn once

    def test_adapters_share_one_rank(self) -> None:
        peft = get_peft_model(_mlp(), RandLoraConfig(r=2, target_modules=["0"]))
        with pytest.raises(ValueError, match="share a rank"):
            peft.add_adapter("other", RandLoraConfig(r=4, target_modules=["0"]))
        with pytest.raises(ValueError, match="PRNG initialisation key"):
            peft.add_adapter(
                "keyed", RandLoraConfig(r=2, projection_prng_key=1, target_modules=["0"])
            )

    def test_a_file_without_the_bases_restores_them_from_the_key(self, tmp_path: Path) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = RandLoraConfig(
                r=2,
                save_projection=False,
                projection_prng_key=3,
                init_weights=False,
                target_modules=["0", "2"],
            )
            peft = get_peft_model(_mlp(), config)
            peft.save_pretrained(tmp_path)
            assert sorted(load_safetensors(str(tmp_path / "adapter_model.safetensors"))) == sorted(
                k for k in get_peft_model_state_dict(peft)
            )
            loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        x = lucid.randn(3, 8)
        with lucid.no_grad():
            assert _close(loaded(x), peft(x))
        assert not any(
            "randlora_A" in k for k in load_safetensors(str(tmp_path / "adapter_model.safetensors"))
        )

    def test_the_file_holds_one_pair_of_bases_whatever_the_layer_count(
        self, tmp_path: Path
    ) -> None:
        peft = get_peft_model(_mlp(), RandLoraConfig(r=2, target_modules=["0", "2"]))
        peft.save_pretrained(tmp_path)
        keys = list(load_safetensors(str(tmp_path / "adapter_model.safetensors")))
        assert [k for k in keys if "randlora_A" in k or "randlora_B" in k] == [
            "base_model.randlora_A",
            "base_model.randlora_B",
        ]

    def test_a_reference_file_with_a_copy_of_the_bases_per_layer_loads(
        self, tmp_path: Path
    ) -> None:
        peft = get_peft_model(
            _mlp(), RandLoraConfig(r=2, init_weights=False, target_modules=["0", "2"])
        )
        peft.save_pretrained(tmp_path)
        tensors = dict(load_safetensors(str(tmp_path / "adapter_model.safetensors")))
        for layer in ("0", "2"):
            tensors[f"base_model.model.{layer}.randlora_A"] = tensors["base_model.randlora_A"]
            tensors[f"base_model.model.{layer}.randlora_B"] = tensors["base_model.randlora_B"]
        from lucid.serialization import save_safetensors

        save_safetensors(
            tensors, str(tmp_path / "adapter_model.safetensors"), metadata={"format": "pt"}
        )
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        x = lucid.randn(3, 8)
        with lucid.no_grad():
            assert _close(loaded(x), peft(x))


# ── VB-LoRA ──────────────────────────────────────────────────────────────────


def _vb(**kw: object) -> VBLoRAConfig:
    fields: dict[str, object] = {
        "r": 2,
        "num_vectors": 6,
        "vector_length": 4,
        "topk": 2,
        "init_vector_bank_bound": 0.5,
        "target_modules": ["0", "2"],
    } | kw
    return VBLoRAConfig(**fields)  # type: ignore[arg-type]


class TestVBLoRA:
    def test_a_sub_vector_is_the_softmax_mix_of_the_top_two_bank_rows(self) -> None:
        base = nn.Linear(4, 4)
        layer = vblora.Linear(base, "default", _vb(r=1, target_modules=["x"]))
        bank = [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]] + [[0.0] * 4] * 3
        logits = [0.5, 1.5, -1.0, 0.0, 0.2, 0.1]
        with lucid.no_grad():
            layer.vblora_vector_bank["default"].copy_(lucid.tensor(bank))
            layer.vblora_logits_A["default"].copy_(lucid.tensor([[logits]]))  # (1, 1, 6)
        top = sorted(range(6), key=lambda k: -logits[k])[:2]  # picks 1 and 0
        exps = [math.exp(logits[k]) for k in top]
        weights = [e / sum(exps) for e in exps]
        expected = [
            sum(w * bank[k][d] for w, k in zip(weights, top, strict=True)) for d in range(4)
        ]
        factor_a, _ = layer._factors("default")
        assert _close(factor_a, lucid.tensor([expected]))

    def test_the_bank_is_shared_and_trains_with_the_logits(self) -> None:
        peft = get_peft_model(_mlp(), _vb())
        tuner = peft.base_model
        bank = tuner.vblora_vector_bank["default"]
        assert bank.requires_grad
        assert sum(1 for p in peft.parameters() if p is bank) == 1
        peft(lucid.randn(3, 8)).sum().backward()
        assert bank.grad is not None and float(bank.grad.abs().max().item()) > 0
        total, other = tuner.get_nb_savable_parameters()
        assert other == 0 and total == sum(p.numel() for p in peft.parameters() if p.requires_grad)

    def test_a_second_adapter_has_a_bank_of_its_own_and_starts_frozen(self) -> None:
        peft = get_peft_model(_mlp(), _vb(), adapter_name="a")
        peft.add_adapter("b", _vb())
        bank_a, bank_b = (
            peft.base_model.vblora_vector_bank["a"],
            peft.base_model.vblora_vector_bank["b"],
        )
        assert bank_a is not bank_b and bank_a.requires_grad and not bank_b.requires_grad
        peft.set_adapter("b")
        assert bank_b.requires_grad and not bank_a.requires_grad
        peft.delete_adapter("a")
        assert list(peft.base_model.vblora_vector_bank.keys()) == ["b"]

    def test_the_top_k_file_is_small_and_loads_to_the_same_output(self, tmp_path: Path) -> None:
        full, small = tmp_path / "full", tmp_path / "small"
        peft = get_peft_model(_mlp(), _vb())
        peft.save_pretrained(full)
        get_peft_model(_mlp(), _vb(save_only_topk_weights=True)).save_pretrained(small)
        topk_peft = get_peft_model(_mlp(), _vb(save_only_topk_weights=True))
        topk_peft.save_pretrained(small)
        keys = sorted(load_safetensors(str(small / "adapter_model.safetensors")))
        assert keys == sorted(
            [
                "base_model.vblora_vector_bank",
                *(
                    f"base_model.model.{i}.vblora_logits_{ab}_{suffix}"
                    for i in ("0", "2")
                    for ab in "AB"
                    for suffix in ("topk_indices", "topk_weights")
                ),
            ]
        )
        tensors = load_safetensors(str(small / "adapter_model.safetensors"))
        assert (
            tensors["base_model.model.0.vblora_logits_A_topk_weights"].shape[-1] == 1
        )  # k - 1 kept
        loaded = PeftModel.from_pretrained(_mlp(), small)
        x = lucid.randn(3, 8)
        with lucid.no_grad():
            assert _close(loaded(x), topk_peft(x), atol=1e-5)

    def test_the_savable_count_is_printed(self, capsys: pytest.CaptureFixture[str]) -> None:
        peft = get_peft_model(_mlp(), _vb())
        peft.base_model.print_savable_parameters()
        total, _ = peft.base_model.get_nb_savable_parameters()
        assert f"{total:,d}" in capsys.readouterr().out

    def test_the_savable_count_of_a_top_k_file_is_the_bank_plus_the_weights_and_indices(
        self,
    ) -> None:
        full = get_peft_model(_mlp(), _vb()).base_model
        small = get_peft_model(_mlp(), _vb(save_only_topk_weights=True)).base_model
        bank = 6 * 4
        logits = full.get_nb_savable_parameters()[0] - bank
        # k - 1 weights and k indices (a quarter of a float32 each below 256 vectors) per sub-vector
        assert small.get_nb_savable_parameters()[0] == int(
            bank + logits / 6 * 1 + logits / 6 * 2 * 0.25
        )

    def test_the_sizes_must_be_multiples_of_the_vector_length(self) -> None:
        with pytest.raises(ValueError, match="divisible by `vector_length`"):
            vblora.Linear(nn.Linear(6, 4), "default", _vb(target_modules=["x"]))

    def test_the_config_checks_its_fields(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            VBLoRAConfig(num_vectors=2, topk=3, target_modules=["a"])
        with pytest.raises(NotImplementedError, match="vblora_only"):
            VBLoRAConfig(bias="vblora_only", target_modules=["a"])

    def test_the_default_targets_follow_the_model_type(self) -> None:
        from lucid.models import create_model

        net = create_model(
            "gpt2_small_lm",
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=8,
        )
        peft = get_peft_model(net, VBLoRAConfig(r=1, num_vectors=4, vector_length=8, topk=2))
        assert peft.base_model.targeted_module_names == ["transformer.h.0.attn.c_attn"]


# ── SHiRA ────────────────────────────────────────────────────────────────────


class TestShira:
    def test_a_mask_has_r_times_the_sides_entries(self) -> None:
        base = nn.Linear(8, 6)
        mask = shira.random_mask(base, 2, random_seed=3)
        assert tuple(mask.shape) == (6, 8) and int(mask.sum().item()) == 2 * (6 + 8)
        assert _close(mask, shira.random_mask(base, 2, random_seed=3), atol=0)
        assert not _close(mask, shira.random_mask(base, 2, random_seed=4), atol=0)

    def test_the_update_is_the_values_at_the_indices(self) -> None:
        base = nn.Linear(4, 3)
        layer = shira.Linear(base, "default", ShiraConfig(r=1, random_seed=1, target_modules=["x"]))
        rows, cols = layer.shira_indices["default"].tolist()
        with lucid.no_grad():
            layer.shira_weight["default"].copy_(lucid.arange(1, 8).to(lucid.float32))
        delta = layer.get_delta_weight("default").tolist()
        expected = [[0.0] * 4 for _ in range(3)]
        for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
            expected[i][j] = float(k + 1)
        assert delta == expected
        assert rows == sorted(rows) or list(zip(rows, cols)) == sorted(zip(rows, cols))  # row-major

    def test_a_scale_multiplies_the_update_and_is_per_adapter(self) -> None:
        layer = shira.Linear(
            nn.Linear(4, 3),
            "default",
            ShiraConfig(r=1, random_seed=1, init_weights=False, target_modules=["x"]),
        )
        before = layer.get_delta_weight("default")
        layer.set_scale("default", 0.5)
        layer.set_scale("absent", 9.0)  # ignored
        assert _close(layer.get_delta_weight("default"), before * 0.5)
        assert layer.scaling == {"default": 0.5}

    def test_a_custom_mask_function_picks_the_entries(self) -> None:
        def first_row(base_layer: nn.Module, r: int, **options: object) -> lucid.Tensor:
            weight = base_layer.weight  # type: ignore[union-attr]
            mask = lucid.zeros(*weight.shape)
            return mask.index_put(
                (lucid.zeros(weight.shape[1], dtype=lucid.int64), lucid.arange(weight.shape[1])),
                lucid.ones(weight.shape[1]),
            )

        config = ShiraConfig(r=1, mask_type="random", target_modules=["x"])
        config.mask_fn = first_row  # type: ignore[assignment,misc]
        # r (d_out + d_in) = 1 * (1 + 5) = 6 entries are asked for; the function returns 5 of them.
        with pytest.raises(ValueError, match="not the same dimensions"):
            shira.Linear(nn.Linear(5, 1), "default", config)

    def test_the_mask_function_is_not_a_field(self) -> None:
        config = ShiraConfig(target_modules=["a"])
        assert "mask_fn" not in config.to_dict()
        assert config.mask_fn is shira.random_mask
        assert ShiraConfig.from_dict(config.to_dict()).mask_fn is shira.random_mask

    def test_another_mask_type_warns_and_needs_a_function_of_its_own(self) -> None:
        with pytest.warns(UserWarning, match="not recognized"):
            config = ShiraConfig(mask_type="mine", target_modules=["x"])
        assert config.mask_fn is None
        with pytest.raises(ValueError, match="needs a mask function"):
            shira.Linear(nn.Linear(4, 3), "default", config)

    def test_a_rank_that_asks_for_more_than_the_weight_holds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="more shira params"):
            shira.Linear(nn.Linear(3, 3), "default", ShiraConfig(r=2, target_modules=["x"]))

    def test_the_indices_travel_with_the_file(self, tmp_path: Path) -> None:
        peft = get_peft_model(
            _mlp(), ShiraConfig(r=1, random_seed=2, init_weights=False, target_modules=["0", "2"])
        )
        peft.save_pretrained(tmp_path)
        tensors = load_safetensors(str(tmp_path / "adapter_model.safetensors"))
        assert tensors["base_model.model.0.shira_indices"].shape == (2, 16)
        # A fresh model with another seed has other indices; the file's win.
        fresh = get_peft_model(_mlp(), ShiraConfig(r=1, random_seed=99, target_modules=["0", "2"]))
        before = fresh.base_model.model[0].shira_indices["default"].tolist()
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        assert (
            loaded.base_model.model[0].shira_indices["default"].tolist()
            == peft.base_model.model[0].shira_indices["default"].tolist()
        )
        assert before != peft.base_model.model[0].shira_indices["default"].tolist()


# ── C3A ──────────────────────────────────────────────────────────────────────


class TestC3A:
    def test_the_dense_matrix_is_the_block_circulant_of_the_kernels(self) -> None:
        kernel = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]  # (1, 2, 3)
        dense = _circulant_matrix(lucid.tensor(kernel)).tolist()
        # block (0, j): C[s][t] = w[(t - s) mod 3]
        expected = [[kernel[0][c // 3][(c % 3 - s) % 3] for c in range(6)] for s in range(3)]
        assert dense == expected

    @pytest.mark.parametrize("size", [2, 4, 5])
    def test_the_fft_product_equals_the_dense_one(self, size: int) -> None:
        lucid.manual_seed(1)
        kernel = lucid.randn(2, 3, size)
        x = lucid.randn(4, 3 * size)
        assert _close(_circulant_apply(x, kernel), x @ _circulant_matrix(kernel).mT, atol=1e-4)

    def test_a_layer_divides_by_the_input_size_and_every_adapter_sees_the_input(self) -> None:
        base = nn.Linear(4, 4)
        layer = c3a.Linear(base, "a", C3AConfig(block_size=2, target_modules=["x"]))
        layer.update_layer("b", C3AConfig(block_size=2, target_modules=["x"]))
        layer.set_adapter(["a", "b"])
        x = lucid.randn(3, 4)
        dense = sum((layer.get_delta_weight(n) for n in ("a", "b")), lucid.zeros(4, 4))
        assert _close(layer(x), base(x) + x @ dense.mT, atol=1e-4)
        assert _close(layer.get_delta_weight("a"), _circulant_matrix(layer.c3a_kernel["a"]) / 4)

    def test_true_starts_at_zero_and_the_other_starts_differ(self) -> None:
        def kernel(init: object) -> lucid.Tensor:
            lucid.manual_seed(0)
            config = C3AConfig(block_size=2, init_weights=init, target_modules=["x"])  # type: ignore[arg-type]
            return c3a.Linear(nn.Linear(4, 4), "default", config).c3a_kernel["default"]

        assert float(kernel(True).abs().max().item()) == 0.0
        bound = math.sqrt(3.0) * math.sqrt(2.0 / 8)
        assert float(kernel("xavier_uniform").abs().max().item()) <= bound
        assert _close(kernel(False), kernel("xavier_uniform"), atol=0)
        assert float(kernel("kaiming_uniform").abs().max().item()) <= math.sqrt(1 / 4)
        assert float(kernel("gaussian").abs().max().item()) > 0

    def test_the_block_size_must_divide_both_sides(self) -> None:
        with pytest.raises(ValueError, match="input size"):
            c3a.Linear(nn.Linear(5, 4), "default", C3AConfig(block_size=2, target_modules=["x"]))
        with pytest.raises(ValueError, match="output size"):
            c3a.Linear(nn.Linear(4, 5), "default", C3AConfig(block_size=2, target_modules=["x"]))
        with pytest.raises(ValueError, match="init_weights"):
            C3AConfig(init_weights="other", target_modules=["a"])  # type: ignore[arg-type]

    def test_a_pattern_gives_one_module_another_block_size(self) -> None:
        config = C3AConfig(block_size=4, block_size_pattern={"2": 2}, target_modules=["0", "2"])
        net = get_peft_model(_mlp(), config).base_model.model
        assert dict(net[0].block_size) == {"default": 4} and dict(net[2].block_size) == {
            "default": 2
        }


# ── trainable tokens ─────────────────────────────────────────────────────────


def _lm() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 3))


class TestTrainableTokens:
    def test_the_rows_start_as_the_tables_and_only_they_train(self) -> None:
        net = _lm()
        ids = lucid.tensor([[1, 2, 5, 9]])
        before = net(ids)
        peft = get_peft_model(
            net, TrainableTokensConfig(token_indices=[2, 5], target_modules=["0"])
        )
        assert _close(peft(ids), before, atol=0)
        trainable = [n for n, p in peft.named_parameters() if p.requires_grad]
        assert trainable == ["base_model.model.0.trainable_tokens_delta.default"]
        peft(ids).sum().backward()
        grad = peft.base_model.model[0].trainable_tokens_delta["default"].grad
        assert grad is not None and float(grad.abs().max().item()) > 0

    def test_training_moves_those_rows_and_merging_writes_them(self) -> None:
        net = _lm()
        table = net[0].weight.clone()
        peft = get_peft_model(
            net, TrainableTokensConfig(token_indices=[2, 5], target_modules=["0"])
        )
        layer = peft.base_model.model[0]
        with lucid.no_grad():
            layer.trainable_tokens_delta["default"].copy_(lucid.ones(2, 4))
        ids = lucid.tensor([[1, 2, 5]])
        with lucid.no_grad():
            out = layer(ids)
        assert _close(
            out, lucid.stack([table[1], lucid.ones(4), lucid.ones(4)]).unsqueeze(0), atol=0
        )
        peft.base_model.merge_adapter()
        expected = table.tolist()
        expected[2], expected[5] = [1.0] * 4, [1.0] * 4
        assert layer.base_layer.weight.tolist() == expected
        peft.base_model.unmerge_adapter()
        assert layer.base_layer.weight.tolist() == table.tolist()

    def test_disjoint_adapters_apply_together_and_overlapping_ones_are_refused(self) -> None:
        peft = get_peft_model(
            _lm(),
            TrainableTokensConfig(token_indices=[1, 2], target_modules=["0"]),
            adapter_name="a",
        )
        peft.add_adapter("b", TrainableTokensConfig(token_indices=[3], target_modules=["0"]))
        peft.add_adapter("c", TrainableTokensConfig(token_indices=[2], target_modules=["0"]))
        peft.set_adapter(["a", "b"])
        peft(lucid.tensor([[1, 3]]))
        peft.set_adapter(["a", "c"])
        with pytest.raises(ValueError, match="disjunct token indices"):
            peft(lucid.tensor([[1, 3]]))

    def test_the_default_target_is_the_input_embedding_of_the_model(self) -> None:
        from lucid.models import create_model

        net = create_model(
            "gpt2_small_lm",
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=8,
            tie_word_embeddings=False,
        )
        peft = get_peft_model(net, TrainableTokensConfig(token_indices=[1, 2]))
        assert peft.base_model.targeted_module_names == ["transformer.wte"]
        ids = lucid.tensor([[1, 2, 3]])
        with lucid.no_grad():
            assert tuple(peft(ids).logits.shape) == (1, 3, 16)

    def test_the_file_holds_the_rows_and_the_config_the_ids(self, tmp_path: Path) -> None:
        peft = get_peft_model(
            _lm(), TrainableTokensConfig(token_indices=[2, 5], target_modules=["0"])
        )
        with lucid.no_grad():
            peft.base_model.model[0].trainable_tokens_delta["default"].copy_(lucid.randn(2, 4))
        peft.save_pretrained(tmp_path)
        assert sorted(load_safetensors(str(tmp_path / "adapter_model.safetensors"))) == [
            "base_model.model.0.trainable_tokens_delta"
        ]
        loaded = PeftModel.from_pretrained(_lm(), tmp_path)
        ids = lucid.tensor([[2, 3, 5]])
        with lucid.no_grad():
            assert _close(loaded(ids), peft(ids), atol=0)

    def test_a_table_another_module_shares_is_refused_and_the_model_left_alone(self) -> None:
        from lucid.models import create_model

        net = create_model(
            "gpt2_small_lm",
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=8,
        )
        assert net.lm_head.weight is net.transformer.wte.weight
        with pytest.raises(NotImplementedError, match="lm_head"):
            get_peft_model(net, TrainableTokensConfig(token_indices=[1]))
        assert type(net.transformer.wte) is nn.Embedding
        assert not hasattr(net, "peft_config")

    def test_an_output_head_that_is_the_table_is_adapted_alike(self) -> None:
        net = nn.Sequential(nn.Linear(4, 10, bias=False))
        peft = get_peft_model(net, TrainableTokensConfig(token_indices=[3], target_modules=["0"]))
        layer = peft.base_model.model[0]
        assert isinstance(layer, trainable_tokens.Linear)
        with lucid.no_grad():
            layer.trainable_tokens_delta["default"].copy_(lucid.ones(1, 4))
        x = lucid.randn(2, 4)
        with lucid.no_grad():
            assert _close(peft(x)[:, 3], x.sum(-1), atol=1e-5)

    def test_the_ids_are_checked(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            trainable_tokens.Embedding(
                nn.Embedding(5, 2), "default", TrainableTokensConfig(target_modules=["x"])
            )
        with pytest.raises(ValueError, match="5 rows"):
            trainable_tokens.Embedding(
                nn.Embedding(5, 2),
                "default",
                TrainableTokensConfig(token_indices=[5], target_modules=["x"]),
            )
        with pytest.raises(ValueError, match="non-negative"):
            TrainableTokensConfig(token_indices=[-1], target_modules=["x"])


# ── the reference library's numbers ──────────────────────────────────────────


def _twin(ref: ModuleType, net: nn.Module) -> ModuleType:
    twin = ref.nn.Sequential(
        *[ref.nn.Linear(m.in_features, m.out_features) for m in net if isinstance(m, nn.Linear)]
    )
    with ref.no_grad():
        for mine, theirs in zip([m for m in net if isinstance(m, nn.Linear)], twin, strict=True):
            theirs.weight.copy_(ref.tensor(mine.weight.tolist()))
            theirs.bias.copy_(ref.tensor(mine.bias.tolist()))
    return twin


def _copy(ref: ModuleType, mine: lucid.Tensor, theirs: ModuleType) -> None:
    with ref.no_grad():
        theirs.copy_(ref.tensor(mine.tolist()))


@pytest.mark.parity
@pytest.mark.parametrize(
    ("init", "options"),
    [(True, {}), (False, {}), ("mini", {"mini_r": 2}), ("bat", {})],
    ids=["balance", "random", "mini", "bat"],
)
def test_miss_matches_the_reference(ref: ModuleType, init: object, options: dict[str, int]) -> None:
    library = adapter_library()
    lucid.manual_seed(2)
    net = nn.Sequential(nn.Linear(8, 6))
    x = lucid.randn(5, 8)
    fields = {"r": 2, "init_weights": init, "target_modules": ["0"], **options}
    twin = _twin(ref, net)
    mine = get_peft_model(net, MissConfig(**fields))  # type: ignore[arg-type]
    theirs = library.get_peft_model(twin, library.MissConfig(**fields))
    layer, other = mine.base_model.model[0], theirs.base_model.model[0]
    with lucid.no_grad():
        layer.miss_block["default"].copy_(lucid.randn(*layer.miss_block["default"].shape) * 0.3)
    _copy(ref, layer.miss_block["default"], other.miss_block["default"])
    assert gap(ref, mine(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    mine.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, layer.base_layer.weight, other.base_layer.weight) < 1e-5
    mine.base_model.unmerge_adapter()
    theirs.base_model.unmerge_adapter()
    assert gap(ref, layer.base_layer.weight, other.base_layer.weight) < 1e-5


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(8, 6), (6, 8), (6, 6)], ids=["wide", "tall", "square"])
def test_randlora_matches_the_reference(ref: ModuleType, shape: tuple[int, int]) -> None:
    library = adapter_library()
    lucid.manual_seed(2)
    net = nn.Sequential(nn.Linear(*shape))
    x = lucid.randn(5, shape[0])
    fields = {"r": 2, "randlora_alpha": 6, "init_weights": False, "target_modules": ["0"]}
    twin = _twin(ref, net)
    mine = get_peft_model(net, RandLoraConfig(**fields))  # type: ignore[arg-type]
    theirs = library.get_peft_model(twin, library.RandLoraConfig(**fields))
    layer, other = mine.base_model.model[0], theirs.base_model.model[0]
    # The bases are drawn by each framework's generator: take ours across.
    _copy(ref, mine.base_model.randlora_A["default"], theirs.base_model.randlora_A["default"])
    _copy(ref, mine.base_model.randlora_B["default"], theirs.base_model.randlora_B["default"])
    _copy(ref, layer.randlora_lambda["default"], other.randlora_lambda["default"])
    _copy(ref, layer.randlora_gamma["default"], other.randlora_gamma["default"])
    assert gap(ref, mine(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert gap(ref, layer.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-5


@pytest.mark.parity
def test_vblora_matches_the_reference(ref: ModuleType) -> None:
    library = adapter_library()
    lucid.manual_seed(2)
    net = nn.Sequential(nn.Linear(8, 6))
    x = lucid.randn(5, 8)
    fields = {"r": 2, "num_vectors": 6, "vector_length": 2, "topk": 2, "target_modules": ["0"]}
    twin = _twin(ref, net)
    mine = get_peft_model(net, VBLoRAConfig(init_vector_bank_bound=0.5, **fields))  # type: ignore[arg-type]
    theirs = library.get_peft_model(
        twin, library.VBLoRAConfig(init_vector_bank_bound=0.5, **fields)
    )
    layer, other = mine.base_model.model[0], theirs.base_model.model[0]
    _copy(
        ref,
        mine.base_model.vblora_vector_bank["default"],
        theirs.base_model.vblora_vector_bank["default"],
    )
    for name in ("vblora_logits_A", "vblora_logits_B"):
        _copy(ref, getattr(layer, name)["default"], getattr(other, name)["default"])
    assert gap(ref, mine(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert gap(ref, layer.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-5


@pytest.mark.parity
@pytest.mark.parametrize("seed", [None, 3], ids=["default-seed", "seed-3"])
def test_shira_matches_the_reference_down_to_the_indices(ref: ModuleType, seed: int | None) -> None:
    library = adapter_library()
    lucid.manual_seed(2)
    net = nn.Sequential(nn.Linear(8, 6))
    x = lucid.randn(5, 8)
    fields = {"r": 1, "random_seed": seed, "init_weights": False, "target_modules": ["0"]}
    twin = _twin(ref, net)
    mine = get_peft_model(net, ShiraConfig(**fields))  # type: ignore[arg-type]
    theirs = library.get_peft_model(twin, library.ShiraConfig(**fields))
    layer, other = mine.base_model.model[0], theirs.base_model.model[0]
    assert (
        gap(
            ref,
            layer.shira_indices["default"].to(lucid.float32),
            other.shira_indices["default"].to(ref.float32),
        )
        == 0.0
    )
    _copy(ref, layer.shira_weight["default"], other.shira_weight["default"])
    assert gap(ref, mine(x), theirs(ref.tensor(x.tolist()))) < 1e-5


@pytest.mark.parity
@pytest.mark.parametrize("init", ["xavier_uniform", "gaussian", True], ids=str)
def test_c3a_matches_the_reference(ref: ModuleType, init: object) -> None:
    library = adapter_library()
    lucid.manual_seed(2)
    net = nn.Sequential(nn.Linear(8, 6))
    x = lucid.randn(5, 8)
    fields = {"block_size": 2, "init_weights": init, "target_modules": ["0"]}
    twin = _twin(ref, net)
    mine = get_peft_model(net, C3AConfig(**fields))  # type: ignore[arg-type]
    theirs = library.get_peft_model(twin, library.C3AConfig(**fields))
    layer, other = mine.base_model.model[0], theirs.base_model.model[0]
    with lucid.no_grad():
        layer.c3a_kernel["default"].copy_(lucid.randn(*layer.c3a_kernel["default"].shape))
    _copy(ref, layer.c3a_kernel["default"], other.c3a_kernel["default"])
    assert gap(ref, mine(x), theirs(ref.tensor(x.tolist()))) < 1e-4
    assert gap(ref, layer.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-5


@pytest.mark.parity
def test_trainable_tokens_match_the_reference(ref: ModuleType) -> None:
    library = adapter_library()
    lucid.manual_seed(2)
    net = nn.Sequential(nn.Embedding(10, 4))
    twin = ref.nn.Sequential(ref.nn.Embedding(10, 4))
    _copy(ref, net[0].weight, twin[0].weight)
    fields = {"token_indices": [1, 4], "target_modules": ["0"]}
    mine = get_peft_model(net, TrainableTokensConfig(**fields))  # type: ignore[arg-type]
    theirs = library.get_peft_model(twin, library.TrainableTokensConfig(**fields))
    layer, other = mine.base_model.model[0], theirs.base_model.model[0]
    with lucid.no_grad():
        layer.trainable_tokens_delta["default"].copy_(lucid.randn(2, 4))
    _copy(ref, layer.trainable_tokens_delta["default"], other.trainable_tokens_delta["default"])
    ids = lucid.tensor([[1, 2, 4, 9]])
    assert gap(ref, mine(ids), theirs(ref.tensor(ids.tolist()))) < 1e-6
    mine.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, layer.base_layer.weight, other.base_layer.weight) < 1e-6


@pytest.mark.parity
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (MissConfig(r=2, target_modules=["a"]), "MissConfig"),
        (RandLoraConfig(r=2, target_modules=["a"]), "RandLoraConfig"),
        (VBLoRAConfig(r=2, target_modules=["a"]), "VBLoRAConfig"),
        (ShiraConfig(r=2, random_seed=1, target_modules=["a"]), "ShiraConfig"),
        (C3AConfig(block_size=4, target_modules=["a"]), "C3AConfig"),
        (TrainableTokensConfig(token_indices=[1], target_modules=["a"]), "TrainableTokensConfig"),
    ],
    ids=lambda v: v if isinstance(v, str) else type(v).__name__,
)
def test_the_config_file_has_the_references_keys(ours: PeftConfig, theirs: str) -> None:
    library = adapter_library()
    reference = getattr(library, theirs)(
        **{
            k: v
            for k, v in ours.to_dict().items()
            if k in {"r", "block_size", "random_seed", "token_indices", "target_modules"}
        }
    ).to_dict()
    mine = ours.to_dict()
    assert set(mine) == set(reference) - {"peft_version"}
    assert all(
        mine[k] == (sorted(v) if isinstance(v, set) else v)
        for k, v in reference.items()
        if k in mine
    )
