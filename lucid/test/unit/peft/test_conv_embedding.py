"""``lora.Embedding`` and ``lora.Conv{1,2,3}d``: the closed forms, on CPU and Metal.

An embedding's update is ``s (A^T[x]) B^T`` and merges as ``s (B A)^T`` into
the table; a convolution's is two convolutions in a row, which is itself one
convolution with the kernel ``s B∘A`` — checked against a real convolution
of the base's geometry (stride, dilation, ``"same"`` padding, padding mode),
not against the layer's own arithmetic.
"""

import math
import warnings

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.peft import LoraConfig, get_peft_model
from lucid.peft.tuners import lora


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-4) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


# ── Embedding ────────────────────────────────────────────────────────────────


def _embedding(
    device: str, padding_idx: int | None = None, **kwargs: object
) -> tuple[nn.Embedding, lora.Embedding]:
    lucid.manual_seed(0)
    base = nn.Embedding(10, 4, padding_idx=padding_idx).to(device)
    options: dict[str, object] = {"r": 3, "lora_alpha": 6} | kwargs
    return base, lora.Embedding(base, "default", **options)  # type: ignore[arg-type]


def _tokens(device: str) -> lucid.Tensor:
    return lucid.tensor([[1, 3, 3], [7, 0, 9]], device=device)


class TestEmbedding:
    def test_the_factors_are_the_reference_s(self, device: str) -> None:
        base, layer = _embedding(device)
        assert layer.get_base_layer() is base and layer.weight is base.weight
        assert (layer.in_features, layer.out_features) == (10, 4)
        assert tuple(layer.lora_embedding_A["default"].shape) == (3, 10)  # (r, num_embeddings)
        assert tuple(layer.lora_embedding_B["default"].shape) == (4, 3)  # (embedding_dim, r)
        assert sorted(layer.state_dict()) == [
            "base_layer.weight",
            "lora_embedding_A.default",
            "lora_embedding_B.default",
        ]
        assert repr(layer).startswith("lora.Embedding(")
        assert layer.scaling["default"] == 2.0

    def test_the_default_init_is_zero_a_and_normal_b(self, device: str) -> None:
        _, layer = _embedding(device)
        a, b = layer.lora_embedding_A["default"], layer.lora_embedding_B["default"]
        assert _same(a, lucid.zeros(3, 10, device=device))
        assert float(b.std().item()) > 0.3  # N(0, 1), 12 values
        _, plain = _embedding(device, init_lora_weights=False)
        assert float(plain.lora_embedding_A["default"].abs().max().item()) > 0

    def test_a_zero_update_is_the_base_bit_for_bit(self, device: str) -> None:
        base, layer = _embedding(device)
        assert _same(layer(_tokens(device)), base(_tokens(device)))

    @pytest.mark.parametrize("use_rslora", [False, True])
    def test_the_update_is_the_scaled_product_looked_up(
        self, device: str, use_rslora: bool
    ) -> None:
        base, layer = _embedding(device, init_lora_weights=False, use_rslora=use_rslora)
        scale = 6 / math.sqrt(3) if use_rslora else 6 / 3
        assert layer.scaling["default"] == scale
        a, b = layer.lora_embedding_A["default"], layer.lora_embedding_B["default"]
        ids = _tokens(device)
        # Row ``t`` of the update is ``s * B @ A[:, t]``.
        update = lucid.stack([(b @ a[:, int(t)]) * scale for t in ids.reshape(-1).tolist()])
        expected = base(ids) + update.reshape(2, 3, 4)
        assert _close(layer(ids), expected, atol=1e-5)

    def test_the_delta_weight_is_the_transposed_product(self, device: str) -> None:
        _, layer = _embedding(device, init_lora_weights=False)
        a, b = layer.lora_embedding_A["default"], layer.lora_embedding_B["default"]
        delta = layer.get_delta_weight("default")
        assert tuple(delta.shape) == (10, 4)
        for token in (0, 4, 9):
            assert _close(delta[token], (b @ a[:, token]) * 2.0, atol=1e-5)

    def test_merging_folds_the_delta_into_the_table_and_unmerging_takes_it_out(
        self, device: str
    ) -> None:
        base, layer = _embedding(device, init_lora_weights=False)
        table = base.weight.clone()
        ids = _tokens(device)
        with lucid.no_grad():
            adapted = layer(ids)
            weight = base.weight
            layer.merge()
            assert base.weight is weight and layer.merged
            assert _close(base.weight, table + layer.get_delta_weight("default"), atol=1e-6)
            assert _close(layer(ids), adapted, atol=1e-5)
            layer.unmerge()
            assert _close(base.weight, table, atol=1e-5) and not layer.merged

    def test_the_padding_row_receives_no_gradient(self, device: str) -> None:
        _, layer = _embedding(device, padding_idx=0, init_lora_weights=False)
        a = layer.lora_embedding_A["default"]
        layer(_tokens(device)).sum().backward()  # token 0 is in the batch
        assert a.grad is not None
        assert _same(a.grad[:, 0], lucid.zeros(3, device=device))
        assert float(a.grad[:, 3].abs().max().item()) > 0

    def test_the_base_layer_s_lookup_options_are_passed_through(self, device: str) -> None:
        base, layer = _embedding(device, init_lora_weights=False)
        base.max_norm = (
            1e6  # far above any row: the renorm is a no-op and must not break the lookup
        )
        assert _close(
            layer(_tokens(device)),
            base(_tokens(device)) + layer.get_delta_weight("default")[_tokens(device)],
            atol=1e-5,
        )

    def test_dropout_is_not_applied_and_says_so(self, device: str) -> None:
        with pytest.warns(UserWarning, match="not applied by lora.Embedding"):
            _, layer = _embedding(device, init_lora_weights=False, lora_dropout=0.5)
        layer.train()
        ids = _tokens(device)
        assert _same(layer(ids), layer(ids))

    def test_a_bfloat16_table_gets_a_float32_adapter(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Embedding(10, 4).to(device).bfloat16()
        layer = lora.Embedding(base, "default", r=2, lora_alpha=2, init_lora_weights=False)
        assert layer.lora_embedding_A["default"].dtype == lucid.float32
        ids = _tokens(device)
        out = layer(ids)
        assert out.dtype == lucid.bfloat16
        expected = (base(ids).float() + layer.get_delta_weight("default")[ids]).bfloat16()
        assert _same(out, expected)

    def test_gaussian_has_the_same_zero_a(self, device: str) -> None:
        _, layer = _embedding(device, init_lora_weights="gaussian")
        assert _same(layer.lora_embedding_A["default"], lucid.zeros(3, 10, device=device))

    def test_a_second_adapter_is_inactive_until_chosen(self, device: str) -> None:
        base, layer = _embedding(device, init_lora_weights=False)
        ids = _tokens(device)
        first = layer(ids)
        layer.update_layer("other", 2, 4, 0.0, False, False)
        assert layer.active_adapters == ["default"] and _same(layer(ids), first)
        layer.set_adapter("other")
        assert not _same(layer(ids), first)
        assert [n for n in layer.state_dict() if n.endswith(".other")] == [
            "lora_embedding_A.other",
            "lora_embedding_B.other",
        ]
        layer.delete_adapter("other")
        assert "other" not in layer.lora_embedding_A and layer.active_adapters == []

    def test_the_tuner_reaches_a_tied_free_embedding_by_name(self, device: str) -> None:
        lucid.manual_seed(0)
        model = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 2)).to(device)
        peft = get_peft_model(model, LoraConfig(r=2, target_modules=["0"]))
        assert isinstance(model[0], lora.Embedding)
        assert peft.base_model.targeted_module_names == ["0"]
        assert not model[0].base_layer.weight.requires_grad
        assert model[0].lora_embedding_B["default"].requires_grad

    def test_a_negative_rank_is_refused_by_the_layer(self, device: str) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _embedding(device, r=0)


# ── convolutions ─────────────────────────────────────────────────────────────

CONV = {
    1: (nn.Conv1d, lora.Conv1d, F.conv1d, (2, 3, 9)),
    2: (nn.Conv2d, lora.Conv2d, F.conv2d, (2, 3, 8, 7)),
    3: (nn.Conv3d, lora.Conv3d, F.conv3d, (1, 3, 5, 5, 4)),
}

#: Geometries a base may have, each as the keyword arguments of its constructor.
GEOMETRIES = {
    "plain": {},
    "padded": {"padding": 1},
    "strided": {"stride": 2, "padding": 1},
    "dilated": {"dilation": 2, "padding": 2},
    "same": {"padding": "same"},
    "reflect": {"padding": 1, "padding_mode": "reflect"},
}


def _conv(
    dims: int, device: str, geometry: str = "padded", **kwargs: object
) -> tuple[nn.Module, lora.LoraLayer, lucid.Tensor]:
    kind, layer_kind, _, shape = CONV[dims]
    lucid.manual_seed(0)
    base = kind(3, 4, 3, **GEOMETRIES[geometry]).to(device)  # type: ignore[arg-type]
    options: dict[str, object] = {"r": 2, "lora_alpha": 6, "init_lora_weights": False} | kwargs
    layer = layer_kind(base, "default", **options)
    return base, layer, lucid.randn(*shape, device=device)


class TestConvolution:
    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_the_factors_are_a_kernel_wide_conv_and_a_one_wide_one(
        self, dims: int, device: str
    ) -> None:
        base, layer, _ = _conv(dims, device)
        a, b = layer.lora_A["default"], layer.lora_B["default"]
        assert tuple(a.weight.shape) == (2, 3, *(3,) * dims)
        assert tuple(b.weight.shape) == (4, 2, *(1,) * dims)
        assert a.bias is None and b.bias is None
        assert sorted(layer.state_dict()) == [
            "base_layer.bias",
            "base_layer.weight",
            "lora_A.default.weight",
            "lora_B.default.weight",
        ]
        assert layer.weight is base.weight and layer.bias is base.bias
        assert repr(layer).startswith(f"lora.Conv{dims}d(")
        assert layer.scaling["default"] == 3.0

    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_a_zero_b_is_the_base_bit_for_bit(self, dims: int, device: str) -> None:
        base, layer, x = _conv(dims, device, init_lora_weights=True)
        assert _same(layer(x), base(x))

    @pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_the_two_convolutions_are_one_with_the_delta_kernel(
        self, dims: int, geometry: str, device: str
    ) -> None:
        base, layer, x = _conv(dims, device, geometry)
        _, _, conv_fn, _ = CONV[dims]
        delta = layer.get_delta_weight("default")
        assert tuple(delta.shape) == tuple(base.weight.shape)
        # The lora branch alone, as one real convolution of the base's geometry.
        lora_branch = layer(x) - base(x)
        merged_kernel_branch = _conv_with(base, x, delta, conv_fn)
        assert _close(lora_branch, merged_kernel_branch, atol=1e-4)

    @pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
    def test_merging_gives_the_unmerged_output(self, geometry: str, device: str) -> None:
        base, layer, x = _conv(2, device, geometry)
        with lucid.no_grad():
            adapted = layer(x)
            weight = base.weight
            kernel = base.weight.clone()
            layer.merge()
            assert base.weight is weight and layer.merged
            assert _close(layer(x), adapted)
            assert _close(base(x), adapted)  # the base alone now computes the adapted layer
            layer.unmerge()
            assert _close(base.weight, kernel, atol=1e-5)

    def test_the_update_is_scale_times_b_after_a(self, device: str) -> None:
        base, layer, x = _conv(2, device)
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        expected = base(x) + F.conv2d(F.conv2d(x, a, padding=1), b) * 3.0
        assert _close(layer(x), expected, atol=1e-4)

    def test_the_bias_is_the_base_s_and_unchanged_by_merging(self, device: str) -> None:
        base, layer, _ = _conv(2, device)
        bias = base.bias.clone()
        layer.merge()
        assert _same(base.bias, bias)

    def test_a_bfloat16_base_gets_a_float32_adapter(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv2d(3, 4, 3, padding=1).to(device).bfloat16()
        layer = lora.Conv2d(base, "default", r=2, lora_alpha=2, init_lora_weights=False)
        assert layer.lora_A["default"].weight.dtype == lucid.float32
        x = lucid.randn(1, 3, 5, 5, device=device).bfloat16()
        out = layer(x)
        assert out.dtype == lucid.bfloat16
        assert _close(out.float(), base(x).float(), atol=1.0)

    def test_dropout_acts_only_in_training(self, device: str) -> None:
        _, layer, x = _conv(2, device, lora_dropout=0.5)
        layer.eval()
        assert _same(layer(x), layer(x))
        layer.train()
        assert not _same(layer(x), layer(x))

    def test_the_tuner_wraps_each_convolution_by_its_type(self, device: str) -> None:
        lucid.manual_seed(0)
        model = nn.Sequential(nn.Conv1d(3, 4, 3), nn.Conv2d(4, 4, 3), nn.Conv3d(4, 4, 3)).to(device)
        get_peft_model(model, LoraConfig(r=2, target_modules=["0", "1", "2"]))
        assert [type(m) for m in model] == [lora.Conv1d, lora.Conv2d, lora.Conv3d]


def _conv_with(
    base: nn.Module, x: lucid.Tensor, kernel: lucid.Tensor, conv_fn: object
) -> lucid.Tensor:
    """``x`` convolved with ``kernel`` the way ``base`` convolves with its weight, no bias."""
    assert callable(conv_fn)
    options = {
        "stride": base.stride,
        "dilation": base.dilation,
        "padding": "same" if base._padding_str else base.padding,
    }
    if base.padding_mode != "zeros":
        pad = (
            [p for p in reversed(base.padding) for _ in range(2)]
            if isinstance(base.padding, tuple)
            else [base.padding] * 2
        )
        x = F.pad(x, pad, mode=base.padding_mode)
        options["padding"] = 0
    out = conv_fn(x, kernel, **options)
    assert isinstance(out, lucid.Tensor)
    return out


class TestGroupedConvolution:
    def test_a_grouped_base_works_forward_and_cannot_merge(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv2d(4, 6, 3, padding=1, groups=2).to(device)
        with pytest.warns(UserWarning, match="groups > 1"):
            layer = lora.Conv2d(base, "default", r=2, lora_alpha=2, init_lora_weights=False)
        b = layer.lora_B["default"]
        assert tuple(b.weight.shape) == (6, 1, 1, 1)  # groups = 2, so 2 // 2 input channels each
        x = lucid.randn(1, 4, 5, 5, device=device)
        a = layer.lora_A["default"].weight
        expected = base(x) + F.conv2d(F.conv2d(x, a, padding=1), b.weight, groups=2)
        assert _close(layer(x), expected, atol=1e-4)
        for call in (layer.merge, lambda: layer.get_delta_weight("default")):
            with pytest.raises(NotImplementedError, match="groups > 1"):
                call()
        assert not layer.merged

    def test_the_rank_must_divide_the_groups(self, device: str) -> None:
        base = nn.Conv2d(4, 6, 3, groups=2).to(device)
        with pytest.warns(UserWarning), pytest.raises(ValueError, match="divisible by groups"):
            lora.Conv2d(base, "default", r=3, lora_alpha=3)

    def test_a_refused_merge_writes_no_layer(self, device: str) -> None:
        lucid.manual_seed(0)
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv2d(4, 6, 3, groups=2)).to(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peft = get_peft_model(
                model, LoraConfig(r=2, target_modules=["0", "1"], init_lora_weights=False)
            )
        weights = [m.base_layer.weight.clone() for m in model]
        with pytest.raises(NotImplementedError, match="groups > 1"):
            peft.base_model.merge_adapter()
        assert all(_same(m.base_layer.weight, w) for m, w in zip(model, weights, strict=True))
        assert not any(m.merged for m in model)


class TestRefusals:
    def test_a_lazy_convolution_is_refused_until_it_is_built(self) -> None:
        lazy = nn.LazyConv2d(4, 3)
        with pytest.raises(ValueError, match="first input"):
            lora.Conv2d(lazy, "default", r=2)  # type: ignore[arg-type]
        lazy(lucid.randn(1, 3, 6, 6))
        assert isinstance(lora.Conv2d(lazy, "default", r=2), lora.Conv2d)

    def test_a_layer_wraps_only_its_own_module_type(self) -> None:
        with pytest.raises(TypeError, match="adapts nn.Conv2d"):
            lora.Conv2d(nn.Conv1d(2, 2, 1), "default", r=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="adapts nn.Embedding"):
            lora.Embedding(nn.Linear(2, 2), "default", r=1)  # type: ignore[arg-type]

    def test_a_conv_target_in_the_tuner_is_not_the_old_refusal(self) -> None:
        model = nn.Sequential(nn.Conv2d(3, 4, 3))
        get_peft_model(model, LoraConfig(r=2, target_modules=["0"]))
        assert isinstance(model[0], lora.Conv2d)
