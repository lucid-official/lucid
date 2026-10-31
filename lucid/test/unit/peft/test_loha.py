"""LoHa: the closed forms, on CPU and Metal.

The update of an adapter is ``s (W1a W1b) ⊙ (W2a W2b)`` (``s = alpha / r``),
applied to the input the way the base layer applies its weight.  The expected
update is written out entry by entry from the factors with plain loops — not
through the layer's matrix products — and a convolution is checked against a
real convolution with the base's stride, padding, dilation and groups.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.peft import LoHaConfig, LoHaModel, get_peft_model
from lucid.peft.tuners import loha
from lucid.peft.tuners._base import _tuner_layers


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _config(**kwargs: object) -> LoHaConfig:
    fields: dict[str, object] = {"r": 2, "alpha": 6, "target_modules": ["x"]} | kwargs
    return LoHaConfig(**fields)  # type: ignore[arg-type]


def _factor(layer: loha.LoHaLayer, table: str, adapter: str = "default") -> lucid.Tensor:
    entry = getattr(layer, table)[adapter]
    assert isinstance(entry, lucid.Tensor)
    return entry


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def _hadamard_linear(layer: loha.LoHaLayer, adapter: str = "default") -> lucid.Tensor:
    """``s (W1a W1b) ⊙ (W2a W2b)`` of a linear layer, entry by entry."""
    first = _matmul(
        _factor(layer, "hada_w1_a", adapter).tolist(), _factor(layer, "hada_w1_b", adapter).tolist()
    )
    second = _matmul(
        _factor(layer, "hada_w2_a", adapter).tolist(), _factor(layer, "hada_w2_b", adapter).tolist()
    )
    s = layer.alpha[adapter] / layer.r[adapter]
    return lucid.tensor(
        [
            [s * a * b for a, b in zip(r1, r2, strict=True)]
            for r1, r2 in zip(first, second, strict=True)
        ]
    )


def _tucker_entries(t: lucid.Tensor, rows: lucid.Tensor, columns: lucid.Tensor) -> lucid.Tensor:
    """``K[p, q, k, l] = sum_ij T[i, j, k, l] U[i, p] V[j, q]``, by loops."""
    r, _, kh, kw = t.shape
    out_p, out_q = rows.shape[1], columns.shape[1]
    tv, uv, vv = t.tolist(), rows.tolist(), columns.tolist()
    return lucid.tensor(
        [
            [
                [
                    [
                        sum(
                            tv[i][j][k][m] * uv[i][p] * vv[j][q] for i in range(r) for j in range(r)
                        )
                        for m in range(kw)
                    ]
                    for k in range(kh)
                ]
                for q in range(out_q)
            ]
            for p in range(out_p)
        ]
    )


class TestLinear:
    def test_the_factors_have_the_reference_shapes_and_zero_start(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = loha.Linear(base, "default", _config(), r=2, alpha=6)
        assert [
            tuple(_factor(layer, n).shape)
            for n in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
        ] == [(5, 2), (2, 8), (5, 2), (2, 8)]
        assert "hada_t1" in layer.adapter_layer_names and "default" not in layer.hada_t1
        assert float(_factor(layer, "hada_w2_b").abs().sum().item()) == 0
        assert float(_factor(layer, "hada_w1_a").abs().max().item()) > 0
        assert _factor(layer, "hada_w1_a").device == base.weight.device
        assert layer.scaling["default"] == 3.0 and layer.rank_dropout_scale["default"] is True
        x = lucid.randn(3, 8, device=device)
        assert _same(layer(x), base(x))

    def test_the_update_is_the_hadamard_product_of_the_two_low_rank_products(
        self, device: str
    ) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = loha.Linear(base, "default", _config(init_weights=False), r=2, alpha=6)
        expected = _hadamard_linear(layer).to(device)
        assert _close(layer.get_delta_weight("default"), expected, atol=1e-5)
        x = lucid.randn(3, 8, device=device)
        assert _close(layer(x), x @ base.weight.mT + base.bias + x @ expected.mT, atol=1e-4)

    def test_random_init_leaves_no_factor_zero(self, device: str) -> None:
        layer = loha.Linear(
            nn.Linear(8, 5).to(device), "default", _config(init_weights=False), r=2, alpha=2
        )
        for table in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"):
            assert float(_factor(layer, table).abs().max().item()) > 0

    def test_merging_adds_the_update_to_the_weight(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = loha.Linear(base, "default", _config(init_weights=False), r=2, alpha=6)
        weight = base.weight.clone()
        delta = _hadamard_linear(layer).to(device)
        x = lucid.randn(3, 8, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert _close(base.weight, weight + delta, atol=1e-5)
            assert _close(layer(x), adapted, atol=1e-4)
            layer.unmerge()
            assert _close(base.weight, weight, atol=1e-5)

    def test_two_active_adapters_add_their_updates(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = loha.Linear(base, "a", _config(init_weights=False), r=2, alpha=6)
        layer.update_layer("b", 3, 3, _config(init_weights=False))
        layer.set_adapter(["a", "b"])
        x = lucid.randn(3, 8, device=device)
        total = _hadamard_linear(layer, "a") + _hadamard_linear(layer, "b")
        assert _close(layer(x), base(x) + x @ total.to(device).mT, atol=1e-4)

    def test_the_gradient_reaches_every_factor_through_a_frozen_base(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        layer = loha.Linear(base, "default", _config(init_weights=False), r=2, alpha=6)
        base.requires_grad_(False)
        layer(lucid.randn(3, 8, device=device)).sum().backward()
        for table in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"):
            grad = _factor(layer, table).grad
            assert grad is not None and float(grad.abs().max().item()) > 0, table
        assert base.weight.grad is None

    def test_rank_and_alpha_must_be_given(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            loha.Linear(nn.Linear(4, 4), "default", _config())  # r defaults to 0, the reference's
        layer = loha.Linear(nn.Linear(4, 4), "default", _config(), r=2, alpha=2)
        with pytest.raises(ValueError, match="already exists"):
            layer.update_layer("default", 2, 2, _config())
        with pytest.raises(TypeError, match="adapts nn.Linear"):
            loha.Linear(nn.Conv2d(2, 2, 1), "default", _config(), r=2, alpha=2)  # type: ignore[arg-type]

    def test_the_repr_names_the_method(self) -> None:
        layer = loha.Linear(nn.Linear(4, 4), "default", _config(), r=2, alpha=2)
        assert repr(layer).startswith("loha.Linear(") and "r=2" in repr(layer)
        layer.update_layer("b", 3, 3, _config())
        layer.set_adapter(["default", "b"])
        assert "r={'default': 2, 'b': 3}" in repr(layer)


class TestDropout:
    def test_rank_dropout_zeroes_rows_and_rescales_the_rest(self, device: str) -> None:
        base = nn.Linear(8, 16).to(device)
        layer = loha.Linear(
            base, "default", _config(init_weights=False, rank_dropout=0.5), r=2, alpha=6
        )
        full = layer.get_delta_weight("default")
        lucid.manual_seed(5)
        dropped = layer.get_delta_weight("default", apply_rank_dropout=True)
        kept = [i for i in range(16) if float(dropped[i].abs().sum().item()) > 0]
        assert 0 < len(kept) < 16
        gain = 16 / len(kept)  # the kept rows are scaled up to keep the mean
        for i in range(16):
            expected = full[i] * gain if i in kept else lucid.zeros_like(full[i])
            assert _close(dropped[i], expected, atol=1e-5), i

    def test_rank_dropout_acts_in_training_only(self, device: str) -> None:
        base = nn.Linear(8, 16).to(device)
        layer = loha.Linear(
            base, "default", _config(init_weights=False, rank_dropout=0.5), r=2, alpha=6
        )
        x = lucid.randn(4, 8, device=device)
        layer.eval()
        assert _same(layer(x), layer(x))
        layer.train()
        lucid.manual_seed(1)
        first = layer(x)
        lucid.manual_seed(2)
        assert not _close(first, layer(x), atol=1e-6)

    def test_an_all_dropped_mask_is_zero_not_nan(self, device: str) -> None:
        layer = loha.Linear(
            nn.Linear(8, 1).to(device),
            "default",
            _config(init_weights=False, rank_dropout=0.9999999),
            r=2,
            alpha=6,
        )
        dropped = layer.get_delta_weight("default", apply_rank_dropout=True)
        assert bool(lucid.isfinite(dropped).all().item())

    def test_module_dropout_skips_the_adapter_in_training(self, device: str) -> None:
        base = nn.Linear(8, 5).to(device)
        always = loha.Linear(
            base, "default", _config(init_weights=False, module_dropout=0.9999999), r=2, alpha=6
        )
        x = lucid.randn(3, 8, device=device)
        always.train()
        assert _same(always(x), base(x))
        always.eval()
        assert not _close(always(x), base(x), atol=1e-4)
        never = loha.Linear(base, "n", _config(init_weights=False), r=2, alpha=6)
        never.train()
        assert not _close(never(x), base(x), atol=1e-4)


class TestConvolutions:
    def test_the_flattened_kernel_is_convolved_like_the_base(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv2d(4, 6, 3, stride=2, padding=1, dilation=1).to(device)
        layer = loha.Conv2d(base, "default", _config(init_weights=False), r=2, alpha=6)
        assert tuple(_factor(layer, "hada_w1_a").shape) == (6, 2)
        assert tuple(_factor(layer, "hada_w1_b").shape) == (2, 4 * 3 * 3)
        flat = _hadamard_linear(layer).to(device)  # the same formula on the flattened kernel
        kernel = flat.reshape(6, 4, 3, 3)
        assert _close(layer.get_delta_weight("default"), kernel, atol=1e-5)
        x = lucid.randn(2, 4, 7, 7, device=device)
        expected = base(x) + F.conv2d(x, kernel, None, 2, 1, 1, 1)
        assert _close(layer(x), expected, atol=1e-4)

    def test_a_grouped_convolution_uses_the_per_group_input_channels(self, device: str) -> None:
        base = nn.Conv2d(4, 6, 3, padding=1, groups=2).to(device)
        layer = loha.Conv2d(base, "default", _config(init_weights=False), r=2, alpha=6)
        assert tuple(_factor(layer, "hada_w1_b").shape) == (2, 2 * 3 * 3)
        x = lucid.randn(2, 4, 5, 5, device=device)
        kernel = layer.get_delta_weight("default")
        assert tuple(kernel.shape) == (6, 2, 3, 3)
        assert _close(layer(x), base(x) + F.conv2d(x, kernel, None, 1, 1, 1, 2), atol=1e-4)

    def test_the_effective_kernel_contracts_a_spatial_core(self, device: str) -> None:
        base = nn.Conv2d(3, 4, 2).to(device)
        layer = loha.Conv2d(
            base, "default", _config(init_weights=False, use_effective_conv2d=True), r=2, alpha=6
        )
        shapes = {n: tuple(_factor(layer, n).shape) for n in layer.adapter_layer_names}
        assert shapes == {
            "hada_w1_a": (2, 4),
            "hada_w1_b": (2, 3),
            "hada_w2_a": (2, 4),
            "hada_w2_b": (2, 3),
            "hada_t1": (2, 2, 2, 2),
            "hada_t2": (2, 2, 2, 2),
        }
        first = _tucker_entries(
            _factor(layer, "hada_t1"), _factor(layer, "hada_w1_a"), _factor(layer, "hada_w1_b")
        )
        second = _tucker_entries(
            _factor(layer, "hada_t2"), _factor(layer, "hada_w2_a"), _factor(layer, "hada_w2_b")
        )
        expected = (first * second * 3.0).to(device)
        assert _close(layer.get_delta_weight("default"), expected, atol=1e-4)
        x = lucid.randn(2, 3, 4, 4, device=device)
        assert _close(layer(x), base(x) + F.conv2d(x, expected), atol=1e-4)

    def test_the_effective_kernel_starts_as_the_base(self, device: str) -> None:
        base = nn.Conv2d(3, 4, 2).to(device)
        layer = loha.Conv2d(base, "default", _config(use_effective_conv2d=True), r=2, alpha=6)
        x = lucid.randn(2, 3, 4, 4, device=device)
        assert _same(layer(x), base(x))
        assert float(_factor(layer, "hada_t1").abs().max().item()) > 0

    def test_a_pointwise_convolution_is_not_factored_with_a_core(self, device: str) -> None:
        layer = loha.Conv2d(
            nn.Conv2d(3, 4, 1).to(device),
            "default",
            _config(use_effective_conv2d=True),
            r=2,
            alpha=6,
        )
        assert "default" not in layer.hada_t1
        assert tuple(_factor(layer, "hada_w1_b").shape) == (2, 3)

    def test_conv1d_flattened_and_effective(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv1d(3, 4, 3, padding=1).to(device)
        flat = loha.Conv1d(base, "default", _config(init_weights=False), r=2, alpha=6)
        kernel = _hadamard_linear(flat).to(device).reshape(4, 3, 3)
        x = lucid.randn(2, 3, 6, device=device)
        assert _close(flat(x), base(x) + F.conv1d(x, kernel, None, 1, 1), atol=1e-4)
        effective = loha.Conv1d(
            base, "default", _config(init_weights=False, use_effective_conv2d=True), r=2, alpha=6
        )
        assert tuple(_factor(effective, "hada_t1").shape) == (2, 2, 3, 1)  # the reference's layout
        delta = effective.get_delta_weight("default")
        assert tuple(delta.shape) == (4, 3, 3)
        first = _tucker_entries(
            _factor(effective, "hada_t1"),
            _factor(effective, "hada_w1_a"),
            _factor(effective, "hada_w1_b"),
        )
        second = _tucker_entries(
            _factor(effective, "hada_t2"),
            _factor(effective, "hada_w2_a"),
            _factor(effective, "hada_w2_b"),
        )
        assert _close(delta, ((first * second) * 3.0).reshape(4, 3, 3).to(device), atol=1e-4)

    def test_padding_modes_and_same_padding_follow_the_base(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv2d(3, 4, 3, padding=1, padding_mode="circular").to(device)
        layer = loha.Conv2d(base, "default", _config(init_weights=False), r=2, alpha=6)
        x = lucid.randn(1, 3, 5, 5, device=device)
        kernel = layer.get_delta_weight("default")
        probe = nn.Conv2d(3, 4, 3, padding=1, padding_mode="circular", bias=False).to(device)
        with lucid.no_grad():
            probe.weight.copy_(kernel)
        assert _close(layer(x), base(x) + probe(x), atol=1e-4)
        same = nn.Conv2d(3, 4, 3, padding="same").to(device)
        assert _close(loha.Conv2d(same, "default", _config(), r=2, alpha=6)(x), same(x))

    @pytest.mark.parametrize("effective", [False, True])
    def test_merging_a_convolution_gives_the_same_output(
        self, device: str, effective: bool
    ) -> None:
        base = nn.Conv2d(3, 4, 2).to(device)
        layer = loha.Conv2d(
            base,
            "default",
            _config(init_weights=False, use_effective_conv2d=effective),
            r=2,
            alpha=6,
        )
        weight = base.weight.clone()
        x = lucid.randn(2, 3, 4, 4, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert _close(base.weight, weight + layer.get_delta_weight("default"), atol=1e-5)
            assert _close(layer(x), adapted, atol=1e-4)
            layer.unmerge()
            assert _close(base.weight, weight, atol=1e-5)


class TestTuner:
    def test_rank_and_alpha_patterns_set_them_per_module(self) -> None:
        net = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 6))
        config = LoHaConfig(
            r=2, alpha=4, target_modules=["0", "2"], rank_pattern={"2": 3}, alpha_pattern={"0": 8}
        )
        LoHaModel(net, config)
        layers = dict(_tuner_layers(net))
        assert (layers["0"].r["default"], layers["0"].alpha["default"]) == (2, 8)  # type: ignore[attr-defined]
        assert (layers["2"].r["default"], layers["2"].alpha["default"]) == (3, 4)  # type: ignore[attr-defined]
        assert layers["2"].scaling["default"] == 4 / 3  # type: ignore[attr-defined]

    def test_a_pattern_key_is_checked_as_a_regex(self) -> None:
        with pytest.raises(ValueError, match="regular expression"):
            LoHaConfig(rank_pattern={"(": 2})

    def test_a_convolution_model_is_adapted_and_its_class_follows_the_module(self) -> None:
        net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv1d(2, 2, 1), nn.Linear(4, 2))
        LoHaModel(net, LoHaConfig(r=2, target_modules=["0", "1", "2"]))
        kinds = [type(m) for m in net]
        assert kinds == [loha.Conv2d, loha.Conv1d, loha.Linear]

    def test_an_unsupported_module_is_named(self) -> None:
        net = nn.Sequential(nn.Embedding(5, 4), nn.Conv3d(2, 2, 1))
        with pytest.raises(ValueError, match=r"Target module 0 \(Embedding\) is not supported"):
            LoHaModel(net, LoHaConfig(target_modules=["0"]))
        with pytest.raises(ValueError, match=r"Target module 1 \(Conv3d\) is not supported"):
            LoHaModel(
                nn.Sequential(nn.Embedding(5, 4), nn.Conv3d(2, 2, 1)),
                LoHaConfig(target_modules=["1"]),
            )

    def test_all_linear_adapts_the_linears_but_the_head(self) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
                self.lm_head = nn.Linear(4, 3)

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return self.lm_head(self.body(x))

        net = Net()
        tuner = LoHaModel(net, LoHaConfig(r=2, target_modules="all-linear"))
        assert tuner.targeted_module_names == ["body.0", "body.1"]

    def test_get_peft_model_trains_only_the_factors(self) -> None:
        peft = get_peft_model(nn.Sequential(nn.Linear(6, 6)), LoHaConfig(r=2, target_modules=["0"]))
        trainable = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trainable == {
            f"base_model.model.0.hada_{t}.default" for t in ("w1_a", "w1_b", "w2_a", "w2_b")
        }


class TestConfig:
    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"r": 0}, ValueError, "positive integer"),
            ({"r": 2.5}, TypeError, "int"),
            ({"alpha": "8"}, TypeError, "real number"),
            ({"rank_dropout": 1.0}, ValueError, r"\[0, 1\)"),
            ({"module_dropout": -0.1}, ValueError, r"\[0, 1\)"),
            ({"use_effective_conv2d": 1}, TypeError, "bool"),
            ({"init_weights": "lycoris"}, TypeError, "bool"),
        ],
    )
    def test_a_bad_field_is_refused(
        self, kwargs: dict[str, object], error: type[Exception], message: str
    ) -> None:
        with pytest.raises(error, match=message):
            LoHaConfig(**kwargs)  # type: ignore[arg-type]

    def test_the_reference_s_defaults_and_round_trip(self) -> None:
        config = LoHaConfig()
        assert (config.r, config.alpha, config.rank_dropout, config.module_dropout) == (
            8,
            8,
            0.0,
            0.0,
        )
        assert config.use_effective_conv2d is False and config.init_weights is True
        assert LoHaConfig.from_dict(config.to_dict()) == config
        assert str(config.peft_type) == "LOHA"
