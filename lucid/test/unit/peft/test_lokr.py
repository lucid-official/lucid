"""LoKr: the closed forms, on CPU and Metal.

The update of an adapter is ``s W1 ⊗ W2`` — block ``(i, j)`` of the weight is
``s W1[i, j] W2`` — applied to the input the way the base layer applies its
weight.  The expected weight is assembled block by block from the factors
(never through ``kron``), and a convolution is checked against a real
convolution with the base's stride, padding and groups.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.peft import LoKrConfig, LoKrModel, get_peft_model
from lucid.peft.tuners import lokr
from lucid.peft.tuners._base import _tuner_layers


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _config(**kwargs: object) -> LoKrConfig:
    fields: dict[str, object] = {"r": 2, "alpha": 6, "target_modules": ["x"]} | kwargs
    return LoKrConfig(**fields)  # type: ignore[arg-type]


def _factor(layer: lokr.LoKrLayer, table: str, adapter: str = "default") -> lucid.Tensor:
    entry = getattr(layer, table)[adapter]
    assert isinstance(entry, lucid.Tensor)
    return entry


def _shapes(layer: lokr.LoKrLayer, adapter: str = "default") -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(getattr(layer, name)[adapter].shape)
        for name in layer.adapter_layer_names
        if adapter in getattr(layer, name)
    }


def _blocks(left: lucid.Tensor, right: lucid.Tensor) -> lucid.Tensor:
    """``left ⊗ right``: block ``(i, j)`` is ``left[i, j] * right`` (``right`` may carry trailing axes)."""
    rows = [
        lucid.cat([right * left[i, j] for j in range(left.shape[1])], dim=1)
        for i in range(left.shape[0])
    ]
    return lucid.cat(rows, dim=0)


def _w1(layer: lokr.LoKrLayer) -> lucid.Tensor:
    if "default" in layer.lokr_w1:
        return _factor(layer, "lokr_w1")
    return _factor(layer, "lokr_w1_a") @ _factor(layer, "lokr_w1_b")


class TestFactorization:
    @pytest.mark.parametrize(
        ("dimension", "factor", "expected"),
        [
            (256, -1, (16, 16)),
            (128, -1, (8, 16)),
            (127, -1, (1, 127)),
            (128, 4, (4, 32)),
            (12, -1, (3, 4)),
            (1, -1, (1, 1)),
            (36, -1, (6, 6)),
            (30, 7, (5, 6)),  # 7 does not divide 30: the search stops near it
            (64, 8, (8, 8)),
            (64, 1, (1, 64)),
        ],
    )
    def test_the_split_is_the_reference_s(
        self, dimension: int, factor: int, expected: tuple[int, int]
    ) -> None:
        assert lokr.factorization(dimension, factor) == expected

    @pytest.mark.parametrize("dimension", [2, 6, 9, 24, 50, 96, 768, 1000])
    @pytest.mark.parametrize("factor", [-1, 2, 3, 8])
    def test_the_two_numbers_always_multiply_back(self, dimension: int, factor: int) -> None:
        m, n = lokr.factorization(dimension, factor)
        assert m * n == dimension and m >= 1 and n >= 1


class TestLinear:
    def test_the_full_factors_have_the_reference_shapes_and_zero_start(self, device: str) -> None:
        base = nn.Linear(8, 6).to(device)  # 6 = 2 x 3, 8 = 2 x 4
        layer = lokr.Linear(base, "default", _config(), r=2, alpha=6)
        assert _shapes(layer) == {"lokr_w1": (2, 2), "lokr_w2": (3, 4)}
        assert float(_factor(layer, "lokr_w1").abs().sum().item()) == 0
        assert float(_factor(layer, "lokr_w2").abs().max().item()) > 0
        assert _factor(layer, "lokr_w2").device == base.weight.device
        assert layer.scaling["default"] == 3.0 and layer.rank_dropout_scale["default"] is False
        x = lucid.randn(3, 8, device=device)
        assert _same(layer(x), base(x))

    def test_the_update_is_the_kronecker_product(self, device: str) -> None:
        base = nn.Linear(8, 6).to(device)
        layer = lokr.Linear(base, "default", _config(init_weights=False), r=2, alpha=6)
        expected = _blocks(_factor(layer, "lokr_w1"), _factor(layer, "lokr_w2")) * 3.0
        assert tuple(expected.shape) == (6, 8)
        assert _close(layer.get_delta_weight("default"), expected, atol=1e-5)
        x = lucid.randn(3, 8, device=device)
        assert _close(layer(x), x @ base.weight.mT + base.bias + x @ expected.mT, atol=1e-4)

    def test_a_low_rank_factor_is_a_product_of_two_matrices(self, device: str) -> None:
        base = nn.Linear(32, 32).to(device)  # 32 = 4 x 8 both ways
        layer = lokr.Linear(
            base, "default", _config(init_weights=False, decompose_both=True), r=1, alpha=2
        )
        assert _shapes(layer) == {
            "lokr_w1_a": (4, 1),
            "lokr_w1_b": (1, 4),
            "lokr_w2_a": (8, 1),
            "lokr_w2_b": (1, 8),
        }
        w2 = _factor(layer, "lokr_w2_a") @ _factor(layer, "lokr_w2_b")
        expected = _blocks(_w1(layer), w2) * 2.0
        assert _close(layer.get_delta_weight("default"), expected, atol=1e-5)

    def test_decompose_both_only_decomposes_when_the_rank_is_small_enough(self) -> None:
        big = lokr.Linear(nn.Linear(32, 32), "d", _config(decompose_both=True), r=4, alpha=4)
        assert "lokr_w1" in big.adapter_layer_names and "d" in big.lokr_w1  # r >= 4 / 2
        alone = lokr.Linear(nn.Linear(32, 32), "d", _config(), r=1, alpha=1)
        assert "d" in alone.lokr_w1  # without decompose_both the left factor is always full
        assert "d" in alone.lokr_w2_a  # but the right one is low-rank once r < 8 / 2

    def test_decompose_factor_changes_the_split(self, device: str) -> None:
        base = nn.Linear(32, 32).to(device)
        layer = lokr.Linear(base, "default", _config(decompose_factor=2), r=8, alpha=8)
        assert _shapes(layer) == {"lokr_w1": (2, 2), "lokr_w2": (16, 16)}

    def test_the_default_start_zeroes_the_left_factor_the_lycoris_one_the_right(
        self, device: str
    ) -> None:
        base = nn.Linear(32, 32).to(device)
        x = lucid.randn(2, 32, device=device)
        for how in (True, "lycoris"):
            plain = lokr.Linear(base, "p", _config(init_weights=how), r=1, alpha=1)
            assert _same(plain(x), base(x)), how
        default = lokr.Linear(base, "d", _config(decompose_both=True), r=1, alpha=1)
        assert float(_factor(default, "lokr_w1_a", "d").abs().sum().item()) == 0
        assert float(_factor(default, "lokr_w1_b", "d").abs().max().item()) > 0
        lycoris = lokr.Linear(
            base, "l", _config(init_weights="lycoris", decompose_both=True), r=1, alpha=1
        )
        assert float(_factor(lycoris, "lokr_w2_b", "l").abs().sum().item()) == 0
        assert float(_factor(lycoris, "lokr_w2_a", "l").abs().max().item()) > 0
        assert float(_factor(lycoris, "lokr_w1_a", "l").abs().max().item()) > 0
        full = lokr.Linear(nn.Linear(8, 6), "f", _config(init_weights="lycoris"), r=2, alpha=2)
        assert float(_factor(full, "lokr_w2", "f").abs().sum().item()) == 0
        assert float(_factor(full, "lokr_w1", "f").abs().max().item()) > 0

    def test_random_init_leaves_no_factor_zero(self, device: str) -> None:
        layer = lokr.Linear(
            nn.Linear(8, 6).to(device), "default", _config(init_weights=False), r=2, alpha=2
        )
        for table in ("lokr_w1", "lokr_w2"):
            assert float(_factor(layer, table).abs().max().item()) > 0

    def test_merging_adds_the_update_to_the_weight(self, device: str) -> None:
        base = nn.Linear(8, 6).to(device)
        layer = lokr.Linear(base, "default", _config(init_weights=False), r=2, alpha=6)
        weight = base.weight.clone()
        delta = _blocks(_factor(layer, "lokr_w1"), _factor(layer, "lokr_w2")) * 3.0
        x = lucid.randn(3, 8, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert _close(base.weight, weight + delta, atol=1e-5)
            assert _close(layer(x), adapted, atol=1e-4)
            layer.unmerge()
            assert _close(base.weight, weight, atol=1e-5)

    def test_the_gradient_reaches_every_factor_through_a_frozen_base(self, device: str) -> None:
        base = nn.Linear(32, 32).to(device)
        layer = lokr.Linear(
            base, "default", _config(init_weights=False, decompose_both=True), r=1, alpha=1
        )
        base.requires_grad_(False)
        layer(lucid.randn(3, 32, device=device)).sum().backward()
        for name in layer.adapter_layer_names:
            if "default" in getattr(layer, name):
                grad = _factor(layer, name).grad
                assert grad is not None and float(grad.abs().max().item()) > 0, name
        assert base.weight.grad is None

    def test_it_refuses_what_it_cannot_do(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            lokr.Linear(nn.Linear(4, 4), "default", _config())
        layer = lokr.Linear(nn.Linear(4, 4), "default", _config(), r=2, alpha=2)
        with pytest.raises(ValueError, match="already exists"):
            layer.update_layer("default", 2, 2, _config())
        assert repr(layer).startswith("lokr.Linear(")


class TestRankDropout:
    def test_the_kept_rows_are_scaled_only_when_asked(self, device: str) -> None:
        base = nn.Linear(8, 16).to(device)
        for scale in (False, True):
            layer = lokr.Linear(
                base,
                "default",
                _config(init_weights=False, rank_dropout=0.5, rank_dropout_scale=scale),
                r=2,
                alpha=6,
            )
            full = layer.get_delta_weight("default")
            lucid.manual_seed(5)
            dropped = layer.get_delta_weight("default", apply_rank_dropout=True)
            kept = [i for i in range(16) if float(dropped[i].abs().sum().item()) > 0]
            assert 0 < len(kept) < 16
            gain = 16 / len(kept) if scale else 1.0
            for i in range(16):
                expected = full[i] * gain if i in kept else lucid.zeros_like(full[i])
                assert _close(dropped[i], expected, atol=1e-5), (scale, i)

    def test_module_dropout_skips_the_adapter_in_training(self, device: str) -> None:
        base = nn.Linear(8, 6).to(device)
        layer = lokr.Linear(
            base, "default", _config(init_weights=False, module_dropout=0.9999999), r=2, alpha=6
        )
        x = lucid.randn(3, 8, device=device)
        layer.train()
        assert _same(layer(x), base(x))
        layer.eval()
        assert not _close(layer(x), base(x), atol=1e-4)


class TestConvolutions:
    def test_a_full_right_factor_carries_the_kernel(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv2d(4, 6, 3, stride=2, padding=1).to(device)  # out 6 = 2 x 3, in 4 = 2 x 2
        layer = lokr.Conv2d(base, "default", _config(init_weights=False), r=2, alpha=6)
        assert _shapes(layer) == {"lokr_w1": (2, 2), "lokr_w2": (3, 2, 3, 3)}
        w1 = _factor(layer, "lokr_w1")
        w2 = _factor(layer, "lokr_w2")
        # ΔW[i*3 + p, j*2 + q, k, l] = s w1[i, j] w2[p, q, k, l]
        expected = _blocks(w1, w2) * 3.0
        assert tuple(expected.shape) == (6, 4, 3, 3)
        assert _close(layer.get_delta_weight("default"), expected, atol=1e-5)
        x = lucid.randn(2, 4, 7, 7, device=device)
        assert _close(layer(x), base(x) + F.conv2d(x, expected, None, 2, 1, 1, 1), atol=1e-4)

    def test_a_low_rank_right_factor_is_flattened_over_the_kernel(self, device: str) -> None:
        base = nn.Conv2d(4, 6, 3, padding=1).to(device)
        layer = lokr.Conv2d(base, "default", _config(init_weights=False), r=1, alpha=3)
        assert _shapes(layer) == {
            "lokr_w1": (2, 2),
            "lokr_w2_a": (3, 1),
            "lokr_w2_b": (1, 2 * 3 * 3),
        }
        flat = _factor(layer, "lokr_w2_a") @ _factor(layer, "lokr_w2_b")  # (3, 2*3*3)
        right = flat.reshape(3, 2, 3, 3)
        expected = _blocks(_factor(layer, "lokr_w1"), right) * 3.0
        assert _close(layer.get_delta_weight("default"), expected, atol=1e-5)
        x = lucid.randn(2, 4, 5, 5, device=device)
        assert _close(layer(x), base(x) + F.conv2d(x, expected, None, 1, 1), atol=1e-4)

    def test_the_effective_kernel_contracts_a_spatial_core(self, device: str) -> None:
        base = nn.Conv2d(4, 6, 3, padding=1).to(device)
        layer = lokr.Conv2d(
            base, "default", _config(init_weights=False, use_effective_conv2d=True), r=1, alpha=3
        )
        assert _shapes(layer) == {
            "lokr_w1": (2, 2),
            "lokr_t2": (1, 1, 3, 3),
            "lokr_w2_a": (1, 3),
            "lokr_w2_b": (1, 2),
        }
        t2, rows, columns = (_factor(layer, n) for n in ("lokr_t2", "lokr_w2_a", "lokr_w2_b"))
        # right[p, q, k, l] = t2[0, 0, k, l] rows[0, p] columns[0, q] for r = 1
        right = lucid.stack(
            [
                lucid.stack([t2[0, 0] * (rows[0, p] * columns[0, q]) for q in range(2)])
                for p in range(3)
            ]
        )
        expected = _blocks(_factor(layer, "lokr_w1"), right) * 3.0
        assert _close(layer.get_delta_weight("default"), expected, atol=1e-5)

    def test_a_grouped_convolution_uses_the_per_group_input_channels(self, device: str) -> None:
        base = nn.Conv2d(4, 6, 3, padding=1, groups=2).to(device)  # 2 input channels per group
        layer = lokr.Conv2d(base, "default", _config(init_weights=False), r=2, alpha=6)
        kernel = layer.get_delta_weight("default")
        assert tuple(kernel.shape) == (6, 2, 3, 3)
        x = lucid.randn(2, 4, 5, 5, device=device)
        assert _close(layer(x), base(x) + F.conv2d(x, kernel, None, 1, 1, 1, 2), atol=1e-4)

    def test_conv1d_full_and_effective_kernels(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv1d(4, 6, 3, padding=1).to(device)
        full = lokr.Conv1d(base, "default", _config(init_weights=False), r=2, alpha=6)
        assert tuple(_factor(full, "lokr_w2").shape) == (3, 2, 3)
        # The right factor is 3-d: the block product must pair its trailing axis with the kernel's.
        expected = _blocks(_factor(full, "lokr_w1"), _factor(full, "lokr_w2")) * 3.0
        assert tuple(expected.shape) == (6, 4, 3)
        assert _close(full.get_delta_weight("default"), expected, atol=1e-5)
        x = lucid.randn(2, 4, 6, device=device)
        assert _close(full(x), base(x) + F.conv1d(x, expected, None, 1, 1), atol=1e-4)
        effective = lokr.Conv1d(
            base, "default", _config(init_weights=False, use_effective_conv2d=True), r=1, alpha=3
        )
        assert tuple(_factor(effective, "lokr_t2").shape) == (1, 1, 3, 1)  # the reference's layout
        assert tuple(effective.get_delta_weight("default").shape) == (6, 4, 3)

    def test_a_pointwise_convolution_is_not_factored_with_a_core(self, device: str) -> None:
        base = nn.Conv2d(4, 6, 1).to(device)
        layer = lokr.Conv2d(base, "default", _config(use_effective_conv2d=True), r=1, alpha=1)
        assert "default" not in layer.lokr_t2 and "default" in layer.lokr_w2_b

    @pytest.mark.parametrize("effective", [False, True])
    def test_merging_a_convolution_gives_the_same_output(
        self, device: str, effective: bool
    ) -> None:
        base = nn.Conv2d(4, 6, 3, padding=1).to(device)
        layer = lokr.Conv2d(
            base,
            "default",
            _config(init_weights=False, use_effective_conv2d=effective),
            r=1,
            alpha=3,
        )
        weight = base.weight.clone()
        x = lucid.randn(2, 4, 5, 5, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert _close(base.weight, weight + layer.get_delta_weight("default"), atol=1e-5)
            assert _close(layer(x), adapted, atol=1e-4)
            layer.unmerge()
            assert _close(base.weight, weight, atol=1e-5)


class TestTuner:
    def test_rank_and_alpha_patterns_change_the_shapes_per_module(self) -> None:
        net = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        config = LoKrConfig(r=8, alpha=8, target_modules=["0", "2"], rank_pattern={"2": 1})
        LoKrModel(net, config)
        layers = dict(_tuner_layers(net))
        assert "default" in layers["0"].lokr_w2 and "default" in layers["2"].lokr_w2_a  # type: ignore[attr-defined]
        assert layers["2"].scaling["default"] == 8.0  # type: ignore[attr-defined]

    def test_a_model_of_each_kind_is_adapted(self) -> None:
        net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv1d(2, 2, 1), nn.Linear(4, 2))
        LoKrModel(net, LoKrConfig(r=2, target_modules=["0", "1", "2"]))
        assert [type(m) for m in net] == [lokr.Conv2d, lokr.Conv1d, lokr.Linear]

    def test_an_unsupported_module_is_named(self) -> None:
        with pytest.raises(ValueError, match=r"Target module 0 \(Embedding\) is not supported"):
            LoKrModel(nn.Sequential(nn.Embedding(5, 4)), LoKrConfig(target_modules=["0"]))

    def test_get_peft_model_trains_only_the_factors(self) -> None:
        peft = get_peft_model(nn.Sequential(nn.Linear(6, 6)), LoKrConfig(r=2, target_modules=["0"]))
        trainable = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trainable == {
            "base_model.model.0.lokr_w1.default",
            "base_model.model.0.lokr_w2.default",
        }


class TestConfig:
    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"r": 0}, ValueError, "positive integer"),
            ({"decompose_both": 1}, TypeError, "bool"),
            ({"decompose_factor": 0}, ValueError, "-1 or a positive"),
            ({"decompose_factor": -2}, ValueError, "-1 or a positive"),
            ({"decompose_factor": 2.0}, TypeError, "int"),
            ({"rank_dropout_scale": "yes"}, TypeError, "bool"),
            ({"init_weights": "kaiming"}, ValueError, "True, False or 'lycoris'"),
            ({"init_weights": 1}, ValueError, "True, False or 'lycoris'"),
        ],
    )
    def test_a_bad_field_is_refused(
        self, kwargs: dict[str, object], error: type[Exception], message: str
    ) -> None:
        with pytest.raises(error, match=message):
            LoKrConfig(**kwargs)  # type: ignore[arg-type]

    def test_the_reference_s_defaults_and_round_trip(self) -> None:
        config = LoKrConfig()
        assert (config.r, config.alpha, config.decompose_both, config.decompose_factor) == (
            8,
            8,
            False,
            -1,
        )
        assert config.rank_dropout_scale is False and config.init_weights is True
        assert LoKrConfig.from_dict(config.to_dict()) == config
        assert (
            LoKrConfig.from_dict({**config.to_dict(), "init_weights": "lycoris"}).init_weights
            == "lycoris"
        )
        assert str(config.peft_type) == "LOKR"
