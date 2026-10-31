"""``init_lora_weights``: each spelling keeps the property it promises.

``True`` and ``"gaussian"`` start the update at zero; ``False`` does not.
PiSSA and OLoRA read the base weight: the adapted layer starts as the
original (``W' + s B A == W``), ``B A`` carries the weight's principal
directions (PiSSA: the rank-``r`` SVD truncation; OLoRA: the leading ``Q``
columns and ``R`` rows of a QR), and the residual is written into the base
weight in place.  The spellings live in a registry; the others of the
reference raise ``NotImplementedError``.
"""

import math
import warnings

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.peft import LoraConfig, get_peft_model
from lucid.peft.tuners.lora import Conv2d, Embedding, Linear, MultiheadAttention
from lucid.peft.tuners.lora._inits import _check_init, _Init, _Olora, _Pissa


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol, rtol=1e-4))


def _layer(
    device: str, init: bool | str, n_in: int = 8, n_out: int = 6, **kwargs: object
) -> tuple[nn.Linear, lucid.Tensor, Linear]:
    """``(base, its original weight, the adapted layer)``."""
    lucid.manual_seed(0)
    base = nn.Linear(n_in, n_out).to(device)
    original = base.weight.detach().clone()
    options: dict[str, object] = {"r": 3, "lora_alpha": 6} | kwargs
    return base, original, Linear(base, "default", init_lora_weights=init, **options)  # type: ignore[arg-type]


class TestPlainSpellings:
    def test_true_starts_at_zero_with_a_kaiming_a(self, device: str) -> None:
        _, _, layer = _layer(device, True, n_in=64)
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        assert float(b.abs().max().item()) == 0.0
        bound = math.sqrt(6 / ((1 + 5) * 64))  # kaiming-uniform, a = sqrt(5)
        assert float(a.abs().max().item()) <= bound + 1e-6 and float(a.abs().max().item()) > 0

    def test_gaussian_has_zero_b_and_a_std_of_one_over_r(self, device: str) -> None:
        _, _, layer = _layer(device, "gaussian", n_in=400, r=4, lora_alpha=8)
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        assert float(b.abs().max().item()) == 0.0
        assert abs(float(a.std().item()) - 1 / 4) < 0.02
        assert abs(float(a.mean().item())) < 0.03

    def test_false_leaves_a_nonzero_update(self, device: str) -> None:
        _, _, layer = _layer(device, False)
        assert float(layer.lora_B["default"].weight.abs().max().item()) > 0

    def test_a_fresh_adapter_does_not_touch_the_base_for_these(self, device: str) -> None:
        for init in (True, False, "gaussian"):
            _, original, layer = _layer(device, init)
            assert bool((layer.weight == original).all().item()), init


class TestPissa:
    def test_the_layer_starts_as_the_original(self, device: str) -> None:
        base, original, layer = _layer(device, "pissa")
        x = lucid.randn(5, 8, device=device)
        assert _close(layer(x), nn.functional.linear(x, original, base.bias), atol=1e-5)
        assert not bool((base.weight == original).all().item())

    def test_base_plus_the_update_is_the_original_weight(self, device: str) -> None:
        base, original, layer = _layer(device, "pissa")
        assert _close(base.weight + layer.get_delta_weight("default"), original, atol=1e-5)

    def test_the_update_is_the_rank_r_svd_truncation(self, device: str) -> None:
        _, original, layer = _layer(device, "pissa", r=3)
        u, s, vh = linalg.svd(original, full_matrices=False)
        truncation = (u[:, :3] * s[:3]) @ vh[:3]
        assert _close(layer.get_delta_weight("default"), truncation, atol=1e-4)

    def test_the_residual_has_the_remaining_singular_values(self, device: str) -> None:
        base, original, _ = _layer(device, "pissa", r=3)
        s_full = linalg.svdvals(original)
        s_rest = linalg.svdvals(base.weight.detach())
        assert _close(s_rest[:3], s_full[3:6], atol=1e-4)
        assert float(s_rest[3:].abs().max().item()) < 1e-4

    def test_the_factors_split_the_singular_values_evenly(self, device: str) -> None:
        _, _, layer = _layer(device, "pissa", r=3, lora_alpha=3)  # scale 1
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        assert _close(a @ a.mT, b.mT @ b, atol=1e-4)  # diag(S) both ways

    def test_the_scale_is_divided_out(self, device: str) -> None:
        _, original, layer = _layer(device, "pissa", r=2, lora_alpha=8)  # scale 4
        assert layer.scaling["default"] == 4.0
        u, s, vh = linalg.svd(original, full_matrices=False)
        assert _close(layer.get_delta_weight("default"), (u[:, :2] * s[:2]) @ vh[:2], atol=1e-4)

    def test_weights_keep_their_identity(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(8, 6).to(device)
        weight = base.weight
        layer = Linear(base, "default", r=2, init_lora_weights="pissa")
        assert base.weight is weight and layer.weight is weight

    def test_niter_reconstructs_the_original_exactly(self, device: str) -> None:
        base, original, layer = _layer(device, "pissa_niter_4")
        assert _close(base.weight + layer.get_delta_weight("default"), original, atol=1e-5)

    def test_niter_finds_the_principal_space_of_a_low_rank_weight(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(10, 7).to(device)
        with lucid.no_grad():
            low_rank = lucid.randn(7, 3, device=device) @ lucid.randn(3, 10, device=device)
            base.weight.copy_(low_rank + 1e-3 * lucid.randn(7, 10, device=device))
        original = base.weight.detach().clone()
        layer = Linear(base, "default", r=3, lora_alpha=3, init_lora_weights="pissa_niter_6")
        u, s, vh = linalg.svd(original, full_matrices=False)
        truncation = (u[:, :3] * s[:3]) @ vh[:3]
        assert _close(layer.get_delta_weight("default"), truncation, atol=2e-2)

    def test_niter_handles_a_wide_weight(self, device: str) -> None:
        base, original, layer = _layer(device, "pissa_niter_2", n_in=12, n_out=5, r=2)
        assert _close(base.weight + layer.get_delta_weight("default"), original, atol=1e-5)

    def test_a_half_precision_base_keeps_its_dtype_and_gets_float32_factors(self) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(8, 6).bfloat16()
        original = base.weight.detach().clone()
        layer = Linear(base, "default", r=2, lora_alpha=4, init_lora_weights="pissa")
        assert base.weight.dtype == lucid.bfloat16
        assert layer.lora_A["default"].weight.dtype == lucid.float32
        rebuilt = base.weight.float() + layer.get_delta_weight("default")
        assert _close(rebuilt, original.float(), atol=5e-2)

    def test_a_rank_beyond_the_weight_is_refused_untouched(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(8, 6).to(device)
        original = base.weight.detach().clone()
        with pytest.raises(ValueError, match="r <= min"):
            Linear(base, "default", r=7, init_lora_weights="pissa")
        assert bool((base.weight == original).all().item())

    def test_through_the_model(self, device: str) -> None:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3)).to(device)
        x = lucid.randn(4, 6, device=device)
        expected = net(x)
        peft = get_peft_model(
            net, LoraConfig(r=3, lora_alpha=6, target_modules=["0", "2"], init_lora_weights="pissa")
        )
        assert _close(peft(x), expected, atol=1e-4)
        trained = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trained and all("lora_" in n for n in trained)

    def test_the_spelling_is_stored_in_lower_case(self) -> None:
        assert LoraConfig(init_lora_weights="PiSSA").init_lora_weights == "pissa"
        assert LoraConfig(init_lora_weights="PISSA_niter_8").init_lora_weights == "pissa_niter_8"


class TestOlora:
    def test_the_layer_starts_as_the_original(self, device: str) -> None:
        base, original, layer = _layer(device, "olora")
        x = lucid.randn(5, 8, device=device)
        assert _close(layer(x), nn.functional.linear(x, original, base.bias), atol=1e-5)
        assert _close(base.weight + layer.get_delta_weight("default"), original, atol=1e-5)

    def test_b_has_orthonormal_columns_and_a_is_the_leading_r_of_r(self, device: str) -> None:
        _, _, layer = _layer(device, "olora", r=3)
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        assert _close(b.mT @ b, lucid.eye(3, device=device), atol=1e-5)
        upper = lucid.triu(a)
        assert _close(a, upper, atol=1e-6)  # R's leading rows are upper-triangular

    def test_the_update_is_q_r_for_the_leading_rank(self, device: str) -> None:
        _, original, layer = _layer(device, "olora", r=3, lora_alpha=3)
        q, r_factor = linalg.qr(original)
        assert _close(layer.get_delta_weight("default"), q[:, :3] @ r_factor[:3], atol=1e-5)

    def test_the_base_changes_in_place(self, device: str) -> None:
        base, original, _ = _layer(device, "olora")
        assert not bool((base.weight == original).all().item())

    def test_a_rank_beyond_the_weight_is_refused(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(8, 6).to(device)
        with pytest.raises(ValueError, match="r <= min"):
            Linear(base, "default", r=7, init_lora_weights="olora")


class TestRefusals:
    @pytest.mark.parametrize("init", ["pissa", "olora", "pissa_niter_2"])
    def test_layers_without_a_dense_weight_refuse_the_factorising_inits(
        self, init: str, device: str
    ) -> None:
        conv = nn.Conv2d(3, 4, kernel_size=3).to(device)
        embedding = nn.Embedding(9, 4).to(device)
        attention = nn.MultiheadAttention(8, 2, batch_first=True).to(device)
        for build in (
            lambda: Conv2d(conv, "default", r=2, init_lora_weights=init),
            lambda: Embedding(embedding, "default", r=2, init_lora_weights=init),
            lambda: MultiheadAttention(attention, "default", r=2, init_lora_weights=init),
        ):
            with pytest.raises(ValueError, match="dense"):
                build()

    @pytest.mark.parametrize("init", ["eva", "corda", "loftq", "lora_ga", "orthogonal", "mica"])
    def test_the_reference_s_other_inits_say_what_is_missing(self, init: str) -> None:
        with pytest.raises(NotImplementedError, match=f"{init}.*not implemented in lucid.peft"):
            _check_init("init_lora_weights", init)

    def test_unknown_names_are_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown initialization"):
            _check_init("init_lora_weights", "kaiming")

    @pytest.mark.parametrize("bad", ["pissa_niter_", "pissa_niter_x", "pissa_niter_-1", "pissax"])
    def test_a_malformed_pissa_spelling_is_refused(self, bad: str) -> None:
        with pytest.raises(ValueError, match="pissa_niter_"):
            _check_init("init_lora_weights", bad)

    def test_a_non_string_non_bool_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="init_lora_weights"):
            _check_init("init_lora_weights", 1)

    def test_a_failed_init_leaves_no_adapter(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(8, 6).to(device)
        layer = Linear(base, "default", r=2)
        with pytest.raises(ValueError, match="r <= min"):
            layer.update_layer("big", 7, 7.0, 0.0, "pissa", False)
        assert list(layer.lora_A) == ["default"] and layer.r == {"default": 2}


class TestStrategies:
    def test_the_spellings_are_registered_by_class(self) -> None:
        assert {"default", "random", "gaussian", "pissa", "olora"} <= set(_Init._strategies)

    def test_canonical_values(self) -> None:
        assert _Init.of(True).value is True and _Init.of(False).value is False
        assert _Init.of("Gaussian").value == "gaussian"
        assert _Init.of("pissa").value == "pissa"
        pissa = _Init.of("pissa_niter_16")
        assert isinstance(pissa, _Pissa) and pissa.niter == 16 and pissa.value == "pissa_niter_16"
        assert isinstance(_Init.of("OLoRA"), _Olora)

    def test_a_new_spelling_is_one_subclass(
        self, device: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_Init, "_strategies", dict(_Init._strategies))

        class _Ones(_Init, init="ones_b"):
            __slots__ = ()

            @property
            def value(self) -> str:
                return "ones_b"

            def apply(self, layer, adapter) -> None:  # type: ignore[no-untyped-def]
                layer._reset_factors(adapter, gaussian=False)
                with lucid.no_grad():
                    layer._factor_pairs(adapter)[0][1].copy_(
                        lucid.ones_like(layer._factor_pairs(adapter)[0][1])
                    )

        _, _, layer = _layer(device, "ones_b")
        assert bool((layer.lora_B["default"].weight == 1).all().item())

    def test_the_reset_method_applies_a_spelling_again(self, device: str) -> None:
        _, _, layer = _layer(device, False)
        layer.reset_lora_parameters("default", True)
        assert float(layer.lora_B["default"].weight.abs().max().item()) == 0.0


class TestEmbeddingAndConv:
    def test_embedding_gaussian_is_the_default_init(self, device: str) -> None:
        lucid.manual_seed(0)
        layer = Embedding(
            nn.Embedding(9, 4).to(device), "default", r=2, init_lora_weights="gaussian"
        )
        assert float(layer.lora_embedding_A["default"].abs().max().item()) == 0.0

    def test_conv_gaussian(self, device: str) -> None:
        lucid.manual_seed(0)
        layer = Conv2d(
            nn.Conv2d(3, 4, kernel_size=3).to(device), "default", r=2, init_lora_weights="gaussian"
        )
        assert float(layer.lora_B["default"].weight.abs().max().item()) == 0.0


class TestRsloraWarning:
    def test_pissa_with_rslora_and_a_pattern_warns(self) -> None:
        with pytest.warns(UserWarning, match="Rank-Stabilized"):
            LoraConfig(use_rslora=True, rank_pattern={"q": 2}, init_lora_weights="pissa")

    def test_plain_rslora_with_a_pattern_does_not(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LoraConfig(use_rslora=True, rank_pattern={"q": 2})
