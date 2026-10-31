"""``lora.ConvTranspose{1,2,3}d``: a transposed convolution, then a 1x1 one.

The two compose into one transposed convolution with the kernel
``s * sum_r A[i, r, k] B[o, r]`` — checked against a real
``F.conv_transposeNd`` of the base's geometry (stride, padding, output padding,
dilation), not against the layer's own arithmetic.  The kernel's axes are
``(in, out, *k)``, so DoRA's per-output-unit norm runs along axis 1.
"""

import warnings

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.peft import LoraConfig, get_peft_model
from lucid.peft.tuners import lora

#: ``dims`` → ``(base class, lora class, functional, input shape)``
TRANSPOSE = {
    1: (nn.ConvTranspose1d, lora.ConvTranspose1d, F.conv_transpose1d, (2, 3, 7)),
    2: (nn.ConvTranspose2d, lora.ConvTranspose2d, F.conv_transpose2d, (2, 3, 5, 6)),
    3: (nn.ConvTranspose3d, lora.ConvTranspose3d, F.conv_transpose3d, (1, 3, 3, 3, 4)),
}

#: Geometries a base may have, each as the keyword arguments of its constructor.
GEOMETRIES = {
    "plain": {},
    "padded": {"padding": 1},
    "strided": {"stride": 2, "padding": 1},
    "strided_output_padded": {"stride": 2, "padding": 1, "output_padding": 1},
    "dilated": {"dilation": 2, "padding": 2},
}


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-4) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _transposed(
    dims: int, device: str, geometry: str = "strided", **kwargs: object
) -> tuple[nn.Module, lora.LoraLayer, lucid.Tensor]:
    kind, layer_kind, _, shape = TRANSPOSE[dims]
    lucid.manual_seed(0)
    base = kind(3, 4, 3, **GEOMETRIES[geometry]).to(device)  # type: ignore[arg-type]
    options: dict[str, object] = {"r": 2, "lora_alpha": 6, "init_lora_weights": False} | kwargs
    layer = layer_kind(base, "default", **options)
    return base, layer, lucid.randn(*shape, device=device)


class TestConvTranspose:
    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_the_factors_are_a_transposed_conv_and_a_one_wide_conv(
        self, dims: int, device: str
    ) -> None:
        base, layer, _ = _transposed(dims, device)
        a, b = layer.lora_A["default"], layer.lora_B["default"]
        assert isinstance(a, TRANSPOSE[dims][0]) and not isinstance(b, TRANSPOSE[dims][0])
        assert tuple(a.weight.shape) == (3, 2, *(3,) * dims)  # (in, r, *k)
        assert tuple(b.weight.shape) == (4, 2, *(1,) * dims)  # (out, r, 1..)
        assert a.bias is None and b.bias is None
        assert sorted(layer.state_dict()) == [
            "base_layer.bias",
            "base_layer.weight",
            "lora_A.default.weight",
            "lora_B.default.weight",
        ]
        assert layer.weight is base.weight and layer.bias is base.bias
        assert repr(layer).startswith(f"lora.ConvTranspose{dims}d(")

    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_a_zero_b_is_the_base_bit_for_bit(self, dims: int, device: str) -> None:
        base, layer, x = _transposed(dims, device, init_lora_weights=True)
        assert _same(layer(x), base(x))

    @pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_the_two_layers_are_one_transposed_conv_with_the_delta_kernel(
        self, dims: int, geometry: str, device: str
    ) -> None:
        base, layer, x = _transposed(dims, device, geometry)
        _, _, transpose_fn, _ = TRANSPOSE[dims]
        delta = layer.get_delta_weight("default")
        assert tuple(delta.shape) == tuple(base.weight.shape)
        options = {
            name: getattr(base, name)
            for name in ("stride", "padding", "output_padding", "dilation")
        }
        assert _close(layer(x) - base(x), transpose_fn(x, delta, **options))

    @pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_merging_gives_the_unmerged_output_and_unmerging_the_kernel(
        self, dims: int, geometry: str, device: str
    ) -> None:
        base, layer, x = _transposed(dims, device, geometry)
        with lucid.no_grad():
            adapted = layer(x)
            weight, kernel = base.weight, base.weight.clone()
            layer.merge()
            assert base.weight is weight and layer.merged
            assert _close(layer(x), adapted)
            assert _close(base(x), adapted)  # the base alone now computes the adapted layer
            layer.unmerge()
            assert _close(base.weight, kernel, atol=1e-5)

    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_dora_rescales_the_output_channels_and_merges(self, dims: int, device: str) -> None:
        base, layer, x = _transposed(dims, device, use_dora=True)
        with lucid.no_grad():
            adapted = layer(x)
            kernel = base.weight.clone()
            # The magnitude starts at the norm of W + sBA per output channel (axis 1 of the kernel).
            total = base.weight + layer.get_delta_weight("default")
            norms = (total * total).sum(dim=[0, *range(2, total.ndim)]).sqrt()
            assert _close(layer.lora_magnitude_vector["default"].weight, norms)
            layer.lora_magnitude_vector["default"].weight.mul_(1.5)
            adapted = layer(x)
            layer.merge()
            assert _close(layer(x), adapted)
            layer.unmerge()
            assert _close(base.weight, kernel, atol=1e-5)

    def test_lora_bias_is_scaled_and_merged_into_the_base_bias(self, device: str) -> None:
        base, layer, x = _transposed(2, device, lora_bias=True)
        with lucid.no_grad():
            layer.lora_B["default"].bias.copy_(lucid.randn(4, device=device))
            adapted = layer(x)
            bias = base.bias.clone()
            layer.merge()
            assert not _same(base.bias, bias)
            assert _close(layer(x), adapted)
            layer.unmerge()
            assert _close(base.bias, bias, atol=1e-5)

    def test_gradients_reach_both_factors_and_not_the_frozen_base(self, device: str) -> None:
        base, layer, x = _transposed(2, device)
        base.requires_grad_(False)
        layer(x).sum().backward()
        assert layer.lora_A["default"].weight.grad is not None
        assert layer.lora_B["default"].weight.grad is not None
        assert base.weight.grad is None

    def test_a_bfloat16_base_gets_a_float32_adapter(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.ConvTranspose2d(3, 4, 3, stride=2).to(device).bfloat16()
        layer = lora.ConvTranspose2d(base, "default", r=2, lora_alpha=2, init_lora_weights=False)
        assert layer.lora_A["default"].weight.dtype == lucid.float32
        x = lucid.randn(1, 3, 4, 4, device=device).bfloat16()
        assert layer(x).dtype == lucid.bfloat16

    def test_the_tuner_wraps_each_transposed_convolution_by_its_type(self, device: str) -> None:
        lucid.manual_seed(0)
        model = nn.Sequential(
            nn.ConvTranspose1d(3, 4, 3), nn.ConvTranspose2d(4, 4, 3), nn.ConvTranspose3d(4, 4, 3)
        ).to(device)
        get_peft_model(model, LoraConfig(r=2, target_modules=["0", "1", "2"]))
        assert [type(m) for m in model] == [
            lora.ConvTranspose1d,
            lora.ConvTranspose2d,
            lora.ConvTranspose3d,
        ]

    def test_the_tuner_trains_and_merges_a_decoder(self, device: str) -> None:
        lucid.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, 4, stride=2, padding=1),
        ).to(device)
        peft = get_peft_model(
            model, LoraConfig(r=2, target_modules=["2"], init_lora_weights=False)
        ).eval()
        x = lucid.randn(1, 3, 6, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            merged = peft.merge_and_unload()
            assert isinstance(merged[2], nn.ConvTranspose2d)
            assert _close(merged(x), adapted)


def _asked_size(x: lucid.Tensor, base: nn.Module) -> list[int]:
    """The size one past the default on every spatial axis: the base's ``output_padding`` raised by one."""
    default = base(x).shape[2:]
    return [n + 1 for n in default]


class TestOutputSize:
    """An adapted transposed convolution takes the base's ``output_size``, and the update follows it."""

    @pytest.mark.parametrize("dims", [1, 2, 3])
    def test_the_output_is_the_size_asked_for_and_the_update_is_at_that_size(
        self, dims: int, device: str
    ) -> None:
        base, layer, x = _transposed(dims, device)
        size = _asked_size(x, base)
        out = layer(x, output_size=size)
        assert list(out.shape[2:]) == size
        branch = layer.lora_B["default"](layer.lora_A["default"](x, output_size=size))
        assert _close(out, base(x, output_size=size) + branch * layer.scaling["default"])
        assert not _same(out, base(x, output_size=size))  # a non-zero update

    def test_it_may_be_given_positionally(self, device: str) -> None:
        base, layer, x = _transposed(2, device)
        size = _asked_size(x, base)
        assert _same(layer(x, size), layer(x, output_size=size))

    def test_without_it_the_output_is_the_default_size(self, device: str) -> None:
        base, layer, x = _transposed(2, device)
        assert _same(layer(x), layer(x, output_size=None))
        assert layer(x).shape == base(x).shape

    def test_a_size_the_stride_cannot_give_is_the_base_layers_error(self, device: str) -> None:
        _, layer, x = _transposed(2, device)
        with pytest.raises(ValueError, match="requested an output size"):
            layer(x, output_size=[40, 40])

    def test_a_merged_layer_gives_the_unmerged_output_at_that_size(self, device: str) -> None:
        base, layer, x = _transposed(2, device)
        size = _asked_size(x, base)
        before = layer(x, output_size=size)
        layer.merge()
        assert _close(layer(x, output_size=size), before)

    def test_dora_follows_it_with_and_without_dropout(self, device: str) -> None:
        base, layer, x = _transposed(2, device, use_dora=True, lora_dropout=0.5)
        size = _asked_size(x, base)
        layer.eval()
        eval_out = layer(x, output_size=size)
        layer.train()
        assert list(eval_out.shape[2:]) == size
        assert list(layer(x, output_size=size).shape[2:]) == size

    def test_every_adapted_layer_takes_what_its_base_forward_takes(self) -> None:
        """The class-wide guard: a layer type registered for a base type keeps that type's forward parameters."""
        from lucid.peft.tuners.lora._layer import _Residual, _extra_parameters

        layers = {
            base: layer
            for base, layer in lora.LoraLayer._layer_types.items()
            if issubclass(layer, _Residual)
        }
        assert nn.ConvTranspose2d in layers
        for base, layer in layers.items():
            assert set(_extra_parameters(base.forward)) <= set(_extra_parameters(layer.forward))

    def test_a_layer_that_drops_a_base_parameter_fails_at_its_class(self) -> None:
        from lucid.peft.tuners.lora._conv import _ConvLayer

        with pytest.raises(TypeError, match="output_size"):

            class _Lossy(_ConvLayer, base_type=nn.ConvTranspose2d):  # type: ignore[type-var,misc]
                pass

        assert lora.LoraLayer._layer_types[nn.ConvTranspose2d] is lora.ConvTranspose2d

    @pytest.mark.parametrize("positional", [False, True])
    def test_a_mixed_batch_routes_rows_at_that_size(self, positional: bool, device: str) -> None:
        class _Up(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.up = nn.ConvTranspose2d(3, 4, 3, stride=2, padding=1)

            def forward(
                self, x: lucid.Tensor, output_size: list[int] | None = None
            ) -> lucid.Tensor:
                if positional:
                    return self.up(x, output_size)
                return self.up(x, output_size=output_size)

        lucid.manual_seed(0)
        config = LoraConfig(r=2, lora_alpha=4, target_modules=["up"], init_lora_weights=False)
        peft = get_peft_model(_Up().to(device), config, adapter_name="a")
        peft.add_adapter("b", config)
        peft.eval()
        x = lucid.randn(3, 3, 5, 6, device=device)
        names = ["a", "b", "__base__"]
        size = [10, 12]
        with lucid.no_grad():
            mixed = peft(x, adapter_names=names, output_size=size)
            rows = []
            for i, name in enumerate(names):
                if name == "__base__":
                    with peft.disable_adapter():
                        rows.append(peft(x[i : i + 1], output_size=size))
                else:
                    peft.set_adapter(name)
                    rows.append(peft(x[i : i + 1], output_size=size))
        assert list(mixed.shape[2:]) == size
        assert _close(mixed, lucid.cat(rows))


class TestGroupedConvTranspose:
    def test_a_grouped_base_works_forward_and_cannot_merge(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.ConvTranspose2d(4, 6, 3, padding=1, groups=2).to(device)
        with pytest.warns(UserWarning, match="groups > 1"):
            layer = lora.ConvTranspose2d(
                base, "default", r=2, lora_alpha=2, init_lora_weights=False
            )
        x = lucid.randn(1, 4, 5, 5, device=device)
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        expected = base(x) + F.conv2d(F.conv_transpose2d(x, a, padding=1), b, groups=2)
        assert _close(layer(x), expected)
        for call in (layer.merge, lambda: layer.get_delta_weight("default")):
            with pytest.raises(NotImplementedError, match="groups > 1"):
                call()
        assert not layer.merged

    def test_the_rank_must_divide_the_groups(self, device: str) -> None:
        base = nn.ConvTranspose2d(4, 6, 3, groups=2).to(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="divisible by groups"):
                lora.ConvTranspose2d(base, "default", r=3, lora_alpha=3)
