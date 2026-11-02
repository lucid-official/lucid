"""``lucid.nn.utils.fusion``: every conv family and Linear fuse to the same function.

The fused module must compute what ``bn(layer(x))`` computes in eval mode, on
CPU and Metal, and the entry points must refuse what the fold is wrong for
(train mode, a BatchNorm with no running statistics, a feature-count mismatch)
at the boundary instead of returning a quietly wrong layer.
"""

from collections.abc import Callable
from typing import Any

import pytest

import lucid
import lucid.nn as nn
from lucid.nn.utils import fusion
from lucid.test._fixtures.devices import metal_available

DEVICES: list[str] = ["cpu", "metal"] if metal_available() else ["cpu"]

# (conv factory, bn factory, input shape, transpose)
CONV_CASES: dict[str, tuple[Callable[[], Any], Callable[[int], Any], tuple[int, ...], bool]] = {
    "conv1d": (lambda: nn.Conv1d(3, 6, 3), nn.BatchNorm1d, (2, 3, 9), False),
    "conv2d": (lambda: nn.Conv2d(3, 6, 3, padding=1), nn.BatchNorm2d, (2, 3, 7, 7), False),
    "conv2d_grouped": (lambda: nn.Conv2d(4, 6, 3, groups=2), nn.BatchNorm2d, (2, 4, 7, 7), False),
    "conv3d_nobias": (
        lambda: nn.Conv3d(3, 6, 3, bias=False),
        nn.BatchNorm3d,
        (2, 3, 5, 5, 5),
        False,
    ),
    "convt1d": (lambda: nn.ConvTranspose1d(3, 6, 3), nn.BatchNorm1d, (2, 3, 9), True),
    "convt2d": (lambda: nn.ConvTranspose2d(3, 6, 3, stride=2), nn.BatchNorm2d, (2, 3, 5, 5), True),
    "convt2d_grouped": (
        lambda: nn.ConvTranspose2d(4, 6, 3, groups=2),
        nn.BatchNorm2d,
        (2, 4, 5, 5),
        True,
    ),
    "convt3d_nobias": (
        lambda: nn.ConvTranspose3d(3, 6, 3, bias=False),
        nn.BatchNorm3d,
        (2, 3, 4, 4, 4),
        True,
    ),
}


def _eval_pair(conv: Any, bn: Any, device: str) -> tuple[Any, Any]:
    """Put both modules on ``device`` in eval mode with non-trivial statistics."""
    conv, bn = conv.to(device=device), bn.to(device=device)
    features = bn.num_features
    bn.running_mean.copy_(lucid.randn(features, device=device))
    bn.running_var.copy_(lucid.rand(features, device=device) + 0.5)
    if bn.affine:
        bn.weight.data.copy_(lucid.randn(features, device=device))
        bn.bias.data.copy_(lucid.randn(features, device=device))
    return conv.eval(), bn.eval()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CONV_CASES)
def test_fused_conv_matches_the_unfused_pair(case: str, device: str) -> None:
    make_conv, make_bn, shape, transpose = CONV_CASES[case]
    conv, bn = _eval_pair(make_conv(), make_bn(6), device)
    x = lucid.randn(*shape, device=device)
    fused = fusion.fuse_conv_bn_eval(conv, bn, transpose=transpose)
    assert type(fused) is type(conv)
    assert (fused(x) - bn(conv(x))).abs().max().item() < 1e-4


@pytest.mark.parametrize("device", DEVICES)
def test_fused_conv_without_affine_matches(device: str) -> None:
    conv, bn = _eval_pair(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6, affine=False), device)
    x = lucid.randn(2, 3, 7, 7, device=device)
    fused = fusion.fuse_conv_bn_eval(conv, bn)
    assert (fused(x) - bn(conv(x))).abs().max().item() < 1e-4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_fused_linear_matches_the_unfused_pair(bias: bool, device: str) -> None:
    linear, bn = _eval_pair(nn.Linear(5, 6, bias=bias), nn.BatchNorm1d(6), device)
    x = lucid.randn(4, 5, device=device)
    fused = fusion.fuse_linear_bn_eval(linear, bn)
    assert (fused(x) - bn(linear(x))).abs().max().item() < 1e-4
    assert tuple(fused.bias.shape) == (6,)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_linear_with_a_one_feature_batchnorm_broadcasts(device: str) -> None:
    # BatchNorm1d(1) normalises dim 1 of a (N, 1, H) input, so it applies to every output.
    linear, bn = _eval_pair(nn.Linear(5, 6), nn.BatchNorm1d(1), device)
    x = lucid.randn(4, 1, 5, device=device)
    fused = fusion.fuse_linear_bn_eval(linear, bn)
    assert (fused(x) - bn(linear(x))).abs().max().item() < 1e-4


def test_originals_are_untouched_and_results_are_parameters() -> None:
    conv, bn = _eval_pair(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6), "cpu")
    before = (conv.weight.tolist(), conv.bias.tolist())
    fused = fusion.fuse_conv_bn_eval(conv, bn)
    assert (conv.weight.tolist(), conv.bias.tolist()) == before
    assert isinstance(fused.weight, nn.Parameter) and isinstance(fused.bias, nn.Parameter)
    assert fused.weight.requires_grad


def test_weights_functions_return_parameters_of_the_source_dtype() -> None:
    weight = lucid.randn(6, 3, 3, 3).to(dtype=lucid.float16)
    ones, zeros = lucid.ones(6), lucid.zeros(6)
    w, b = fusion.fuse_conv_bn_weights(weight, None, zeros, ones, 1e-5, None, None)
    assert isinstance(w, nn.Parameter) and isinstance(b, nn.Parameter)
    assert w.dtype == lucid.float16 and b.dtype == lucid.float16
    wt, _ = fusion.fuse_conv_bn_weights(
        lucid.randn(3, 6, 3, 3), None, zeros, ones, 1e-5, None, None, transpose=True
    )
    assert tuple(wt.shape) == (3, 6, 3, 3)


class TestRefusals:
    """What the fold is wrong for is refused at the boundary, with the reference's classes."""

    @pytest.mark.parametrize("which", ["conv", "bn", "both"])
    def test_train_mode_is_refused(self, which: str) -> None:
        conv, bn = _eval_pair(nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6), "cpu")
        if which in ("conv", "both"):
            conv.train()
        if which in ("bn", "both"):
            bn.train()
        with pytest.raises(AssertionError, match="Fusion only for eval!"):
            fusion.fuse_conv_bn_eval(conv, bn)

    def test_linear_train_mode_is_refused(self) -> None:
        linear, bn = _eval_pair(nn.Linear(5, 6), nn.BatchNorm1d(6), "cpu")
        with pytest.raises(AssertionError, match="Fusion only for eval!"):
            fusion.fuse_linear_bn_eval(linear.train(), bn)
        with pytest.raises(AssertionError, match="Fusion only for eval!"):
            fusion.fuse_linear_bn_eval(linear.eval(), bn.train())

    def test_bn_without_running_statistics_is_refused(self) -> None:
        bn = nn.BatchNorm2d(6, track_running_stats=False).eval()
        with pytest.raises(AssertionError, match="must not be None"):
            fusion.fuse_conv_bn_eval(nn.Conv2d(3, 6, 3).eval(), bn)
        bn1 = nn.BatchNorm1d(6, track_running_stats=False).eval()
        with pytest.raises(AssertionError, match="must not be None"):
            fusion.fuse_linear_bn_eval(nn.Linear(5, 6).eval(), bn1)

    def test_conv_feature_mismatch_names_both_counts(self) -> None:
        conv, bn = nn.Conv2d(3, 6, 3).eval(), nn.BatchNorm2d(4).eval()
        with pytest.raises(ValueError, match="4 features .* 6 output channels"):
            fusion.fuse_conv_bn_eval(conv, bn)

    def test_linear_feature_mismatch_uses_the_reference_message(self) -> None:
        linear, bn = nn.Linear(5, 6).eval(), nn.BatchNorm1d(4).eval()
        with pytest.raises(AssertionError, match=r"linear\.out_features=6 and bn\.num_features=4"):
            fusion.fuse_linear_bn_eval(linear, bn)

    def test_weights_feature_mismatch_names_both_counts(self) -> None:
        zeros = lucid.zeros(4)
        ones = zeros + 1
        with pytest.raises(ValueError, match="4 features"):
            fusion.fuse_conv_bn_weights(
                lucid.randn(6, 3, 3, 3), None, zeros, ones, 1e-5, None, None
            )
        with pytest.raises(ValueError, match="4 features"):
            fusion.fuse_linear_bn_weights(lucid.randn(6, 5), None, zeros, ones, 1e-5, None, None)

    @pytest.mark.parametrize("transpose", [True, False])
    def test_transpose_must_match_the_module(self, transpose: bool) -> None:
        conv = (nn.Conv2d(3, 6, 3) if transpose else nn.ConvTranspose2d(3, 6, 3)).eval()
        with pytest.raises(ValueError, match="does not match"):
            fusion.fuse_conv_bn_eval(conv, nn.BatchNorm2d(6).eval(), transpose=transpose)

    def test_wrong_module_types_are_refused(self) -> None:
        with pytest.raises(TypeError, match="conv must be"):
            fusion.fuse_conv_bn_eval(nn.Linear(2, 2), nn.BatchNorm1d(2))
        with pytest.raises(TypeError, match="bn must be"):
            fusion.fuse_conv_bn_eval(nn.Conv2d(3, 6, 3), nn.Linear(2, 2))
        with pytest.raises(TypeError, match="linear must be Linear"):
            fusion.fuse_linear_bn_eval(nn.Conv2d(3, 6, 3), nn.BatchNorm1d(2))
        with pytest.raises(TypeError, match="bn must be a BatchNorm"):
            fusion.fuse_linear_bn_eval(nn.Linear(2, 2), nn.Linear(2, 2))
