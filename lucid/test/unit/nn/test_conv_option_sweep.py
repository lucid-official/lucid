"""Every convolution variant, across the options that select a code path.

``conv.py`` sat at 43.7% line coverage, and almost all of the missing
lines were *body* rather than error handling: the ``same``-padding
arithmetic, the transposed-convolution output-size computation, the
grouped and dilated branches.  Those are not corners — they are the
options a user picks — and none of them ran.

Shapes are asserted against the arithmetic rather than against recorded
numbers, so a test that passes says the formula is right and not merely
that it has not changed.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

RANKS = [1, 2, 3]


def _input(rank: int, channels: int = 3, size: int = 8, batch: int = 2):
    shape = (batch, channels) + (size,) * rank
    return lucid.tensor(np.random.default_rng(rank).standard_normal(shape).astype(np.float32))


def _out_size(size, kernel, stride, padding, dilation):
    effective = dilation * (kernel - 1) + 1
    return (size + 2 * padding - effective) // stride + 1


# ── the option space ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dilation", [1, 2])
def test_conv_output_shape_follows_the_arithmetic(rank, stride, padding, dilation):
    layer = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank](
        3, 4, kernel_size=3, stride=stride, padding=padding, dilation=dilation
    )
    out = layer(_input(rank))
    expected = _out_size(8, 3, stride, padding, dilation)
    assert tuple(out.shape) == (2, 4) + (expected,) * rank


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("groups", [1, 3])
def test_grouped_convolution(rank, groups):
    layer = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank](
        3, 6, kernel_size=3, padding=1, groups=groups
    )
    out = layer(_input(rank))
    assert tuple(out.shape) == (2, 6) + (8,) * rank
    assert layer.weight.shape[1] == 3 // groups


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("padding", ["same", "valid"])
def test_string_padding_modes(rank, padding):
    """``same`` keeps the spatial size; ``valid`` is no padding at all."""
    layer = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank](3, 4, kernel_size=3, padding=padding)
    out = layer(_input(rank))
    expected = 8 if padding == "same" else 6
    assert tuple(out.shape) == (2, 4) + (expected,) * rank


@pytest.mark.parametrize("rank", RANKS)
def test_same_padding_with_an_even_kernel(rank):
    """An even kernel cannot pad symmetrically, so the split is uneven —
    the branch that computes ``pad_lo`` separately from ``pad_hi``."""
    layer = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank](3, 4, kernel_size=4, padding="same")
    assert tuple(layer(_input(rank)).shape) == (2, 4) + (8,) * rank


@pytest.mark.parametrize("rank", RANKS)
def test_bias_free_convolution(rank):
    layer = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank](
        3, 4, kernel_size=3, padding=1, bias=False
    )
    assert layer.bias is None
    assert tuple(layer(_input(rank)).shape) == (2, 4) + (8,) * rank


# ── transposed ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize(("stride", "output_padding"), [(1, 0), (2, 0), (2, 1)])
def test_conv_transpose_output_shape(rank, stride, output_padding):
    """The usual combinations, where ``output_padding`` is below the stride.

    ``output_padding`` at the stride or above is refused, asserted by
    the test below.
    """
    layer = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[rank](
        3, 4, kernel_size=3, stride=stride, padding=1, output_padding=output_padding
    )
    out = layer(_input(rank))
    expected = (8 - 1) * stride - 2 * 1 + 3 + output_padding
    assert tuple(out.shape) == (2, 4) + (expected,) * rank


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize(
    ("stride", "dilation", "output_padding"), [(1, 1, 1), (2, 1, 2), (2, 1, -1)]
)
def test_output_padding_outside_the_stride_is_refused(rank, stride, dilation, output_padding):
    """``output_padding`` must be non-negative and below the stride or dilation.

    It picks one of the ``stride`` input sizes a strided convolution maps to
    the same output; at the stride or above it reaches past every input
    position.  The engine used to answer anyway, with rows holding the bias
    alone — both the module and the function now refuse, as the reference
    framework does.
    """
    cls = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[rank]
    fn = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}[rank]
    x = _input(rank)
    layer = cls(
        3,
        4,
        kernel_size=3,
        stride=stride,
        padding=1,
        dilation=dilation,
        output_padding=output_padding,
    )
    with pytest.raises(ValueError, match="output_padding"):
        layer(x)
    with pytest.raises(ValueError, match="output_padding"):
        fn(
            x,
            layer.weight,
            stride=stride,
            padding=1,
            dilation=dilation,
            output_padding=output_padding,
        )


@pytest.mark.parametrize("rank", RANKS)
def test_a_dilation_above_the_stride_admits_a_larger_output_padding(rank):
    """Below the dilation is enough, even when it is not below the stride."""
    cls = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[rank]
    layer = cls(3, 4, kernel_size=3, stride=1, padding=1, dilation=2, output_padding=1)
    out = layer(_input(rank))
    expected = (8 - 1) * 1 - 2 * 1 + 2 * (3 - 1) + 1 + 1
    assert tuple(out.shape) == (2, 4) + (expected,) * rank


@pytest.mark.parametrize("rank", RANKS)
def test_transpose_inverts_the_stride_of_a_convolution(rank):
    """A stride-2 conv halves the size and its transpose restores it."""
    down = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank](
        3, 4, kernel_size=3, stride=2, padding=1
    )
    up = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[rank](
        4, 3, kernel_size=3, stride=2, padding=1, output_padding=1
    )
    assert tuple(up(down(_input(rank))).shape) == (2, 3) + (8,) * rank


# ── gradients reach every parameter ───────────────────────────────────────────


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("transposed", [False, True])
def test_every_parameter_receives_a_gradient(rank, transposed):
    cls = {
        (1, False): nn.Conv1d,
        (2, False): nn.Conv2d,
        (3, False): nn.Conv3d,
        (1, True): nn.ConvTranspose1d,
        (2, True): nn.ConvTranspose2d,
        (3, True): nn.ConvTranspose3d,
    }[(rank, transposed)]
    layer = cls(3, 4, kernel_size=3, padding=1)
    layer(_input(rank)).sum().backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, name
        assert np.abs(np.asarray(param.grad.numpy())).sum() > 0.0, name


# ── the functional forms agree with the layers ────────────────────────────────


@pytest.mark.parametrize("rank", RANKS)
def test_functional_matches_the_module(rank):
    layer = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank](3, 4, kernel_size=3, padding=1)
    fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
    x = _input(rank)
    assert np.allclose(
        np.asarray(layer(x).numpy()),
        np.asarray(fn(x, layer.weight, layer.bias, padding=1).numpy()),
        atol=1e-5,
    )


@pytest.mark.parametrize("rank", RANKS)
def test_repr_names_the_configuration(rank):
    layer = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank](
        3, 4, kernel_size=3, stride=2, padding=1
    )
    text = repr(layer)
    assert "3" in text and "4" in text


# ── the same options on the other stream ──────────────────────────────────────


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("groups", [1, 2])
def test_grouped_convolution_agrees_across_devices(rank, groups):
    """MLX declines grouping above two spatial dimensions.

    A grouped ``conv3d`` therefore used to run on the CPU stream and
    raise ``Can only handle groups != 1 in 1D or 2D convolutions`` on
    Metal — the same model working on one device and not the other. The
    GPU path now falls back to one convolution per group, so this asserts
    both that it runs and that the group blocks are paired correctly.
    """
    x = _input(rank, channels=4, size=5)
    fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]
    weight = lucid.tensor(
        np.random.default_rng(rank + groups)
        .standard_normal((4, 4 // groups) + (3,) * rank)
        .astype(np.float32)
    )
    bias = lucid.tensor(np.zeros(4, dtype=np.float32))

    cpu = fn(x, weight, bias, padding=1, groups=groups)
    metal = fn(x.to("metal"), weight.to("metal"), bias.to("metal"), padding=1, groups=groups)
    assert tuple(metal.shape) == tuple(cpu.shape)
    scale = max(float(cpu.abs().max().item()), 1e-30)
    assert float((cpu - metal.to("cpu")).abs().max().item()) / scale < 1e-5
