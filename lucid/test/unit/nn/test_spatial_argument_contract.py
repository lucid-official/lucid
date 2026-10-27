"""The spatial functionals read their arguments the way the reference does (LCD-405).

Convolution, transposed convolution and pooling each used to read their own
arguments, and they drifted from the reference and from each other:

- ``conv2d(x, w, padding="same")`` split the string into its characters;
- ``max_pool2d(x, 2, stride=0)`` pooled with the kernel's stride, because
  the engine reads 0 as "the kernel size";
- no op took an unbatched input.

Every spatial entry point now reads its arguments through one normaliser,
``lucid.nn.functional._spatial_args``.  These cases hold the whole matrix
to the reference — batched and unbatched input, every padding and stride
spelling, the functional and the module — on CPU and Metal, and check that
the reference's refusals are refusals here too.
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

pytestmark = pytest.mark.parity

RANKS = (1, 2, 3)
_SIZE = 7


def _data(*shape: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _input(rank: int, batched: bool, channels: int = 3) -> np.ndarray:
    lead = (2, channels) if batched else (channels,)
    return _data(*lead, *(_SIZE,) * rank)


def _close(got: lucid.Tensor, want: object, atol: float = 1e-4) -> None:
    want_np = want.detach().numpy()
    assert tuple(got.shape) == tuple(want_np.shape)
    np.testing.assert_allclose(got.numpy(), want_np, rtol=1e-4, atol=atol)


def _per_axis(value: int, rank: int, bump: int) -> tuple[int, ...]:
    """A per-axis spelling whose axes differ, so a swapped axis shows."""
    return tuple(value + (bump if axis == rank - 1 else 0) for axis in range(rank))


# ── convolution ───────────────────────────────────────────────────────────────

_CONV_PADDINGS = ("0", "1", "per_axis", "valid", "same")
_CONV_STRIDES = ("1", "2", "per_axis")


def _conv_padding(spelling: str, rank: int) -> object:
    if spelling in ("valid", "same"):
        return spelling
    if spelling == "per_axis":
        return _per_axis(0, rank, 1)
    return int(spelling)


def _conv_stride(spelling: str, rank: int) -> object:
    if spelling == "per_axis":
        return _per_axis(1, rank, 1)
    return int(spelling)


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("padding", _CONV_PADDINGS)
@pytest.mark.parametrize("stride", _CONV_STRIDES)
def test_conv_matches_the_reference(
    ref: ModuleType, device: str, rank: int, batched: bool, padding: str, stride: str
) -> None:
    pad = _conv_padding(padding, rank)
    step = _conv_stride(stride, rank)
    if pad == "same" and stride != "1":
        pytest.skip("covered by test_same_padding_refuses_a_stride")
    x = _input(rank, batched)
    # An even kernel: "same" then pads an odd total, one more after.
    w = _data(4, 3, *(4,) * rank, seed=1)
    b = _data(4, seed=2)
    fn = getattr(F, f"conv{rank}d")
    ref_fn = getattr(ref.nn.functional, f"conv{rank}d")
    got = fn(
        lucid.tensor(x, device=device),
        lucid.tensor(w, device=device),
        lucid.tensor(b, device=device),
        stride=step,
        padding=pad,
        dilation=1,
    )
    want = ref_fn(ref.from_numpy(x), ref.from_numpy(w), ref.from_numpy(b), stride=step, padding=pad)
    _close(got, want)


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("padding", ["1", "same", "valid"])
@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_conv_module_matches_the_reference(
    ref: ModuleType, device: str, rank: int, batched: bool, padding: str, mode: str
) -> None:
    pad = _conv_padding(padding, rank)
    kwargs = dict(kernel_size=4, padding=pad, padding_mode=mode, dilation=2)
    layer = getattr(nn, f"Conv{rank}d")(3, 4, device=device, **kwargs)
    ref_layer = getattr(ref.nn, f"Conv{rank}d")(3, 4, **kwargs)
    with ref.no_grad():
        ref_layer.weight.copy_(ref.from_numpy(layer.weight.numpy()))
        ref_layer.bias.copy_(ref.from_numpy(layer.bias.numpy()))
    x = _input(rank, batched)
    _close(layer(lucid.tensor(x, device=device)), ref_layer(ref.from_numpy(x)))


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("stride", _CONV_STRIDES)
@pytest.mark.parametrize("padding", ["0", "1", "per_axis"])
def test_conv_transpose_matches_the_reference(
    ref: ModuleType, device: str, rank: int, batched: bool, stride: str, padding: str
) -> None:
    step = _conv_stride(stride, rank)
    pad = _conv_padding(padding, rank)
    x = _input(rank, batched)
    w = _data(3, 2, *(3,) * rank, seed=1)
    fn = getattr(F, f"conv_transpose{rank}d")
    ref_fn = getattr(ref.nn.functional, f"conv_transpose{rank}d")
    got = fn(
        lucid.tensor(x, device=device), lucid.tensor(w, device=device), stride=step, padding=pad
    )
    want = ref_fn(ref.from_numpy(x), ref.from_numpy(w), stride=step, padding=pad)
    _close(got, want)


@pytest.mark.parametrize("rank", RANKS)
def test_lazy_conv_reads_the_channels_of_an_unbatched_input(device: str, rank: int) -> None:
    layer = getattr(nn, f"LazyConv{rank}d")(4, 3, device=device)
    out = layer(lucid.tensor(_input(rank, batched=False, channels=5), device=device))
    assert layer.in_channels == 5
    assert tuple(out.shape) == (4,) + (_SIZE - 2,) * rank


# ── pooling ───────────────────────────────────────────────────────────────────

_POOL_STRIDES = ("none", "empty", "1", "2", "per_axis")


def _pool_stride(spelling: str, rank: int) -> object:
    return {"none": None, "empty": (), "1": 1, "2": 2, "per_axis": _per_axis(1, rank, 1)}[spelling]


_POOLS: dict[str, Callable[[ModuleType, int], tuple[Callable, Callable]]] = {
    "max": lambda r, n: (getattr(F, f"max_pool{n}d"), getattr(r.nn.functional, f"max_pool{n}d")),
    "avg": lambda r, n: (getattr(F, f"avg_pool{n}d"), getattr(r.nn.functional, f"avg_pool{n}d")),
}


@pytest.mark.parametrize("kind", ["max", "avg"])
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("stride", _POOL_STRIDES)
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("ceil_mode", [False, True], ids=["floor", "ceil"])
def test_pool_matches_the_reference(
    ref: ModuleType,
    device: str,
    kind: str,
    rank: int,
    batched: bool,
    stride: str,
    padding: int,
    ceil_mode: bool,
) -> None:
    fn, ref_fn = _POOLS[kind](ref, rank)
    step = _pool_stride(stride, rank)
    x = _input(rank, batched)
    got = fn(lucid.tensor(x, device=device), 3, step, padding, ceil_mode=ceil_mode)
    want = ref_fn(ref.from_numpy(x), 3, step, padding, ceil_mode=ceil_mode)
    _close(got, want)


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
def test_max_pool_indices_match_the_reference(
    ref: ModuleType, device: str, rank: int, batched: bool
) -> None:
    x = _input(rank, batched)
    got, got_idx = getattr(F, f"max_pool{rank}d")(
        lucid.tensor(x, device=device), 3, 2, 1, return_indices=True
    )
    want, want_idx = getattr(ref.nn.functional, f"max_pool{rank}d")(
        ref.from_numpy(x), 3, 2, 1, return_indices=True
    )
    _close(got, want)
    np.testing.assert_array_equal(got_idx.numpy(), want_idx.numpy())


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("ceil_mode", [False, True], ids=["floor", "ceil"])
def test_lp_pool_matches_the_reference(
    ref: ModuleType, device: str, rank: int, batched: bool, ceil_mode: bool
) -> None:
    x = _input(rank, batched)
    got = getattr(F, f"lp_pool{rank}d")(lucid.tensor(x, device=device), 2.0, 2, ceil_mode=ceil_mode)
    want = getattr(ref.nn.functional, f"lp_pool{rank}d")(
        ref.from_numpy(x), 2.0, 2, ceil_mode=ceil_mode
    )
    _close(got, want)
    module = getattr(nn, f"LPPool{rank}d")(2.0, 2, ceil_mode=ceil_mode)
    _close(module(lucid.tensor(x, device=device)), want)


@pytest.mark.parametrize("kind", ["avg", "max"])
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("size", [1, 3, 7], ids=["global", "uneven", "identity"])
def test_adaptive_pool_matches_the_reference(
    ref: ModuleType, device: str, kind: str, rank: int, batched: bool, size: int
) -> None:
    x = _input(rank, batched)
    name = f"adaptive_{kind}_pool{rank}d"
    got = getattr(F, name)(lucid.tensor(x, device=device), size)
    want = getattr(ref.nn.functional, name)(ref.from_numpy(x), size)
    _close(got, want)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_instance_norm_module_takes_an_unbatched_input(
    ref: ModuleType, device: str, rank: int
) -> None:
    x = _input(rank, batched=False)
    layer = getattr(nn, f"InstanceNorm{rank}d")(3, affine=True, device=device)
    want = getattr(ref.nn, f"InstanceNorm{rank}d")(3, affine=True)(ref.from_numpy(x))
    _close(layer(lucid.tensor(x, device=device)), want)


# ── refusals ──────────────────────────────────────────────────────────────────

_X2 = _data(1, 3, 8, 8)
_W2 = _data(4, 3, 3, 3, seed=1)
_WT2 = _data(3, 4, 3, 3, seed=1)

_REFUSED: dict[str, Callable[[ModuleType, bool], object]] = {
    # The engine reads a pooling stride of 0 as "the kernel size".
    "max_pool_stride_0": lambda m, _: m.nn.functional.max_pool2d(_t(m, _X2), 2, 0),
    "max_pool_stride_negative": lambda m, _: m.nn.functional.max_pool2d(_t(m, _X2), 2, -1),
    "avg_pool_stride_0": lambda m, _: m.nn.functional.avg_pool2d(_t(m, _X2), 2, 0),
    "avg_pool1d_stride_0": lambda m, _: m.nn.functional.avg_pool1d(_t(m, _X2[0]), 2, 0),
    "max_pool_kernel_0": lambda m, _: m.nn.functional.max_pool2d(_t(m, _X2), 0),
    "max_pool_padding_past_half": lambda m, _: m.nn.functional.max_pool2d(_t(m, _X2), 2, padding=2),
    "avg_pool_padding_past_half": lambda m, _: m.nn.functional.avg_pool2d(_t(m, _X2), 3, padding=2),
    "max_pool_kernel_too_long": lambda m, _: m.nn.functional.max_pool2d(_t(m, _X2), (2, 2, 2)),
    "conv_stride_0": lambda m, _: m.nn.functional.conv2d(_t(m, _X2), _t(m, _W2), stride=0),
    "conv_padding_negative": lambda m, _: m.nn.functional.conv2d(
        _t(m, _X2), _t(m, _W2), padding=-1
    ),
    "conv_padding_too_long": lambda m, _: m.nn.functional.conv2d(
        _t(m, _X2), _t(m, _W2), padding=(1, 1, 1)
    ),
    "conv_padding_unknown_string": lambda m, _: m.nn.functional.conv2d(
        _t(m, _X2), _t(m, _W2), padding="full"
    ),
    "conv_transpose_stride_0": lambda m, _: m.nn.functional.conv_transpose2d(
        _t(m, _X2), _t(m, _WT2), stride=0
    ),
}


def _t(module: ModuleType, a: np.ndarray) -> object:
    return module.from_numpy(a) if module is not lucid else lucid.tensor(a)


@pytest.mark.parametrize("case", list(_REFUSED))
def test_a_refused_argument_is_refused_here_too(ref: ModuleType, case: str) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        _REFUSED[case](ref, True)
    with pytest.raises(ValueError):
        _REFUSED[case](lucid, True)


@pytest.mark.parametrize("rank", RANKS)
def test_same_padding_refuses_a_stride(ref: ModuleType, rank: int) -> None:
    x = _input(rank, batched=True)
    w = _data(4, 3, *(3,) * rank, seed=1)
    name = f"conv{rank}d"
    with pytest.raises(RuntimeError, match="strided"):
        getattr(ref.nn.functional, name)(
            ref.from_numpy(x), ref.from_numpy(w), stride=2, padding="same"
        )
    with pytest.raises(ValueError, match="strided"):
        getattr(F, name)(lucid.tensor(x), lucid.tensor(w), stride=2, padding="same")
    with pytest.raises(ValueError, match="strided"):
        getattr(nn, f"Conv{rank}d")(3, 4, 3, stride=2, padding="same")
