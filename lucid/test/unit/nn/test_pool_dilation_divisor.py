"""Dilated max pooling and ``divisor_override`` average pooling (LCD-57).

``max_pool{1,2,3}d(dilation=2)`` and ``nn.MaxPool2d(dilation=2)`` raised
``NotImplementedError``: the engine kernel pools undilated windows only.
``avg_pool{2,3}d(divisor_override=)`` raised too, and ``nn.AvgPool2d`` had no
such parameter.  A dilated window is now gathered and reduced in Python, and
``divisor_override`` rescales the padded average to the window's sum over the
divisor.  The cases hold both to the reference — values, indices and
gradients — on CPU and Metal, batched and unbatched.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

pytestmark = pytest.mark.parity


def _data(*shape: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _close(got: lucid.Tensor, want: object, atol: float = 1e-5) -> None:
    want_np = want.detach().numpy()
    assert tuple(got.shape) == tuple(want_np.shape)
    np.testing.assert_allclose(got.numpy(), want_np, rtol=1e-5, atol=atol)


def _input(rank: int, batched: bool, size: int = 9) -> np.ndarray:
    return _data(*((2, 3) if batched else (3,)), *(size,) * rank)


# (kernel, stride, padding, dilation); padding is at most kernel // 2.
_DILATED = [(2, 2, 0, 2), (3, 2, 1, 2), (3, 1, 1, 3), (2, 3, 1, 2)]


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("window", _DILATED, ids=lambda w: "k{}s{}p{}d{}".format(*w))
@pytest.mark.parametrize("ceil_mode", [False, True], ids=["floor", "ceil"])
def test_dilated_max_pool_matches_the_reference(
    ref: ModuleType,
    device: str,
    rank: int,
    batched: bool,
    window: tuple[int, int, int, int],
    ceil_mode: bool,
) -> None:
    k, s, p, d = window
    x = _input(rank, batched)
    xs = lucid.tensor(x, device=device, requires_grad=True)
    xr = ref.tensor(x, requires_grad=True)
    got, got_idx = getattr(F, f"max_pool{rank}d")(
        xs, k, s, p, d, ceil_mode=ceil_mode, return_indices=True
    )
    want, want_idx = getattr(ref.nn.functional, f"max_pool{rank}d")(
        xr, k, s, p, d, ceil_mode=ceil_mode, return_indices=True
    )
    _close(got, want)
    np.testing.assert_array_equal(got_idx.numpy(), want_idx.numpy())
    got.sum().backward()
    want.sum().backward()
    _close(xs.grad, xr.grad)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_max_pool_modules_take_dilation(ref: ModuleType, device: str, rank: int) -> None:
    x = _input(rank, batched=True)
    layer = getattr(nn, f"MaxPool{rank}d")(3, stride=2, padding=1, dilation=2)
    want = getattr(ref.nn, f"MaxPool{rank}d")(3, stride=2, padding=1, dilation=2)
    _close(layer(lucid.tensor(x, device=device)), want(ref.from_numpy(x)))


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("kernel,padding", [(3, 1), (3, 2), (4, 2), (4, 3), (2, 1)])
def test_padding_past_half_the_kernel_is_refused_at_any_dilation(
    ref: ModuleType, rank: int, dilation: int, kernel: int, padding: int
) -> None:
    # The limit is kernel_size // 2 whatever the dilation, though a dilated
    # window padded further would still reach the input.
    x = _input(rank, batched=True, size=12)
    name = f"max_pool{rank}d"
    args = (kernel, 1, padding, dilation)
    if padding <= kernel // 2:
        want = getattr(ref.nn.functional, name)(ref.from_numpy(x), *args)
        _close(getattr(F, name)(lucid.tensor(x), *args), want)
        return
    with pytest.raises(RuntimeError, match="half"):
        getattr(ref.nn.functional, name)(ref.from_numpy(x), *args)
    with pytest.raises(ValueError, match="half"):
        getattr(F, name)(lucid.tensor(x), *args)


def test_a_dilated_window_wider_than_the_input_is_refused(ref: ModuleType) -> None:
    x = _data(1, 1, 4, 4)
    with pytest.raises(RuntimeError):
        ref.nn.functional.max_pool2d(ref.from_numpy(x), 3, dilation=3)
    with pytest.raises(ValueError, match="too small"):
        F.max_pool2d(lucid.tensor(x), 3, dilation=3)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("divisor", [1, 3, -2])
@pytest.mark.parametrize("ceil_mode", [False, True], ids=["floor", "ceil"])
@pytest.mark.parametrize("count_include_pad", [True, False], ids=["cip", "nocip"])
@pytest.mark.parametrize("window", [(2, 2, 0), (3, 2, 1)], ids=["k2s2", "k3s2p1"])
def test_divisor_override_matches_the_reference(
    ref: ModuleType,
    device: str,
    rank: int,
    batched: bool,
    divisor: int,
    ceil_mode: bool,
    count_include_pad: bool,
    window: tuple[int, int, int],
) -> None:
    k, s, p = window
    x = _input(rank, batched, size=8)
    xs = lucid.tensor(x, device=device, requires_grad=True)
    xr = ref.tensor(x, requires_grad=True)
    args = (k, s, p, ceil_mode, count_include_pad, divisor)
    got = getattr(F, f"avg_pool{rank}d")(xs, *args)
    want = getattr(ref.nn.functional, f"avg_pool{rank}d")(xr, *args)
    _close(got, want)
    got.sum().backward()
    want.sum().backward()
    _close(xs.grad, xr.grad)
    module = getattr(nn, f"AvgPool{rank}d")(*args)
    _close(module(lucid.tensor(x, device=device)), want)


@pytest.mark.parametrize("rank", [2, 3])
def test_a_zero_divisor_is_refused(ref: ModuleType, rank: int) -> None:
    x = _input(rank, batched=True)
    with pytest.raises(RuntimeError, match="zero"):
        getattr(ref.nn.functional, f"avg_pool{rank}d")(ref.from_numpy(x), 2, divisor_override=0)
    with pytest.raises(ValueError, match="zero"):
        getattr(F, f"avg_pool{rank}d")(lucid.tensor(x), 2, divisor_override=0)
