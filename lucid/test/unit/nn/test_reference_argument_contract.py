"""Attention, transposed conv, adaptive and fractional pooling, and the
recurrent layers take the reference's arguments (LCD-135).

Each of these had an argument the reference accepts and Lucid did not:

- ``scaled_dot_product_attention`` had no ``enable_gqa`` and refused
  leading axes that broadcast (a key of batch 1 against a query of batch 2);
- ``ConvTranspose*d.forward`` had no ``output_size``;
- the adaptive pools read ``None`` in ``output_size`` ("keep this axis")
  with ``int()`` and raised;
- GRU and RNN named their parameters ``cell_l0.weight_ih`` where LSTM and
  the reference say ``weight_ih_l0``;
- ``fractional_max_pool2d`` read its fixed samples in (height, width)
  order where the reference reads (width, height).

The cases hold each one to the reference — values and gradients — on CPU
and Metal, and check that a GRU / RNN checkpoint saved under the old
parameter names still loads.
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


def _close(got: lucid.Tensor, want: object, atol: float = 1e-4) -> None:
    want_np = want.detach().numpy()
    assert tuple(got.shape) == tuple(want_np.shape)
    np.testing.assert_allclose(got.numpy(), want_np, rtol=1e-4, atol=atol)


# ── scaled_dot_product_attention ──────────────────────────────────────────────

_SDPA_SHAPES = {
    # name: (query, key / value, enable_gqa)
    "batch_broadcast": ((2, 4, 3, 8), (1, 4, 5, 8), False),
    "head_broadcast": ((2, 4, 3, 8), (2, 1, 5, 8), False),
    "rank3_key": ((2, 4, 3, 8), (4, 5, 8), False),
    "gqa_2_of_4": ((2, 4, 3, 8), (2, 2, 5, 8), True),
    "gqa_1_of_4": ((2, 4, 3, 8), (2, 1, 5, 8), True),
    "gqa_equal_heads": ((2, 4, 3, 8), (2, 4, 5, 8), True),
    "gqa_and_batch_broadcast": ((2, 6, 3, 8), (1, 2, 5, 8), True),
}


@pytest.mark.parametrize("case", list(_SDPA_SHAPES))
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
def test_sdpa_matches_the_reference(
    ref: ModuleType, device: str, case: str, causal: bool, masked: bool
) -> None:
    if causal and masked:
        pytest.skip("the reference refuses a mask together with is_causal")
    q_shape, kv_shape, gqa = _SDPA_SHAPES[case]
    q, k, v = _data(*q_shape), _data(*kv_shape, seed=1), _data(*kv_shape, seed=2)
    mask = _data(q_shape[-2], kv_shape[-2], seed=3) if masked else None
    ours = [lucid.tensor(a, device=device, requires_grad=True) for a in (q, k, v)]
    theirs = [ref.tensor(a, requires_grad=True) for a in (q, k, v)]
    got = F.scaled_dot_product_attention(
        *ours,
        attn_mask=None if mask is None else lucid.tensor(mask, device=device),
        is_causal=causal,
        enable_gqa=gqa,
    )
    want = ref.nn.functional.scaled_dot_product_attention(
        *theirs,
        attn_mask=None if mask is None else ref.from_numpy(mask),
        is_causal=causal,
        enable_gqa=gqa,
    )
    _close(got, want)
    got.sum().backward()
    want.sum().backward()
    for a, b in zip(ours, theirs, strict=True):
        _close(a.grad, b.grad, atol=1e-3)


@pytest.mark.parametrize(
    "q_shape,kv_shape,gqa",
    [
        ((1, 4, 3, 8), (1, 3, 5, 8), True),  # 3 heads do not divide 4
        ((1, 4, 3, 8), (1, 2, 5, 8), False),  # fewer heads without GQA
        ((3, 4, 3, 8), (2, 4, 5, 8), False),  # batches that do not broadcast
    ],
)
def test_sdpa_refuses_what_the_reference_refuses(
    ref: ModuleType, q_shape: tuple, kv_shape: tuple, gqa: bool
) -> None:
    q, kv = _data(*q_shape), _data(*kv_shape, seed=1)
    with pytest.raises(RuntimeError):
        ref.nn.functional.scaled_dot_product_attention(
            ref.from_numpy(q), ref.from_numpy(kv), ref.from_numpy(kv), enable_gqa=gqa
        )
    with pytest.raises(ValueError):
        F.scaled_dot_product_attention(
            lucid.tensor(q), lucid.tensor(kv), lucid.tensor(kv), enable_gqa=gqa
        )


# ── ConvTranspose*d.forward(output_size=) ────────────────────────────────────


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("extra", [0, 1, 2], ids=["smallest", "plus1", "largest"])
@pytest.mark.parametrize("spelling", ["spatial", "full"])
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_conv_transpose_output_size_matches_the_reference(
    ref: ModuleType, device: str, rank: int, batched: bool, extra: int, spelling: str, lazy: bool
) -> None:
    kwargs = dict(kernel_size=3, stride=3, padding=1, dilation=1)
    ref_layer = getattr(ref.nn, f"ConvTranspose{rank}d")(3, 2, **kwargs)
    if lazy:
        layer = getattr(nn, f"LazyConvTranspose{rank}d")(2, device=device, **kwargs)
    else:
        layer = getattr(nn, f"ConvTranspose{rank}d")(3, 2, device=device, **kwargs)
    x = _data(*((2, 3) if batched else (3,)), *(4,) * rank)
    layer(lucid.tensor(x, device=device))  # sizes the lazy one
    with ref.no_grad():
        ref_layer.weight.copy_(ref.from_numpy(layer.weight.numpy()))
        ref_layer.bias.copy_(ref.from_numpy(layer.bias.numpy()))
    # (4 - 1) * 3 - 2 + 3 = 10 is the smallest; a stride of 3 allows 10..12.
    spatial = tuple(10 + extra + (axis % 2 if extra < 2 else 0) for axis in range(rank))
    size = spatial if spelling == "spatial" else x.shape[: x.ndim - rank] + spatial
    got = layer(lucid.tensor(x, device=device), output_size=size)
    want = ref_layer(ref.from_numpy(x), output_size=size)
    _close(got, want)


@pytest.mark.parametrize("size", [(9, 10), (10, 13), (10,), (1, 1, 1, 10, 10)])
def test_conv_transpose_output_size_out_of_range_is_refused(ref: ModuleType, size: tuple) -> None:
    x = _data(1, 3, 4, 4)
    with pytest.raises(ValueError):
        ref.nn.ConvTranspose2d(3, 2, 3, stride=3, padding=1)(ref.from_numpy(x), output_size=size)
    with pytest.raises(ValueError):
        nn.ConvTranspose2d(3, 2, 3, stride=3, padding=1)(lucid.tensor(x), output_size=size)


# ── adaptive pools: None keeps the axis ───────────────────────────────────────


def _none_sizes(rank: int) -> list[tuple[int | None, ...]]:
    sizes: list[tuple[int | None, ...]] = []
    for keep in range(rank):
        sizes.append(tuple(None if axis == keep else 2 for axis in range(rank)))
    sizes.append((None,) * rank)
    return sizes


@pytest.mark.parametrize("kind", ["avg", "max"])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("which", [0, 1, 2, 3])
def test_adaptive_pool_keeps_an_axis_given_as_none(
    ref: ModuleType, device: str, kind: str, rank: int, batched: bool, which: int
) -> None:
    sizes = _none_sizes(rank)
    if which >= len(sizes):
        pytest.skip("one None per axis, plus all of them")
    size = sizes[which]
    x = _data(*((2, 3) if batched else (3,)), *(5 + axis for axis in range(rank)))
    name = f"adaptive_{kind}_pool{rank}d"
    want = getattr(ref.nn.functional, name)(ref.from_numpy(x), size)
    _close(getattr(F, name)(lucid.tensor(x, device=device), size), want)
    cls = f"Adaptive{kind.capitalize()}Pool{rank}d"
    _close(getattr(nn, cls)(size)(lucid.tensor(x, device=device)), want)


@pytest.mark.parametrize("kind", ["avg", "max"])
@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("spelling", ["int", "one_axis", "with_none"])
def test_adaptive_pool_to_zero_is_empty(
    ref: ModuleType, device: str, kind: str, rank: int, batched: bool, spelling: str
) -> None:
    # A size of 0 used to divide by zero in the window arithmetic.
    size: object = {
        "int": 0,
        "one_axis": (0,) + (2,) * (rank - 1),
        "with_none": (None,) * (rank - 1) + (0,),
    }[spelling]
    x = _data(*((2, 3) if batched else (3,)), *(5 + axis for axis in range(rank)))
    name = f"adaptive_{kind}_pool{rank}d"
    want = getattr(ref.nn.functional, name)(ref.from_numpy(x), size)
    xs = lucid.tensor(x, device=device, requires_grad=True)
    got = getattr(F, name)(xs, size)
    assert tuple(got.shape) == tuple(want.shape)
    got.sum().backward()
    assert xs.grad is None or float(xs.grad.abs().sum().item()) == 0.0
    if kind == "max":
        _, idx = getattr(F, name)(xs, size, return_indices=True)
        assert tuple(idx.shape) == tuple(want.shape)


@pytest.mark.parametrize("kind", ["avg", "max"])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_adaptive_pool_refuses_a_negative_size(ref: ModuleType, kind: str, rank: int) -> None:
    x = _data(2, 3, *(5,) * rank)
    name = f"adaptive_{kind}_pool{rank}d"
    with pytest.raises(RuntimeError):
        getattr(ref.nn.functional, name)(ref.from_numpy(x), -1)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        getattr(F, name)(lucid.tensor(x), -1)


def test_fractional_samples_that_are_not_real_are_refused() -> None:
    samples = lucid.tensor([[[0.5 + 0j, 0.25 + 0j]]])
    with pytest.raises(TypeError, match="_random_samples"):
        F.fractional_max_pool2d(lucid.randn(1, 1, 8, 8), 2, output_size=4, _random_samples=samples)


# ── GRU / RNN parameter names ─────────────────────────────────────────────────

_RECURRENT = {
    "GRU": dict(num_layers=2, bidirectional=True),
    "RNN": dict(num_layers=2, bidirectional=True, nonlinearity="relu"),
    "RNN-nobias": dict(num_layers=1, bias=False),
}


def _recurrent(module: ModuleType, case: str, device: str | None = None) -> object:
    kind = case.split("-")[0]
    kwargs = dict(_RECURRENT[case])
    if device is not None:
        kwargs["device"] = device
    return getattr(module.nn, kind)(4, 5, **kwargs)


@pytest.mark.parametrize("case", list(_RECURRENT))
def test_recurrent_parameter_names_are_the_references(ref: ModuleType, case: str) -> None:
    ours, theirs = _recurrent(lucid, case), _recurrent(ref, case)
    assert [n for n, _ in ours.named_parameters()] == [n for n, _ in theirs.named_parameters()]
    assert list(ours.state_dict()) == list(theirs.state_dict())
    for name, p in theirs.named_parameters():
        assert tuple(getattr(ours, name).shape) == tuple(p.shape)


@pytest.mark.parametrize("case", list(_RECURRENT))
@pytest.mark.parametrize("legacy", [False, True], ids=["flat_keys", "cell_keys"])
@pytest.mark.parametrize("assign", [False, True], ids=["copy", "assign"])
def test_recurrent_checkpoint_round_trips(
    ref: ModuleType, device: str, case: str, legacy: bool, assign: bool
) -> None:
    theirs = _recurrent(ref, case)
    checkpoint: dict[str, lucid.Tensor] = {}
    for name, value in theirs.state_dict().items():
        if legacy:
            # How GRU / RNN checkpoints were keyed before the names were flat.
            param, _, tag = name.partition("_l")
            name = f"cell_l{tag}.{param}"
        checkpoint[name] = lucid.tensor(value.numpy(), device=device)
    ours = _recurrent(lucid, case, device)
    result = ours.load_state_dict(checkpoint, assign=assign)
    assert not result.missing_keys and not result.unexpected_keys
    x = _data(6, 2, 4)
    got, h = ours(lucid.tensor(x, device=device))
    want, want_h = theirs(ref.from_numpy(x))
    _close(got, want)
    _close(h, want_h)
    again = _recurrent(lucid, case, device)
    again.load_state_dict(ours.state_dict())
    _close(again(lucid.tensor(x, device=device))[0], want)


def test_rnn_refuses_an_unknown_nonlinearity(ref: ModuleType) -> None:
    with pytest.raises(ValueError, match="nonlinearity"):
        ref.nn.RNN(4, 5, nonlinearity="gelu")
    with pytest.raises(ValueError, match="nonlinearity"):
        nn.RNN(4, 5, nonlinearity="gelu")
    with pytest.raises(ValueError, match="nonlinearity"):
        nn.RNNCell(4, 5, nonlinearity="gelu")


# ── fractional max pooling with fixed samples ────────────────────────────────


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("batched", [True, False], ids=["batched", "unbatched"])
@pytest.mark.parametrize("by", ["size", "ratio"])
@pytest.mark.parametrize("kernel", [2, 3])
def test_fractional_max_pool_matches_the_reference(
    ref: ModuleType, device: str, rank: int, batched: bool, by: str, kernel: int
) -> None:
    spatial = (11, 13, 9)[:rank]
    x = _data(*((2, 3) if batched else (3,)), *spatial)
    samples = np.random.default_rng(4).random((2 if batched else 1, 3, rank))
    samples = samples.astype(np.float32)
    kwargs: dict[str, object] = (
        {"output_size": (5, 6, 4)[:rank]}
        if by == "size"
        else {"output_ratio": (0.5, 0.6, 0.7)[:rank]}
    )
    name = f"fractional_max_pool{rank}d"
    xs = lucid.tensor(x, device=device, requires_grad=True)
    xr = ref.tensor(x, requires_grad=True)
    got, got_idx = getattr(F, name)(
        xs,
        kernel,
        return_indices=True,
        _random_samples=lucid.tensor(samples, device=device),
        **kwargs,
    )
    want, want_idx = getattr(ref.nn.functional, name)(
        xr, kernel, return_indices=True, _random_samples=ref.from_numpy(samples), **kwargs
    )
    _close(got, want)
    np.testing.assert_array_equal(got_idx.numpy(), want_idx.numpy())
    got.sum().backward()
    want.sum().backward()
    _close(xs.grad, xr.grad)
