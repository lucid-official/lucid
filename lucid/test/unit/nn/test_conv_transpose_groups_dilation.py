"""Grouped and dilated transposed convolution, checked against definitions.

Both options were accepted, documented, and then refused: the functional
layer raised ``Only groups=1 is supported`` because the engine kernel had
no parameter to carry them.  That left depthwise transposed convolution —
the standard building block of an efficient segmentation decoder —
unreachable on every backend at once.

The references here are the definitions rather than recorded numbers.  A
grouped transposed convolution *is* ``groups`` independent ungrouped ones
over channel slices, and a dilated one *is* the undilated convolution of
a kernel with ``dilation - 1`` zeros between its taps.  Building both out
of the op's own ``groups=1`` / ``dilation=1`` path means a wrong grouping
convention cannot hide behind a self-consistent implementation.
"""

import itertools

import pytest

import lucid
import lucid.nn.functional as F

TRANSPOSED = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


def _rel(a: lucid.Tensor, b: lucid.Tensor) -> float:
    """Relative max error, so a large bias sum is not judged in absolutes."""
    scale = max(float(a.abs().max().item()), 1e-30)
    return float((a - b).abs().max().item()) / scale


def _stuff_zeros(w: lucid.Tensor, dilation: tuple[int, ...]) -> lucid.Tensor:
    """The undilated kernel that computes the same dilated convolution."""
    rank = len(dilation)
    taps = [int(w.shape[2 + i]) for i in range(rank)]
    spread = [dilation[i] * (taps[i] - 1) + 1 for i in range(rank)]
    out = lucid.zeros(int(w.shape[0]), int(w.shape[1]), *spread, dtype=w.dtype, device=w.device)
    for idx in itertools.product(*[range(t) for t in taps]):
        at = tuple(idx[i] * dilation[i] for i in range(rank))
        out[(slice(None), slice(None)) + at] = w[(slice(None), slice(None)) + idx]
    return out


def _by_group(fn, x, w, b, groups: int, **kwargs) -> lucid.Tensor:
    """``groups`` independent ungrouped convolutions, joined on the channels."""
    per_in = int(x.shape[1]) // groups
    per_out = int(w.shape[1])
    parts = [
        fn(
            x[:, g * per_in : (g + 1) * per_in],
            w[g * per_in : (g + 1) * per_in],
            b[g * per_out : (g + 1) * per_out],
            **kwargs,
        )
        for g in range(groups)
    ]
    return lucid.concat(parts, dim=1)


def _sample(rank: int, c_in: int, c_out_per_group: int, groups: int, size: int = 5):
    lucid.manual_seed(rank * 10 + groups)
    x = lucid.randn(2, c_in, *([size] * rank))
    w = lucid.randn(c_in, c_out_per_group, *([3] * rank))
    b = lucid.randn(c_out_per_group * groups)
    return x, w, b


# ── grouping ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize(
    ("groups", "per_group"),
    [(2, 3), (3, 1), (6, 1)],
    ids=["two-groups", "three-groups", "depthwise"],
)
def test_grouping_equals_independent_convolutions(rank, groups, per_group):
    x, w, b = _sample(rank, 6, per_group, groups)
    fn = TRANSPOSED[rank]
    got = fn(x, w, b, stride=2, padding=1, groups=groups)
    want = _by_group(fn, x, w, b, groups, stride=2, padding=1)

    assert tuple(got.shape) == tuple(want.shape)
    assert int(got.shape[1]) == per_group * groups
    assert _rel(want, got) < 1e-6


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_grouped_gradients_match_the_decomposition(rank):
    groups, per_group = 3, 2
    x, w, b = _sample(rank, 6, per_group, groups)
    fn = TRANSPOSED[rank]

    grouped = [t.detach() for t in (x, w, b)]
    for t in grouped:
        t.requires_grad = True
    y = fn(*grouped, stride=2, padding=1, groups=groups)
    (y * y).sum().backward()

    per_in = 6 // groups
    slices, outs = [], []
    for g in range(groups):
        parts = [
            x[:, g * per_in : (g + 1) * per_in].detach(),
            w[g * per_in : (g + 1) * per_in].detach(),
            b[g * per_group : (g + 1) * per_group].detach(),
        ]
        for t in parts:
            t.requires_grad = True
        slices.append(parts)
        outs.append(fn(*parts, stride=2, padding=1))
    reference = lucid.concat(outs, dim=1)
    (reference * reference).sum().backward()

    for axis, slot in ((1, 0), (0, 1), (0, 2)):
        want = lucid.concat([s[slot].grad for s in slices], dim=axis)
        assert _rel(want, grouped[slot].grad) < 1e-5


def test_bias_length_follows_c_out_and_not_the_weight_axis():
    """The transposed weight's second axis is ``C_out // groups``.

    Sizing an implicit bias from that axis alone leaves it short by
    exactly ``groups`` — the shape is wrong rather than the values, so it
    surfaces as a broadcast far from the cause.
    """
    x = lucid.randn(1, 8, 5)
    w = lucid.randn(8, 2, 3)  # groups=4 -> C_out = 8
    y = F.conv_transpose1d(x, w, None, groups=4)
    assert int(y.shape[1]) == 8


def test_channels_must_divide_into_groups():
    x = lucid.randn(1, 5, 6)
    w = lucid.randn(5, 2, 3)
    with pytest.raises(Exception, match="divisible by groups"):
        F.conv_transpose1d(x, w, None, groups=2)


# ── dilation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("dilation", [2, 3], ids=["dilation-2", "dilation-3"])
def test_dilation_equals_a_zero_stuffed_kernel(rank, dilation):
    x, w, b = _sample(rank, 3, 4, 1)
    fn = TRANSPOSED[rank]
    got = fn(x, w, b, stride=2, padding=1, dilation=dilation)
    want = fn(x, _stuff_zeros(w, (dilation,) * rank), b, stride=2, padding=1)

    assert tuple(got.shape) == tuple(want.shape)
    # The extent grows by dilation * (k - 1) rather than k - 1.
    expected = (int(x.shape[2]) - 1) * 2 - 2 + dilation * (3 - 1) + 1
    assert int(got.shape[2]) == expected
    assert _rel(want, got) < 1e-6


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_dilated_gradients_match_the_zero_stuffed_kernel(rank):
    dilation = 2
    x, w, b = _sample(rank, 3, 4, 1)
    fn = TRANSPOSED[rank]

    direct = [t.detach() for t in (x, w, b)]
    for t in direct:
        t.requires_grad = True
    y = fn(*direct, stride=2, padding=1, dilation=dilation)
    (y * y).sum().backward()

    stuffed = [x.detach(), _stuff_zeros(w, (dilation,) * rank).detach(), b.detach()]
    for t in stuffed:
        t.requires_grad = True
    reference = fn(*stuffed, stride=2, padding=1)
    (reference * reference).sum().backward()

    # Only the stuffed positions carry a real tap; the interleaved zeros
    # accumulate gradient that the dilated kernel has no slot for.
    taps = (slice(None), slice(None)) + (slice(None, None, dilation),) * rank
    assert _rel(stuffed[0].grad, direct[0].grad) < 1e-5
    assert _rel(stuffed[1].grad[taps], direct[1].grad) < 1e-5
    assert _rel(stuffed[2].grad, direct[2].grad) < 1e-5


# ── the two together, and the other stream ───────────────────────────


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
@pytest.mark.parametrize(
    ("rank", "kwargs"),
    [
        (1, {"stride": 2, "padding": 1, "groups": 3}),
        (1, {"stride": 1, "dilation": 2}),
        (2, {"stride": 2, "padding": 1, "groups": 6}),
        (2, {"stride": 2, "padding": 2, "groups": 2, "dilation": 2}),
        (3, {"stride": 2, "padding": 1, "groups": 3}),
        (3, {"stride": 1, "dilation": 2}),
    ],
    ids=[
        "1d-grouped",
        "1d-dilated",
        "2d-depthwise",
        "2d-grouped-dilated",
        "3d-grouped",
        "3d-dilated",
    ],
)
def test_the_two_streams_agree(rank, kwargs):
    """MLX declines grouping above two spatial dimensions.

    The rank-5 grouped case therefore runs a per-group loop rather than
    one kernel; this is what catches that loop pairing the wrong channel
    blocks, which no CPU-only test would see.
    """
    groups = kwargs.get("groups", 1)
    x, w, b = _sample(rank, 6, 6 // groups if groups > 1 else 4, groups, size=4)
    fn = TRANSPOSED[rank]

    def run(device):
        args = [t.to(device).detach() for t in (x, w, b)]
        for t in args:
            t.requires_grad = True
        out = fn(*args, **kwargs)
        (out * out).sum().backward()
        return [out] + [t.grad for t in args]

    for cpu, metal in zip(run("cpu"), run("metal")):
        assert _rel(cpu, metal.to("cpu")) < 1e-5
