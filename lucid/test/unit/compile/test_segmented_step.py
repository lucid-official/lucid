"""Segmented compiled training step (``make_step(model, loss, segments=K)``).

Executable splitting compiles each contiguous group of an
:class:`nn.Sequential`'s children into its own forward + backward executable and
stitches them with eager autograd (gradient-checkpointing-style: each segment's
backward recomputes only its own forward, so no single executable holds the full
activation stack).  These tests assert the property that matters — the split
path is **token-identical** to both the eager path and the monolithic
(``segments=1``) compiled path — plus the API guards (Sequential-only,
segment-count clamping, compiled-not-fallback).
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import make_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _ce(y: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return F.cross_entropy(y, target)


def _grads(step_callable, model, x, t, *, eager=False):
    """Run one step (eager or compiled), return (loss, {param_index: grad})."""
    for p in model.parameters():
        p.grad = None
    if eager:
        loss = F.cross_entropy(model(x), t)
    else:
        loss = step_callable(x, t)
    loss.backward()
    return float(loss.item()), {
        i: (p.grad.clone() if p.grad is not None else None)
        for i, p in enumerate(model.parameters())
    }


def _maxdiff(ga, gb):
    m = 0.0
    for k in ga:
        if ga[k] is None or gb[k] is None:
            continue
        m = max(m, float((ga[k] - gb[k]).abs().max().item()))
    return m


def _mlp(d: int, c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d, d),
        nn.ReLU(),
        nn.Linear(d, d),
        nn.ReLU(),
        nn.Linear(d, d),
        nn.ReLU(),
        nn.Linear(d, c),
    ).to(COMPILE_DEVICE)


class _ConvBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.relu = nn.ReLU()

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _Head(nn.Module):
    def __init__(self, c: int, n: int) -> None:
        super().__init__()
        self.fc = nn.Linear(c, n)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(x.mean(dim=(2, 3)))


def test_segmented_grads_match_eager_and_mono_mlp() -> None:
    lucid.manual_seed(0)
    b, d, c = 16, 64, 10
    model = _mlp(d, c)
    x = lucid.randn(b, d).to(COMPILE_DEVICE)
    t = lucid.randint(0, c, (b,)).to(COMPILE_DEVICE)

    _, g_eager = _grads(None, model, x, t, eager=True)
    _, g_mono = _grads(make_step(model, _ce), model, x, t)
    seg = make_step(model, _ce, segments=3)
    _, g_seg = _grads(seg, model, x, t)

    assert not seg.eager_only["flag"], "segmented step fell back to eager"
    assert seg.segments == 3
    assert _maxdiff(g_seg, g_eager) < 1e-4
    assert _maxdiff(g_seg, g_mono) < 1e-4


def test_segmented_grads_match_eager_conv_bn() -> None:
    lucid.manual_seed(0)
    b, c, n, depth = 8, 16, 10, 6
    model = nn.Sequential(*[_ConvBlock(c) for _ in range(depth)], _Head(c, n)).to(COMPILE_DEVICE)
    x = lucid.randn(b, c, 14, 14).to(COMPILE_DEVICE)
    t = lucid.randint(0, n, (b,)).to(COMPILE_DEVICE)

    _, g_eager = _grads(None, model, x, t, eager=True)
    seg = make_step(model, _ce, segments=3)
    _, g_seg = _grads(seg, model, x, t)

    assert not seg.eager_only["flag"]
    assert _maxdiff(g_seg, g_eager) < 1e-4


def test_segmented_cache_hit_stays_correct() -> None:
    lucid.manual_seed(0)
    b, d, c = 8, 32, 5
    model = _mlp(d, c)
    x = lucid.randn(b, d).to(COMPILE_DEVICE)
    t = lucid.randint(0, c, (b,)).to(COMPILE_DEVICE)

    _, g_eager = _grads(None, model, x, t, eager=True)
    seg = make_step(model, _ce, segments=2)
    _grads(seg, model, x, t)  # first call compiles
    _, g_seg2 = _grads(seg, model, x, t)  # second call is a cache hit
    assert _maxdiff(g_seg2, g_eager) < 1e-4


def test_segments_clamped_to_child_count() -> None:
    lucid.manual_seed(0)
    model = _mlp(16, 4)
    x = lucid.randn(8, 16).to(COMPILE_DEVICE)
    t = lucid.randint(0, 4, (8,)).to(COMPILE_DEVICE)
    # More segments requested than children -> clamped, still compiles + runs.
    seg = make_step(model, _ce, segments=999)
    loss = seg(x, t)
    loss.backward()
    assert seg.segments == len(list(model))


def test_segments_requires_sequential() -> None:
    model = nn.Linear(8, 4).to(COMPILE_DEVICE)
    import pytest

    with pytest.raises(TypeError, match="nn.Sequential"):
        make_step(model, _ce, segments=2)


def test_segments_auto_selects_and_stays_correct() -> None:
    lucid.manual_seed(0)
    b, d, c = 16, 64, 10
    model = _mlp(d, c)
    x = lucid.randn(b, d).to(COMPILE_DEVICE)
    t = lucid.randint(0, c, (b,)).to(COMPILE_DEVICE)

    _, g_eager = _grads(None, model, x, t, eager=True)
    auto = make_step(model, _ce, segments="auto")
    _, g_auto = _grads(auto, model, x, t)  # first call probes + selects

    chosen = auto.chosen_segments()
    n_children = len(list(model))
    assert chosen is not None and 2 <= chosen <= n_children
    assert auto.probe_results()  # populated
    assert _maxdiff(g_auto, g_eager) < 1e-4
    # second call delegates to the chosen step, still correct
    _, g_auto2 = _grads(auto, model, x, t)
    assert _maxdiff(g_auto2, g_eager) < 1e-4


def test_segments_auto_rejects_non_sequential() -> None:
    model = nn.Linear(8, 4).to(COMPILE_DEVICE)
    import pytest

    with pytest.raises(TypeError, match="nn.Sequential"):
        make_step(model, _ce, segments="auto")
