"""A normalisation's accuracy must not depend on how much it reduces.

``group_norm`` and ``batch_norm`` sum a whole group — channels times
spatial, or batch times spatial — to get a mean and a variance.  Done
serially in one ``float``, that sum loses accuracy in proportion to how
many terms it has taken, so the *same layer on the same distribution*
gets worse as the feature map grows: measured 7.8e-07 over four thousand
elements and 2.1e-04 over a million.

That is the shape of the bug, and the shape is what these tests check.
An absolute threshold would not have caught it — 2.1e-04 looks like
float32 accumulation over a deep network, and in the case that found it
(a converted Stable Diffusion autoencoder) it was recorded as exactly
that and left alone for a release.  What gave it away was that the
number *moved with the input's resolution*, so the property pinned here
is flatness rather than any particular value.

Float64 is the reference, not another framework: the question is whether
the f32 path agrees with the same arithmetic carried out precisely, and
Lucid can answer that itself.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn

# Roughly a thousand-fold range in what a single group reduces.  The
# defect grew linearly across it; anything that still does will show up
# as a ratio far above `_FLATNESS`.
_SMALL = (1, 64, 8, 8)
_LARGE = (1, 64, 256, 256)
_GROUPS = 32

# The f32 path is allowed to be a few times worse on the larger
# reduction — some growth is real.  Linear growth is not: across this
# range the defect gave about 300x.
_FLATNESS = 20.0
_CEILING = 5e-06


def _drift(layer_factory, shape, groups=None):
    """Relative disagreement between the f32 and f64 paths of one layer.

    Both get the same weights and the same input; only the precision of
    the reduction differs, so what comes back is the f32 path's error.
    """
    rng = np.random.default_rng(0)
    channels = shape[1]
    x = rng.standard_normal(shape).astype(np.float64)
    weight = rng.standard_normal(channels).astype(np.float64)
    bias = rng.standard_normal(channels).astype(np.float64)

    outputs = {}
    for dtype, cast in ((lucid.float32, np.float32), (lucid.float64, np.float64)):
        layer = layer_factory(channels, groups)
        state = {
            "weight": lucid.tensor(weight.astype(cast)),
            "bias": lucid.tensor(bias.astype(cast)),
        }
        if isinstance(layer, nn.BatchNorm2d):
            state |= {
                "running_mean": lucid.zeros((channels,), dtype=dtype),
                "running_var": lucid.ones((channels,), dtype=dtype),
                "num_batches_tracked": lucid.tensor(0),
            }
            layer.train()
        layer.load_state_dict(state)
        with lucid.no_grad():
            outputs[cast] = layer(lucid.tensor(x.astype(cast))).numpy().astype(np.float64)

    exact = outputs[np.float64]
    return float(np.abs(outputs[np.float32] - exact).max() / np.abs(exact).max())


def _group_norm(channels, groups):
    return nn.GroupNorm(groups, channels, eps=1e-6)


def _batch_norm(channels, _groups):
    return nn.BatchNorm2d(channels, eps=1e-5)


class TestGroupNorm:
    def test_a_small_group_is_accurate(self) -> None:
        """The baseline the large case is compared against."""
        assert _drift(_group_norm, _SMALL, _GROUPS) < _CEILING

    def test_a_million_element_group_is_just_as_accurate(self) -> None:
        assert _drift(_group_norm, _LARGE, _GROUPS) < _CEILING

    def test_the_error_does_not_grow_with_the_group(self) -> None:
        """The one that would have caught it.

        Both cases can pass a fixed threshold while the second is
        hundreds of times worse than the first, which is what a network
        deep enough to compound it turns into a wrong answer.
        """
        small = _drift(_group_norm, _SMALL, _GROUPS)
        large = _drift(_group_norm, _LARGE, _GROUPS)
        assert large < max(small, 1e-08) * _FLATNESS, (
            f"group-norm error grew from {small:.2e} to {large:.2e} as the "
            f"group went from {_SMALL[1] // _GROUPS * 64:,} to "
            f"{_LARGE[1] // _GROUPS * 256 * 256:,} elements"
        )


class TestTheBackwardReducesToo:
    """The gradient sums the same group, four times over.

    ``dx`` needs two group-wide sums; ``dweight`` and ``dbias`` each
    accumulate over batch times spatial per channel.  Widening only the
    forward would leave a layer whose activations are accurate and whose
    gradients are not, which is the harder half to notice — a training
    run does not print its gradients.

    ``dweight`` is the sharpest of the three because it sums a quantity
    with mean zero, so the total is far smaller than the terms and the
    cancellation exposes whatever precision the accumulator has.
    """

    @staticmethod
    def _grads(shape, dtype, cast):
        rng = np.random.default_rng(1)
        channels = shape[1]
        x = rng.standard_normal(shape)
        upstream = rng.standard_normal(shape)
        weight = rng.standard_normal(channels)
        bias = rng.standard_normal(channels)

        layer = nn.GroupNorm(_GROUPS, channels, eps=1e-6)
        layer.load_state_dict(
            {"weight": lucid.tensor(weight.astype(cast)), "bias": lucid.tensor(bias.astype(cast))}
        )
        inp = lucid.tensor(x.astype(cast), requires_grad=True)
        layer(inp).backward(lucid.tensor(upstream.astype(cast)))
        return (
            inp.grad.numpy().astype(np.float64),
            layer.weight.grad.numpy().astype(np.float64),
            layer.bias.grad.numpy().astype(np.float64),
        )

    def _drifts(self, shape):
        got = self._grads(shape, lucid.float32, np.float32)
        exact = self._grads(shape, lucid.float64, np.float64)
        return [float(np.abs(g - e).max() / np.abs(e).max()) for g, e in zip(got, exact)]

    @pytest.mark.parametrize("index,name", [(0, "dx"), (1, "dweight"), (2, "dbias")])
    def test_a_million_element_reduction_stays_accurate(self, index, name) -> None:
        assert self._drifts(_LARGE)[index] < _CEILING, name

    def test_no_gradient_degrades_with_the_group(self) -> None:
        small = self._drifts(_SMALL)
        large = self._drifts(_LARGE)
        for name, s, l in zip(("dx", "dweight", "dbias"), small, large):
            assert l < max(s, 1e-08) * _FLATNESS, f"{name} error grew from {s:.2e} to {l:.2e}"


class TestBatchNorm:
    """Same reduction, over batch and space rather than channels."""

    @pytest.mark.parametrize("shape", [_SMALL, _LARGE])
    def test_it_is_accurate_at_either_size(self, shape) -> None:
        assert _drift(_batch_norm, shape) < _CEILING

    def test_the_error_does_not_grow_with_the_reduction(self) -> None:
        small = _drift(_batch_norm, _SMALL)
        large = _drift(_batch_norm, _LARGE)
        assert large < max(small, 1e-08) * _FLATNESS


# ── where the data sits ───────────────────────────────────────────────────
#
# A variance taken as E[x²] − mean² subtracts two numbers that grow with
# the mean while their difference does not, so once the mean outgrows the
# spread the result is mostly rounding.  The CPU float32 batch norm did
# exactly that, and scaled its output as x·scale + (β − mean·scale), which
# cancels the same way: at a mean a hundred times the spread its input
# gradient was off by 7e-03 against 1e-05 on Metal.  The inputs above are
# zero-mean, so the flatness checks could never see it.  What the ReLUs
# and residual sums of a real network hand a norm is not zero-mean.

_OFFSETS = [(0.0, 1.0), (100.0, 1.0), (10.0, 0.1)]
_OFFSET_CEILING = 1e-04

_NORMS = {
    "batch_norm_2d": (lambda: nn.BatchNorm2d(64), (2, 64, 2, 2)),
    "batch_norm_1d": (lambda: nn.BatchNorm1d(64), (8, 64)),
    "layer_norm": (lambda: nn.LayerNorm(64), (4, 8, 64)),
    "group_norm": (lambda: nn.GroupNorm(8, 64), (4, 64, 4, 4)),
    "instance_norm": (lambda: nn.InstanceNorm2d(64, affine=True), (4, 64, 4, 4)),
    "rms_norm": (lambda: nn.RMSNorm(64), (4, 8, 64)),
}


def _devices() -> list[str]:
    out = ["cpu"]
    try:
        lucid.zeros(1).to("metal")
        out.append("metal")
    except Exception:  # noqa: BLE001 — no Metal on this machine
        pass
    return out


def _output_and_grad(name, device, dtype, offset, spread):
    factory, shape = _NORMS[name]
    lucid.manual_seed(0)
    layer = factory().to(dtype).to(device).train()
    rng = np.random.default_rng(3)
    x = lucid.tensor(rng.standard_normal(shape) * spread + offset).to(dtype).to(device)
    x.requires_grad = True
    y = layer(x)
    probe = lucid.tensor(rng.standard_normal(shape)).to(dtype).to(device)
    (y * probe).sum().backward()
    return y.numpy().astype(np.float64), x.grad.numpy().astype(np.float64)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("offset,spread", _OFFSETS)
@pytest.mark.parametrize("name", sorted(_NORMS))
def test_accuracy_does_not_depend_on_the_mean(name, offset, spread, device) -> None:
    got_y, got_dx = _output_and_grad(name, device, lucid.float32, offset, spread)
    want_y, want_dx = _output_and_grad(name, "cpu", lucid.float64, offset, spread)
    for label, g, w in (("output", got_y, want_y), ("input grad", got_dx, want_dx)):
        err = float(np.abs(g - w).max() / np.abs(w).max())
        assert err < _OFFSET_CEILING, f"{label}: {err:.2e}"
