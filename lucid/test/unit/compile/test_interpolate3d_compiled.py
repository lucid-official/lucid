"""Three-dimensional resampling compiles.

MPSGraph's resamplers are two-dimensional, and these two were stubs.
Resampling is separable, so each axis is its own one-dimensional resample:
a gather for nearest, a gather of two taps and a blend for linear, at the
engine's shared sample positions (``OpEmitters/_Resample.h``; the exact
agreement with eager on Metal is held in
``test_resize_sample_positions_compiled.py``).  Here the reference is the
CPU stream.

A gather lets the same path serve a fractional ratio and a downsample.
``align_corners`` changes which source plane each output plane reads, so
both settings are covered: agreeing on the default would say nothing about
the flag.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


class _Resample(nn.Module):
    def __init__(self, size: tuple[int, ...], mode: str, align: bool) -> None:
        super().__init__()
        self.size = size
        self.mode = mode
        self.align = align

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        if self.mode == "nearest":
            return F.interpolate(x, size=self.size, mode=self.mode)
        return F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align)


def _run(
    shape: tuple[int, ...], size: tuple[int, ...], mode: str, align: bool = False
) -> tuple[bool, float]:
    lucid.manual_seed(len(size))
    model = _Resample(size, mode, align).eval()
    x = lucid.randn(*shape)
    reference = model(x)

    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got = compiled(x.to("metal")).to("cpu")

    assert tuple(got.shape) == tuple(reference.shape)
    scale = max(float(reference.abs().max().item()), 1e-30)
    return (
        _C_engine.compile.session_cache_size() > 0,
        float((got - reference).abs().max().item()) / scale,
    )


class TestNearest3dCompiles:
    """Depth is a gather, so no ratio is a special case."""

    @pytest.mark.parametrize(
        ("shape", "size"),
        [
            ((1, 2, 3, 4, 4), (6, 8, 8)),
            ((1, 2, 3, 4, 4), (5, 7, 6)),
            ((1, 2, 4, 8, 8), (2, 4, 4)),
            ((2, 3, 2, 3, 5), (2, 6, 10)),
        ],
        ids=["doubled", "fractional", "downsampled", "depth-unchanged"],
    )
    def test_it_compiles_and_matches(self, shape, size):
        compiled, error = _run(shape, size, "nearest")
        assert compiled, "fell back to eager — the emitter declined this node"
        # Nearest is a pure selection: the compiled graph reads the same
        # elements, so anything but exact agreement means a wrong index.
        assert error == 0.0


class TestTrilinearCompiles:
    @pytest.mark.parametrize("align", [False, True], ids=["centres", "corners"])
    @pytest.mark.parametrize(
        ("shape", "size"),
        [
            ((1, 2, 3, 4, 4), (6, 8, 8)),
            ((1, 2, 3, 4, 4), (5, 7, 6)),
            ((1, 2, 3, 4, 4), (3, 8, 8)),
            ((2, 3, 4, 5, 5), (2, 3, 3)),
        ],
        ids=["doubled", "fractional", "depth-unchanged", "downsampled"],
    )
    def test_it_compiles_and_matches(self, shape, size, align):
        compiled, error = _run(shape, size, "trilinear", align)
        assert compiled, "fell back to eager — the emitter declined this node"
        assert error < 1e-5


class TestTheyAreNoLongerStubs:
    def test_both_have_emitters(self):
        for name in ("interpolate_nearest_3d", "interpolate_trilinear"):
            assert _C_engine.compile.emitter_registered(name), name

    def test_a_dynamic_batch_answers_at_every_size(self):
        """Only the trailing axes are gathered, so the batch stays symbolic:
        the trace-time batch is never baked into the graph."""
        model = _Resample((6, 8, 8), "nearest", False).eval().to("metal")
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(model, dynamic=True)
        for batch in (1, 3):
            x = lucid.randn(batch, 2, 3, 4, 4)
            got = compiled(x.to("metal")).to("cpu")
            reference = _Resample((6, 8, 8), "nearest", False).eval()(x)
            assert tuple(got.shape) == tuple(reference.shape)
            assert float((got - reference).abs().max().item()) == 0.0
