"""Sampling a feature map at a flow field, compiled.

The stub reason said MPSGraph's ``gatherAlongAxis`` cannot express a
per-pixel bilinear gather. It can, once the spatial axes are flattened:
the sample coordinates collapse to one integer index per output
position, and ``gatherAlongAxis`` takes an index *tensor*, so the
data-dependence was never the obstacle. What remains is arithmetic.

The coordinates here run past ``[-1, 1]`` on purpose. Everything inside
the image agrees whatever the padding mode is, so a grid that stays
inside would pass while ``zeros`` and ``border`` were swapped.
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


class _Sampler(nn.Module):
    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        mode: str,
        padding_mode: str,
        align_corners: bool,
        spread: float = 2.6,
    ) -> None:
        super().__init__()
        lucid.manual_seed(1)
        batch, height, width = grid_shape
        self.register_buffer("grid", lucid.rand(batch, height, width, 2) * spread - spread / 2)
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return F.grid_sample(
            x,
            self.grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


def _run(
    shape: tuple[int, int, int, int],
    grid_shape: tuple[int, int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    spread: float = 2.6,
) -> tuple[bool, float]:
    lucid.manual_seed(0)
    model = _Sampler(grid_shape, mode, padding_mode, align_corners, spread).eval()
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


class TestEverySettingCompiles:
    @pytest.mark.parametrize("align_corners", [False, True], ids=["edges", "corners"])
    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    def test_it_matches_eager(self, mode, padding_mode, align_corners):
        compiled, error = _run(
            (1, 3, 5, 5),
            (1, 4, 4),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        assert compiled, "fell back to eager — the emitter declined this node"
        if mode == "nearest":
            # Nearest selects a pixel; the compiled graph reads the same
            # one or it read the wrong index.
            assert error == 0.0
        else:
            assert error < 1e-5


class TestTheShapesThatIndexArithmeticCouldGetWrong:
    """Flattening H and W hides a stride; a square image would not show it."""

    @pytest.mark.parametrize(
        ("shape", "grid_shape"),
        [
            ((1, 3, 4, 7), (1, 5, 5)),
            ((1, 3, 7, 4), (1, 5, 5)),
            ((2, 5, 6, 3), (2, 3, 8)),
            ((1, 1, 2, 2), (1, 6, 6)),
        ],
        ids=["wide", "tall", "batched", "tiny-source"],
    )
    def test_it_matches_eager(self, shape, grid_shape):
        compiled, error = _run(shape, grid_shape)
        assert compiled
        assert error < 1e-5

    @pytest.mark.parametrize("spread", [0.5, 2.0, 6.0], ids=["inside", "edge", "far"])
    def test_coordinates_far_outside_still_agree(self, spread):
        """Zero padding has to hold however far out the coordinate lands."""
        compiled, error = _run((1, 2, 4, 4), (1, 5, 5), spread=spread)
        assert compiled
        assert error < 1e-5


class TestTheSpatialTransformerComposes:
    """``affine_grid`` then ``grid_sample`` — the pair this exists for."""

    def test_the_whole_transformer_compiles(self):
        class Transformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                lucid.manual_seed(3)
                self.register_buffer("theta", lucid.randn(1, 2, 3) * 0.3)

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                grid = F.affine_grid(self.theta, (1, 3, 6, 6), align_corners=False)
                return F.grid_sample(x, grid, align_corners=False)

        lucid.manual_seed(0)
        model = Transformer().eval()
        x = lucid.randn(1, 3, 6, 6)
        reference = model(x)

        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile.compile(model.to("metal"))
        got = compiled(x.to("metal")).to("cpu")

        assert _C_engine.compile.session_cache_size() > 0
        scale = max(float(reference.abs().max().item()), 1e-30)
        assert float((got - reference).abs().max().item()) / scale < 1e-5


class TestItIsNoLongerAStub:
    def test_a_dynamic_batch_declines_rather_than_pinning_one(self):
        """The index arithmetic is built around a concrete batch."""
        model = _Sampler((1, 4, 4), "bilinear", "zeros", False).eval().to("metal")
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(model, dynamic=True)
        x = lucid.randn(1, 3, 5, 5)
        got = compiled(x.to("metal")).to("cpu")
        reference = _Sampler((1, 4, 4), "bilinear", "zeros", False).eval()(x)
        scale = max(float(reference.abs().max().item()), 1e-30)
        assert float((got - reference).abs().max().item()) / scale < 1e-5
