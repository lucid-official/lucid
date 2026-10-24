"""Grouped and dilated transposed convolution through the compile path.

Until the engine carried them, ``groups`` and ``dilation`` were never
recorded on a ``conv_transpose`` node, so every emitter read the default
and no test could tell the difference.  Once the trace started carrying
them, the rank-5 emitter's hardcoded ``groups:1`` met a grouped weight
and MPSGraph rejected the whole module — an abort inside the compiler,
not a Python exception, so nothing downstream could catch or report it.

These are the shapes that reach a real emitter, so the assertion that
matters is ``session_cache_size() > 0``: without it a silent fall back to
eager would agree with eager perfectly and prove nothing.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine

LAYERS = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = [
    pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable"),
    pytest.mark.usefixtures("inference"),
]


class _Deconv(nn.Module):
    def __init__(self, rank: int, c_in: int, c_out: int, **kwargs: object) -> None:
        super().__init__()
        self.up = LAYERS[rank](c_in, c_out, kernel_size=3, **kwargs)  # type: ignore[arg-type]

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.up(x)


def _compile_and_compare(
    rank: int, c_in: int, c_out: int, size: int, **kwargs: object
) -> tuple[bool, float]:
    lucid.manual_seed(rank)
    model = _Deconv(rank, c_in, c_out, **kwargs).eval()
    x = lucid.randn(1, c_in, *([size] * rank))
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


class TestGroupedAndDilatedCompile:
    @pytest.mark.parametrize(
        ("rank", "c_in", "c_out", "kwargs"),
        [
            (1, 4, 4, {"stride": 2, "padding": 1, "groups": 2}),
            (1, 3, 4, {"stride": 1, "dilation": 2}),
            (2, 6, 6, {"stride": 2, "padding": 1, "groups": 3}),
            (2, 6, 6, {"stride": 2, "padding": 1, "groups": 6}),
            (2, 3, 4, {"stride": 2, "padding": 1, "dilation": 2}),
            (2, 4, 4, {"stride": 2, "padding": 2, "groups": 2, "dilation": 2}),
            (3, 4, 4, {"stride": 2, "padding": 1, "groups": 2}),
            (3, 4, 4, {"stride": 2, "padding": 1, "groups": 4}),
            (3, 2, 4, {"stride": 1, "dilation": 2}),
        ],
        ids=[
            "1d-grouped",
            "1d-dilated",
            "2d-grouped",
            "2d-depthwise",
            "2d-dilated",
            "2d-grouped-dilated",
            "3d-grouped",
            "3d-depthwise",
            "3d-dilated",
        ],
    )
    def test_it_compiles_and_matches(self, rank, c_in, c_out, kwargs):
        compiled, error = _compile_and_compare(rank, c_in, c_out, 5, **kwargs)
        assert compiled, "fell back to eager — the emitter declined this node"
        assert error < 1e-5

    def test_the_default_path_is_unchanged(self):
        """The ungrouped, undilated case still compiles and agrees."""
        compiled, error = _compile_and_compare(2, 3, 4, 5, stride=2, padding=1)
        assert compiled
        assert error < 1e-5
