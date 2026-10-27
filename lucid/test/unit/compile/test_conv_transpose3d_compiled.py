"""Transposed 3-D convolution, through the convolution's data gradient.

MPSGraph exposes ``convolutionTranspose2D`` and no 3-D counterpart, so
this was a stub filed under "SDK 2-D only".  That reason was true about
the name and wrong about the capability: a transposed convolution *is*
the data gradient of a convolution, and ``convolution3DDataGradient``
has been exposed since macOS 13.2.

The layout falls out of the same reading.  The forward convolution being
differentiated consumes this op's output and produces its input, so its
weight is OIDHW with O the transposed op's input channels — which is
exactly how Lucid stores the weight, ``(Cin, Cout/g, kD, kH, kW)``.
Nothing is permuted.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine


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


class Deconv(nn.Module):
    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(3, 4, kernel_size=3, **kwargs)  # type: ignore[arg-type]

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.up(x)


def _run(shape: tuple[int, ...], **kwargs: object) -> tuple[bool, float, int]:
    model = Deconv(**kwargs).eval()
    x = lucid.randn(*shape)
    reference = model(x)
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got = compiled(x.to("metal")).to("cpu")
    assert tuple(got.shape) == tuple(reference.shape)
    return (
        _C_engine.compile.session_cache_size() > 0,
        float((got - reference).abs().max().item()),
        len(got.shape),
    )


class TestConvTranspose3dCompiles:
    """Agreement to float32 rounding, across the shape-bearing options.

    ``stride`` and ``padding`` reach the descriptor; ``output_padding``
    does not, because the trace has already folded it into the output
    shape and the gradient leaves the extra tail at zero.  That last one
    is the case a wrong reading would break, so it is covered.
    """

    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("plain", {}),
            ("strided", {"stride": 2}),
            ("strided and padded", {"stride": 2, "padding": 1}),
            (
                "output padding",
                {"stride": 2, "padding": 1, "output_padding": 1},
            ),
        ],
        ids=["plain", "strided", "strided-padded", "output-padding"],
    )
    def test_it_compiles_and_matches(self, name: str, kwargs: dict[str, object]) -> None:
        compiled, difference, rank = _run((2, 3, 5, 6, 7), **kwargs)
        assert compiled, f"{name} fell back to eager"
        assert difference < 1e-5, f"{name} disagrees by {difference:.3e}"
        assert rank == 5

    def test_bias_is_added_on_the_channel_axis(self) -> None:
        """A bias broadcast along the wrong axis still yields rank 5."""
        with_bias, difference, _rank = _run((1, 3, 4, 4, 4), stride=2)
        assert with_bias
        assert difference < 1e-5

    def test_it_is_no_longer_a_stub(self) -> None:
        """Registration never changed; compiling did."""
        assert _C_engine.compile.emitter_registered("conv_transpose3d")
        compiled, _difference, _rank = _run((1, 3, 4, 4, 4))
        assert compiled
