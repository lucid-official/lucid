"""``lucid.nn.intrinsic.qat`` — fused QAT modules (Conv+BN fold-in-training)."""

from lucid.nn.intrinsic.qat.modules import (
    ConvBn1d,
    ConvBn2d,
    ConvBn3d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBnReLU3d,
)

# The internal ``convert`` builder ``_convbn_to_quantized`` lives in ``.modules``
# and is imported there directly by ``lucid.quantization._quantize`` —
# deliberately not re-exported here (it is not part of the public surface).
__all__ = ["ConvBn1d", "ConvBn2d", "ConvBn3d", "ConvBnReLU1d", "ConvBnReLU2d", "ConvBnReLU3d"]


def __dir__() -> list[str]:
    """Restrict introspection to the public module."""
    return list(__all__)
