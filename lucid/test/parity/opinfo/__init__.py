"""OpInfo-based multi-argument, strided-view, and backward differential harness."""

from lucid.test.parity.opinfo._catalog_fft_ext import FFT_EXT_OPINFOS
from lucid.test.parity.opinfo._catalog_linalg_nn import LINALG_NN_OPINFOS
from lucid.test.parity.opinfo._catalog_math import MATH_OPINFOS
from lucid.test.parity.opinfo._catalog_reduction_index import REDUCTION_INDEX_OPINFOS
from lucid.test.parity.opinfo._core import (
    OpInfo,
    SampleInput,
    verify_opinfo_backward,
    verify_opinfo_forward,
    verify_opinfo_gradcheck,
    verify_opinfo_second_order_strided,
    verify_opinfo_strided_views,
)

OPINFO_REGISTRY: tuple[OpInfo, ...] = (
    *MATH_OPINFOS,
    *REDUCTION_INDEX_OPINFOS,
    *LINALG_NN_OPINFOS,
    *FFT_EXT_OPINFOS,
)

__all__ = [
    "FFT_EXT_OPINFOS",
    "LINALG_NN_OPINFOS",
    "MATH_OPINFOS",
    "OPINFO_REGISTRY",
    "OpInfo",
    "REDUCTION_INDEX_OPINFOS",
    "SampleInput",
    "verify_opinfo_backward",
    "verify_opinfo_forward",
    "verify_opinfo_gradcheck",
    "verify_opinfo_second_order_strided",
    "verify_opinfo_strided_views",
]
