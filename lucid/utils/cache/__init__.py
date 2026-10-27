"""
lucid.utils.cache: key/value caches for incremental attention (generation).

Class names, method names, and argument names mirror the reference
``cache_utils`` module so that generation code ports across with only an
import change.  The eager default is :class:`DynamicCache` (grows by
concatenation); :class:`StaticCache` is a fixed pre-allocated buffer that keeps
shapes constant for compiled single-token decoding; and encoder-decoder models
use :class:`EncoderDecoderCache` (paired self- + cross-attention).  Canonical
paths: ``lucid.utils.cache.Cache`` / ``DynamicCache`` / ``StaticCache`` /
``EncoderDecoderCache``.
"""

from lucid.utils.cache._base import Cache
from lucid.utils.cache._dynamic import DynamicCache
from lucid.utils.cache._encoder_decoder import EncoderDecoderCache
from lucid.utils.cache._static import StaticCache

__all__ = ["Cache", "DynamicCache", "EncoderDecoderCache", "StaticCache"]
