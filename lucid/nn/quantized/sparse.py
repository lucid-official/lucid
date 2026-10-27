"""Quantized ``Embedding`` — int8 table storage, dequantize-on-lookup.

The embedding table is stored as int8 with a per-row (per-token) scale, which
is the dominant memory win for large-vocabulary embeddings; the forward
dequantizes the table and performs the ordinary lookup.
"""

from typing import TYPE_CHECKING, Protocol, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.quantized._utils import quantize_weight
from lucid.quantization._functional import dequantize
from lucid.quantization._qscheme import qint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    class _FloatEmbedding(Protocol):
        num_embeddings: int
        embedding_dim: int
        weight: Tensor
        padding_idx: int | None
        qconfig: object


class Embedding(nn.Module):
    r"""Quantized embedding lookup table — per-row int8, dequantize-on-lookup.

    The inference-time replacement that :func:`lucid.quantization.convert`
    installs for a calibrated float :class:`~lucid.nn.Embedding`. Embedding
    tables dominate the parameter budget of large-vocabulary language models —
    a ``(50000, 768)`` table is ~150 MB in ``float32`` — so storing the table as
    int8 is the single largest memory win quantization offers on the text side,
    and this layer is where it lands.

    **Representation.** The learned table ``(num_embeddings, embedding_dim)`` is
    quantized once, at :meth:`from_float` time, into an ``int8`` code tensor plus
    a **per-row** (per-token) ``scale`` / ``zero_point``; the float table is then
    dropped. Because each vocabulary row gets its own scale, tokens with very
    different embedding magnitudes are all tracked tightly — far more accurately
    than a single per-tensor scale. Each forward *dequantizes* the table back to
    float and runs the ordinary :func:`~lucid.nn.functional.embedding` lookup, so
    the layer needs no int8 gather kernel and runs unchanged on any device.

    Encoding happens once; decode + gather run on every forward. For row
    :math:`i`, column :math:`j`:

    .. math::

        \hat{W}_{ij} = (W^{q}_{ij} - z_i)\, s_i,
        \qquad
        \operatorname{out}[k] = \hat{W}[\, x_k\, ]

    where :math:`s_i, z_i` are the row-:math:`i` scale / zero-point (symmetric
    ``qint8``, so :math:`z_i = 0`) and :math:`x_k` the :math:`k`-th input index.
    The lookup gathers dequantized rows; only rows actually indexed are touched.

    Parameters
    ----------
    num_embeddings : int
        Size of the vocabulary (number of rows in the table).
    embedding_dim : int
        Dimensionality of each embedding vector.
    padding_idx : int or None, optional
        If given, the embedding at this index is treated as padding (its row is
        not accumulated into gradients upstream). Defaults to ``None``.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` table codes, shape ``(num_embeddings, embedding_dim)``.
    weight_scale : Tensor
        Per-row (per-token) scale, shape ``(num_embeddings,)``.
    weight_zero_point : Tensor
        Per-row (per-token) zero-point, shape ``(num_embeddings,)`` — all zero
        under the default symmetric ``qint8`` scheme.

    Notes
    -----
    - Instances are normally produced by :func:`lucid.quantization.convert` (or
      :meth:`from_float`), **not** constructed directly: a bare instance holds a
      zeroed table with identity qparams and returns zeros until ``from_float`` /
      ``load_state_dict`` populates the buffers.
    - The table is quantized **per-row on axis 0** (one ``scale`` per token) with
      a symmetric ``qint8`` grid (``[-128, 127]``). When the source module
      carries no ``qconfig``, a default per-channel ``qint8`` observer calibrates
      the rows during ``from_float``.
    - Memory: the int8 codes plus the per-row float scale shrink the table
      payload ~``3.55x`` versus ``float32`` — the dominant checkpoint saving for
      text models, whose weight mass is mostly embeddings.
    - This layer wins on **memory**, not compute — the gather runs in float after
      an on-the-fly dequantize.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> emb = nn.Embedding(1000, 64)
    >>> qemb = nn.quantized.Embedding.from_float(emb)   # per-row int8 table
    >>> qemb.weight_int8.dtype
    lucid.int8
    >>> qemb(lucid.tensor([1, 5, 999])).shape           # dequantize-on-lookup
    (3, 64)

    A directly constructed layer is zeroed — a common mistake is to use it
    without ``from_float`` / ``load_state_dict``:

    >>> broken = nn.quantized.Embedding(1000, 64)
    >>> bool((broken.weight_int8 == 0).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.EmbeddingBag : The pooled (bag-reducing) analogue.
    lucid.nn.quantized.Linear : Per-channel int8 weight for dense layers.
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    weight_int8: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight_ch_axis = 0
        self.register_buffer(
            "weight_int8",
            lucid.zeros((num_embeddings, embedding_dim), dtype=lucid.int8),
        )
        self.register_buffer("weight_scale", lucid.ones(num_embeddings))
        self.register_buffer("weight_zero_point", lucid.zeros(num_embeddings))

    @override
    def forward(self, indices: Tensor) -> Tensor:
        """Dequantize the table and look up ``indices``."""
        weight = dequantize(self.weight_int8, self.weight_scale, self.weight_zero_point, ch_axis=0)
        return F.embedding(indices, weight, self.padding_idx)

    @classmethod
    def from_float(cls, mod: nn.Module) -> Embedding:
        """Quantize a float :class:`~lucid.nn.Embedding`'s table (per-row int8)."""
        f = cast("_FloatEmbedding", mod)
        qmod = cls(f.num_embeddings, f.embedding_dim, padding_idx=f.padding_idx)
        # Per-row weight quantization (ch_axis 0 = per token) via the shared helper,
        # falling back to a default per-channel qint8 observer if no qconfig is set.
        if getattr(mod, "qconfig", None) is None:
            from lucid.quantization.observer import PerChannelMinMaxObserver
            from lucid.quantization._functional import quantize
            from lucid.quantization._qscheme import per_channel_symmetric

            obs = PerChannelMinMaxObserver(ch_axis=0, qscheme=per_channel_symmetric, qdtype=qint8)
            obs(f.weight)
            scale, zero_point = obs.calculate_qparams()
            codes = quantize(f.weight, scale, zero_point, qint8, ch_axis=0)
        else:
            codes, scale, zero_point, _ = quantize_weight(mod)
        qmod.register_buffer("weight_int8", codes)
        qmod.register_buffer("weight_scale", scale)
        qmod.register_buffer("weight_zero_point", zero_point)
        return qmod

    @override
    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class EmbeddingBag(nn.Module):
    r"""Quantized ``EmbeddingBag`` — per-row int8 table, pooled dequant lookup.

    The inference-time replacement for a calibrated float
    :class:`~lucid.nn.EmbeddingBag`, installed by
    :func:`lucid.quantization.convert` / :meth:`from_float`. Like
    :class:`~lucid.nn.quantized.Embedding` it stores the table as int8 with a
    per-row (per-token) ``scale`` — the same dominant memory win for
    large-vocabulary models — but instead of returning one row per index it
    *pools* each bag of indices into a single vector, the fused primitive behind
    recommendation and bag-of-words text models.

    **Representation.** The table ``(num_embeddings, embedding_dim)`` is
    quantized per-row into ``int8`` codes plus a per-row ``scale`` /
    ``zero_point`` at :meth:`from_float` time; the float table is dropped. Each
    forward dequantizes the table, gathers the rows in each bag, and reduces them
    by ``mode`` — fused so the per-token embedding matrix is never materialised.

    Given a flat index sequence partitioned into bags by ``offsets``, the
    :math:`i`-th pooled output is

    .. math::

        \hat{W}_{rc} = (W^{q}_{rc} - z_r)\, s_r,
        \qquad
        \operatorname{out}[i] = \operatorname{reduce}_{k \in \operatorname{bag}_i}
            \hat{W}[\, x_k\, ]

    where :math:`s_r, z_r` are the row-:math:`r` scale / zero-point (symmetric
    ``qint8``) and ``reduce`` is one of ``sum``, ``mean``, or ``max`` selected by
    ``mode``. The reduction runs on the dequantized (float) rows.

    Parameters
    ----------
    num_embeddings : int
        Size of the vocabulary (number of rows in the table).
    embedding_dim : int
        Dimensionality of each embedding vector.
    mode : {"sum", "mean", "max"}, optional
        Pooling reduction applied over each bag of gathered rows. Defaults to
        ``"mean"``.
    padding_idx : int or None, optional
        If given, the embedding at this index is treated as padding. Defaults
        to ``None``.

    Attributes
    ----------
    weight_int8 : Tensor
        The ``int8`` table codes, shape ``(num_embeddings, embedding_dim)``.
    weight_scale : Tensor
        Per-row (per-token) scale, shape ``(num_embeddings,)``.
    weight_zero_point : Tensor
        Per-row (per-token) zero-point, shape ``(num_embeddings,)`` — all zero
        under the default symmetric ``qint8`` scheme.

    Notes
    -----
    - Instances are normally produced by :func:`lucid.quantization.convert` (or
      :meth:`from_float`), **not** constructed directly: a bare instance holds a
      zeroed table with identity qparams and pools to zeros until ``from_float`` /
      ``load_state_dict`` populates the buffers.
    - The table is quantized **per-row on axis 0** (one ``scale`` per token) with
      a symmetric ``qint8`` grid. When the source module carries no ``qconfig``,
      a default per-channel ``qint8`` observer calibrates the rows.
    - Memory: the int8 codes plus per-row scale shrink the table payload
      ~``3.55x`` versus ``float32`` — as with :class:`Embedding`, the layer wins
      on memory, not compute (pooling runs in float post-dequantize).
    - ``offsets`` marks where each bag begins in a flat index tensor; pass a 2-D
      index tensor instead to treat every row as its own fixed-length bag.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> bag = nn.EmbeddingBag(1000, 32, mode="mean")
    >>> qbag = nn.quantized.EmbeddingBag.from_float(bag)
    >>> idx = lucid.tensor([1, 2, 4, 5, 4])
    >>> offsets = lucid.tensor([0, 3])              # two bags: [1,2,4] and [5,4]
    >>> qbag(idx, offsets).shape                    # one pooled vector per bag
    (2, 32)

    See Also
    --------
    lucid.nn.quantized.Embedding : The unpooled (one-row-per-index) analogue.
    lucid.nn.functional.embedding_bag : The pooled lookup run each forward.
    lucid.quantization.convert : Installs this layer from a calibrated model.
    """

    weight_int8: Tensor
    weight_scale: Tensor
    weight_zero_point: Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str = "mean",
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.padding_idx = padding_idx
        self.weight_ch_axis = 0
        self.register_buffer(
            "weight_int8",
            lucid.zeros((num_embeddings, embedding_dim), dtype=lucid.int8),
        )
        self.register_buffer("weight_scale", lucid.ones(num_embeddings))
        self.register_buffer("weight_zero_point", lucid.zeros(num_embeddings))

    @override
    def forward(
        self,
        indices: Tensor,
        offsets: Tensor | None = None,
        per_sample_weights: Tensor | None = None,
        compressed_indices_mapping: Tensor | None = None,
    ) -> Tensor:
        """Dequantize the table and run the pooled bag lookup."""
        if compressed_indices_mapping is not None:
            raise NotImplementedError(
                "quantized.EmbeddingBag: compressed_indices_mapping is not supported"
            )
        weight = dequantize(self.weight_int8, self.weight_scale, self.weight_zero_point, ch_axis=0)
        return F.embedding_bag(
            indices,
            weight,
            offsets,
            mode=self.mode,
            per_sample_weights=per_sample_weights,
            padding_idx=self.padding_idx,
        )

    @classmethod
    def from_float(cls, mod: nn.Module) -> EmbeddingBag:
        """Quantize a float :class:`~lucid.nn.EmbeddingBag`'s table (per-row int8)."""
        f = cast("_FloatEmbedding", mod)
        qmod = cls(
            f.num_embeddings,
            f.embedding_dim,
            mode=cast("str", getattr(mod, "mode", "mean")),
            padding_idx=getattr(mod, "padding_idx", None),
        )
        if getattr(mod, "qconfig", None) is None:
            from lucid.quantization._functional import quantize
            from lucid.quantization._qscheme import per_channel_symmetric
            from lucid.quantization.observer import PerChannelMinMaxObserver

            obs = PerChannelMinMaxObserver(ch_axis=0, qscheme=per_channel_symmetric, qdtype=qint8)
            obs(f.weight)
            scale, zero_point = obs.calculate_qparams()
            codes = quantize(f.weight, scale, zero_point, qint8, ch_axis=0)
        else:
            codes, scale, zero_point, _ = quantize_weight(mod)
        qmod.register_buffer("weight_int8", codes)
        qmod.register_buffer("weight_scale", scale)
        qmod.register_buffer("weight_zero_point", zero_point)
        return qmod

    @override
    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, mode={self.mode}"
        )
