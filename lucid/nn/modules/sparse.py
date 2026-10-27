"""
Sparse / embedding modules.
"""

from typing import NamedTuple, override

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid as _lucid
import lucid.nn.init as init
from lucid.nn.functional.activations import straight_through
from lucid.nn.functional.sparse import embedding, nearest_codebook, one_hot


class Embedding(Module):
    r"""Learnable dense lookup table that maps integer token indices to vectors.

    An embedding table is a matrix :math:`W \in \mathbb{R}^{V \times D}`
    where :math:`V` = ``num_embeddings`` (vocabulary size) and
    :math:`D` = ``embedding_dim``.  The forward pass is a simple index
    operation:

    .. math::

        y = W[\text{idx}]

    Each row :math:`W[i]` is the dense representation (embedding vector)
    of token :math:`i`.  Because indexing is not differentiable with
    respect to the integer indices themselves, gradients flow only into
    the rows of :math:`W` that were selected during the forward pass.

    **Padding index.** When ``padding_idx`` is set, the corresponding
    row is initialised to zero and its gradient is masked to zero during
    backpropagation.  This is the standard way to mark a special
    ``<PAD>`` token whose embedding should never influence the model.

    **Max-norm renormalisation.** When ``max_norm`` is set, every row
    looked up whose :math:`L_p`-norm exceeds ``max_norm`` is rescaled
    *in place* at each forward call before the lookup:

    .. math::

        W[i] \leftarrow W[i] \cdot \frac{\texttt{max\_norm}}{\|W[i]\|_p}
        \quad \text{if } \|W[i]\|_p > \texttt{max\_norm}

    Parameters
    ----------
    num_embeddings : int
        Size of the embedding dictionary (vocabulary size :math:`V`).
    embedding_dim : int
        Dimensionality of each embedding vector (:math:`D`).
    padding_idx : int or None, optional
        If provided, the embedding for this index is fixed at zero and
        receives no gradient updates.  Negative values are normalised
        to ``padding_idx + num_embeddings``.  Default: ``None``.
    max_norm : float or None, optional
        If provided, the rows looked up whose :math:`L_p`-norm exceeds
        this value are renormalised in place at every forward call.
        Default: ``None``.
    norm_type : float, optional
        The :math:`p` in the :math:`L_p`-norm used by ``max_norm``.
        Default: ``2.0``.
    scale_grad_by_freq : bool, optional
        Divide each row's gradient by the number of times that index
        appeared in the lookup, so a token seen fifty times in a batch
        does not take a fifty-times larger step than one seen once.  The
        count belongs to each forward call: two lookups before one
        ``backward()`` are each scaled by their own.  Default: ``False``.
    sparse : bool, optional
        Gradients are always dense, so ``True`` raises
        ``NotImplementedError`` at the first forward.  Default: ``False``.
    device : DeviceLike, optional
        Device for the weight tensor.
    dtype : DTypeLike, optional
        Data type for the weight tensor.

    Attributes
    ----------
    weight : Parameter, shape ``(num_embeddings, embedding_dim)``
        The embedding matrix :math:`W`.  Rows are initialised from
        :math:`\mathcal{N}(0, 1)`.  If ``padding_idx`` is set, that
        row is zeroed immediately after initialisation.

    Shape
    -----
    * **Input** ``x``: ``(*)`` — integer tensor of arbitrary shape
      with values in ``[0, num_embeddings)``.
    * **Output**: ``(*, embedding_dim)`` — the input shape with an
      extra trailing dimension of size ``embedding_dim``.

    Notes
    -----
    The weight matrix is initialised from a standard normal distribution
    :math:`\mathcal{N}(0, 1)`, which gives each embedding a unit-order
    magnitude.  For downstream layers that are sensitive to input scale
    (e.g. transformers), consider dividing by :math:`\sqrt{D}` after
    construction.

    Embedding is commonly used in natural-language processing (token
    embeddings, position encodings), recommendation systems (item /
    user embeddings), and any domain where discrete categorical inputs
    must be projected into a continuous representation space.

    Examples
    --------
    Simple token embedding for a vocabulary of 100 with 16-dim vectors:

    >>> import lucid, lucid.nn as nn
    >>> emb = nn.Embedding(num_embeddings=100, embedding_dim=16)
    >>> idx = lucid.tensor([[1, 5, 3], [0, 2, 7]], dtype=lucid.int64)
    >>> y = emb(idx)
    >>> y.shape    # (2, 3, 16)
    (2, 3, 16)

    Using ``padding_idx`` to mark a ``<PAD>`` token (index 0):

    >>> emb_pad = nn.Embedding(50, 8, padding_idx=0)
    >>> # Row 0 is always zero and never updated
    >>> import lucid.linalg
    >>> float(lucid.linalg.norm(emb_pad.weight[0])) == 0.0
    True
    >>> idx2 = lucid.tensor([0, 3, 0, 7], dtype=lucid.int64)
    >>> emb_pad(idx2).shape
    (4, 8)

    See Also
    --------
    EmbeddingBag : Efficient embedding lookup with per-bag reduction.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the Embedding module. See the class docstring for parameter semantics."""
        super().__init__()
        if padding_idx is not None and not (-num_embeddings <= padding_idx < num_embeddings):
            raise ValueError(
                f"padding_idx must be within [-{num_embeddings}, "
                f"{num_embeddings}); got {padding_idx}"
            )
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        # Normalise negative padding_idx for downstream comparisons.
        self.padding_idx: int | None = (
            padding_idx + num_embeddings
            if padding_idx is not None and padding_idx < 0
            else padding_idx
        )
        self.max_norm: float | None = max_norm
        self.norm_type: float = norm_type
        self.scale_grad_by_freq: bool = scale_grad_by_freq
        self.sparse: bool = sparse
        self.weight: Parameter = Parameter(
            empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        init.normal_(self.weight)
        # Zero out the pad row on init — matches the reference framework
        # so untouched models do not leak random values through the pad slot.
        if self.padding_idx is not None:
            self._zero_pad_row()

    def _zero_pad_row(self) -> None:
        """Set ``weight[padding_idx]`` to zero, written through in place.

        Through the weight's own storage, the way ``nn.init`` writes, so the
        parameter stays a leaf that requires grad.  This used to compute the
        zeroed table with gradient tracking on and swap it in as the
        weight's impl: the weight became the output of an ``index_fill``
        node rather than a leaf, and backward never accumulated into its
        ``.grad``.  Every embedding built with a ``padding_idx`` — BERT's and
        RoFormer's word tables among them — was untrainable, with no error.
        """
        pad = _lucid.tensor(
            [int(self.padding_idx)], dtype=_lucid.int64, device=self.weight.device  # type: ignore[arg-type]
        )
        with _lucid.no_grad():
            zeroed: Tensor = _lucid.index_fill(self.weight, 0, pad, 0.0)
        init._fill_from_impl(self.weight, zeroed._impl)

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Look up embeddings for the given indices.

        Parameters
        ----------
        input : Tensor
            Tensor of integer indices.

        Returns
        -------
        Tensor
            Tensor of embedding vectors of shape ``(*input.shape, embedding_dim)``.
        """
        return embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        s: str = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        if self.norm_type != 2.0:
            s += f", norm_type={self.norm_type}"
        return s


class EmbeddingBag(Module):
    r"""Embedding lookup table with per-bag reduction.

    Computes a summary embedding for each *bag* (a variable-length set
    of token indices) without materialising the full per-token embedding
    tensor.  Given an embedding matrix
    :math:`W \in \mathbb{R}^{V \times D}` and a bag of indices
    :math:`\mathcal{B} = \{i_1, i_2, \ldots, i_k\}`, the output
    embedding for that bag is:

    .. math::

        y =
        \begin{cases}
            \displaystyle\sum_{j \in \mathcal{B}} W[j]
              & \text{mode} = \texttt{'sum'} \\[6pt]
            \displaystyle\frac{1}{|\mathcal{B}|}\sum_{j \in \mathcal{B}} W[j]
              & \text{mode} = \texttt{'mean'} \\[6pt]
            \displaystyle\max_{j \in \mathcal{B}} W[j]
              & \text{mode} = \texttt{'max'}
        \end{cases}

    where the :math:`\max` is taken element-wise across the embedding
    dimension.

    **2-D input (fixed-length bags).** When ``offsets`` is ``None`` the
    input ``x`` must have shape ``(B, L)`` — ``B`` bags each containing
    exactly ``L`` indices.  The reduction is applied over the ``L``
    axis, yielding an output of shape ``(B, D)``.

    **1-D input with offsets (variable-length bags).** When ``offsets``
    is provided, ``x`` is a flat 1-D integer tensor of all indices
    concatenated, and ``offsets`` is a 1-D integer tensor of length
    ``B`` marking the start position of each bag.  Bag :math:`b`
    consists of indices ``x[offsets[b] : offsets[b+1]]`` (the last bag
    runs to the end of ``x``).

    Parameters
    ----------
    num_embeddings : int
        Size of the embedding dictionary (vocabulary size :math:`V`).
    embedding_dim : int
        Dimensionality of each embedding vector (:math:`D`).
    max_norm : float or None, optional
        If provided, rows with :math:`L_p`-norm exceeding this value
        are renormalised before the lookup (see :class:`Embedding`).
        Default: ``None``.
    norm_type : float, optional
        The :math:`p` for ``max_norm`` renormalisation.  Default: ``2.0``.
    scale_grad_by_freq : bool, optional
        Divide each row's gradient by how many times its index occurs in
        the input (see :class:`Embedding`).  Not with ``mode='max'``.
        Default: ``False``.
    mode : {'sum', 'mean', 'max'}, optional
        Reduction applied over each bag.  Default: ``'mean'``.
    sparse : bool, optional
        Gradients are always dense, so ``True`` raises
        ``NotImplementedError`` at the first forward.  Default: ``False``.
    padding_idx : int or None, optional
        Indices equal to ``padding_idx`` contribute zero to the
        reduction and do not receive gradient updates.
        Default: ``None``.
    device : DeviceLike, optional
        Device for the weight tensor.
    dtype : DTypeLike, optional
        Data type for the weight tensor.

    Attributes
    ----------
    weight : Parameter, shape ``(num_embeddings, embedding_dim)``
        The embedding matrix :math:`W`, initialised from
        :math:`\mathcal{N}(0, 1)`.

    Shape
    -----
    * **x** (2-D path): ``(B, L)`` integer indices → output ``(B, D)``.
    * **x** (1-D + offsets path): ``(total_indices,)`` integer indices,
      ``offsets`` ``(B,)`` → output ``(B, D)``.

    Notes
    -----
    :class:`EmbeddingBag` is more memory-efficient than computing
    ``Embedding(x).sum(dim=1)`` because it fuses the lookup and
    reduction into a single kernel call, avoiding the intermediate
    ``(B, L, D)`` tensor.  This is especially beneficial for large
    vocabularies and long bags.

    ``'max'`` mode is not differentiable with respect to ties; the
    gradient is propagated only through the index that achieved the
    maximum.

    Examples
    --------
    Fixed-length bags (2-D input), mean pooling:

    >>> import lucid, lucid.nn as nn
    >>> emb_bag = nn.EmbeddingBag(num_embeddings=20, embedding_dim=8, mode='mean')
    >>> idx = lucid.tensor([[1, 3, 5], [0, 2, 4]], dtype=lucid.int64)  # (B=2, L=3)
    >>> y = emb_bag(idx)
    >>> y.shape    # (B=2, D=8)
    (2, 8)

    Variable-length bags via offsets (sum pooling):

    >>> emb_sum = nn.EmbeddingBag(10, 4, mode='sum')
    >>> flat_idx = lucid.tensor([0, 1, 2, 3, 4, 5], dtype=lucid.int64)  # 6 total
    >>> offsets = lucid.tensor([0, 2, 5], dtype=lucid.int64)  # bags: [0,1], [2,3,4], [5]
    >>> y2 = emb_sum(flat_idx, offsets)
    >>> y2.shape    # (B=3, D=4)
    (3, 4)

    See Also
    --------
    Embedding : Per-token embedding lookup without reduction.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        _weight: Tensor | None = None,
        include_last_offset: bool = False,
        padding_idx: int | None = None,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the EmbeddingBag module. See the class docstring for parameter semantics."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset
        self.padding_idx = padding_idx
        if _weight is None:
            self.weight = Parameter(
                empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
            )
            init.normal_(self.weight)
        else:
            self.weight = Parameter(_weight)

    @override
    def forward(
        self, input: Tensor, offsets: Tensor | None = None, per_sample_weights: Tensor | None = None
    ) -> Tensor:
        """Look up embeddings for the given indices.

        Parameters
        ----------
        input : Tensor
            Tensor of integer indices.

        Returns
        -------
        Tensor
            Tensor of embedding vectors of shape ``(*input.shape, embedding_dim)``.
        """
        from lucid.nn.functional.sampling import embedding_bag as _eb

        return _eb(
            input,
            self.weight,
            offsets,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.mode,
            self.sparse,
            per_sample_weights=per_sample_weights,
            include_last_offset=self.include_last_offset,
            padding_idx=self.padding_idx,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"mode={self.mode!r}, padding_idx={self.padding_idx}"
        )


class VectorQuantizerOutput(NamedTuple):
    r"""What :class:`VectorQuantizer` returns.

    Attributes
    ----------
    quantized : Tensor
        The quantised field, same shape as the input.  Numerically equal
        to the selected codebook entries, but differentiable with respect
        to the input via the straight-through estimator.
    indices : Tensor
        Selected entry per position — the input's shape with the trailing
        feature axis dropped, ``int64`` dtype.
    codebook_loss : Tensor
        Scalar :math:`\| \mathrm{sg}[x] - e \|_2^2`, which trains the
        codebook and is detached from the input.
    commitment_loss : Tensor
        Scalar :math:`\| x - \mathrm{sg}[e] \|_2^2`, which trains the
        input's producer and is detached from the codebook.  Reported
        *unweighted* — apply ``commitment_cost`` when summing the terms.
    perplexity : Tensor
        Scalar :math:`\exp(-\sum_k p_k \log p_k)` over the batch's usage
        histogram, in ``[1, num_embeddings]``.  A value falling toward 1
        means the codebook is collapsing to a few live entries.
    """

    quantized: Tensor
    indices: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor
    perplexity: Tensor


class VectorQuantizer(Module):
    r"""Nearest-neighbour codebook lookup with a straight-through gradient.

    Maps each position of a continuous feature field to the closest entry
    of a learned codebook :math:`e \in \mathbb{R}^{K \times D}`:

    .. math::

        k = \arg\min_j \big\| x - e_j \big\|_2,
        \qquad
        z_q = e_k .

    This is the discretisation layer of van den Oord, Vinyals, and
    Kavukcuoglu, *"Neural Discrete Representation Learning"* (2017) — the
    bottleneck that turns an auto-encoder into a **tokeniser** whose
    latent is an integer field a downstream model can treat as a
    vocabulary.

    Like :class:`Linear` and :class:`Embedding`, the layer acts on the
    **trailing** axis: an input of shape ``(*, embedding_dim)`` produces a
    quantised field of the same shape and an index field of shape ``(*)``.
    Image models holding ``(N, C, H, W)`` should permute the channel axis
    last before calling and back afterwards.

    Because :math:`\arg\min` has zero gradient almost everywhere, the
    forward value is routed through :func:`lucid.nn.functional.
    straight_through`, so the producer of ``x`` trains as if the
    quantisation were the identity.  The codebook itself receives *no*
    gradient from that path, which is why the layer also returns the two
    terms that train it — see :class:`VectorQuantizerOutput`.

    Parameters
    ----------
    num_embeddings : int
        Codebook size :math:`K`.
    embedding_dim : int
        Code dimension :math:`D`; must match the input's trailing axis.
    commitment_cost : float, optional
        The coefficient :math:`\beta` the caller is expected to apply to
        ``commitment_loss``.  Stored for introspection and reused by
        :meth:`loss`; the returned ``commitment_loss`` is unweighted so
        callers can log the raw term.  Default ``0.25`` (the paper's
        value).
    device : DeviceLike, optional
        Device for the codebook parameter.
    dtype : DTypeLike, optional
        Dtype for the codebook parameter.

    Attributes
    ----------
    weight : Parameter
        The codebook, shaped ``(num_embeddings, embedding_dim)``.
        Initialised ``Uniform(-1/K, 1/K)``: entries must start inside the
        range the producer emits early in training, or a subset is never
        selected and never receives gradient.

    Notes
    -----
    Reference: van den Oord, Vinyals, and Kavukcuoglu, NeurIPS, 2017
    (arXiv:1711.00937).  The straight-through estimator is Bengio,
    Léonard, and Courville (arXiv:1308.3432).

    Distances go through :func:`lucid.cdist`, whose ``p=2`` path uses the
    numerically stable expansion
    :math:`\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2 a b^\top` rather than
    materialising an ``(N, K, D)`` difference.

    Perplexity is computed on every forward.  It costs one ``(N, K)``
    one-hot, the same order as the distance matrix already built, so it is
    always on rather than gated behind a flag — codebook collapse is the
    dominant failure mode of this layer and is invisible in the loss.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> vq = nn.VectorQuantizer(num_embeddings=64, embedding_dim=8)
    >>> x = lucid.randn((2, 5, 8))           # (batch, positions, D)
    >>> out = vq(x)
    >>> out.quantized.shape, out.indices.shape
    ((2, 5, 8), (2, 5))
    >>> bool((out.indices.max() < 64).item())
    True
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the VectorQuantizer. See the class docstring for parameter semantics."""
        super().__init__()
        if num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if commitment_cost < 0.0:
            raise ValueError(f"commitment_cost must be non-negative, got {commitment_cost}")

        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        self.commitment_cost: float = commitment_cost

        self.weight: Parameter = Parameter(
            empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        bound = 1.0 / float(num_embeddings)
        init.uniform_(self.weight, -bound, bound)

    def assign(self, x: Tensor) -> Tensor:
        """Return the nearest-entry index field for ``x``, shape ``(*)``."""
        return nearest_codebook(x, self.weight)

    def lookup(self, indices: Tensor) -> Tensor:
        """Map an index field ``(*)`` back to codes ``(*, embedding_dim)``."""
        lead = tuple(int(s) for s in indices.shape)
        flat = embedding(indices.reshape(-1), self.weight)
        return flat.reshape(*lead, self.embedding_dim)

    @override
    def forward(self, x: Tensor) -> VectorQuantizerOutput:
        """Quantise ``x`` along its trailing axis."""
        indices = nearest_codebook(x, self.weight)
        # The *hard* lookup, not the straight-through result: both loss
        # terms need one side genuinely detached, and reading them off the
        # estimator's output would leak a gradient path back into ``x``.
        # ``F.vector_quantize`` is the same two calls for callers who only
        # want the quantised field.
        hard = self.lookup(indices)

        codebook_loss = ((hard - x.detach()) ** 2).mean()
        commitment_loss = ((x - hard.detach()) ** 2).mean()

        onehot = one_hot(indices.reshape(-1), num_classes=self.num_embeddings).to(x.dtype)
        probs = onehot.mean(dim=0)
        perplexity = _lucid.exp(-(probs * (probs + 1e-10).log()).sum())

        return VectorQuantizerOutput(
            quantized=straight_through(hard, x),
            indices=indices,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    def loss(self, out: VectorQuantizerOutput) -> Tensor:
        """Combine the two codebook terms using this layer's ``commitment_cost``."""
        return out.codebook_loss + self.commitment_cost * out.commitment_loss

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"commitment_cost={self.commitment_cost}"
        )
