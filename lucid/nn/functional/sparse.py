"""
nn.functional sparse / embedding operations.
"""

from typing import TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._autocast import _autocast_suspended
from lucid._unsupported import unsupported_if
from lucid.nn.functional._index_checks import _check_indices, _check_table, _padding_index
from lucid.nn.functional.activations import straight_through

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def check_embedding_indices(x: Tensor, weight: Tensor, op: str) -> None:
    """Raise if any index in ``x`` falls outside the embedding table ``weight``.

    This is the check :func:`embedding` and
    :func:`~lucid.nn.functional.embedding_bag` run before their gather.  It
    is kept under this name for code that calls it directly.  The rule
    itself has one home, shared with every other op that reads with a
    class or table index (``one_hot``, the class-index losses,
    ``ctc_loss``), so they cannot disagree on what a valid index is.

    Parameters
    ----------
    x : Tensor
        Index tensor of any shape and any integer dtype (or bool).  An
        empty one passes.
    weight : Tensor
        Embedding table of shape ``(num_embeddings, embedding_dim)``.  Only
        its leading dimension is consulted.
    op : str
        Name of the calling op, used as the prefix of the raised message.

    Raises
    ------
    TypeError
        If ``x`` is not an integer (or bool) tensor.  A float index would
        pass a range check trivially and then be read by the gather as
        integer bits.
    IndexError
        If a CPU ``x`` holds an index that is negative or
        ``>= weight.shape[0]``.  A Metal ``x`` is not read back: the
        engine's gather isolates an out-of-range row on device (NaN for a
        floating table, 0 for an integer one).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional.sparse import check_embedding_indices
    >>> table = lucid.zeros(4, 2)
    >>> check_embedding_indices(lucid.tensor([0, 3]), table, "lookup")
    >>> check_embedding_indices(lucid.tensor([4]), table, "lookup")
    Traceback (most recent call last):
        ...
    IndexError: lookup: index 4 is out of range for a table with 4 entries (valid range [0, 3])
    """
    _check_table(x, weight, op)


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    r"""Look up rows of an embedding table by integer indices.

    A learned embedding table maps integer tokens / categorical features
    into dense vectors:

    .. math::

        \mathrm{out}[i_1, \dots, i_k] = W[\, x[i_1, \dots, i_k]\, ]

    where ``W`` of shape ``(num_embeddings, embedding_dim)`` is the
    lookup table and ``input`` holds integer indices in
    ``[0, num_embeddings)``.  Equivalent to a one-hot matmul
    :math:`\mathrm{onehot}(x) W` but computed with an indexed gather.

    Parameters
    ----------
    input : Tensor
        Integer index tensor of arbitrary shape ``(*)``, with entries in
        ``[0, num_embeddings)``.  A CPU entry outside that range raises
        ``IndexError`` before the lookup and a Metal one is isolated in
        the gather (see :func:`check_embedding_indices`); a float tensor
        raises ``TypeError``.
    weight : Tensor
        Embedding table of shape ``(num_embeddings, embedding_dim)``.
    padding_idx : int, optional
        If given, the embedding vector at ``weight[padding_idx]`` is
        treated as a padding slot: its gradient is forced to zero so the
        padding embedding stays at its initialised value (typically a
        zero vector) throughout training.  It must lie in
        ``[-num_embeddings, num_embeddings)`` (a negative one counts from
        the end), else ``ValueError``.
    max_norm : float, optional
        If given, every row of ``weight`` that ``input`` looks up and whose
        :math:`L_p` norm exceeds ``max_norm`` is rescaled in place, outside
        autograd, to norm ``max_norm`` before the lookup (with :math:`p` =
        ``norm_type``).  ``weight`` stays a leaf; a tensor that saved it for
        an earlier backward sees the write.
    norm_type : float, optional
        The :math:`p` value of the :math:`L_p` norm used by ``max_norm``.
        Default ``2.0``.
    scale_grad_by_freq : bool, optional
        If ``True``, the gradient reaching each row of ``weight`` from this
        call is divided by how many times that row's index occurs in
        ``input`` — useful for highly skewed token distributions.  The count
        is per call: two lookups before one backward are each scaled by
        their own.
    sparse : bool, optional
        Request a sparse gradient w.r.t. ``weight``.  Gradients are always
        dense, so ``True`` raises ``NotImplementedError``.

    Returns
    -------
    Tensor
        Embedded tensor of shape ``(*, embedding_dim)``.

    Notes
    -----
    The backward pass for ``embedding`` accumulates gradient contributions
    from repeated indices via scatter-add — multiple tokens of the same
    type in a batch correctly sum into the same row of
    :math:`\partial L / \partial W`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import embedding
    >>> table = lucid.randn(10, 4)              # 10 tokens, dim 4
    >>> ids = lucid.tensor([[1, 2, 4], [4, 3, 2]], dtype=lucid.int64)
    >>> out = embedding(ids, table)
    >>> out.shape
    (2, 3, 4)

    With ``scale_grad_by_freq``, row 1 — looked up three times — gets the
    mean of its gradients rather than their sum:

    >>> w = lucid.ones(5, 2, requires_grad=True)
    >>> ids = lucid.tensor([1, 1, 1, 3], dtype=lucid.int64)
    >>> embedding(ids, w, scale_grad_by_freq=True).sum().backward()
    >>> w.grad[1].tolist(), w.grad[3].tolist()
    ([1.0, 1.0], [1.0, 1.0])
    """
    unsupported_if(sparse, "embedding", "sparse", sparse, detail="Gradients are always dense.")
    padding_idx = _padding_index(padding_idx, weight, "embedding")
    _check_table(input, weight, "embedding")
    if max_norm is not None:
        _renorm_rows(weight, input, max_norm, norm_type)
    pad = padding_idx if padding_idx is not None else -1
    # Normalised, not passed through.  The engine gather reads the index
    # buffer at a fixed width, so an int8 or bool index of more than a
    # handful of entries is read past its own allocation — SIGBUS, the
    # same shape as the scatter_add defect.  int16/int32/int64 happened to
    # survive; relying on that is relying on an over-read staying inside
    # the page.
    idx = input if input.dtype == lucid.int64 else input.to(lucid.int64)
    by_freq = bool(scale_grad_by_freq)
    return _wrap(_C_engine.nn.embedding(_unwrap(weight), _unwrap(idx), pad, by_freq))


def _renorm_rows(weight: Tensor, idx: Tensor, max_norm: float, norm_type: float) -> None:
    """Rescale the rows of ``weight`` that ``idx`` names down to ``max_norm``, in place.

    The ``max_norm`` of every lookup — ``embedding``, ``embedding_bag`` and
    their modules — is this one write, as the reference makes it: only the
    rows looked up, only those whose :math:`L_p` norm exceeds ``max_norm``,
    each scaled by ``max_norm / (norm + 1e-7)``, under ``no_grad`` and
    through ``weight``'s own storage so a trainable table stays a leaf.  A
    row named twice is written twice with the same values.
    """
    flat = idx.reshape(-1)
    if flat.dtype != lucid.int64:
        flat = flat.to(lucid.int64)
    # An in-place write computes in ``weight``'s dtype, as the reference's
    # ``embedding_renorm_`` does, so it has no autocast row: under a Metal
    # autocast ``vector_norm``'s FP32 row made the rows float32 and the copy
    # into a float16 table was refused.
    with lucid.no_grad(), _autocast_suspended(_unwrap(weight).device):
        rows = weight.index_select(0, flat)
        norms = lucid.linalg.vector_norm(rows, norm_type, dim=1, keepdim=True)
        scale = lucid.where(norms > max_norm, max_norm / (norms + 1e-7), lucid.ones_like(norms))
        weight.index_copy_(0, flat, rows * scale)


def _bag_mode(mode: str, sparse: bool, scale_grad_by_freq: bool) -> int:
    """The engine's code for ``mode``, once the flags beside it are known to fit.

    Raises
    ------
    ValueError
        An unknown ``mode`` — ``.get(mode, 1)`` used to answer "mean" for a
        typo — or ``scale_grad_by_freq`` with ``mode="max"``, where one
        index per bag and dimension takes the whole gradient and there is
        no frequency to divide by (the reference refuses it too).
    NotImplementedError
        ``sparse=True``: gradients are always dense, as in ``embedding``.
    """
    modes = {"sum": 0, "mean": 1, "max": 2}
    if mode not in modes:
        raise ValueError(f"embedding_bag: mode must be one of {sorted(modes)}, got {mode!r}")
    unsupported_if(sparse, "embedding_bag", "sparse", sparse, detail="Gradients are always dense.")
    if scale_grad_by_freq and mode == "max":
        raise ValueError(
            "embedding_bag: scale_grad_by_freq is not defined for mode='max' — "
            "use mode='sum' or 'mean'"
        )
    return modes[mode]


def _bag_by_lookup(
    x: Tensor,
    weight: Tensor,
    bags: tuple[Tensor, int],
    per_sample_weights: Tensor | None,
    mode: str,
    padding_idx: int | None,
    scale_grad_by_freq: bool,
) -> Tensor:
    """``embedding_bag`` built from :func:`embedding`: rows weighted, then summed per bag.

    The path for what the fused engine op cannot do.  ``per_sample_weights``
    used to be accepted and dropped, so a weighted bag came back a plain sum
    with no error; ``scale_grad_by_freq`` likewise, giving every row the
    unscaled gradient.  Here each index's row comes from :func:`embedding`,
    which owns the frequency scaling at its lookup node, and is weighted by
    its sample weight — or by one over its bag's count for ``"mean"`` —
    before :func:`lucid.index_add` sums it into its bag.  It differentiates
    in both the table and the weights.

    Parameters
    ----------
    x : Tensor
        Indices, 1-D with ``offsets`` or 2-D with one bag per row.
    weight : Tensor
        The ``(num_embeddings, dim)`` table.
    bags : (Tensor, int)
        Each index's bag and the number of bags (:func:`_bag_of_each_index`).
    per_sample_weights : Tensor or None
        One weight per index, the shape of ``x``; ``"sum"`` only, as in
        the reference.
    mode : str
        ``"sum"`` or ``"mean"`` (``_bag_mode`` has refused the rest).
    padding_idx : int or None
        Rows with this index contribute nothing and are not counted.
    scale_grad_by_freq : bool
        Divide each row's gradient by its index's count in ``x``.

    Returns
    -------
    Tensor
        ``(num_bags, dim)`` pooled rows.

    Raises
    ------
    NotImplementedError
        ``per_sample_weights`` with a mode other than ``"sum"`` — ``"max"``
        included.
    ValueError
        If the weights do not match ``x``.
    """
    if per_sample_weights is not None:
        if mode != "sum":
            raise NotImplementedError(
                "embedding_bag: per_sample_weights is only supported for "
                f"mode='sum' (got mode={mode!r})"
            )
        if tuple(per_sample_weights.shape) != tuple(x.shape):
            raise ValueError(
                "embedding_bag: per_sample_weights must have the shape of the "
                f"indices {tuple(x.shape)}, got {tuple(per_sample_weights.shape)}"
            )
    flat = x.reshape(-1)
    rows = embedding(flat, weight, scale_grad_by_freq=scale_grad_by_freq)
    bag, n_bags = bags
    counted = bag >= 0
    if padding_idx is not None:
        counted = counted & (flat != padding_idx)
    if per_sample_weights is None:
        scale = counted.to(rows.dtype)
    else:
        scale = lucid.where(
            counted,
            per_sample_weights.reshape(-1).to(rows.dtype),
            lucid.zeros((), dtype=rows.dtype, device=rows.device),
        )
    bag = bag.clamp(min=0)
    zeros = lucid.zeros(n_bags, dtype=rows.dtype, device=rows.device)
    if mode == "mean":
        sizes = lucid.index_add(zeros, 0, bag, counted.to(rows.dtype))
        scale = scale / sizes.clamp(min=1.0).index_select(0, bag)
    out = lucid.zeros((n_bags, int(weight.shape[1])), dtype=rows.dtype, device=rows.device)
    return lucid.index_add(out, 0, bag, rows * scale.unsqueeze(-1))


def _bag_of_each_index(
    x: Tensor, offsets: Tensor | None, include_last_offset: bool
) -> tuple[Tensor, int]:
    """Which bag each index of ``x`` (flattened) falls in, ``-1`` for none; and how many bags.

    A 2-D ``x`` is one bag per row.  For a 1-D one, bag ``b`` owns
    ``[offsets[b], offsets[b + 1])``; under ``include_last_offset`` the last
    offset ends the final bag, and an index past it belongs to none.
    """
    device = x.device
    if x.ndim == 2:
        n_rows, length = int(x.shape[0]), int(x.shape[1])
        positions = lucid.arange(n_rows * length, dtype=lucid.int64, device=device)
        return positions // max(length, 1), n_rows
    if offsets is None:
        raise ValueError("embedding_bag: offsets required for 1-D input")
    bounds = offsets.to(lucid.int64).to(device)
    n_bags = int(bounds.shape[0]) - (1 if include_last_offset else 0)
    positions = lucid.arange(int(x.shape[0]), dtype=lucid.int64, device=device)
    bag = lucid.searchsorted(bounds[:n_bags], positions, right=True) - 1
    if include_last_offset:
        bag = lucid.where(positions < bounds[-1], bag, lucid.full_like(bag, -1))
    return bag, n_bags


def one_hot(input: Tensor, num_classes: int = -1) -> Tensor:
    r"""One-hot encode an integer class index tensor.

    Maps each integer entry into a one-hot vector along a new trailing
    axis of size ``num_classes``:

    .. math::

        \mathrm{out}[\ldots, c] =
            \begin{cases} 1 & \text{if } \mathrm{input}[\ldots] = c \\
                          0 & \text{otherwise} \end{cases}

    Parameters
    ----------
    input : Tensor
        Integer tensor of arbitrary shape ``(*)`` whose entries are class
        indices in ``[0, num_classes)``.
    num_classes : int, optional
        Total number of classes :math:`C`.  If ``-1`` (the default), it
        is inferred as ``input.max() + 1``; supplying an explicit value
        avoids a host round-trip and is preferred in hot loops.

    Returns
    -------
    Tensor
        One-hot encoded ``int64`` tensor of shape ``(*, num_classes)``.
        Cast to a floating dtype if it will participate in gradient-based
        computation.

    Raises
    ------
    ValueError
        If ``num_classes`` is ``-1`` and ``input`` is empty — there is no
        largest index to infer the count from — or if ``num_classes`` is
        otherwise not positive.
    IndexError
        If a CPU ``input`` holds a class outside ``[0, num_classes)``,
        a negative one included; it used to give a row of zeros.  A Metal
        ``input`` is not read back for the check (that would stall every
        step), and a class outside the range gives a row of zeros there.

    Notes
    -----
    For loss functions like cross-entropy, prefer passing raw integer
    targets to the loss directly — one-hot encoding then immediately
    contracting against a softmax wastes memory and breaks the
    log-sum-exp fused path.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import one_hot
    >>> idx = lucid.tensor([0, 2, 1, 2], dtype=lucid.int64)
    >>> one_hot(idx, num_classes=3)
    tensor([[1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]], dtype=lucid.int64)
    >>> one_hot(lucid.tensor([1, 0]))  # num_classes inferred as max + 1
    tensor([[0, 1],
            [1, 0]], dtype=lucid.int64)
    """
    if num_classes != -1 and num_classes <= 0:
        raise ValueError(
            f"one_hot: num_classes must be positive, or -1 to infer it, " f"got {num_classes}"
        )
    if num_classes == -1:
        # The engine takes only a positive count, so the default this
        # function has always advertised went straight through and was
        # refused.  Inferred here instead, as documented.
        if input.numel() == 0:
            raise ValueError(
                "one_hot: cannot infer num_classes from an empty tensor; "
                "pass num_classes explicitly"
            )
        num_classes = int(input.max().item()) + 1
    _check_indices(input, num_classes, "one_hot", what="class", axis=f"num_classes={num_classes}")
    out = _C_engine.nn.one_hot(_unwrap(input), num_classes)
    # int64, the dtype the reference framework answers in and the one class
    # indices are carried in everywhere else; the engine builds int8.
    return _wrap(_C_engine.astype(out, _C_engine.Dtype.I64))


def nearest_codebook(x: Tensor, codebook: Tensor) -> Tensor:
    r"""Index of the closest codebook entry for each row of ``x``.

    .. math::

        k_i = \arg\min_j \big\| x_i - e_j \big\|_2

    The search is the non-differentiable half of vector quantisation:
    the result is an integer field, so no gradient flows through it and
    none is defined.  :func:`vector_quantize` pairs it with the
    straight-through estimator to make the surrounding network trainable.

    Parameters
    ----------
    x : Tensor
        Query field of shape ``(*, D)``.  Only the trailing axis is
        treated as the feature dimension; everything before it is
        flattened into the search's row axis.
    codebook : Tensor
        Codebook of shape ``(K, D)``.

    Returns
    -------
    Tensor
        ``int64`` index field of shape ``(*)`` with values in ``[0, K)``.

    Notes
    -----
    Distances go through :func:`lucid.cdist`, whose ``p=2`` path uses the
    stable expansion :math:`\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2ab^\top`
    rather than materialising an ``(N, K, D)`` difference.  It does
    still build the ``(N, K)`` matrix, which at a large latent grid and
    a large codebook is the dominant allocation of a quantiser; a fused
    engine kernel that reduces over ``K`` without materialising it is the
    natural next step and would slot in behind this exact signature.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> codebook = lucid.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> x = lucid.tensor([[0.9, 1.1], [0.1, 0.0]])
    >>> F.nearest_codebook(x, codebook).tolist()
    [1, 0]
    """
    if codebook.ndim != 2:
        raise ValueError(f"codebook must be 2-D (K, D), got shape {codebook.shape}")
    dim = int(codebook.shape[-1])
    if int(x.shape[-1]) != dim:
        raise ValueError(
            f"x's trailing axis must match the codebook's, got " f"{int(x.shape[-1])} and {dim}"
        )

    lead = tuple(int(s) for s in x.shape[:-1])
    flat = x.reshape(-1, dim)
    idx = lucid.argmin(lucid.cdist(flat, codebook), dim=1)
    return idx.reshape(*lead) if lead else idx


def vector_quantize(x: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor]:
    r"""Snap ``x`` to its nearest codebook entries, straight-through.

    The functional core of :class:`lucid.nn.VectorQuantizer` — van den
    Oord, Vinyals, and Kavukcuoglu, *"Neural Discrete Representation
    Learning"* (2017).  Each row of ``x`` is replaced by the closest
    entry of ``codebook``, and the result is routed through
    :func:`straight_through` so the producer of ``x`` trains as though
    quantisation were the identity.

    Parameters
    ----------
    x : Tensor
        Field of shape ``(*, D)``; the trailing axis is the feature
        dimension.
    codebook : Tensor
        Codebook of shape ``(K, D)``.

    Returns
    -------
    quantized : Tensor
        Shape ``(*, D)``, numerically equal to the selected entries and
        differentiable with respect to ``x``.
    indices : Tensor
        ``int64`` field of shape ``(*)`` naming the selected entries.

    Notes
    -----
    The returned ``quantized`` carries **no** gradient to ``codebook`` —
    the straight-through path routes past it by construction.  Training
    the codebook needs the separate term
    :math:`\|\mathrm{sg}[x] - e\|_2^2`, which
    :class:`lucid.nn.VectorQuantizer` builds and returns alongside the
    commitment term.  Calling this function directly and optimising only
    a reconstruction loss leaves the codebook frozen at its
    initialisation — a silent failure that looks like a model which
    simply will not learn.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> codebook = lucid.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> x = lucid.tensor([[0.9, 1.1]], requires_grad=True)
    >>> quantized, indices = F.vector_quantize(x, codebook)
    >>> quantized.tolist(), indices.tolist()
    ([[1.0, 1.0]], [1])
    """
    indices = nearest_codebook(x, codebook)
    dim = int(codebook.shape[-1])
    lead = tuple(int(s) for s in x.shape[:-1])
    hard = embedding(indices.reshape(-1), codebook).reshape(*lead, dim)
    return straight_through(hard, x), indices


def two_hot(values: Tensor, bins: Tensor) -> Tensor:
    r"""Encode continuous scalars as a distribution over two adjacent bins.

    The generalisation of :func:`one_hot` to values that fall *between*
    classes: all entries are zero except at the two bins bracketing the
    scalar, which carry linearly interpolated weight summing to one.

    Parameters
    ----------
    values : Tensor
        Scalars to encode, any shape ``(...)``.
    bins : Tensor
        Bin locations, ``(K,)``, strictly increasing.  Values outside the
        range are clamped onto the end bins.

    Returns
    -------
    Tensor
        ``(..., K)``, non-negative and summing to 1 along the last axis.

    Notes
    -----
    Why a regression head would want this: predicting a scalar with a
    squared error forces the network to commit to one number, and the
    mean of a bimodal target is a value the target never takes.  A
    distribution over bins can represent "either 0 or 10", and the scalar
    is recovered afterwards as :math:`\sum_k p_k b_k`.  Trained with a
    cross-entropy, the gradient no longer scales with the error either,
    which is what lets one set of hyperparameters cover reward scales
    that differ by orders of magnitude.

    Exact on a bin: a value landing on ``bins[k]`` puts all its weight
    there, so this reduces to :func:`one_hot` on the grid.

    See Also
    --------
    one_hot : The discrete case.
    lucid.nn.functional.symlog : Usually applied before encoding, so that
        the grid can be uniform while the bins are not.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> F.two_hot(lucid.tensor([1.5]), lucid.tensor([0.0, 1.0, 2.0]))
    tensor([[0., 0.5, 0.5]])
    """
    count = int(bins.shape[0])
    if bins.ndim != 1 or count < 2:
        raise ValueError(
            f"bins must be a 1-D tensor of at least two entries, got shape "
            f"{tuple(int(s) for s in bins.shape)}"
        )

    # The ends of the grid as tensors, not host floats: reading them with
    # ``.item()`` synced the device on every call, and inside a compile
    # trace it read a traced value on the host, which sends the whole step
    # eager — DreamerV3's reward and value heads encode through here.
    lower, upper = bins[0].to(values.dtype), bins[count - 1].to(values.dtype)
    clamped = lucid.minimum(lucid.maximum(values, lower), upper).unsqueeze(dim=-1)

    # Weight every bin by how close it is, then keep only the two that
    # bracket the value: `below` counts the bins at or under it, so
    # `below - 1` and `below` are the neighbours.
    below = (bins <= clamped).to(clamped.dtype).sum(dim=-1) - 1.0
    below = below.clip(0.0, float(count - 2))
    index = below.unsqueeze(dim=-1)

    positions = lucid.arange(0, count, 1, dtype=clamped.dtype, device=clamped.device)
    left = (positions == index).to(clamped.dtype)
    right = (positions == index + 1.0).to(clamped.dtype)

    left_edge = (left * bins).sum(dim=-1, keepdim=True)
    right_edge = (right * bins).sum(dim=-1, keepdim=True)
    span = (right_edge - left_edge).clip(1e-12, None)
    weight = ((clamped - left_edge) / span).clip(0.0, 1.0)

    return left * (1.0 - weight) + right * weight
