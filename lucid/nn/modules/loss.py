"""Loss function modules."""

from collections.abc import Callable
from typing import ClassVar, override

from lucid._tensor.tensor import Tensor
from lucid._types import Reduction, ReductionKL
from lucid.nn.functional._loss_checks import _Loss, _default_distance, _legacy_reduction
from lucid.nn.functional.loss import _check_beta, _check_delta, _check_label_smoothing
from lucid.nn.functional.loss import _check_multi_margin, _check_reduction, _check_triplet_margin
from lucid.nn.functional.loss import binary_cross_entropy, binary_cross_entropy_with_logits
from lucid.nn.functional.loss import cosine_embedding_loss, cross_entropy, ctc_loss, huber_loss
from lucid.nn.functional.loss import gaussian_nll_loss, hinge_embedding_loss, kl_div, l1_loss
from lucid.nn.functional.loss import margin_ranking_loss, mse_loss, multi_margin_loss, nll_loss
from lucid.nn.functional.loss import multilabel_margin_loss, multilabel_soft_margin_loss
from lucid.nn.functional.loss import poisson_nll_loss, smooth_l1_loss, soft_margin_loss
from lucid.nn.functional.loss import triplet_margin_loss, triplet_margin_with_distance_loss
from lucid.nn.module import Module

type _Flag = bool | None
type _Meta = dict[str, object]


class MSELoss(_Loss):
    r"""Mean squared error (MSE) loss between each element of the prediction
    and the target.

    This is the canonical regression loss that penalises large deviations
    quadratically.  With ``reduction='mean'`` the per-element squared
    differences are averaged over all elements:

    .. math::

        \mathcal{L}(x, y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - y_i)^2

    With ``reduction='sum'`` the sum is taken instead, and with
    ``reduction='none'`` the full element-wise tensor is returned unchanged.

    Parameters
    ----------
    reduction : str, optional
        Specifies the reduction to apply to the output.
        ``'none'`` — no reduction, element-wise output.
        ``'mean'`` — average over all elements (default).
        ``'sum'``  — sum over all elements.

    Attributes
    ----------
    reduction : str
        The reduction mode set at construction time.

    Shape
    -----
    - **Input** ``x``   : :math:`(*)` — any shape.
    - **Target** ``y``  : :math:`(*)` — same shape as ``x``.
    - **Output**        : scalar when ``reduction`` is ``'mean'`` or
      ``'sum'``; :math:`(*)` when ``reduction='none'``.

    Notes
    -----
    - Gradients scale linearly with the residual magnitude, which can
      make training sensitive to outliers and large-scale targets.
      Consider :class:`HuberLoss` or :class:`L1Loss` when outliers are
      present.
    - MSE is proportional to the negative log-likelihood under a Gaussian
      observation model with unit variance.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MSELoss()
    >>> x = lucid.tensor([2.5, 0.0, 2.0, 8.0])
    >>> y = lucid.tensor([3.0, -0.5, 2.0, 7.0])
    >>> loss = criterion(x, y)  # scalar

    Element-wise output with ``reduction='none'``:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MSELoss(reduction="none")
    >>> x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> y = lucid.tensor([[1.5, 1.5], [2.5, 4.5]])
    >>> loss = criterion(x, y)  # shape (2, 2)
    """

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return mse_loss(input, target, reduction=self.reduction)


class L1Loss(_Loss):
    r"""Mean absolute error (MAE) loss between each element of the prediction
    and the target.

    Also known as the *L1 loss*, this criterion computes the element-wise
    absolute difference and optionally reduces it:

    .. math::

        \mathcal{L}(x, y) = \frac{1}{n} \sum_{i=1}^{n} |x_i - y_i|

    With ``reduction='sum'`` the sum is taken instead, and with
    ``reduction='none'`` the full element-wise tensor is returned.

    Parameters
    ----------
    reduction : str, optional
        Specifies the reduction applied to the output.
        ``'none'`` — no reduction, element-wise output.
        ``'mean'`` — average over all elements (default).
        ``'sum'``  — sum over all elements.

    Attributes
    ----------
    reduction : str
        The reduction mode set at construction time.

    Shape
    -----
    - **Input** ``x``   : :math:`(*)` — any shape.
    - **Target** ``y``  : :math:`(*)` — same shape as ``x``.
    - **Output**        : scalar when ``reduction`` is ``'mean'`` or
      ``'sum'``; :math:`(*)` when ``reduction='none'``.

    Notes
    -----
    - Unlike :class:`MSELoss`, gradients are constant in magnitude
      (the subgradient is ±1), making L1 loss more robust to outliers.
    - The non-differentiability at zero can cause numerical instability
      near convergence for some optimisers; :class:`SmoothL1Loss` provides
      a twice-differentiable alternative.
    - L1 loss is proportional to the negative log-likelihood under a
      Laplace (double-exponential) observation model.

    Examples
    --------
    Scalar regression:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.L1Loss()
    >>> x = lucid.tensor([2.5, 0.0, 2.0, 8.0])
    >>> y = lucid.tensor([3.0, -0.5, 2.0, 7.0])
    >>> loss = criterion(x, y)  # scalar ≈ 0.5

    Comparing element-wise residuals:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.L1Loss(reduction="none")
    >>> x = lucid.tensor([[1.0, 3.0], [0.0, -1.0]])
    >>> y = lucid.tensor([[2.0, 1.0], [0.0,  1.0]])
    >>> loss = criterion(x, y)  # shape (2, 2): [[1., 2.], [0., 2.]]
    """

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return l1_loss(input, target, reduction=self.reduction)


class _WeightedLoss(_Loss):
    """Base of the losses that hold a ``weight`` (and ``pos_weight``) tensor.

    The tensors are registered buffers, as in the reference: ``.to()`` moves
    them with the module and ``state_dict`` saves them.  They used to be
    plain attributes, so ``.to("metal")`` left them on the CPU — the first
    forward raised ``DeviceMismatch`` — and a checkpoint did not hold them.

    Version 2 is the version that saves them.  A checkpoint written before
    (version 1, or a plain dict without metadata) has no key for them;
    loading it, strict or not, keeps the tensors the module was built with
    rather than reporting the keys missing.
    """

    _version: ClassVar[int] = 2
    #: The buffers an older checkpoint may lack.
    _weight_buffers: ClassVar[tuple[str, ...]] = ("weight",)

    weight: Tensor | None

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialise the weighted loss module."""
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)

    @override
    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: _Meta,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load as usual, excusing the weight keys a version-1 checkpoint lacks."""
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        version = local_metadata.get("version") if local_metadata else None
        if isinstance(version, int) and version >= 2:
            return
        for name in self._weight_buffers:
            key = f"{prefix}{name}"
            if key not in state_dict and key in missing_keys:
                missing_keys.remove(key)


class CrossEntropyLoss(_WeightedLoss):
    r"""Cross-entropy loss for multi-class classification.

    This criterion combines a log-softmax and a negative log-likelihood
    step in a single numerically stable operation.  For a batch of ``N``
    samples, each of class index ``y_n`` from ``C`` classes, and raw
    logit vector ``x_n``:

    .. math::

        \mathcal{L}(x, y) = -x_{n,y_n}
            + \log \sum_{c=1}^{C} \exp(x_{n,c})

    The log-sum-exp trick is applied internally so that large logit values
    do not cause overflow or underflow.

    **Label smoothing** — when ``label_smoothing`` :math:`= \varepsilon > 0`
    the hard target is softened to a mixture of the one-hot label and the
    uniform distribution:

    .. math::

        \tilde{y}_{n,c} = (1-\varepsilon)\,\mathbf{1}[c = y_n]
            + \frac{\varepsilon}{C}

    which replaces the loss with:

    .. math::

        \mathcal{L}_\varepsilon
            = (1-\varepsilon)\,\mathcal{L}(x,y)
            + \frac{\varepsilon}{C} \sum_{c=1}^{C} \mathcal{L}(x,c)

    Parameters
    ----------
    weight : Tensor of shape (C,), optional
        Manual rescaling weight assigned to each class.  Useful for
        imbalanced datasets.  Must be a 1-D float tensor of length ``C``.
    ignore_index : int, optional
        Specifies a target value that is ignored and does not contribute
        to the gradient.  Default ``-100``.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.
    label_smoothing : float, optional
        Smoothing parameter :math:`\varepsilon \in [0, 1]`.  Default ``0.0``
        (no smoothing).

    Attributes
    ----------
    weight : Tensor or None
        Per-class weight tensor, or ``None`` if not provided.
        A registered buffer: moved by ``.to()`` and saved in
        ``state_dict``.
    ignore_index : int
        Target index excluded from loss and gradient computation.
    reduction : str
        The reduction mode.
    label_smoothing : float
        The smoothing coefficient :math:`\varepsilon`.

    Shape
    -----
    - **Input** ``x``     : :math:`(N, C)` or :math:`(N, C, d_1, d_2, \ldots)`
      — raw unnormalised logits.
    - **Target** ``y``    : :math:`(N,)` or :math:`(N, d_1, d_2, \ldots)`
      — integer class indices in :math:`[0, C)`.
    - **Output**          : scalar when ``reduction`` is ``'mean'`` or
      ``'sum'``; :math:`(N,)` or :math:`(N, d_1, \ldots)` for ``'none'``.

    Notes
    -----
    - Passing logits rather than softmax probabilities is strongly
      recommended for numerical stability — the internal log-sum-exp
      implementation avoids catastrophic cancellation.
    - Equivalent to ``NLLLoss(LogSoftmax(x, dim=1), y)`` but computed
      in a single pass.

    Examples
    --------
    Three-class classification with a batch of two samples:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.CrossEntropyLoss()
    >>> x = lucid.tensor([[0.1, 0.9, 0.0], [2.0, 0.5, 0.1]])
    >>> y = lucid.tensor([1, 0])
    >>> loss = criterion(x, y)  # scalar

    With label smoothing and per-class weights:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.CrossEntropyLoss(
    ...     weight=lucid.tensor([1.0, 2.0, 1.0]),
    ...     label_smoothing=0.1,
    ... )
    >>> x = lucid.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 1.5]])
    >>> y = lucid.tensor([1, 2])
    >>> loss = criterion(x, y)
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: Reduction = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        """Initialise the CrossEntropyLoss module. See the class docstring for parameter semantics."""
        super().__init__(weight, size_average, reduce, reduction)
        _check_label_smoothing(label_smoothing, "CrossEntropyLoss")
        _check_multi_margin(1, weight, "CrossEntropyLoss")
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"ignore_index={self.ignore_index}, reduction={self.reduction!r}, "
            f"label_smoothing={self.label_smoothing}"
        )


class NLLLoss(_WeightedLoss):
    r"""Negative log-likelihood loss.

    Operates on **log-probabilities** — the input is expected to already
    be the output of a ``LogSoftmax`` layer.  For sample ``n`` with true
    class label ``y_n`` the per-sample loss is:

    .. math::

        \ell_n = -x_{n,\,y_n}

    and the final scalar is obtained via the chosen ``reduction``.

    This is the second stage of the two-step formulation of multi-class
    cross-entropy: ``LogSoftmax`` followed by ``NLLLoss``.  When a single
    fused module is preferred, use :class:`CrossEntropyLoss` instead.

    Parameters
    ----------
    weight : Tensor of shape (C,), optional
        Manual rescaling weight for each class.  Useful for imbalanced
        class distributions.
    ignore_index : int, optional
        Target value that is ignored and does not contribute to the
        gradient.  Default ``-100``.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    weight : Tensor or None
        Per-class weight tensor.
        A registered buffer: moved by ``.to()`` and saved in
        ``state_dict``.
    ignore_index : int
        Excluded target index.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **Input** ``x``   : :math:`(N, C)` or :math:`(N, C, d_1, \ldots)`
      — log-probabilities (e.g. output of ``LogSoftmax``).
    - **Target** ``y``  : :math:`(N,)` or :math:`(N, d_1, \ldots)`
      — integer class indices.
    - **Output**        : scalar for ``'mean'`` / ``'sum'``;
      :math:`(N,)` for ``'none'``.

    Notes
    -----
    - The input must contain log-probabilities.  Feeding raw logits or
      softmax probabilities produces incorrect results.
    - Typical usage: ``log_p = lucid.nn.functional.log_softmax(logits, dim=1)``,
      then ``loss = NLLLoss()(log_p, targets)``.

    Examples
    --------
    Manual log-softmax then NLLLoss:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.nn.functional as F
    >>> log_softmax = nn.LogSoftmax(dim=1)
    >>> criterion  = nn.NLLLoss()
    >>> logits = lucid.tensor([[0.5, 1.5, 0.3], [1.2, 0.1, 0.9]])
    >>> log_probs = log_softmax(logits)
    >>> targets   = lucid.tensor([1, 0])
    >>> loss = criterion(log_probs, targets)

    With a custom per-class weight:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.NLLLoss(weight=lucid.tensor([1.0, 3.0, 1.0]))
    >>> log_probs = lucid.tensor([[-1.2, -0.5, -2.1], [-0.8, -1.5, -0.3]])
    >>> targets   = lucid.tensor([1, 2])
    >>> loss = criterion(log_probs, targets)
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialise the NLLLoss module. See the class docstring for parameter semantics."""
        super().__init__(weight, size_average, reduce, reduction)
        _check_multi_margin(1, weight, "NLLLoss")
        self.ignore_index: int = ignore_index

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return nll_loss(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        s: str = f"reduction={self.reduction!r}"
        if self.ignore_index != -100:
            s += f", ignore_index={self.ignore_index}"
        return s


class BCELoss(_WeightedLoss):
    r"""Binary cross-entropy loss.

    Measures the element-wise binary cross-entropy between predictions
    ``x`` (probabilities in :math:`(0, 1)`) and binary targets ``y``
    (values in :math:`\{0, 1\}`):

    .. math::

        \ell(x, y) = -\bigl[y \log x + (1 - y) \log(1 - x)\bigr]

    The input **must** be valid probabilities, i.e. produced by a
    ``Sigmoid`` activation.  Passing raw logits directly leads to
    undefined behaviour; use :class:`BCEWithLogitsLoss` instead for
    better numerical stability when starting from logits.

    Parameters
    ----------
    weight : Tensor, optional
        Element-wise weight tensor that is broadcast over the input and
        target tensors.  Must be broadcastable to their shape.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    weight : Tensor or None
        Optional element-wise weighting.
        A registered buffer: moved by ``.to()`` and saved in
        ``state_dict``.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **Input** ``x``   : :math:`(*)` — probabilities in :math:`(0, 1)`.
    - **Target** ``y``  : :math:`(*)` — binary labels in :math:`\{0, 1\}`.
    - **Output**        : scalar for ``'mean'`` / ``'sum'``;
      :math:`(*)` for ``'none'``.

    Notes
    -----
    - Values of ``x`` outside :math:`(0, 1)` will produce ``NaN``
      or ``inf`` losses due to the logarithm.
    - For logits (pre-sigmoid values), prefer :class:`BCEWithLogitsLoss`
      which uses the identity
      :math:`\log(1 + e^{-x}) = \max(x,0) + \log(1 + e^{-|x|})`.

    Examples
    --------
    Basic binary classification:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.BCELoss()
    >>> probs   = lucid.tensor([0.9, 0.1, 0.8, 0.3])
    >>> targets = lucid.tensor([1.0, 0.0, 1.0, 0.0])
    >>> loss = criterion(probs, targets)

    With element-wise sample weighting:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.BCELoss(weight=lucid.tensor([2.0, 1.0, 2.0, 1.0]))
    >>> probs   = lucid.tensor([0.7, 0.2, 0.6, 0.4])
    >>> targets = lucid.tensor([1.0, 0.0, 1.0, 0.0])
    >>> loss = criterion(probs, targets)
    """

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(_WeightedLoss):
    r"""Binary cross-entropy loss that accepts raw logits.

    Combines a ``Sigmoid`` activation with a binary cross-entropy loss in
    a single numerically stable expression.  Using the identity:

    .. math::

        \log(1 + e^x) = \max(x, 0) + \log\!\bigl(1 + e^{-|x|}\bigr)

    the per-element loss is computed as:

    .. math::

        \ell(x, y) = \max(x, 0) - x\,y
            + \log\!\bigl(1 + e^{-|x|}\bigr)

    When a **positive-class weight** :math:`p` is supplied (``pos_weight``),
    the loss becomes:

    .. math::

        \ell(x, y) = -p \cdot y \log \sigma(x)
            - (1 - y) \log(1 - \sigma(x))

    where :math:`\sigma` is the sigmoid function.  A value of ``pos_weight
    > 1`` up-weights the positive class, useful when positives are rare.

    Parameters
    ----------
    weight : Tensor, optional
        Element-wise weight tensor broadcast over input and target.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.
    pos_weight : Tensor, optional
        Weight for the positive class, shape ``(C,)`` or scalar.
        Provides class-level rebalancing independent of ``weight``.

    Attributes
    ----------
    weight : Tensor or None
        Element-wise weight.
        A registered buffer: moved by ``.to()`` and saved in
        ``state_dict``.
    reduction : str
        The reduction mode.
    pos_weight : Tensor or None
        Positive-class weight.
        A registered buffer: moved by ``.to()`` and saved in
        ``state_dict``.

    Shape
    -----
    - **Input** ``x``   : :math:`(*)` — raw logits (any real value).
    - **Target** ``y``  : :math:`(*)` — binary labels in :math:`\{0, 1\}`.
    - **Output**        : scalar for ``'mean'`` / ``'sum'``;
      :math:`(*)` for ``'none'``.

    Notes
    -----
    - Numerically superior to ``BCELoss(Sigmoid(x), y)`` because the
      log-domain computation avoids squashing gradients to zero near
      saturation.
    - ``pos_weight`` values significantly larger than 1 can destabilise
      training; values in ``[1, 10]`` are typically safe.

    Examples
    --------
    Raw logits, no weighting:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.BCEWithLogitsLoss()
    >>> logits  = lucid.tensor([ 2.0, -1.0, 0.5, -3.0])
    >>> targets = lucid.tensor([ 1.0,  0.0, 1.0,  0.0])
    >>> loss = criterion(logits, targets)

    Up-weighting the positive class by a factor of 10:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.BCEWithLogitsLoss(
    ...     pos_weight=lucid.tensor([10.0])
    ... )
    >>> logits  = lucid.tensor([0.2, -0.8, 1.5])
    >>> targets = lucid.tensor([1.0,  0.0, 1.0])
    >>> loss = criterion(logits, targets)
    """

    _weight_buffers: ClassVar[tuple[str, ...]] = ("weight", "pos_weight")

    pos_weight: Tensor | None

    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: Reduction = "mean",
        pos_weight: Tensor | None = None,
    ) -> None:
        """Initialise the BCEWithLogitsLoss module. See the class docstring for parameter semantics."""
        super().__init__(weight, size_average, reduce, reduction)
        self.register_buffer("pos_weight", pos_weight)

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return binary_cross_entropy_with_logits(
            input, target, weight=self.weight, pos_weight=self.pos_weight, reduction=self.reduction
        )


class HuberLoss(_Loss):
    r"""Huber loss — a smooth interpolation between MSE and MAE.

    For each element the per-element loss is:

    .. math::

        \ell_\delta(x, y) =
        \begin{cases}
            \tfrac{1}{2}(x - y)^2
                & \text{if } |x - y| < \delta \\[4pt]
            \delta \!\left(|x - y| - \tfrac{\delta}{2}\right)
                & \text{otherwise}
        \end{cases}

    The scalar loss is the mean (or sum, or element-wise) of
    :math:`\ell_\delta` over all elements in the batch.

    For small residuals the loss is quadratic (same as MSE) and for large
    residuals it is linear (same as MAE), with a smooth join at
    :math:`|x - y| = \delta`.

    Parameters
    ----------
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.
    delta : float, optional
        Threshold that separates the quadratic and linear regions.
        Default ``1.0``.

    Attributes
    ----------
    reduction : str
        The reduction mode.
    delta : float
        The threshold :math:`\delta`.

    Shape
    -----
    - **Input** ``x``   : :math:`(*)`.
    - **Target** ``y``  : :math:`(*)` — same shape as ``x``.
    - **Output**        : scalar for ``'mean'`` / ``'sum'``;
      :math:`(*)` for ``'none'``.

    Notes
    -----
    - Choosing :math:`\delta` controls the robustness/sensitivity
      trade-off: larger :math:`\delta` behaves more like MSE, smaller
      :math:`\delta` behaves more like MAE.
    - :class:`SmoothL1Loss` is identical to :class:`HuberLoss` but
      parameterised by :math:`\beta = \delta` and normalised so that the
      two formulations match when :math:`\delta = \beta = 1`.

    Examples
    --------
    Default :math:`\delta = 1`:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.HuberLoss()
    >>> x = lucid.tensor([1.0,  3.0, -1.5])
    >>> y = lucid.tensor([1.5, -1.0,  0.0])
    >>> loss = criterion(x, y)

    Custom threshold to tolerate larger outliers:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.HuberLoss(delta=2.0)
    >>> x = lucid.tensor([0.0, 5.0, -4.0])
    >>> y = lucid.tensor([0.0, 1.0,  0.0])
    >>> loss = criterion(x, y)
    """

    def __init__(self, reduction: Reduction = "mean", delta: float = 1.0) -> None:
        """Initialise the HuberLoss module. See the class docstring for parameter semantics."""
        super().__init__(reduction=reduction)
        _check_delta(delta, "HuberLoss")
        self.delta = delta

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return huber_loss(input, target, reduction=self.reduction, delta=self.delta)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"delta={self.delta}, reduction={self.reduction!r}"


class SmoothL1Loss(_Loss):
    r"""Smooth L1 loss — a :math:`\beta`-parameterised Huber loss.

    This loss is equivalent to :class:`HuberLoss` with :math:`\delta = \beta`
    but uses a slightly different normalisation convention.  The per-element
    form is:

    .. math::

        \ell_\beta(x, y) =
        \begin{cases}
            \dfrac{(x - y)^2}{2\,\beta}
                & \text{if } |x - y| < \beta \\[6pt]
            |x - y| - \dfrac{\beta}{2}
                & \text{otherwise}
        \end{cases}

    When :math:`\beta = 1` this coincides exactly with the Huber loss.

    Parameters
    ----------
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.
    beta : float, optional
        Transition threshold between the quadratic and linear regions.
        Default ``1.0``.

    Attributes
    ----------
    reduction : str
        The reduction mode.
    beta : float
        The threshold :math:`\beta`.

    Shape
    -----
    - **Input** ``x``   : :math:`(*)`.
    - **Target** ``y``  : :math:`(*)` — same shape as ``x``.
    - **Output**        : scalar for ``'mean'`` / ``'sum'``;
      :math:`(*)` for ``'none'``.

    Notes
    -----
    - Smooth L1 is commonly used in object detection (bounding-box
      regression) because it is less sensitive to outlier predictions
      than MSE while still being differentiable everywhere.
    - Setting :math:`\beta \to 0` approaches MAE; :math:`\beta \to \infty`
      approaches MSE (for bounded residuals).

    Examples
    --------
    Default :math:`\beta = 1`:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.SmoothL1Loss()
    >>> x = lucid.tensor([0.5, 2.0, -1.0])
    >>> y = lucid.tensor([0.0, 0.0,  0.0])
    >>> loss = criterion(x, y)

    Tighter quadratic region (:math:`\beta = 0.5`):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.SmoothL1Loss(beta=0.5)
    >>> x = lucid.tensor([0.3, 1.5, -0.2])
    >>> y = lucid.tensor([0.0, 0.0,  0.0])
    >>> loss = criterion(x, y)
    """

    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: Reduction = "mean",
        beta: float = 1.0,
    ) -> None:
        """Initialise the SmoothL1Loss module. See the class docstring for parameter semantics."""
        super().__init__(size_average, reduce, reduction)
        _check_beta(beta, "SmoothL1Loss")
        self.beta = beta

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return smooth_l1_loss(input, target, beta=self.beta, reduction=self.reduction)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"beta={self.beta}, reduction={self.reduction!r}"


class KLDivLoss(Module):
    r"""Kullback–Leibler divergence loss.

    Measures how one probability distribution diverges from a reference
    distribution.  The input ``x`` must be **log-probabilities** (e.g.
    output of ``LogSoftmax``), and the target ``y`` must be probabilities
    (or log-probabilities when ``log_target=True``).

    With ``log_target=False`` (default):

    .. math::

        \ell(x, y) = y \cdot (\log y - x)

    With ``log_target=True`` (target already in log-space):

    .. math::

        \ell(x, y) = e^{y} \cdot (y - x)

    The scalar is obtained by reducing :math:`\ell` according to
    ``reduction``.  Note that ``'batchmean'`` (if supported) divides by
    the batch size :math:`N`, which corresponds to the mathematical KL
    definition.

    Parameters
    ----------
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'`` | ``'batchmean'``.
    log_target : bool, optional
        If ``True``, ``target`` is interpreted as log-probabilities.
        Default ``False``.

    Attributes
    ----------
    reduction : str
        The reduction mode.
    log_target : bool
        Whether the target is in log-space.

    Shape
    -----
    - **Input** ``x``   : :math:`(*)` — log-probabilities.
    - **Target** ``y``  : :math:`(*)` — probabilities (or log-probabilities
      when ``log_target=True``).
    - **Output**        : scalar for ``'mean'`` / ``'sum'`` / ``'batchmean'``;
      :math:`(*)` for ``'none'``.

    Notes
    -----
    - KL divergence is asymmetric: :math:`\text{KL}(P \| Q) \neq \text{KL}(Q \| P)`.
    - Common applications include variational autoencoders (VAE), knowledge
      distillation, and training language models.
    - Passing raw probabilities (non-log) as ``x`` is a common mistake and
      will produce incorrect and potentially negative values.

    Examples
    --------
    Comparing two discrete distributions (``'batchmean'`` follows the KL
    mathematical convention):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.nn.functional as F
    >>> criterion = nn.KLDivLoss(reduction="batchmean")
    >>> log_pred = F.log_softmax(lucid.tensor([[0.5, 1.0, 0.2]]), dim=1)
    >>> target   = F.softmax(lucid.tensor([[0.3, 0.9, 0.4]]),   dim=1)
    >>> loss = criterion(log_pred, target)

    With log-space target (knowledge distillation style):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.nn.functional as F
    >>> criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
    >>> log_p = F.log_softmax(lucid.tensor([[1.0, 2.0, 0.5]]), dim=1)
    >>> log_q = F.log_softmax(lucid.tensor([[0.8, 1.5, 0.9]]), dim=1)
    >>> loss = criterion(log_p, log_q)
    """

    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: ReductionKL = "mean",
        log_target: bool = False,
    ) -> None:
        """Initialise the KLDivLoss module. See the class docstring for parameter semantics."""
        super().__init__()
        self.reduction: ReductionKL = _legacy_reduction(
            size_average, reduce, reduction, "KLDivLoss", batchmean=True
        )
        self.log_target = log_target

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return kl_div(input, target, reduction=self.reduction, log_target=self.log_target)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"reduction={self.reduction!r}, log_target={self.log_target}"


class TripletMarginLoss(_Loss):
    r"""Triplet margin loss for metric learning.

    Trains an embedding such that an *anchor* sample is closer to a
    *positive* (same-class) sample than to a *negative* (different-class)
    sample by at least ``margin``.  Given embeddings
    :math:`(a, p, n)`:

    .. math::

        \mathcal{L}(a, p, n) =
            \max\!\bigl(d(a, p) - d(a, n) + \text{margin},\; 0\bigr)

    where the distance is the :math:`L_p` norm:

    .. math::

        d(x, y) = \left\| x - y \right\|_p

    The **swap** option: if enabled, the loss also considers
    :math:`d(p, n)` as an alternative negative distance and uses the
    smaller of :math:`d(a, n)` and :math:`d(p, n)`.

    Parameters
    ----------
    margin : float, optional
        Minimum required distance gap.  Default ``1.0``.
    p : float, optional
        The norm degree for the distance computation.  Default ``2.0`` (L2).
    eps : float, optional
        Small value added inside the norm to avoid zero-division.
        Default ``1e-6``.
    swap : bool, optional
        If ``True``, uses the triangle-inequality-based swap.
        Default ``False``.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    margin : float
        The margin threshold.
    p : float
        The norm degree.
    eps : float
        Numerical stabilisation constant.
    swap : bool
        Whether the distance swap is active.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **anchor**   : :math:`(N, D)`.
    - **positive** : :math:`(N, D)`.
    - **negative** : :math:`(N, D)`.
    - **Output**   : scalar for ``'mean'`` / ``'sum'``;
      :math:`(N,)` for ``'none'``.

    Notes
    -----
    - Used extensively in face verification, image retrieval, and
      few-shot learning.
    - For variable distance functions (e.g. cosine distance), see
      :class:`TripletMarginWithDistanceLoss`.
    - Choosing ``margin`` depends on the scale of the embedding space;
      ``0.2`` to ``1.0`` is typical for L2-normalised embeddings.

    Examples
    --------
    Basic triplet training step:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.TripletMarginLoss(margin=1.0)
    >>> anchor   = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> positive = lucid.tensor([[1.1, 2.1, 3.1]])
    >>> negative = lucid.tensor([[5.0, 6.0, 7.0]])
    >>> loss = criterion(anchor, positive, negative)

    With L1 distance and larger margin:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.TripletMarginLoss(margin=2.0, p=1.0)
    >>> anchor   = lucid.tensor([[0.0, 0.0]])
    >>> positive = lucid.tensor([[0.2, 0.2]])
    >>> negative = lucid.tensor([[3.0, 3.0]])
    >>> loss = criterion(anchor, positive, negative)
    """

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        size_average: _Flag = None,
        reduce: _Flag = None,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialise the TripletMarginLoss module. See the class docstring for parameter semantics."""
        super().__init__(size_average, reduce, reduction)
        _check_triplet_margin(margin, "TripletMarginLoss")
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    @override
    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        anchor : Tensor
            Input tensor.
        positive : Tensor
            Input tensor.
        negative : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return triplet_margin_loss(
            anchor,
            positive,
            negative,
            self.margin,
            self.p,
            self.eps,
            self.swap,
            reduction=self.reduction,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"margin={self.margin}, p={self.p}, reduction={self.reduction!r}"


class CosineEmbeddingLoss(_Loss):
    r"""Cosine embedding loss for learning similarity/dissimilarity.

    Measures whether two inputs are similar or dissimilar using cosine
    similarity.  The label ``y`` must be ``+1`` (similar) or ``-1``
    (dissimilar):

    .. math::

        \ell(x_1, x_2, y) =
        \begin{cases}
            1 - \cos(x_1, x_2)
                & \text{if } y = +1 \\[4pt]
            \max\!\bigl(0,\; \cos(x_1, x_2) - \text{margin}\bigr)
                & \text{if } y = -1
        \end{cases}

    where :math:`\cos(x_1, x_2) = \dfrac{x_1 \cdot x_2}{\|x_1\| \|x_2\|}`.

    Parameters
    ----------
    margin : float, optional
        Minimum cosine similarity below which dissimilar pairs incur
        zero loss.  Must be in :math:`(-1, 1)`.  Default ``0.0``.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    margin : float
        The cosine similarity margin for dissimilar pairs.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **x1** : :math:`(N, D)`.
    - **x2** : :math:`(N, D)`.
    - **y**  : :math:`(N,)` with values in :math:`\{-1, +1\}`.
    - **Output** : scalar for ``'mean'`` / ``'sum'``;
      :math:`(N,)` for ``'none'``.

    Notes
    -----
    - Useful when the magnitude of embeddings is not informative — only
      the angle between them matters (e.g. sentence similarity, image
      pairing).
    - Setting ``margin > 0`` means dissimilar pairs are only penalised
      when their cosine similarity exceeds the margin.

    Examples
    --------
    Paired sentence embeddings:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.CosineEmbeddingLoss(margin=0.5)
    >>> x1 = lucid.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> x2 = lucid.tensor([[0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
    >>> y  = lucid.tensor([1.0, -1.0])
    >>> loss = criterion(x1, x2, y)

    Zero margin for dissimilar pairs:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.CosineEmbeddingLoss()
    >>> x1 = lucid.tensor([[1.0, 2.0]])
    >>> x2 = lucid.tensor([[2.0, 1.0]])
    >>> y  = lucid.tensor([1.0])
    >>> loss = criterion(x1, x2, y)
    """

    def __init__(
        self,
        margin: float = 0.0,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialise the CosineEmbeddingLoss module. See the class docstring for parameter semantics."""
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    @override
    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input1 : Tensor
            Input tensor.
        input2 : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return cosine_embedding_loss(
            input1, input2, target, margin=self.margin, reduction=self.reduction
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"margin={self.margin}, reduction={self.reduction!r}"


class MarginRankingLoss(_Loss):
    r"""Margin ranking loss for pairwise ranking problems.

    Given two inputs :math:`x_1`, :math:`x_2` and a binary label
    :math:`y \in \{-1, +1\}`, the loss enforces that the correctly ranked
    input exceeds the other by at least ``margin``:

    .. math::

        \mathcal{L}(x_1, x_2, y) = \max\!\bigl(0,\; -y\,(x_1 - x_2) + \text{margin}\bigr)

    When :math:`y = +1`, :math:`x_1` should be larger than :math:`x_2`
    (e.g. a more relevant document score).  When :math:`y = -1`,
    :math:`x_2` should be larger.

    Parameters
    ----------
    margin : float, optional
        Minimum required score gap between the two inputs.
        Default ``0.0``.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    margin : float
        The margin threshold.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **x1** : :math:`(N,)` or :math:`(N, D)`.
    - **x2** : :math:`(N,)` or :math:`(N, D)`.
    - **y**  : :math:`(N,)` with values in :math:`\{-1, +1\}`.
    - **Output** : scalar for ``'mean'`` / ``'sum'``;
      same shape as input for ``'none'``.

    Notes
    -----
    - Commonly used in learning-to-rank tasks such as information
      retrieval and recommendation systems.
    - A positive margin :math:`> 0` requires a minimum gap between
      scores; setting it to ``0`` only penalises inversions.

    Examples
    --------
    Ranking pairs with a margin of 0.3:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MarginRankingLoss(margin=0.3)
    >>> x1 = lucid.tensor([1.0,  0.5, 2.0])
    >>> x2 = lucid.tensor([0.5,  1.0, 1.5])
    >>> y  = lucid.tensor([1.0, -1.0, 1.0])
    >>> loss = criterion(x1, x2, y)

    Strict ranking (no margin):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MarginRankingLoss()
    >>> x1 = lucid.tensor([3.0, 1.0])
    >>> x2 = lucid.tensor([2.0, 2.0])
    >>> y  = lucid.tensor([1.0, 1.0])
    >>> loss = criterion(x1, x2, y)
    """

    def __init__(
        self,
        margin: float = 0.0,
        size_average: _Flag = None,
        reduce: _Flag = None,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialise the MarginRankingLoss module. See the class docstring for parameter semantics."""
        self.margin = margin
        super().__init__(size_average, reduce, reduction)

    @override
    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input1 : Tensor
            Input tensor.
        input2 : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return margin_ranking_loss(
            input1, input2, target, margin=self.margin, reduction=self.reduction
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"margin={self.margin}, reduction={self.reduction!r}"


class HingeEmbeddingLoss(_Loss):
    r"""Hinge embedding loss for binary similarity learning.

    Given an input ``x`` representing a scalar distance measure and a
    binary label ``y``, the per-element loss is:

    .. math::

        \ell(x, y) =
        \begin{cases}
            x
                & \text{if } y = +1 \\[4pt]
            \max\!\bigl(0,\; \text{margin} - x\bigr)
                & \text{if } y = -1
        \end{cases}

    With :math:`y = +1` the loss is the distance itself (the model should
    produce small distances for positive pairs).  With :math:`y = -1` the
    model is penalised only if the distance is smaller than ``margin``
    (dissimilar pairs should be pushed apart).

    Parameters
    ----------
    margin : float, optional
        Minimum required distance for dissimilar pairs.  Default ``1.0``.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    margin : float
        The minimum distance for dissimilar pairs.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **x** : :math:`(N,)` or :math:`(*)` — non-negative distance values.
    - **y** : same shape as ``x`` — labels in :math:`\{-1, +1\}`.
    - **Output** : scalar for ``'mean'`` / ``'sum'``;
      same shape as input for ``'none'``.

    Notes
    -----
    - This loss is a one-sided generalisation of the hinge loss used in
      SVMs, applied here to embedding distances rather than margin-based
      decision functions.
    - The input is typically a pre-computed distance (e.g. L2 norm between
      representations), not raw logits.

    Examples
    --------
    Embedding distances with margin of 1:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.HingeEmbeddingLoss(margin=1.0)
    >>> dists   = lucid.tensor([0.3, 1.5, 0.8, 2.1])
    >>> labels  = lucid.tensor([1.0, 1.0, -1.0, -1.0])
    >>> loss = criterion(dists, labels)

    Tighter margin to encourage more separation:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.HingeEmbeddingLoss(margin=2.0)
    >>> dists  = lucid.tensor([0.5, 1.0, 3.0])
    >>> labels = lucid.tensor([-1.0, -1.0, 1.0])
    >>> loss = criterion(dists, labels)
    """

    def __init__(
        self,
        margin: float = 1.0,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialise the HingeEmbeddingLoss module. See the class docstring for parameter semantics."""
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return hinge_embedding_loss(input, target, self.margin, reduction=self.reduction)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"margin={self.margin}, reduction={self.reduction!r}"


class PoissonNLLLoss(_Loss):
    r"""Poisson negative log-likelihood loss for count data.

    Models the negative log-likelihood under a Poisson observation model.
    Two modes are supported, controlled by ``log_input``:

    **log_input=True** — the input ``x`` is the natural logarithm of the
    Poisson rate :math:`\lambda` (i.e. :math:`x = \log\lambda`):

    .. math::

        \ell(x, y) = e^{x} - y \cdot x

    **log_input=False** — the input ``x`` is the Poisson rate
    :math:`\lambda` directly:

    .. math::

        \ell(x, y) = x - y \cdot \log(x + \varepsilon)

    where :math:`\varepsilon` is added for numerical stability.

    When ``full=True``, the Stirling approximation term
    :math:`y\log y - y + 0.5\log(2\pi y)` is added to approximate the
    full log-likelihood including the factorial term.

    Parameters
    ----------
    log_input : bool, optional
        If ``True``, the input is :math:`\log\lambda`.  Default ``True``.
    full : bool, optional
        If ``True``, adds the Stirling approximation term.  Default
        ``False``.
    eps : float, optional
        Small constant for numerical stability when ``log_input=False``.
        Default ``1e-8``.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    log_input : bool
        Whether the input is in log-space.
    full : bool
        Whether the full approximation is applied.
    eps : float
        Numerical stability constant.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **Input** ``x``   : :math:`(*)`.
    - **Target** ``y``  : :math:`(*)` — non-negative counts.
    - **Output**        : scalar for ``'mean'`` / ``'sum'``;
      :math:`(*)` for ``'none'``.

    Notes
    -----
    - Commonly used for count-valued outputs such as word counts in
      language modelling or photon counts in imaging.
    - The ``log_input=True`` variant is the standard form for neural
      network outputs because it guarantees :math:`\lambda > 0` and
      simplifies the gradient.

    Examples
    --------
    Log-input mode (standard neural network output):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.PoissonNLLLoss(log_input=True)
    >>> log_rates = lucid.tensor([1.5, 0.3, -0.5, 2.0])
    >>> counts    = lucid.tensor([4.0, 1.0,  0.0, 7.0])
    >>> loss = criterion(log_rates, counts)

    Direct rate mode with full approximation:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.PoissonNLLLoss(log_input=False, full=True)
    >>> rates  = lucid.tensor([4.5, 1.0, 0.5, 7.5])
    >>> counts = lucid.tensor([4.0, 1.0, 0.0, 7.0])
    >>> loss = criterion(rates, counts)
    """

    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        size_average: bool | None = None,
        eps: float = 1e-8,
        reduce: bool | None = None,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialise the PoissonNLLLoss module. See the class docstring for parameter semantics."""
        super().__init__(size_average, reduce, reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps

    @override
    def forward(self, log_input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        log_input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return poisson_nll_loss(
            log_input, target, self.log_input, self.full, eps=self.eps, reduction=self.reduction
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"log_input={self.log_input}, reduction={self.reduction!r}"


class GaussianNLLLoss(_Loss):
    r"""Gaussian negative log-likelihood loss for heteroscedastic regression.

    Models the negative log-likelihood of a Gaussian distribution whose
    **mean** and **variance** are both predicted by the network.  For
    predicted mean :math:`\mu` (``x``), target :math:`y`, and predicted
    variance :math:`\sigma^2` (``var``):

    .. math::

        \ell(\mu, y, \sigma^2) =
            \frac{1}{2}\left(\log \sigma^2
            + \frac{(y - \mu)^2}{\sigma^2}\right)

    When ``full=True``, the constant term :math:`\frac{1}{2}\log(2\pi)` is
    also included.  The variance is clamped below by ``eps`` to prevent
    division by zero.

    Parameters
    ----------
    full : bool, optional
        If ``True``, adds the :math:`\frac{1}{2}\log(2\pi)` constant.
        Default ``False``.
    eps : float, optional
        Minimum value for the variance.  Default ``1e-6``.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    full : bool
        Whether the full Gaussian constant is included.
    eps : float
        Variance lower bound.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **x** (mean)   : :math:`(N, *)` — predicted means.
    - **target**     : :math:`(N, *)` — observed values.
    - **var**        : :math:`(N, *)` (the input's shape), :math:`(N, *)`
      without its last dimension (one variance per sample, broadcast over
      that dimension), the input's shape with one dimension of size 1, or
      a float — predicted variances, non-negative.
    - **Output**     : scalar for ``'mean'`` / ``'sum'``;
      :math:`(N, *)` for ``'none'``.

    Notes
    -----
    - This loss is also called the *heteroscedastic regression loss*
      because the noise level is input-dependent (predicted by the model
      rather than fixed).
    - Minimising this loss simultaneously trains the model to produce
      accurate predictions (via the residual term) and well-calibrated
      uncertainty estimates (via the log-variance term).

    Examples
    --------
    Predicting mean and variance simultaneously:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.GaussianNLLLoss()
    >>> mean = lucid.tensor([1.5, 2.0, 0.5])
    >>> var  = lucid.tensor([0.5, 1.0, 0.2])
    >>> y    = lucid.tensor([1.0, 2.5, 0.3])
    >>> loss = criterion(mean, y, var)

    With full Gaussian constant:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.GaussianNLLLoss(full=True, eps=1e-4)
    >>> mean = lucid.tensor([[0.0, 1.0], [2.0, 3.0]])
    >>> var  = lucid.tensor([[0.1, 0.2], [0.3, 0.4]])
    >>> y    = lucid.tensor([[0.1, 0.8], [2.2, 2.8]])
    >>> loss = criterion(mean, y, var)
    """

    def __init__(
        self, *, full: bool = False, eps: float = 1e-6, reduction: Reduction = "mean"
    ) -> None:
        """Initialise the GaussianNLLLoss module. See the class docstring for parameter semantics."""
        super().__init__()
        _check_reduction(reduction, "GaussianNLLLoss")
        self.full = full
        self.eps = eps
        self.reduction = reduction

    @override
    def forward(self, input: Tensor, target: Tensor, var: Tensor | float) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.
        var : Tensor or float
            Predicted variances; see the class docstring for the shapes
            accepted.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return gaussian_nll_loss(input, target, var, self.full, self.eps, self.reduction)


class CTCLoss(_Loss):
    r"""Connectionist Temporal Classification (CTC) loss.

    Computes the CTC loss for sequence-to-sequence learning where the
    alignment between the input and the target is unknown.  The loss
    marginalises over all valid monotonic alignments between the input
    sequence and the target sequence:

    .. math::

        \mathcal{L} = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)}
            \prod_{t=1}^{T} p\!\left(\pi_t \mid x_t\right)

    where :math:`\mathcal{B}` is the CTC collapsing function (removes
    blanks and repeated tokens) and the sum is over all valid paths
    :math:`\pi` that decode to the target sequence :math:`y`.

    Parameters
    ----------
    blank : int, optional
        Index of the blank label.  Default ``0``.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.
    zero_infinity : bool, optional
        If ``True``, infinite losses and their gradients are set to zero.
        Prevents instability for very long sequences or mismatched lengths.
        Default ``False``.

    Attributes
    ----------
    blank : int
        The blank label index.
    reduction : str
        The reduction mode.
    zero_infinity : bool
        Whether to zero out infinite-valued losses.

    Shape
    -----
    - **log_probs**      : :math:`(T, N, C)` — log-probabilities over the
      alphabet (including blank), typically output of ``LogSoftmax``.
      :math:`T` = input length, :math:`N` = batch size, :math:`C` = number
      of classes.
    - **targets**        : :math:`(N, S)` or :math:`(\sum S_n,)` — target
      sequences (without blank labels).
    - **input_lengths**  : :math:`(N,)` — length of each input sequence.
    - **target_lengths** : :math:`(N,)` — length of each target sequence.
    - **Output**         : scalar for ``'mean'`` / ``'sum'``;
      :math:`(N,)` for ``'none'``.

    Notes
    -----
    - CTC is widely used in automatic speech recognition (ASR) and optical
      character recognition (OCR) because it does not require aligned
      training data.
    - The input length for each sample must satisfy
      :math:`T_n \geq S_n` (input cannot be shorter than the target).
    - ``input_lengths`` and ``target_lengths`` must be integer tensors.

    Examples
    --------
    Single-sample sequence with 5 frames, 3 target characters, 6 classes:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.nn.functional as F
    >>> T, N, C = 5, 1, 6
    >>> criterion = nn.CTCLoss(blank=0)
    >>> log_probs = F.log_softmax(
    ...     lucid.zeros(T, N, C), dim=2
    ... )
    >>> targets        = lucid.tensor([[1, 2, 3]])
    >>> input_lengths  = lucid.tensor([T])
    >>> target_lengths = lucid.tensor([3])
    >>> loss = criterion(log_probs, targets, input_lengths, target_lengths)

    With ``zero_infinity=True`` for robustness:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="sum")
    >>> log_probs = lucid.zeros(10, 2, 8)
    >>> targets        = lucid.tensor([[1, 2, 3], [4, 5, 6]])
    >>> input_lengths  = lucid.tensor([10, 10])
    >>> target_lengths = lucid.tensor([ 3,  3])
    >>> loss = criterion(log_probs, targets, input_lengths, target_lengths)
    """

    def __init__(
        self, blank: int = 0, reduction: Reduction = "mean", zero_infinity: bool = False
    ) -> None:
        """Initialise the CTCLoss module. See the class docstring for parameter semantics."""
        super().__init__()
        _check_reduction(reduction, "CTCLoss")
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    @override
    def forward(
        self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor
    ) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        log_probs : Tensor
            Input tensor.
        targets : Tensor
            Input tensor.
        input_lengths : Tensor
            Input tensor.
        target_lengths : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            self.blank,
            self.reduction,
            self.zero_infinity,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"blank={self.blank}, reduction={self.reduction!r}"


class MultiMarginLoss(_WeightedLoss):
    r"""Multi-class hinge (SVM-style) margin loss.

    For each sample with predicted scores :math:`x \in \mathbb{R}^C`
    and correct class index :math:`y`, the per-sample loss sums hinge
    contributions from all incorrect classes:

    .. math::

        \mathcal{L}(x, y) = \frac{1}{C}
            \sum_{\substack{j=1 \\ j \neq y}}^{C}
            \max\!\bigl(0,\; \text{margin} - x[y] + x[j]\bigr)^p

    When ``weight`` is provided, each class contribution is scaled by
    ``weight[y]``.

    Parameters
    ----------
    p : int, optional
        Exponent of the hinge term; ``1`` (default) gives the standard
        hinge, ``2`` gives the squared hinge.
    margin : float, optional
        Minimum required score gap between the true class and all others.
        Default ``1.0``.
    weight : Tensor of shape (C,), optional
        Manual rescaling weight per class.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    p : int
        Hinge exponent.
    margin : float
        The score margin.
    weight : Tensor or None
        Per-class weight tensor.
        A registered buffer: moved by ``.to()`` and saved in
        ``state_dict``.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **Input** ``x``   : :math:`(N, C)` — class scores.
    - **Target** ``y``  : :math:`(N,)` — integer class indices in
      :math:`[0, C)`.
    - **Output**        : scalar for ``'mean'`` / ``'sum'``;
      :math:`(N,)` for ``'none'``.

    Notes
    -----
    - This is the multi-class generalisation of the SVM hinge loss.
    - With ``p=2`` the loss is the squared hinge, which penalises margin
      violations more aggressively.

    Examples
    --------
    Standard hinge loss over 3 classes:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MultiMarginLoss(margin=1.0)
    >>> scores  = lucid.tensor([[0.3, 2.1, 0.5], [1.8, 0.2, 1.0]])
    >>> targets = lucid.tensor([1, 0])
    >>> loss = criterion(scores, targets)

    Squared hinge with class weights:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MultiMarginLoss(
    ...     p=2,
    ...     weight=lucid.tensor([1.0, 2.0, 1.0]),
    ... )
    >>> scores  = lucid.tensor([[0.5, 1.5, 0.1]])
    >>> targets = lucid.tensor([1])
    >>> loss = criterion(scores, targets)
    """

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialise the MultiMarginLoss module. See the class docstring for parameter semantics."""
        super().__init__(weight, size_average, reduce, reduction)
        _check_multi_margin(p, weight, "MultiMarginLoss")
        self.p = p
        self.margin = margin

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return multi_margin_loss(
            input,
            target,
            p=self.p,
            margin=self.margin,
            weight=self.weight,
            reduction=self.reduction,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"p={self.p}, margin={self.margin}, reduction={self.reduction!r}"


class MultilabelMarginLoss(_Loss):
    r"""Multi-label ranking margin loss.

    Computes a pairwise ranking loss for multi-label classification where
    each sample can belong to multiple classes simultaneously.  Given a
    sample with predicted scores ``x`` of shape :math:`(C,)` and a target
    ``y`` listing the *ground-truth class indices* (padded with ``-1``),
    the loss penalises every pair where a non-target class is ranked above
    a target class:

    .. math::

        \mathcal{L}(x, y) =
            \frac{1}{n} \sum_{i \in \text{pos}} \sum_{j \notin \text{pos}}
            \max\!\bigl(0,\; 1 - x[y_i] + x[j]\bigr)

    where :math:`n` is the total number of classes :math:`C`.

    Parameters
    ----------
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    reduction : str
        The reduction mode.

    Shape
    -----
    - **Input** ``x``   : :math:`(N, C)` or :math:`(C,)`.
    - **Target** ``y``  : :math:`(N, C)` or :math:`(C,)` — integer class
      indices of positive labels; ``-1`` entries pad the list.
    - **Output**        : scalar for ``'mean'`` / ``'sum'``;
      :math:`(N,)` for ``'none'``.

    Notes
    -----
    - Unlike :class:`MultiLabelSoftMarginLoss`, this loss uses pairwise
      ranking comparisons and does not require probabilities.
    - The ``-1`` padding convention means: fill the target tensor with
      positive label indices from the left and pad the remaining entries
      with ``-1``.

    Examples
    --------
    One sample with 5 classes, two positive labels (0 and 3):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MultilabelMarginLoss()
    >>> scores = lucid.tensor([[0.1, 0.5, 0.3, 0.8, 0.2]])
    >>> target = lucid.tensor([[0, 3, -1, -1, -1]])
    >>> loss = criterion(scores, target)

    Batch of two samples:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MultilabelMarginLoss(reduction="sum")
    >>> scores = lucid.tensor([[1.0, 0.5, 0.2], [0.3, 0.9, 0.4]])
    >>> target = lucid.tensor([[0, -1, -1], [1, 2, -1]])
    >>> loss = criterion(scores, target)
    """

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return multilabel_margin_loss(input, target, reduction=self.reduction)


# CamelCase alias for parity with the reference framework's
# ``MultiLabelMarginLoss`` (capital ``L``).  Kept as a subclass rather
# than a simple ``= MultilabelMarginLoss`` so ``__name__`` and
# ``__repr__`` carry the canonical name.
class MultiLabelMarginLoss(MultilabelMarginLoss):
    r"""CamelCase alias for :class:`MultilabelMarginLoss`.

    Provided so that ``nn.MultiLabelMarginLoss`` (the capitalisation used
    by the reference framework) resolves to the same implementation.
    There is no behavioural difference — all parameters, attributes, and
    shapes are identical to :class:`MultilabelMarginLoss`.

    Parameters
    ----------
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Examples
    --------
    Interchangeable with :class:`MultilabelMarginLoss`:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MultiLabelMarginLoss()
    >>> scores = lucid.tensor([[0.5, 0.2, 0.9, 0.1]])
    >>> target = lucid.tensor([[2, 0, -1, -1]])
    >>> loss = criterion(scores, target)

    Verifying alias equivalence:

    >>> import lucid.nn as nn
    >>> assert nn.MultiLabelMarginLoss is nn.MultiLabelMarginLoss
    >>> c1 = nn.MultilabelMarginLoss()
    >>> c2 = nn.MultiLabelMarginLoss()
    >>> # Both produce identical results for the same inputs.

    Notes
    -----
    Identical loss formula to :class:`MultilabelMarginLoss`:

    .. math::

        \text{loss}(x, y) = \sum_{ij}
            \frac{\max\!\bigl(0,\; 1 - (x[y[j]] - x[i])\bigr)}{|x|}

    where the sum is taken over all pairs :math:`(i, j)` such that
    :math:`i \notin \{y[0], y[1], \ldots\}`.  Targets are padded with
    :math:`-1` to mark end-of-list; entries at or beyond the first
    :math:`-1` are ignored.
    """


# ── P4 fill: SoftMargin / MultiLabelSoftMargin / TripletMarginWithDistance ──


class SoftMarginLoss(_Loss):
    r"""Soft-margin binary hinge loss.

    Computes the element-wise two-class logistic loss between the input
    ``x`` and the binary label ``y \in \{-1, +1\}``:

    .. math::

        \ell(x, y) = \log\!\bigl(1 + e^{-y \cdot x}\bigr)

    This is the logistic (log-sigmoid) surrogate for the binary hinge.
    It is smooth and differentiable everywhere, unlike the hard hinge.

    Parameters
    ----------
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    reduction : str
        The reduction mode.

    Shape
    -----
    - **Input** ``x``   : :math:`(*)` — real-valued scores.
    - **Target** ``y``  : :math:`(*)` — binary labels in
      :math:`\{-1, +1\}`.
    - **Output**        : scalar for ``'mean'`` / ``'sum'``;
      :math:`(*)` for ``'none'``.

    Notes
    -----
    - As :math:`x \to +\infty` with :math:`y = +1` the loss approaches
      zero; as :math:`x \to -\infty` it grows linearly, which provides
      outlier robustness compared to cross-entropy.
    - This loss is essentially :class:`BCEWithLogitsLoss` re-labelled for
      :math:`\{-1, +1\}` targets.

    Examples
    --------
    Positive and negative labels:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.SoftMarginLoss()
    >>> x = lucid.tensor([ 1.5, -1.0,  0.5, -2.0])
    >>> y = lucid.tensor([ 1.0,  1.0, -1.0, -1.0])
    >>> loss = criterion(x, y)

    Element-wise output:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.SoftMarginLoss(reduction="none")
    >>> x = lucid.tensor([0.0, 2.0])
    >>> y = lucid.tensor([1.0, -1.0])
    >>> loss = criterion(x, y)  # shape (2,)
    """

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return soft_margin_loss(input, target, reduction=self.reduction)


class MultiLabelSoftMarginLoss(_WeightedLoss):
    r"""Multi-label soft-margin loss (BCE with logits averaged over classes).

    Treats each class as an independent binary classification problem and
    applies numerically stable binary cross-entropy with logits across all
    :math:`C` classes, then averages over the class dimension:

    .. math::

        \mathcal{L}(x, y) =
            -\frac{1}{C} \sum_{c=1}^{C}
            \Bigl[
                y_c \log \sigma(x_c)
                + (1 - y_c) \log(1 - \sigma(x_c))
            \Bigr]

    where :math:`\sigma` is the sigmoid function.  If ``weight`` is
    provided, each class term is multiplied by ``weight[c]``.

    Parameters
    ----------
    weight : Tensor of shape (C,), optional
        Manual rescaling weight per class.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    weight : Tensor or None
        Per-class weight tensor.
        A registered buffer: moved by ``.to()`` and saved in
        ``state_dict``.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **Input** ``x``   : :math:`(N, C)` — raw logits.
    - **Target** ``y``  : :math:`(N, C)` — binary labels in
      :math:`\{0, 1\}`.
    - **Output**        : scalar for ``'mean'`` / ``'sum'``;
      :math:`(N,)` for ``'none'``.

    Notes
    -----
    - This loss is appropriate when each sample can belong to multiple
      classes simultaneously (multi-label classification), as opposed to
      :class:`CrossEntropyLoss` which assumes mutually exclusive classes.
    - The input is expected to be raw logits; do not apply sigmoid before
      passing to this module.

    Examples
    --------
    Multi-label classification with 4 classes:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MultiLabelSoftMarginLoss()
    >>> logits = lucid.tensor([[1.5, -0.5, 0.3, 2.1],
    ...                        [0.2,  1.0, -1.5, 0.8]])
    >>> targets = lucid.tensor([[1.0, 0.0, 1.0, 1.0],
    ...                         [0.0, 1.0,  0.0, 0.0]])
    >>> loss = criterion(logits, targets)

    With class-frequency re-weighting:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.MultiLabelSoftMarginLoss(
    ...     weight=lucid.tensor([1.0, 2.0, 1.0, 0.5])
    ... )
    >>> logits  = lucid.tensor([[0.5, 1.0, -0.3, 0.8]])
    >>> targets = lucid.tensor([[1.0, 0.0,  1.0, 1.0]])
    >>> loss = criterion(logits, targets)
    """

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        input : Tensor
            Input tensor.
        target : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return multilabel_soft_margin_loss(
            input, target, weight=self.weight, reduction=self.reduction
        )


class TripletMarginWithDistanceLoss(_Loss):
    r"""Triplet margin loss with a user-supplied distance function.

    A generalisation of :class:`TripletMarginLoss` that replaces the
    fixed :math:`L_p` norm with an arbitrary differentiable distance
    function.  Given anchor :math:`a`, positive :math:`p`, and negative
    :math:`n` embeddings and a callable :math:`d`:

    .. math::

        \mathcal{L}(a, p, n) =
            \max\!\bigl(d(a, p) - d(a, n) + \text{margin},\; 0\bigr)

    When ``distance_function`` is ``None``, the default falls back to the
    Euclidean (L2) distance, making this equivalent to
    :class:`TripletMarginLoss` with :math:`p = 2`.

    The **swap** option: if ``True``, the loss also considers
    :math:`d(p, n)` as an alternative negative distance and replaces
    :math:`d(a, n)` with :math:`\max(d(a, n), d(p, n))`, exploiting the
    triangle inequality.

    Parameters
    ----------
    distance_function : callable, optional
        A function ``(Tensor, Tensor) -> Tensor`` that computes pairwise
        distances along the batch dimension.  Defaults to the L2
        (Euclidean) distance when ``None``.
    margin : float, optional
        Minimum required distance gap.  Default ``1.0``.
    swap : bool, optional
        If ``True``, uses the triangle-inequality-based swap.
        Default ``False``.
    reduction : str, optional
        ``'none'`` | ``'mean'`` (default) | ``'sum'``.

    Attributes
    ----------
    distance_function : callable
        The distance function in use.
    margin : float
        The margin threshold.
    swap : bool
        Whether the swap heuristic is active.
    reduction : str
        The reduction mode.

    Shape
    -----
    - **anchor**   : :math:`(N, D)`.
    - **positive** : :math:`(N, D)`.
    - **negative** : :math:`(N, D)`.
    - **Output**   : scalar for ``'mean'`` / ``'sum'``;
      :math:`(N,)` for ``'none'``.

    Notes
    -----
    - Custom distance functions allow cosine distance, Mahalanobis
      distance, or any learned distance to be used as the training
      objective without changing the loss formulation.
    - The ``distance_function`` must return a 1-D tensor of shape
      :math:`(N,)` — one scalar distance per sample in the batch.

    Examples
    --------
    Default L2 distance (equivalent to :class:`TripletMarginLoss`):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> criterion = nn.TripletMarginWithDistanceLoss(margin=1.0)
    >>> anchor   = lucid.tensor([[1.0, 0.0, 0.0]])
    >>> positive = lucid.tensor([[1.1, 0.0, 0.0]])
    >>> negative = lucid.tensor([[5.0, 0.0, 0.0]])
    >>> loss = criterion(anchor, positive, negative)

    Custom cosine distance:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.nn.functional as F
    >>> def cosine_dist(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
    ...     cos_sim = F.cosine_similarity(a, b, dim=1)
    ...     return 1.0 - cos_sim
    >>> criterion = nn.TripletMarginWithDistanceLoss(
    ...     distance_function=cosine_dist, margin=0.5
    ... )
    >>> anchor   = lucid.tensor([[1.0, 0.0]])
    >>> positive = lucid.tensor([[0.9, 0.1]])
    >>> negative = lucid.tensor([[0.0, 1.0]])
    >>> loss = criterion(anchor, positive, negative)
    """

    def __init__(
        self,
        *,
        distance_function: Callable[[Tensor, Tensor], Tensor] | None = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: Reduction = "mean",
    ) -> None:
        """Initialise the TripletMarginWithDistanceLoss module. See the class docstring for parameter semantics."""
        super().__init__(reduction=reduction)
        _check_triplet_margin(margin, "TripletMarginWithDistanceLoss")
        self.distance_function: Callable[[Tensor, Tensor], Tensor] = _default_distance(
            distance_function
        )
        self.margin = margin
        self.swap = swap

    @override
    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        # Delegate to the functional implementation so the F. and nn.
        # surfaces stay byte-equivalent.
        """Compute the loss between predictions and targets.

        Parameters
        ----------
        anchor : Tensor
            Input tensor.
        positive : Tensor
            Input tensor.
        negative : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Scalar loss (or unreduced tensor depending on ``reduction``).
        """
        return triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            distance_function=self.distance_function,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"margin={self.margin}, swap={self.swap}, reduction={self.reduction!r}"
