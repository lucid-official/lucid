"""
Adaptive softmax (Grave et al., 2016).
"""

from collections.abc import Sequence
from typing import NamedTuple, override

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.functional.activations import log_softmax
from lucid.nn.module import Module
from lucid.nn.modules.container import ModuleList, Sequential
from lucid.nn.modules.linear import Linear


class _ASMoutput(NamedTuple):
    """What :class:`AdaptiveLogSoftmaxWithLoss` returns: per-example target
    log-probabilities and their negative mean."""

    output: Tensor
    loss: Tensor


def _check_cutoffs(cutoffs: Sequence[int], n_classes: int) -> list[int]:
    """``cutoffs`` as a list, or ``ValueError`` if they cannot partition ``n_classes``."""
    cuts = list(cutoffs)
    if len(cuts) == 0:
        raise ValueError("cutoffs should be a sequence of length larger than 0")
    if (
        cuts != sorted(cuts)
        or min(cuts) <= 0
        or max(cuts) > n_classes - 1
        or len(set(cuts)) != len(cuts)
        or any(int(c) != c for c in cuts)
    ):
        raise ValueError(
            "cutoffs should be a sequence of unique, positive integers sorted in an "
            "increasing order, where each value is between 1 and n_classes-1"
        )
    return cuts


class AdaptiveLogSoftmaxWithLoss(Module):
    r"""Efficient softmax approximation for large output spaces (Grave et al., 2016).

    Adaptive softmax suits a label distribution that is highly imbalanced,
    such as the word frequencies of a language model, which follow Zipf's
    law.  The classes are partitioned into a *head* of frequent classes and
    several *tail clusters* of rarer ones.  The head softmax scores the
    frequent classes directly plus one extra logit per cluster; a cluster's
    own, lower-dimensional softmax is evaluated only for the examples whose
    target lies in it.  The log-probability of a tail class is

    .. math::

        \log p(c) = \log p_{\text{head}}(\text{cluster}(c))
                    + \log p_{\text{cluster}}(c \mid \text{cluster}(c)).

    Parameters
    ----------
    in_features : int
        Number of features of the input.
    n_classes : int
        Number of classes.
    cutoffs : sequence of int
        Strictly increasing class indices in ``[1, n_classes - 1]`` that
        split the classes into the head and the clusters: with
        ``cutoffs=[10, 100, 1000]``, classes ``0..9`` are the head,
        ``10..99`` and ``100..999`` are the first two clusters and
        ``1000..n_classes-1`` the last.
    div_value : float, optional
        Cluster ``i`` (from 1) projects the input to
        ``floor(in_features / div_value ** i)`` features before its
        classifier.  Default: ``4.0``.
    head_bias : bool, optional
        Add a bias to the head classifier.  Default: ``False``.
    device : DeviceLike, optional
        Device for the parameters.
    dtype : DTypeLike, optional
        Dtype for the parameters.

    Attributes
    ----------
    head : Linear
        The head classifier, ``in_features -> shortlist_size + n_clusters``.
    tail : ModuleList
        One ``Sequential(Linear, Linear)`` projection per cluster.
    cutoffs : list of int
        The constructor's cutoffs followed by ``n_classes``.

    Shape
    -----
    * **input**: ``(N, in_features)`` or ``(in_features)``.
    * **target**: ``(N)`` or ``()``, integers in ``[0, n_classes - 1]``,
      classes sorted by frequency (the most frequent is ``0``).
    * **output**: ``(N)`` or ``()`` — each example's target log-probability.
    * **loss**: scalar — the mean negative log-likelihood.

    Notes
    -----
    ``forward`` returns a ``NamedTuple`` ``(output, loss)``.  Use
    :meth:`log_prob` for the log-probabilities of every class and
    :meth:`predict` for the most probable class.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> asm = nn.AdaptiveLogSoftmaxWithLoss(16, 100, cutoffs=[10, 40])
    >>> x = lucid.randn(5, 16)
    >>> y = lucid.tensor([0, 7, 12, 55, 99], dtype=lucid.int64)
    >>> out = asm(x, y)
    >>> out.output.shape, out.loss.shape
    ((5,), ())
    >>> asm.log_prob(x).shape
    (5, 100)
    >>> asm.predict(x).shape
    (5,)
    """

    in_features: int
    n_classes: int
    cutoffs: list[int]
    div_value: float
    head_bias: bool
    head: Linear
    tail: ModuleList

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        div_value: float = 4.0,
        head_bias: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the module. See the class docstring for parameter semantics."""
        super().__init__()
        cuts = _check_cutoffs(cutoffs, n_classes)

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cuts + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.shortlist_size: int = self.cutoffs[0]
        self.n_clusters: int = len(self.cutoffs) - 1
        self.head_size: int = self.shortlist_size + self.n_clusters

        self.head = Linear(in_features, self.head_size, bias=head_bias, device=device, dtype=dtype)
        self.tail = ModuleList()
        for i in range(self.n_clusters):
            hidden = int(in_features // (div_value ** (i + 1)))
            size = self.cutoffs[i + 1] - self.cutoffs[i]
            self.tail.append(
                Sequential(
                    Linear(in_features, hidden, bias=False, device=device, dtype=dtype),
                    Linear(hidden, size, bias=False, device=device, dtype=dtype),
                )
            )

    def reset_parameters(self) -> None:
        """Re-draw every parameter the way the constructor did."""
        for layer in self.modules():
            if isinstance(layer, Linear):
                layer.reset_parameters()

    @override
    def forward(self, input_: Tensor, target_: Tensor) -> _ASMoutput:
        """Score ``target_`` under the adaptive softmax.

        Parameters
        ----------
        input_ : Tensor
            Features, ``(N, in_features)`` or ``(in_features)``.
        target_ : Tensor
            Class indices, ``(N)`` or ``()`` to match ``input_``.

        Returns
        -------
        output : Tensor
            Each example's target log-probability, ``(N)`` or ``()``.
        loss : Tensor
            ``(-output).mean()``.

        Raises
        ------
        ValueError
            If ``input_`` and ``target_`` disagree in rank or batch size, or
            ``target_`` has more than one dimension.
        IndexError
            If a target lies outside ``[0, n_classes - 1]``.
        """
        batched = self._check_batch(input_, target_)
        x = input_ if batched else input_.unsqueeze(0)
        y = target_ if batched else target_.unsqueeze(0)
        batch = int(y.shape[0])

        output = _lucid.zeros(batch, dtype=x.dtype, device=x.device)
        # The head column each example scores: its own class in the shortlist,
        # else the logit of the cluster that holds it.
        head_column = y
        covered = 0
        bounds = [0] + self.cutoffs
        for i in range(len(bounds) - 1):
            low, high = bounds[i], bounds[i + 1]
            in_bucket = (y >= low) & (y < high)
            rows = _lucid.nonzero(in_bucket).reshape(-1)
            if rows.numel() == 0:
                continue
            covered += int(rows.numel())
            if i == 0:
                continue
            cluster = self.shortlist_size + i - 1
            head_column = _lucid.where(in_bucket, cluster, head_column)
            relative = y.index_select(0, rows) - low
            scores = log_softmax(self.tail[i - 1](x.index_select(0, rows)), dim=1)
            local = scores.gather(1, relative.unsqueeze(1)).reshape(-1)
            output = _lucid.index_add(output, 0, rows, local)

        if covered != batch:
            raise _C_engine.IndexError(
                f"Target values should be in [0, {self.n_classes - 1}], but values in range "
                f"[{y.min().item()}, {y.max().item()}] were found."
            )

        head_logprob = log_softmax(self.head(x), dim=1)
        output = output + head_logprob.gather(1, head_column.unsqueeze(1)).reshape(-1)
        loss = (-output).mean()
        return _ASMoutput(output if batched else output.squeeze(0), loss)

    def _check_batch(self, input: Tensor, target: Tensor) -> bool:
        """Whether the call is batched; ``ValueError`` if the operands disagree."""
        if target.ndim == 1:
            if input.shape[0] != target.shape[0]:
                raise ValueError(
                    "Input and target should have the same size in the batch dimension."
                )
            if input.ndim != 2:
                raise ValueError(
                    f"1D target tensor expects 2D input tensors, but found inputs with "
                    f"size {tuple(input.shape)}"
                )
            return True
        if target.ndim == 0:
            if input.ndim != 1:
                raise ValueError(
                    f"0D target tensor expects 1D input tensors, but found inputs with "
                    f"size {tuple(input.shape)}"
                )
            return False
        raise ValueError("0D or 1D target tensor expected, multi-target not supported")

    def _full_log_prob(self, input: Tensor, head_output: Tensor) -> Tensor:
        """The log-probability of every class, given ``head`` applied to ``input``."""
        head_logprob = log_softmax(head_output, dim=1)
        parts = [head_logprob[:, : self.shortlist_size]]
        for i in range(self.n_clusters):
            cluster_logprob = log_softmax(self.tail[i](input), dim=1)
            parts.append(cluster_logprob + head_logprob[:, self.shortlist_size + i].unsqueeze(1))
        return _lucid.cat(parts, 1)

    def log_prob(self, input: Tensor) -> Tensor:
        r"""Log-probabilities of all :attr:`n_classes` classes.

        Parameters
        ----------
        input : Tensor
            A minibatch, ``(N, in_features)``.

        Returns
        -------
        Tensor
            ``(N, n_classes)``; each row's exponential sums to one.
        """
        return self._full_log_prob(input, self.head(input))

    def predict(self, input: Tensor) -> Tensor:
        r"""The most probable class of each example.

        Equal to ``log_prob(input).argmax(dim=1)`` but skips the clusters
        when every example's best head column is a shortlist class.

        Parameters
        ----------
        input : Tensor
            A minibatch, ``(N, in_features)``.

        Returns
        -------
        Tensor
            ``(N)`` integer class indices.
        """
        head_output = self.head(input)
        best = head_output.argmax(dim=1)
        tail_rows = _lucid.nonzero(best >= self.shortlist_size).reshape(-1)
        if tail_rows.numel() == 0:
            return best
        if tail_rows.numel() == best.shape[0]:
            return self._full_log_prob(input, head_output).argmax(dim=1)
        full = self._full_log_prob(
            input.index_select(0, tail_rows), head_output.index_select(0, tail_rows)
        )
        return _lucid.index_copy(best, 0, tail_rows, full.argmax(dim=1))

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"in_features={self.in_features}, n_classes={self.n_classes}, "
            f"cutoffs={self.cutoffs[:-1]}, div_value={self.div_value}, "
            f"head_bias={self.head_bias}"
        )
