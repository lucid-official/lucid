"""How ``add_weighted_adapter`` folds several LoRA adapters into one.

A combination type is a class: it states the rank of the adapter it makes
from the ranks of the adapters it is given, and, layer by layer, the
``lora_A`` and ``lora_B`` weights of the new adapter.  Each registers itself
under its name as it is defined (``combination_type=`` in the class
statement), so ``LoraModel.add_weighted_adapter`` looks the name up and
runs the class, and a new type is one subclass.

The new adapter's ``lora_alpha`` is its rank, so its scale is exactly 1:
every weight and every source adapter's own scale is already inside the
tensors a strategy returns.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, override

import lucid
import lucid.linalg as linalg
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners.lora._layer import LoraLayer

# The strategies are reached through ``_Combination.named``, by the name each
# registers under — which is also why they are listed here.
__all__ = ["_Cat", "_Combination", "_CombinationType", "_Linear", "_MergeOptions", "_Svd"]

#: The names ``add_weighted_adapter`` takes — the reference's, built or not.
_CombinationType = Literal[
    "svd",
    "linear",
    "cat",
    "ties",
    "ties_svd",
    "dare_ties",
    "dare_linear",
    "dare_ties_svd",
    "dare_linear_svd",
    "magnitude_prune",
    "magnitude_prune_svd",
]

#: Combination types the reference offers that need the task-vector
#: machinery (trimming, sign election, random drops) lucid.peft does not have.
_LATER: Final = frozenset(
    {
        "ties",
        "ties_svd",
        "dare_ties",
        "dare_linear",
        "dare_ties_svd",
        "dare_linear_svd",
        "magnitude_prune",
        "magnitude_prune_svd",
    }
)


@dataclass(frozen=True, slots=True)
class _MergeOptions:
    """The rank of the new adapter and the knobs of ``add_weighted_adapter``.

    Each strategy reads the ones it uses.
    """

    rank: int
    svd_clamp: float | None
    svd_full_matrices: bool
    svd_driver: str | None


def _pad(weight: Tensor, size: int, dim: int) -> Tensor:
    """``weight`` with zeros appended along ``dim`` up to ``size``."""
    short = size - weight.shape[dim]
    if short <= 0:
        return weight
    shape = list(weight.shape)
    shape[dim] = short
    zeros = lucid.zeros(tuple(shape), dtype=weight.dtype, device=weight.device)
    return lucid.cat([weight, zeros], dim=dim)


def _weighted_sum(tensors: Sequence[Tensor], coefficients: Sequence[float]) -> Tensor:
    total = tensors[0] * coefficients[0]
    for tensor, coefficient in zip(tensors[1:], coefficients[1:], strict=True):
        total = total + tensor * coefficient
    return total


class _Combination(ABC):
    """One way to fold several adapters into a new one, registered by name.

    A subclass names itself with ``combination_type=`` and supplies the
    new rank (``new_rank``) and the weights for one layer (``combine``).
    """

    _strategies: ClassVar[dict[str, type[_Combination]]] = {}

    @override
    def __init_subclass__(cls, *, combination_type: str | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if combination_type is not None:
            cls._strategies[combination_type] = cls

    @classmethod
    def named(cls, combination_type: str) -> type[_Combination]:
        """The strategy registered under ``combination_type``."""
        found = cls._strategies.get(combination_type)
        if found is not None:
            return found
        if combination_type in _LATER:
            raise NotImplementedError(
                f"combination_type={combination_type!r} (the TIES / DARE / magnitude-prune "
                f"family) is not implemented in lucid.peft yet; use one of "
                f"{', '.join(map(repr, cls._strategies))}"
            )
        raise ValueError(f"Invalid combination_type: {combination_type}")

    def __init__(self, options: _MergeOptions) -> None:
        self.options = options

    @classmethod
    @abstractmethod
    def new_rank(cls, ranks: Sequence[int], svd_rank: int | None) -> int:
        """The rank of the adapter made from adapters of ``ranks``."""

    @abstractmethod
    def combine(
        self, layer: LoraLayer, adapters: Sequence[str], weights: Sequence[float]
    ) -> tuple[Tensor, Tensor]:
        """``(lora_A, lora_B)`` folding ``adapters`` of ``layer``; at most the new rank wide."""

    def weights_for(
        self, layer: LoraLayer, adapters: Sequence[str], weights: Sequence[float]
    ) -> tuple[Tensor, Tensor]:
        """``combine`` for the adapters ``layer`` holds, zero-padded to exactly the new rank.

        Raises
        ------
        ValueError
            ``layer`` holds none of ``adapters``.
        """
        held = [(n, w) for n, w in zip(adapters, weights, strict=True) if n in layer.lora_A]
        if not held:
            raise ValueError(f"None of the adapters {list(adapters)} is held by this layer.")
        lora_a, lora_b = self.combine(layer, [n for n, _ in held], [w for _, w in held])
        return _pad(lora_a, self.options.rank, 0), _pad(lora_b, self.options.rank, 1)


class _Linear(_Combination, combination_type="linear"):
    r"""Weight ``A`` and ``B`` separately: sum :math:`\sqrt{|w_i s_i|}` times each.

    :math:`A' = \sum_i \mathrm{sgn}(w_i s_i)\sqrt{|w_i s_i|}\,A_i` and
    :math:`B' = \sum_i \sqrt{|w_i s_i|}\,B_i`.  Cheap, and exact only for
    one adapter: the product :math:`B'A'` also holds the cross terms
    :math:`B_i A_j`.  The adapters must share a rank.
    """

    @classmethod
    @override
    def new_rank(cls, ranks: Sequence[int], svd_rank: int | None) -> int:
        del svd_rank  # the rank is the sources' own
        if len(set(ranks)) != 1:
            raise ValueError(
                "All adapters must have the same r value when using combination_type 'linear'."
            )
        return ranks[0]

    @override
    def combine(
        self, layer: LoraLayer, adapters: Sequence[str], weights: Sequence[float]
    ) -> tuple[Tensor, Tensor]:
        scaling = layer.scaling
        scaled = [w * scaling[name] for name, w in zip(adapters, weights, strict=True)]
        roots = [math.sqrt(abs(s)) for s in scaled]
        # The sign goes on one side only, or the two would cancel in B'A'.
        signed = [root if s >= 0 else -root for s, root in zip(scaled, roots, strict=True)]
        lora_a = _weighted_sum([layer.lora_A[name].weight for name in adapters], signed)
        lora_b = _weighted_sum([layer.lora_B[name].weight for name in adapters], roots)
        return lora_a, lora_b


class _Cat(_Combination, combination_type="cat"):
    r"""Stack the ranks: :math:`A'` row-wise from :math:`w_i s_i A_i`, :math:`B'` column-wise.

    Exact — :math:`B'A' = \sum_i w_i s_i B_i A_i` — at the price of a rank
    that is the sum of the sources'.
    """

    @classmethod
    @override
    def new_rank(cls, ranks: Sequence[int], svd_rank: int | None) -> int:
        del svd_rank  # the rank is the sum of the sources'
        return sum(ranks)

    @override
    def combine(
        self, layer: LoraLayer, adapters: Sequence[str], weights: Sequence[float]
    ) -> tuple[Tensor, Tensor]:
        scaling = layer.scaling
        lora_a = [
            layer.lora_A[name].weight * (w * scaling[name])
            for name, w in zip(adapters, weights, strict=True)
        ]
        lora_b: list[Tensor] = [layer.lora_B[name].weight for name in adapters]
        return lucid.cat(lora_a, dim=0), lucid.cat(lora_b, dim=1)


class _Svd(_Combination, combination_type="svd"):
    r"""The best rank-:math:`r` approximation of :math:`\sum_i w_i\,\Delta W_i`.

    Sums the sources' updates, :math:`\Delta W_i = s_i B_i A_i`, factors
    the sum as :math:`U \Sigma V^\top` and keeps the :math:`r` largest
    singular values: :math:`B' = U_r \Sigma_r`, :math:`A' = V_r^\top`.  The
    reconstruction error is exactly the discarded singular values
    (Eckart-Young).
    """

    @classmethod
    @override
    def new_rank(cls, ranks: Sequence[int], svd_rank: int | None) -> int:
        return svd_rank or max(ranks)

    @override
    def combine(
        self, layer: LoraLayer, adapters: Sequence[str], weights: Sequence[float]
    ) -> tuple[Tensor, Tensor]:
        options = self.options
        delta = _weighted_sum([layer.get_delta_weight(name) for name in adapters], weights)
        u, s, vh = linalg.svd(delta, options.svd_full_matrices, driver=options.svd_driver)
        # There are only min(out, in) singular values; a larger rank is zero-padded by the caller.
        keep = min(options.rank, s.shape[0])
        left = u[:, :keep] * s[:keep]
        right = vh[:keep, :]
        if options.svd_clamp is not None:
            pooled = lucid.cat([left.flatten(), right.flatten()])
            bound = float(lucid.quantile(pooled, options.svd_clamp).item())
            left, right = left.clamp(-bound, bound), right.clamp(-bound, bound)
        return right, left
