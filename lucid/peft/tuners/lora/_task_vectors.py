r"""The task-vector merges behind ``add_weighted_adapter``'s TIES / DARE / magnitude-prune types.

A merge of several same-shaped tensors ("task vectors") is a pipeline of
three interchangeable parts, each a small registered class:

* a :class:`_Pruner` thins every tensor first (keep the largest by
  magnitude, or keep a random share and rescale as DARE does);
* a :class:`_SignElection` decides, per element, which sign the merged
  value follows (TIES: the sign of the sum, or the more frequent sign), and
  elements of the other sign are dropped;
* the weighted sum itself, averaged over the elements that survived the
  election when there was one (the "disjoint merge").

:class:`_TaskMerge` runs the pipeline; a combination type of
``lucid.peft.tuners.lora._merging`` is a choice of parts.  A part is added
by one subclass naming itself in its class statement.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, override

import lucid
from lucid._tensor.tensor import Tensor

#: How the sign of a merged element is elected — the reference's names.
_SignMethod = Literal["total", "frequency"]


class _Methods[Part]:
    """The parts of one kind by name; each part is registered by a class decorator."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._by_name: dict[str, type[Part]] = {}

    def register(self, name: str) -> Callable[[type[Part]], type[Part]]:
        """A class decorator registering the class it decorates as ``name``."""

        def add(part: type[Part]) -> type[Part]:
            self._by_name[name] = part
            return part

        return add

    def get(self, name: str) -> type[Part]:
        """The part registered as ``name``; ``ValueError`` listing the known names if there is none."""
        found = self._by_name.get(name)
        if found is None:
            raise ValueError(f"Unknown {self._kind} {name!r}; one of {sorted(self._by_name)}")
        return found


class _Pruner(ABC):
    """Thins a task vector, keeping ``density`` of its elements; registered by name.

    Parameters
    ----------
    density : float
        The share of elements to keep, in ``[0, 1]``; a density of 1 or more
        keeps everything (with a warning).
    """

    def __init__(self, density: float) -> None:
        if density < 0:
            raise ValueError(f"Density should be >= 0, got {density}")
        self.density = density

    @classmethod
    def named(cls, name: str, density: float) -> _Pruner:
        """The pruner registered under ``name``, keeping ``density``."""
        return _PRUNERS.get(name)(density)

    def __call__(self, tensor: Tensor) -> Tensor:
        if self.density >= 1:
            warnings.warn(
                f"The density {self.density} is greater than or equal to 1, no pruning will be "
                "performed.",
                stacklevel=2,
            )
            return tensor
        return self.prune(tensor)

    @abstractmethod
    def prune(self, tensor: Tensor) -> Tensor:
        """``tensor`` with all but ``density`` of its elements zeroed."""


_PRUNERS: Final = _Methods[_Pruner]("pruning method")


@_PRUNERS.register("magnitude")
class _MagnitudePruner(_Pruner):
    """Keep the ``density * numel`` elements of largest absolute value."""

    @override
    def prune(self, tensor: Tensor) -> Tensor:
        flat = tensor.abs().reshape(-1)
        keep = int(self.density * flat.numel())
        if keep == 0:
            return lucid.zeros_like(tensor)
        chosen = lucid.topk(flat, k=keep).indices
        mask = lucid.zeros_like(flat).scatter(0, chosen, lucid.ones_like(chosen, dtype=flat.dtype))
        return tensor * mask.reshape(tensor.shape)


@_PRUNERS.register("random")
class _RandomPruner(_Pruner):
    r"""Keep each element with probability ``density``, rescaled by :math:`1/\mathrm{density}`.

    The rescale keeps the expected value of the tensor (DARE).  The draws
    come from Lucid's global generator, so ``lucid.manual_seed`` makes them
    repeat.
    """

    def __init__(self, density: float) -> None:
        super().__init__(density)
        if density == 0:
            raise ValueError("Density should be > 0 for random pruning: the rescale divides by it.")

    @override
    def prune(self, tensor: Tensor) -> Tensor:
        mask = lucid.bernoulli(lucid.full_like(tensor, self.density))
        return tensor * mask / self.density


class _SignElection(ABC):
    """Decides which elements of a stack of task vectors agree with the elected sign; registered by name."""

    @classmethod
    def named(cls, name: str) -> _SignElection:
        """The election registered under ``name``."""
        return _ELECTIONS.get(name)()

    def mask(self, stacked: Tensor) -> Tensor:
        """``True`` where a vector's element has the elected sign; ``stacked`` is vectors along dim 0."""
        majority = lucid.where(self.magnitude(stacked) >= 0, 1, -1)
        return stacked.sign() == majority

    @abstractmethod
    def magnitude(self, stacked: Tensor) -> Tensor:
        """The per-element quantity whose sign is the elected one."""


_ELECTIONS: Final = _Methods[_SignElection]("majority_sign_method")


@_ELECTIONS.register("total")
class _TotalElection(_SignElection):
    """The sign of the sum — the side with more total mass wins."""

    @override
    def magnitude(self, stacked: Tensor) -> Tensor:
        return stacked.sum(dim=0)


@_ELECTIONS.register("frequency")
class _FrequencyElection(_SignElection):
    """The sign that most vectors have."""

    @override
    def magnitude(self, stacked: Tensor) -> Tensor:
        return stacked.sign().sum(dim=0)


@dataclass(frozen=True, slots=True)
class _Recipe:
    """Which parts a combination type is made of: a pruning method, and whether signs are elected."""

    pruning: str | None = None
    elects_sign: bool = False


@dataclass(frozen=True, slots=True)
class _TaskMerge:
    """The merge pipeline for one set of options: prune, elect a sign, weight and sum.

    ``pruner`` and ``election`` are ``None`` when the recipe has no such
    step; with neither it is a plain weighted sum (task arithmetic).
    """

    pruner: _Pruner | None = None
    election: _SignElection | None = None

    @classmethod
    def make(
        cls, recipe: _Recipe, density: float | None, majority_sign_method: _SignMethod
    ) -> _TaskMerge:
        """The pipeline ``recipe`` describes, at ``density`` and electing by ``majority_sign_method``.

        Raises
        ------
        ValueError
            The recipe prunes and ``density`` is ``None``.
        """
        pruner = None
        if recipe.pruning is not None:
            if density is None:
                raise ValueError(
                    "`density` must be given for the ties, dare and magnitude_prune "
                    "combination types."
                )
            pruner = _Pruner.named(recipe.pruning, density)
        election = _SignElection.named(majority_sign_method) if recipe.elects_sign else None
        return cls(pruner, election)

    def __call__(self, tensors: Sequence[Tensor], weights: Sequence[float]) -> Tensor:
        """Merge ``tensors`` (one per weight, all one shape) into one."""
        pruner = self.pruner
        stacked = lucid.stack([tensor if pruner is None else pruner(tensor) for tensor in tensors])
        scale = lucid.tensor(list(weights), dtype=stacked.dtype, device=stacked.device)
        weighted = stacked * scale.reshape((len(weights),) + (1,) * (stacked.ndim - 1))
        if self.election is None:
            return weighted.sum(dim=0)
        # Elected on the pruned vectors, before the weights: a negative weight must not flip it.
        agreeing = self.election.mask(stacked).to(stacked.dtype)
        return (weighted * agreeing).sum(dim=0) / agreeing.sum(dim=0).clamp(min=1.0)
