r"""Pruning utilities — the reference framework's ``nn.utils.prune``.

Pruning reparametrises a parameter ``W`` as ``W = W_orig * mask``: the original
:class:`~lucid.nn.parameter.Parameter` stays, as the *same object*, under
``<name>_orig`` (so an optimizer built before pruning keeps updating it and a
frozen weight stays frozen), the mask becomes the buffer ``<name>_mask``, and a
forward pre-hook recomputes ``<name>`` before every call.  The pre-hook **is** the
pruning-method object — a plain instance, so a pruned module pickles.

The methods form a hierarchy rooted at :class:`BasePruningMethod`: a subclass
defines :meth:`~BasePruningMethod.compute_mask` and declares its
``PRUNING_TYPE``.  Pruning a parameter that is already pruned wraps the methods
in a :class:`PruningContainer`, which applies the newest one to what the earlier
masks left alive; the free functions (:func:`l1_unstructured`,
:func:`ln_structured`, ...) are thin entry points onto the classes.
"""

import math
import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import ClassVar, Literal, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid.nn.utils._checks import _require_module, _require_parameter
from lucid.nn.utils._reparam import _ReparamHook
from lucid.nn.utils.convert_parameters import parameters_to_vector

__all__ = [
    "BasePruningMethod",
    "PruningContainer",
    "Identity",
    "RandomUnstructured",
    "L1Unstructured",
    "RandomStructured",
    "LnStructured",
    "CustomFromMask",
    "identity",
    "random_unstructured",
    "l1_unstructured",
    "random_structured",
    "ln_structured",
    "global_unstructured",
    "custom_from_mask",
    "remove",
    "is_pruned",
]

type _Amount = int | float
type _Norm = int | float | Literal["fro", "nuc"]


# ── argument validation (the boundary of every method) ──────────────────


def _validate_pruning_amount_init(amount: _Amount) -> None:
    """Refuse an ``amount`` that no tensor could make sense of.

    A float is a fraction in ``[0, 1]``; an int is an absolute count, so
    non-negative.  Whether the count fits the tensor is known only at prune
    time (:func:`_validate_pruning_amount`).
    """
    if isinstance(amount, bool) or not isinstance(amount, numbers.Real):
        raise TypeError(f"Invalid type for amount: {amount!r}. Must be int or float.")
    ceiling = math.inf if isinstance(amount, numbers.Integral) else 1.0
    if not 0 <= amount <= ceiling:
        raise ValueError(f"amount={amount} must be a non-negative integer or a float in [0, 1]")


def _validate_pruning_amount(amount: _Amount, tensor_size: int) -> None:
    """An absolute ``amount`` cannot exceed the ``tensor_size`` it prunes from."""
    if isinstance(amount, numbers.Integral) and amount > tensor_size:
        raise ValueError(
            f"amount={amount} should be smaller than the number of parameters to prune="
            f"{tensor_size}"
        )


def _validate_structured_pruning(t: Tensor) -> None:
    """Channels exist only on a tensor of at least two dimensions."""
    if len(t.shape) <= 1:
        raise ValueError(
            "Structured pruning can only be applied to multidimensional tensors. "
            f"Found tensor of shape {tuple(t.shape)} with {len(t.shape)} dims"
        )


def _validate_pruning_dim(t: Tensor, dim: int) -> None:
    """``dim`` must name an axis of ``t`` (counting from the end when negative)."""
    ndim: int = len(t.shape)
    if not -ndim <= dim < ndim:
        raise IndexError(f"Invalid index {dim} for tensor of size {tuple(t.shape)}")


def _compute_nparams_toprune(amount: _Amount, tensor_size: int) -> int:
    """The absolute number of units to prune: an int is itself, a float a rounded fraction."""
    if isinstance(amount, numbers.Integral):
        return int(amount)
    return round(amount * tensor_size)


# ── mask building blocks ────────────────────────────────────────────────


def _zero_flat(mask: Tensor, indices: Tensor) -> Tensor:
    """A copy of ``mask`` with the entries at flat positions ``indices`` set to zero."""
    flat: Tensor = mask.clone().reshape([-1])
    flat[indices] = 0
    return flat.reshape(list(mask.shape))


def _channel_mask(t: Tensor, dim: int, kept: Tensor) -> Tensor:
    """Ones on the channels of ``t`` along ``dim`` listed in ``kept``, zeros elsewhere."""
    n: int = int(t.shape[dim])
    keep: Tensor = lucid.zeros(n, dtype=t.dtype, device=t.device)
    keep[kept] = 1
    shape: list[int] = [1] * len(t.shape)
    shape[dim] = n
    return keep.reshape(shape) * lucid.ones_like(t)


def _top_indices(scores: Tensor, k: int, largest: bool) -> Tensor:
    """Indices of the ``k`` largest (or smallest) entries of a 1-D ``scores``; ``k=0`` is empty."""
    if k == 0:
        return lucid.zeros(0, dtype=lucid.int64, device=scores.device)
    return lucid.topk(scores, k, largest=largest).indices


def _compute_norm(t: Tensor, n: _Norm, dim: int) -> Tensor:
    """The ``L_n`` norm of ``t`` over every axis except ``dim``: one value per channel."""
    dims: list[int] = [d for d in range(len(t.shape)) if d != dim % len(t.shape)]
    if n == "nuc":
        if len(dims) != 2:
            raise ValueError(
                f"the 'nuc' norm needs a matrix per channel; got {len(dims)} axes besides dim"
            )
        return lucid.linalg.matrix_norm(t, ord="nuc", dim=(dims[0], dims[1]))
    if n == "fro":
        return lucid.linalg.vector_norm(t, ord=2, dim=dims)
    if not isinstance(n, int | float):
        raise TypeError(f"n must be a number, 'fro' or 'nuc', got {n!r}")
    return lucid.linalg.vector_norm(t, ord=n, dim=dims)


def _scores_of(
    current: Tensor, importance_scores: Tensor | None, name: str, module: Module
) -> Tensor:
    """What ranks the entries of ``module.<name>``: the caller's scores, else its current values."""
    if importance_scores is None:
        return current.detach()
    if tuple(importance_scores.shape) != tuple(current.shape):
        raise ValueError(
            f"importance_scores should have the same shape as parameter {name} of "
            f"{type(module).__name__}, got {tuple(importance_scores.shape)} vs "
            f"{tuple(current.shape)}"
        )
    return importance_scores.detach()


# ── the hierarchy ───────────────────────────────────────────────────────


class BasePruningMethod(_ReparamHook, ABC):
    r"""Base of every pruning technique: a mask recipe and the plumbing to apply it.

    A subclass declares :attr:`PRUNING_TYPE` and implements
    :meth:`compute_mask`; it usually also defines a classmethod ``apply`` with
    its own arguments that forwards to :meth:`BasePruningMethod.apply`.
    Installed on a module, an instance is the forward pre-hook that rebuilds
    ``<name> = <name>_orig * <name>_mask`` before each call.

    Attributes
    ----------
    PRUNING_TYPE : {"unstructured", "structured", "global"}
        What :class:`PruningContainer` slices out for a later pruning round:
        the unpruned entries, the unpruned channels along ``dim``, or the
        whole tensor.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> class EveryOther(prune.BasePruningMethod):
    ...     PRUNING_TYPE = "unstructured"
    ...     def compute_mask(self, t, default_mask):
    ...         mask = default_mask.clone().reshape([-1])
    ...         mask[::2] = 0
    ...         return mask.reshape(list(t.shape))
    >>> layer = nn.Linear(2, 2)
    >>> _ = EveryOther.apply(layer, "weight")
    >>> int(layer.weight_mask.sum().item())
    2
    """

    PRUNING_TYPE: ClassVar[str]
    _LABEL = "pruning"
    _tensor_name: str

    @property
    @override
    def target(self) -> str:
        """Name of the pruned parameter."""
        return self._tensor_name

    @abstractmethod
    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """Return the mask for the importance scores ``t``.

        Parameters
        ----------
        t : Tensor
            Importance scores of the parameter, the parameter itself unless the
            caller supplied scores.
        default_mask : Tensor
            The mask earlier rounds left (ones for a first round), same shape as
            ``t``.  What it already zeroes stays zero.

        Returns
        -------
        Tensor
            The mask to multiply into the parameter, same shape as ``t``.
        """

    def apply_mask(self, module: Module) -> Tensor:
        """``<name>_orig * <name>_mask`` for ``module``, the pruned tensor."""
        mask: Tensor = getattr(module, self._tensor_name + "_mask")
        orig: Tensor = getattr(module, self._tensor_name + "_orig")
        return mask.to(dtype=orig.dtype) * orig

    @override
    def refresh(self, module: Module) -> None:
        """Rebuild the pruned tensor ``module.<name>`` — what the pre-hook does."""
        self._publish(module, self.apply_mask(module))

    @classmethod
    def apply(
        cls,
        module: Module,
        name: str,
        *args: object,
        importance_scores: Tensor | None = None,
        **kwargs: object,
    ) -> BasePruningMethod:
        """Prune ``module.<name>`` with ``cls(*args, **kwargs)`` and return the installed hook.

        The parameter keeps its identity as ``<name>_orig``; the mask is the
        buffer ``<name>_mask``.  If ``name`` is already pruned the new method
        and the old one are wrapped in a :class:`PruningContainer`, and the new
        mask is computed only over what the old masks left alive.  Nothing on
        ``module`` changes unless the mask computes.

        Parameters
        ----------
        module : Module
            Host of the parameter.
        name : str
            Name of the parameter to prune.
        *args, **kwargs
            Passed to the constructor of ``cls``.
        importance_scores : Tensor, optional
            Scores, shaped like the parameter, that rank its entries; the
            parameter's own values when ``None``.

        Returns
        -------
        BasePruningMethod
            The pre-hook now installed: the new method, or the container
            holding it.
        """
        _require_module(module, "prune")
        method: BasePruningMethod = cls(*args, **kwargs)
        method._tensor_name = name
        entry = BasePruningMethod._entry(module, name)
        if entry is None:
            return BasePruningMethod._first_round(module, name, method, importance_scores)
        return BasePruningMethod._next_round(module, name, method, entry, importance_scores)

    @staticmethod
    def _first_round(
        module: Module, name: str, method: BasePruningMethod, importance_scores: Tensor | None
    ) -> BasePruningMethod:
        """Prune a parameter for the first time: it moves aside as ``<name>_orig``, unchanged."""
        orig: Parameter = _require_parameter(module, name, "prune")
        scores: Tensor = _scores_of(orig, importance_scores, name, module)
        mask: Tensor = method.compute_mask(scores, default_mask=lucid.ones_like(orig))
        module.register_parameter(name + "_orig", orig)
        del module._parameters[name]
        module.register_buffer(name + "_mask", mask)
        method.refresh(module)
        _ReparamHook._register(module, method)
        return method

    @staticmethod
    def _next_round(
        module: Module,
        name: str,
        method: BasePruningMethod,
        entry: tuple[int, BasePruningMethod],
        importance_scores: Tensor | None,
    ) -> BasePruningMethod:
        """Prune again: the old method and the new one go in a container, the new mask on the old."""
        key, old = entry
        # A fresh container, so a mask that fails to compute leaves the old one as it was.
        container = PruningContainer(
            *(old if isinstance(old, PruningContainer) else (old,)), method
        )
        scores: Tensor = _scores_of(old.apply_mask(module), importance_scores, name, module)
        mask: Tensor = container.compute_mask(
            scores, default_mask=getattr(module, name + "_mask").detach().clone()
        )
        setattr(module, name + "_mask", mask)
        del module._forward_pre_hooks[key]
        container.refresh(module)
        _ReparamHook._register(module, container)
        return container

    def prune(
        self, t: Tensor, default_mask: Tensor | None = None, importance_scores: Tensor | None = None
    ) -> Tensor:
        """Return ``t`` with this method's mask applied — no module involved.

        Parameters
        ----------
        t : Tensor
            Tensor to prune.
        default_mask : Tensor, optional
            Mask of an earlier round to respect; ones when ``None``.
        importance_scores : Tensor, optional
            Scores ranking ``t``'s entries; ``t`` itself when ``None``.

        Returns
        -------
        Tensor
            ``t * mask``.
        """
        if importance_scores is not None and tuple(importance_scores.shape) != tuple(t.shape):
            raise ValueError(
                "importance_scores should have the same shape as tensor t, got "
                f"{tuple(importance_scores.shape)} vs {tuple(t.shape)}"
            )
        scores: Tensor = t if importance_scores is None else importance_scores
        base: Tensor = lucid.ones_like(t) if default_mask is None else default_mask
        return t * self.compute_mask(scores, default_mask=base)

    @override
    def remove(self, module: Module) -> None:
        """Make the pruning permanent: ``<name>`` is again the one Parameter, holding ``W * mask``.

        ``<name>_orig`` is written in place and handed back as ``<name>`` — the
        same object, so its optimizer state and ``requires_grad`` survive.  The
        mask buffer and the pre-hook go away.
        """
        name: str = self._tensor_name
        orig: Parameter = _require_parameter(module, name + "_orig", "remove pruning")
        with lucid.no_grad():
            orig.copy_(self.apply_mask(module))
        self._vacate(module)
        del module._parameters[name + "_orig"]
        del module._buffers[name + "_mask"]
        module.register_parameter(name, orig)


class PruningContainer(BasePruningMethod):
    r"""The methods applied, in order, to one parameter — iterative pruning.

    Created by :meth:`BasePruningMethod.apply` the second time a parameter is
    pruned.  The newest method decides what to prune; the earlier ones matter
    through the mask they left, which it must respect.

    Parameters
    ----------
    *methods : BasePruningMethod
        Methods already applied to one parameter, oldest first.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> layer = nn.Linear(4, 3)
    >>> _ = prune.l1_unstructured(layer, "weight", amount=2)
    >>> _ = prune.l1_unstructured(layer, "weight", amount=3)
    >>> int((layer.weight_mask == 0).sum().item())
    5
    """

    def __init__(self, *methods: BasePruningMethod) -> None:
        self._pruning_methods: tuple[BasePruningMethod, ...] = ()
        for method in methods:
            self.add_pruning_method(method)

    def add_pruning_method(self, method: BasePruningMethod) -> None:
        """Append ``method``; it must prune the same parameter as the ones held.

        Raises
        ------
        TypeError
            If ``method`` is not a :class:`BasePruningMethod`.
        ValueError
            If it acts on another parameter.
        """
        if not isinstance(method, BasePruningMethod):
            raise TypeError(f"{type(method)} is not a BasePruningMethod subclass")
        if not self._pruning_methods:
            self._tensor_name = method._tensor_name
        elif self._tensor_name != method._tensor_name:
            raise ValueError(
                f"Can only add pruning methods acting on the parameter named "
                f"'{self._tensor_name}' to PruningContainer {self}. Found '{method._tensor_name}'"
            )
        self._pruning_methods += (method,)

    def __len__(self) -> int:
        return len(self._pruning_methods)

    def __iter__(self) -> Iterator[BasePruningMethod]:
        return iter(self._pruning_methods)

    def __getitem__(self, idx: int) -> BasePruningMethod:
        return self._pruning_methods[idx]

    @override
    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """Combine ``default_mask`` with the newest method's mask over what is still alive.

        The slice the newest method sees depends on its ``PRUNING_TYPE``: the
        unmasked entries (raveled) for ``"unstructured"``, the channels not
        entirely masked for ``"structured"``, everything for ``"global"``.

        Parameters
        ----------
        t : Tensor
            Importance scores of the parameter.
        default_mask : Tensor
            The mask the earlier rounds left, same shape as ``t``.

        Returns
        -------
        Tensor
            The combined mask, same shape as ``t``.
        """
        method: BasePruningMethod = self._pruning_methods[-1]
        new_mask: Tensor = default_mask.to(dtype=t.dtype).clone()
        kind: str = method.PRUNING_TYPE
        if kind == "unstructured":
            alive: Tensor = default_mask == 1
            partial: Tensor = method.compute_mask(t[alive], default_mask=default_mask[alive])
            new_mask[alive] = partial.to(dtype=new_mask.dtype)
        elif kind == "structured":
            return self._combine_structured(method, t, default_mask, new_mask)
        elif kind == "global":
            new_mask = method.compute_mask(t, default_mask=default_mask).to(dtype=new_mask.dtype)
        else:
            raise ValueError(f"Unrecognized PRUNING_TYPE {kind}")
        return new_mask

    @staticmethod
    def _combine_structured(
        method: BasePruningMethod, t: Tensor, mask: Tensor, new_mask: Tensor
    ) -> Tensor:
        """Run a structured ``method`` over the channels of ``mask`` that are not all zero."""
        axis = getattr(method, "dim", None)
        if axis is None:
            raise AttributeError(
                'Pruning methods of PRUNING_TYPE "structured" need to have the attribute '
                "`dim` defined."
            )
        dim: int = int(axis)
        ndim: int = len(t.shape)
        if not -ndim <= dim < ndim:
            raise IndexError(f"Index is out of bounds for tensor with dimensions {ndim}")
        dim %= ndim
        others: list[int] = [d for d in range(ndim) if d != dim]
        alive: Tensor = lucid.nonzero(mask.sum(dim=others) != 0).reshape([-1])
        partial: Tensor = method.compute_mask(
            lucid.index_select(t, dim, alive), default_mask=lucid.index_select(mask, dim, alive)
        )
        moved: Tensor = new_mask.movedim(dim, 0).contiguous()
        moved[alive] = partial.movedim(dim, 0).to(dtype=moved.dtype)
        return moved.movedim(0, dim).contiguous()


class Identity(BasePruningMethod):
    """Prune nothing: install the ``orig * mask`` plumbing with a mask of ones.

    The weight stays bitwise what it was; later rounds then compose onto the
    all-ones mask through a :class:`PruningContainer`.
    """

    PRUNING_TYPE = "unstructured"

    @override
    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """The mask of the earlier rounds, unchanged (ones for a first round)."""
        return default_mask


class _CountedUnstructured(BasePruningMethod):
    """Prune a counted ``amount`` of the entries not yet pruned, lowest score first.

    A subclass says what the score of an entry is (:meth:`_scores`); the count
    — a float fraction rounded, or an int taken as is — and the ties are settled
    here, once: exactly that many entries go, whatever the scores.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount: _Amount) -> None:
        _validate_pruning_amount_init(amount)
        self.amount: _Amount = amount

    @abstractmethod
    def _scores(self, t: Tensor) -> Tensor:
        """One score per entry of ``t``; the lowest are pruned."""

    @override
    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """``default_mask`` with exactly the requested number of lowest-score entries zeroed."""
        size: int = t.numel()
        k: int = _compute_nparams_toprune(self.amount, size)
        _validate_pruning_amount(k, size)
        if k == 0:
            return default_mask.clone()
        drop: Tensor = _top_indices(self._scores(t).reshape([-1]), k, largest=False)
        return _zero_flat(default_mask, drop)


class RandomUnstructured(_CountedUnstructured):
    """Prune a random ``amount`` of the entries not yet pruned.

    Parameters
    ----------
    amount : int or float
        A float is the fraction of entries to prune, an int their exact count.
    """

    @override
    def _scores(self, t: Tensor) -> Tensor:
        return lucid.rand_like(t)


class L1Unstructured(_CountedUnstructured):
    """Prune the ``amount`` entries, not yet pruned, with the smallest magnitude.

    Parameters
    ----------
    amount : int or float
        A float is the fraction of entries to prune, an int their exact count.
        Ties at the cutoff are broken so that exactly that many go.
    """

    @override
    def _scores(self, t: Tensor) -> Tensor:
        return lucid.abs(t)


class _CountedStructured(BasePruningMethod):
    """Prune a counted ``amount`` of the channels along ``dim`` not yet pruned, lowest score first.

    A subclass says what the score of a channel is (:meth:`_scores`); the count,
    the axis checks and the whole-channel mask are settled here, once.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount: _Amount, dim: int = -1) -> None:
        _validate_pruning_amount_init(amount)
        self.amount: _Amount = amount
        self.dim: int = dim

    @abstractmethod
    def _scores(self, t: Tensor) -> Tensor:
        """One score per channel of ``t`` along ``dim``; the lowest are pruned."""

    @override
    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """``default_mask`` with the requested number of lowest-score channels zeroed whole."""
        _validate_structured_pruning(t)
        _validate_pruning_dim(t, self.dim)
        size: int = int(t.shape[self.dim])
        k: int = _compute_nparams_toprune(self.amount, size)
        _validate_pruning_amount(k, size)
        if k == 0:
            return default_mask.clone()
        kept: Tensor = _top_indices(self._scores(t), size - k, largest=True)
        return _channel_mask(t, self.dim, kept) * default_mask.to(dtype=t.dtype)


class RandomStructured(_CountedStructured):
    """Prune a random ``amount`` of the channels, along ``dim``, not yet pruned.

    Parameters
    ----------
    amount : int or float
        A float is the fraction of channels to prune, an int their exact count.
    dim : int, optional
        Axis whose slices are the channels.  Default ``-1``.
    """

    @override
    def _scores(self, t: Tensor) -> Tensor:
        return lucid.rand(int(t.shape[self.dim]), device=t.device)


class LnStructured(_CountedStructured):
    """Prune the ``amount`` channels, along ``dim``, with the smallest ``L_n`` norm.

    Parameters
    ----------
    amount : int or float
        A float is the fraction of channels to prune, an int their exact count.
    n : int, float, "fro" or "nuc"
        The norm taken over each channel (every axis but ``dim``): a vector
        order including ``inf`` / ``-inf``, ``"fro"``, or ``"nuc"`` for a
        channel that is a matrix.
    dim : int, optional
        Axis whose slices are the channels.  Default ``-1``.
    """

    def __init__(self, amount: _Amount, n: _Norm, dim: int = -1) -> None:
        super().__init__(amount, dim)
        self.n: _Norm = n

    @override
    def _scores(self, t: Tensor) -> Tensor:
        return _compute_norm(t, self.n, self.dim)


class CustomFromMask(BasePruningMethod):
    """Prune with a mask the caller computed.

    Parameters
    ----------
    mask : Tensor
        Binary mask, shaped like the parameter; multiplied into the mask of any
        earlier round.
    """

    PRUNING_TYPE = "global"

    def __init__(self, mask: Tensor) -> None:
        self.mask: Tensor = mask

    @override
    def compute_mask(self, t: Tensor, default_mask: Tensor) -> Tensor:
        """``default_mask * mask``; the mask's shape must match."""
        if tuple(default_mask.shape) != tuple(self.mask.shape):
            raise ValueError(
                f"default_mask shape {tuple(default_mask.shape)} must match "
                f"self.mask shape {tuple(self.mask.shape)}"
            )
        return default_mask * self.mask.to(dtype=default_mask.dtype)


# ── entry points ────────────────────────────────────────────────────────


def identity(module: Module, name: str = "weight") -> Module:
    r"""Install the pruning plumbing on ``module.<name>`` without pruning anything.

    The weight is bitwise unchanged, but ``<name>_orig`` / ``<name>_mask`` exist
    and a forward pre-hook rebuilds ``<name>``.  A baseline, a wiring check, and
    the starting point for later rounds.

    Parameters
    ----------
    module : Module
        Host module.  Mutated in place.
    name : str, optional
        Parameter to prune.  Default ``"weight"``.

    Returns
    -------
    Module
        The same module, with ``<name>_orig``, ``<name>_mask`` (all ones) and
        the pre-hook.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> layer = nn.Linear(4, 3)
    >>> _ = prune.identity(layer)
    >>> prune.is_pruned(layer)
    True
    """
    Identity.apply(module, name)
    return module


def random_unstructured(module: Module, name: str = "weight", amount: _Amount = 0.5) -> Module:
    r"""Prune random entries of ``module.<name>``.

    Parameters
    ----------
    module : Module
        Host module.  Mutated in place.
    name : str, optional
        Parameter to prune.  Default ``"weight"``.
    amount : int or float, optional
        A float is the fraction of the entries not yet pruned to prune, an int
        their exact count.  Default ``0.5``.

    Returns
    -------
    Module
        The module with the pruning hook installed.

    Raises
    ------
    ValueError
        If ``amount`` is a float outside :math:`[0, 1]`, a negative int, or an
        int larger than the number of entries.

    Notes
    -----
    Exactly ``amount`` (as a count) entries are zeroed — the ``k`` largest of a
    uniform random draw.  Pruning the same parameter again prunes among the
    entries still alive.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> layer = nn.Linear(10, 10)
    >>> _ = prune.random_unstructured(layer, amount=30)
    >>> int((layer.weight_mask == 0).sum().item())
    30
    """
    RandomUnstructured.apply(module, name, amount)
    return module


def l1_unstructured(
    module: Module,
    name: str = "weight",
    amount: _Amount = 0.5,
    importance_scores: Tensor | None = None,
) -> Module:
    r"""Prune the smallest-magnitude entries of ``module.<name>`` (Han et al., 2015).

    Parameters
    ----------
    module : Module
        The module that owns the parameter; changed in place.
    name : str, optional
        Which parameter to prune; ``"weight"`` unless told otherwise.
    amount : int or float, optional
        How much to prune.  A float is a fraction of the entries still alive,
        an int is a count of them.  Default ``0.5``.
    importance_scores : Tensor, optional
        Scores, shaped like the parameter, that rank its entries in place of
        ``|W|``.

    Returns
    -------
    Module
        The same module, now carrying the mask and the hook.

    Raises
    ------
    ValueError
        If ``amount`` is out of range for its type, or as a count exceeds the
        number of entries.

    Notes
    -----
    The ``k = \operatorname{round}(\text{amount}\cdot N)`` entries of smallest
    :math:`|W_i|` are zeroed — exactly ``k``, ties at the cutoff included.
    Pruning again composes through a :class:`PruningContainer`: the second round
    ranks only the entries the first left.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> layer = nn.Linear(10, 10)
    >>> _ = prune.l1_unstructured(layer, amount=0.7)
    >>> int((layer.weight_mask == 0).sum().item())   # the 70 smallest |w| go
    70
    """
    L1Unstructured.apply(module, name, amount=amount, importance_scores=importance_scores)
    return module


def random_structured(module: Module, name: str, amount: _Amount, dim: int) -> Module:
    r"""Prune random channels of ``module.<name>`` along ``dim``.

    Parameters
    ----------
    module : Module
        Host module.  Mutated in place.
    name : str
        Parameter to prune.
    amount : int or float
        A float is the fraction of the channels not yet pruned to prune, an int
        their exact count.
    dim : int
        Axis whose slices are the channels.

    Returns
    -------
    Module
        The module with the pruning hook installed.

    Raises
    ------
    ValueError
        If the parameter has fewer than two dimensions or ``amount`` is invalid.
    IndexError
        If ``dim`` is not an axis of the parameter.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> layer = nn.Linear(5, 3)
    >>> _ = prune.random_structured(layer, "weight", amount=3, dim=1)
    >>> int((layer.weight_mask.sum(dim=0) == 0).sum().item())    # three whole columns
    3
    """
    RandomStructured.apply(module, name, amount, dim)
    return module


def ln_structured(
    module: Module,
    name: str,
    amount: _Amount,
    n: _Norm,
    dim: int,
    importance_scores: Tensor | None = None,
) -> Module:
    r"""Prune the lowest-norm channels of ``module.<name>`` along ``dim``.

    Parameters
    ----------
    module : Module
        The module that owns the parameter; changed in place.
    name : str
        Which parameter to prune.
    amount : int or float
        How much to prune.  A float is a fraction of the channels still alive,
        an int is a count of them.
    n : int, float, "fro" or "nuc"
        Norm of each channel, taken over every axis but ``dim``.
    dim : int
        The axis whose slices are the channels.
    importance_scores : Tensor, optional
        Scores, shaped like the parameter, that the norms are taken of in
        place of the parameter.

    Returns
    -------
    Module
        The same module, now carrying the mask and the hook.

    Raises
    ------
    ValueError
        If the parameter is below two dimensions, or ``amount`` is out of range.
    IndexError
        If ``dim`` names no axis of the parameter.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> layer = nn.Conv2d(5, 4, 2)
    >>> _ = prune.ln_structured(layer, "weight", amount=0.5, n=2, dim=0)
    >>> int((layer.weight_mask.sum(dim=(1, 2, 3)) == 0).sum().item())   # two filters
    2
    """
    LnStructured.apply(module, name, amount, n, dim, importance_scores=importance_scores)
    return module


def global_unstructured(
    parameters: Iterable[tuple[Module, str]],
    pruning_method: type[BasePruningMethod],
    importance_scores: dict[tuple[Module, str], Tensor] | None = None,
    **kwargs: object,
) -> None:
    r"""Prune several parameters at once, ranking all their entries together.

    The entries of every ``(module, name)`` are pooled, the (unstructured)
    ``pruning_method`` picks among the ones still alive, and each parameter
    receives its slice of the resulting mask through :func:`custom_from_mask`.

    Parameters
    ----------
    parameters : iterable of (Module, str)
        The parameters to prune together.
    pruning_method : type
        A pruning class with ``PRUNING_TYPE == "unstructured"`` such as
        :class:`L1Unstructured`.
    importance_scores : dict, optional
        ``(module, name) -> scores`` shaped like the parameter; a parameter
        absent from it is ranked by its own values.
    **kwargs
        Constructor arguments of ``pruning_method``, e.g. ``amount``.

    Raises
    ------
    TypeError
        If ``parameters`` is not iterable, ``importance_scores`` not a dict, or
        the method is not unstructured.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> net = nn.Sequential(nn.Linear(10, 4), nn.Linear(4, 1))
    >>> prune.global_unstructured(
    ...     [(net[0], "weight"), (net[1], "weight")], prune.L1Unstructured, amount=10
    ... )
    >>> sum(int((m.weight_mask == 0).sum().item()) for m in net)
    10
    """
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")
    if importance_scores is not None and not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")
    scores_of: dict[tuple[Module, str], Tensor] = importance_scores or {}
    sites: list[tuple[Module, str]] = list(parameters)
    method: BasePruningMethod = pruning_method(**kwargs)
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for the `pruning_method`. '
            f"Found method {pruning_method} of type {method.PRUNING_TYPE}"
        )
    tensors: list[Tensor] = [getattr(module, name) for module, name in sites]
    scores: Tensor = parameters_to_vector(
        [scores_of.get(site, tensor).detach() for site, tensor in zip(sites, tensors, strict=True)]
    )
    alive: Tensor = parameters_to_vector(
        [
            getattr(module, name + "_mask", lucid.ones_like(tensor)).detach()
            for (module, name), tensor in zip(sites, tensors, strict=True)
        ]
    )
    method._tensor_name = "global"
    container = PruningContainer(method)
    final: Tensor = container.compute_mask(scores, alive)
    start: int = 0
    for (module, name), tensor in zip(sites, tensors, strict=True):
        size: int = tensor.numel()
        custom_from_mask(module, name, final[start : start + size].reshape(list(tensor.shape)))
        start += size


def custom_from_mask(module: Module, name: str, mask: Tensor) -> Module:
    r"""Prune ``module.<name>`` with a mask the caller computed.

    Parameters
    ----------
    module : Module
        Host module.  Mutated in place.
    name : str
        Parameter to prune.
    mask : Tensor
        Binary mask, shaped like the parameter.

    Returns
    -------
    Module
        The module with the pruning hook installed.

    Raises
    ------
    ValueError
        If ``mask``'s shape differs from the parameter's.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> layer = nn.Linear(2, 2)
    >>> _ = prune.custom_from_mask(layer, "weight", lucid.tensor([[1.0, 0.0], [0.0, 1.0]]))
    >>> layer.weight_mask.tolist()
    [[1.0, 0.0], [0.0, 1.0]]
    """
    CustomFromMask.apply(module, name, mask)
    return module


def remove(module: Module, name: str = "weight") -> Module:
    r"""Make the pruning of ``module.<name>`` permanent and drop the hook.

    ``<name>`` becomes a plain Parameter again — the *same* object that was
    ``<name>_orig`` — holding ``W_orig * mask``; ``<name>_orig``, the mask
    buffer and the pre-hook are gone.  Pruning is not undone: the zeros stay.

    Parameters
    ----------
    module : Module
        Module previously passed through one of the prune methods.
    name : str, optional
        Parameter name to finalise.  Default ``"weight"``.

    Returns
    -------
    Module
        The same module.

    Raises
    ------
    ValueError
        If ``name`` is not pruned.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> layer = nn.Linear(4, 3)
    >>> _ = prune.l1_unstructured(layer, amount=0.5)
    >>> _ = prune.remove(layer)
    >>> prune.is_pruned(layer)
    False
    """
    return BasePruningMethod.uninstall(module, name)


def is_pruned(module: Module) -> bool:
    r"""Whether ``module`` or any module under it carries a pruning hook.

    Parameters
    ----------
    module : Module
        Module to inspect; its submodules are searched too.

    Returns
    -------
    bool
        ``True`` if some module in the tree has a pruning pre-hook installed.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import prune
    >>> net = nn.Sequential(nn.Linear(4, 3))
    >>> prune.is_pruned(net)
    False
    >>> _ = prune.l1_unstructured(net[0], amount=0.2)
    >>> prune.is_pruned(net)
    True
    """
    return any(
        isinstance(hook, BasePruningMethod)
        for sub in module.modules()
        for _, hook in _ReparamHook.installed(sub)
    )
