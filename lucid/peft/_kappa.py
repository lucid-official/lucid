"""KappaTune: pick the modules to adapt by how well-conditioned their weights are."""

from typing import Final, NamedTuple

import lucid
import lucid.linalg
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter

#: Names that end the parameter paths of fused mixture-of-experts weights.
_MOE_SUFFIXES: Final = (".gate_up_proj", ".down_proj", ".gate_proj", ".up_proj")
#: Added to the smallest singular value so that a singular matrix has a finite number.
_EPSILON: Final = 1e-8


def _condition_number(singular_values: Tensor) -> float:
    """Largest over smallest singular value (``svdvals`` returns them in descending order)."""
    return float(singular_values[0].item() / (singular_values[-1].item() + _EPSILON))


class _Kappas(NamedTuple):
    """Condition numbers by name: of the linear modules, and of the fused expert parameters."""

    modules: dict[str, float]
    parameters: dict[str, float]


class _KappaTuneSelector:
    """Condition numbers of a model's linear weights and fused expert weights, lowest first.

    A matrix with a small condition number (largest over smallest singular value) is
    well-conditioned, which makes it a safer place for a low-rank update than one
    close to singular.  The numbers cost an SVD of every weight, so they are computed
    once, on the first request, and kept.
    """

    def __init__(
        self,
        model: nn.Module,
        max_dim_size_to_analyze: int,
        moe_param_suffixes: tuple[str, ...] | None,
    ) -> None:
        self._model = model
        self._max_dim = max_dim_size_to_analyze
        self._suffixes = moe_param_suffixes or _MOE_SUFFIXES
        self._kappas: _Kappas | None = None

    def _analysable(self, weight: Tensor) -> bool:
        return all(dim <= self._max_dim for dim in weight.shape)

    @staticmethod
    def _full_precision(parameter: Parameter) -> Tensor:
        return parameter.data.to(lucid.float32)

    def _compute(self) -> _Kappas:
        if self._kappas is not None:
            return self._kappas
        modules: dict[str, float] = {}
        for name, module in self._model.named_modules():
            if isinstance(module, nn.Linear):
                weight = self._full_precision(module.weight)
                if self._analysable(weight):
                    modules[name] = _condition_number(lucid.linalg.svdvals(weight))
        parameters: dict[str, float] = {}
        for name, parameter in self._model.named_parameters():
            if parameter.ndim == 3 and name.endswith(self._suffixes):
                experts = self._full_precision(parameter)
                if self._analysable(experts[0]):
                    kappas = [_condition_number(lucid.linalg.svdvals(e)) for e in experts]
                    parameters[name] = sum(kappas) / len(kappas)
        self._kappas = _Kappas(modules, parameters)
        return self._kappas

    @staticmethod
    def _lowest(kappas: dict[str, float], top_p: float) -> list[str]:
        """The names of the ``top_p`` fraction with the lowest numbers (at least one, if any)."""
        ordered = sorted(kappas, key=kappas.__getitem__)
        return ordered[: max(1, int(len(ordered) * top_p))] if ordered else []

    def best_modules(self, top_p: float) -> list[str]:
        """The linear modules to adapt."""
        return self._lowest(self._compute().modules, top_p)

    def best_parameters(self, top_p: float) -> list[str]:
        """The fused expert parameters to adapt."""
        return self._lowest(self._compute().parameters, top_p)


def find_kappa_target_modules(
    model: nn.Module,
    top_p: float = 0.2,
    max_dim_size_to_analyze: int = 16384,
    moe_param_suffixes: tuple[str, ...] | None = None,
    show_progress: bool = True,
) -> dict[str, list[str] | None]:
    r"""Choose what to adapt by the condition number of the weights (KappaTune).

    The condition number of a weight matrix is its largest singular value over its
    smallest.  The ``top_p`` fraction of ``nn.Linear`` modules with the lowest numbers
    are returned as ``target_modules``, and the same fraction of the fused
    mixture-of-experts weights (three-dimensional parameters named ``gate_up_proj``,
    ``down_proj`` and the like) as ``target_parameters``, a pair ready to give to
    :class:`~lucid.peft.LoraConfig`.  An expert weight's number is the mean over its
    experts.  Every weight is decomposed, so a large model takes a while.

    Parameters
    ----------
    model : nn.Module
        The base model whose weights are analysed.
    top_p : float, default=0.2
        The fraction, between 0 and 1, of candidates to keep; at least one is kept
        when there are any.
    max_dim_size_to_analyze : int, default=16384
        A weight with a dimension larger than this is skipped.
    moe_param_suffixes : tuple of str, optional
        The endings of the names of fused expert parameters; ``.gate_up_proj``,
        ``.down_proj``, ``.gate_proj`` and ``.up_proj`` when ``None``.
    show_progress : bool, default=True
        Accepted for the reference's signature; no progress bar is drawn.

    Returns
    -------
    dict of str to list of str or None
        ``{"target_modules": [...], "target_parameters": [...] or None}``; the second is
        ``None`` when the model has no fused expert weights.

    Raises
    ------
    ValueError
        ``top_p`` is outside ``[0, 1]``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import find_kappa_target_modules
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8))
    >>> picked = find_kappa_target_modules(net, top_p=0.4)
    >>> len(picked["target_modules"]), picked["target_parameters"]
    (1, None)
    """
    del show_progress  # the reference draws a progress bar; this library has none to draw
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"top_p must be between 0 and 1, got {top_p}")
    selector = _KappaTuneSelector(model, max_dim_size_to_analyze, moe_param_suffixes)
    return {
        "target_modules": selector.best_modules(top_p),
        "target_parameters": selector.best_parameters(top_p) or None,
    }
