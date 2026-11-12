"""``MixedModel``: one model holding the adapters of several methods, their layers stacked on each module.

A single method's tuner puts one layer on each module it adapts.  ``MixedModel`` lets the layers of
several methods share a module: the second method's layer is built around the first's, so ``lin0``
is ``loha.Linear(lora.Linear(nn.Linear))`` and the forward pass is the composition, lowest first.
Everything but the choice of layer is the generic tuner's: planning is atomic, switching, freezing,
deleting and unloading walk the adapter slots without knowing whose they are, and the hooks that
are one method's (what a layer is built from, what trains besides the adapters) go to the tuner
of the method whose config is at hand.

What the host owns is what only a stack can get wrong:

* **One layer per module and method.**  A method whose layer is already in the module's stack
  gets its adapter added there; otherwise a new layer goes on top.  So the stack's order is
  the order the methods first appeared in, lowest first.
* **What may stand on what.**  A layer class stands on another method's layer only when it has
  declared that it can (``_stackable``): a class that computes from the weight would drop the
  layer beneath it, and one that reads the weight reads the wrong one.  The innermost layer may
  be of any class.  Every refusal is made while planning, before the model is touched.
* **The order layers merge in.**  A layer's update is computed from the weight the layers beneath
  it leave, so a module's layers merge lowest first and unmerge highest first
  (``_merge_order``), and a merge that would fold a layer's update under a transform it was not
  computed beneath is refused before a weight is written.

A method is mixable once it is declared (``class XModel(..., peft_type=..., mixed=True)``) and
has passed ``test_mixed_model_contract``, which builds every ordered pair of the declared
methods and checks the stack against a dense probe.
"""

from collections.abc import Sequence
from typing import ClassVar, TypeIs, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, _TargetedConfig
from lucid.peft._slots import _adapter_slots, _outermost_slots
from lucid.peft.tuners._host import _registered
from lucid.peft.tuners._layer import _BaseTunerLayer, _check_adapter_name, _qualified
from lucid.peft.tuners._plan import _NamesParameters, _not_found, _selected_modules, _Target
from lucid.peft.tuners._tuner import _MIXED_TUNERS, _BaseTuner


def _compatible_types() -> str:
    return ", ".join(sorted(peft_type.value for peft_type in _MIXED_TUNERS))


def _family(config: object) -> type[_BaseTuner]:
    """The tuner of the method ``config`` belongs to, if that method can be mixed.

    Raises
    ------
    TypeError
        ``config`` is not a ``PeftConfig``.
    ValueError
        The method is not declared ``mixed=True``.
    """
    if not isinstance(config, PeftConfig):
        raise TypeError(f"config must be a PeftConfig, got {type(config).__name__}")
    family = _registered(_MIXED_TUNERS, config)
    if family is None:
        raise ValueError(
            f"The provided `peft_type` '{config.peft_type}' is not compatible with the "
            f"`PeftMixedModel`. Compatible types are: {_compatible_types()}"
        )
    return family


def _family_of(layer: _BaseTunerLayer) -> type[_BaseTuner]:
    """The tuner of the method a layer belongs to: the one whose table of layer classes it shares."""
    for tuner in _MIXED_TUNERS.values():
        if tuner._layer_types is layer._layer_types:
            return tuner
    raise RuntimeError(
        f"{_qualified(type(layer))} is a layer of a method that is not declared mixed=True"
    )


def _chain(top: nn.Module) -> list[_BaseTunerLayer]:
    """The tuner layers standing on one another from ``top`` down, outermost first."""
    layers: list[_BaseTunerLayer] = []
    module = top
    while isinstance(module, _BaseTunerLayer):
        layers.append(module)
        module = module.base_layer
    return layers


def _adds_to_output(layer: _BaseTunerLayer) -> bool:
    return _family_of(layer)._layer_adds_to_output(layer)


class MixedModel(_BaseTuner):
    r"""A model holding adapters of several methods, their layers stacked on each module.

    The first adapter of a method puts that method's layer on the module it targets; the layers of
    later methods stand on it, so the forward pass runs the layers lowest first.  A method that
    already has a layer on a module gets its further adapters in that layer.  ``set_adapter``
    names the adapters in use, of any methods; a layer holding none of them passes its input
    through to the layer beneath.  Merging folds each module's layers into its weight lowest
    first, and ``unmerge_adapter`` takes them out highest first.

    Only methods declared mixable (``mixed=True`` in the tuner's class statement) take part.
    A layer class stands on another method's layer only if it declares so, so the adapters that
    read the base weight (HiRA, DoRA) are the lowest in a stack, and a method that computes from
    the weight (BOFT, HRA, MiSS) takes a module alone.  Adding an adapter that cannot sit where
    it would land raises ``ValueError`` and leaves the model as it was.

    AdaLoRA takes part for inference: the host does not drive its rank allocator or its
    orthogonality loss, which belong to ``AdaLoraModel``.  Adapters of methods that train a
    shared table (VeRA) or hold one adapter at a time (prompt learning) are not mixable.

    Parameters
    ----------
    model : nn.Module
        The model to adapt; it must hold no adapters yet.
    peft_config : PeftConfig
        The first adapter's config: any mixable method's.
    adapter_name : str
        Name of the first adapter.
    low_cpu_mem_usage, state_dict
        As in :class:`~lucid.peft.LoraModel`.

    Raises
    ------
    ValueError
        The method is not mixable, the model already holds adapters, or the adapter cannot be
        applied (no module matches, or a layer cannot stand where this one would).

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoHaConfig, LoraConfig
    >>> from lucid.peft.tuners import MixedModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> mixed = MixedModel(net, LoraConfig(r=2, target_modules=["0"]), "lora")
    >>> mixed.add_adapter("loha", LoHaConfig(r=2, target_modules=["0"]))
    >>> type(net[0]).__module__.rsplit(".", 2)[-2:], type(net[0].base_layer).__module__.rsplit(".", 2)[-2:]
    (['loha', '_layer'], ['lora', '_layer'])
    >>> mixed.set_adapter(["lora", "loha"])
    >>> mixed.active_adapters
    ['lora', 'loha']
    >>> mixed.unload() is net, type(net[0]) is nn.Linear
    (True, True)
    """

    _config_class: ClassVar[type[_TargetedConfig]] = _TargetedConfig

    @override
    def __init__(
        self,
        model: nn.Module,
        peft_config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        """Adapt ``model`` with ``peft_config``'s method, as ``adapter_name``, ready to take other methods' adapters."""
        _family(peft_config)
        super().__init__(model, peft_config, adapter_name, low_cpu_mem_usage, state_dict)

    @override
    def _accepts(self, config: object) -> TypeIs[_TargetedConfig]:
        return (
            isinstance(config, _TargetedConfig) and _registered(_MIXED_TUNERS, config) is not None
        )

    @override
    def _admit(self, adapter_name: str, peft_config: PeftConfig) -> str:
        adapter_name = _check_adapter_name(adapter_name)
        if adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        _family(peft_config)
        if not isinstance(peft_config, _TargetedConfig):
            raise TypeError(f"config must be a _TargetedConfig, got {type(peft_config).__name__}")
        return adapter_name

    @override
    @classmethod
    def _adopted_configs(cls, model: nn.Module, adapter_name: str) -> dict[str, PeftConfig] | None:
        """``None``: a mixed model is built on a model that holds no adapters.

        Raises
        ------
        ValueError
            The model already holds adapter layers or ``modules_to_save`` copies.
        """
        if _adapter_slots(model):
            raise ValueError(
                f"Cannot build a mixed model on a model that already holds adapters (adapter "
                f"{adapter_name!r} was to be added): call unload() on it first."
            )
        return None

    # ── a method's hooks, answered by the method the config belongs to ────────

    @override
    @classmethod
    def _prepare_config(cls, config: _TargetedConfig, model: nn.Module) -> _TargetedConfig:
        return _family(config)._prepare_config(config, model)

    @override
    @classmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        return _family(config)._create_new_module(config, adapter_name, target)

    @override
    @classmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        _family(config)._add_to_layer(config, adapter_name, target)

    @override
    @classmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        return _family(config)._extra_trainable_names(model, config)

    @override
    @classmethod
    def _rewrites_base(cls, config: _TargetedConfig) -> bool:
        return _family(config)._rewrites_base(config)

    @override
    def _check_new_adapter_config(self, config: PeftConfig) -> None:
        self._refuse_second_biased(config)

    # ── planning: one layer per module and method, refused before the model is touched ──

    @override
    @classmethod
    def _plan_targets(cls, model: nn.Module, config: _TargetedConfig) -> list[_Target]:
        family = _family(config)
        if isinstance(config, _NamesParameters) and config.target_parameters:
            raise ValueError(
                f"{family.__name__} was given target_parameters, which wrap one parameter of a "
                "module rather than the module; a mixed model stacks whole modules, so name "
                "target_modules"
            )
        targets = [
            cls._site_target(family, config, key, module)
            for key, module in _selected_modules(model, config)
        ]
        if not targets:
            raise _not_found(config)
        return targets

    @classmethod
    def _site_target(
        cls, family: type[_BaseTuner], config: _TargetedConfig, key: str, top: nn.Module
    ) -> _Target:
        """Where ``config``'s adapter goes on the module at ``key``, whose stack of layers (if any) ends in ``top``.

        The layer of ``family`` already in the stack, to take the adapter; otherwise a new layer of
        the family on top of the stack.

        Raises
        ------
        ValueError
            The module is not one the family can adapt, a layer cannot stand where this one would,
            or the family's own conditions for a mixed model refuse the config.
        """
        chain = _chain(top)
        for depth, layer in enumerate(chain):
            if _family_of(layer) is family:
                _refuse(family, config, key, stacked=depth < len(chain) - 1)
                return _Target(key, layer, None)
        root = chain[0]._root() if chain else top
        layer_class = family._target_layer(key, root)
        refusal = family._mixed_layer_refusal(layer_class)
        if refusal is not None:
            raise ValueError(f"{family.__name__} cannot go on {key!r} of a mixed model: {refusal}")
        if chain and not layer_class._stackable:
            raise ValueError(
                f"{family.__name__} cannot add an adapter on {key!r}: {_qualified(layer_class)} "
                "has not declared that it can stand on another method's layer (_stackable), and "
                f"{key!r} already holds {_held_by(chain)}.  Add this method's adapter before the "
                "others, or unload() and start with it."
            )
        _refuse(family, config, key, stacked=bool(chain))
        return _Target(key, top, layer_class)

    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
        combination_type: str = "svd",
        svd_rank: int | None = None,
        svd_clamp: float | None = None,
        svd_full_matrices: bool = True,
        svd_driver: str | None = None,
        density: float | None = None,
        majority_sign_method: str = "total",
    ) -> None:
        """Refused: a weighted sum of adapters is one method's operation, and a mixed model holds several.

        Parameters
        ----------
        adapters : list of str
            The adapters to combine; not read.
        weights : list of float
            One weight per adapter; not read.
        adapter_name : str
            Name of the combined adapter; not read.
        combination_type : str, default="svd"
            How the adapters are combined, as in :meth:`~lucid.peft.LoraModel.add_weighted_adapter`; not read.
        svd_rank : int, optional
            Rank of an ``"svd"`` result; not read.
        svd_clamp : float, optional
            Quantile the ``"svd"`` factors are clamped to; not read.
        svd_full_matrices : bool, default=True
            Whether the SVD computes full matrices; not read.
        svd_driver : str, optional
            The SVD routine; not read.
        density : float, optional
            The share of elements kept by the pruning types; not read.
        majority_sign_method : str, default="total"
            How TIES elects a sign; not read.

        Raises
        ------
        NotImplementedError
            Always; combine the adapters inside the method's own model, then mix.
        """
        del adapters, weights, adapter_name, combination_type  # the signature a call is written for
        del svd_rank, svd_clamp, svd_full_matrices, svd_driver, density, majority_sign_method
        raise NotImplementedError(
            "add_weighted_adapter is not supported by a mixed model: combine the adapters of "
            "one method before mixing"
        )

    # ── merging: lowest layer first, and nothing that would merge a layer under the wrong transform ──

    @override
    def _merge_order(
        self, layers: list[tuple[str, _BaseTunerLayer]]
    ) -> list[tuple[str, _BaseTunerLayer]]:
        """Each module's layers lowest first.

        ``layers`` lists a module's layer, then the layers it stands on; a run of them is turned around.
        """
        ordered: list[tuple[str, _BaseTunerLayer]] = []
        run: list[tuple[str, _BaseTunerLayer]] = []
        for key, layer in layers:
            if run and not key.startswith(f"{run[0][0]}."):
                ordered.extend(reversed(run))
                run = []
            run.append((key, layer))
        ordered.extend(reversed(run))
        return ordered

    @override
    def merge_adapter(
        self, adapter_names: list[str] | None = None, safe_merge: bool = False
    ) -> None:
        r"""Fold adapter updates into their base weights, each module's layers lowest first.

        A layer's update is computed from the weight the layers beneath it have left, so the
        order is the stack's, and a merge that would fold an update under a layer that
        transforms it differently is refused before any weight is written.  A layer
        standing on others folds into the user's module, so what is left unmerged beneath a
        merged layer must be a sum of terms that commutes with it: refused are the merge of a
        layer while a lower layer keeps adapters in use (its active adapters not merged, now
        or before), and of a lower layer after a higher one has been merged, unless both
        layers add a term to the output (LoRA, LoHa, LoKr, AdaLoRA, SHiRA); a layer above a
        merged one may stay as it is.  A layer merged for some of its own adapters goes on
        applying the others, as in a single method's model, and refuses the splits that method
        cannot reproduce (``_BaseTuner.merge_adapter``).

        Parameters
        ----------
        adapter_names : list of str, optional
            Adapters to merge; the active ones by default.
        safe_merge : bool, default=False
            Check every merged weight for NaN / inf before writing any.

        Raises
        ------
        RuntimeError
            A base weight is shared with another module, or the adapter is disabled.
        ValueError
            A name is not one of the model's adapters, the merge would fold a layer's update
            under a layer that transforms it or leave a layer unable to apply what stays
            unmerged, or ``safe_merge`` found a non-finite merged weight.
        """
        super().merge_adapter(adapter_names, safe_merge)

    @override
    def _merge(
        self, adapter_names: list[str] | None, safe_merge: bool, *, keeps_layers: bool
    ) -> None:
        """Check the stack's merge order, then fold: the shared step of ``merge_adapter`` and ``merge_and_unload``."""
        names = self.active_adapters if adapter_names is None else list(adapter_names)
        self._require_known(names)
        self._check_merge(names)
        super()._merge(adapter_names, safe_merge, keeps_layers=keeps_layers)

    def _check_merge(self, names: Sequence[str]) -> None:
        """Raise ``ValueError`` if merging ``names`` would leave a module's layers in an order that changes its function.

        With the module's layers lowest first, a layer ``above`` and a layer ``below`` it that do
        not both add a term to the output must be folded the way the forward pass runs them:
        the layer below first, and not left in use beneath a layer that was folded without it.
        """
        for key, top in _outermost_slots(self.model):
            if not isinstance(top, _BaseTunerLayer):
                continue
            chain = _chain(top)[::-1]
            folding = [_adapters_of(layer, names, pending=True) for layer in chain]
            for low, below in enumerate(chain):
                for high in range(low + 1, len(chain)):
                    problem = _fold_conflict(below, chain[high], folding[low], folding[high])
                    if problem is not None:
                        raise ValueError(f"cannot merge into {key!r}: {problem}")


def _fold_conflict(
    below: _BaseTunerLayer, above: _BaseTunerLayer, below_now: list[str], above_now: list[str]
) -> str | None:
    """Why folding ``below_now`` and ``above_now`` (the adapters of two layers of one module that merge now) is refused, or ``None``.

    The two layers are ordered correctly when the lower folds first, and when nothing of the lower
    is left in use beneath the upper once that is folded; a pair that both add a term to the output
    commutes, so neither condition matters.
    """
    if _adds_to_output(below) and _adds_to_output(above):
        return None
    if above.merged and below_now:
        return (
            f"{below_now} stand beneath {_qualified(type(above))}, which is merged already, and "
            "an update is folded after the layers beneath it.  Call unmerge_adapter() first."
        )
    # A merged layer goes on applying the active adapters it has not folded.
    left = [
        name
        for name in _adapters_of(below, below.active_adapters)
        if name not in below.merged_adapters and name not in below_now
    ]
    if left and (above.merged or above_now):
        return (
            f"{above_now or above.merged_adapters} would be folded into a weight that "
            f"{_qualified(type(above))} transforms, while {left} stay unmerged beneath it.  "
            "Name those adapters in adapter_names too."
        )
    return None


def _refuse(family: type[_BaseTuner], config: _TargetedConfig, key: str, *, stacked: bool) -> None:
    """Raise ``ValueError`` with the family's reason if it cannot take ``config`` on ``key``."""
    refusal = family._mixed_refusal(config, stacked=stacked)
    if refusal is not None:
        raise ValueError(f"{family.__name__} cannot add this adapter on {key!r}: {refusal}")


def _held_by(chain: list[_BaseTunerLayer]) -> list[str]:
    """The adapters the layers of ``chain`` hold, lowest layer first."""
    return [name for layer in chain[::-1] for name in layer._adapter_names()]


def _adapters_of(
    layer: _BaseTunerLayer, names: Sequence[str], *, pending: bool = False
) -> list[str]:
    """The adapters among ``names`` that ``layer`` holds; with ``pending``, those not merged yet."""
    held = layer._adapter_names()
    chosen = [name for name in names if name in held]
    return layer._pending_merges(chosen) if pending else chosen
