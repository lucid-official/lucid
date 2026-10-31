"""What the LyCORIS family (LoHa, LoKr) shares: config fields, layer and tuner.

A LyCORIS adapter learns a weight-shaped update :math:`\\Delta W` through a
structured factorisation (a Hadamard product of low-rank products, a
Kronecker product) and applies it to the input as the base layer would
(``lucid.peft.tuners._kinds``).  Everything but the factorisation is the
same for the two, so a method is a subclass of each of the three bases here
and supplies only:

* the config — its own fields on top of ``_LycorisConfig``'s;
* the layer — ``_factor_shapes`` (what to allocate for a module of a given
  geometry), ``_initialise`` and ``_scaled_delta`` (the update);
* the tuner — which config and layer family it is.

Rank, alpha, rank dropout, module dropout, the forward pass, merging and
the per-adapter bookkeeping are here once.
"""

import math
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Final, NamedTuple, Self, overload, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import (
    PeftConfig,
    _Check,
    _check_bool,
    _check_dropout,
    _check_number,
    _check_rank,
    _check_regex,
    _keyed,
    _pattern_key_regex,
    _TargetedConfig,
)
from lucid.peft._targeting import _pattern_value
from lucid.peft.tuners._base import (
    _BaseTuner,
    _NO_PARAMETERS,
    _BaseTunerLayer,
    _check_adapter_name,
    _parameter_at,
    _Target,
)
from lucid.peft.tuners._kinds import _Geometry, _Kind

_WEIGHT: Final = "base_layer.weight"
#: Below this a mean of kept rows counts as "none kept": scaling by it would not make a number.
_TINY: Final = 1e-12


@dataclass(kw_only=True)
class _LycorisConfig(_TargetedConfig):
    r"""The fields every LyCORIS method has, on top of the shared targeting fields.

    Parameters
    ----------
    r : int, default=8
        Rank of the factorisation.
    alpha : float, default=8
        :math:`\alpha`; the update is scaled by :math:`\alpha / r`.
    rank_dropout : float, default=0.0
        During training, the probability of zeroing an output unit's row of
        the update, in ``[0, 1)``.
    module_dropout : float, default=0.0
        During training, the probability of skipping the adapter for a
        whole forward pass, in ``[0, 1)``.
    use_effective_conv2d : bool, default=False
        Factor a convolution's kernel with a small spatial core ("Proposition
        3" of FedPara) instead of flattening it; ignored for ``nn.Linear``
        and for kernels of size 1.
    rank_pattern : dict of str to int, optional
        Per-module rank overrides, keyed by name or regex suffix.
    alpha_pattern : dict of str to float, optional
        Per-module :math:`\alpha` overrides, keyed the same way.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "alpha": _check_number,
        "rank_dropout": _check_dropout,
        "module_dropout": _check_dropout,
        "use_effective_conv2d": _check_bool,
        "rank_pattern": _keyed(_check_rank),
        "alpha_pattern": _keyed(_check_number),
    }

    r: int = 8
    alpha: float = 8
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    use_effective_conv2d: bool = False
    rank_pattern: dict[str, int] = field(default_factory=dict)
    alpha_pattern: dict[str, float] = field(default_factory=dict)

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        for name, keys in (
            ("rank_pattern", self.rank_pattern),
            ("alpha_pattern", self.alpha_pattern),
        ):
            for key in keys:
                _check_regex(f"{name}[{key!r}]", _pattern_key_regex(key))


class _LycorisSpec(NamedTuple):
    """What a layer remembers of one adapter besides its parameters.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes
    it: the scale is baked into the graph.
    """

    r: int
    alpha: float
    scaling: float
    rank_dropout: float
    module_dropout: float
    rank_dropout_scale: bool


class _SpecField[T]:
    """A read-only view of one field of every adapter's spec: ``{adapter name: value}``.

    The field is the attribute's own name, so ``scaling = _SpecField[float](doc)`` reads
    ``spec.scaling`` of each adapter in ``layer._specs``.
    """

    def __init__(self, doc: str) -> None:
        self.__doc__ = doc
        self._field = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._field = name

    @overload
    def __get__(self, layer: None, owner: type) -> Self: ...

    @overload
    def __get__(self, layer: _LycorisLayer, owner: type) -> Mapping[str, T]: ...

    def __get__(self, layer: _LycorisLayer | None, owner: type) -> Self | Mapping[str, T]:
        if layer is None:
            return self
        return MappingProxyType({n: getattr(spec, self._field) for n, spec in layer._specs.items()})


class _LycorisLayer(_BaseTunerLayer):
    r"""A frozen base module with a structured, weight-shaped update :math:`\Delta W` applied beside it.

    The forward pass is :math:`f(x) + \sum_a g_a(x)`, :math:`g_a` being the
    adapter's :math:`\Delta W_a` applied to :math:`x` as the base layer would
    apply a weight (the bias is not in it).  ``merge`` adds
    :math:`\Delta W` to the base weight in place and ``unmerge`` subtracts it.
    Per adapter a layer keeps the rank, alpha, scale and dropouts (read-only
    mappings ``r``, ``alpha``, ``scaling``, ``rank_dropout``,
    ``module_dropout``, ``rank_dropout_scale``) and the method's factors in
    its ``ParameterDict`` tables (``adapter_layer_names``); an adapter need not
    have a factor in every table.

    Parameters
    ----------
    base_layer : nn.Module
        The module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : _LycorisConfig
        The method's config; only the fields the layer needs are read.
    r : int, default=0
        Rank.  The default is the reference's and is refused: pass a positive rank.
    alpha : float, default=0.0
        Numerator of the scale.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``r`` is not positive, or ``base_layer`` builds parameters on its first input.
    """

    #: The prefix of the layer's ``repr`` (``"loha."``).
    _repr_prefix: ClassVar[str] = ""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: _LycorisConfig,
        r: int = 0,
        alpha: float = 0.0,
    ) -> None:
        super().__init__(base_layer)
        self._specs: dict[str, _LycorisSpec] = {}
        for table in self.adapter_layer_names:
            setattr(self, table, nn.ParameterDict())
        self.update_layer(adapter_name, r, alpha, config)

    # ── what a method supplies ────────────────────────────────────────────

    @abstractmethod
    def _factor_shapes(
        self, r: int, geometry: _Geometry, config: _LycorisConfig, effective: bool
    ) -> dict[str, tuple[int, ...]]:
        """The shape of each factor of one adapter, by table name; ``effective`` says to use a spatial core."""

    @abstractmethod
    def _initialise(self, adapter: str, init_weights: object) -> None:
        """Write the starting values of ``adapter``'s factors."""

    @abstractmethod
    def _scaled_delta(self, adapter: str) -> Tensor:
        r""":math:`s\,\Delta W` of ``adapter`` from its factors, any shape that reshapes to the base weight's."""

    def _kaiming(self, adapter: str, *tables: nn.ParameterDict) -> None:
        """Kaiming-uniform (``a = sqrt(5)``, what ``nn.Linear`` uses) the factors ``adapter`` has in ``tables``."""
        for table in tables:
            if adapter in table:
                init.kaiming_uniform_(_parameter_at(table, adapter), a=math.sqrt(5))

    def _zero(self, adapter: str, *tables: nn.ParameterDict) -> None:
        """Zero the factors ``adapter`` has in ``tables``."""
        for table in tables:
            if adapter in table:
                init.zeros_(_parameter_at(table, adapter))

    @staticmethod
    def _rank_dropout_scale(config: _LycorisConfig) -> bool:
        """Whether the rows kept under rank dropout are scaled up to keep the mean."""
        del config
        return True

    # ── per-adapter state ─────────────────────────────────────────────────

    #: Read-only ``{adapter: value}`` views of the spec fields, by the reference's spelling.
    r = _SpecField[int]("Rank of each adapter, by name (read-only).")
    alpha = _SpecField[float](
        "The numerator :math:`\\alpha` of each adapter's scale, by name (read-only)."
    )
    scaling = _SpecField[float](
        "The scale :math:`\\alpha / r` of each adapter, by name (read-only)."
    )
    rank_dropout = _SpecField[float](
        "Rank-dropout probability of each adapter, by name (read-only)."
    )
    module_dropout = _SpecField[float](
        "Module-dropout probability of each adapter, by name (read-only)."
    )
    rank_dropout_scale = _SpecField[bool](
        "Whether each adapter scales the rows it keeps under rank dropout, by name (read-only)."
    )

    @override
    def _adapter_names(self) -> list[str]:
        return list(self._specs)

    @property
    def _geometry(self) -> _Geometry:
        return _Kind.of(type(self.base_layer)).geometry(self.base_layer)

    @override
    def __repr__(self) -> str:
        return type(self)._repr_prefix + super().__repr__()

    @override
    def extra_repr(self) -> str:
        held = [name for name in self.active_adapters if name in self._specs]
        ranks = {name: self._specs[name].r for name in held}
        detail = (
            f"r={ranks[held[0]]}" if isinstance(self.active_adapter, str) and held else f"r={ranks}"
        )
        return (
            f"adapter={self.active_adapter!r}, {detail}, "
            f"merged={self.merged}, disabled={self.disable_adapters}"
        )

    # ── creating and removing an adapter ─────────────────────────────────

    def _effective(self, geometry: _Geometry, config: _LycorisConfig) -> bool:
        """Whether this module is factored with a spatial core: asked for, and it has a kernel bigger than 1."""
        return config.use_effective_conv2d and geometry.has_kernel and not geometry.pointwise

    def update_layer(self, adapter_name: str, r: int, alpha: float, config: _LycorisConfig) -> None:
        r"""Create adapter ``adapter_name``: its factors, initialised, and its scale :math:`\alpha / r`.

        The factors live on the base weight's device, in its dtype.  A layer
        that holds an adapter already keeps applying the adapters it applied:
        the new one is inactive until ``set_adapter`` names it.  A layer
        with none starts applying this one.

        Parameters
        ----------
        adapter_name : str
            Key of the adapter in each table.
        r : int
            Rank; positive.
        alpha : float
            Numerator of the scale.
        config : _LycorisConfig
            The method's config: dropouts, ``init_weights``, ``use_effective_conv2d``
            and whatever the method's factorisation reads.

        Raises
        ------
        ValueError
            The layer already holds an adapter of this name, or ``r`` is not positive.
        """
        adapter_name = _check_adapter_name(adapter_name)
        r = _check_rank("r", r)
        alpha = _check_number("alpha", alpha)
        if adapter_name in self._specs:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        shapes = self._factor_shapes(
            r, self._geometry, config, self._effective(self._geometry, config)
        )
        anchor = self.weight
        # Built first, entered together: an allocation that fails leaves no half adapter.
        factors = {
            name: Parameter(lucid.zeros(*shape, device=anchor.device, dtype=anchor.dtype))
            for name, shape in shapes.items()
        }
        self._specs[adapter_name] = _LycorisSpec(
            r,
            alpha,
            alpha / r,
            config.rank_dropout,
            config.module_dropout,
            self._rank_dropout_scale(config),
        )
        for name, factor in factors.items():
            getattr(self, name)[adapter_name] = factor
        self._initialise(adapter_name, getattr(config, "init_weights", True))
        if not self.active_adapters:
            self.active_adapter = adapter_name

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        self._specs.pop(adapter_name, None)

    # ── the update ────────────────────────────────────────────────────────

    def _row_mask(self, adapter: str, weight: Tensor) -> Tensor:
        """A random per-output-row keep mask for ``weight``, shaped to broadcast against it."""
        spec = self._specs[adapter]
        keep = (lucid.rand(weight.shape[0], device=weight.device) > spec.rank_dropout).to(
            weight.dtype
        )
        if spec.rank_dropout_scale:
            keep = keep / keep.mean().clamp(min=_TINY)
        return keep.reshape(-1, *([1] * (weight.ndim - 1)))

    @override
    def get_delta_weight(self, adapter: str, *, apply_rank_dropout: bool = False) -> Tensor:
        r"""Return :math:`s\,\Delta W`, the update ``adapter`` adds to the base weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.
        apply_rank_dropout : bool, default=False
            Zero a random subset of output rows first, as training does.

        Returns
        -------
        Tensor
            Shaped like the base weight, in its dtype.
        """
        delta = self._scaled_delta(adapter).reshape(*self.weight.shape)
        return delta * self._row_mask(adapter, delta) if apply_rank_dropout else delta

    def _skips(self, adapter: str) -> bool:
        """Whether module dropout leaves ``adapter`` out of this forward pass."""
        probability = self._specs[adapter].module_dropout
        return self.training and probability > 0 and float(lucid.rand(1).item()) <= probability

    @override
    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.base_layer(x)
        if self.disable_adapters or self.merged:
            return result
        kind = _Kind.of(type(self.base_layer))
        out = result
        for adapter in self.active_adapters:
            if adapter not in self._specs or self._skips(adapter):
                continue
            delta = self.get_delta_weight(
                adapter,
                apply_rank_dropout=self.training and bool(self._specs[adapter].rank_dropout),
            )
            out = out + kind.apply(self.base_layer, x, delta)
        return out if out.dtype == result.dtype else out.to(result.dtype)

    # ── merging ───────────────────────────────────────────────────────────

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        return {_WEIGHT: self.weight}

    @override
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        return {_WEIGHT: self.weight + self.get_delta_weight(adapter)}

    @override
    def _fold(self, adapter: str) -> None:
        self.weight.copy_(self.weight + self.get_delta_weight(adapter))

    @override
    def _unfold(self) -> None:
        for adapter in reversed(self.merged_adapters):
            self.weight.copy_(self.weight - self.get_delta_weight(adapter))


def _as_lycoris(config: PeftConfig) -> _LycorisConfig:
    # ``_BaseTuner`` has checked it against ``_config_class``.
    assert isinstance(config, _LycorisConfig)
    return config


class _LycorisTuner(_BaseTuner):
    """The tuner of a LyCORIS method: layers built with the rank and alpha the config gives each module."""

    _default_parameters: ClassVar[Mapping[str, frozenset[str]]] = _NO_PARAMETERS

    @override
    @classmethod
    def _create_new_module(
        cls, config: _TargetedConfig, adapter_name: str, target: _Target
    ) -> nn.Module:
        layer = target.layer
        assert layer is not None and issubclass(layer, _LycorisLayer)
        r, alpha = cls._rank_and_alpha(_as_lycoris(config), target.full_name)
        return layer(target.module, adapter_name, _as_lycoris(config), r=r, alpha=alpha)

    @override
    @classmethod
    def _add_to_layer(cls, config: _TargetedConfig, adapter_name: str, target: _Target) -> None:
        layer = target.module
        assert isinstance(layer, _LycorisLayer)
        r, alpha = cls._rank_and_alpha(_as_lycoris(config), target.full_name)
        layer.update_layer(adapter_name, r, alpha, _as_lycoris(config))

    @staticmethod
    def _rank_and_alpha(config: _LycorisConfig, key: str) -> tuple[int, float]:
        return (
            _pattern_value(config.rank_pattern, key, config.r),
            _pattern_value(config.alpha_pattern, key, config.alpha),
        )

    @override
    @classmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        del model, config  # a LyCORIS adapter trains its factors only
        return []
