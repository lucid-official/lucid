"""The LoRA layers: a frozen base module with a low-rank update beside it.

``LoraLayer`` is the family root.  It keeps what every LoRA layer keeps per
adapter (rank, alpha, scale, dropout, the factor tables), creates and
initialises adapters (``update_layer``, with an ``_Init`` strategy for the
initialisation), and merges them into the base weights.  What differs
between layer types is a handful of hooks: the shape of the factors
(``_build_factors``), the update an adapter makes (``get_delta_weight``
and, for the layers that add it to the output, ``_lowrank``) and which base
tensors a merge writes (``_merge_targets``).  ``_Residual`` is the layers
that add the update to the base output (``Linear``, the convolutions,
``Embedding``) and so can host a variant of the update (``_Variant``:
plain LoRA or DoRA); the ones that instead shift the base weights for the
duration of its forward are in ``_weight_space``.
"""

import inspect
import math
import warnings
from abc import abstractmethod
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import ClassVar, Final, NamedTuple, override

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.parameter import Parameter
from lucid.peft._config import _check_dropout, _check_number, _check_rank
from lucid.peft.tuners._base import _BaseTunerLayer, _check_adapter_name, _entry_parameters
from lucid.peft.tuners.lora._inits import _check_init, _Init
from lucid.peft.tuners.lora._variants import MAGNITUDE_TABLE, _Magnitude, _rows_view, _Variant
from lucid.peft.tuners.lora._weight_site import _WeightSite

_HALF: Final = (lucid.float16, lucid.bfloat16)


class _AdapterSpec(NamedTuple):
    """What a layer remembers of one adapter besides its modules: rank, alpha, scale, bias and variant.

    A tuple, so the scalar state ``lucid.compile`` keys a trace on includes
    it: the scale, and which variant computes the update, are baked into the graph.
    """

    r: int
    lora_alpha: float
    scaling: float
    lora_bias: bool = False
    variant: str = "lora"


type _Factors = tuple[nn.Module | Parameter, ...]

#: What a base layer's ``forward`` takes beside its input (a transposed convolution's ``output_size``),
#: by parameter name: handed to the base layer and to the low-rank branch alike.
type _Extras = Mapping[str, object]
_NO_EXTRAS: Final[_Extras] = MappingProxyType({})


def _extra_parameters(forward: Callable[..., object]) -> tuple[str, ...]:
    """The names a ``forward`` takes beside ``self`` and its input, in order."""
    return tuple(inspect.signature(forward).parameters)[2:]


class _Place(NamedTuple):
    """Where and in what dtype an adapter's factors are allocated: the base tensor's, widened if half.

    ``bias`` is whether the adapter's second factor carries a bias (``lora_bias``).
    """

    device: DeviceLike
    dtype: DTypeLike
    bias: bool = False

    def linear(self, n_in: int, n_out: int, *, bias: bool = False) -> nn.Linear:
        """An ``nn.Linear`` here, bias-free unless ``bias``."""
        return nn.Linear(n_in, n_out, bias=bias, device=self.device, dtype=self.dtype)


def _enter(
    table: nn.ModuleDict[nn.Module] | nn.ParameterDict, name: str, factor: nn.Module | Parameter
) -> None:
    """Put ``factor`` in ``table`` under ``name``: a module in a ``ModuleDict``, a parameter in a ``ParameterDict``."""
    if isinstance(factor, Parameter):
        if not isinstance(table, nn.ParameterDict):
            raise TypeError(f"a {type(table).__name__} cannot hold the parameter {name!r}")
        table[name] = factor
    elif isinstance(factor, nn.Module):
        if not isinstance(table, nn.ModuleDict):
            raise TypeError(f"a {type(table).__name__} cannot hold the module {name!r}")
        table[name] = factor
    else:
        raise TypeError(f"{type(factor).__name__} is not an adapter factor")


def _weight_of(factor: nn.Module | Parameter | None) -> Parameter:
    """The weight of a table entry: a module's ``weight`` (its bias, if any, is not a factor), or the bare parameter."""
    if isinstance(factor, Parameter):
        return factor
    weight = getattr(factor, "weight", None)
    if isinstance(weight, Parameter):
        return weight
    raise TypeError(f"{type(factor).__name__} holds no weight to use as an adapter factor")


class LoraLayer(_BaseTunerLayer):
    r"""Adapter bookkeeping shared by the LoRA layers.

    Holds the wrapped layer as ``base_layer`` — the very module, so its
    parameters stay the same ``Parameter`` objects (an optimizer, a tied
    head and Core ML's identity-keyed weight sharing all keep seeing
    them) — and, per adapter name, the rank ``r``, ``lora_alpha``,
    ``scaling``, ``lora_bias`` and ``use_dora`` and the factor tables
    (``lora_A`` / ``lora_B`` and the ``lora_dropout`` modules in
    ``ModuleDict``\ s; ``lora_embedding_A`` / ``lora_embedding_B`` in
    ``ParameterDict``\ s for an embedding; DoRA's ``lora_magnitude_vector``),
    so a state dict reads ``....lora_A.default.weight`` as the reference's
    does.  A layer holds any number of adapters, one entry per name in
    those tables and in ``_specs`` (read as the ``r``, ``lora_alpha``,
    ``scaling``, ``lora_bias`` and ``use_dora`` mappings), and the forward
    pass applies the ones named in ``active_adapter`` that it holds.
    Merging folds an update into the base weights in place; each LoRA layer
    type supplies the update (``get_delta_weight``) and the forward pass.  A
    DoRA adapter folds in as the row-rescaled weight
    :math:`m\,(W + sBA)/\lVert W + sBA\rVert`; an adapter with ``lora_bias``
    also adds its scaled bias to the base bias.

    What the forward pass branches on — ``active_adapter``,
    ``disable_adapters``, ``merged`` — are plain public attributes, not
    properties as in the reference: ``lucid.compile`` keys an executable
    on each module's scalar attributes, so state kept anywhere else would
    let a compiled model replay the branch of a state it has left
    (LCD-703).  They change only through ``update_layer``,
    ``set_adapter``, ``delete_adapter``, ``enable_adapters``, ``merge``
    and ``unmerge``.

    Parameters
    ----------
    base_layer : nn.Module
        The layer to adapt.
    adapter_name : str
        Name of the adapter created at once (``update_layer``).
    r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
        The adapter's hyper-parameters, as ``update_layer`` takes them.

    Raises
    ------
    TypeError
        ``base_layer`` is not the module type the layer class adapts.
    ValueError
        ``base_layer`` has parameters it builds on its first input.
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("lora_A", "lora_B")
    #: The variants of the update (``_Variant`` names) this layer type can host.
    _variant_names: ClassVar[frozenset[str]] = frozenset({"lora"})
    #: Tables a variant keeps state in (a subset of ``adapter_layer_names``, not factor tables).
    _variant_tables: ClassVar[tuple[str, ...]] = ()
    #: Whether the base weight is a dense ``(out, in)`` matrix an initialisation may factorise.
    _dense_weight: ClassVar[bool] = False
    #: Whether the second factor may carry a bias (``lora_bias``).
    _lora_bias_ok: ClassVar[bool] = False

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: float = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        *,
        lora_bias: bool = False,
    ) -> None:
        super().__init__(base_layer)
        self._specs: dict[str, _AdapterSpec] = {}
        self.lora_dropout: nn.ModuleDict[nn.Dropout | nn.Identity] = nn.ModuleDict()
        self._setup()
        for table in self._variant_tables:
            setattr(self, table, nn.ModuleDict())
        self.update_layer(
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            use_rslora,
            use_dora,
            lora_bias=lora_bias,
        )

    # ── what a layer type supplies ────────────────────────────────────────

    @abstractmethod
    def _setup(self) -> None:
        """Create the empty factor tables and read the base layer's sizes the factors are shaped by."""

    @abstractmethod
    def _anchor(self) -> Tensor:
        """The base tensor whose device and dtype an adapter's factors follow."""

    def _check_rank_fits(self, r: int) -> None:
        """Raise ``ValueError`` if rank ``r`` cannot be built on this base layer."""

    def _check_dropout(self, p: float) -> None:
        """Refuse, or warn of, a dropout probability ``p`` this layer type cannot apply."""

    @abstractmethod
    def _build_factors(self, r: int, place: _Place) -> _Factors:
        """One adapter's new factors, one per factor table (``adapter_layer_names`` less the variant tables), in order."""

    def _adapter_built(self, adapter: str) -> None:
        """Called once ``adapter``'s factors are initialised; a layer that keeps variant state creates it here."""

    def _factor_tables(self) -> list[nn.ModuleDict[nn.Module] | nn.ParameterDict]:
        """The factor tables (``adapter_layer_names`` less the variant tables), in order."""
        return [
            table
            for name in self.adapter_layer_names
            if name not in self._variant_tables
            and isinstance(table := getattr(self, name), (nn.ModuleDict, nn.ParameterDict))
        ]

    def _factor_pairs(self, adapter: str) -> list[tuple[Parameter, Parameter]]:
        r"""``(A, B)`` of each low-rank pair ``adapter`` holds: the update is :math:`s\,BA`.

        The factor tables come in ``(A, B)`` pairs, each entry one weight.
        """
        tables = self._factor_tables()
        return [
            (_weight_of(lora_a[adapter]), _weight_of(lora_b[adapter]))
            for lora_a, lora_b in zip(tables[0::2], tables[1::2], strict=True)
        ]

    # ── per-adapter state ─────────────────────────────────────────────────

    @property
    def r(self) -> Mapping[str, int]:
        """Rank of each adapter, by name (read-only)."""
        return MappingProxyType({name: spec.r for name, spec in self._specs.items()})

    @property
    def lora_alpha(self) -> Mapping[str, float]:
        """:math:`\\alpha` of each adapter, by name (read-only)."""
        return MappingProxyType({name: spec.lora_alpha for name, spec in self._specs.items()})

    @property
    def scaling(self) -> Mapping[str, float]:
        """The scale :math:`s` each adapter's update is multiplied by, by name (read-only)."""
        return MappingProxyType({name: spec.scaling for name, spec in self._specs.items()})

    @property
    def lora_bias(self) -> Mapping[str, bool]:
        """Whether each adapter's second factor carries a bias, by name (read-only)."""
        return MappingProxyType({name: spec.lora_bias for name, spec in self._specs.items()})

    @property
    def use_dora(self) -> Mapping[str, bool]:
        """Whether each adapter is a DoRA adapter, by name (read-only)."""
        return MappingProxyType(
            {name: spec.variant == "dora" for name, spec in self._specs.items()}
        )

    def _base_path(self) -> str:
        """The dotted path from this layer to the module holding the base parameters."""
        return "base_layer"

    @override
    def __repr__(self) -> str:
        return "lora." + super().__repr__()

    @override
    def extra_repr(self) -> str:
        held = [name for name in self.active_adapters if name in self._specs]
        shown = {name: self._specs[name] for name in held}
        if isinstance(self.active_adapter, str) and held:
            spec = shown[held[0]]
            detail = f"r={spec.r}, lora_alpha={spec.lora_alpha}"
        else:
            detail = f"r={ {name: spec.r for name, spec in shown.items()} }"
        return (
            f"adapter={self.active_adapter!r}, {detail}, "
            f"merged={self.merged}, disabled={self.disable_adapters}"
        )

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        init_lora_weights: bool | str,
        use_rslora: bool,
        use_dora: bool = False,
        *,
        lora_bias: bool = False,
    ) -> None:
        r"""Create adapter ``adapter_name``: its factors, dropout and scale.

        The adapter lives on the base weight's device.  Its dtype is the
        base's, except that a float16 / bfloat16 base gets a float32
        adapter — the reference's default, which keeps the low-rank update
        from underflowing in training.  A layer that holds an adapter already
        keeps applying the adapters it applied: the new one is inactive until
        ``set_adapter`` names it.  A layer with none starts applying this one.

        Everything the layer cannot do is refused before anything is
        allocated or any base weight is touched.

        Parameters
        ----------
        adapter_name : str
            Key of the adapter in each table.
        r : int
            Rank; positive.
        lora_alpha : float
            Numerator of the scale.
        lora_dropout : float
            Dropout on the update's input, in ``[0, 1)``.
        init_lora_weights : bool or str
            ``True``, ``False``, ``"gaussian"``, ``"pissa"``,
            ``"pissa_niter_<k>"`` or ``"olora"`` (see ``LoraConfig``).
            PiSSA and OLoRA rewrite the base weight (the residual), in place.
        use_rslora : bool
            Scale by :math:`\alpha/\sqrt r` instead of :math:`\alpha/r`.
        use_dora : bool, default=False
            Weight-decomposed update (DoRA): a magnitude vector per output
            unit beside the factors.
        lora_bias : bool, default=False
            Give the second factor a bias.

        Raises
        ------
        NotImplementedError
            An initialisation lucid.peft does not have.
        ValueError
            The layer already holds an adapter of this name, the rank does
            not fit the base layer, or the layer type cannot host the
            variant, the bias or the initialisation asked for.
        """
        adapter_name = _check_adapter_name(adapter_name)
        r = _check_rank("r", r)
        lora_alpha = _check_number("lora_alpha", lora_alpha)
        init_strategy = _Init.of(_check_init("init_lora_weights", init_lora_weights))
        p = _check_dropout("lora_dropout", lora_dropout)
        variant = _Variant.requested(use_dora=use_dora)
        if adapter_name in self._specs:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        variant.check(self)
        self._check_lora_bias(lora_bias)
        init_strategy.check(self, r)
        self._check_rank_fits(r)
        self._check_dropout(p)
        anchor = self._anchor()
        dtype = lucid.float32 if anchor.dtype in _HALF else anchor.dtype
        # Built first, entered together: an allocation that fails leaves no half adapter.
        factors = self._build_factors(r, _Place(anchor.device, dtype, lora_bias))
        self._specs[adapter_name] = _AdapterSpec(
            r, lora_alpha, lora_alpha / (math.sqrt(r) if use_rslora else r), lora_bias, variant.name
        )
        self.lora_dropout[adapter_name] = nn.Dropout(p=p) if p > 0 else nn.Identity()
        for table, factor in zip(self._factor_tables(), factors, strict=True):
            _enter(table, adapter_name, factor)
        init_strategy.apply(self, adapter_name)
        self._adapter_built(adapter_name)
        if not self.active_adapters:
            self.active_adapter = adapter_name

    def _check_lora_bias(self, lora_bias: bool) -> None:
        """Refuse ``lora_bias`` on a layer type that cannot carry it; warn if a merge could not hold it."""
        if not lora_bias:
            return
        if not self._lora_bias_ok:
            raise ValueError(f"lora_bias=True is not supported for {type(self).__name__}.")
        if self.bias is None:
            warnings.warn(
                f"`lora_bias=True` was passed but the targeted layer of type "
                f"{type(self.get_base_layer()).__name__} has no bias. This means that merging "
                "LoRA weights won't be possible.",
                stacklevel=3,
            )

    def _reset_factors(self, adapter: str, gaussian: bool) -> None:
        r"""The default initialisation of every factor pair of ``adapter``: :math:`B = 0`, :math:`A` random.

        :math:`A` is Kaiming-uniform with ``a = sqrt(5)`` (what ``nn.Linear``
        uses), or :math:`\mathcal N(0, 1/r^2)` when ``gaussian``.
        """
        for lora_a, lora_b in self._factor_pairs(adapter):
            if gaussian:
                init.normal_(lora_a, std=1 / self._specs[adapter].r)
            else:
                init.kaiming_uniform_(lora_a, a=math.sqrt(5))
            init.zeros_(lora_b)

    def reset_lora_parameters(self, adapter_name: str, init_lora_weights: bool | str) -> None:
        r"""Initialise adapter ``adapter_name`` the way ``init_lora_weights`` asks.

        ``True``: every :math:`A` Kaiming-uniform with ``a = sqrt(5)`` (what
        ``nn.Linear`` uses) and every :math:`B = 0`, so the update starts at
        zero.  ``"gaussian"``: :math:`A \sim \mathcal N(0, 1/r^2)`,
        :math:`B = 0`.  ``False``: leave both as the layers made them.
        ``"pissa"``, ``"pissa_niter_<k>"`` and ``"olora"`` factorise the
        base weight and write the residual back into it, in place.

        Parameters
        ----------
        adapter_name : str
            The adapter to initialise.
        init_lora_weights : bool or str
            The initialisation, as ``update_layer`` takes it.
        """
        _Init.of(_check_init("init_lora_weights", init_lora_weights)).apply(self, adapter_name)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        super().delete_adapter(adapter_name)
        self._specs.pop(adapter_name, None)
        # Not in ``adapter_layer_names`` — it holds no parameters to train or save.
        if adapter_name in self.lora_dropout:
            del self.lora_dropout[adapter_name]

    # ── merging ───────────────────────────────────────────────────────────

    @override
    def _pending_merges(self, adapter_names: list[str] | None) -> list[str]:
        pending = super()._pending_merges(adapter_names)
        if self.bias is None and any(self._specs[name].lora_bias for name in pending):
            raise RuntimeError(
                "Impossible to merge LoRA with `lora_bias=True` because the base layer has no bias."
            )
        return pending


class _Plan(NamedTuple):
    """What folding one adapter into the base does: the change of each merge target, and the DoRA row gain."""

    shifts: dict[str, Tensor]
    gain: Tensor | None


_WEIGHT: Final = "base_layer.weight"
_BIAS: Final = "base_layer.bias"


def _added(param: Tensor, shift: Tensor) -> Tensor:
    """``param + shift`` summed in the wider dtype and rounded once, in ``param``'s dtype."""
    value = param + shift
    return value if value.dtype == param.dtype else value.to(param.dtype)


class _Residual(LoraLayer):
    r"""A LoRA layer whose update is added to the base layer's output: :math:`y = f(x) + \sum_a u_a(x)`.

    A subclass supplies the low-rank update from the layer's input
    (``_lowrank``) and the same update as a change of the base weight
    (``get_delta_weight``), which is what merging writes.  What an adapter
    makes of that update — plain, or DoRA's magnitude-rescaled one — is its
    variant's (``_Variant``): the forward pass and the merge delegate to it.

    The hooks below are how a variant reads a layer without knowing its type:
    the weight as one row per output unit (``_rows``), a per-unit vector
    shaped to broadcast against the output (``_output_vector``) and the
    output without its bias (``_without_bias``).
    """

    _variant_names = frozenset({"lora", "dora"})
    _variant_tables = (MAGNITUDE_TABLE,)
    lora_magnitude_vector: nn.ModuleDict[_Magnitude]

    @override
    def __init_subclass__(
        cls, *, base_type: type[nn.Module] | None = None, family: bool = False, **kwargs: object
    ) -> None:
        if base_type is not None:
            # An adapted layer takes whatever its base's forward takes: a base that gains a
            # parameter fails here, at the class (before it registers), not at a user's call.
            missing = set(_extra_parameters(base_type.forward)) - set(
                _extra_parameters(cls.forward)
            )
            if missing:
                raise TypeError(
                    f"{cls.__name__}.forward must take {sorted(missing)}, as "
                    f"{base_type.__name__}.forward does"
                )
        super().__init_subclass__(base_type=base_type, family=family, **kwargs)

    @abstractmethod
    def _lowrank(self, adapter: str, dropped: Tensor, extras: _Extras) -> Tensor:
        """The scaled low-rank update ``adapter`` makes from the (dropped-out) input.

        ``extras`` are the base layer's other forward arguments, which the update must follow.
        """

    def _extras_of(self, args: tuple[object, ...], kwargs: Mapping[str, object]) -> _Extras:
        """The extras of a call ``self(*args, **kwargs)``, by name (defaults filled in)."""
        bound = inspect.signature(self.forward).bind(*args, **kwargs)
        bound.apply_defaults()
        return {name: bound.arguments[name] for name in _extra_parameters(type(self).forward)}

    def _dropped(self, adapter: str, x: Tensor) -> Tensor:
        """``x`` as the update sees it: through the adapter's dropout."""
        dropped: Tensor = self.lora_dropout[adapter](x)
        return dropped

    def _applies_dropout(self, adapter: str) -> bool:
        """Whether ``_dropped`` changes its input now (a dropout module, in training)."""
        return self.training and not isinstance(self.lora_dropout[adapter], nn.Identity)

    def _rows(self, weight: Tensor) -> Tensor:
        """``weight`` (or an update shaped like it) with one row per output unit; its own inverse."""
        return weight

    def _output_vector(self, vector: Tensor) -> Tensor:
        """A per-output-unit ``vector`` shaped to broadcast against this layer's output."""
        return vector

    def _without_bias(self, result: Tensor) -> Tensor:
        """The base output ``result`` less its bias."""
        bias = self.bias
        return result if bias is None else result - self._output_vector(bias)

    def _check_weight_norm(self) -> None:
        """Raise if the per-unit weight norm DoRA needs cannot be formed for this base layer."""

    def _bias_delta(self, adapter: str) -> Tensor | None:
        """What merging ``adapter`` adds to the base bias, or ``None``."""
        del adapter
        return None

    def _variant(self, adapter: str) -> _Variant:
        return _Variant.named(self._specs[adapter].variant)

    def _site(self) -> _WeightSite:
        """Where the base layer keeps its weight: the one thing a merge reads and writes it through."""
        return _WeightSite.of(self.base_layer)

    def _base_weight(self) -> Tensor:
        """The weight the base layer computes with now (a weight norm's derived one included)."""
        return self._site().read()

    @override
    def _adapter_built(self, adapter: str) -> None:
        self._variant(adapter).attach(self, adapter)

    def _original_rows(self, rows: Tensor) -> Tensor:
        r"""``rows``, the weight now, with every merged adapter taken back out: the weight before any merge.

        A merged weight is :math:`c\,W + \sum_a g_a\,\Delta_a` with
        :math:`c = 1 + \sum_a (g_a - 1)` (:math:`g_a = 1` for a plain
        adapter), so :math:`W` follows from the gains the merges
        remembered.
        """
        original = rows
        spread: Tensor | None = None
        for adapter in reversed(self.merged_adapters):
            delta = self._rows(self.get_delta_weight(adapter))
            gain = self._variant(adapter).recall(self, adapter)
            if gain is None:
                original = original - delta
                continue
            original = original - _rows_view(gain, original) * delta
            spread = gain - 1 if spread is None else spread + (gain - 1)
        return original if spread is None else original / _rows_view(1 + spread, original)

    def _plan(self, adapter: str) -> _Plan:
        change, gain = self._variant(adapter).shift(
            self,
            adapter,
            self._rows(self._base_weight()),
            self._rows(self.get_delta_weight(adapter)),
        )
        shifts = {_WEIGHT: self._rows(change)}
        bias = self._bias_delta(adapter)
        if bias is not None:
            shifts[_BIAS] = bias
        return _Plan(shifts, gain)

    @override
    def _merge_targets(self) -> dict[str, Parameter]:
        targets = {f"base_layer.{name}": p for name, p in self._site().parameters().items()}
        bias = self.bias
        if bias is not None and any(spec.lora_bias for spec in self._specs.values()):
            targets[_BIAS] = bias
        return targets

    def _bias_target(self) -> Parameter:
        bias = self.bias
        if bias is None:
            raise RuntimeError("the base layer has no bias to merge into")
        return bias

    @override
    def _merged_values(self, adapter: str) -> dict[str, Tensor]:
        shifts = self._plan(adapter).shifts
        values = {_WEIGHT: _added(self._base_weight(), shifts[_WEIGHT])}
        if _BIAS in shifts:
            values[_BIAS] = _added(self._bias_target(), shifts[_BIAS])
        return values

    @override
    def _fold(self, adapter: str) -> None:
        plan = self._plan(adapter)
        site = self._site()
        site.write(_added(site.read(), plan.shifts[_WEIGHT]))
        if _BIAS in plan.shifts:
            bias = self._bias_target()
            bias.copy_(_added(bias, plan.shifts[_BIAS]))
        self._variant(adapter).remember(self, adapter, plan.gain)

    @override
    def _unfold(self) -> None:
        site = self._site()
        weight = site.read()
        original = self._rows(self._original_rows(self._rows(weight)))
        for adapter in self.merged_adapters:
            bias = self._bias_delta(adapter)
            if bias is not None:
                target = self._bias_target()
                target.copy_(_added(target, -bias))
            self._variant(adapter).remember(self, adapter, None)
        site.write(original if original.dtype == weight.dtype else original.to(weight.dtype))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x, _NO_EXTRAS)

    def _forward(self, x: Tensor, extras: _Extras) -> Tensor:
        """The forward pass: the base layer on ``x`` and ``extras``, plus each active adapter's update."""
        result: Tensor = self.base_layer(x, **extras)
        if self.disable_adapters or self.merged:
            return result
        out = result
        for adapter in self.active_adapters:
            if adapter not in self._specs:
                continue  # an adapter aimed at other layers: this one runs its base for it
            out = out + self._variant(adapter).contribution(self, adapter, x, result, extras)
        return out if out.dtype == result.dtype else out.to(result.dtype)


class _Factored[Factor: nn.Module](_Residual):
    r"""A residual layer whose factors are two modules in a row: :math:`u(x) = s\,B(A(\mathrm{drop}(x)))`.

    ``lora_A`` and ``lora_B`` hold one ``Factor`` per adapter (an ``nn.Linear``,
    a convolution); a subclass builds them and says how they merge.  The
    second factor carries a bias when the adapter has ``lora_bias`` — scaled
    with the rest of the update in the forward pass, and merged, scaled, into
    the base bias.
    """

    adapter_layer_names = ("lora_A", "lora_B", MAGNITUDE_TABLE)
    _lora_bias_ok = True
    lora_A: nn.ModuleDict[Factor]
    lora_B: nn.ModuleDict[Factor]

    @override
    def _lowrank(self, adapter: str, dropped: Tensor, extras: _Extras) -> Tensor:
        lora_a = self.lora_A[adapter]
        dtype = _entry_parameters(lora_a)[0].dtype
        update: Tensor = self.lora_B[adapter](
            lora_a(dropped if dropped.dtype == dtype else dropped.to(dtype), **extras)
        )
        return update * self._specs[adapter].scaling

    def _second_bias(self, adapter: str) -> Parameter | None:
        """The bias of ``adapter``'s second factor (``lora_bias``), or ``None``."""
        bias = getattr(self.lora_B[adapter], "bias", None)
        return bias if isinstance(bias, Parameter) else None

    @override
    def _reset_factors(self, adapter: str, gaussian: bool) -> None:
        super()._reset_factors(adapter, gaussian)
        bias = self._second_bias(adapter)
        if bias is not None:
            init.zeros_(bias)

    @override
    def _bias_delta(self, adapter: str) -> Tensor | None:
        bias = self._second_bias(adapter)
        return None if bias is None else bias * self._specs[adapter].scaling


class Linear(_Factored[nn.Linear], base_type=nn.Linear):
    r"""An ``nn.Linear`` with a LoRA update: :math:`y = W x + b + s\,B A\,\mathrm{drop}(x)`.

    The base layer runs as it is and stays frozen; the update is computed
    in the adapter's dtype and added in the wider of the two dtypes before
    the result is cast back to the base output's dtype.  ``merge`` folds
    :math:`s B A` into :math:`W` in place, so a merged layer costs what
    the base costs; ``unmerge`` takes it out again.

    Parameters
    ----------
    base_layer : nn.Linear
        The layer to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter, the key in ``lora_A`` / ``lora_B``.
    r : int, default=0
        Rank.  The default is the reference's and is refused: pass a
        positive rank.
    lora_alpha : float, default=1
        Numerator of the scale.
    lora_dropout : float, default=0.0
        Dropout on the update's input.
    fan_in_fan_out : bool, default=False
        Must be ``False``: ``nn.Linear`` stores its weight ``(out, in)``.
    is_target_conv_1d_layer : bool, default=False
        Must be ``False``; kept so positional calls line up with the
        reference, whose fused-QKV ``Conv1D`` base v1 does not have.
    init_lora_weights : bool or str, default=True
        See ``LoraConfig``; ``"pissa"`` and ``"olora"`` also rewrite the
        base weight (the residual).
    use_rslora : bool, default=False
        Scale by :math:`\alpha/\sqrt r`.
    use_dora : bool, default=False
        Weight-decomposed update (DoRA): a magnitude vector
        (``lora_magnitude_vector``) beside the factors, so the layer computes
        :math:`m\,(W + s\,BA)/\lVert W + s\,BA\rVert` row by row.
    lora_bias : bool, default=False
        Give ``lora_B`` a bias (``lora_B.<adapter>.bias``); merging folds it,
        scaled, into the base bias.

    Raises
    ------
    ValueError
        ``r`` is not positive, ``fan_in_fan_out`` or
        ``is_target_conv_1d_layer`` is ``True``, or ``init_lora_weights``
        asks for more rank than the weight has.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft.tuners.lora import Linear
    >>> base = nn.Linear(8, 4)
    >>> layer = Linear(base, "default", r=2, lora_alpha=4)
    >>> layer.base_layer is base, layer.scaling["default"]
    (True, 2.0)
    >>> x = lucid.randn(3, 8)
    >>> bool((layer(x) == base(x)).all().item())   # B starts at zero
    True
    """

    _dense_weight = True
    base_layer: nn.Linear

    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        r: int = 0,
        lora_alpha: float = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        *,
        lora_bias: bool = False,
    ) -> None:
        if fan_in_fan_out:
            raise ValueError(
                "fan_in_fan_out=True is for a weight stored (in, out); nn.Linear stores (out, in)"
            )
        if is_target_conv_1d_layer:
            raise ValueError(
                "is_target_conv_1d_layer=True describes a Conv1D base; this layer wraps nn.Linear"
            )
        super().__init__(
            base_layer,
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            use_rslora,
            use_dora,
            lora_bias=lora_bias,
        )

    @override
    def _setup(self) -> None:
        self.in_features = self.base_layer.in_features
        self.out_features = self.base_layer.out_features
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()

    @override
    def _anchor(self) -> Tensor:
        return self.base_layer.weight

    @override
    def _build_factors(self, r: int, place: _Place) -> _Factors:
        return (
            place.linear(self.in_features, r),
            place.linear(r, self.out_features, bias=place.bias),
        )

    @override
    def get_delta_weight(self, adapter: str) -> Tensor:
        r"""Return :math:`s\,B A`, the update ``adapter`` adds to the weight.

        Parameters
        ----------
        adapter : str
            The adapter's name.

        Returns
        -------
        Tensor
            ``(out_features, in_features)``, in the adapter's dtype.
        """
        scaling = self._specs[adapter].scaling
        return (self.lora_B[adapter].weight @ self.lora_A[adapter].weight) * scaling
