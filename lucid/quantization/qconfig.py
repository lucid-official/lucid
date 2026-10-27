"""``QConfig`` — the (activation, weight) observer/fake-quant recipe.

A :class:`QConfig` pairs the observer recipe for **activations** with the
one for **weights**; a :class:`QConfigMapping` then says which QConfig
applies to which part of a model (globally, by module type, or by
qualified name).  This is the reference framework's structure, and it is
the analogue of the AMP autocast policy: a small declarative object that
``prepare`` / ``prepare_qat`` consult when inserting observers.

The ``activation`` / ``weight`` fields hold **factories** (zero-arg
callables, typically produced by ``Observer.with_args(...)`` /
``FakeQuantize.with_args(...)``) so one QConfig can be instantiated fresh
at every insertion site.
"""

from typing import Callable, NamedTuple

import lucid.nn as nn
from lucid.quantization._qscheme import per_channel_symmetric, per_tensor_affine, qint8, quint8
from lucid.quantization._fake_quantize import FakeQuantize
from lucid.quantization.observer import (
    HistogramObserver,
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
)

_Factory = Callable[..., nn.Module]


class QConfig(NamedTuple):
    """The atomic ``(activation, weight)`` observer / fake-quant recipe.

    A :class:`QConfig` is the smallest unit of quantization *policy*: it pairs the
    recipe for how **activations** are observed / fake-quantized with the recipe for
    how **weights** are. It carries no state of its own — only two *factories*. When
    :func:`~lucid.quantization.prepare` (static PTQ) or
    :func:`~lucid.quantization.prepare_qat` (QAT) walks a model and decides a module
    should be quantized, it calls ``qconfig.activation()`` and ``qconfig.weight()`` to
    mint a **fresh** observer (PTQ) or :class:`~lucid.quantization.FakeQuantize` (QAT)
    for that specific insertion site. One :class:`QConfig` therefore parameterizes the
    quantization of an entire network without any two layers sharing observer state.

    Conceptually it is the direct analogue of an AMP autocast policy: a small,
    declarative object the framework consults while rewriting the graph, rather than
    something that participates in the forward pass itself. A :class:`QConfigMapping`
    sits one level above and answers the orthogonal question of *which* QConfig
    applies to *which* module.

    **Why factories, not instances.** Both fields hold zero-arg callables — almost
    always produced by ``Observer.with_args(...)`` or ``FakeQuantize.with_args(...)``,
    which capture the desired :class:`~lucid.quantization.QScheme` /
    :class:`~lucid.quantization.QDtype` / ``ch_axis`` and defer construction. Storing
    factories (not built modules) is what lets one recipe instantiate independent,
    per-layer observers across the whole model; sharing a single observer instance
    between layers would conflate their statistics and corrupt calibration.

    Attributes
    ----------
    activation : Callable[..., nn.Module]
        Factory for the **activation** observer / fake-quant. Called with no arguments
        at every insertion site to build a fresh module that tracks the dynamic range
        of that layer's *output* activations.
    weight : Callable[..., nn.Module]
        Factory for the **weight** observer / fake-quant. Called with no arguments to
        build a fresh module that quantizes that layer's learned *weight* (typically
        per-output-channel, symmetric ``qint8``).

    Notes
    -----
    - **Factory semantics (``with_args``).** ``HistogramObserver.with_args(qscheme=...,
      qdtype=...)`` returns a zero-arg callable (a captured partial); calling it yields
      a configured observer. This is why the two fields are typed
      ``Callable[..., nn.Module]`` and not ``nn.Module``.
    - **PTQ vs QAT.** For static PTQ the factories build plain *observers* (which only
      record statistics); for QAT they build :class:`~lucid.quantization.FakeQuantize`
      modules (which also apply straight-through rounding so gradients see the
      quantization error). The two default recipes differ only in that wrapper.
    - **Pairing, not a scalar.** Activations and weights almost always want *different*
      schemes — activations are asymmetric (post-ReLU ranges start at 0, hence
      per-tensor-affine) while weights are near-zero-mean (hence per-channel-symmetric).
      Bundling both keeps that pairing explicit and consistent.
    - Being a :class:`~typing.NamedTuple`, a :class:`QConfig` is immutable and unpacks
      positionally as ``activation, weight = qconfig``.

    Examples
    --------
    Build a recipe by hand from two ``with_args`` factories:

    >>> import lucid.quantization as Q
    >>> from lucid.quantization.observer import (
    ...     HistogramObserver, PerChannelMinMaxObserver)
    >>> qcfg = Q.QConfig(
    ...     activation=HistogramObserver.with_args(
    ...         qscheme=Q.per_tensor_affine, qdtype=Q.quint8),
    ...     weight=PerChannelMinMaxObserver.with_args(
    ...         ch_axis=0, qscheme=Q.per_channel_symmetric, qdtype=Q.qint8),
    ... )
    >>> act_observer = qcfg.activation()   # fresh observer for one insertion site
    >>> wt_observer = qcfg.weight()        # independent observer, no shared state
    >>> type(act_observer).__name__
    'HistogramObserver'

    In practice the defaults cover the common case, so hand-building is rare:

    >>> qcfg = Q.get_default_qconfig()       # static-PTQ recipe
    >>> qat = Q.get_default_qat_qconfig()    # QAT recipe (FakeQuantize-wrapped)

    See Also
    --------
    QConfigMapping : Decides which :class:`QConfig` applies to which module.
    get_default_qconfig : The standard static-PTQ recipe.
    get_default_qat_qconfig : The standard QAT recipe.
    lucid.quantization.FakeQuantize : The QAT observer + straight-through wrapper.
    """

    activation: _Factory
    weight: _Factory


def get_default_qconfig() -> QConfig:
    r"""Return the standard **static post-training-quantization** recipe.

    The recipe to reach for first when quantizing a trained float model for int8
    inference without any fine-tuning. It pairs the two schemes the literature and the
    reference framework have settled on as the accuracy-preserving default for
    convolution / linear stacks:

    * **Activations** — a per-tensor-affine ``quint8`` :class:`HistogramObserver`.
      Activations are fed through a histogram so the calibrated ``(scale, zero_point)``
      minimizes quantization error under the *observed distribution* rather than merely
      clamping to min/max; this matters because activations are typically heavy-tailed
      (a few large outliers would otherwise stretch the grid and starve the bulk of the
      mass of resolution). The affine (asymmetric) grid with a free ``zero_point`` fits
      post-ReLU ranges that start at 0 without wasting half the codes on negatives.
    * **Weights** — a per-channel-symmetric ``qint8`` :class:`PerChannelMinMaxObserver`
      on ``ch_axis=0``. Each output channel gets its own scale, so channels with very
      different magnitudes are each tracked tightly; the symmetric grid pins
      ``zero_point`` to 0, which the low-precision GEMM kernels assume for the weight
      operand.

    Returns
    -------
    QConfig
        A recipe whose ``activation`` factory builds a histogram observer and whose
        ``weight`` factory builds a per-channel min/max observer.

    Notes
    -----
    - This is the *global* recipe installed by :func:`get_default_qconfig_mapping`;
      override it per-type or per-name through a :class:`QConfigMapping` when a specific
      layer needs different treatment (or should stay in float).
    - Both fields are ``with_args`` factories, so :func:`~lucid.quantization.prepare`
      builds an independent observer at every site (see :class:`QConfig`).
    - The activation histogram is costlier to calibrate than a plain min/max observer
      but is the single biggest lever on PTQ accuracy for real activations; swap in a
      ``MinMaxObserver`` only when calibration speed dominates.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> qcfg = Q.get_default_qconfig()
    >>> type(qcfg.activation()).__name__
    'HistogramObserver'
    >>> type(qcfg.weight()).__name__
    'PerChannelMinMaxObserver'
    >>> # Apply it globally, then keep every Linear in float:
    >>> mapping = (Q.QConfigMapping()
    ...     .set_global(qcfg)
    ...     .set_object_type(nn.Linear, None))

    See Also
    --------
    get_default_qat_qconfig : The QAT counterpart (same schemes, FakeQuantize-wrapped).
    get_default_qconfig_mapping : Installs this recipe as a model-wide mapping.
    QConfig : The recipe object this returns.
    """
    return QConfig(
        activation=HistogramObserver.with_args(qscheme=per_tensor_affine, qdtype=quint8),
        weight=PerChannelMinMaxObserver.with_args(
            ch_axis=0, qscheme=per_channel_symmetric, qdtype=qint8
        ),
    )


def get_default_qat_qconfig() -> QConfig:
    r"""Return the standard **quantization-aware-training** recipe.

    The QAT analogue of :func:`get_default_qconfig`: it uses the *same* grid geometry
    for activations and weights, but wraps each observer in a
    :class:`~lucid.quantization.FakeQuantize` module. During training that wrapper both
    refreshes ``(scale, zero_point)`` from live statistics *and* applies
    straight-through fake-quantization, so the training graph experiences the exact
    rounding / clamping of int8 inference while gradients keep flowing. Fine-tuning
    under that simulated error lets the weights adapt to it, recovering accuracy that
    plain PTQ leaves on the table for aggressive (low-bit or quantization-sensitive)
    models.

    * **Activations** — a per-tensor-affine ``quint8`` fake-quant driven by a
      :class:`MovingAverageMinMaxObserver`. The moving average smooths the per-batch
      range so the simulated grid does not jitter from step to step during training.
    * **Weights** — a per-channel-symmetric ``qint8`` fake-quant on ``ch_axis=0``,
      matching the eventual inference-time weight quantization exactly.

    Returns
    -------
    QConfig
        A recipe whose two factories build :class:`~lucid.quantization.FakeQuantize`
        modules (rather than bare observers) for activations and weights.

    Notes
    -----
    - Install it with :func:`~lucid.quantization.prepare_qat`, not
      :func:`~lucid.quantization.prepare`; after training,
      :func:`~lucid.quantization.convert` folds the fake-quant statistics into a
      genuinely quantized model.
    - The standard QAT schedule toggles the two ``FakeQuantize`` switches over time —
      observe + fake-quant early, then freeze the observer (and BN stats) as training
      converges — via :meth:`~lucid.quantization.FakeQuantize.disable_observer`, etc.
    - Activations use a *moving-average* min/max here rather than a histogram: the range
      is re-estimated every step, so a cheap running statistic is both sufficient and
      more stable than a per-step histogram.

    Examples
    --------
    >>> import lucid.quantization as Q
    >>> qat = Q.get_default_qat_qconfig()
    >>> type(qat.activation()).__name__
    'FakeQuantize'
    >>> type(qat.weight()).__name__
    'FakeQuantize'

    See Also
    --------
    get_default_qconfig : The static-PTQ counterpart (bare observers, no fake-quant).
    get_default_qat_qconfig_mapping : Installs this recipe as a model-wide mapping.
    lucid.quantization.FakeQuantize : The wrapper both factories build.
    """
    return QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver, qscheme=per_tensor_affine, qdtype=quint8
        ),
        weight=FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            ch_axis=0,
            qscheme=per_channel_symmetric,
            qdtype=qint8,
        ),
    )


class QConfigMapping:
    r"""Resolves which :class:`QConfig` applies to each module in a model.

    A :class:`QConfig` says *how* to quantize; a :class:`QConfigMapping` says *where*.
    :func:`~lucid.quantization.prepare` / :func:`~lucid.quantization.prepare_qat`
    consult a mapping for every candidate module and insert (or skip) observers
    according to the resolved recipe. The mapping is a small three-tier lookup table
    plus a resolution rule.

    **Three tiers, most-specific-wins.** For a module of type ``T`` at qualified name
    ``"a.b.c"`` the mapping is queried in order:

    1. **module name** — an exact :meth:`set_module_name` entry;
    2. **module type** — a :meth:`set_object_type` entry;
    3. **global** — the :meth:`set_global` fallback.

    The first tier that has an entry wins, so a per-name rule overrides a per-type
    rule, which overrides the global default. This lets you set one recipe for the
    whole model and then carve out exceptions with surgical precision.

    **Mapping to ``None`` means "do not quantize".** Any tier may map to ``None`` to
    *exclude* a module from quantization even when a broader tier would have quantized
    it — e.g. keep a numerically sensitive final classifier in float while quantizing
    everything else.

    **Fluent construction.** Every setter returns ``self``, so a mapping is typically
    built in a single chained expression.

    Notes
    -----
    - The name / type tiers are plain dicts, so re-setting the same key overwrites the
      previous entry; there is no merge.
    - Type matching is by *exact* type identity (the dict is keyed on the class
      object), not ``isinstance`` — a subclass is not matched by a base-class rule
      unless you register the subclass explicitly.
    - The default mappings from :func:`get_default_qconfig_mapping` /
      :func:`get_default_qat_qconfig_mapping` set only the global tier; you add
      per-type / per-name overrides on top.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> mapping = (
    ...     Q.QConfigMapping()
    ...     .set_global(Q.get_default_qconfig())               # default everywhere
    ...     .set_object_type(nn.Linear, None)                  # ... except Linears
    ...     .set_module_name("head", Q.get_default_qconfig())  # ... but quantize "head"
    ... )
    >>> # "head" is a Linear, but the per-name rule (tier 1) beats the per-type rule:
    >>> mapping.get_qconfig(nn.Linear, "head") is not None
    True
    >>> # any other Linear falls to the per-type rule → excluded from quantization:
    >>> mapping.get_qconfig(nn.Linear, "encoder.fc") is None
    True
    >>> # a Conv2d has no name / type rule → the global default applies:
    >>> mapping.get_qconfig(nn.Conv2d, "features.0") is not None
    True

    See Also
    --------
    QConfig : The recipe a mapping resolves to.
    get_default_qconfig_mapping : A mapping preloaded with the static-PTQ default.
    get_default_qat_qconfig_mapping : A mapping preloaded with the QAT default.
    """

    def __init__(self) -> None:
        self._global: QConfig | None = None
        self._object_type: dict[type, QConfig | None] = {}
        self._module_name: dict[str, QConfig | None] = {}

    def set_global(self, qconfig: QConfig | None) -> QConfigMapping:
        r"""Set the fallback :class:`QConfig` used when no more specific rule matches.

        The global tier is the lowest-priority tier (below per-type and per-name), so
        it defines the model-wide default behaviour. Pass ``None`` to make "leave in
        float" the default and then *opt in* specific types / names on top.

        Parameters
        ----------
        qconfig : QConfig or None
            The recipe applied to any module without a matching per-type or per-name
            entry, or ``None`` to leave such modules unquantized.

        Returns
        -------
        QConfigMapping
            ``self``, so setters can be chained fluently.
        """
        self._global = qconfig
        return self

    def set_object_type(self, module_type: type, qconfig: QConfig | None) -> QConfigMapping:
        r"""Bind a :class:`QConfig` to every module of an exact type.

        A per-type rule sits above the global fallback but below a per-name rule.
        Matching is by exact class identity, not ``isinstance`` — register a subclass
        explicitly if you need it. Pass ``None`` to exclude every module of this type
        from quantization.

        Parameters
        ----------
        module_type : type
            The module class to match (e.g. ``lucid.nn.Linear``).
        qconfig : QConfig or None
            The recipe for modules of ``module_type``, or ``None`` to skip them.

        Returns
        -------
        QConfigMapping
            ``self``, so setters can be chained fluently.
        """
        self._object_type[module_type] = qconfig
        return self

    def set_module_name(self, name: str, qconfig: QConfig | None) -> QConfigMapping:
        r"""Bind a :class:`QConfig` to the single module at a qualified name.

        A per-name rule is the highest-priority tier: it overrides both the per-type
        rule and the global fallback for exactly the module whose dotted path in the
        parent model equals ``name`` (e.g. ``"encoder.layer.0.attention"``). Pass
        ``None`` to exclude just that one module.

        Parameters
        ----------
        name : str
            The fully-qualified module name (dotted path from the root model).
        qconfig : QConfig or None
            The recipe for that module, or ``None`` to leave it unquantized.

        Returns
        -------
        QConfigMapping
            ``self``, so setters can be chained fluently.
        """
        self._module_name[name] = qconfig
        return self

    def get_qconfig(self, module_type: type, module_name: str) -> QConfig | None:
        r"""Resolve the :class:`QConfig` for a module, most-specific tier first.

        Implements the resolution rule: a per-name entry (tier 1) wins over a per-type
        entry (tier 2), which wins over the global fallback (tier 3). Returns ``None``
        when the winning tier maps to ``None`` *or* when no tier matches and no global
        default was set — in both cases the caller leaves the module in float.

        Parameters
        ----------
        module_type : type
            The exact class of the module being resolved.
        module_name : str
            The fully-qualified (dotted) name of the module in the root model.

        Returns
        -------
        QConfig or None
            The resolved recipe, or ``None`` to signal "do not quantize".
        """
        if module_name in self._module_name:
            return self._module_name[module_name]
        if module_type in self._object_type:
            return self._object_type[module_type]
        return self._global


def get_default_qconfig_mapping() -> QConfigMapping:
    r"""Build the default **static-PTQ** :class:`QConfigMapping`.

    Convenience constructor returning a fresh mapping whose *global* tier is
    :func:`get_default_qconfig` and whose per-type / per-name tiers are empty. Every
    eligible module is therefore quantized with the standard histogram-activation /
    per-channel-weight recipe unless you add an override. This is the mapping
    :func:`~lucid.quantization.prepare` uses when called without an explicit mapping.

    Returns
    -------
    QConfigMapping
        A mapping whose global :class:`QConfig` is the default static-PTQ config and
        whose type / name tiers are empty (ready to receive overrides).

    Notes
    -----
    - Add exceptions with :meth:`QConfigMapping.set_object_type` /
      :meth:`~QConfigMapping.set_module_name` (e.g. map a sensitive head to ``None``).
    - For quantization-aware training use :func:`get_default_qat_qconfig_mapping`
      instead — it installs :class:`~lucid.quantization.FakeQuantize` recipes.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> mapping = Q.get_default_qconfig_mapping()
    >>> mapping.get_qconfig(nn.Conv2d, "features.0") is not None
    True
    >>> # Keep the classifier in float while quantizing the rest:
    >>> _ = mapping.set_module_name("classifier", None)
    >>> mapping.get_qconfig(nn.Linear, "classifier") is None
    True

    See Also
    --------
    get_default_qconfig : The recipe installed as the global tier.
    get_default_qat_qconfig_mapping : The QAT counterpart.
    QConfigMapping : The three-tier resolution object this returns.
    """
    return QConfigMapping().set_global(get_default_qconfig())


def get_default_qat_qconfig_mapping() -> QConfigMapping:
    r"""Build the default **QAT** :class:`QConfigMapping`.

    Convenience constructor returning a fresh mapping whose *global* tier is
    :func:`get_default_qat_qconfig` and whose per-type / per-name tiers are empty. Every
    eligible module therefore trains with :class:`~lucid.quantization.FakeQuantize` on
    its activations and weights unless you add an override. This is the mapping
    :func:`~lucid.quantization.prepare_qat` uses when called without an explicit one.

    Returns
    -------
    QConfigMapping
        A mapping whose global :class:`QConfig` is the default QAT config and whose
        type / name tiers are empty (ready to receive overrides).

    Notes
    -----
    - Add exceptions with :meth:`QConfigMapping.set_object_type` /
      :meth:`~QConfigMapping.set_module_name`; mapping a module to ``None`` trains it
      in float (no fake-quant inserted).
    - For static post-training quantization use :func:`get_default_qconfig_mapping`
      instead — it installs bare observers rather than fake-quant modules.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> mapping = Q.get_default_qat_qconfig_mapping()
    >>> type(mapping.get_qconfig(nn.Linear, "fc").activation()).__name__
    'FakeQuantize'

    See Also
    --------
    get_default_qat_qconfig : The recipe installed as the global tier.
    get_default_qconfig_mapping : The static-PTQ counterpart.
    QConfigMapping : The three-tier resolution object this returns.
    """
    return QConfigMapping().set_global(get_default_qat_qconfig())
