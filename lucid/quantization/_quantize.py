"""Eager-mode static post-training quantization: ``prepare`` → ``convert``.

The classic three-step flow:

1. :func:`prepare` — attach an activation observer (as a forward hook) to
   every quantizable module the :class:`QConfigMapping` selects.
2. *calibration* — the caller runs representative data through the prepared
   model so the observers collect activation ranges.
3. :func:`convert` — swap each calibrated float module for its quantized
   counterpart (int8 weights baked in, activation qparams frozen).

``nn.quantized`` is imported lazily inside the functions so this module can
live in ``lucid.quantization`` without an import cycle (``nn.quantized``
imports the quantization primitives back).
"""

import copy
import warnings
from typing import TYPE_CHECKING, Callable, cast

import lucid.nn as nn
from lucid.quantization._qscheme import QDtype, qint8
from lucid.quantization.qconfig import (
    QConfig,
    QConfigMapping,
    get_default_qat_qconfig_mapping,
    get_default_qconfig_mapping,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    _Hook = Callable[[object, tuple[Tensor, ...], Tensor | tuple[Tensor, ...]], None]
    _FromFloat = Callable[[nn.Module], nn.Module]
    _DynFromFloat = Callable[[nn.Module, QDtype], nn.Module]


def _observed_types() -> tuple[type, ...]:
    """Module types that receive an activation observer during ``prepare``."""
    import lucid.nn.intrinsic as nni
    import lucid.nn.quantized as nnq

    return (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nni.ConvReLU1d,
        nni.ConvReLU2d,
        nni.ConvReLU3d,
        nni.LinearReLU,
        nn.Sigmoid,
        nn.Hardswish,
        nn.Hardsigmoid,
        nn.Tanh,
        nn.ELU,
        nn.LeakyReLU,
        nnq.QuantStub,
        nnq.FloatFunctional,
    )


def _quant_mapping() -> dict[type, _FromFloat]:
    """Map each float / QAT module type to its quantized ``from_float`` builder."""
    import lucid.nn.intrinsic as nni
    import lucid.nn.intrinsic.qat as nniqat
    import lucid.nn.qat as nnqat
    import lucid.nn.quantized as nnq
    from lucid.nn.intrinsic.qat.modules import _convbn_to_quantized

    return {
        nniqat.ConvBn1d: _convbn_to_quantized,
        nniqat.ConvBn2d: _convbn_to_quantized,
        nniqat.ConvBn3d: _convbn_to_quantized,
        nniqat.ConvBnReLU1d: _convbn_to_quantized,
        nniqat.ConvBnReLU2d: _convbn_to_quantized,
        nniqat.ConvBnReLU3d: _convbn_to_quantized,
        nn.Linear: nnq.Linear.from_float,
        nn.Conv1d: nnq.Conv1d.from_float,
        nn.Conv2d: nnq.Conv2d.from_float,
        nn.Conv3d: nnq.Conv3d.from_float,
        nn.ConvTranspose1d: nnq.ConvTranspose1d.from_float,
        nn.ConvTranspose2d: nnq.ConvTranspose2d.from_float,
        nn.ConvTranspose3d: nnq.ConvTranspose3d.from_float,
        nn.Embedding: nnq.Embedding.from_float,
        nn.EmbeddingBag: nnq.EmbeddingBag.from_float,
        nn.Sigmoid: nnq.Sigmoid.from_float,
        nn.Hardswish: nnq.Hardswish.from_float,
        nn.Hardsigmoid: nnq.Hardsigmoid.from_float,
        nn.Tanh: nnq.Tanh.from_float,
        nn.ELU: nnq.ELU.from_float,
        nn.LeakyReLU: nnq.LeakyReLU.from_float,
        nni.ConvReLU1d: nnq.ConvReLU1d.from_float,
        nni.ConvReLU2d: nnq.ConvReLU2d.from_float,
        nni.ConvReLU3d: nnq.ConvReLU3d.from_float,
        nni.LinearReLU: nnq.LinearReLU.from_float,
        # QAT modules → the same quantized inference layers.
        nnqat.Linear: nnq.Linear.from_float,
        nnqat.Conv1d: nnq.Conv1d.from_float,
        nnqat.Conv2d: nnq.Conv2d.from_float,
        nnqat.Conv3d: nnq.Conv3d.from_float,
        nnqat.LinearReLU: nnq.LinearReLU.from_float,
        nnqat.ConvReLU1d: nnq.ConvReLU1d.from_float,
        nnqat.ConvReLU2d: nnq.ConvReLU2d.from_float,
        nnqat.ConvReLU3d: nnq.ConvReLU3d.from_float,
        nnqat.Embedding: nnq.Embedding.from_float,
        nnq.QuantStub: nnq.Quantize.from_float,
        nnq.DeQuantStub: nnq.DeQuantize.from_float,
        nnq.FloatFunctional: nnq.QFunctional.from_float,
    }


def _mlx_weight_bits(mod: nn.Module) -> int:
    """Weight bit-width (4/8) for the MLX fast path, read from the module's qconfig."""
    qcfg = getattr(mod, "qconfig", None)
    if qcfg is not None:
        bits = getattr(getattr(qcfg.weight(), "qdtype", None), "bits", None)
        if isinstance(bits, int) and bits in (4, 8):
            return bits
    return 8


def _mlx_group_size(in_features: int) -> int | None:
    """Largest valid MLX group size (32/64/128) dividing ``in_features``.

    ``mlx.core.quantize`` requires the quantized dimension to be a multiple of
    the group size, so a layer whose ``in_features`` divides none of the three
    supported sizes cannot use the group-wise kernel — return ``None`` and let
    the caller fall back to the dequantize path.
    """
    for gs in (64, 128, 32):
        if in_features % gs == 0:
            return gs
    return None


def _mlx_linear_builder(relu: bool) -> _FromFloat:
    """1-arg ``convert`` builder that routes a (fused) Linear to the real MLX GEMM.

    Weight-only group-wise quantization (``quantized_matmul``) — the actual
    speed + memory win, in exchange for W8A16 numerics (activations stay float).
    Selected only when ``backends.quantized.use_mlx()`` is true; layers whose
    ``in_features`` is not group-size-divisible fall back to the dequant path.
    """
    from lucid.nn.quantized import Linear as QLinear
    from lucid.nn.quantized import LinearReLU as QLinearReLU
    from lucid.nn.quantized import QuantizedLinearMLX

    dequant: _FromFloat = QLinearReLU.from_float if relu else QLinear.from_float

    def build(child: nn.Module) -> nn.Module:
        inner = cast("nn.Sequential", child)[0] if relu else child
        gs = _mlx_group_size(cast("nn.Linear", inner).in_features)
        if gs is None:
            return dequant(child)
        return QuantizedLinearMLX.from_float(
            inner, bits=_mlx_weight_bits(child), group_size=gs, relu=relu
        )

    return build


def _mlx_dynamic_builder() -> _DynFromFloat:
    """2-arg ``quantize_dynamic`` builder routing Linear to the real MLX GEMM."""
    import lucid.nn.quantized.dynamic as nnqd
    from lucid.nn.quantized import QuantizedLinearMLX

    def build(child: nn.Module, dtype: QDtype) -> nn.Module:
        gs = _mlx_group_size(cast("nn.Linear", child).in_features)
        if gs is None:
            return nnqd.Linear.from_float(child, dtype)
        return QuantizedLinearMLX.from_float(child, bits=dtype.bits, group_size=gs, relu=False)

    return build


def _make_observer_hook(obs: nn.Module) -> _Hook:
    """Forward hook that feeds a module's output to its activation observer."""

    def hook(
        _module: object, _args: tuple[Tensor, ...], output: Tensor | tuple[Tensor, ...]
    ) -> None:
        obs(cast("Tensor", output))

    return hook


def _as_mapping(qconfig: QConfig | QConfigMapping | None) -> QConfigMapping:
    """Coerce a bare QConfig (or ``None``) to a global :class:`QConfigMapping`."""
    if qconfig is None:
        return get_default_qconfig_mapping()
    if isinstance(qconfig, QConfig):
        return QConfigMapping().set_global(qconfig)
    return qconfig


def prepare(
    model: nn.Module, qconfig: QConfig | QConfigMapping | None = None, inplace: bool = False
) -> nn.Module:
    r"""Attach activation observers to a float model ahead of calibration.

    ``prepare`` is step one of the eager-mode static post-training quantization
    (PTQ) workflow — the classic ``prepare`` → *calibrate* → :func:`convert`
    pipeline. It does not touch weights or change any numerics; it walks the
    model and, for every quantizable module the :class:`QConfigMapping` selects,
    attaches an **activation observer** as a forward hook. As calibration data
    flows through the returned model, each observer records the running range
    (min / max, or a histogram) of that module's output, and :func:`convert`
    later reads those ranges to freeze the per-layer activation ``scale`` /
    ``zero_point``.

    Only module types in the observed set — Linear, the Conv / ConvTranspose
    family, the intrinsic fused Conv/Linear+ReLU modules, a handful of
    activations, and the ``QuantStub`` / ``FloatFunctional`` boundaries — receive
    an observer, and only where the mapping returns a non-``None`` ``QConfig``
    for them. The model is switched to ``eval`` mode so BatchNorm / Dropout stay
    deterministic during calibration. Use this when you want the best accuracy
    and can afford a calibration pass over representative data; for a
    calibration-free path see :func:`quantize_dynamic`, and to make quantization
    error visible to the optimizer see :func:`prepare_qat`.

    Parameters
    ----------
    model : nn.Module
        The float model to observe. Explicit ``QuantStub`` / ``DeQuantStub``
        boundaries are picked up automatically; if you would rather not place
        them by hand, use :func:`prepare_fx`, which wraps them for you.
    qconfig : QConfig or QConfigMapping, optional
        The quantization recipe — a single :class:`QConfig` is promoted to a
        global mapping, and ``None`` (default) falls back to
        :func:`get_default_qconfig_mapping` (per-channel int8 weights,
        per-tensor affine activations).
    inplace : bool, default False
        If ``True`` mutate and return ``model`` itself; if ``False`` (default)
        observe a :func:`copy.deepcopy` and leave the original untouched.

    Returns
    -------
    nn.Module
        The prepared, ``eval``-mode model carrying activation observers. Run
        representative data through it to calibrate, then pass it to
        :func:`convert`.

    Notes
    -----
    - Non-destructive: with ``inplace=False`` the source model is deep-copied,
      so the returned graph is a separate object.
    - Nothing is quantized yet — the returned model still runs in full float and
      is marginally *slower* than the original because of the observer hooks.
      The size / speed win only lands after :func:`convert`.
    - If the mapping matches no quantizable module a ``UserWarning`` is raised
      and a later :func:`convert` leaves the model unchanged — usually a sign
      the ``QConfigMapping`` targets types the model does not contain.
    - Fuse Conv/BN/ReLU runs with :func:`fuse_modules` *before* ``prepare`` so
      the observer sees the true post-fusion (e.g. post-ReLU) activation range.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare(model)            # observers inserted (deep copy)
    >>> for _ in range(8):                      # calibrate on representative data
    ...     _ = prepared(lucid.randn(32, 128))
    >>> qmodel = Q.convert(prepared)            # freeze qparams, bake int8 weights

    Targeting a type the model does not contain warns, and the downstream
    :func:`convert` becomes a no-op:

    >>> mapping = Q.QConfigMapping().set_object_type(nn.Conv2d, Q.get_default_qconfig())
    >>> _ = Q.prepare(nn.Sequential(nn.Linear(8, 8)), mapping)   # doctest: +SKIP
    UserWarning: prepare(): the QConfigMapping matched no quantizable modules ...

    See Also
    --------
    lucid.quantization.convert : Bake the calibrated model into quantized modules.
    lucid.quantization.quantize_dynamic : Calibration-free dynamic quantization.
    lucid.quantization.prepare_qat : Insert fake-quant for quantization-aware training.
    lucid.quantization.fuse_modules : Fuse Conv/BN/ReLU runs before ``prepare``.
    """
    mapping = _as_mapping(qconfig)
    model = model if inplace else copy.deepcopy(model)
    model.eval()
    observed = _observed_types()
    n_matched = 0
    for name, mod in model.named_modules():
        qcfg = mapping.get_qconfig(type(mod), name)
        if qcfg is None or not isinstance(mod, observed):
            continue
        mod.qconfig = qcfg
        obs = qcfg.activation()
        mod.activation_post_process = obs
        mod.register_forward_hook(_make_observer_hook(obs))
        n_matched += 1
    if n_matched == 0:
        warnings.warn(
            "prepare(): the QConfigMapping matched no quantizable modules — "
            "convert() will leave this model unchanged.",
            stacklevel=2,
        )
    return model


def convert(model: nn.Module, inplace: bool = False) -> nn.Module:
    r"""Bake a calibrated model into quantized inference modules.

    ``convert`` is the final step of the static-PTQ pipeline (``prepare`` →
    *calibrate* → ``convert``). It walks the calibrated model and swaps every
    observed float submodule for its quantized counterpart: the learned float
    weight is quantized once into int8 codes plus per-channel ``scale`` /
    ``zero_point`` (dropping the float weight), and each activation observer's
    recorded range is frozen into that layer's output qparams. What comes back
    is a self-contained, ``eval``-mode model with no observers left — ready to
    run or serialize, with a checkpoint roughly ``4x`` smaller than the float
    original.

    A submodule is convertible when calibration attached an
    ``activation_post_process`` to it; a small always-convert set
    (``DeQuantStub``, ``Embedding``, ``EmbeddingBag``) converts even without an
    activation observer. Each replacement takes the slot of the child it
    replaces, so a ``Sequential``'s child order is preserved.

    When ``lucid.backends.quantized.use_mlx()`` is true (the MLX quantized engine
    is present), the Linear family is routed to the real weight-only
    ``quantized_matmul`` GEMM — the genuine compute + memory win, at W8A16
    numerics (weights int8/int4, activations kept float) — instead of the
    dequantize-to-float reference path. On a fully-Metal model this is bandwidth-
    bound and measures ~``3.15x`` faster at the memory-bound decode shape
    ``M = 1`` with a weight payload ~``3.55x`` smaller; layers whose
    ``in_features`` is not divisible by a supported MLX group size fall back to
    the dequant path automatically. Set ``backends.quantized.engine =
    "reference"`` to force the exact W8A8 dequant numerics on every layer.

    Parameters
    ----------
    model : nn.Module
        A model previously returned by :func:`prepare` and then run through
        calibration, so every quantizable submodule carries frozen activation
        qparams. QAT models (from :func:`prepare_qat`, fine-tuned) also convert
        here — their ``nn.qat`` layers map to the same quantized inference layers.
    inplace : bool, default False
        If ``True`` mutate and return ``model`` itself; if ``False`` (default)
        convert a :func:`copy.deepcopy` and leave the calibrated model intact.

    Returns
    -------
    nn.Module
        The converted, ``eval``-mode model with int8 weights baked in and its
        float submodules replaced by their quantized counterparts.

    Notes
    -----
    - Non-destructive by default (deep-copy); pass ``inplace=True`` to convert
      the calibrated model in place and reclaim its float weights.
    - Only int8 codes + qparams + float bias enter the resulting ``state_dict``;
      the float weight never does (a whole checkpoint shrinks ~``3.97x`` on
      ``resnet_18``).
    - The reference (non-MLX) path wins on **memory**, not compute — its GEMM
      runs in float and so is roughly float-speed on the CPU stream. Convert with
      the MLX backend active for the genuine low-precision compute speed-up.
    - Calling ``convert`` on a model that was never calibrated is a silent
      correctness bug: the output qparams are whatever the fresh observers hold
      (often identity), so run calibration data through :func:`prepare`'s output
      first.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.Linear(128, 64))
    >>> model.qconfig = Q.get_default_qconfig()
    >>> prepared = Q.prepare(model)
    >>> for _ in range(8):                       # calibrate before converting
    ...     _ = prepared(lucid.randn(32, 128))
    >>> qmodel = Q.convert(prepared)
    >>> type(qmodel[0]).__name__ == (            # which layer depends on the backend
    ...     "QuantizedLinearMLX" if lucid.backends.quantized.use_mlx() else "Linear"
    ... )
    True

    The common mistake — converting straight after ``prepare`` with **no**
    calibration — silently produces a model with meaningless activation qparams:

    >>> prepared = Q.prepare(nn.Sequential(nn.Linear(128, 64)))  # doctest: +SKIP
    >>> qmodel = Q.convert(prepared)   # no data ran through -> garbage output scale

    See Also
    --------
    lucid.quantization.prepare : Insert observers ahead of calibration.
    lucid.quantization.convert_fx : The graph-mode counterpart.
    lucid.quantization.quantize_dynamic : Calibration-free alternative.
    lucid.nn.quantized.Linear : The quantized layer this installs for Linear.
    """
    import lucid
    import lucid.nn.intrinsic as nni
    import lucid.nn.qat as nnqat
    import lucid.nn.quantized as nnq

    model = model if inplace else copy.deepcopy(model)
    mapping = _quant_mapping()
    if lucid.backends.quantized.use_mlx():
        mapping = dict(mapping)
        mapping[nn.Linear] = _mlx_linear_builder(relu=False)
        mapping[nni.LinearReLU] = _mlx_linear_builder(relu=True)
        mapping[nnqat.Linear] = _mlx_linear_builder(relu=False)
    # DeQuantStub + Embedding(Bag) convert without calibration (no activation obs).
    _convert_recursive(model, mapping, always=(nnq.DeQuantStub, nn.Embedding, nn.EmbeddingBag))
    model.eval()
    return model


def _dynamic_mapping() -> dict[type, _DynFromFloat]:
    """Map each float module type to its dynamic-quant ``from_float`` builder."""
    import lucid.nn.quantized.dynamic as nnqd

    return {
        nn.Linear: nnqd.Linear.from_float,
        nn.LSTM: nnqd.LSTM.from_float,
        nn.GRU: nnqd.GRU.from_float,
        nn.LSTMCell: nnqd.LSTMCell.from_float,
        nn.GRUCell: nnqd.GRUCell.from_float,
        nn.RNNCell: nnqd.RNNCell.from_float,
    }


def quantize_dynamic(
    model: nn.Module,
    qconfig_spec: set[type] | None = None,
    dtype: QDtype = qint8,
    inplace: bool = False,
) -> nn.Module:
    r"""Dynamically quantize a model — int8 weights, runtime-quantized activations.

    ``quantize_dynamic`` is the whole dynamic-PTQ workflow in one call: unlike
    the static ``prepare`` → *calibrate* → ``convert`` flow it needs **no
    calibration data and no observers**. Every module whose type is in
    ``qconfig_spec`` (default: Linear, LSTM, GRU and the LSTM / GRU / RNN
    cells) is swapped in place for its
    dynamic counterpart, which stores its weight as int8 codes but measures the
    activation range and quantizes activations *per forward*, at runtime. That
    makes it the go-to for models where activation statistics vary widely between
    inputs and a fixed calibrated range would be a poor fit — most notably the
    Linear-heavy stacks of Transformers and RNN language models, where the weight
    memory dominates and per-call activation quantization costs little.

    Because activation qparams are recomputed each call, dynamic quantization
    captures the ``4x`` weight-memory win without an offline calibration pass, at
    the cost of a small per-forward quantize overhead. Prefer static PTQ
    (:func:`prepare` / :func:`convert`) when you can calibrate and want the
    lowest inference latency; prefer this when calibration is impractical or the
    stack is Linear-only.

    When ``lucid.backends.quantized.use_mlx()`` is true, the Linear family is
    routed to the real weight-only ``quantized_matmul`` MLX GEMM (the genuine
    speed + memory win, W8A16 numerics), falling back to the dequantize path for
    layers whose ``in_features`` is not group-size-divisible.

    Parameters
    ----------
    model : nn.Module
        The float model to quantize. A bare top-level target (e.g. calling this
        directly on an ``nn.Linear``) is handled too — it has no parent to swap
        it into, so the quantized module is returned directly.
    qconfig_spec : set of module types, optional
        Which module types to quantize; ``None`` (default) uses the mapping's
        keys — ``{nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell, nn.GRUCell,
        nn.RNNCell}``, the reference framework's dynamic set.  ``nn.RNN`` and
        the convolutions have no dynamic form.
    dtype : QDtype, default ``qint8``
        The quantized dtype for the weights (its ``bits`` also selects the MLX
        fast-path bit width when that backend is active).
    inplace : bool, default False
        If ``True`` mutate and return ``model`` itself; if ``False`` (default)
        quantize a :func:`copy.deepcopy` and leave the original untouched.

    Returns
    -------
    nn.Module
        The dynamically-quantized, ``eval``-mode model (or the single quantized
        module when a bare target was passed).

    Notes
    -----
    - No calibration and no observers are inserted — the model is ready to run
      immediately after this call.
    - Only the targeted types are swapped; everything else (activations,
      norms, attention softmax) stays float, so this is a *partial* quantization
      aimed at the weight-heavy layers.
    - If no module of a targeted type is found, a ``UserWarning`` is raised and
      the model is returned unchanged.
    - A type named in ``qconfig_spec`` that has no dynamic form (``nn.Conv2d``,
      ``nn.RNN``) stays float, with a ``UserWarning`` naming it — the
      reference framework leaves it float silently.
    - The model is switched to ``eval`` mode before swapping.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> mlp = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
    >>> qmlp = Q.quantize_dynamic(mlp)          # no calibration needed
    >>> type(qmlp[0]).__name__ == (             # which layer depends on the backend
    ...     "QuantizedLinearMLX" if lucid.backends.quantized.use_mlx() else "Linear"
    ... )
    True
    >>> qmlp(lucid.randn(1, 512)).shape
    (1, 10)

    A model with no targeted layers warns and comes back unchanged — e.g. a
    conv-only stack, since ``Conv2d`` is not a dynamic-quant target:

    >>> conv = nn.Sequential(nn.Conv2d(3, 8, 3))
    >>> Q.quantize_dynamic(conv) is not None    # doctest: +SKIP
    UserWarning: quantize_dynamic(): no module of a targeted type (GRU, ...) found ...

    See Also
    --------
    lucid.quantization.prepare : Static-PTQ observer insertion (needs calibration).
    lucid.quantization.convert : Static-PTQ bake step.
    lucid.quantization.prepare_qat : Quantization-aware training entry point.
    lucid.nn.quantized.dynamic.Linear : The dynamic quantized Linear installed here.
    lucid.nn.quantized.dynamic.GRU : The dynamic quantized GRU installed here.
    """
    import lucid

    model = model if inplace else copy.deepcopy(model)
    model.eval()
    mapping = _dynamic_mapping()
    if lucid.backends.quantized.use_mlx():
        mapping = dict(mapping)
        mapping[nn.Linear] = _mlx_dynamic_builder()
    types = qconfig_spec if qconfig_spec is not None else set(mapping)
    _warn_left_float(model, types, set(mapping))
    # Handle a bare top-level target (no parent to swap it in).
    build = mapping.get(type(model))
    if build is not None and type(model) in types:
        return build(model, dtype)
    _swap_dynamic(model, mapping, types, dtype)
    return model


def _warn_left_float(model: nn.Module, types: set[type], mapped: set[type]) -> None:
    """Warn when ``quantize_dynamic`` would leave every targeted module float.

    Either no module of a targeted type is in ``model``, or a targeted type
    present in it has no dynamically quantized form.  Both name the types.
    """
    present = {type(m) for m in model.modules()} & types
    if not present:
        warnings.warn(
            "quantize_dynamic(): no module of a targeted type "
            f"({', '.join(sorted(t.__name__ for t in types))}) found — "
            "the model is returned unchanged.",
            # warn ← _warn_left_float ← quantize_dynamic ← caller
            stacklevel=3,
        )
    unmapped = present - mapped
    if unmapped:
        warnings.warn(
            f"quantize_dynamic(): {', '.join(sorted(t.__name__ for t in unmapped))} "
            "has no dynamically quantized form and stays float; dynamic "
            f"quantization covers {', '.join(sorted(t.__name__ for t in mapped))}.",
            stacklevel=3,
        )


def _swap_dynamic(
    module: nn.Module, mapping: dict[type, _DynFromFloat], types: set[type], dtype: QDtype
) -> None:
    """Depth-first swap of each targeted module for its dynamic-quant form."""
    for name, child in list(module.named_children()):
        build = mapping.get(type(child))
        if build is not None and type(child) in types:
            # ``setattr`` keeps the child's slot in place (LCD-551), so a
            # ``Sequential`` still runs it in the same turn; no direct
            # ``_modules`` write needed (LCD-556).
            setattr(module, name, build(child, dtype))
        else:
            _swap_dynamic(child, mapping, types, dtype)


def _qat_mapping() -> dict[type, _FromFloat]:
    """Map each float module type to its QAT (fake-quant, trainable) form."""
    import lucid.nn.intrinsic as nni
    import lucid.nn.qat as nnqat

    return {
        nn.Linear: nnqat.Linear.from_float,
        nn.Conv1d: nnqat.Conv1d.from_float,
        nn.Conv2d: nnqat.Conv2d.from_float,
        nn.Conv3d: nnqat.Conv3d.from_float,
        nn.Embedding: nnqat.Embedding.from_float,
        nni.LinearReLU: nnqat.LinearReLU.from_float,
        nni.ConvReLU1d: nnqat.ConvReLU1d.from_float,
        nni.ConvReLU2d: nnqat.ConvReLU2d.from_float,
        nni.ConvReLU3d: nnqat.ConvReLU3d.from_float,
    }


def prepare_qat(
    model: nn.Module, qconfig: QConfig | QConfigMapping | None = None, inplace: bool = False
) -> nn.Module:
    r"""Insert fake-quant modules for quantization-aware training.

    ``prepare_qat`` is the entry point of the QAT workflow — ``prepare_qat`` →
    *fine-tune* → :func:`convert`. Where static PTQ only *observes* a frozen
    float model, QAT makes the quantization grid part of the forward pass: this
    swaps every weighted layer for its :mod:`lucid.nn.qat` counterpart (which
    fake-quantizes both its weight and its output) and attaches a fake-quant to
    every ``QuantStub``. During the subsequent fine-tuning the network *feels*
    the int8 grid on each step — via the straight-through estimator of
    :func:`fake_quantize` — and learns weights that minimize the rounding error,
    typically recovering most of the accuracy a low-bit model would otherwise
    lose. Reach for it when plain static PTQ drops too much accuracy (aggressive
    bit widths, quantization-sensitive architectures).

    Unlike :func:`prepare`, the returned model is left in **``train`` mode**: BN
    keeps updating its running stats and the fake-quant grids track the evolving
    ranges. Fuse Conv/BN with :func:`fuse_modules_qat` first so the folded
    ``ConvBn*`` modules stay trainable and fold BN per forward under the STE.
    After fine-tuning, :func:`convert` maps the ``nn.qat`` layers to the same
    int8 inference layers a static-PTQ ``convert`` produces.

    Parameters
    ----------
    model : nn.Module
        Float model with ``QuantStub`` / ``DeQuantStub`` boundaries (and,
        ideally, Conv/BN runs already fused for QAT).
    qconfig : QConfig or QConfigMapping, optional
        The QAT recipe — a single :class:`QConfig` is promoted to a global
        mapping, and ``None`` (default) falls back to
        :func:`get_default_qat_qconfig_mapping` (fused weight + activation
        fake-quant).
    inplace : bool, default False
        If ``True`` mutate and return ``model`` itself; if ``False`` (default)
        prepare a :func:`copy.deepcopy` and leave the original untouched.

    Returns
    -------
    nn.Module
        The QAT-ready model in ``train`` mode. Fine-tune it, then call
        :func:`convert` to obtain the int8 inference model.

    Notes
    -----
    - Returned in ``train`` mode (contrast :func:`prepare`, which returns
      ``eval``) — fake-quant and BN both need training-time behavior.
    - Weighted layers become trainable fake-quant modules; their float weights
      are kept and updated by the optimizer (nothing is baked to int8 until
      :func:`convert`).
    - Fuse with :func:`fuse_modules_qat` (not :func:`fuse_modules`) beforehand:
      the eval-time BN fold would freeze BN, which QAT needs to keep learning.
    - Non-destructive by default (deep-copy).

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    >>> model.qconfig = Q.get_default_qat_qconfig()
    >>> qat = Q.prepare_qat(model)              # fake-quant inserted, train mode
    >>> qat.training
    True
    >>> # ... fine-tune qat for a few epochs with real labels ...
    >>> qmodel = Q.convert(qat.eval())          # bake into the int8 inference model

    See Also
    --------
    lucid.quantization.prepare : Static-PTQ observers (no fake-quant, eval mode).
    lucid.quantization.convert : Bake the fine-tuned QAT model into int8.
    lucid.quantization.fuse_modules_qat : Trainable Conv/BN fusion for QAT.
    lucid.quantization.fake_quantize : The STE primitive QAT trains through.
    """
    if qconfig is None:
        mapping = get_default_qat_qconfig_mapping()
    else:
        mapping = _as_mapping(qconfig)
    model = model if inplace else copy.deepcopy(model)
    model.train()
    _prepare_qat_recursive(model, mapping, _qat_mapping(), "")
    return model


def _prepare_qat_recursive(
    module: nn.Module, qconfig_mapping: QConfigMapping, qat_map: dict[type, _FromFloat], prefix: str
) -> None:
    """Depth-first swap of weighted layers to QAT + fake-quant on ``QuantStub``."""
    import lucid.nn.quantized as nnq

    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        qcfg = qconfig_mapping.get_qconfig(type(child), full)
        build = qat_map.get(type(child))
        if qcfg is not None and build is not None:
            child.qconfig = qcfg
            # ``setattr`` keeps the child's slot in place (LCD-551); no
            # direct ``_modules`` write needed (LCD-556).
            setattr(module, name, build(child))
        elif qcfg is not None and isinstance(child, nnq.QuantStub):
            child.qconfig = qcfg
            child.activation_post_process = qcfg.activation()
        else:
            _prepare_qat_recursive(child, qconfig_mapping, qat_map, full)


def _convert_recursive(
    module: nn.Module, mapping: dict[type, _FromFloat], always: tuple[type, ...]
) -> None:
    """Depth-first swap of every convertible child in place."""
    for name, child in list(module.named_children()):
        build = mapping.get(type(child))
        convertible = build is not None and (
            isinstance(child, always) or hasattr(child, "activation_post_process")
        )
        if convertible and build is not None:
            # ``setattr`` keeps the child's slot in place (LCD-551), so a
            # ``Sequential`` still runs it in the same turn; no direct
            # ``_modules`` write needed (LCD-556).
            setattr(module, name, build(child))
        else:
            _convert_recursive(child, mapping, always)
