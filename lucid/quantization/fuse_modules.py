"""``fuse_modules`` — collapse Conv/BN/ReLU (or Linear/ReLU) runs before quant.

Given groups of dotted module names, each run is replaced in place: the
first member becomes the fused module and the rest become
:class:`~lucid.nn.Identity` (so the model's original ``forward`` still
routes through them harmlessly).  BatchNorm is folded into the preceding
convolution's weight via the existing eval-time fusion; a trailing ReLU is
absorbed into an ``nn.intrinsic`` fused module so the activation observer
later sees the post-ReLU range.
"""

import copy
from typing import TYPE_CHECKING, cast

import lucid.nn as nn
from lucid.nn.utils.fusion import fuse_conv_bn_eval

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lucid.quantization.qconfig import QConfig

_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_CONV_RANK = {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}


def _get_module(root: nn.Module, dotted: str) -> nn.Module:
    """Resolve a dotted submodule path to the module instance."""
    mod: nn.Module = root
    for part in dotted.split("."):
        mod = cast("nn.Module", getattr(mod, part))
    return mod


def _set_module(root: nn.Module, dotted: str, value: nn.Module) -> None:
    """Assign ``value`` at a dotted submodule path, preserving child order.

    ``setattr`` keeps an existing child's slot in place (LCD-551), so one
    unconditional call covers both the replace and the fresh-name case --
    no direct ``_modules`` write needed (LCD-556).
    """
    parts = dotted.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = cast("nn.Module", getattr(parent, part))
    setattr(parent, parts[-1], value)


def _conv_relu(conv: nn.Module, relu: nn.Module) -> nn.Module:
    """Wrap a conv + ReLU into the matching ``nn.intrinsic`` fused module."""
    import lucid.nn.intrinsic as nni

    mapping: dict[type, type] = {
        nn.Conv1d: nni.ConvReLU1d,
        nn.Conv2d: nni.ConvReLU2d,
        nn.Conv3d: nni.ConvReLU3d,
    }
    return cast("nn.Module", mapping[type(conv)](conv, relu))


def _fuse_run(mods: Sequence[nn.Module]) -> list[nn.Module]:
    """Return replacements for a fusible run (fused first, ``Identity`` rest)."""
    import lucid.nn.intrinsic as nni

    ids = [nn.Identity() for _ in range(len(mods) - 1)]
    first = mods[0]
    # Pattern — Conv (+ BN) (+ ReLU)
    if isinstance(first, _CONV_TYPES):
        conv: nn.Module = first
        rest = list(mods[1:])
        if rest and isinstance(
            rest[0], (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        ):
            conv = cast("nn.Module", fuse_conv_bn_eval(conv, rest.pop(0)))
        if rest and isinstance(rest[0], nn.ReLU):
            rest.pop(0)
            return [_conv_relu(conv, nn.ReLU()), *ids]
        return [conv, *ids]
    # Linear + ReLU
    if isinstance(first, nn.Linear) and len(mods) == 2 and isinstance(mods[1], nn.ReLU):
        return [nni.LinearReLU(first, nn.ReLU()), *ids]
    raise ValueError(
        f"fuse_modules: unsupported fusion pattern {[type(m).__name__ for m in mods]}"
    )


def fuse_modules(
    model: nn.Module,
    modules_to_fuse: Sequence[Sequence[str]],
    inplace: bool = False,
) -> nn.Module:
    r"""Collapse Conv/BN/ReLU (or Linear/ReLU) runs into single fused modules.

    ``fuse_modules`` is a pre-processing step you run **before**
    :func:`~lucid.quantization.prepare` in the static-PTQ workflow. It rewrites
    the model so each named run of adjacent layers becomes one module: the first
    member is replaced by the fused module and the rest by
    :class:`~lucid.nn.Identity` (so the model's original ``forward`` still routes
    through them harmlessly). BatchNorm is folded into the preceding convolution's
    weight via the eval-time fusion, and a trailing ReLU is absorbed into an
    ``nn.intrinsic`` fused module.

    Fusion matters for quantization on two counts. First, folding BN into the
    conv weight removes a separately-quantized layer and its rounding error, and
    it means there is only one weight tensor to quantize per block. Second,
    absorbing the ReLU makes the downstream activation observer see the *post-*
    ReLU range, so the calibrated grid is not wasted on the negative half that
    the ReLU would clip anyway — a direct accuracy win. Put the model in ``eval``
    mode first: the BN fold uses the frozen running statistics, which is only
    correct in eval.

    Parameters
    ----------
    model : nn.Module
        The model to fuse. Put it in ``eval`` mode first — BN folding is
        eval-time (it consumes the frozen running mean / variance).
    modules_to_fuse : sequence of name groups
        Each group is a list of dotted module names to fuse together, e.g.
        ``[["conv1", "bn1"], ["layer1.0.conv1", "layer1.0.bn1", "layer1.0.relu"]]``.
        Supported runs: ``[Conv, BN]``, ``[Conv, BN, ReLU]``, ``[Conv, ReLU]``,
        and ``[Linear, ReLU]``; any other pattern raises ``ValueError``.
    inplace : bool, default False
        If ``True`` fuse and return ``model`` itself; if ``False`` (default) fuse
        a :func:`copy.deepcopy` and leave the original untouched.

    Returns
    -------
    nn.Module
        The fused model, with each run's first module replaced by the fused
        module and the remaining members turned into ``Identity``.

    Notes
    -----
    - Order-preserving: each replacement takes the slot of the module it
      replaces, so a ``Sequential``'s execution order is unchanged.
    - The trailing members become ``Identity`` rather than being deleted, so any
      hand-written ``forward`` that still calls them keeps working.
    - This is the **eval / static-PTQ** fusion. For quantization-aware training
      use :func:`fuse_modules_qat`, which keeps Conv+BN trainable and folds BN
      per forward instead of freezing it.
    - Non-destructive by default (deep-copy).

    Examples
    --------
    >>> import lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> class Block(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv = nn.Conv2d(3, 8, 3)
    ...         self.bn = nn.BatchNorm2d(8)
    ...         self.relu = nn.ReLU()
    ...     def forward(self, x):
    ...         return self.relu(self.bn(self.conv(x)))
    >>> m = Block().eval()
    >>> fused = Q.fuse_modules(m, [["conv", "bn", "relu"]])
    >>> type(fused.conv).__name__               # Conv+BN+ReLU -> one fused module
    'ConvReLU2d'
    >>> type(fused.bn).__name__, type(fused.relu).__name__
    ('Identity', 'Identity')

    An unsupported run (here ``[BN, ReLU]`` with no leading Conv/Linear) raises:

    >>> Q.fuse_modules(m.eval(), [["bn", "relu"]])   # doctest: +SKIP
    ValueError: fuse_modules: unsupported fusion pattern ['BatchNorm2d', 'ReLU']

    See Also
    --------
    lucid.quantization.fuse_modules_qat : Trainable Conv/BN fusion for QAT.
    lucid.quantization.prepare : Run this before inserting observers.
    lucid.nn.utils.fusion.fuse_conv_bn_eval : The eval-time BN-into-conv fold.
    """
    model = model if inplace else copy.deepcopy(model)
    for group in modules_to_fuse:
        mods = [_get_module(model, name) for name in group]
        replacements = _fuse_run(mods)
        for name, repl in zip(group, replacements, strict=True):
            _set_module(model, name, repl)
    return model


def _fuse_run_qat(mods: Sequence[nn.Module], qconfig: QConfig) -> list[nn.Module]:
    """Return QAT replacements — Conv+BN(+ReLU) stay **trainable** (fold in forward)."""
    import lucid.nn.intrinsic.qat as nniqat

    ids = [nn.Identity() for _ in range(len(mods) - 1)]
    first = mods[0]
    if isinstance(first, _CONV_TYPES):
        rank = _CONV_RANK[type(first)]
        rest = list(mods[1:])
        if rest and isinstance(rest[0], _BN_TYPES):
            bn = rest.pop(0)
            relu = bool(rest and isinstance(rest[0], nn.ReLU))
            plain = (nniqat.ConvBn1d, nniqat.ConvBn2d, nniqat.ConvBn3d)[rank - 1]
            fused = (nniqat.ConvBnReLU1d, nniqat.ConvBnReLU2d, nniqat.ConvBnReLU3d)[
                rank - 1
            ]
            cls = fused if relu else plain
            return [cast("nn.Module", cls(first, bn, qconfig=qconfig)), *ids]
        # Conv (+ ReLU), no BN → plain intrinsic fused; prepare_qat makes it qat.
        if rest and isinstance(rest[0], nn.ReLU):
            return [_conv_relu(first, nn.ReLU()), *ids]
        return [first, *ids]
    if isinstance(first, nn.Linear) and len(mods) == 2 and isinstance(mods[1], nn.ReLU):
        import lucid.nn.intrinsic as nni

        return [nni.LinearReLU(first, nn.ReLU()), *ids]
    raise ValueError(
        f"fuse_modules_qat: unsupported pattern {[type(m).__name__ for m in mods]}"
    )


def fuse_modules_qat(
    model: nn.Module,
    modules_to_fuse: Sequence[Sequence[str]],
    qconfig: QConfig,
    inplace: bool = False,
) -> nn.Module:
    r"""Fuse for **QAT** — Conv+BN runs become trainable ``nn.intrinsic.qat`` modules.

    ``fuse_modules_qat`` is the fusion step of the quantization-aware-training
    workflow, run **before** :func:`~lucid.quantization.prepare_qat`. It mirrors
    :func:`fuse_modules` but keeps BatchNorm *learnable*. Where the eval-time
    :func:`fuse_modules` folds BN into the conv weight once and freezes it — fine
    for a frozen PTQ model — that would stop BN from training, which QAT needs.
    So here a ``[Conv, BN]`` / ``[Conv, BN, ReLU]`` run becomes a trainable
    ``ConvBn*`` / ``ConvBnReLU*`` module that folds BN **per forward**: the BN
    parameters keep updating and the fold is recomputed each step, all under the
    straight-through estimator of the fake-quant grid. ``[Conv, ReLU]`` and
    ``[Linear, ReLU]`` runs (no BN) become the plain ``nn.intrinsic`` fused
    module, which :func:`~lucid.quantization.prepare_qat` then swaps to its QAT
    form.

    Because the trainable ``ConvBn*`` modules fake-quantize at construction time,
    the ``qconfig`` must be supplied *here* rather than later — it is baked into
    the fused module so its weight fake-quant grid is active from the first
    forward. As with :func:`fuse_modules`, the first member of each run is
    replaced by the fused module and the rest become ``Identity``.

    Parameters
    ----------
    model : nn.Module
        The model to fuse. Kept **trainable** (unlike :func:`fuse_modules`, do
        not put it in eval — BN must keep learning).
    modules_to_fuse : sequence of name groups
        Dotted-name groups exactly as in :func:`fuse_modules` — e.g.
        ``[["conv1", "bn1"], ["layer1.0.conv1", "layer1.0.bn1"]]``. Same supported
        runs: ``[Conv, BN]``, ``[Conv, BN, ReLU]``, ``[Conv, ReLU]``,
        ``[Linear, ReLU]``.
    qconfig : QConfig
        The QAT recipe handed to the fused ``ConvBn*`` modules; they fake-quant
        at construction, so it is needed at fusion time rather than at
        :func:`~lucid.quantization.prepare_qat` time.
    inplace : bool, default False
        If ``True`` fuse and return ``model`` itself; if ``False`` (default) fuse
        a :func:`copy.deepcopy` and leave the original untouched.

    Returns
    -------
    nn.Module
        The fused, still-trainable model — hand it to
        :func:`~lucid.quantization.prepare_qat`, fine-tune, then
        :func:`~lucid.quantization.convert`.

    Notes
    -----
    - Conv+BN stays a single trainable module that folds BN **per forward**; BN
      is never frozen (contrast :func:`fuse_modules`, which folds once at eval).
    - The ``qconfig`` argument is mandatory precisely because the fused
      ``ConvBn*`` fake-quantizes at construction — there is no later hook to pass
      it through.
    - Order-preserving splice via ``_modules``; trailing members become
      ``Identity`` so the original ``forward`` still routes through them.
    - Non-destructive by default (deep-copy).

    Examples
    --------
    >>> import lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> class Block(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv = nn.Conv2d(3, 8, 3)
    ...         self.bn = nn.BatchNorm2d(8)
    ...     def forward(self, x):
    ...         return self.bn(self.conv(x))
    >>> m = Block()                              # kept in train mode
    >>> qcfg = Q.get_default_qat_qconfig()
    >>> fused = Q.fuse_modules_qat(m, [["conv", "bn"]], qcfg)
    >>> type(fused.conv).__name__               # trainable Conv+BN, folds per forward
    'ConvBn2d'
    >>> qat = Q.prepare_qat(fused)               # continue the QAT workflow

    See Also
    --------
    lucid.quantization.fuse_modules : Eval-time (static-PTQ) fusion; freezes BN.
    lucid.quantization.prepare_qat : Run this after fusing, before fine-tuning.
    lucid.quantization.convert : Bake the fine-tuned QAT model into int8.
    """
    model = model if inplace else copy.deepcopy(model)
    for group in modules_to_fuse:
        mods = [_get_module(model, name) for name in group]
        replacements = _fuse_run_qat(mods, qconfig)
        for name, repl in zip(group, replacements, strict=True):
            _set_module(model, name, repl)
    return model
