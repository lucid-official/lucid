"""Shared BatchNorm running-stats compile plumbing.

A fused-momentum BatchNorm (``track_running_stats=True`` and a finite
``momentum``) traces as a 5-input / 3-output ``batch_norm`` node: the running
buffers come in as ``inputs[3]/[4]`` and the EMA-updated stats come out as
``outputs[1]/[2]``.  The compile entry points surface those extra outputs and
write them back into the live module buffers each run so ``model.eval()`` reads
fresh stats (see [[retro-3-5-bn-runningstats-compile-writeback]]).

Two running-stats configurations CANNOT ride the graph and must be handled by
the caller (eager fallback for ``make_step`` / ``CompiledModule``; a clear error
for ``fused_step`` / ``compiled_step`` which have no fallback):

* a BatchNorm with ``track_running_stats=True`` + ``momentum=None`` (cumulative
  moving average) — the update needs ``num_batches_tracked`` as a host scalar (a
  GPU sync), so it can't be lowered into a pure graph.
  :func:`model_has_cumulative_bn` detects it.

A ``track_running_stats=False`` BatchNorm, by contrast, keeps NO buffers and uses
batch statistics in both train and eval — it traces as a plain 3-input node with
nothing to write back, so it compiles unchanged.  The distinction between these
two 3-input cases is invisible in the trace IR, so the discriminator is the model
itself, not the trace.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from lucid._C.engine import TensorImpl
    from lucid.nn.module import Module

__all__ = [
    "BnWriteback",
    "model_has_cumulative_bn",
    "model_has_tracking_bn",
    "bn_writeback_targets",
]

_BN_NAMES: tuple[str, ...] = ("batch_norm", "batch_norm1d", "batch_norm3d")


def model_has_cumulative_bn(model: Module) -> bool:
    """Return ``True`` if any submodule keeps running stats via cumulative moving
    average: a BatchNorm-family layer with ``track_running_stats=True`` and
    ``momentum is None``.

    Such a layer's running-stats update can't be lowered into the compiled graph
    (it reads ``num_batches_tracked`` as a host scalar), so the caller must fall
    back to eager (``make_step`` / ``CompiledModule``) or raise (``fused_step`` /
    ``compiled_step``).

    The test is the layer's update rule, not the shape of its attributes.  An
    InstanceNorm with ``momentum=None`` passes no weight to its update, as the
    reference does, so it reads no count; matched by attribute it sent a
    compilable step to eager, and made ``compiled_step`` refuse it (LCD-521).
    """
    from lucid.nn.modules.normalization import _BatchNormBase

    return any(
        isinstance(m, _BatchNormBase) and m.track_running_stats and m.momentum is None
        for m in model.modules()
    )


def model_has_tracking_bn(model: Module) -> bool:
    """Return ``True`` if any submodule tracks running stats
    (``track_running_stats=True``, any momentum).

    Used by the forward-only :class:`CompiledModule`: in training mode such a
    layer's EMA must advance, but the forward-only compile path has no write-back
    hook, so the caller falls back to eager.  (In eval mode BatchNorm dispatches
    the distinct ``batch_norm_eval`` op, which is unaffected.)
    """
    for m in model.modules():
        if getattr(m, "track_running_stats", False):
            return True
    return False


def bn_writeback_targets(
    graph: object, ext: dict[int, TensorImpl]
) -> list[tuple[int, int, TensorImpl]]:
    """Collect the running-stat write-back targets for every fused-momentum BN.

    One ``(feed_id, out_id, buffer_impl)`` triple is produced per live running-stat
    buffer. Repeated 5-input / 3-output nodes chain their EMA values; retain the
    original external feed and the final output, not intermediate writes.
    A 3-input node (``track_running_stats=False``) keeps no buffers and contributes
    nothing.

    Parameters
    ----------
    graph : TraceGraph
        The recorded trace whose ``ops`` are scanned for BatchNorm nodes.
    ext : dict[int, TensorImpl]
        The trace's external-feed map (trace id to live TensorImpl), used to
        resolve each running-stat feed id to the module buffer to write back.

    Returns
    -------
    list[tuple[int, int, TensorImpl]]
        ``(feed_id, out_id, buffer_impl)`` per slot, where ``feed_id`` is the
        original trace id of the live buffer fed in, ``out_id`` the final
        EMA output (``outputs[1]/[2]``), and ``buffer_impl`` the live module
        buffer the new value must be copied/swapped into.
    """
    targets: dict[int, tuple[int, int, TensorImpl]] = {}
    owners = {feed_id: (feed_id, impl) for feed_id, impl in ext.items()}
    ops = getattr(graph, "ops", [])
    for node in ops:
        if node.name not in _BN_NAMES:
            continue
        if len(node.inputs) >= 5 and len(node.outputs) >= 3:
            for in_idx, out_idx in ((3, 1), (4, 2)):
                feed_id = int(node.inputs[in_idx])
                owner = owners.get(feed_id)
                if owner is None:
                    continue
                root_id, impl = owner
                out_id = int(node.outputs[out_idx].id)
                owners[out_id] = owner
                # Only the last update is written to each live buffer.
                targets[root_id] = (root_id, out_id, impl)
    return list(targets.values())


def bn_counter_targets(
    model: Module, graph: object, ext: dict[int, TensorImpl]
) -> list[tuple[Module, int]]:
    """Map observed training BN nodes to their live counters and call counts.

    A registered but uncalled layer must not advance. Counting trace nodes also
    preserves repeated calls to the same layer; walking modules alone cannot.

    Args:
        model: Module tree that owns the live BatchNorm buffers.
        graph: Traced graph whose BatchNorm nodes record actual calls.
        ext: Mapping from external graph value IDs to their tensor implementations.

    Returns:
        ``(module, call_count)`` pairs for observed layers with live counters.
    """
    from lucid._dispatch import _unwrap

    counts: dict[int, int] = {}
    owners = dict(ext)
    for node in getattr(graph, "ops", []):
        if node.name in _BN_NAMES and len(node.inputs) >= 5:
            impl = owners.get(int(node.inputs[3]))
            if impl is not None:
                if len(node.outputs) >= 3:
                    owners[int(node.outputs[1].id)] = impl
                key = id(impl)
                counts[key] = counts.get(key, 0) + 1
    targets: list[tuple[Module, int]] = []
    for module in model.modules():
        mean = module._buffers.get("running_mean")
        counter = module._buffers.get("num_batches_tracked")
        if mean is not None and counter is not None:
            count = counts.get(id(_unwrap(mean)), 0)
            if count:
                targets.append((module, count))
    return targets


def advance_bn_counters(targets: list[tuple[Module, int]]) -> None:
    """Advance observed BatchNorm counters after a successful executable run.

    This runs outside the traced graph so failed executions leave every live
    counter unchanged.  Each counter advances in place, as the eager layer's
    does: rebinding the buffer to ``counter + count`` left a reference taken
    earlier — a ``state_dict()``, a hook — at the old count (LCD-503).

    Args:
        targets: Observed ``(module, call_count)`` pairs from
            :func:`bn_counter_targets`.
    """
    for module, count in targets:
        counter = module._buffers["num_batches_tracked"]
        assert counter is not None
        counter.add_(count)


@final
@dataclass(frozen=True, slots=True)
class BnWriteback:
    """The BatchNorm state a traced call updates, and its write-back after each run.

    Under a tracer a training-mode BatchNorm leaves its buffers alone: the new
    statistics are trace values (:func:`bn_writeback_targets`) and the counter
    is bookkeeping the trace omits (:func:`bn_counter_targets`).  An entry point
    that lowers the trace asks the executable for :attr:`out_ids` as extra
    outputs and hands them to :meth:`apply` after every run that succeeded, so
    the buffers move once per call as an eager call's do.  One that never asks
    leaves them at their values from before the first call (LCD-719: the
    segmented ``make_step`` trained with ``running_var`` stuck at its initial
    ones and ``num_batches_tracked`` at 0).

    Attributes
    ----------
    buffers : tuple of TensorImpl
        The live ``running_mean`` / ``running_var`` buffers, in the order of
        :attr:`out_ids`.
    out_ids : tuple of int
        The trace id of each buffer's final statistic.
    counters : tuple of (Module, int)
        Each layer the trace ran in training mode, and how many times.
    """

    buffers: tuple[TensorImpl, ...] = ()
    out_ids: tuple[int, ...] = ()
    counters: tuple[tuple[Module, int], ...] = ()

    @classmethod
    def of(cls, model: Module, graph: object, ext: dict[int, TensorImpl]) -> BnWriteback:
        """Read it off a finished trace of ``model``.

        Parameters
        ----------
        model : Module
            The tree that owns the BatchNorm layers.
        graph : TraceGraph
            The recorded trace.
        ext : dict of int to TensorImpl
            The trace's external feeds.

        Returns
        -------
        BnWriteback
            Empty when the trace ran no BatchNorm that keeps running stats.
        """
        targets = bn_writeback_targets(graph, ext)
        return cls(
            buffers=tuple(impl for _, _, impl in targets),
            out_ids=tuple(out_id for _, out_id, _ in targets),
            counters=tuple(bn_counter_targets(model, graph, ext)),
        )

    def apply(self, stats: Sequence[TensorImpl]) -> None:
        """Write a run's statistics into the buffers and advance the counters.

        The statistics go in through :func:`~lucid.compile._core.buffer_writes.write_back`,
        the path every other tensor a step updates from outside the trace
        takes — in place, cast to the buffer's dtype.

        Parameters
        ----------
        stats : sequence of TensorImpl
            The executable's outputs for :attr:`out_ids`, in order.
        """
        from lucid.compile._core.buffer_writes import write_back

        write_back(list(self.buffers), list(stats))
        advance_bn_counters(list(self.counters))
