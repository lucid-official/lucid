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

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

if TYPE_CHECKING:
    from lucid._C.engine import TensorImpl
    from lucid._C.engine.compile import TraceGraph, Tracer
    from lucid.nn.module import Module

__all__ = [
    "BnWriteback",
    "bn_statistics_lost",
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
    from lucid.compile._core.signature import tree_modules
    from lucid.nn.modules.normalization import _BatchNormBase

    return any(
        isinstance(m, _BatchNormBase) and m.track_running_stats and m.momentum is None
        for m in tree_modules(model)
    )


def model_has_tracking_bn(model: Module) -> bool:
    """Return ``True`` if any submodule tracks running stats
    (``track_running_stats=True``, any momentum).

    Used by :class:`CompiledModule`, which runs a norm module in training mode
    eagerly: carrying its statistics, as it does a functional batch norm's, is
    not proven for that entry (LCD-882).  (In eval mode BatchNorm dispatches the
    distinct ``batch_norm_eval`` op, which is unaffected.)
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


def bn_statistics_lost(graph: TraceGraph, carried: Collection[int]) -> bool:
    """Whether a batch norm handed out a statistic that nothing writes back.

    The statistics of a tensor with no feed of its own, a view taken in the call
    or a tensor the call made, have nowhere to go: compiled, they vanish, where
    eager updates the view's bytes or the tensor the call keeps.  A value counts
    as written back if a carried one is computed from it: ``rm.add_(1)`` after
    the norm carries the norm's statistic with it.

    Parameters
    ----------
    graph : TraceGraph
        The recorded trace.
    carried : collection of int
        The trace ids a run writes back.

    Returns
    -------
    bool
        ``True`` when one of the trace's batch norms would lose a statistic.
    """
    reaches = set(carried)
    for node in reversed(graph.ops):
        outputs = [int(out.id) for out in node.outputs]
        keeps = node.name in _BN_NAMES and len(node.inputs) >= 5 and len(outputs) >= 3
        if keeps and not reaches.issuperset(outputs[1:3]):
            return True
        if not reaches.isdisjoint(outputs):
            reaches.update(int(i) for i in node.inputs)
    return False


def bn_counter_targets(model: Module, tracer: Tracer) -> list[tuple[Module, int]]:
    """Map observed training BN nodes to their live counters and call counts.

    A registered but uncalled layer must not advance. Counting trace nodes also
    preserves repeated calls to the same layer; walking modules alone cannot.

    Only a training ``_BatchNormBase`` counts, the one eager layer that does:
    ``F.batch_norm`` on buffers that happen to be named ``running_mean``, or on an
    evaluating layer's, leaves any counter alone.  A layer's statistics are found
    by the feed they came in as, or for ones built inside the trace (a lazy
    layer's first call) by the value its buffer last took.

    Args:
        model: Module tree that owns the live BatchNorm buffers.
        tracer: The finished trace.

    Returns:
        ``(module, call_count)`` pairs for observed layers with live counters.
    """
    from lucid._dispatch import _unwrap
    from lucid.compile._core.signature import tree_modules
    from lucid.nn.modules.normalization import _BatchNormBase

    counts: dict[int, int] = {}
    roots: dict[int, int] = {}
    for node in tracer.graph.ops:
        if node.name in _BN_NAMES and len(node.inputs) >= 5:
            fed = int(node.inputs[3])
            root = roots.get(fed, fed)
            counts[root] = counts.get(root, 0) + 1
            if len(node.outputs) >= 3:
                roots[int(node.outputs[1].id)] = root
    feeds = {id(impl): tid for tid, impl in tracer.external_feeds.items()}
    targets: list[tuple[Module, int]] = []
    for module in tree_modules(model):
        if not (isinstance(module, _BatchNormBase) and module.training):
            continue
        mean = module._buffers.get("running_mean")
        if mean is None or module._buffers.get("num_batches_tracked") is None:
            continue
        impl = _unwrap(mean)
        latest = tracer.lookup_id(impl)
        # Both: an in-place write to the buffer before the norm starts a chain
        # of its own, and one after it leaves the buffer's last value off the
        # norm's chain.
        starts = {feeds.get(id(impl)), None if latest is None else roots.get(latest, latest)}
        count = sum(counts.get(start, 0) for start in starts if start is not None)
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
    def of(cls, model: Module, tracer: Tracer) -> BnWriteback:
        """Read it off a finished trace of ``model``.

        Parameters
        ----------
        model : Module
            The tree that owns the BatchNorm layers.
        tracer : Tracer
            The finished trace.

        Returns
        -------
        BnWriteback
            Empty when the trace ran no BatchNorm that keeps running stats.
        """
        targets = bn_writeback_targets(tracer.graph, tracer.external_feeds)
        return cls(
            buffers=tuple(impl for _, _, impl in targets),
            out_ids=tuple(out_id for _, out_id, _ in targets),
            counters=tuple(bn_counter_targets(model, tracer)),
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
