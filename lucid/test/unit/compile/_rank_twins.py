"""Operands and rank-pass IR shared by the unit and performance twins."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from unittest import mock

import lucid
from lucid._C import engine as _C_engine
from lucid.compile import _tracing

Shape = tuple[int, ...]
ENV = "LUCID_COMPILE_RANK_CANON_MIN_LEAD"


@dataclass(frozen=True)
class Side:
    """One compiled call: ``fn`` on ``args``."""

    fn: Callable[..., lucid.Tensor]
    args: tuple[lucid.Tensor, ...]


@dataclass(frozen=True)
class RankTwin:
    #: the recorded op the rank pass rewrites (the node both sides must hold)
    op: str
    wide: Side
    twin: Side

    def __post_init__(self) -> None:
        wide = [arg.numel() for arg in self.wide.args]
        twin = [arg.numel() for arg in self.twin.args]
        assert wide == twin, f"{self.op} operand element counts differ: {wide} and {twin}"


def _trace(
    fn: Callable[..., lucid.Tensor], *args: lucid.Tensor
) -> tuple[_C_engine.compile.TraceGraph, dict[int, _C_engine.TensorImpl]]:
    with lucid.no_grad(), _tracing() as tracer:
        fn(*args)
    return tracer.graph, tracer.external_feeds


IrOps = list[_C_engine.compile.OpNode]


def rank_ir(side: Side, gate: str | None) -> tuple[IrOps, dict[int, Shape]]:
    """The pass's output for ``side`` and the shape of every tensor id in it (feeds included)."""
    graph, feeds = _trace(side.fn, *side.args)
    with mock.patch.dict(os.environ, {} if gate is None else {ENV: gate}):
        ops = list(_C_engine.compile.canonicalize_rank(graph, feeds, False).ops)
    shapes = {fid: tuple(feed.shape) for fid, feed in feeds.items()}
    shapes |= {out.id: tuple(out.shape) for node in ops for out in node.outputs}
    return ops, shapes


def computes(ops: IrOps) -> list[tuple[str, Shape, dict[str, object]]]:
    """What the graph computes, views aside: each op but a reshape, its answer's shape and attrs."""
    return [(n.name, tuple(n.outputs[0].shape), dict(n.attrs)) for n in ops if n.name != "reshape"]


def wide_ops(ops: IrOps, shapes: dict[int, Shape]) -> list[str]:
    """The ops, a reshape (a free view) aside, that read or write a tensor of rank >= 5."""
    return [
        n.name
        for n in ops
        if n.name != "reshape"
        and any(len(shapes[i]) > 4 for i in (*n.inputs, *(out.id for out in n.outputs)))
    ]


def _filled(shape: Shape, phase: float = 0.0) -> lucid.Tensor:
    n = 1
    for s in shape:
        n *= s
    return (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 0.37 + phase) + 1.5).reshape(*shape)
