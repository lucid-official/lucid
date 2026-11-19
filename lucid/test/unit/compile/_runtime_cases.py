"""Shared inputs for replay behavior and its steady-state cost."""

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile import _tracing
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _deep_model(n: int = 200) -> nn.Module:
    return nn.Sequential(*[nn.Linear(4, 4) for _ in range(n)])


def _cached_executable(
    scale: float,
) -> tuple[_C_engine.compile.CompiledExecutable, list[_C_engine.TensorImpl]]:
    """An executable of ``x * scale + 1`` through the session cache, and its feeds."""
    x = lucid.ones(4, device=COMPILE_DEVICE)
    step = lucid.ones_like(x)
    with lucid.no_grad(), _tracing() as tracer:
        y = x * scale + step
    lucid.eval(x, step, y)
    exe = _C_engine.compile.compile_or_cached(tracer.graph, tracer.external_feeds, False, [], [])
    assert exe is not None
    by_id = {int(tid): impl for tid, impl in tracer.external_feeds.items()}
    return exe, [by_id[int(tid)] for tid in exe.input_ids]
