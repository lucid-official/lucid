"""Where a compiled rank-5 multiply loses time against its rank-4 twin (LCD-935).

``test_rank_canon.py::test_a_rank_5_multiply_is_not_slower_than_its_rank_4_twin`` measured the
compiled ``(x * w).sum(-1)`` over a ``(6144, 16, 2, 16, 2)`` feed at 2.1x its ``(6144, 16, 2, 32)``
twin on an M1 Pro and at 4.4x on the hosted runner's virtual GPU.  Every path ``RankCanon``
added (920a-c, 928) is gated by one setting, ``LUCID_COMPILE_RANK_CANON_MIN_LEAD`` (``-1`` turns
all of them off), and none has a switch of its own, so a path is isolated by the graph it is
built from.  Each row says which path it exercises (the IR facts printed first confirm it):

* **A** the guard's own graph: ``w`` a parameter, the product reduced over its last axis (size 2).
  The guard's twin reduces an axis of size 32, so it is not a twin in the reduction:
  **A same-reduce** is a rank-4 graph that reduces the same size-2 axis, and **A (-2,-1)** is
  the rank-5 graph reduced over the same 32 elements per output as the twin;
* **B** the same with ``w`` a feed instead of a parameter (a parameter keeps its logical shape);
* **C** ``x`` also read whole by ``x.sum()``: the feed is not read by views alone, so LCD-928's
  declared feed shape is off for it (and the twin pays the same extra read);
* **D** the rank-5 product returned itself, LCD-920c's rank-4 output binding, against **D flat**,
  the same product reshaped to rank 1 inside the graph, which is what that binding does.

Every variant is timed in alternating rounds against all the others (a load spike lands on every
one); the medians and minimums over every sample are printed beside the IR facts.  Run it where
the guard failed, through ``.github/workflows/rank5-probe.yml`` (``workflow_dispatch``), or here::

    python -m tools.probe_rank5_multiply [--rounds N] [--reps N] [--quick]
"""

import argparse
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile import _tracing
from lucid.test.unit.compile._helpers import COMPILE_DEVICE, with_env

ENV = "LUCID_COMPILE_RANK_CANON_MIN_LEAD"

Shape = tuple[int, ...]
Fn = Callable[[lucid.Tensor], lucid.Tensor]

S5: Shape = (6144, 16, 2, 16, 2)
S4: Shape = (6144, 16, 2, 32)
#: Rank 4, with the last axis the rank-5 case reduces (size 2), over as many rows.
S4_SAME: Shape = (98304, 2, 16, 2)


def _reduce_last(product: lucid.Tensor) -> lucid.Tensor:
    return product.sum(dim=-1)


def _reduce_last_two(product: lucid.Tensor) -> lucid.Tensor:
    return product.sum(dim=(-2, -1))


def _whole(product: lucid.Tensor) -> lucid.Tensor:
    return product


def _flat(product: lucid.Tensor) -> lucid.Tensor:
    return product.reshape(-1)


def _total(x: lucid.Tensor) -> lucid.Tensor:
    return x.sum()


@dataclass(frozen=True)
class Variant:
    name: str
    shape: Shape
    #: What reads the product ``x * w``.
    tail: Fn
    #: ``"param"``: ``w`` is a module parameter; ``"feed"``: a call argument.
    operand: str = "param"
    #: Also returned, read from ``x`` itself: a feed read as it is keeps its logical shape.
    keep: Fn | None = None
    #: ``LUCID_COMPILE_RANK_CANON_MIN_LEAD`` for this variant (``None`` = unset = the default).
    gate: str | None = None


VARIANTS: tuple[Variant, ...] = (
    Variant("A  r5 default", S5, _reduce_last),
    Variant("A  r5 pass off (-1)", S5, _reduce_last, gate="-1"),
    Variant("A  r5 reduce (-2,-1)", S5, _reduce_last_two),
    Variant("A  r4 twin (the guard's)", S4, _reduce_last),
    Variant("A  r4 same-reduce", S4_SAME, _reduce_last),
    Variant("B  r5 default, w a feed", S5, _reduce_last, operand="feed"),
    Variant("B  r5 pass off, w a feed", S5, _reduce_last, operand="feed", gate="-1"),
    Variant("B  r4 twin, w a feed", S4, _reduce_last, operand="feed"),
    Variant("C  r5 default, x also summed", S5, _reduce_last, keep=_total),
    Variant("C  r4 twin, x also summed", S4, _reduce_last, keep=_total),
    Variant("D  r5 returned default", S5, _whole),
    Variant("D  r5 returned pass off", S5, _whole, gate="-1"),
    Variant("D  r5 flat in graph", S5, _flat),
    Variant("D  r4 returned twin", S4, _whole),
)

#: The twin the ratio columns divide by: the variants in a group share a graph but for rank.
TWIN_OF = {
    "A  r5 default": "A  r4 twin (the guard's)",
    "A  r5 pass off (-1)": "A  r4 twin (the guard's)",
    "A  r5 reduce (-2,-1)": "A  r4 twin (the guard's)",
    "A  r4 same-reduce": "A  r4 twin (the guard's)",
    "B  r5 default, w a feed": "B  r4 twin, w a feed",
    "B  r5 pass off, w a feed": "B  r4 twin, w a feed",
    "C  r5 default, x also summed": "C  r4 twin, x also summed",
    "D  r5 returned default": "D  r4 returned twin",
    "D  r5 returned pass off": "D  r4 returned twin",
    "D  r5 flat in graph": "D  r4 returned twin",
}


def _filled(shape: Shape, phase: float = 0.0) -> lucid.Tensor:
    n = 1
    for s in shape:
        n *= s
    return (lucid.sin(lucid.arange(n, dtype=lucid.float32) * 0.37 + phase) + 1.5).reshape(*shape)


class _Product(nn.Module):
    """``tail(x * w)`` with a parameter ``w``; ``keep`` also returns ``keep(x)``."""

    def __init__(self, shape: Shape, tail: Fn, keep: Fn | None) -> None:
        super().__init__()
        self.w = nn.Parameter(_filled(shape, 0.5).to(COMPILE_DEVICE))
        self.tail = tail
        self.keep = keep

    @override
    def forward(self, x: lucid.Tensor) -> lucid.Tensor | tuple[lucid.Tensor, lucid.Tensor]:
        y = self.tail(x * self.w)
        return y if self.keep is None else (y, self.keep(x))


def _callable_of(variant: Variant) -> tuple[Callable[..., object], tuple[lucid.Tensor, ...]]:
    x = _filled(variant.shape).to(COMPILE_DEVICE)
    if variant.operand == "feed":
        w = _filled(variant.shape, 0.5).to(COMPILE_DEVICE)
        return (lambda a, b: variant.tail(a * b)), (x, w)
    model = _Product(variant.shape, variant.tail, variant.keep).to(COMPILE_DEVICE).eval()
    return model, (x,)


def _build(variant: Variant) -> Callable[[], float]:
    """A self-synchronising call of the variant's compiled graph, under its own gate."""
    fn, args = _callable_of(variant)
    compiled = lucid.compile(fn)

    def call() -> float:
        with lucid.no_grad():
            out = compiled(*args)
        first = out[0] if isinstance(out, tuple) else out
        # The guard's own read: one slice of the result, summed, so the device work is waited on.
        return float(first[0].sum().item())

    return with_env(call, ENV, variant.gate)


def _facts(variant: Variant) -> str:
    """The graph the builder emits: ops and shapes, and which feeds only views read."""
    fn, args = _callable_of(variant)
    with lucid.no_grad(), _tracing() as tracer:
        fn(*args)
    canon = with_env(
        lambda: list(_C_engine.compile.canonicalize_rank(tracer.graph, tracer.external_feeds).ops),
        ENV,
        variant.gate,
    )
    ops = canon()
    chain = " > ".join(f"{n.name}{tuple(n.outputs[0].shape)}" for n in ops)
    by_views = [
        fid
        for fid in tracer.external_feeds
        if {n.name for n in ops if fid in n.inputs} == {"reshape"}
    ]
    return f"{chain}\n      feeds read by views alone (declared at their views' shape): {by_views}"


def _measure(
    calls: dict[str, Callable[[], float]], rounds: int, reps: int, warmup: int
) -> dict[str, list[float]]:
    for call in calls.values():
        for _ in range(warmup):
            call()
    samples: dict[str, list[float]] = {name: [] for name in calls}
    names = list(calls)
    for rnd in range(rounds):
        for name in names if rnd % 2 == 0 else reversed(names):
            for _ in range(reps):
                t0 = time.perf_counter()
                calls[name]()
                samples[name].append((time.perf_counter() - t0) * 1e3)
    return samples


def _sysctl(key: str) -> str:
    done = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
    return done.stdout.strip()


def _host() -> str:
    one, five, fifteen = os.getloadavg()
    return "\n".join(
        [
            f"platform   {platform.platform()}",
            f"cpu        {_sysctl('machdep.cpu.brand_string')}",
            f"hypervisor {_sysctl('kern.hv_vmm_present')} (kern.hv_vmm_present)",
            f"gpu        {lucid.metal.get_device_name()}",
            f"load       {one:.2f} {five:.2f} {fifteen:.2f}",
        ]
    )


def _table(med: dict[str, float], low: dict[str, float]) -> str:
    width = max(len(v.name) for v in VARIANTS)
    head = f"{'variant':<{width}}  {'median':>9}  {'min':>9}  {'med/twin':>9}  {'min/twin':>9}"
    rows = [head]
    for v in VARIANTS:
        twin = TWIN_OF.get(v.name)
        ratios = (
            f"{med[v.name] / med[twin]:>8.2f}x  {low[v.name] / low[twin]:>8.2f}x" if twin else ""
        )
        rows.append(f"{v.name:<{width}}  {med[v.name]:>6.3f} ms  {low[v.name]:>6.3f} ms  {ratios}")
    return "\n".join(rows)


def _reading(med: dict[str, float]) -> str:
    """The comparisons the table is for, each as the ratio or the difference of two medians."""
    r5, twin = med["A  r5 default"], med["A  r4 twin (the guard's)"]
    same = med["A  r4 same-reduce"]
    feed_kept = (med["C  r5 default, x also summed"] - r5) - (
        med["C  r4 twin, x also summed"] - twin
    )
    return "\n".join(
        [
            f"the guard's ratio (A r5 default / A twin)           {r5 / twin:>6.2f}x",
            f"  of it, the reduction's axis (same-reduce / twin)  {same / twin:>6.2f}x",
            f"  rank alone (A r5 default / same-reduce)           {r5 / same:>6.2f}x",
            f"  rank alone, same reduction (A (-2,-1) / twin)     "
            f"{med['A  r5 reduce (-2,-1)'] / twin:>6.2f}x",
            f"the pass's gain (A pass off / A default)            "
            f"{med['A  r5 pass off (-1)'] / r5:>6.2f}x",
            f"w a parameter, not a feed (A default - B default)   "
            f"{r5 - med['B  r5 default, w a feed']:>+6.3f} ms",
            f"LCD-928 off, net of the extra read (C - A, r5 - r4) {feed_kept:>+6.3f} ms",
            f"LCD-920c (D returned default - D flat in graph)     "
            f"{med['D  r5 returned default'] - med['D  r5 flat in graph']:>+6.3f} ms",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Time a compiled rank-5 multiply and its twins.")
    parser.add_argument("--rounds", type=int, default=12, help="alternating rounds (default 12)")
    parser.add_argument("--reps", type=int, default=5, help="calls per variant per round")
    parser.add_argument("--warmup", type=int, default=3, help="untimed calls per variant")
    parser.add_argument("--quick", action="store_true", help="4 rounds: a smoke run")
    args = parser.parse_args()
    rounds = 4 if args.quick else args.rounds

    print(_host())
    print(f"\nrank-5 {S5}, the guard's twin {S4}, same-reduce twin {S4_SAME}")
    print("\n== graphs after RankCanon ==")
    for v in VARIANTS:
        print(f"  {v.name}\n      {_facts(v)}")

    _C_engine.compile.session_cache_clear()
    calls = {v.name: _build(v) for v in VARIANTS}
    print(f"\n== timing: {rounds} rounds x {args.reps} calls, alternating, {args.warmup} warmup ==")
    samples = _measure(calls, rounds, args.reps, args.warmup)
    med = {name: statistics.median(ms) for name, ms in samples.items()}
    low = {name: min(ms) for name, ms in samples.items()}
    print(_table(med, low))
    print("\n== reading ==")
    print(_reading(med))
    print(f"\nload at the end: {os.getloadavg()[0]:.2f}")


if __name__ == "__main__":
    main()
