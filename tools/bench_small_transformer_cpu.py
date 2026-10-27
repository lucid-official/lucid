"""Small causal Transformer training step on the CPU, split into phases.

The workload is the long-run parity recipe's GPT
(``lucid/test/parity/_long_harness.py``): two pre-norm blocks, width 64, four
heads, context 64, batch 32, vocabulary 32, AdamW with gradient clipping at
1.0.  Each step is timed as four phases — forward (model and loss), backward,
gradient clipping, optimizer (``zero_grad`` and ``step``) — so a gap to the
reference framework can be attributed before anyone opens a profiler.

Lucid and the reference (when it is installed) build the same model from the
same weights and see the same fixed batch.  Repeats alternate between the two
so machine load drifts onto both, and the best repeat of each is reported —
on a busy machine the minimum is the closest thing to the uncontended cost.

``--profile`` adds an op-level table from both profilers for one extra block
of steps.  Lucid's profiler records each backward node as an event of its own,
named after the node (``OpEvent.backward`` marks them), so the backward phase
is attributed in the same table.

Usage::

    .venv/bin/python3 tools/bench_small_transformer_cpu.py
    .venv/bin/python3 tools/bench_small_transformer_cpu.py --steps 50 --repeat 5
    .venv/bin/python3 tools/bench_small_transformer_cpu.py --profile --top 15
"""

import argparse
import os
import time
from collections import defaultdict
from types import ModuleType
from typing import Any

import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._fixtures.ref_framework import REF_DISPLAY_NAME, ref_module
from lucid.test.parity._long_harness import CONTEXT, GPT, TEXT_BATCH, VOCAB
from lucid.test.parity._mnist_harness import copy_state

PHASES = ("forward", "backward", "clip", "optimizer")


class Step:
    """One framework's model, optimizer, batch and a timed training step."""

    def __init__(self, framework: str, lib: ModuleType, model: Any) -> None:
        self.framework = framework
        self.lib = lib
        self.model = model
        if framework == "lucid":
            self.fn: Any = F
            self.clip: Any = lucid.nn.utils.clip_grad_norm_
            self.opt = GPT.make_optimizer(lucid.optim, model.parameters())
        else:
            self.fn = lib.nn.functional
            self.clip = lib.nn.utils.clip_grad_norm_
            self.opt = GPT.make_optimizer(lib.optim, model.parameters())
        batch = np.random.default_rng(0).integers(0, VOCAB, (TEXT_BATCH, CONTEXT + 1))
        if framework == "lucid":
            self.inputs = lucid.tensor(batch[:, :-1])
            self.targets = lucid.tensor(batch[:, 1:]).reshape(-1)
        else:
            self.inputs = lib.from_numpy(batch[:, :-1])
            self.targets = lib.from_numpy(batch[:, 1:]).reshape(-1)
        model.train()

    def run(self, totals: dict[str, float]) -> float:
        clock = time.perf_counter
        t0 = clock()
        out = self.model(self.inputs)
        loss = self.fn.cross_entropy(out.reshape(-1, VOCAB), self.targets)
        t1 = clock()
        loss.backward()
        t2 = clock()
        self.clip(self.model.parameters(), GPT.clip)
        t3 = clock()
        self.opt.step()
        self.opt.zero_grad()
        value = float(loss.item())
        t4 = clock()
        totals["forward"] += t1 - t0
        totals["backward"] += t2 - t1
        totals["clip"] += t3 - t2
        totals["optimizer"] += t4 - t3
        return value

    def block(self, steps: int) -> dict[str, float]:
        """Mean ms/step per phase over ``steps`` steps."""
        totals: dict[str, float] = defaultdict(float)
        for _ in range(steps):
            self.run(totals)
        return {k: totals[k] * 1000.0 / steps for k in PHASES}


def build(ref: ModuleType | None) -> list[Step]:
    lucid.manual_seed(0)
    mine = GPT.make_model(lucid, nn, F)
    steps = [Step("lucid", lucid, mine)]
    if ref is not None:
        theirs = GPT.make_model(ref, ref.nn, ref.nn.functional)
        copy_state(mine, theirs, ref)
        steps.append(Step(REF_DISPLAY_NAME, ref, theirs))
    return steps


def profile_lucid(step: Step, steps: int) -> dict[str, tuple[int, float]]:
    totals: dict[str, float] = defaultdict(float)
    with lucid.profiler.profile() as prof:
        for _ in range(steps):
            step.run(totals)
    agg: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for ev in prof.events():
        agg[ev.name][0] += 1
        agg[ev.name][1] += ev.time_us
    return {k: (int(v[0] / steps), v[1] / steps) for k, v in agg.items()}


def profile_ref(step: Step, ref: ModuleType, steps: int) -> dict[str, tuple[int, float]]:
    totals: dict[str, float] = defaultdict(float)
    with ref.profiler.profile(activities=[ref.profiler.ProfilerActivity.CPU]) as prof:
        for _ in range(steps):
            step.run(totals)
    out: dict[str, tuple[int, float]] = {}
    for ev in prof.key_averages():
        out[ev.key] = (int(ev.count / steps), ev.self_cpu_time_total / steps)
    return out


def print_profile(name: str, table: dict[str, tuple[int, float]], top: int) -> None:
    rows = sorted(table.items(), key=lambda kv: -kv[1][1])
    total = sum(v[1] for v in table.values())
    print(f"\n{name}: op self time per step, total {total / 1000:.2f} ms")
    print(f"  {'op':<44}{'calls':>6}{'us/step':>10}{'us/call':>10}{'share':>8}")
    for key, (calls, us) in rows[:top]:
        per = us / calls if calls else 0.0
        print(f"  {key[:44]:<44}{calls:>6}{us:>10.1f}{per:>10.1f}{us / total:>8.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=50, help="timed steps per repeat")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3, help="alternating repeats")
    parser.add_argument("--no-ref", action="store_true", help="time Lucid alone")
    parser.add_argument("--profile", action="store_true", help="op-level tables")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    ref = None if args.no_ref else ref_module()
    runs = build(ref)
    print(
        f"load average {os.getloadavg()[0]:.1f}  steps {args.steps}  "
        f"warmup {args.warmup}  repeat {args.repeat}"
    )
    if ref is not None:
        print(f"{REF_DISPLAY_NAME} {ref.__version__}, {ref.get_num_threads()} threads")

    for run in runs:
        run.block(args.warmup)
    results: dict[str, list[dict[str, float]]] = {r.framework: [] for r in runs}
    for _ in range(args.repeat):
        for run in runs:
            results[run.framework].append(run.block(args.steps))

    best: dict[str, dict[str, float]] = {}
    print(
        f"\n{'ms/step (best repeat)':<24}"
        + "".join(f"{p:>11}" for p in PHASES)
        + f"{'total':>11}  repeats (total)"
    )
    for name, blocks in results.items():
        pick = min(blocks, key=lambda b: sum(b.values()))
        best[name] = pick
        spread = " ".join(f"{sum(b.values()):.2f}" for b in blocks)
        print(
            f"{name:<24}"
            + "".join(f"{pick[p]:>11.2f}" for p in PHASES)
            + f"{sum(pick.values()):>11.2f}  {spread}"
        )
    if ref is not None:
        mine, theirs = best["lucid"], best[REF_DISPLAY_NAME]
        print(
            f"{'ratio':<24}"
            + "".join(f"{mine[p] / theirs[p]:>10.2f}x" for p in PHASES)
            + f"{sum(mine.values()) / sum(theirs.values()):>10.2f}x"
        )

    if args.profile:
        print_profile("lucid", profile_lucid(runs[0], args.steps), args.top)
        if ref is not None:
            print_profile(REF_DISPLAY_NAME, profile_ref(runs[1], ref, args.steps), args.top)


if __name__ == "__main__":
    main()
