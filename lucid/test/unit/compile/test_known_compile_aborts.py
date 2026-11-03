"""Compiled steps that used to abort the process inside MPSGraph, pinned (LCD-786, LCD-793).

Each was found by ``test_traced_sampler_compile_budget`` and is not about
samplers: an elementwise graph is enough.  MPSGraph's verifier rejected a node
the emitter produced and the process died on a failed assertion -- not a Python
exception, so a user's training script had nothing to catch.  Each runs in a
child process; a randomized small-graph fuzz over rank-0 and constant-reduction
mixes guards the whole class.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import lucid

_PRELUDE = """
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import make_step


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(lucid.ones(1))

    def forward(self, x):
        return BODY


def main():
    x = lucid.randn(4).to("metal")
    t = lucid.randn(4).to("metal")
    step = make_step(Probe().to("metal"), F.mse_loss)
    step(x, t).backward()
    print("compiled", not step.eager_only)


if __name__ == "__main__":
    main()
"""

_BODIES: dict[str, str] = {}

#: Aborts that were fixed; they stay as plain tests:
#: - ``rank0_pow_rank0`` / ``rank0_nextafter_rank0``: ``mps.reinterpret_cast`` rejected a
#:   0-d ``si32`` operand; ``reinterpret_cast_mps_tensor`` expands 0-d to ``(1,)`` and
#:   squeezes back to ``()`` (LCD-793).
#: - ``param_beside_constant_rank2_reduction``: a constant rank-2 full reduction was bound
#:   ``(1, 1)`` against a 0-d trace shape, so the backward reshaped it into the wrong element
#:   count (``mps.reshape``); a bound value now has its trace's shape (LCD-804).
_FIXED = {
    "rank0_pow_rank0": "x + (self.w.reshape(()) + 1.0) ** (self.w.reshape(()) * 2.0)",
    "rank0_nextafter_rank0": (
        "x + lucid.nextafter(self.w.reshape(()) + 1.0, self.w.reshape(()) * 2.0)"
    ),
    "param_beside_constant_rank2_reduction": (
        'x * self.w + lucid.ones(3, 3, device="metal").sum()'
    ),
}


_BROKEN = [
    pytest.param(
        name,
        marks=pytest.mark.xfail(
            strict=True, reason=f"LCD-786 FOUND: {name} aborts the process in MPSGraph"
        ),
    )
    for name in _BODIES
]


@pytest.mark.parametrize("name", [*_BROKEN, *_FIXED])
def test_compiled_step_does_not_abort(name: str, device_gpu_only: str) -> None:
    root = str(Path(lucid.__file__).resolve().parent.parent)
    done = subprocess.run(
        [sys.executable, "-c", _PRELUDE.replace("BODY", {**_BODIES, **_FIXED}[name])],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-300:]
    assert "compiled True" in done.stdout, done.stdout


_FUZZ_SCRIPT = """
import random
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import make_step

ATOMS = [
    "(self.w.reshape(()) + 1.5)",
    "(self.w.reshape(()) * 0.5 + 1.25)",
    "lucid.ones(2, 3, device='metal').sum()",
    "lucid.full((2, 2), 1.5, device='metal').mean()",
    "lucid.linspace(1.0, 2.0, 4, device='metal').amax()",
]
BINOPS = [
    "({a} + {b})",
    "({a} * {b})",
    "(({a}).abs() + 0.5) ** (({b}) * 0.25)",
    "lucid.nextafter({a}, {b})",
]


def build_expr(rng: random.Random) -> str:
    cur = rng.choice(ATOMS)
    for _ in range(2):
        rhs = rng.choice(ATOMS)
        cur = rng.choice(BINOPS).format(a=cur, b=rhs)
    return f"x * self.w + ({cur})"


def main() -> None:
    rng = random.Random(793)
    x = lucid.randn(4, device="metal")
    t = lucid.randn(4, device="metal")
    for idx in range(12):
        expr = build_expr(rng)

        class FuzzProbe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(lucid.ones(1))

            def forward(self, inp: lucid.Tensor) -> lucid.Tensor:
                return eval(expr, {"lucid": lucid, "self": self, "x": inp})

        model = FuzzProbe().to("metal")
        step = make_step(model, F.mse_loss)
        loss = step(x, t)
        loss.backward()
        assert not step.eager_only, f"case {idx} fell back: {expr}"
    print("fuzz_ok 12")


if __name__ == "__main__":
    main()
"""


def test_small_graph_rank0_and_constant_reduction_fuzz(device_gpu_only: str) -> None:
    del device_gpu_only
    root = str(Path(lucid.__file__).resolve().parent.parent)
    done = subprocess.run(
        [sys.executable, "-c", _FUZZ_SCRIPT],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-400:]
    assert "fuzz_ok 12" in done.stdout, done.stdout
