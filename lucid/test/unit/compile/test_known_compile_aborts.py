"""Compiled steps that abort the process inside MPSGraph, pinned (LCD-786 FOUND).

Both were found by ``test_traced_sampler_compile_budget`` and are not about
samplers: an elementwise graph is enough.  MPSGraph's verifier rejects a node
the emitter produced and the process dies on a failed assertion -- not a Python
exception, so a user's training script has nothing to catch.  Each runs in a
child process; strict ``xfail`` so that the fix shows up as an XPASS to delete.
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

_BODIES = {
    # ``mps.reinterpret_cast`` rejects a rank-0 si32 operand.  Pareto(0-d, 0-d).rsample() is this.
    "rank0_pow_rank0": "x + (self.w.reshape(()) + 1.0) ** (self.w.reshape(()) * 2.0)",
    # ``mps.reshape``: result shape not compatible with the input shape.
    "param_beside_constant_rank2_reduction": (
        'x * self.w + lucid.ones(3, 3, device="metal").sum()'
    ),
}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True, reason=f"LCD-786 FOUND: {name} aborts the process in MPSGraph"
            ),
        )
        for name in _BODIES
    ],
)
def test_compiled_step_does_not_abort(name: str, device_gpu_only: str) -> None:
    root = str(Path(lucid.__file__).resolve().parent.parent)
    done = subprocess.run(
        [sys.executable, "-c", _PRELUDE.replace("BODY", _BODIES[name])],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-300:]
    assert "compiled True" in done.stdout, done.stdout
