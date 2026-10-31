"""The Gamma rejection sampler must not make the Metal graph compiler blow up.

LCD-784: the sampler carried an "accepted so far" mask from one retry round to
the next, so every round read the previous mask twice.  MPSGraph's stitching
pass walks the uses of a node without memoising them, so compiling a step that
holds one ``Gamma.rsample`` took ``2 ** rounds`` units of time: 56 s for the
default 8 rounds, minutes once a few distributions share a step.  The compile
is native and cannot be interrupted, so the guard runs it in a child process
with a deadline and fails fast instead of hanging the suite.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import lucid

_CHILD = """
import time

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import make_step
from lucid.distributions import Gamma


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 3)

    def forward(self, x):
        raw = F.softplus(self.encoder(x)) + 0.5
        return self.decoder(Gamma(raw, lucid.ones_like(raw)).rsample())


def main():
    x = lucid.randn(4, 4).to("metal")
    t = lucid.randn(4, 3).to("metal")
    step = make_step(Block().to("metal"), F.mse_loss)
    start = time.perf_counter()
    step(x, t).backward()
    print(f"compiled {not step.eager_only} in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
"""

_DEADLINE_S = 30.0  # fixed: ~1 s; the exponential-in-rounds blow-up: 56 s


def test_gamma_rsample_step_compiles_in_bounded_time(device_gpu_only: str) -> None:
    root = str(Path(lucid.__file__).resolve().parent.parent)
    try:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_DEADLINE_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"compiling a make_step over Gamma.rsample took more than {_DEADLINE_S:.0f} s: "
            "the sampler's retry rounds feed one another (LCD-784)"
        )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "compiled True" in done.stdout, done.stdout
