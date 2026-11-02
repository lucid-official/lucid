"""A full reduction keeps its graph shape equal to its trace shape (LCD-800, LCD-804).

``emit_reduce`` leaves a reduction over every axis of a rank-r tensor in the
graph as ``keepdim`` -- r ones -- while the trace records ``()``.  The tensor
bound to that trace id therefore disagrees with the trace, and the first op
that broadcasts it against something of lower rank promotes its result:
``x(4) + a(2, 3).sum()`` is ``(1, 4)`` in the graph and ``(4,)`` in the trace.
Everything downstream is then built from the wrong-rank tensor.

* ``sum(dim=0)`` reduces the size-1 axis, so a compiled step returns the first
  element instead of the total, with no error;
* ``softmax`` and ``cumsum`` run over the wrong axis;
* ``cat`` / ``stack`` against a true rank-1 tensor abort the process
  (``all inputs must have equal rank``);
* the add's VJP reduces the gradient over axes computed from the trace shape
  (``axes_for_unreduce``: ``-1`` from an unsigned underflow, plus an
  out-of-bounds read) and reshapes the result into ``(1, 1)``.  The MPSGraph on
  macOS 27 tolerates that; the six sampler cases of
  ``test_traced_sampler_compile_budget`` that aborted on the CI image reshaped
  ``(1, 4)`` into ``(1, 1)`` in exactly this place.

The emitter now squeezes a full reduction to ``()`` and the builder refuses any
value bound at a shape other than the trace's, so the compiled result matches eager.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="Metal device not available on this host"),
    pytest.mark.deadline(300),
]

#: Reductions over every axis of a rank-2 tensor, by the call that makes them.
_REDUCTIONS = {
    "sum": lambda a: a.sum(),
    "sum_dims": lambda a: a.sum(dim=(0, 1)),
    "mean": lambda a: a.mean(),
    "max": lambda a: a.max(),
    "min": lambda a: a.min(),
    "amax": lambda a: lucid.amax(a, dim=(0, 1)),
    "prod": lambda a: a.prod(),
    "var": lambda a: a.var(),
    "std": lambda a: a.std(),
    "norm": lambda a: lucid.linalg.norm(a),
    "logsumexp": lambda a: lucid.logsumexp(a, dim=(0, 1)),
}
_CONSUMERS = {
    "sum0": lambda v: v.sum(dim=0),
    "softmax0": lambda v: F.softmax(v, dim=0),
    "cumsum0": lambda v: lucid.cumsum(v, dim=0),
}


def _inputs() -> tuple[lucid.Tensor, lucid.Tensor]:
    a = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="metal")
    x = lucid.tensor([0.0, 1.0, 2.0, 3.0], device="metal")
    return x, a


@pytest.mark.parametrize("consumer", sorted(_CONSUMERS))
@pytest.mark.parametrize("reduction", sorted(_REDUCTIONS))
def test_compiled_matches_eager_after_a_full_reduction(reduction: str, consumer: str) -> None:
    reduce, consume = _REDUCTIONS[reduction], _CONSUMERS[consumer]

    def fn(x: lucid.Tensor, a: lucid.Tensor) -> lucid.Tensor:
        return consume(x + reduce(a))

    x, a = _inputs()
    compiled = lucid.compile(fn)
    compiled(x, a)  # the first call traces; the second runs the executable
    got, want = compiled(x, a), fn(x, a)
    assert got.shape == want.shape
    assert lucid.allclose(
        got, want, rtol=1e-4, atol=1e-4
    ), f"{reduction} then {consumer}: compiled {got.tolist()} != eager {want.tolist()}"


_CHILD = """
import lucid
a = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="metal")
x = lucid.tensor([0.0, 1.0, 2.0, 3.0], device="metal")
fn = lambda x, a: lucid.{join}([x + a.sum(), x], dim=0)
c = lucid.compile(fn)
c(x, a)
print("RESULT", c(x, a).tolist() == fn(x, a).tolist(), flush=True)
"""


@pytest.mark.parametrize("join", ["cat", "stack"])
def test_joining_a_reduced_scalar_with_a_vector_does_not_abort(join: str) -> None:
    done = subprocess.run(
        [sys.executable, "-c", _CHILD.format(join=join)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=Path(lucid.__file__).resolve().parent.parent,
    )
    assert done.returncode == 0, (done.stderr or done.stdout)[-300:]
    assert "RESULT True" in done.stdout
