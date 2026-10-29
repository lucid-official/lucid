"""``tensordot`` validates ``dims`` before it builds an axis list (LCD-670).

An integer ``dims`` stands for ``range(dims)`` on the second operand, and the
adapter built that list before checking ``dims`` against either operand's
rank: ``tensordot(randn(0), -3, 2**40)`` tried to allocate a 2**40-entry list
and the process was killed.  ``dims`` outside ``[0, min(ndim_a, ndim_b)]`` is
refused first now, as the reference refuses it, without allocating — the
huge cases run in a child process under a timeout, so a regression fails the
test instead of taking pytest down with it.
"""

import subprocess
import sys
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine

_SHAPES = [((3, 4), (4, 5)), ((3, 4, 5), (5,)), ((), ()), ((0,), (0,)), ((2, 3), (2, 3))]
_DIMS = [-1, 0, 1, 2, 3, True, False]


def _outcome(call: object) -> tuple[str, object]:
    try:
        out = call()
    except RuntimeError:
        return "refused", None
    return "shape", tuple(out.shape)


@pytest.mark.parity
@pytest.mark.parametrize("dims", _DIMS, ids=str)
@pytest.mark.parametrize(("sa", "sb"), _SHAPES, ids=str)
def test_integer_dims_are_accepted_or_refused_as_the_reference_does(
    ref: ModuleType, device: str, sa: tuple[int, ...], sb: tuple[int, ...], dims: int
) -> None:
    a, b = lucid.randn(*sa, device=device), lucid.randn(*sb, device=device)
    want = _outcome(lambda: ref.tensordot(ref.randn(sa), ref.randn(sb), dims))
    got = _outcome(lambda: lucid.tensordot(a, b, dims))
    assert got == want


@pytest.mark.parametrize("dims", [-1, 3])
def test_a_refused_dims_is_both_the_reference_and_the_builtin_class(dims: int) -> None:
    a, b = lucid.randn(3, 4), lucid.randn(4, 5)
    with pytest.raises(_C_engine.InvalidArgument):
        lucid.tensordot(a, b, dims)
    with pytest.raises(RuntimeError):
        lucid.tensordot(a, b, dims)
    with pytest.raises(ValueError):
        lucid.tensordot(a, b, dims)


@pytest.mark.parity
@pytest.mark.parametrize("axes", [([2], [0]), ([-3], [0]), ([1], [5]), ([2**40], [0])], ids=str)
def test_an_axis_list_out_of_range_is_an_index_error(
    ref: ModuleType, device: str, axes: tuple[list[int], list[int]]
) -> None:
    a, b = lucid.randn(3, 4, device=device), lucid.randn(4, 5, device=device)
    with pytest.raises(IndexError):
        ref.tensordot(ref.randn(3, 4), ref.randn(4, 5), axes)
    with pytest.raises(IndexError):
        lucid.tensordot(a, b, axes)


@pytest.mark.parametrize("operand", [-3, 2.5])
def test_a_number_is_not_a_tensor_operand(operand: float) -> None:
    with pytest.raises(TypeError):
        lucid.tensordot(lucid.randn(3), operand, 1)
    with pytest.raises(TypeError):
        lucid.tensordot(operand, lucid.randn(3), 1)


_CHILD = """
import lucid

for dims in (2**40, -(2**40), 2**62):
    try:
        lucid.tensordot(lucid.randn(0), lucid.randn(0), dims)
    except RuntimeError:
        continue
    raise SystemExit(f"dims={dims} was accepted")
try:
    lucid.tensordot(lucid.randn(0), -3, 2**40)
except TypeError:
    pass
else:
    raise SystemExit("a number operand was accepted")
"""


def test_a_huge_dims_is_refused_without_allocating() -> None:
    # A regression would build range(2**40) and be killed; the timeout and the
    # return code turn that into a failure of this test alone.
    run = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=120, check=False
    )
    assert run.returncode == 0, run.stderr[-500:]
