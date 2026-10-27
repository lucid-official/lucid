"""The engine's ``norm`` checks its axes on both devices.

The CPU kernel took an axis the input lacks without a word — ``dims=[1]`` on
a 1-D input returned ``|x|`` unreduced, ``dims=[0, 1]`` ignored axis 1 —
while Metal raised MLX's own "Invalid axis".  Both now refuse the same way.
"""

import pytest

import lucid
from lucid._C import engine as _C_engine


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("axis", [[1], [0, 1], [-2]])
def test_an_axis_the_input_lacks_is_refused(device, axis):
    x = lucid.tensor([3.0, 4.0], device=device)
    with pytest.raises(IndexError, match="out of range"):
        _C_engine.linalg.norm(x._impl, 2.0, axis, False)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_a_repeated_axis_is_refused(device):
    x = lucid.tensor([[3.0, 4.0]], device=device)
    with pytest.raises(RuntimeError, match="appears multiple times"):
        _C_engine.linalg.norm(x._impl, 2.0, [1, -1], False)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_valid_axes_still_reduce(device):
    x = lucid.tensor([[3.0, 4.0]], device=device)
    got = lucid.Tensor.__new_from_impl__(_C_engine.linalg.norm(x._impl, 2.0, [-1], False))
    assert float(got.item()) == pytest.approx(5.0)
