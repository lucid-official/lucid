"""``lucid.metal.synchronize`` raises when pending work cannot run (LCD-462).

6dc7aea0f made ``evaluate_pending`` retry each array on its own after a batch
evaluation failed, so that one bad graph would not keep the rest from running
— and dropped every error of the retry.  A graph MLX could not evaluate
(an allocation past the device's buffer limit, the out-of-memory case) left
``synchronize`` returning as if the work were done; the error surfaced later,
at whatever read touched the tensor, or never.

The retry stays: the healthy work still runs.  The first failure is then
raised from ``synchronize`` with how many arrays failed.

The failure provoked here is an allocation of 4 TiB, which MLX's Metal
allocator refuses against ``maxBufferLength`` before it allocates anything.
"""

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

_TOO_LARGE = (1 << 20, 1 << 20)  # float32: 4 TiB, past any device's buffer limit


def test_an_unrunnable_graph_raises_from_synchronize() -> None:
    _C_engine.synchronize_gpu()  # nothing earlier left queued counts here
    healthy = lucid.ones(4, device="metal") * 3.0
    doomed = lucid.zeros(*_TOO_LARGE, device="metal")
    with pytest.raises(_C_engine.LucidError, match="maximum allowed buffer size") as info:
        lucid.metal.synchronize()
    assert isinstance(info.value, RuntimeError)
    assert str(info.value).startswith("metal synchronize: 1 of ")
    # The healthy work ran and reads back; the failure was not kept queued.
    assert healthy.tolist() == [3.0] * 4
    assert _C_engine.synchronize_gpu() == 0
    del doomed


def test_every_failure_is_counted_and_the_rest_still_run() -> None:
    _C_engine.synchronize_gpu()
    first = lucid.zeros(*_TOO_LARGE, device="metal")
    healthy = lucid.arange(6, device="metal").reshape(2, 3) + 1
    second = lucid.full(_TOO_LARGE, 2.0, device="metal")
    with pytest.raises(_C_engine.LucidError, match=r"^metal synchronize: 2 of \d+ pending"):
        _C_engine.synchronize_gpu()
    assert healthy.tolist() == [[1, 2, 3], [4, 5, 6]]
    del first, second


def test_a_read_of_the_failed_tensor_still_raises() -> None:
    _C_engine.synchronize_gpu()
    doomed = lucid.zeros(*_TOO_LARGE, device="metal")
    with pytest.raises(_C_engine.LucidError):
        lucid.metal.synchronize()
    with pytest.raises(RuntimeError, match="maximum allowed buffer size"):
        doomed[0, :2].tolist()
