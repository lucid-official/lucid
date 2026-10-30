"""``lucid.metal.synchronize`` raises when pending work cannot run (LCD-462).

6dc7aea0f made ``evaluate_pending`` retry each array on its own after a batch
evaluation failed, so that one bad graph would not keep the rest from running
— and dropped every error of the retry.  A graph MLX could not evaluate
(an allocation past the device's buffer limit, the out-of-memory case) left
``synchronize`` returning as if the work were done; the error surfaced later,
at whatever read touched the tensor, or never.

The retry stays: the healthy work still runs.  The first failure is then
raised from ``synchronize`` with how many arrays failed.

The failure provoked here is a 4 TiB allocation that only evaluation makes.
A *result* that large is refused when it is built (LCD-669: no Metal buffer
can be larger than ``maxBufferLength``, so ``zeros(2**20, 2**20)`` raises
``OutOfMemory`` at once), which leaves intermediates: ``embedding_bag``
gathers every looked-up row before it sums them into bags, so 2^20 lookups
of a 2^20-wide row are a 4 TiB gather behind a 4 MiB result.  MLX's Metal
allocator refuses that buffer against ``maxBufferLength`` before it
allocates anything.
"""

import pytest

import lucid
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

_WIDE = 1 << 20  # float32 rows of 4 MiB; 2^20 of them gathered are 4 TiB


def _unrunnable(mode: str = "sum") -> lucid.Tensor:
    """One bag of ``_WIDE`` lookups of a ``_WIDE``-wide row: a (1, _WIDE)
    result whose gather of (_WIDE, _WIDE) no device can hold."""
    weight = lucid.ones(1, _WIDE, device="metal")
    indices = lucid.zeros(_WIDE, dtype=lucid.int64, device="metal")
    offsets = lucid.zeros(1, dtype=lucid.int64, device="metal")
    return F.embedding_bag(indices, weight, offsets, mode=mode)


def test_a_result_past_the_buffer_limit_is_refused_when_built() -> None:
    # Why the graph above goes through an intermediate: a result that size
    # never becomes pending work at all.
    with pytest.raises(_C_engine.OutOfMemory, match="maxBufferLength") as info:
        lucid.zeros(_WIDE, _WIDE, device="metal")
    assert isinstance(info.value, MemoryError)
    assert _C_engine.synchronize_gpu() >= 0


def test_an_unrunnable_graph_raises_from_synchronize() -> None:
    _C_engine.synchronize_gpu()  # nothing earlier left queued counts here
    healthy = lucid.ones(4, device="metal") * 3.0
    doomed = _unrunnable()
    assert tuple(doomed.shape) == (1, _WIDE)
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
    first = _unrunnable("sum")
    healthy = lucid.arange(6, device="metal").reshape(2, 3) + 1
    second = _unrunnable("max")
    with pytest.raises(_C_engine.LucidError, match=r"^metal synchronize: 2 of \d+ pending"):
        _C_engine.synchronize_gpu()
    assert healthy.tolist() == [[1, 2, 3], [4, 5, 6]]
    del first, second


def test_a_read_of_the_failed_tensor_still_raises() -> None:
    _C_engine.synchronize_gpu()
    doomed = _unrunnable()
    with pytest.raises(_C_engine.LucidError):
        lucid.metal.synchronize()
    with pytest.raises(RuntimeError, match="maximum allowed buffer size"):
        doomed[0, :2].tolist()
