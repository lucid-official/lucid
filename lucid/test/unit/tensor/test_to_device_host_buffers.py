"""A move between devices makes no full-size host buffer of its own (LCD-1007, LCD-1012).

``randn(...).to("metal")`` of a new shape left about four times the tensor in the process
footprint after everything was freed, against one for ``randn(...)`` alone: a freed block
above the allocator pool's 4 MB limit stays in malloc's cache for tens of seconds, and the
move made three of them.  ``_transfer_impl`` densified the source first (``contiguous``, a
full copy of a dense tensor), ``transfer_to_device`` packed it into a zero-filled vector,
and only then handed a host block to MLX, which wrapped it: the Metal tensor kept that block
for its whole life, so the CPU and the device counters held the same bytes.

``upload_cpu_to_gpu`` now copies into a buffer MLX allocates and keeps nothing of the host
block, and the callers that staged a copy only to upload it (``to``, NumPy ingestion,
``from_bytes``) read their source where it lies.  A Metal tensor is counted on the device
alone, and the loop below leaves what the CPU-only loop leaves.

The footprint test measures a child process and has a control, the CPU-only version of the
same loop, so an allocator that stops retaining freed blocks fails it instead of passing.
The peak tests count what the engine itself allocates and do not depend on the allocator;
the snapshot tests, whose temporaries the engine does not track, read a child's peak
resident size.

The footprint and snapshot tests read process memory, which AddressSanitizer's quarantine and
redzones inflate (212 MB against a 140 MB bound for the pickle case, LCD-1056), so they carry
``requires_unsanitized_memory``; the peak tests read the engine's counters and keep running
under a sanitizer.
"""

import gc
import json
import os
import pickle
import subprocess
import sys
import textwrap
from collections.abc import Callable

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal not available")

_MB = 1024 * 1024
_CPU = _C_engine.Device.CPU
_GPU = _C_engine.Device.GPU

# Above the pool's 4 MB class limit, where a freed block goes to malloc's cache.
_BIG = (2048, 1024)  # 8 MiB of float32


def _peak(device: _C_engine.Device, body: Callable[[], object]) -> int:
    """The most bytes the engine's tracker held on ``device`` while ``body`` ran, above the start."""
    _C_engine.reset_peak_memory_stats(device)
    start = _C_engine.memory_stats(device).current_bytes
    body()
    return int(_C_engine.memory_stats(device).peak_bytes) - int(start)


def _current(device: _C_engine.Device) -> int:
    return int(_C_engine.memory_stats(device).current_bytes)


def test_a_dense_cpu_to_metal_move_allocates_no_host_buffer() -> None:
    x = lucid.randn(*_BIG, device="cpu")
    kept: list[lucid.Tensor] = []
    above = _peak(_CPU, lambda: kept.append(x.to("metal")))
    # The upload copies into the buffer Metal keeps, straight out of the source.
    assert above <= _MB, f"{above / _MB:.1f} MB of host memory made for a dense move"
    assert kept[0].tolist() == x.tolist()


def test_copy_of_a_transposed_view_allocates_no_host_buffer() -> None:
    x = lucid.zeros(_BIG[1], _BIG[0], device="cpu")
    y = lucid.randn(*_BIG, device="cpu")
    above = _peak(_CPU, lambda: x.copy_(y.t()))
    assert above <= _MB, f"{above / _MB:.1f} MB of host memory made for copy_(y.t())"
    keep = x[0:2]
    above_aliased = _peak(_CPU, lambda: x.copy_(y.t()))
    assert (
        above_aliased <= _MB
    ), f"{above_aliased / _MB:.1f} MB of host memory made for aliased copy_(y.t())"
    assert keep.shape == (2, _BIG[0])
    np.testing.assert_array_equal(x.numpy(), y.t().numpy())

    sh = lucid.metal.shared_tensor((_BIG[1], _BIG[0]), dtype=lucid.float32)
    above_shared = _peak(_CPU, lambda: sh.copy_(y.t()))
    assert (
        above_shared <= _MB
    ), f"{above_shared / _MB:.1f} MB of host memory made for shared_tensor.copy_(y.t())"
    np.testing.assert_array_equal(sh.numpy(), y.t().numpy())


def test_a_strided_cpu_to_metal_move_packs_through_one_temporary_block() -> None:
    x = lucid.randn(*_BIG, device="cpu")
    nbytes = x.numel() * 4
    kept: list[lucid.Tensor] = []
    above = _peak(_CPU, lambda: kept.append(x.t().to("metal")))
    # A view has to be packed somewhere; the copy that densified it first made a second block.
    assert (
        above <= nbytes + _MB
    ), f"{above / _MB:.1f} MB above the source for a {nbytes / _MB:.0f} MB tensor"
    assert kept[0].tolist() == x.t().tolist()


def test_a_metal_tensor_is_counted_on_the_device_and_not_on_the_host() -> None:
    gc.collect()
    cpu0, gpu0 = _current(_CPU), _current(_GPU)
    src = lucid.randn(*_BIG, device="cpu")
    nbytes = src.numel() * 4
    moved = src.to("metal")
    del src
    gc.collect()
    # The moved tensor held the host block it was wrapped around: both counters had its bytes.
    assert _current(_CPU) - cpu0 <= _MB, "the moved tensor still holds a host buffer"
    assert _current(_GPU) - gpu0 >= nbytes
    del moved
    gc.collect()
    assert abs(_current(_GPU) - gpu0) <= _MB


def test_ingesting_onto_metal_stages_no_host_copy() -> None:
    arr = np.random.default_rng(0).standard_normal(_BIG).astype(np.float32)
    kept: list[lucid.Tensor] = []
    assert _peak(_CPU, lambda: kept.append(lucid.tensor(arr, device="metal"))) <= _MB
    raw = arr.tobytes()
    impl = lucid.tensor(arr, device="cpu")._impl
    stats: list[object] = []

    def from_bytes() -> None:
        stats.append(
            _C_engine.TensorImpl.from_bytes(
                raw, list(arr.shape), impl.dtype, _C_engine.Device.GPU, False
            )
        )

    assert _peak(_CPU, from_bytes) <= _MB
    unpickled: list[lucid.Tensor] = []
    blob = pickle.dumps(kept[0])
    assert _peak(_CPU, lambda: unpickled.append(pickle.loads(blob))) <= _MB
    np.testing.assert_array_equal(unpickled[0].numpy(), arr)
    np.testing.assert_array_equal(kept[0].numpy(), arr)


def test_a_metal_to_cpu_move_makes_no_device_side_copy() -> None:
    g = lucid.randn(*_BIG, device="metal")
    lucid.eval(g)
    nbytes = g.numel() * 4
    kept: list[lucid.Tensor] = []
    above = _peak(_GPU, lambda: kept.append(g.to("cpu")))
    assert (
        above <= _MB
    ), f"{above / _MB:.1f} MB of Metal memory made for a {nbytes / _MB:.0f} MB move"
    assert kept[0].tolist() == g.tolist()


_LAYOUTS = ("dense", "transposed", "row slice at an offset", "column slice", "every other row")


def _view(name: str, t: lucid.Tensor) -> lucid.Tensor:
    """``t`` (2-D) as one of the layouts a move has to read through its strides."""
    return {
        "dense": t,
        "transposed": t.t(),
        "row slice at an offset": t[1:3],
        "column slice": t[:, 1:5],
        "every other row": t[::2],
    }[name]


@pytest.mark.parametrize("name", _LAYOUTS)
def test_a_cpu_view_moves_to_metal_with_its_own_values(name: str) -> None:
    src = _view(name, lucid.arange(24, dtype=lucid.float32, device="cpu").reshape(4, 6))
    moved = src.to("metal")
    assert moved.device.type == "metal"
    assert moved.tolist() == src.tolist()
    assert moved.to("cpu").tolist() == src.tolist()


@pytest.mark.parametrize("name", _LAYOUTS)
def test_a_metal_view_moves_to_the_cpu_with_its_own_values(name: str) -> None:
    src = _view(name, lucid.arange(24, dtype=lucid.float32, device="metal").reshape(4, 6))
    moved = src.to("cpu")
    assert moved.device.type == "cpu"
    assert moved.tolist() == src.tolist()


def test_a_move_that_casts_keeps_the_values() -> None:
    src = lucid.arange(12, dtype=lucid.float32, device="cpu").reshape(3, 4).t()
    half = src.to("metal", lucid.float16)
    assert half.dtype == lucid.float16 and half.tolist() == src.tolist()
    back = half.to("cpu", lucid.float64)
    assert back.dtype == lucid.float64 and back.tolist() == src.tolist()


def test_the_gradient_of_a_strided_move_reaches_the_cpu_leaf() -> None:
    x = lucid.arange(6, dtype=lucid.float32, device="cpu").reshape(2, 3).requires_grad_()
    y = x.t().to("metal")
    (y * y).sum().backward()
    assert x.grad is not None and x.grad.device.type == "cpu"
    assert x.grad.tolist() == (x * 2).tolist()

    leaf = lucid.tensor([1.0, 2.0, 3.0], device="cpu", requires_grad=True)
    (leaf.to("metal") * 3).sum().backward()
    assert leaf.grad is not None and leaf.grad.tolist() == [3.0, 3.0, 3.0]


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_a_same_device_copy_is_still_a_separate_buffer(device: str) -> None:
    """``contiguous`` is the copy of ``to(copy=True)``: dropping it for moves must not drop it here."""
    x = lucid.zeros(8, device=device)
    whole, part = x.to(copy=True), x[2:6].to(copy=True)
    assert whole is not x
    x.add_(1.0)
    assert whole.tolist() == [0.0] * 8
    assert part.tolist() == [0.0] * 4


# ── the footprint of a loop of distinct shapes, with the CPU-only loop as its control ──

_CHILD = textwrap.dedent("""
    import gc
    import json
    import os
    import sys

    import lucid
    from lucid.test.audit._zoo_budget import footprint_gb

    MODE, N, LEAD = sys.argv[1], 8, 1024


    def footprint_mb():
        return footprint_gb(os.getpid()) * 1024


    # One-time Metal and kernel set-up stays out of what is measured.
    lucid.randn(64, 8, 16, 16, device="cpu").to("metal").sum().item()
    gc.collect()
    lucid.metal.empty_cache()
    base = footprint_mb()

    total = 0
    for i in range(N):
        shape = (LEAD + 32 * i, 8, 16, 16)
        x = lucid.randn(*shape, device="cpu")
        total += x.numel() * 4
        if MODE == "move":
            y = x.to("metal")
            del y
        del x
        gc.collect()
        lucid.metal.empty_cache()
    # Read at once: a freed block stays in the allocator's cache for tens of seconds.
    print(json.dumps({"grew_mb": footprint_mb() - base, "inputs_mb": total / 2**20}))
    """)


def _grew(mode: str) -> dict[str, float]:
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, mode],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return {k: float(v) for k, v in json.loads(done.stdout.strip().splitlines()[-1]).items()}


@pytest.mark.slow
@pytest.mark.requires_unsanitized_memory
def test_a_loop_of_cpu_to_metal_moves_leaves_what_the_cpu_loop_leaves() -> None:
    """Eight new shapes: the move was at four times the control, then at two, now at one."""
    control = _grew("control")
    moved = _grew("move")
    assert control["grew_mb"] > control["inputs_mb"] / 2, (
        f"the control grew {control['grew_mb']:.0f} MB for {control['inputs_mb']:.0f} MB of "
        "inputs: freed large blocks are no longer retained, so this test proves nothing"
    )
    # The moved tensor no longer holds a host block of its own, so the loop leaves what the
    # CPU-only loop leaves: the three-copy path was at four times, the landed one-pass pack
    # at twice (the Metal tensor kept the block it was packed into).
    assert moved["grew_mb"] < 1.5 * control["grew_mb"], (
        f"move {moved['grew_mb']:.0f} MB vs control {control['grew_mb']:.0f} MB "
        f"for {moved['inputs_mb']:.0f} MB of inputs"
    )


# ── the snapshots a read or a write through a view used to take, in a child's peak size ──

_PEAK_CHILD = textwrap.dedent("""
    import gc
    import json
    import resource
    import sys

    import numpy  # imported here so the body's peak does not include loading it

    import lucid

    N = 4096  # 4096 x 4096: 64 MiB of float32


    def peak():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


    SETUP
    gc.collect()
    before = peak()
    BODY
    print(json.dumps({"rise_mb": (peak() - before) / 2**20, "tensor_mb": N * N * 4 / 2**20}))
    """)

_SETUP_F32 = "x = lucid.ones(N, N, device='cpu'); y = lucid.ones(N, N, device='cpu')"
_SETUP_F32_LIVE_VIEW = _SETUP_F32 + "; keep = x[0:2]"
# Built from a column, so no full-size block is freed during setup for the body to reuse: a
# freed large block stays in malloc's cache and would hide the temporary this measures.
_SETUP_BF16 = "x = lucid.ones(N, 1, device='cpu').to(lucid.bfloat16).expand(N, N).contiguous()"
# Case id -> (setup, body, megabytes of result the case has to hold anyway): a bytes object or
# a float32 array of one tensor's size (``pickle`` also writes them into its own output), and
# nothing for a write or a repr.
_SNAPSHOT_CASES: dict[str, tuple[str, str, float]] = {
    "to_bytes": (_SETUP_F32, "b = x._impl.to_bytes()", 64.0),
    "to_bytes of a transposed view": (_SETUP_F32, "b = x.t()._impl.to_bytes()", 64.0),
    "pickle": (_SETUP_F32, "import pickle; b = pickle.dumps(x)", 128.0),
    "bfloat16 numpy": (_SETUP_BF16, "a = x.numpy()", 64.0),
    "bfloat16 numpy of a transposed view": (_SETUP_BF16, "a = x.t().numpy()", 64.0),
    "repr": (_SETUP_F32, "s = repr(x)", 0.0),
    "assignment of a view": (_SETUP_F32, "x[:] = y.t()", 0.0),
    "assignment of a view to a tensor with a live view": (
        _SETUP_F32_LIVE_VIEW,
        "x[:] = y.t()",
        0.0,
    ),
    "copy_ of a transposed view": (_SETUP_F32, "x.copy_(y.t())", 0.0),
    "copy_ of a transposed view to a tensor with a live view": (
        _SETUP_F32_LIVE_VIEW,
        "x.copy_(y.t())",
        0.0,
    ),
    # ``broadcast_to`` allocates one full-size tensor; dropping Python ``contiguous``
    # avoids a second 64 MiB copy on top of it (128 MB -> 64 MB).
    "copy_ of a broadcast view": (
        "x = lucid.ones(N, N, device='cpu'); y = lucid.ones(1, N, device='cpu')",
        "x.copy_(y)",
        64.0,
    ),
}


def _rise(case: str) -> dict[str, float]:
    setup, body, _ = _SNAPSHOT_CASES[case]
    code = _PEAK_CHILD.replace("SETUP", setup).replace("BODY", body)
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return {k: float(v) for k, v in json.loads(done.stdout.strip().splitlines()[-1]).items()}


@pytest.mark.slow
@pytest.mark.requires_unsanitized_memory
@pytest.mark.parametrize("case", list(_SNAPSHOT_CASES))
def test_a_read_or_write_through_a_view_takes_no_full_size_snapshot(case: str) -> None:
    """The peak above the setup is the result it hands back and a little, not a second copy of it.

    Every one of these went through a zero-filled ``std::vector`` the size of the tensor
    first (the engine's tracker never sees one), then a copy out of it.
    """
    holds = _SNAPSHOT_CASES[case][2]
    got = _rise(case)
    assert got["rise_mb"] <= holds + 12.0, (
        f"{case}: peak rose {got['rise_mb']:.0f} MB for a {got['tensor_mb']:.0f} MB tensor, "
        f"of which {holds:.0f} MB is the result itself"
    )
