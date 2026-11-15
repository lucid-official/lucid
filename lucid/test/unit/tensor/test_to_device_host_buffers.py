"""A move between devices makes no full-size buffer beyond the one it hands back (LCD-1007).

``randn(...).to("metal")`` of a new shape left about four times the tensor in the process
footprint after everything was freed, against one for ``randn(...)`` alone: a freed block
above the allocator pool's 4 MB limit stays in malloc's cache for tens of seconds, and the
move made three of them.  ``_transfer_impl`` densified the source first (``contiguous``, a
full copy of a dense tensor), ``transfer_to_device`` packed it into a zero-filled vector,
and only then copied it into the buffer the Metal tensor keeps.  The move reads the source
through its own strides and writes the new buffer in one pass, so neither is needed.

The footprint test measures a child process and has a control, the CPU-only version of the
same loop, so an allocator that stops retaining freed blocks fails it instead of passing.
The peak tests count what the engine itself allocates and do not depend on the allocator.
"""

import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable

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


def test_a_cpu_to_metal_move_holds_one_host_buffer_beside_the_source() -> None:
    x = lucid.randn(*_BIG, device="cpu")
    nbytes = x.numel() * 4
    kept: list[lucid.Tensor] = []
    above = _peak(_CPU, lambda: kept.append(x.to("metal")))
    # The tensor the move returns is backed by the host buffer it was packed into, so one
    # block above the source is the floor; a second one is the densifying copy.
    assert above <= nbytes + _MB, (
        f"{above / _MB:.1f} MB above the source for a {nbytes / _MB:.0f} MB tensor"
    )
    assert kept[0].tolist() == x.tolist()


def test_a_metal_to_cpu_move_makes_no_device_side_copy() -> None:
    g = lucid.randn(*_BIG, device="metal")
    lucid.eval(g)
    nbytes = g.numel() * 4
    kept: list[lucid.Tensor] = []
    above = _peak(_GPU, lambda: kept.append(g.to("cpu")))
    assert above <= _MB, (
        f"{above / _MB:.1f} MB of Metal memory made for a {nbytes / _MB:.0f} MB move"
    )
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
def test_a_loop_of_cpu_to_metal_moves_leaves_what_the_cpu_loop_leaves_not_four_times_it() -> None:
    """Eight new shapes: the move was at four times the control, one block per copy it made."""
    control = _grew("control")
    moved = _grew("move")
    assert control["grew_mb"] > control["inputs_mb"] / 2, (
        f"the control grew {control['grew_mb']:.0f} MB for {control['inputs_mb']:.0f} MB of "
        "inputs: freed large blocks are no longer retained, so this test proves nothing"
    )
    # The returned tensor is backed by one host block, so twice the control is the floor
    # until that block belongs to Metal; the three-copy path was at four times.
    assert moved["grew_mb"] < 3.0 * control["grew_mb"], (
        f"move {moved['grew_mb']:.0f} MB vs control {control['grew_mb']:.0f} MB "
        f"for {moved['inputs_mb']:.0f} MB of inputs"
    )
