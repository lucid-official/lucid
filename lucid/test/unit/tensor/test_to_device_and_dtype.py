"""``.to()`` that moves and casts in one call never asks Metal for a dtype it cannot hold.

``lucid.ones(2, device="metal").to(device="cpu", dtype=lucid.float64)`` cast
first, on Metal, and raised ``to_mlx_dtype: float64 is not supported on GPU``,
though ``t.to("cpu").to(lucid.float64)`` worked.  A move to the CPU now casts
after the move; a move to Metal still casts before it.
"""

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal not available")


def test_a_metal_tensor_moves_to_the_cpu_and_widens_in_one_call() -> None:
    t = lucid.tensor([1.5, -2.0, 3.25], device="metal")
    for moved in (
        t.to(device="cpu", dtype=lucid.float64),
        t.to("cpu", lucid.float64),
        t.to("cpu").to(lucid.float64),
    ):
        assert moved.device.type == "cpu" and moved.dtype == lucid.float64
        assert moved.tolist() == [1.5, -2.0, 3.25]


def test_a_complex64_metal_tensor_widens_to_complex128_on_the_cpu() -> None:
    t = lucid.tensor([1.0, 2.0], device="metal").to(lucid.complex64)
    moved = t.to(device="cpu", dtype=lucid.complex128)
    assert moved.device.type == "cpu" and moved.dtype == lucid.complex128


def test_a_strided_metal_source_is_made_dense_before_the_move() -> None:
    base = lucid.arange(6, dtype=lucid.float32, device="metal").reshape(2, 3)
    moved = base.mT.to(device="cpu", dtype=lucid.float64)
    assert moved.dtype == lucid.float64 and moved.device.type == "cpu"
    assert moved.tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]


def test_a_shared_metal_buffer_widens_on_the_cpu() -> None:
    shared = lucid.metal.shared_tensor((3,)).to("metal")
    shared.fill_(2.5)
    moved = shared.to(device="cpu", dtype=lucid.float64)
    assert moved.dtype == lucid.float64 and moved.tolist() == [2.5, 2.5, 2.5]


def test_the_gradient_of_the_combined_move_returns_to_the_metal_leaf() -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], device="metal", requires_grad=True)
    y = x.to(device="cpu", dtype=lucid.float64)
    assert y.requires_grad and y.dtype == lucid.float64
    (y * y).sum().backward()
    assert x.grad is not None and x.grad.device.type == "metal" and x.grad.dtype == lucid.float32
    assert x.grad.tolist() == [2.0, 4.0, 6.0]


def test_no_grad_leaves_no_graph() -> None:
    x = lucid.tensor([1.0], device="metal", requires_grad=True)
    with lucid.no_grad():
        y = x.to(device="cpu", dtype=lucid.float64)
    assert not y.requires_grad and y.dtype == lucid.float64


def test_an_integer_target_ends_the_graph() -> None:
    x = lucid.tensor([1.9, 2.1], device="metal", requires_grad=True)
    y = x.to(device="cpu", dtype=lucid.int64)
    assert not y.requires_grad and y.dtype == lucid.int64 and y.tolist() == [1, 2]


def test_the_other_direction_still_narrows_before_it_moves() -> None:
    t = lucid.tensor([1.5, 2.5], dtype=lucid.float64)
    moved = t.to(device="metal", dtype=lucid.float32)
    assert moved.device.type == "metal" and moved.dtype == lucid.float32
    assert moved.tolist() == [1.5, 2.5]


def test_a_cast_on_one_device_is_unchanged() -> None:
    t = lucid.tensor([1.5, 2.5], device="metal")
    assert t.to(dtype=lucid.float16).dtype == lucid.float16
    assert lucid.tensor([1.5]).to(dtype=lucid.float64).dtype == lucid.float64
