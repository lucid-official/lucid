"""A CPU index addresses a metal tensor; the engine never sees mixed storages.

Index tensors are usually built on the CPU — a list made into a tensor, a
mask from a comparison done on the host — and the reference framework
accepts them against a tensor on any device.  Lucid passed them to the
engine unchanged, and every indexing path on a metal tensor failed with
``bad_variant_access`` (a backend reading a CPU storage it assumed was a
GPU one), assignment included.  Indices now follow the indexed tensor;
the opposite direction is refused by name, as the reference refuses it,
and the engine's index ops check devices instead of crashing.
"""

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")


def _grid() -> lucid.Tensor:
    return lucid.arange(12.0).reshape(3, 4).to("metal")


def test_cpu_indices_read_a_metal_tensor() -> None:
    x = _grid()
    assert x[lucid.tensor([0, 2])].tolist() == [[0, 1, 2, 3], [8, 9, 10, 11]]
    assert x[:, lucid.tensor([1, 3])].tolist() == [[1, 3], [5, 7], [9, 11]]
    assert x[lucid.tensor([True, False, True])].tolist() == [[0, 1, 2, 3], [8, 9, 10, 11]]
    assert x[lucid.tensor([0, 1]), lucid.tensor([2, 3])].tolist() == [2.0, 7.0]


def test_cpu_indices_write_a_metal_tensor() -> None:
    y = lucid.zeros(3, 4).to("metal")
    y[lucid.tensor([0, 2])] = 5.0
    assert y.tolist() == [[5.0] * 4, [0.0] * 4, [5.0] * 4]


def test_a_metal_index_on_a_cpu_tensor_is_refused_by_name() -> None:
    with pytest.raises(RuntimeError, match="index is on metal"):
        lucid.arange(12.0).reshape(3, 4)[lucid.tensor([0, 2], device="metal")]


@pytest.mark.parametrize(
    "call",
    [
        lambda x: lucid.index_select(x, 0, lucid.tensor([0, 2])),
        lambda x: lucid.gather(x, 1, lucid.tensor([[0, 1]])),
    ],
    ids=["index_select", "gather"],
)
def test_engine_index_ops_report_a_device_mismatch(call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(Exception, match="DeviceMismatch|expected"):
        call(_grid())
