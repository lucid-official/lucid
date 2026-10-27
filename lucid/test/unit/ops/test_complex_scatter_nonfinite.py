"""A complex scatter-add on Metal keeps an infinite part as it is (LCD-462).

MLX's scatter-add kernels have no complex64 form, so the GPU stream sums the
real and the imaginary parts separately.  6dc7aea0f put the parts back
together as ``re + im * 1j``, which multiplies an infinite imaginary part by
the zero real lane of ``1j``: scattering (1 + inf j) answered (nan + inf j) on
Metal while the CPU answered (1 + inf j).  The parts are now interleaved and
reinterpreted, as ``nan_to_num`` already did, and every caller of the
complex split — the multi-axis ``scatter_add``, its single-axis form, and the
axis scatter behind ``index_add`` and ``scatter_add`` — shares that helper.
"""

import math
from collections.abc import Callable

import pytest

import lucid

_INF = math.inf
_UPDATES = [
    complex(1.0, _INF),
    complex(-_INF, 2.0),
    complex(_INF, -_INF),
    complex(3.0, -4.0),
]

Scatter = Callable[[lucid.Tensor, lucid.Tensor, lucid.Tensor], lucid.Tensor]

_OPS: dict[str, Scatter] = {
    "scatter_add": lambda base, idx, src: base.scatter_add(0, idx, src),
    "index_add": lambda base, idx, src: lucid.index_add(base, 0, idx, src),
    "scatter_reduce_sum": lambda base, idx, src: lucid.scatter_reduce(base, 0, idx, src, "sum"),
    "index_put_accumulate": lambda base, idx, src: base.index_put((idx,), src, accumulate=True),
}


def _same(got: list[complex], want: list[complex]) -> bool:
    def key(z: complex) -> tuple[str, str]:
        return (repr(z.real), repr(z.imag))

    return [key(z) for z in got] == [key(z) for z in want]


@pytest.mark.parametrize("op", sorted(_OPS))
def test_metal_matches_cpu_on_infinite_parts(device: str, op: str) -> None:
    def run(dev: str) -> list[complex]:
        base = lucid.zeros(6, dtype=lucid.complex64, device=dev)
        idx = lucid.tensor([0, 2, 5, 2], dtype=lucid.int64, device=dev)
        src = lucid.tensor(_UPDATES, dtype=lucid.complex64, device=dev)
        return _OPS[op](base, idx, src).tolist()

    want = run("cpu")
    assert want[0] == complex(1.0, _INF)
    assert _same(run(device), want)


def test_two_dimensional_scatter_add_along_the_inner_axis(device: str) -> None:
    def run(dev: str) -> list[list[complex]]:
        base = lucid.zeros(2, 3, dtype=lucid.complex64, device=dev)
        idx = lucid.tensor([[0, 2, 1], [1, 1, 0]], dtype=lucid.int64, device=dev)
        src = lucid.tensor(
            [[complex(1, _INF), complex(2, 3), complex(-_INF, 0)], [_UPDATES[3]] * 3],
            dtype=lucid.complex64,
            device=dev,
        )
        return base.scatter_add(1, idx, src).tolist()

    want = run("cpu")
    got = run(device)
    assert all(_same(g, w) for g, w in zip(got, want, strict=True))


def test_finite_complex_scatter_add_is_unchanged(device: str) -> None:
    base = lucid.tensor([1 + 1j, 2 - 2j, 0j], dtype=lucid.complex64, device=device)
    idx = lucid.tensor([2, 0, 2], dtype=lucid.int64, device=device)
    src = lucid.tensor([1j, 3 + 0j, -1 + 2j], dtype=lucid.complex64, device=device)
    assert base.scatter_add(0, idx, src).tolist() == [4 + 1j, 2 - 2j, -1 + 3j]
