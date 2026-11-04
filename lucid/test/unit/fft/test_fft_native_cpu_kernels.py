"""The CPU's native FFT kernels, checked against a float64 oracle.

float32 and complex64 reach the vDSP plans at power-of-two lengths only, since
every other length routes to MLX; float64 and complex128 reach them at any length.
The rest of this directory mostly asks about arguments, so without this file a kernel
could change on the paths that matter and nothing would notice: a cropped or padded
line, a strided axis, the real-input path, and the parallel driver, which only runs
past a size threshold (at least 4 lines and 32768 elements in all).
"""

from collections.abc import Iterator

import numpy as np
import pytest

import lucid

_POW2 = (2, 8, 64, 1024)
_ANY = (*_POW2, 3, 12, 97, 1000, 4097)
_LENGTHS = {"float32": _POW2, "complex64": _POW2, "float64": _ANY, "complex128": _ANY}
_TOLERANCE = {"float32": 2e-5, "complex64": 2e-5, "float64": 1e-11, "complex128": 1e-11}
_OUTPUT = {
    "float32": lucid.complex64,
    "complex64": lucid.complex64,
    "float64": lucid.complex128,
    "complex128": lucid.complex128,
}
# Each stays past the parallel threshold whichever axis is transformed.
_PARALLEL = (((64, 1024), 0), ((64, 1024), 1), ((1024, 64), 0), ((1024, 64), 1))
_NORMS = (None, "ortho")


def _sample(shape: tuple[int, ...], dtype: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(shape)
    if dtype.startswith("complex"):
        return (real + 1j * rng.standard_normal(shape)).astype(dtype)
    return real.astype(dtype)


def _complex_of(dtype: str) -> str:
    return "complex64" if dtype == "float32" else "complex128"


def _shapes(length: int) -> Iterator[tuple[tuple[int, ...], int]]:
    yield (length,), 0
    yield (4, length), -1
    yield (length, 4), 0
    yield (3, length, 5), 1


def _line_cases() -> Iterator[tuple[str, tuple[int, ...], int, int | None]]:
    for dtype, lengths in _LENGTHS.items():
        for length in lengths:
            for shape, axis in _shapes(length):
                for n in (None, max(length // 2, 1), length * 2):
                    yield dtype, shape, axis, n
    for dtype in _LENGTHS:
        for shape, axis in _PARALLEL:
            yield dtype, shape, axis, None


def _case_id(case: tuple[str, tuple[int, ...], int, int | None]) -> str:
    dtype, shape, axis, n = case
    return f"{dtype}-{'x'.join(map(str, shape))}-axis{axis}-n{n}"


_LINES = list(_line_cases())


def _assert_close(got: lucid.Tensor, want: np.ndarray, dtype: str) -> None:
    values = got.numpy()
    scale = max(float(np.abs(want).max(initial=0.0)), 1.0)
    np.testing.assert_allclose(values, want, rtol=0.0, atol=_TOLERANCE[dtype] * scale)


@pytest.mark.parametrize("norm", _NORMS)
@pytest.mark.parametrize("case", _LINES, ids=_case_id)
def test_forward_and_inverse_lines_match_the_oracle(
    case: tuple[str, tuple[int, ...], int, int | None], norm: str | None
) -> None:
    dtype, shape, axis, n = case
    data = _sample(shape, dtype)
    tensor = lucid.tensor(data, device="cpu")
    wide = data.astype(np.complex128)

    forward = lucid.fft.fft(tensor, n=n, dim=axis, norm=norm)
    assert forward.dtype == _OUTPUT[dtype]
    _assert_close(forward, np.fft.fft(wide, n=n, axis=axis, norm=norm), dtype)

    inverse = lucid.fft.ifft(tensor, n=n, dim=axis, norm=norm)
    assert inverse.dtype == _OUTPUT[dtype]
    _assert_close(inverse, np.fft.ifft(wide, n=n, axis=axis, norm=norm), dtype)


@pytest.mark.parametrize("norm", _NORMS)
@pytest.mark.parametrize("case", [c for c in _LINES if c[0].startswith("float")], ids=_case_id)
def test_real_input_lines_match_the_oracle(
    case: tuple[str, tuple[int, ...], int, int | None], norm: str | None
) -> None:
    dtype, shape, axis, n = case
    data = _sample(shape, dtype)
    tensor = lucid.tensor(data, device="cpu")
    wide = data.astype(np.float64)

    half = lucid.fft.rfft(tensor, n=n, dim=axis, norm=norm)
    assert half.dtype == _OUTPUT[dtype]
    _assert_close(half, np.fft.rfft(wide, n=n, axis=axis, norm=norm), dtype)


@pytest.mark.parametrize("norm", _NORMS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("n", [8, 64, 1024])
@pytest.mark.parametrize("shape_axis", [((16,), 0), ((4, 16), -1), ((16, 4), 0), ((64, 513), 1)])
def test_inverse_real_lines_match_the_oracle(
    shape_axis: tuple[tuple[int, ...], int], n: int, dtype: str, norm: str | None
) -> None:
    shape, axis = shape_axis
    spectrum = np.fft.rfft(_sample(shape, "float64"), axis=axis)
    tensor = lucid.tensor(spectrum.astype(_complex_of(dtype)), device="cpu")
    got = lucid.fft.irfft(tensor, n=n, dim=axis, norm=norm)
    want = np.fft.irfft(spectrum, n=n, axis=axis, norm=norm)
    _assert_close(got, want, dtype)


@pytest.mark.parametrize("norm", _NORMS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(8, 16), (64, 1024), (1024, 64), (4, 8, 16)], ids=str)
def test_nd_transforms_match_the_oracle(
    shape: tuple[int, ...], dtype: str, norm: str | None
) -> None:
    data = _sample(shape, dtype)
    tensor = lucid.tensor(data, device="cpu")
    wide = data.astype(np.float64)

    _assert_close(lucid.fft.fftn(tensor, norm=norm), np.fft.fftn(wide, norm=norm), dtype)
    _assert_close(lucid.fft.ifftn(tensor, norm=norm), np.fft.ifftn(wide, norm=norm), dtype)
    half = np.fft.rfftn(wide, norm=norm)
    _assert_close(lucid.fft.rfftn(tensor, norm=norm), half, dtype)

    back = lucid.tensor(half.astype(_complex_of(dtype)), device="cpu")
    _assert_close(lucid.fft.irfftn(back, s=shape, norm=norm), wide, dtype)
