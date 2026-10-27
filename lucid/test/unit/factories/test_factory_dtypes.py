"""Every factory makes every dtype the reference makes, on both devices.

``full`` refused bfloat16; ``arange`` and ``linspace`` filled only four
widths and refused the half floats and the narrow integers; ``eye``,
``rand`` and ``randn`` refused complex64 on the CPU — and on Metal ``eye``
and ``rand`` passed MLX's own ValueError through, as ``rand`` and
``randn`` did for integers.  What the reference refuses stays refused, now
in Lucid's words: bool and complex ``arange``, bool ``linspace``, and
integer draws.
"""

import numpy as np
import pytest

import lucid

DEVICES = ["cpu", "metal"]
_NARROW = [lucid.float16, lucid.bfloat16, lucid.int8, lucid.int16]


def _values(t: lucid.Tensor) -> np.ndarray:
    if t.dtype in (lucid.float16, lucid.bfloat16):
        t = t.float()
    return t.numpy()


@pytest.mark.parametrize("device", DEVICES)
def test_full_makes_bfloat16(device: str) -> None:
    t = lucid.full((2, 3), 1.3, dtype=lucid.bfloat16, device=device)
    assert t.dtype == lucid.bfloat16
    # 1.3 rounded to bfloat16's 8 significant bits.
    np.testing.assert_array_equal(_values(t), np.full((2, 3), 1.296875))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", _NARROW, ids=str)
def test_arange_makes_the_narrow_dtypes(device: str, dtype: object) -> None:
    t = lucid.arange(-2, 3, dtype=dtype, device=device)
    assert t.dtype == dtype
    np.testing.assert_array_equal(_values(t), [-2, -1, 0, 1, 2])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", _NARROW, ids=str)
def test_linspace_makes_the_narrow_dtypes(device: str, dtype: object) -> None:
    t = lucid.linspace(0, 8, 5, dtype=dtype, device=device)
    assert t.dtype == dtype
    np.testing.assert_array_equal(_values(t), [0, 2, 4, 6, 8])


@pytest.mark.parametrize("device", DEVICES)
def test_linspace_makes_complex(device: str) -> None:
    t = lucid.linspace(0, 1, 3, dtype=lucid.complex64, device=device)
    np.testing.assert_array_equal(t.numpy(), [0, 0.5, 1])


@pytest.mark.parametrize("device", DEVICES)
def test_eye_makes_complex(device: str) -> None:
    t = lucid.eye(3, dtype=lucid.complex64, device=device)
    np.testing.assert_array_equal(t.numpy(), np.eye(3, dtype=np.complex64))


@pytest.mark.parametrize("device", DEVICES)
def test_rand_draws_each_complex_part_uniformly(device: str) -> None:
    z = lucid.rand(100_000, dtype=lucid.complex64, device=device).numpy()
    for part in (z.real, z.imag):
        assert part.min() >= 0.0 and part.max() < 1.0
        assert abs(part.mean() - 0.5) < 0.01
    # The two parts are separate draws, not one value written twice.
    assert not np.array_equal(z.real, z.imag)


@pytest.mark.parametrize("device", DEVICES)
def test_randn_gives_each_complex_part_half_the_variance(device: str) -> None:
    z = lucid.randn(100_000, dtype=lucid.complex64, device=device).numpy()
    assert abs(z.real.var() - 0.5) < 0.02
    assert abs(z.imag.var() - 0.5) < 0.02
    assert abs(np.mean(np.abs(z) ** 2) - 1.0) < 0.03


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "make",
    [
        lambda d: lucid.arange(0, 3, dtype=lucid.bool_, device=d),
        lambda d: lucid.arange(0, 3, dtype=lucid.complex64, device=d),
        lambda d: lucid.linspace(0, 1, 3, dtype=lucid.bool_, device=d),
        lambda d: lucid.rand(2, dtype=lucid.int32, device=d),
        lambda d: lucid.randn(2, dtype=lucid.int64, device=d),
        lambda d: lucid.rand(2, dtype=lucid.bool_, device=d),
    ],
    ids=["arange-bool", "arange-complex", "linspace-bool", "rand-i32", "randn-i64", "rand-bool"],
)
def test_what_the_reference_refuses_is_refused_in_lucids_words(device: str, make: object) -> None:
    with pytest.raises(NotImplementedError, match="not implemented for"):
        make(device)  # type: ignore[operator]
