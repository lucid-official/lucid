"""An op handed a Metal tensor must not answer on the CPU.

Both of these took a Metal input and returned a CPU tensor, so the very
next op raised ``DeviceMismatch``.  The audit had been reporting them as
SKIP — "output landed on device('cpu'), expected metal" — among 45 cells
that were otherwise factories, where landing on the default device is
correct.  A factory has no input device to follow; a transform does.

``linalg.matrix_rank`` hardcoded ``_C_engine.CPU`` on its last line.  The
rank is read back to a Python int, which is the H3 data-dependent
carve-out and not a mistake, but the tensor built from it ignored where
``A`` lived — unlike ``det``, ``svd``, ``eigvalsh`` and ``matrix_norm``,
which take the same round trip and return the input's device.

``histc`` returned float64 whatever it was given, and MLX has no float64,
so the result could only live on the CPU: a device bug whose actual cause
was a dtype.  The reference reports in the input's dtype.
"""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

MATRIX = np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32)


# ── the dtype contract that caused it ─────────────────────────────────────────


@pytest.mark.parametrize("dtype,expected", [(lucid.float32, "float32"), (lucid.float64, "float64")])
def test_histc_reports_in_the_input_dtype(dtype, expected) -> None:
    values = np.random.default_rng(0).standard_normal(20)
    out = lucid.histc(lucid.tensor(values, dtype=dtype), bins=5)
    assert expected in str(out.dtype)


def test_histc_counts_are_unchanged() -> None:
    values = np.random.default_rng(0).standard_normal(20)
    out = np.asarray(lucid.histc(lucid.tensor(values), bins=5).numpy())
    lo, hi = values.min(), values.max()
    expected, _ = np.histogram(values, bins=5, range=(lo, hi))
    assert np.allclose(out, expected)
    assert out.sum() == values.size


def test_histc_refuses_integer_input() -> None:
    """It used to accept one and answer ``[0, 0, 0, 0]``.

    Zeros are the worst possible wrong answer: the right shape, so
    nothing downstream can tell.  The reference refuses integers too.
    """
    ints = lucid.tensor(np.array([1, 2, 2, 3, 5, 5, 5]), dtype=lucid.int32)
    with pytest.raises(Exception, match="floating"):
        lucid.histc(ints, bins=4)


# ── the device contract ───────────────────────────────────────────────────────


def test_matrix_rank_value_is_unchanged() -> None:
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
    assert int(np.asarray(lucid.linalg.matrix_rank(lucid.tensor(singular)).numpy())) == 1
    full = np.eye(4)
    assert int(np.asarray(lucid.linalg.matrix_rank(lucid.tensor(full)).numpy())) == 4


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_matrix_rank_answers_on_the_input_device(device) -> None:
    out = lucid.linalg.matrix_rank(lucid.tensor(MATRIX, device=device))
    assert device in str(out.device)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_histc_answers_on_the_input_device(device) -> None:
    out = lucid.histc(lucid.tensor(MATRIX, device=device))
    assert device in str(out.device)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_the_results_compose_with_other_metal_tensors() -> None:
    """The concrete harm: the next op raised DeviceMismatch."""
    a = lucid.tensor(MATRIX, device="metal")
    rank = lucid.linalg.matrix_rank(a)
    assert "metal" in str((rank + lucid.tensor(np.array(1), device="metal")).device)
    counts = lucid.histc(a)
    ones = lucid.tensor(np.ones(100, dtype=np.float32), device="metal")
    assert "metal" in str((counts * ones).device)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_the_two_devices_agree_on_the_values() -> None:
    cpu = np.asarray(lucid.histc(lucid.tensor(MATRIX, device="cpu")).numpy())
    metal = np.asarray(lucid.histc(lucid.tensor(MATRIX, device="metal")).numpy())
    assert np.allclose(cpu, metal)
    cpu_rank = np.asarray(lucid.linalg.matrix_rank(lucid.tensor(MATRIX, device="cpu")).numpy())
    metal_rank = np.asarray(lucid.linalg.matrix_rank(lucid.tensor(MATRIX, device="metal")).numpy())
    assert np.array_equal(cpu_rank, metal_rank)


# ── factories decide their own device ─────────────────────────────────────────


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize(
    "build",
    [
        lambda d: lucid.zeros(4, device=d),
        lambda d: lucid.ones(4, device=d),
        lambda d: lucid.arange(0, 4, 1, device=d),
        lambda d: lucid.eye(3, device=d),
        lambda d: lucid.linspace(0.0, 1.0, 5, device=d),
        lambda d: lucid.full((2, 2), 1.5, device=d),
        lambda d: lucid.signal.windows.hann(8, device=d),
        lambda d: lucid.signal.windows.general_cosine(8, [0.5, 0.5], device=d),
    ],
)
def test_a_factory_honours_its_device_argument(build) -> None:
    """No input device to follow, so this is the question worth asking."""
    assert "metal" in str(build("metal").device)
    assert "cpu" in str(build("cpu").device)
