"""Microbenchmarks — per-op timing on a fixed shape.

These tests are tagged ``@pytest.mark.perf`` so the default fast tier
skips them.  Run explicitly with::

    python3 -m pytest lucid/test/perf/ -m perf

When ``pytest-benchmark`` is installed each test reports min / mean /
stddev; otherwise the ``bench`` fixture falls back to a one-shot
``time.perf_counter`` so the suite still runs end-to-end.

A golden ``_thresholds.json`` next to this file pins the upper bound
each timing must stay under (with a 25 % tolerance applied).  Tests
without an entry simply record their timing without asserting a
limit — useful while a baseline is being established.
"""

from pathlib import Path

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.perf import assert_no_regression, load_thresholds

_THRESHOLDS = load_thresholds(Path(__file__).parent)


@pytest.mark.perf
class TestArithMicrobench:
    def test_add_2048(self, bench, device: str) -> None:
        a = lucid.tensor(np.random.standard_normal((2048,)).astype(np.float32), device=device)
        b = lucid.tensor(np.random.standard_normal((2048,)).astype(np.float32), device=device)
        bench(lambda: a + b)
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"add_2048_{device}", bench.last_elapsed, _THRESHOLDS)

    def test_mul_2048(self, bench, device: str) -> None:
        a = lucid.tensor(np.random.standard_normal((2048,)).astype(np.float32), device=device)
        b = lucid.tensor(np.random.standard_normal((2048,)).astype(np.float32), device=device)
        bench(lambda: a * b)
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"mul_2048_{device}", bench.last_elapsed, _THRESHOLDS)


@pytest.mark.perf
class TestMatmulMicrobench:
    def test_matmul_64x64(self, bench, device: str) -> None:
        a = lucid.tensor(np.random.standard_normal((64, 64)).astype(np.float32), device=device)
        b = lucid.tensor(np.random.standard_normal((64, 64)).astype(np.float32), device=device)
        bench(lambda: a @ b)
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"matmul_64_{device}", bench.last_elapsed, _THRESHOLDS)

    def test_matmul_256x256(self, bench, device: str) -> None:
        a = lucid.tensor(np.random.standard_normal((256, 256)).astype(np.float32), device=device)
        b = lucid.tensor(np.random.standard_normal((256, 256)).astype(np.float32), device=device)
        bench(lambda: a @ b)
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"matmul_256_{device}", bench.last_elapsed, _THRESHOLDS)


@pytest.mark.perf
class TestReductionMicrobench:
    def test_sum_4096(self, bench, device: str) -> None:
        a = lucid.tensor(np.random.standard_normal((4096,)).astype(np.float32), device=device)
        bench(lambda: lucid.sum(a))
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"sum_4096_{device}", bench.last_elapsed, _THRESHOLDS)


@pytest.mark.perf
class TestActivationMicrobench:
    def test_softmax_1024(self, bench, device: str) -> None:
        from lucid.nn.functional import softmax

        a = lucid.tensor(np.random.standard_normal((32, 1024)).astype(np.float32), device=device)
        bench(lambda: softmax(a, dim=-1))
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"softmax_1024_{device}", bench.last_elapsed, _THRESHOLDS)

    def test_tanh_1m(self, bench, device: str) -> None:
        a = lucid.tensor(np.random.standard_normal((1_048_576,)).astype(np.float32), device=device)
        bench(lambda: lucid.tanh(a))
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"tanh_1m_{device}", bench.last_elapsed, _THRESHOLDS)

    def test_gelu_tanh_1m(self, bench, device: str) -> None:
        from lucid.nn.functional import gelu

        a = lucid.tensor(np.random.standard_normal((1_048_576,)).astype(np.float32), device=device)
        bench(lambda: gelu(a, approximate="tanh"))
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"gelu_tanh_1m_{device}", bench.last_elapsed, _THRESHOLDS)


@pytest.mark.perf
class TestSortMicrobench:
    def test_argsort_100k(self, bench, device: str) -> None:
        a = lucid.tensor(np.random.standard_normal((100_000,)).astype(np.float32), device=device)
        bench(lambda: lucid.argsort(a))
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"argsort_100k_{device}", bench.last_elapsed, _THRESHOLDS)

    def test_unique_100k(self, bench, device: str) -> None:
        a = lucid.tensor(np.random.standard_normal((100_000,)).astype(np.float32), device=device)
        bench(lambda: lucid.unique(a, return_inverse=True, return_counts=True))
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(f"unique_100k_{device}", bench.last_elapsed, _THRESHOLDS)

    def test_argsort_16384x64_int64(self, bench, device: str) -> None:
        rng = np.random.default_rng(0)
        a = lucid.tensor(
            rng.integers(-10_000, 10_000, size=(16_384, 64), dtype=np.int64),
            device=device,
        )
        bench(lambda: lucid.argsort(a, dim=-1))
        if hasattr(bench, "last_elapsed"):
            assert_no_regression(
                f"argsort_16384x64_int64_{device}", bench.last_elapsed, _THRESHOLDS
            )
