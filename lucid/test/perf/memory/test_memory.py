"""Peak-memory regression tests.

These use ``tracemalloc`` to bound the Python-side allocation
footprint of representative ops.  Engine-side (MLX / Accelerate)
allocations aren't tracked here — for those, watch the integration
suite's loss curves and the microbench timings instead.

Tagged ``@pytest.mark.perf`` so they're opt-in via ``-m perf``.
"""

import tracemalloc

import numpy as np
import pytest

import lucid

_PY_PEAK_THRESHOLDS_BYTES = {
    "tensor_alloc_2048_f32": 200_000,
    "matmul_alloc_64x64_f32": 400_000,
}


@pytest.mark.perf
class TestPythonSideMemory:
    def test_tensor_alloc(self, device: str) -> None:
        tracemalloc.start()
        for _ in range(10):
            t = lucid.tensor(np.zeros(2048, dtype=np.float32), device=device)
            del t
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        threshold = _PY_PEAK_THRESHOLDS_BYTES["tensor_alloc_2048_f32"]
        assert peak < threshold * 5, f"python-side peak {peak} > 5x threshold {threshold}"

    def test_matmul_alloc(self, device: str) -> None:
        a = lucid.tensor(np.zeros((64, 64), dtype=np.float32), device=device)
        b = lucid.tensor(np.zeros((64, 64), dtype=np.float32), device=device)
        tracemalloc.start()
        for _ in range(10):
            c = (a @ b).numpy()
            del c
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        threshold = _PY_PEAK_THRESHOLDS_BYTES["matmul_alloc_64x64_f32"]
        assert peak < threshold * 5
