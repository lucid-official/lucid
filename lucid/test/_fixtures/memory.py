"""Engine memory-stats fixtures — a single number a leak test can diff.

``live_bytes`` was redefined byte-for-byte in four test files (autograd's
saved-output, function-apply and spent-frontier suites, plus quantization's
stat-buffer suite) each time a test needed "how many bytes does this device
hold right now" to diff before/after a scope that should free everything
(LCD-536). One copy here keeps the engine call its only owner.

Callers run ``gc.collect()`` themselves immediately before (and often after)
reading this — a CPython reference cycle keeps its tensors, and with them
their engine-side storage, alive until the collector breaks it — this helper
only reads the counter once collection has already happened.
"""

from lucid._C import engine as _C_engine


def live_bytes(device: str) -> int:
    """Bytes the engine currently holds live on ``device`` ("cpu" or "metal")."""
    dev = _C_engine.CPU if device == "cpu" else _C_engine.GPU
    return int(_C_engine.memory_stats(dev).current_bytes)
