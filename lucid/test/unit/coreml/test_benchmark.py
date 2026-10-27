"""Timing a package, and timing it the way it should be timed.

The reason this subsystem exists is that Core ML reaches hardware Lucid
cannot: a ResNet-18 at 224 square takes 18.6 ms eager on this machine,
4.5 ms as a float32 package on the CPU, and 1.5 ms once the Neural
Engine is allowed. That claim had never been re-measured with the
current writer — the figure in the plan came from a prototype that was
thrown away.

These tests do not assert any of those numbers. A latency threshold on a
shared laptop is a test that fails for reasons that have nothing to do
with the code, and the numbers belong in a note rather than in an
assertion. What is checked is that the measurement is a measurement:
that the warmup is thrown away rather than averaged in, that the best is
not above the median, that the settings which decide the answer are
carried with it, and that asking for no timed calls is refused instead of
returning a statistic over nothing.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.coreml as cml
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


def _model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU()).eval()


class TestTheMeasurementIsAMeasurement:
    def test_it_reports_a_median_and_a_best(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        exported = cml.export(_model(), lucid.randn(1, 3, 32, 32), f"{tmp_path}/t.mlpackage")
        try:
            timing = exported.benchmark(lucid.randn(1, 3, 32, 32), repeats=8)
            assert timing.repeats == 8
            assert timing.best_ms > 0.0
            assert timing.best_ms <= timing.median_ms
        finally:
            exported.close()

    def test_it_carries_what_decided_the_answer(self, tmp_path: object) -> None:
        """A latency without its settings says nothing.

        The same package is several times slower with the accelerator
        withheld, and float32 forfeits the accelerator altogether — so a
        number quoted without both is not comparable to any other.
        """
        lucid.manual_seed(0)
        exported = cml.export(
            _model(),
            lucid.randn(1, 3, 32, 32),
            f"{tmp_path}/settings.mlpackage",
            precision=cml.Precision.FLOAT16,
            compute_units=cml.ComputeUnits.CPU_ONLY,
        )
        try:
            timing = exported.benchmark(lucid.randn(1, 3, 32, 32), repeats=4)
            assert timing.compute_units is cml.ComputeUnits.CPU_ONLY
            assert timing.precision == "FLOAT16"
            assert "CPU_ONLY" in repr(timing)
        finally:
            exported.close()

    def test_the_warmup_is_thrown_away(self, tmp_path: object) -> None:
        """Core ML defers work to the first calls.

        Specialising for the units it was given, laying weights out the
        way the accelerator wants — a timing that includes those reports
        the setup rather than the model. Measured by asking twice: the
        second run has nothing left to warm up, so a warmup that was
        being averaged in would show as the first run being slower by a
        margin the second never has.
        """
        lucid.manual_seed(0)
        x = lucid.randn(1, 3, 32, 32)
        exported = cml.export(_model(), x, f"{tmp_path}/warm.mlpackage")
        try:
            first = exported.benchmark(x, repeats=10, warmup=5)
            second = exported.benchmark(x, repeats=10, warmup=5)
            # Generous: this is checking that the first measurement is
            # not dominated by setup, not that the two agree closely.
            assert first.median_ms < second.median_ms * 5.0
        finally:
            exported.close()

    def test_no_timed_calls_is_refused(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        x = lucid.randn(1, 3, 32, 32)
        exported = cml.export(_model(), x, f"{tmp_path}/none.mlpackage")
        try:
            with pytest.raises(ValueError, match="at least one"):
                exported.benchmark(x, repeats=0)
        finally:
            exported.close()

    def test_a_warmup_of_zero_is_allowed(self, tmp_path: object) -> None:
        """Somebody measuring cold start wants exactly that."""
        lucid.manual_seed(0)
        x = lucid.randn(1, 3, 32, 32)
        exported = cml.export(_model(), x, f"{tmp_path}/cold.mlpackage")
        try:
            timing = exported.benchmark(x, repeats=2, warmup=0)
            assert timing.repeats == 2
        finally:
            exported.close()
