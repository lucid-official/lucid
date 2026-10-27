"""What a prediction does to the rest of the process.

A package is a deployment artifact, and deployments are threaded: a
request handler, a metrics thread, an event loop, a health check. None
of those are predicting, and all of them used to stop dead while
something else was.

The engine never released the interpreter lock. It costs nothing where
an operation takes microseconds, which is every other call in Lucid, and
it is a different thing entirely for Core ML: one prediction of a
moderate convolutional stack runs for tens of milliseconds, and the lock
was held across all of it. Measured before the fix, a 53 ms prediction
froze an unrelated Python thread for 78 ms — the prediction plus the
work either side of it. A server predicting continuously would not have
run anything else at all.

Nothing in ``predict`` or ``classify`` touches Python — they take
tensor implementations and return them — so the lock is dropped for the
call itself and taken again before the result is converted. Compiling a
package, which is the longest call here at seconds for a large one, does
the same.

Two things that had been resting on the lock without saying so:

The handle could be closed from another thread while a prediction was in
flight. Serialised by the lock, that was impossible; without it, the
model would be freed underneath the call. It is held by shared ownership
now, so closing during a prediction frees the model after that
prediction rather than during it.

A stateful package keeps one ``MLState`` that every prediction reads and
writes. Core ML's prediction is safe to call concurrently; sharing that
object across concurrent calls is not, and the result would be lost
updates rather than a crash — the quiet kind. Those predictions queue.

Throughput is a separate question and the lock was never the answer to
it: four threads reach 1.3x one thread both before and after, because
the Neural Engine is one piece of hardware and the work queues there.
What changed is that the rest of the process keeps running.
"""

import threading
import time

import pytest

import lucid
import lucid.nn as nn
import lucid.coreml as cml
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


def _slow_enough() -> nn.Module:
    """A prediction long enough that a stall would be unmistakable.

    Around fourteen milliseconds here. The assertions below are written
    against the measured figure rather than a constant, so a faster or
    slower machine moves both sides together.
    """
    return nn.Sequential(
        *[layer for _ in range(12) for layer in (nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())]
    ).eval()


@pytest.fixture(scope="module")
def heavy(tmp_path_factory):
    lucid.manual_seed(0)
    path = tmp_path_factory.mktemp("threading") / "heavy.mlpackage"
    exported = cml.export(
        _slow_enough(), lucid.randn(8, 64, 112, 112), str(path), precision=cml.Precision.FLOAT16
    )
    yield exported, lucid.randn(8, 64, 112, 112), str(path)
    exported.close()


def _longest_stall(during) -> tuple[float, object]:
    """The longest a second thread went without being scheduled.

    A thread that only reads the clock is entirely at the interpreter's
    mercy: if the lock is held elsewhere it records exactly how long for.
    """
    stop, worst = threading.Event(), []

    def watch() -> None:
        longest, last = 0.0, time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            longest = max(longest, now - last)
            last = now
        worst.append(longest)

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        outcome = during()
    finally:
        stop.set()
        watcher.join()
    return worst[0], outcome


class TestOtherThreadsKeepRunning:
    def test_a_prediction_does_not_freeze_the_process(self, heavy) -> None:
        """The defect: a 14 ms prediction was a 14 ms freeze."""
        exported, x, _path = heavy
        for _ in range(3):
            exported.predict(x)
        start = time.perf_counter()
        for _ in range(10):
            exported.predict(x)
        each = (time.perf_counter() - start) / 10

        # The best of three runs, because one unlucky garbage collection
        # can stall a thread for milliseconds and that is not what is
        # being measured. Held, the lock stalls *every* run by at least a
        # whole prediction — there is no lucky round — so the minimum is
        # the statistic that separates the two.
        stalls = [
            _longest_stall(lambda: [exported.predict(x) for _ in range(20)])[0] for _ in range(3)
        ]
        best = min(stalls)
        assert best < each / 3, (
            f"best stall {best * 1e3:.1f} ms of {[f'{s * 1e3:.1f}' for s in stalls]} "
            f"against a {each * 1e3:.1f} ms prediction"
        )

    def test_compiling_a_package_does_not_either(self, heavy) -> None:
        """Loading is the longest call in the subsystem."""
        _exported, _x, path = heavy
        stall, opened = _longest_stall(lambda: cml.load(path))
        try:
            assert stall < 0.05
        finally:
            opened.close()


class TestPredictingFromSeveralThreads:
    def test_the_answers_are_the_ones_a_single_thread_gets(self, heavy) -> None:
        """Releasing the lock must not have made the results a race."""
        exported, x, _path = heavy
        wanted = exported.predict(x)

        answers: list[object] = []
        guard = threading.Lock()

        def work() -> None:
            got = exported.predict(x)
            with guard:
                answers.append(got)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(answers) == 4
        for got in answers:
            assert float((got - wanted).abs().max().item()) == 0.0


class TestAStatefulPackageQueuesInstead:
    """One ``MLState``, shared, read and written by every prediction."""

    def test_no_update_is_lost(self, tmp_path) -> None:
        """The failure would be silent: a smaller total, no error.

        Each prediction adds one to the carried value and returns twice
        it, so the largest answer any thread saw is twice the number of
        predictions — exactly, and only if every one of them read what
        the last had written.
        """

        class _Counts(nn.Module):
            def forward(
                self, x: lucid.Tensor, cache: lucid.Tensor
            ) -> tuple[lucid.Tensor, lucid.Tensor]:
                carried = cache + x
                return carried, carried * 2.0

        lucid.manual_seed(0)
        exported = cml.export(
            _Counts().eval(),
            {"x": lucid.ones(1, 4), "cache": lucid.zeros(1, 4)},
            str(tmp_path / "counted.mlpackage"),
            precision=cml.Precision.FLOAT16,
            state=[cml.State(input="cache", output="output_0")],
        )
        try:
            x = lucid.ones(1, 4)
            each, threads = 40, 4
            seen: list[float] = []
            guard = threading.Lock()

            def work() -> None:
                for _ in range(each):
                    got = float(exported.predict(x).max().item())
                    with guard:
                        seen.append(got)

            running = [threading.Thread(target=work) for _ in range(threads)]
            for thread in running:
                thread.start()
            for thread in running:
                thread.join()

            assert len(seen) == each * threads
            assert max(seen) == pytest.approx(2.0 * each * threads)
        finally:
            exported.close()


class TestClosingWhileAPredictionIsRunning:
    """Held by shared ownership, so the model outlives the call using it."""

    def test_it_does_not_free_the_model_underneath(self, tmp_path) -> None:
        lucid.manual_seed(0)
        exported = cml.export(
            _slow_enough(),
            lucid.randn(8, 64, 112, 112),
            str(tmp_path / "closing.mlpackage"),
            precision=cml.Precision.FLOAT16,
        )
        x = lucid.randn(8, 64, 112, 112)
        exported.predict(x)

        outcome: list[object] = []

        def work() -> None:
            try:
                outcome.append(exported.predict(x))
            except RuntimeError as refusal:
                outcome.append(refusal)

        worker = threading.Thread(target=work)
        worker.start()
        time.sleep(0.002)
        exported.close()
        worker.join()

        # Either it finished with the model it took, or it never got one.
        # What it must not do is run against a freed one.
        assert len(outcome) == 1
        assert isinstance(outcome[0], (lucid.Tensor, RuntimeError))
        with pytest.raises(RuntimeError, match="closed"):
            exported.predict(x)
