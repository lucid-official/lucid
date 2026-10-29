"""A Metal tensor or module made in one thread runs in any other (LCD-671).

MLX gives every thread its own default stream and keeps each stream's
command encoder in thread-local storage.  A graph node records the stream it
was built on, so a tensor whose work was still pending could be run only by
the thread that built it.  Any other thread raised ``There is no
Stream(gpu, N) in current thread``.  A worker that returned a lazy tensor
and exited left it unrunnable for good.  Lucid promises one Metal stream per
process, the way a device's default stream works in the reference framework.

Each case below hands over tensors that are still pending.  Reading a value
before the handover would hide the defect.  The expected values come from
running the same thing again afterwards in a single thread.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import threading

import pytest

import lucid
from lucid._C import engine as _C_engine
import lucid.nn as nn
import lucid.optim as optim
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")


def _in_thread[T](fn: Callable[[], T]) -> T:
    # A fresh thread per call, so the work is that thread's first.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn).result()


def _lazy_input(shift: float = 0.0) -> lucid.Tensor:
    x = lucid.arange(12, dtype=lucid.float32, device="metal").reshape(3, 4)
    return x * 0.25 - 1.0 + shift


def _model() -> nn.Module:
    lucid.manual_seed(0)
    return nn.Sequential(
        nn.Linear(4, 8, device="metal"),
        nn.Tanh(),
        nn.Linear(8, 2, device="metal"),
    )


def _grads(model: nn.Module, x: lucid.Tensor) -> list[list[float]]:
    model.zero_grad()
    (model(x) ** 2).sum().backward()
    return [p.grad.tolist() for p in model.parameters()]


def _train(model: nn.Module, x: lucid.Tensor) -> None:
    opt = optim.Adam(model.parameters(), lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        (model(x) ** 2).sum().backward()
        opt.step()


def _params(model: nn.Module) -> list[list[float]]:
    return [p.tolist() for p in model.parameters()]


def _close(got: object, want: object) -> bool:
    if isinstance(got, list) and isinstance(want, list):
        return len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want, strict=True))
    assert isinstance(got, float) and isinstance(want, float), (got, want)
    return abs(got - want) <= 1e-5 * max(1.0, abs(want))


class TestMadeHereUsedThere:
    def test_tensor_op_and_item(self) -> None:
        x = _lazy_input()
        got = _in_thread(lambda: (x * 2).sum().item())
        assert _close(got, (_lazy_input() * 2).sum().item())

    def test_tolist(self) -> None:
        x = _lazy_input()
        got = _in_thread(x.tolist)
        assert _close(got, _lazy_input().tolist())

    def test_module_forward(self) -> None:
        model, x = _model(), _lazy_input()
        got = _in_thread(lambda: model(x).tolist())
        assert _close(got, _model()(_lazy_input()).tolist())

    def test_backward(self) -> None:
        model, x = _model(), _lazy_input()
        got = _in_thread(lambda: _grads(model, x))
        assert _close(got, _grads(_model(), _lazy_input()))

    def test_optimizer_step(self) -> None:
        model, x = _model(), _lazy_input()
        _in_thread(lambda: _train(model, x))
        want = _model()
        _train(want, _lazy_input())
        assert _close(_params(model), _params(want))

    def test_synchronize_with_another_threads_pending_work(self) -> None:
        x = _lazy_input()
        y = x * 3
        _in_thread(lucid.metal.synchronize)
        assert _close(y.tolist(), (_lazy_input() * 3).tolist())


class TestMadeThereUsedHere:
    def test_tensor_from_a_finished_thread(self) -> None:
        x = _in_thread(lambda: _lazy_input() * 2)
        assert _close(x.sum().item(), (_lazy_input() * 2).sum().item())

    def test_module_from_a_finished_thread(self) -> None:
        model, x = _in_thread(lambda: (_model(), _lazy_input()))
        assert _close(model(x).tolist(), _model()(_lazy_input()).tolist())

    def test_trained_in_a_finished_thread(self) -> None:
        def build_and_train() -> nn.Module:
            model = _model()
            _train(model, _lazy_input())
            return model

        model = _in_thread(build_and_train)
        want = _model()
        _train(want, _lazy_input())
        assert _close(_params(model), _params(want))


class TestConcurrent:
    N = 4

    def _run(self, model: nn.Module, i: int) -> list[list[list[float]]]:
        x = _lazy_input(shift=float(i))
        return [model(x * (k + 1)).tolist() for k in range(5)]

    def test_threads_share_one_model(self) -> None:
        model = _model()
        start = threading.Barrier(self.N)

        def work(i: int) -> list[list[list[float]]]:
            start.wait()
            return self._run(model, i)

        with ThreadPoolExecutor(max_workers=self.N) as pool:
            got = list(pool.map(work, range(self.N)))
        serial = [self._run(_model(), i) for i in range(self.N)]
        assert _close(got, serial)

    def test_threads_each_with_their_own_model(self) -> None:
        start = threading.Barrier(self.N)
        # One generator serves every thread: seeding and drawing must not
        # interleave, or the models differ from the serial ones.
        seeding = threading.Lock()

        def work(i: int) -> list[list[list[float]]]:
            with seeding:
                model = _model()
            start.wait()
            return self._run(model, i)

        with ThreadPoolExecutor(max_workers=self.N) as pool:
            got = list(pool.map(work, range(self.N)))
        serial = [self._run(_model(), i) for i in range(self.N)]
        assert _close(got, serial)

    def test_threads_train_one_model(self) -> None:
        # Backward and optimizer steps from four threads on one model: every
        # thread's backward accumulates into the shared gradients, then each
        # steps the layer it owns.  The barriers fix the order, so the result
        # is the serial one.
        def run(concurrent: bool) -> list[list[float]]:
            lucid.manual_seed(0)
            model = nn.Sequential(*[nn.Linear(4, 4, device="metal") for _ in range(self.N)])
            opts = [optim.SGD(layer.parameters(), lr=0.05, momentum=0.9) for layer in model]
            phase = threading.Barrier(self.N)

            def backward(i: int) -> None:
                loss = (model(_lazy_input(shift=float(i)).tanh()) ** 2).sum()
                loss.item()
                loss.backward()

            def step(i: int) -> None:
                opts[i].step()
                lucid.eval(*model[i].parameters())
                opts[i].zero_grad()

            def work(i: int) -> None:
                for _ in range(2):
                    phase.wait()
                    backward(i)
                    phase.wait()
                    step(i)

            if concurrent:
                with ThreadPoolExecutor(max_workers=self.N) as pool:
                    list(pool.map(work, range(self.N)))
            else:
                for _ in range(2):
                    for i in range(self.N):
                        backward(i)
                    for i in range(self.N):
                        step(i)
            return _params(model)

        assert _close(run(concurrent=True), run(concurrent=False))

    def test_threads_read_back_every_way(self) -> None:
        # Each read waits for the GPU without the GIL; the encoding before it
        # must not overlap another thread's.
        model = _model()
        reads: list[Callable[[lucid.Tensor], object]] = [
            lambda t: t.sum().item(),
            lambda t: t.tolist(),
            lambda t: t.numpy().tolist(),
            lambda t: (lucid.eval(t), t.tolist())[1],
            lambda t: (lucid.metal.synchronize(), t.tolist())[1],
        ]

        def run(i: int) -> list[object]:
            x = _lazy_input(shift=float(i))
            return [read(model(x * (k + 1))) for k, read in enumerate(reads * 4)]

        start = threading.Barrier(self.N)

        def work(i: int) -> list[object]:
            start.wait()
            return run(i)

        with ThreadPoolExecutor(max_workers=self.N) as pool:
            got = list(pool.map(work, range(self.N)))
        assert _close(got, [run(i) for i in range(self.N)])

    def test_threads_run_one_compiled_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A compiled run waits without the GIL.  Its feeds are pending and
        # strided, so they need MLX work before MPSGraph can read them, and
        # that work must happen before the GIL is let go.
        runs = [0]
        counting = threading.Lock()
        run_executable = _C_engine.compile.run_executable

        def counted(*args: object) -> object:
            with counting:
                runs[0] += 1
            return run_executable(*args)

        monkeypatch.setattr(_C_engine.compile, "run_executable", counted)
        model = _model()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        compiled = lucid.compile(model)

        def run(i: int) -> list[list[list[float]]]:
            x = lucid.arange(12, dtype=lucid.float32, device="metal").reshape(4, 3) * 0.1
            return [compiled((x * (k + 1) + i).T).tolist() for k in range(20)]

        want = [run(i) for i in range(self.N)]
        start = threading.Barrier(self.N)

        def work(i: int) -> list[list[list[float]]]:
            start.wait()
            return run(i)

        before = runs[0]
        with ThreadPoolExecutor(max_workers=self.N) as pool:
            got = list(pool.map(work, range(self.N)))
        assert runs[0] - before == 20 * self.N
        assert _close(got, want)


_FFT_FIRST = """
from concurrent.futures import ThreadPoolExecutor
import lucid
x = lucid.arange(16, dtype=lucid.float32, device="metal").reshape(4, 4) * 0.5 + 1.0
with ThreadPoolExecutor(max_workers=1) as pool:
    y = pool.submit(lambda: lucid.fft.fft(x)).result()
print(y.abs().sum().item())
"""


class TestFirstWorkOutsideTheDispatcher:
    # Run apart: today the read in the main thread hangs rather than raises.
    @pytest.mark.xfail(
        strict=True,
        raises=(AssertionError, subprocess.TimeoutExpired),
        reason="LCD-671 follow-up: an op that never asks the dispatcher (fft) as a thread's "
        "first MLX work builds on that thread's own CPU stream, and reading the result in "
        "another thread hangs; wrap_mlx_array must bind the thread (MlxBridge.cpp)",
    )
    def test_fft_is_a_threads_first_op(self) -> None:
        done = subprocess.run(
            [sys.executable, "-c", _FFT_FIRST], capture_output=True, text=True, timeout=20
        )
        assert done.returncode == 0, done.stderr[-400:]
