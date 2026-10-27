"""Thread-safety guard for ``run_executable_inplace`` on Metal view families (LCD-377).

``run_executable_inplace`` runs the MPSGraph executable with the GIL released
and then swaps the resulting GPU buffers into ``output_targets``.  For a target
that belongs to a Metal view family, ``mutable_storage()`` replaces the shared
``family_->root`` and invalidates every sibling's cached ``packed_`` storage
via ``drop_family_caches()``.  Performing that swap while the GIL is released
races with another Python thread holding the GIL and reading a sibling's
``storage()``.
"""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.compile import _tracing, compile_optimizer, fused_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _compile_increment_exe(
    sample: lucid.Tensor,
) -> tuple[
    _C_engine.compile.CompiledExecutable,
    list[_C_engine.TensorImpl],
    _C_engine.TensorImpl,
]:
    step = lucid.ones_like(sample)
    with lucid.no_grad(), _tracing() as tracer:
        out = sample + step
    lucid.eval(sample, step, out)
    assert out.shape == sample.shape
    exe = _C_engine.compile.compile_trace(tracer.graph, tracer.external_feeds)
    assert exe is not None
    feed_map = {int(tid): impl for tid, impl in tracer.external_feeds.items()}
    sample_impl = _unwrap(sample)
    feeds = [
        sample_impl if feed_map[int(tid)] is sample_impl else feed_map[int(tid)]
        for tid in exe.input_ids
    ]
    return exe, feeds, sample_impl


def test_run_executable_inplace_with_concurrent_sibling_reads() -> None:
    base = lucid.zeros(16, device=COMPILE_DEVICE)
    target_4x4 = base.view(4, 4)
    target_2x8 = base.view(2, 8)
    sibling_flat = base.view(16)
    sibling_2d = base.view(2, 8)

    exe_4x4, feeds_4x4, impl_4x4 = _compile_increment_exe(target_4x4)
    exe_2x8, feeds_2x8, impl_2x8 = _compile_increment_exe(target_2x8)

    steps = 160
    barrier = threading.Barrier(3)
    done = threading.Event()

    def writer() -> None:
        barrier.wait()
        try:
            for k in range(steps):
                if k % 2 == 0:
                    _C_engine.compile.run_executable_inplace(exe_4x4, feeds_4x4, [impl_4x4])
                else:
                    _C_engine.compile.run_executable_inplace(exe_2x8, feeds_2x8, [impl_2x8])
        finally:
            done.set()

    def reader(view: lucid.Tensor, expected_shape: tuple[int, ...]) -> None:
        barrier.wait()
        while not done.is_set():
            assert view.shape == expected_shape
            assert view._impl.version >= 0
            vals = view.reshape(-1).tolist()
            assert len(vals) == 16
            first = float(vals[0])
            assert 0.0 <= first <= float(steps)
            assert all(float(v) == first for v in vals)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(writer),
            pool.submit(reader, sibling_flat, (16,)),
            pool.submit(reader, sibling_2d, (2, 8)),
        ]
        for fut in futures:
            fut.result(timeout=30.0)

    expected = [float(steps)] * 16
    assert sibling_flat.tolist() == expected
    assert sibling_2d.reshape(-1).tolist() == expected
    assert base.tolist() == expected


@pytest.mark.parametrize("mode", ["compile_optimizer", "fused_step"])
def test_compiled_step_with_concurrent_parameter_sibling_reads(mode: str) -> None:
    lucid.manual_seed(0)
    model = nn.Linear(8, 4, bias=False).to(COMPILE_DEVICE)
    alias = model.weight.data
    flat_sibling = model.weight.data.view(-1)
    reshaped_sibling = model.weight.data.view(2, 16)

    opt = optim.SGD(model.parameters(), lr=0.01)
    x = lucid.ones(4, 8, device=COMPILE_DEVICE)

    if mode == "compile_optimizer":
        c_opt = compile_optimizer(opt)

        def step_once() -> None:
            opt.zero_grad()
            loss = (model(x) ** 2).mean()
            loss.backward()
            c_opt.step()

    else:
        f_step = fused_step(model, lambda out, tgt: ((out - tgt) ** 2).mean(), opt)
        target_y = lucid.zeros(4, 4, device=COMPILE_DEVICE)

        def step_once() -> None:
            f_step(x, target_y)

    # Warm up (trace + compile) before spawning threads.
    step_once()

    steps = 80
    barrier = threading.Barrier(2)
    done = threading.Event()

    def writer() -> None:
        barrier.wait()
        try:
            for _ in range(steps):
                step_once()
        finally:
            done.set()

    def reader() -> None:
        barrier.wait()
        while not done.is_set():
            assert alias.shape == (4, 8)
            assert flat_sibling.shape == (32,)
            assert reshaped_sibling.shape == (2, 16)
            vals_a = flat_sibling.tolist()
            vals_b = reshaped_sibling.reshape(-1).tolist()
            assert len(vals_a) == 32
            assert len(vals_b) == 32

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_writer = pool.submit(writer)
        f_reader = pool.submit(reader)
        f_writer.result(timeout=30.0)
        f_reader.result(timeout=30.0)

    final_weight = model.weight.reshape(-1).tolist()
    assert alias.reshape(-1).tolist() == final_weight
    assert flat_sibling.tolist() == final_weight
    assert reshaped_sibling.reshape(-1).tolist() == final_weight
