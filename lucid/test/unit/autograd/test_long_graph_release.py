"""Dropping a long autograd graph releases it by a loop, not by recursion (LCD-564).

Each edge of the graph owns the node it points at, so the last tensor of a
chain of N nodes going away destroyed N nodes nested one inside the next.
About 200,000 of them overflowed the main thread's 8 MiB stack and took the
process down with SIGSEGV: dropping the graph without a backward, or after
``backward(retain_graph=True)``, where no pass had freed it node by node.  A
thread with a smaller stack went down far sooner.  A node now hands the nodes
it owned to a worklist that the outermost release on its thread drains.

Every case runs in a child process, so a regression fails its test instead of
killing the run.  The release is host-side work, the same on every device;
Metal is given a shorter chain only because MLX caps the live buffers a
process may hold below a million.
"""

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.slow

_PRELUDE = """
import gc
import json
import threading
import time
import weakref

import lucid
from lucid._C import engine as _C_engine

N = 1_000_000


def live_bytes(device):
    dev = _C_engine.CPU if device == "cpu" else _C_engine.GPU
    return int(_C_engine.memory_stats(dev).current_bytes)


# Half the nodes save their input (mul) and half save nothing (neg), so the
# chain is held both through edges and through the tensors nodes keep.  The
# values stay exact: an even number of negations of x times one.
def chain(y, n):
    for i in range(n):
        y = -y if i % 2 else y * 1.0
    return y
"""


def _run(body: str) -> str:
    """Run ``body`` after the prelude in a child; return its stdout."""
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0 and "survived" in proc.stdout, (
        f"exit {proc.returncode} (negative: killed by that signal)\n"
        f"{proc.stderr[-3000:]}"
    )
    return proc.stdout


def _report(stdout: str) -> object:
    return json.loads(stdout.splitlines()[0])


def test_a_dropped_chain_is_freed_whole_in_linear_time() -> None:
    stdout = _run("""
        x = lucid.randn(4, requires_grad=True)
        gc.collect()
        base = live_bytes("cpu")
        rounds = []
        for _ in range(3):
            loss = chain(x, N).sum()
            held = live_bytes("cpu") - base
            start = time.perf_counter()
            del loss
            seconds = time.perf_counter() - start
            gc.collect()
            rounds.append({"held": held, "seconds": seconds, "left": live_bytes("cpu") - base})
        print(json.dumps(rounds))
        print("survived")
    """)
    rounds = _report(stdout)
    assert isinstance(rounds, list) and len(rounds) == 3
    for r in rounds:
        # The chain held something to free, and freeing it left nothing.
        assert r["held"] > 0, rounds
        assert r["left"] == 0, rounds
        assert r["seconds"] < 2.0, rounds


def test_a_chain_is_dropped_after_a_backward_that_retained_it() -> None:
    stdout = _run("""
        x = lucid.randn(4, requires_grad=True)
        loss = chain(x, N).sum()
        loss.backward(retain_graph=True)
        grad = x.grad.tolist()
        del loss
        print(json.dumps(grad))
        print("survived")
    """)
    assert _report(stdout) == [1.0, 1.0, 1.0, 1.0]


def test_a_chain_is_dropped_on_a_thread_with_a_small_stack(device: str) -> None:
    # On a 512 KiB stack the recursive release overflowed near 10,000 nodes.
    length = 1_000_000 if device == "cpu" else 100_000
    stdout = _run(f"""
        DEVICE, LENGTH = {device!r}, {length}
        result = {{}}


        def run():
            x = lucid.randn(4, requires_grad=True, device=DEVICE)
            loss = chain(x, LENGTH).sum()
            del loss
            loss = chain(x, LENGTH).sum()
            loss.backward(retain_graph=True)
            del loss
            result["grad"] = x.grad.tolist()


        threading.stack_size(512 * 1024)
        worker = threading.Thread(target=run)
        worker.start()
        worker.join()
        print(json.dumps(result.get("grad")))
        print("survived")
    """)
    assert _report(stdout) == [1.0, 1.0, 1.0, 1.0]


def test_a_tree_of_chains_is_dropped_through_its_last_python_held_node() -> None:
    # One node with 1,001 edges: a chain half the graph long, and 1,000
    # branches over a trunk they all share, which only the last branch to go
    # frees.  The worklist grows wide as well as draining deep.  The last
    # owner is a grad_fn held from Python, whose holder frees it.
    stdout = _run("""
        x = lucid.randn(4, requires_grad=True)
        gc.collect()
        base = live_bytes("cpu")
        deep = chain(x, N // 2)
        trunk = chain(x, 1000)
        branches = [chain(trunk, 499) for _ in range(1000)]
        fn = lucid.stack([deep, *branches]).sum().grad_fn
        del deep, trunk, branches
        held = live_bytes("cpu") - base
        del fn
        gc.collect()
        print(json.dumps({"held": held, "left": live_bytes("cpu") - base}))
        print("survived")
    """)
    report = _report(stdout)
    assert report["held"] > 0 and report["left"] == 0, report


def test_a_chain_with_custom_functions_in_it_is_dropped() -> None:
    # A custom Function every 1,000 nodes: its node holds Python objects —
    # the ctx with the saved input and an attribute — that its own destructor
    # releases under the GIL.  Each ctx's marker dies with its node.
    stdout = _run("""
        class Marker:
            pass


        markers = []


        class Scale(lucid.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                ctx.marker = Marker()
                markers.append(weakref.ref(ctx.marker))
                return x * 1.0

            @staticmethod
            def backward(ctx, grad):
                (x,) = ctx.saved_tensors
                return grad * 1.0


        x = lucid.randn(4, requires_grad=True)
        y = x
        for _ in range(N // 1000):
            y = Scale.apply(chain(y, 999))
        loss = y.sum()
        del y
        loss.backward(retain_graph=True)
        grad = x.grad.tolist()
        del loss
        print(json.dumps({"grad": grad, "made": len(markers),
                          "alive": sum(m() is not None for m in markers)}))
        print("survived")
    """)
    report = _report(stdout)
    assert report == {"grad": [1.0, 1.0, 1.0, 1.0], "made": 1000, "alive": 0}, report
