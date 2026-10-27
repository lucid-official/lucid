"""Dropping a long autograd graph releases it by a loop, not by recursion (LCD-564).

Each edge of the graph owns the node it points at, so the last tensor of a
chain of N nodes going away destroyed N nodes nested one inside the next.
About 200,000 of them overflowed the main thread's 8 MiB stack and took the
process down with SIGSEGV: dropping the graph without a backward, or after
``backward(retain_graph=True)``, where no pass had freed it node by node.  A
512 KiB thread went down near 10,000.  A node now hands the nodes it owned to
a worklist that the outermost release on its thread drains.

Every case runs in a child process, so a regression fails its test instead of
killing the run.  All but one run on a 512 KiB stack, where 50,000 nodes are
five times what the recursion survived and cost a fraction of a second; one
runs on the main thread, past where it overflowed there.
"""

import json
import subprocess
import sys
import textwrap

_PRELUDE = """
import gc
import json
import threading
import time
import weakref

import lucid
from lucid._C import engine as _C_engine

N = 50_000


def live_bytes(device="cpu"):
    dev = _C_engine.CPU if device == "cpu" else _C_engine.GPU
    return int(_C_engine.memory_stats(dev).current_bytes)


# Half the nodes save their input (mul) and half save nothing (neg), so the
# chain is held both through edges and through the tensors nodes keep.  The
# values stay exact: an even number of negations of x times one.
def chain(y, n):
    for i in range(n):
        y = -y if i % 2 else y * 1.0
    return y


# Run fn on a 512 KiB stack; its result, or None when it raised (the thread
# prints the traceback).
def on_small_stack(fn):
    out = {}

    def run():
        out["result"] = fn()

    threading.stack_size(512 * 1024)
    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    return out.get("result")


def report(result):
    print(json.dumps(result))
    print("survived")
"""


def _run(body: str) -> tuple[object, str]:
    """Run ``body`` after the prelude in a child; what it reported, and its stderr."""
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0 and "survived" in proc.stdout, (
        f"exit {proc.returncode} (negative: killed by that signal)\n" f"{proc.stderr[-3000:]}"
    )
    result = json.loads(proc.stdout.splitlines()[0])
    assert result is not None, proc.stderr[-3000:]
    return result, proc.stderr[-3000:]


def test_a_chain_is_dropped_on_the_main_thread() -> None:
    # Negations alone: the edges are what recursed, and a node that saves
    # nothing keeps the child near 0.5 GB at this length.
    result, err = _run("""
        x = lucid.randn(4, requires_grad=True)
        gc.collect()
        base = live_bytes()
        y = x
        for _ in range(300_000):
            y = -y
        loss = y.sum()
        del y, loss
        gc.collect()
        report({"left": live_bytes() - base})
    """)
    assert result == {"left": 0}, err


def test_a_dropped_chain_is_freed_whole_and_sooner_than_it_was_built() -> None:
    # Both are linear in the chain, and a release does far less per node than
    # the op that made it; on the best of three rounds, load moves both alike.
    result, err = _run("""
        def rounds():
            x = lucid.randn(4, requires_grad=True)
            gc.collect()
            base = live_bytes()
            out = []
            for _ in range(3):
                start = time.perf_counter()
                loss = chain(x, N).sum()
                built = time.perf_counter() - start
                held = live_bytes() - base
                start = time.perf_counter()
                del loss
                dropped = time.perf_counter() - start
                gc.collect()
                out.append({"built": built, "dropped": dropped, "held": held,
                            "left": live_bytes() - base})
            return out


        report(on_small_stack(rounds))
    """)
    assert isinstance(result, list) and len(result) == 3, err
    for r in result:
        # The chain held something to free, and freeing it left nothing.
        assert r["held"] > 0 and r["left"] == 0, result
    dropped = min(r["dropped"] for r in result)
    assert dropped < min(r["built"] for r in result), result


def test_a_chain_is_dropped_after_a_backward_that_retained_it(device: str) -> None:
    result, err = _run(f"""
        def retained():
            x = lucid.randn(4, requires_grad=True, device={device!r})
            loss = chain(x, N).sum()
            loss.backward(retain_graph=True)
            del loss
            return x.grad.tolist()


        report(on_small_stack(retained))
    """)
    assert result == [1.0, 1.0, 1.0, 1.0], err


def test_a_tree_of_chains_is_dropped_through_its_last_python_held_node() -> None:
    # One node with 1,001 edges: a chain over half the graph, and 1,000
    # branches over a trunk they all share, which only the last branch to go
    # frees.  The worklist grows wide as well as draining deep.  The last
    # owner is a grad_fn held from Python, whose holder frees it.
    result, err = _run("""
        def tree():
            x = lucid.randn(4, requires_grad=True)
            gc.collect()
            base = live_bytes()
            deep = chain(x, N // 2)
            trunk = chain(x, 1000)
            branches = [chain(trunk, 24) for _ in range(1000)]
            fn = lucid.stack([deep, *branches]).sum().grad_fn
            del deep, trunk, branches
            held = live_bytes() - base
            del fn
            gc.collect()
            return {"held": held, "left": live_bytes() - base}


        report(on_small_stack(tree))
    """)
    assert isinstance(result, dict), err
    assert result["held"] > 0 and result["left"] == 0, result


def test_a_chain_with_custom_functions_in_it_is_dropped() -> None:
    # A custom Function every 1,000 nodes: its node holds Python objects —
    # the ctx with the saved input and an attribute — that its own destructor
    # releases under the GIL.  Each ctx's marker dies with its node.
    result, err = _run("""
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


        def with_functions():
            x = lucid.randn(4, requires_grad=True)
            y = x
            for _ in range(N // 1000):
                y = Scale.apply(chain(y, 999))
            loss = y.sum()
            del y
            loss.backward(retain_graph=True)
            del loss
            return {"grad": x.grad.tolist(), "made": len(markers),
                    "alive": sum(m() is not None for m in markers)}


        report(on_small_stack(with_functions))
    """)
    assert result == {"grad": [1.0, 1.0, 1.0, 1.0], "made": 50, "alive": 0}, err


def test_a_graph_dropped_while_another_is_being_released_joins_that_release() -> None:
    # A ctx attribute whose __del__ builds and drops a long chain of its own:
    # it runs inside the release of the outer graph, whose worklist the inner
    # chain's nodes join rather than being released where they stand.
    result, err = _run("""
        seen = []


        class Dropper:
            def __del__(self):
                inner = chain(lucid.randn(4, requires_grad=True), N).sum()
                del inner
                seen.append(True)


        class Scale(lucid.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.dropper = Dropper()
                return x * 1.0

            @staticmethod
            def backward(ctx, grad):
                return grad


        def nested():
            x = lucid.randn(4, requires_grad=True)
            gc.collect()
            base = live_bytes()
            loss = chain(Scale.apply(chain(x, 1000)), 1000).sum()
            del loss
            ran = len(seen)
            gc.collect()
            return {"ran": ran, "left": live_bytes() - base}


        report(on_small_stack(nested))
    """)
    # __del__ ran within the outer release, and nothing of either is left.
    assert result == {"ran": 1, "left": 0}, err
