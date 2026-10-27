"""A compiled step holds every value at its declared dtype and keeps every
index inside its buffer.

The compiled graph is a second implementation of the eager ops, and two of
the eager contracts it has to follow were broken in it:

- **Integer width (LCD-270).** An emitter whose MPSGraph primitive picks its
  own result type handed the next op a value of another width: ``argmax``
  came out int32 where Lucid types it int64. Comparing it with an int64
  constant then aborted the process inside MPSGraph ("not broadcast
  compatible", rc 134) — no exception, no fallback. The builder now holds
  every emitted value at the dtype the trace declared, so a binary op sees
  the dtypes eager's kernel saw (eager inserts the ``astype`` itself).
- **Index bounds (LCD-278).** The gather, embedding and scatter emitters
  handed the caller's indices to MPSGraph unchecked, and the graph is built
  from the first call's indices: a later call's out-of-range index read or
  wrote outside the buffer. They now follow the eager Metal policy B
  (``backend/gpu/AxisIndex.h``): a gather answers NaN (0 for an integer),
  a scatter drops the update, one_hot gives a zero row.

Each program runs in a subprocess, because the failure being guarded
against kills the interpreter.

``2**40`` is used only where eager keeps the index at int64. Where the
Python layer narrows the index to int32 first (``x[i]``, ``scatter``,
``scatter_add``: LCD-273), eager reads ``2**40`` as 0 while the graph,
whose optimiser may assume the narrowing cast does not overflow, drops it;
the two agree once LCD-273 removes the narrowing.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="Metal unavailable")


_PRELUDE = textwrap.dedent("""
    import json, math, warnings
    import lucid, lucid.nn as nn, lucid.nn.functional as F
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap, _wrap
    from lucid.compile import fused_step
    from lucid.compile._core.fallback import CompileFallbackWarning

    # A fallback would make every comparison below pass trivially.
    warnings.simplefilter("error", CompileFallbackWarning)

    def flat(v):
        return [x for row in v for x in flat(row)] if isinstance(v, list) else [v]

    def same(a, b, tol):
        a, b = flat(a), flat(b)
        return len(a) == len(b) and all(
            (math.isnan(x) and math.isnan(y)) or abs(x - y) <= tol * (1 + abs(y))
            for x, y in zip(a, b))

    def emit(case, ok, **detail):
        print(json.dumps({"case": case, "ok": bool(ok), **detail}), flush=True)

    def eager_step(model, loss_fn, x, y):
        opt = lucid.optim.SGD(model.parameters(), lr=0.1)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        return loss
""")


_INTEGER_PROGRAM = _PRELUDE + textwrap.dedent("""
    import itertools

    def operands(y):
        return {
            "int64": y.argmax(dim=1),
            "int32": y.argmin(dim=1).to(lucid.int32),
            "bool": y.sum(dim=1) > 0,
        }

    OPS = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "lt": lambda a, b: a < b,
        "ge": lambda a, b: a >= b,
        "add": lambda a, b: a + b,
        "mul": lambda a, b: a * b,
    }

    def loss_for(ka, kb):
        def loss(p, y):
            # Two sources, so a same-width pair is two tensors: MPSGraph turns
            # an int64 ``x * x`` into a square it has no kernel for, and the
            # mul emitter declines that case.
            a, b = operands(y)[ka], operands(-y)[kb]
            total = ((y.argmax(dim=1) != -100).to(p.dtype) * p.sum(dim=1)).sum()
            for k, f in enumerate(OPS.values()):
                r = f(a, b).to(p.dtype)
                total = total + (r * p.sum(dim=1)).sum() * float(k + 1)
            return total
        return loss

    lucid.manual_seed(0)
    x = lucid.randn(4, 8).to("metal")
    y = lucid.randn(4, 4).to("metal")
    for ka, kb in itertools.product(["int64", "int32", "bool"], repeat=2):
        loss_fn = loss_for(ka, kb)
        compiled = nn.Linear(8, 4).to("metal")
        eager = nn.Linear(8, 4).to("metal")
        eager.load_state_dict(compiled.state_dict())
        step = fused_step(compiled, loss_fn, lucid.optim.SGD(compiled.parameters(), lr=0.1))
        got = step(x, y).item()
        want = eager_step(eager, loss_fn, x, y).item()
        ok = same(got, want, 1e-5) and all(
            same(p.to("cpu").tolist(), q.to("cpu").tolist(), 1e-5)
            for p, q in zip(compiled.parameters(), eager.parameters(), strict=True))
        emit(f"{ka}-{kb}", ok, compiled=got, eager=want)
""")


_INDEX_FORWARD_PROGRAM = _PRELUDE + textwrap.dedent("""
    N = 5
    x = (lucid.arange(20, dtype=lucid.float32).reshape(4, N) / 7.0).to("metal")
    table = lucid.arange(15, dtype=lucid.float32).reshape(N, 3).to("metal")
    ones = lucid.ones(4, 2).to("metal")

    def ix(v):
        return lucid.tensor(v, dtype=lucid.int64).to("metal")

    def engine(fn, *args):
        return _wrap(fn(*[_unwrap(a) if isinstance(a, lucid.Tensor) else a for a in args]))

    PAIRS = [[0, 1], [2, 3], [4, 0], [1, 2]]
    BAD_PAIRS = [[0, N], [-N - 1, 3], [4, -1], [1, 2]]
    CASES = {
        "gather": (lambda i: lucid.gather(x, 1, i), PAIRS,
                   [[0, N], [-N - 1, 3], [4, 2**40], [-1, 2]]),
        "index_select": (lambda i: lucid.index_select(x, 1, i), [0, 1, 2], [N, -N - 1, 2**40]),
        "take": (lambda i: lucid.take(x, i), [0, 1, 2], [20, -21, 2**40]),
        "getitem": (lambda i: x[i], [0, 1, 2], [4, -5, 3]),
        "embedding": (lambda i: engine(_C_engine.nn.embedding, table, i, -1), [0, 1, 2],
                      [N, -1, 2**40]),
        "one_hot": (lambda i: engine(_C_engine.nn.one_hot, i, N), [0, 1, 2], [N, -1, 2**40]),
        "scatter_add": (lambda i: lucid.scatter_add(x, 1, i, ones), PAIRS, BAD_PAIRS),
        "scatter": (lambda i: lucid.scatter(x, 1, i, -ones), PAIRS, BAD_PAIRS),
        "cross_entropy": (lambda t: F.cross_entropy(x, t, reduction="none"),
                          [0, 1, 2, -100], [N, -1, 3, -100]),
        "nll_loss": (lambda t: F.nll_loss(x, t, reduction="none"),
                     [0, 1, 2, -100], [N, -1, 3, -100]),
    }
    for name, (fn, good, bad) in CASES.items():
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile.compile(lambda i, fn=fn: fn(i) * 1)
        compiled(ix(good))
        got = compiled(ix(bad)).to("cpu").tolist()
        want = (fn(ix(bad)) * 1).to("cpu").tolist()
        built = _C_engine.compile.session_cache_size() > 0
        emit(name, built and same(got, want, 1e-6), compiled=got, eager=want)
""")


_INDEX_BACKWARD_PROGRAM = _PRELUDE + textwrap.dedent("""
    N = 5

    class Lookup(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.w = nn.Parameter(lucid.arange(15, dtype=lucid.float32).reshape(N, 3) / 10.0)
            self.s = nn.Parameter(lucid.arange(6, dtype=lucid.float32).reshape(3, 2) / 10.0)

        def forward(self, i):
            if self.kind == "embedding":
                return _wrap(_C_engine.nn.embedding(_unwrap(self.w), _unwrap(i), -1)) + self.s.sum()
            if self.kind == "gather":
                return lucid.gather(self.w.T, 1, i) + self.s.sum()
            if self.kind == "index_select":
                return lucid.index_select(self.w, 0, i) + self.s.sum()
            if self.kind == "scatter_add":
                return lucid.scatter_add(self.w.T * 1.0, 1, i, self.s)
            return lucid.scatter(self.w.T * 1.0, 1, i, self.s)

    def loss_fn(out, y):
        return (out * y).sum()

    rows = (lucid.tensor([0, 1, 2]), lucid.tensor([N, -N - 1, 4]))
    pairs = (lucid.tensor([[0, 1], [2, 3], [4, 0]]), lucid.tensor([[0, N], [-N - 1, 3], [4, -2]]))
    for kind in ("embedding", "gather", "index_select", "scatter_add", "scatter"):
        good, bad = rows if kind in ("embedding", "index_select") else pairs
        good, bad = good.to("metal"), bad.to("metal")
        width = {"embedding": 3, "index_select": 3, "gather": 2}.get(kind, N)
        y = lucid.ones(3, width).to("metal")
        compiled = Lookup(kind).to("metal")
        step = fused_step(compiled, loss_fn, lucid.optim.SGD(compiled.parameters(), lr=0.1))
        step(good, y)
        eager = Lookup(kind).to("metal")
        eager.load_state_dict(compiled.state_dict())
        got = step(bad, y).item()
        want = eager_step(eager, loss_fn, bad, y).item()
        ok = same(got, want, 1e-5) and all(
            same(p.to("cpu").tolist(), q.to("cpu").tolist(), 1e-5)
            for p, q in zip(compiled.parameters(), eager.parameters(), strict=True))
        emit(kind, ok, compiled=[p.to("cpu").tolist() for p in compiled.parameters()],
             eager=[q.to("cpu").tolist() for q in eager.parameters()])
""")


def _run(program: str) -> dict[str, dict[str, object]]:
    done = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "LUCID_COMPILE_VERBOSE"},
        timeout=600,
    )
    assert (
        done.returncode == 0
    ), f"the program died with rc {done.returncode}:\n{done.stderr[-3000:]}"
    records = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
    return {str(r["case"]): r for r in records}


def _assert_all_ok(records: dict[str, dict[str, object]], expected: list[str]) -> None:
    assert sorted(records) == sorted(expected)
    wrong = {name: r for name, r in records.items() if not r["ok"]}
    assert not wrong, json.dumps(wrong, indent=1)


class TestIntegerWidth:
    def test_mixed_integer_operands_compile_and_match_eager(self) -> None:
        """int64 / int32 / bool operands, each pair under every comparison
        and arithmetic op, inside one fused step — the LCD-270 repro
        (``argmax(...) != -100``) included in each."""
        widths = ["int64", "int32", "bool"]
        _assert_all_ok(_run(_INTEGER_PROGRAM), [f"{a}-{b}" for a in widths for b in widths])


class TestIndexBounds:
    def test_compiled_index_ops_answer_policy_b(self) -> None:
        _assert_all_ok(
            _run(_INDEX_FORWARD_PROGRAM),
            [
                "gather",
                "index_select",
                "take",
                "getitem",
                "embedding",
                "one_hot",
                "scatter_add",
                "scatter",
                "cross_entropy",
                "nll_loss",
            ],
        )

    def test_fused_step_gradients_drop_out_of_range_rows(self) -> None:
        _assert_all_ok(
            _run(_INDEX_BACKWARD_PROGRAM),
            ["embedding", "gather", "index_select", "scatter_add", "scatter"],
        )
