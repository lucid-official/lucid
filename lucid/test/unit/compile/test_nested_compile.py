"""A compiled callable invoked from inside another trace is transparent (LCD-626).

``inner = lucid.compile(fn)`` called from inside ``outer = lucid.compile(lambda
t: inner(t) * 3)``'s own trace used to run ``inner``'s cached executable
directly.  That tensor entered ``outer``'s trace as a feed no recorded op
produced, which ``outer`` pins at its trace-time value: every later call to
``outer`` answered with the *first* call's result, silently, however
different the input.  The symptom was even worse on a cold inner cache —
``inner``'s own ``_tracing()`` replaced ``outer``'s active Tracer and then
unconditionally detached it to ``None`` on exit, dropping ``outer``'s
recording for every op traced after the nested call returned.

A ``CompiledModule`` now checks whether a trace is already recording
(``lucid.compile._core.trace_context.recording_active``) before touching its
own cache or tracer, and if so runs its wrapped model inline — its ops
dispatch straight into the already-installed (outer) tracer, exactly as if
the caller had written them out instead of nesting a second
``lucid.compile``. The other ``_tracing()``-based entry points
(``make_step``, ``fused_step``, ``compiled_step``, ``compile_optimizer``,
decode/segment-step tracing) have no inline-eager fallback of their own —
nesting one inside an active trace now raises instead of silently clobbering
it.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile._entry.step import make_step
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")


def _t(*shape: int) -> lucid.Tensor:
    return lucid.randn(*shape, device=COMPILE_DEVICE)


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8).to(COMPILE_DEVICE)
        self.b = nn.Linear(8, 4).to(COMPILE_DEVICE)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.b(self.a(x).relu())


class _Parent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(8, 8).to(COMPILE_DEVICE)
        self.body = _Net()

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.body(self.stem(x))


@pytest.mark.parametrize("warm_inner", [True, False])
def test_a_compiled_callable_nested_in_another_matches_eager(warm_inner: bool) -> None:
    """``outer(inner(t))`` tracks eager for several distinct inputs — warm or cold."""
    plain = lambda t: t * 2 + 1  # the traced function under test
    inner = lucid.compile(plain)
    outer = lucid.compile(lambda t: inner(t) * 3)

    x1, x2, x3 = _t(3), _t(3), _t(3)
    with lucid.no_grad():
        if warm_inner:
            inner(x1)
            inner(x1)
            assert inner.cache_info()["entries"] == 1

        for x in (x1, x2, x3):
            got = outer(x)
            expected = plain(x) * 3
            np.testing.assert_allclose(got.numpy(), expected.numpy(), atol=1e-5)

    # The nested call never ran its own cache entry — it is always inlined.
    assert inner.cache_info()["entries"] == (1 if warm_inner else 0)
    # outer compiled exactly once (one signature seen).
    assert outer.cache_info()["entries"] == 1


def test_the_outer_trace_contains_the_inner_ops() -> None:
    """The outer graph records the inner's ops, not just its own."""
    import json

    inner = lucid.compile(lambda t: t * 2 + 1)
    outer = lucid.compile(lambda t: inner(t) * 3)
    with lucid.no_grad():
        outer(_t(3))

    dump = json.loads(outer.graph_dump())
    ops = {op["name"] for entry in dump for op in entry["graph"]["ops"]}
    # inner's `add`, outer's `mul` by 3 — both from a single recording.
    assert "add" in ops
    assert "mul" in ops


def test_a_compiled_module_child_of_a_compiled_parent_matches_eager() -> None:
    """A ``CompiledModule`` assigned as a submodule, called by a compiled parent."""
    lucid.manual_seed(0)
    parent = _Parent().eval()
    parent.body = lucid.compile(parent.body)
    compiled_parent = lucid.compile(parent)

    eager = _Parent().eval()
    for (_, p1), (_, p2) in zip(parent.named_parameters(), eager.named_parameters(), strict=True):
        p2.data = p1.data.clone()

    with lucid.no_grad():
        for _ in range(2):
            x = _t(2, 8)
            got = compiled_parent(x)
            want = eager(x)
            np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-5)

    assert compiled_parent.cache_info()["entries"] == 1


def test_gradients_through_a_nested_compiled_call_match_eager() -> None:
    """``make_step`` tracing a model whose forward calls a nested compiled child."""
    lucid.manual_seed(0)
    parent = _Parent()
    parent.body = lucid.compile(parent.body)

    eager = _Parent()
    for (_, p1), (_, p2) in zip(parent.named_parameters(), eager.named_parameters(), strict=True):
        p2.data = p1.data.clone()

    x = _t(4, 8)
    target = lucid.randint(0, 4, (4,), device=COMPILE_DEVICE)

    step_fn = make_step(parent, lambda y: F.cross_entropy(y, target))
    loss = step_fn(x)
    loss.backward()

    out = eager(x)
    loss_eager = F.cross_entropy(out, target)
    loss_eager.backward()

    np.testing.assert_allclose(loss.numpy(), loss_eager.numpy(), atol=1e-4)
    for (n1, p1), (n2, p2) in zip(parent.named_parameters(), eager.named_parameters(), strict=True):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None
        np.testing.assert_allclose(p1.grad.numpy(), p2.grad.numpy(), atol=1e-4)


def test_nesting_make_step_inside_an_active_trace_is_refused_loudly() -> None:
    """``make_step`` / ``fused_step`` / ``compiled_step`` have no inline-eager
    escape of their own: reopening ``_tracing()`` while one is active must
    raise, not silently replace (and then drop) the outer Tracer."""
    model = nn.Linear(8, 4).to(COMPILE_DEVICE)
    target = lucid.randint(0, 4, (2,), device=COMPILE_DEVICE)
    step_fn = make_step(model, lambda y: F.cross_entropy(y, target))
    x = _t(2, 8)

    with pytest.raises(RuntimeError, match="already recording"):
        with lucid.compile._tracing():
            step_fn(x)
