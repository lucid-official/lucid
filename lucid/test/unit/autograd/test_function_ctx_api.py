"""The ``Function`` / ``ctx`` surface the reference defines and Lucid lacked (LCD-741a).

Six pieces, each with one owner in ``lucid/autograd``:

* ``ctx.mark_dirty`` — an input ``forward`` wrote in place comes back as itself,
  its history the new node (``function.py``).
* ``ctx.set_materialize_grads(False)`` — an output no gradient reached arrives
  at ``backward`` as ``None`` instead of zeros (``function.py`` →
  ``_python_node.py``).
* ``setup_context`` — ``forward`` written without a ``ctx`` (``function.py``).
* ``once_differentiable`` — ``backward`` run without recording, an error when
  its gradients are backpropagated (``function.py``).
* a ``forward`` that returns non-tensors — ``backward`` has one argument per
  result, ``None`` for those (``_python_node.py``).
* ``saved_tensors_hooks`` and its subclass ``save_on_cpu`` (``graph.py``).

Every case is a scenario written once against a ``lib`` (Lucid or the
reference) and compared against the reference under ``pytest -m parity``; the
cases that follow only from Lucid's own contract are plain tests.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.autograd import Function
from lucid.autograd.graph import saved_tensors_hooks, save_on_cpu

Scenario = Callable[[ModuleType, str], object]


def _vec(lib: ModuleType, device: str, *values: float, grad: bool = True):
    return lib.tensor(list(values), device=device, requires_grad=grad)


def _kind(g) -> str:
    return "None" if g is None else f"tensor{[round(v, 4) for v in g.tolist()]}"


def _raises(run: Callable[[], object]) -> object:
    try:
        return run()
    except Exception as exc:  # noqa: BLE001 — the scenario compares exception *types*
        return type(exc).__name__


# ── mark_dirty ────────────────────────────────────────────────────────────────


def _dirty_class(lib: ModuleType, *, save: bool = False, scale: float | None = None):
    class Dirty(lib.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            if scale is None:
                x.add_(1)
            else:
                x.mul_(scale)
            ctx.mark_dirty(x)
            if save:
                ctx.save_for_backward(x)
            return x

        @staticmethod
        def backward(ctx, grad):
            if save:
                ctx.saved_tensors
            return grad if scale is None else grad * scale

    return Dirty


def _dirty_is_returned(lib: ModuleType, device: str) -> object:
    a = _vec(lib, device, 1.0, 2.0, 3.0)
    b = a * 1
    before = b._version
    out = _dirty_class(lib).apply(b)
    out.sum().backward()
    return out is b, out.grad_fn is not None, out._version > before, _kind(a.grad), out.tolist()


def _dirty_and_saved(lib: ModuleType, device: str) -> object:
    a = _vec(lib, device, 1.0, 2.0)
    b = a * 3
    out = _dirty_class(lib, save=True, scale=2.0).apply(b)
    (out * out).sum().backward()
    return out is b, _kind(a.grad)


def _two_dirty(lib: ModuleType, device: str) -> object:
    class Both(lib.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            x.add_(1)
            y.add_(1)
            ctx.mark_dirty(x, y)
            return x, y

        @staticmethod
        def backward(ctx, gx, gy):
            return gx, gy * 2

    a = _vec(lib, device, 1.0, 2.0)
    p, q = a * 1, a * 2
    o1, o2 = Both.apply(p, q)
    (o1 + o2).sum().backward()
    return o1 is p, o2 is q, _kind(a.grad)


def _dirty_leaf_that_requires_grad(lib: ModuleType, device: str) -> object:
    return _raises(lambda: _dirty_class(lib).apply(_vec(lib, device, 1.0, 2.0)))


def _dirty_leaf_under_no_grad(lib: ModuleType, device: str) -> object:
    a = _vec(lib, device, 1.0, 2.0)
    with lib.no_grad():
        _dirty_class(lib).apply(a)
    return a.tolist(), a.requires_grad, a.is_leaf


def _dirty_untracked(lib: ModuleType, device: str) -> object:
    x = _vec(lib, device, 1.0, 2.0, grad=False)
    out = _dirty_class(lib).apply(x)
    return out is x, out.requires_grad, x.tolist()


def _dirty_not_returned(lib: ModuleType, device: str) -> object:
    class Lost(lib.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x.add_(1)
            ctx.mark_dirty(x)
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            return grad

    a = _vec(lib, device, 1.0)
    return (
        _raises(lambda: Lost.apply(a * 1)),
        _raises(lambda: Lost.apply(_vec(lib, device, 1.0, grad=False))),
    )


def _dirty_marked_in_setup_context(lib: ModuleType, device: str) -> object:
    class Setup(lib.autograd.Function):
        @staticmethod
        def forward(x):
            x.add_(1)
            return x

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.mark_dirty(inputs[0])

        @staticmethod
        def backward(ctx, grad):
            return grad

    a = _vec(lib, device, 1.0, 2.0)
    b = a * 1
    out = Setup.apply(b)
    out.sum().backward()
    return out is b, _kind(a.grad)


# ── set_materialize_grads ─────────────────────────────────────────────────────


def _materializing(lib: ModuleType, seen: list[str], *, materialize: bool, mask: bool = False):
    class Split(lib.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.set_materialize_grads(materialize)
            if mask:
                m = x > 0
                ctx.mark_non_differentiable(m)
                return x * 2, m, x * 3
            return x * 2, x * 3

        @staticmethod
        def backward(ctx, *grads):
            seen.extend(_kind(g) for g in grads)
            total = 0
            for g, s in zip(grads, (2.0, 0.0, 3.0) if mask else (2.0, 3.0), strict=True):
                if g is not None and s:
                    total = total + g * s
            return total if not isinstance(total, int) else None

    return Split


def _run_split(lib, device, use, **kw):
    seen: list[str] = []
    a = _vec(lib, device, 1.0, 2.0)
    outs = _materializing(lib, seen, **kw).apply(a)
    use(outs).sum().backward()
    return seen, _kind(a.grad)


def _materialize_false_first_only(lib: ModuleType, device: str) -> object:
    return _run_split(lib, device, lambda o: o[0], materialize=False)


def _materialize_false_second_only(lib: ModuleType, device: str) -> object:
    return _run_split(lib, device, lambda o: o[1], materialize=False)


def _materialize_false_both(lib: ModuleType, device: str) -> object:
    return _run_split(lib, device, lambda o: o[0] + o[1], materialize=False)


def _materialize_true_second_only(lib: ModuleType, device: str) -> object:
    return _run_split(lib, device, lambda o: o[1], materialize=True)


def _materialize_false_with_a_non_differentiable_output(lib: ModuleType, device: str) -> object:
    return _run_split(lib, device, lambda o: o[2], materialize=False, mask=True)


def _materialize_true_with_a_non_differentiable_output(lib: ModuleType, device: str) -> object:
    return _run_split(lib, device, lambda o: o[2], materialize=True, mask=True)


def _materialize_set_in_setup_context(lib: ModuleType, device: str) -> object:
    seen: list[str] = []

    class Split(lib.autograd.Function):
        @staticmethod
        def forward(x):
            return x * 2, x * 3

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.set_materialize_grads(False)

        @staticmethod
        def backward(ctx, g1, g2):
            seen.extend((_kind(g1), _kind(g2)))
            return g2 * 3

    a = _vec(lib, device, 1.0, 2.0)
    Split.apply(a)[1].sum().backward()
    return seen, _kind(a.grad)


# ── setup_context ─────────────────────────────────────────────────────────────


def _square(lib: ModuleType):
    class Square(lib.autograd.Function):
        @staticmethod
        def forward(x):
            return x * x

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(inputs[0])

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return 2 * x * grad

    return Square


def _setup_context_gradient(lib: ModuleType, device: str) -> object:
    a = _vec(lib, device, 1.0, 2.0, 3.0)
    _square(lib).apply(a).sum().backward()
    return _kind(a.grad)


def _setup_context_inputs(lib: ModuleType, device: str) -> object:
    seen: list[object] = []

    class Scale(lib.autograd.Function):
        @staticmethod
        def forward(x, k=3, label="s"):
            return x * k

        @staticmethod
        def setup_context(ctx, inputs, output):
            seen.append((len(inputs), inputs[1], inputs[2], ctx.needs_input_grad))
            ctx.k = inputs[1]

        @staticmethod
        def backward(ctx, grad):
            return grad * ctx.k, None, None

    a = _vec(lib, device, 1.0, 2.0)
    for call in (lambda: Scale.apply(a), lambda: Scale.apply(a, 5), lambda: Scale.apply(a, k=4)):
        call().sum().backward()
    return seen, _kind(a.grad)


def _setup_context_marks_non_differentiable(lib: ModuleType, device: str) -> object:
    class Pair(lib.autograd.Function):
        @staticmethod
        def forward(x):
            return x * 2, x > 0

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.mark_non_differentiable(output[1])

        @staticmethod
        def backward(ctx, g, gm):
            return g * 2

    a = _vec(lib, device, 1.0, -2.0)
    out, mask = Pair.apply(a)
    out.sum().backward()
    return mask.requires_grad, mask.tolist(), _kind(a.grad)


def _setup_context_missing_argument(lib: ModuleType, device: str) -> object:
    return _raises(lambda: _square(lib).apply())


def _setup_context_attributes_reach_backward(lib: ModuleType, device: str) -> object:
    class Shape(lib.autograd.Function):
        @staticmethod
        def forward(x):
            return x.sum()

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.shape = tuple(inputs[0].shape)

        @staticmethod
        def backward(ctx, grad):
            return grad.expand(*ctx.shape) if hasattr(grad, "expand") else grad

    a = lib.ones(2, 3, device=device, requires_grad=True)
    Shape.apply(a).backward()
    return _kind(a.grad.reshape(-1))


# ── once_differentiable ───────────────────────────────────────────────────────


def _once(lib: ModuleType, mode: list[object]):
    class Square(lib.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        @lib.autograd.function.once_differentiable
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            mode.append(lib.is_grad_enabled())
            return 2 * x * grad

    return Square


def _once_first_order(lib: ModuleType, device: str) -> object:
    mode: list[object] = []
    a = _vec(lib, device, 1.0, 2.0)
    _once(lib, mode).apply(a).sum().backward()
    return _kind(a.grad), mode


def _once_double_backward_raises(lib: ModuleType, device: str) -> object:
    mode: list[object] = []
    a = _vec(lib, device, 1.0, 2.0)
    b = _vec(lib, device, 3.0, 4.0)
    y = _once(lib, mode).apply(a)
    (g,) = lib.autograd.grad((y * b).sum(), a, create_graph=True)
    return g.requires_grad, _kind(g), _raises(lambda: g.sum().backward())


def _once_create_graph_with_constant_seed(lib: ModuleType, device: str) -> object:
    # Nothing the backward was given needs a gradient, so there is nothing to
    # differentiate twice and no error is armed.
    mode: list[object] = []
    a = _vec(lib, device, 1.0, 2.0)
    y = _once(lib, mode).apply(a)
    (g,) = lib.autograd.grad(y.sum(), a, create_graph=True)
    return g.requires_grad, _kind(g)


def _once_with_two_inputs(lib: ModuleType, device: str) -> object:
    class Mul(lib.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            return x * y

        @staticmethod
        @lib.autograd.function.once_differentiable
        def backward(ctx, grad):
            x, y = ctx.saved_tensors
            return grad * y, grad * x

    a, b = _vec(lib, device, 1.0, 2.0), _vec(lib, device, 3.0, 4.0)
    w = _vec(lib, device, 1.0, 1.0)
    (ga, gb) = lib.autograd.grad((Mul.apply(a, b) * w).sum(), (a, b), create_graph=True)
    return _kind(ga), _kind(gb), _raises(lambda: (ga + gb).sum().backward())


# ── non-tensor outputs ────────────────────────────────────────────────────────


def _results_with_non_tensors(lib: ModuleType, device: str) -> object:
    seen: list[str] = []

    class Tagged(lib.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return 7, x * 2, "tag", x * 3, None

        @staticmethod
        def backward(ctx, *grads):
            seen.extend(_kind(g) for g in grads)
            return grads[1] * 2 + grads[3] * 3

    a = _vec(lib, device, 1.0, 2.0)
    n, p, tag, q, none = Tagged.apply(a)
    (p + q).sum().backward()
    return (n, tag, none), seen, _kind(a.grad)


def _one_tensor_and_a_count(lib: ModuleType, device: str) -> object:
    seen: list[str] = []

    class Counted(lib.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            return a * 2, 5

        @staticmethod
        def backward(ctx, g, h):
            seen.extend((_kind(g), _kind(h)))
            return g * 2

    a = _vec(lib, device, 1.0, 2.0)
    out, count = Counted.apply(a)
    out.sum().backward()
    return count, seen, _kind(a.grad)


def _non_tensor_first_with_setup_context(lib: ModuleType, device: str) -> object:
    seen: list[str] = []

    class Counted(lib.autograd.Function):
        @staticmethod
        def forward(a):
            return 3, a * 2

        @staticmethod
        def setup_context(ctx, inputs, output):
            pass

        @staticmethod
        def backward(ctx, h, g):
            seen.extend((_kind(h), _kind(g)))
            return g * 2

    a = _vec(lib, device, 1.0, 2.0)
    count, out = Counted.apply(a)
    out.sum().backward()
    return count, seen, _kind(a.grad)


_PARITY: dict[str, Scenario] = {
    "dirty is returned": _dirty_is_returned,
    "dirty and saved": _dirty_and_saved,
    "two dirty": _two_dirty,
    "dirty leaf that requires grad": _dirty_leaf_that_requires_grad,
    "dirty leaf under no_grad": _dirty_leaf_under_no_grad,
    "dirty untracked": _dirty_untracked,
    "dirty not returned": _dirty_not_returned,
    "dirty marked in setup_context": _dirty_marked_in_setup_context,
    "materialize False, first only": _materialize_false_first_only,
    "materialize False, second only": _materialize_false_second_only,
    "materialize False, both": _materialize_false_both,
    "materialize True, second only": _materialize_true_second_only,
    "materialize False, non-differentiable": _materialize_false_with_a_non_differentiable_output,
    "materialize True, non-differentiable": _materialize_true_with_a_non_differentiable_output,
    "materialize set in setup_context": _materialize_set_in_setup_context,
    "setup_context gradient": _setup_context_gradient,
    "setup_context inputs": _setup_context_inputs,
    "setup_context non-differentiable": _setup_context_marks_non_differentiable,
    "setup_context missing argument": _setup_context_missing_argument,
    "setup_context attributes": _setup_context_attributes_reach_backward,
    "once_differentiable first order": _once_first_order,
    "once_differentiable double backward": _once_double_backward_raises,
    "once_differentiable constant seed": _once_create_graph_with_constant_seed,
    "once_differentiable two inputs": _once_with_two_inputs,
    "non-tensor results": _results_with_non_tensors,
    "one tensor and a count": _one_tensor_and_a_count,
    "non-tensor first, setup_context": _non_tensor_first_with_setup_context,
}


@pytest.mark.parametrize("case", sorted(_PARITY))
def test_scenario_runs_and_is_deterministic(case: str, device: str) -> None:
    assert _PARITY[case](lucid, device) == _PARITY[case](lucid, device)


@pytest.mark.parity
@pytest.mark.parametrize("case", sorted(_PARITY))
def test_scenario_matches_the_reference(ref: ModuleType, case: str, device: str) -> None:
    # What a scenario observes does not depend on where a tensor lives, so the
    # reference runs on its CPU against each Lucid device.
    assert _PARITY[case](lucid, device) == _PARITY[case](ref, "cpu")


# ── what the scenarios leave to a plain assertion ─────────────────────────────


def test_mark_dirty_refuses_a_value_that_is_not_a_tensor() -> None:
    class Bad(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.mark_dirty(5)
            return x * 1

        @staticmethod
        def backward(ctx, grad):
            return grad

    with pytest.raises(TypeError, match="mark_dirty can only accept tensors"):
        Bad.apply(lucid.ones(2, requires_grad=True))


def test_set_materialize_grads_refuses_a_value_that_is_not_a_bool() -> None:
    class Bad(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.set_materialize_grads("no")
            return x * 1

        @staticmethod
        def backward(ctx, grad):
            return grad

    with pytest.raises(TypeError, match="expects a bool"):
        Bad.apply(lucid.ones(2, requires_grad=True))


def test_a_dirty_input_keeps_one_history_through_a_second_call(device: str) -> None:
    a = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    b = a * 1
    c = _dirty_class(lucid).apply(_dirty_class(lucid).apply(b))
    assert c is b
    c.sum().backward()
    assert a.grad is not None and a.grad.tolist() == [1.0, 1.0]
    assert b.tolist() == [3.0, 4.0]


def test_setup_context_form_is_chosen_by_overriding_it(device: str) -> None:
    class WithCtx(Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2

        @staticmethod
        def backward(ctx, grad):
            return grad * 2

    a = lucid.ones(2, device=device, requires_grad=True)
    WithCtx.apply(a).sum().backward()
    assert a.grad is not None and a.grad.tolist() == [2.0, 2.0]


def test_a_function_with_no_forward_says_so() -> None:
    class Bare(Function):
        @staticmethod
        def backward(ctx, grad):
            return grad

    with pytest.raises(NotImplementedError):
        Bare.apply(lucid.ones(2, requires_grad=True))


def test_setup_context_form_composes_with_func_transforms(device: str) -> None:
    square = _square(lucid)
    x = lucid.tensor([1.0, 2.0, 3.0], device=device)
    assert lucid.func.grad(lambda t: square.apply(t).sum())(x).tolist() == [2.0, 4.0, 6.0]
    jac = lucid.func.jacrev(lambda t: square.apply(t))(x)
    assert jac.tolist() == [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]]
    batched = lucid.func.vmap(lambda t: square.apply(t))(lucid.ones(2, 3, device=device) * 2)
    assert batched.tolist() == [[4.0] * 3] * 2
    per_row = lucid.func.vmap(lucid.func.grad(lambda t: square.apply(t).sum()))(
        lucid.ones(2, 3, device=device) * 2
    )
    assert per_row.tolist() == [[4.0] * 3] * 2


def test_once_differentiable_message_names_the_decorator(device: str) -> None:
    mode: list[object] = []
    a = lucid.tensor([1.0], device=device, requires_grad=True)
    b = lucid.tensor([2.0], device=device, requires_grad=True)
    (g,) = lucid.autograd.grad((_once(lucid, mode).apply(a) * b).sum(), a, create_graph=True)
    with pytest.raises(RuntimeError, match="@once_differentiable"):
        g.sum().backward()


# ── saved_tensors_hooks ───────────────────────────────────────────────────────


def _saver(lib: ModuleType):
    class Saver(lib.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x, x * 2)
            return x * 3

        @staticmethod
        def backward(ctx, grad):
            a, b = ctx.saved_tensors
            return grad * a * b

    return Saver


def _tagging(lib: ModuleType, log: list[str], tag: str):
    def pack(t):
        log.append(f"pack {tag}")
        return (tag, t)

    def unpack(packed):
        log.append(f"unpack {packed[0]}")
        return packed[1]

    return lib.autograd.graph.saved_tensors_hooks(pack, unpack)


def _hooks_pack_then_unpack(lib: ModuleType, device: str) -> object:
    log: list[str] = []
    x = _vec(lib, device, 1.0, 2.0)
    with _tagging(lib, log, "h"):
        out = _saver(lib).apply(x)
    packed = list(log)
    out.sum().backward()
    return packed, log, _kind(x.grad)


def _hooks_nest(lib: ModuleType, device: str) -> object:
    log: list[str] = []
    x = _vec(lib, device, 1.0, 2.0)
    with _tagging(lib, log, "outer"):
        with _tagging(lib, log, "inner"):
            inner = _saver(lib).apply(x)
        outer = _saver(lib).apply(x)
    packed = list(log)
    (inner + outer).sum().backward()
    return packed, sorted(log[len(packed) :]), _kind(x.grad)


def _hooks_after_the_block(lib: ModuleType, device: str) -> object:
    log: list[str] = []
    x = _vec(lib, device, 1.0, 2.0)
    with _tagging(lib, log, "h"):
        pass
    _saver(lib).apply(x).sum().backward()
    return log


def _hooks_not_run_without_a_graph(lib: ModuleType, device: str) -> object:
    log: list[str] = []
    x = _vec(lib, device, 1.0, 2.0, grad=False)
    with _tagging(lib, log, "h"):
        _saver(lib).apply(x)
        with lib.no_grad():
            _saver(lib).apply(_vec(lib, device, 1.0, grad=True))
    return log


def _hooks_unpack_on_every_read(lib: ModuleType, device: str) -> object:
    log: list[str] = []
    x = _vec(lib, device, 1.0, 2.0)
    with _tagging(lib, log, "h"):
        out = _saver(lib).apply(x)
    out.sum().backward(retain_graph=True)
    out.sum().backward()
    return log, _kind(x.grad)


def _hooks_unpack_result_must_be_a_tensor(lib: ModuleType, device: str) -> object:
    x = _vec(lib, device, 1.0, 2.0)
    with lib.autograd.graph.saved_tensors_hooks(lambda t: t, lambda p: 5):
        out = _saver(lib).apply(x)
    return _raises(lambda: out.sum().backward())


def _hooks_see_the_saved_tensor(lib: ModuleType, device: str) -> object:
    seen: list[object] = []

    def pack(t):
        seen.append(t.tolist())
        return t

    x = _vec(lib, device, 1.0, 2.0)
    with lib.autograd.graph.saved_tensors_hooks(pack, lambda p: p):
        _saver(lib).apply(x)
    return seen


_HOOK_SCENARIOS: dict[str, Scenario] = {
    "pack then unpack": _hooks_pack_then_unpack,
    "nesting": _hooks_nest,
    "after the block": _hooks_after_the_block,
    "no graph": _hooks_not_run_without_a_graph,
    "unpack on every read": _hooks_unpack_on_every_read,
    "unpack result must be a tensor": _hooks_unpack_result_must_be_a_tensor,
    "pack sees the tensor": _hooks_see_the_saved_tensor,
}


@pytest.mark.parametrize("case", sorted(_HOOK_SCENARIOS))
def test_hook_scenario_is_deterministic(case: str, device: str) -> None:
    assert _HOOK_SCENARIOS[case](lucid, device) == _HOOK_SCENARIOS[case](lucid, device)


@pytest.mark.parity
@pytest.mark.parametrize("case", sorted(_HOOK_SCENARIOS))
def test_hook_scenario_matches_the_reference(ref: ModuleType, case: str, device: str) -> None:
    assert _HOOK_SCENARIOS[case](lucid, device) == _HOOK_SCENARIOS[case](ref, "cpu")


def test_the_innermost_hooks_pack_and_their_own_unpack_reads(device: str) -> None:
    log: list[str] = []
    x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    with _tagging(lucid, log, "outer"):
        with _tagging(lucid, log, "inner"):
            inner = _saver(lucid).apply(x)
        outer = _saver(lucid).apply(x)
    assert log == ["pack inner", "pack inner", "pack outer", "pack outer"]
    log.clear()
    (inner + outer).sum().backward()
    assert sorted(log) == ["unpack inner", "unpack inner", "unpack outer", "unpack outer"]


def test_hooks_are_per_thread() -> None:
    import threading

    log: list[str] = []
    other: list[list[str]] = []

    def run() -> None:
        mine: list[str] = []
        # Entered on the main thread, not this one, so it sees no hooks.
        x = lucid.tensor([1.0], requires_grad=True)
        _saver(lucid).apply(x)
        other.append(mine)

    with _tagging(lucid, log, "main"):
        t = threading.Thread(target=run)
        t.start()
        t.join()
    assert log == [] and other == [[]]


def test_the_hooks_are_left_when_the_block_raises() -> None:
    hooks = saved_tensors_hooks(lambda t: t, lambda p: p)
    with pytest.raises(ValueError, match="boom"):
        with hooks:
            raise ValueError("boom")
    x = lucid.tensor([1.0], requires_grad=True)
    log: list[str] = []
    h = _tagging(lucid, log, "later")
    with h:
        _saver(lucid).apply(x)
    assert log == ["pack later", "pack later"]


def test_a_tensor_saved_through_a_hook_is_not_version_checked(device: str) -> None:
    class Save(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * 2

        @staticmethod
        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * x

    x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    w = x * 1
    with saved_tensors_hooks(lambda t: t * 1, lambda p: p):
        y = Save.apply(w)
    w.add_(1)  # the hook kept a copy, so the write cannot reach what backward reads
    y.sum().backward()
    assert x.grad is not None and x.grad.tolist() == [1.0, 2.0]


def test_save_on_cpu_is_a_saved_tensors_hooks(device: str) -> None:
    assert issubclass(save_on_cpu, saved_tensors_hooks)
    assert save_on_cpu(pin_memory=True).pin_memory is True


def test_save_on_cpu_holds_a_host_copy_and_restores_the_device(device: str) -> None:
    held: list[str] = []
    back: list[str] = []
    ctx_hooks = save_on_cpu()
    pack, unpack = ctx_hooks.pack_hook, ctx_hooks.unpack_hook

    def spy_pack(t):
        packed = pack(t)
        held.append(packed.host.device.type)
        return packed

    def spy_unpack(p):
        t = unpack(p)
        back.append(t.device.type)
        return t

    ctx_hooks.pack_hook, ctx_hooks.unpack_hook = spy_pack, spy_unpack
    x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    with ctx_hooks:
        out = _saver(lucid).apply(x)
    out.sum().backward()
    assert held == ["cpu", "cpu"]
    assert back == [x.device.type] * 2
    assert x.grad is not None and x.grad.tolist() == [2.0, 8.0]


@pytest.mark.parity
def test_save_on_cpu_gradient_matches_the_reference(ref: ModuleType, device: str) -> None:
    def run(lib: ModuleType, dev: str) -> object:
        x = _vec(lib, dev, 1.0, 2.0)
        with lib.autograd.graph.save_on_cpu():
            out = _saver(lib).apply(x)
        out.sum().backward()
        return _kind(x.grad)

    assert run(lucid, device) == run(ref, "cpu")


def test_the_built_in_ops_are_not_routed_through_the_hooks(device: str) -> None:
    # Documented coverage limit: the engine keeps a built-in op's saved
    # tensors itself, so only a custom ``Function`` reaches the hooks.
    log: list[str] = []
    x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    with _tagging(lucid, log, "h"):
        y = (x * x).exp()
    y.sum().backward()
    assert log == []
