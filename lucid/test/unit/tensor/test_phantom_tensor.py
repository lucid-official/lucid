"""Phantom tensors: the engine's shape-only tensors (LCD-466).

While ``_C_engine.set_phantom_mode(True)`` is on for a thread, every factory
and op on that thread returns a real ``TensorImpl`` whose storage is a
phantom: the shape, dtype and device the op computes before it dispatches,
autograd state, and no values.  Nothing is allocated and no kernel runs.

The contract, as these tests hold it:

* each of 56 ops' phantom result has the shape, dtype and device of its eager
  one, views their strides, and the phantom section allocates nothing;
* a read of a phantom's values (``item``, ``tolist``, NumPy, bytes, ``bool``,
  printing) raises ``LucidError``;
* a phantom target takes any write; a real tensor never becomes or takes a
  phantom — an in-place result, a copy, a gradient (its first included), an
  assigned ``.grad``, a running statistic — and an optimizer step under the
  mode refuses before any state moves;
* outside the mode a phantom is inert: every op given one refuses;
* the mode belongs to the thread that set it.

The tests drive the engine switch directly, not ``nn.shadow_alloc``.
"""

import contextlib
import threading
from collections.abc import Callable, Iterator

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.autograd.function import Function, FunctionCtx
from lucid._tensor.tensor import Tensor
from lucid.test._fixtures.devices import metal_available


@contextlib.contextmanager
def phantom() -> Iterator[None]:
    previous = _C_engine.phantom_mode()
    _C_engine.set_phantom_mode(True)
    try:
        yield
    finally:
        _C_engine.set_phantom_mode(previous)


def _allocations() -> tuple[int, int]:
    return (
        _C_engine.memory_stats(_C_engine.CPU).alloc_count,
        _C_engine.memory_stats(_C_engine.GPU).alloc_count,
    )


def _tensors(out: object) -> list[Tensor]:
    if isinstance(out, Tensor):
        return [out]
    assert isinstance(out, (tuple, list)), type(out)
    return [t for item in out for t in _tensors(item)]


# Each case builds its inputs and its result on ``d``, so the inputs are
# phantoms too when the case runs under the mode.
_CASES: dict[str, Callable[[str], object]] = {
    # factories
    "zeros": lambda d: lucid.zeros(2, 3, device=d),
    "ones": lambda d: lucid.ones(4, device=d, dtype=lucid.float16),
    "empty": lambda d: lucid.empty(3, 1, device=d),
    "full": lambda d: lucid.full((2, 2), 7.0, device=d),
    "eye": lambda d: lucid.eye(3, 5, device=d),
    "arange": lambda d: lucid.arange(0, 10, 3, device=d),
    "linspace": lambda d: lucid.linspace(0.0, 1.0, 7, device=d),
    "logspace": lambda d: lucid.logspace(0.0, 2.0, 5, device=d),
    "rand": lambda d: lucid.rand(3, 4, device=d),
    "randn": lambda d: lucid.randn(2, 5, device=d),
    "randint": lambda d: lucid.randint(0, 9, (6,), device=d),
    "zeros_like": lambda d: lucid.zeros_like(lucid.ones(2, 7, device=d)),
    "full_like": lambda d: lucid.full_like(lucid.ones(3, device=d), 2.0),
    "tensor": lambda d: lucid.tensor([[1, 2], [3, 4]], device=d),
    # elementwise, with broadcasting and promotion
    "add_broadcast": lambda d: lucid.ones(2, 1, 4, device=d) + lucid.ones(3, 1, device=d),
    "sub_scalar": lambda d: lucid.ones(3, device=d) - 1,
    "mul_promote": lambda d: lucid.ones(3, device=d) * lucid.arange(3, device=d),
    "div": lambda d: lucid.ones(2, 3, device=d) / lucid.full((3,), 2.0, device=d),
    "pow": lambda d: lucid.ones(4, device=d) ** 2,
    "exp": lambda d: lucid.exp(lucid.ones(2, 2, device=d)),
    "log": lambda d: lucid.log(lucid.ones(5, device=d)),
    "sqrt": lambda d: lucid.sqrt(lucid.ones(3, 3, device=d)),
    "tanh": lambda d: lucid.tanh(lucid.ones(4, device=d)),
    "sigmoid": lambda d: lucid.sigmoid(lucid.ones(4, device=d)),
    "relu": lambda d: F.relu(lucid.randn(3, 2, device=d)),
    "gelu": lambda d: F.gelu(lucid.randn(3, 2, device=d)),
    "maximum": lambda d: lucid.maximum(lucid.ones(3, device=d), lucid.zeros(2, 3, device=d)),
    "compare": lambda d: lucid.ones(3, device=d) > lucid.zeros(3, device=d),
    "astype": lambda d: lucid.ones(2, 2, device=d).to(lucid.float16),
    # reductions
    "sum_dim": lambda d: lucid.ones(2, 3, 4, device=d).sum(dim=1),
    "mean_keepdim": lambda d: lucid.ones(2, 3, device=d).mean(dim=0, keepdim=True),
    "sum_all": lambda d: lucid.ones(5, 6, device=d).sum(),
    "amax": lambda d: lucid.amax(lucid.ones(3, 4, device=d), dim=1),
    "argmax": lambda d: lucid.argmax(lucid.ones(3, 4, device=d), dim=0),
    "var": lambda d: lucid.var(lucid.ones(4, 5, device=d), dim=1),
    # shape and layout
    "reshape": lambda d: lucid.ones(2, 6, device=d).reshape(3, -1),
    "transpose": lambda d: lucid.ones(2, 3, device=d).transpose(0, 1),
    "permute": lambda d: lucid.ones(2, 3, 4, device=d).permute(2, 0, 1),
    "unsqueeze": lambda d: lucid.ones(3, device=d).unsqueeze(0),
    "flatten": lambda d: lucid.ones(2, 3, 4, device=d).flatten(1),
    "expand": lambda d: lucid.ones(3, 1, device=d).expand(3, 4),
    "cat": lambda d: lucid.cat([lucid.ones(2, 3, device=d), lucid.ones(4, 3, device=d)]),
    "stack": lambda d: lucid.stack([lucid.ones(3, device=d), lucid.ones(3, device=d)], dim=1),
    "split": lambda d: lucid.ones(7, 2, device=d).split(3),
    "chunk": lambda d: lucid.ones(6, 4, device=d).chunk(3, dim=1),
    "narrow": lambda d: lucid.ones(5, 4, device=d).narrow(0, 1, 3),
    "flip": lambda d: lucid.flip(lucid.ones(2, 3, device=d), dims=[1]),
    "tril": lambda d: lucid.tril(lucid.ones(4, 4, device=d)),
    "pad": lambda d: F.pad(lucid.ones(2, 3, device=d), (1, 2)),
    # linear algebra and nn
    "matmul": lambda d: lucid.ones(2, 3, 4, device=d) @ lucid.ones(4, 5, device=d),
    "softmax": lambda d: F.softmax(lucid.ones(2, 5, device=d), dim=-1),
    "log_softmax": lambda d: F.log_softmax(lucid.ones(2, 5, device=d), dim=1),
    "linear": lambda d: F.linear(lucid.ones(3, 4, device=d), lucid.ones(6, 4, device=d)),
    "conv2d": lambda d: F.conv2d(
        lucid.ones(1, 3, 8, 8, device=d), lucid.ones(4, 3, 3, 3, device=d), padding=1
    ),
    "max_pool2d": lambda d: F.max_pool2d(lucid.ones(1, 2, 8, 8, device=d), 2),
    "layer_norm": lambda d: F.layer_norm(lucid.ones(2, 6, device=d), (6,)),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_a_phantom_has_the_shape_dtype_and_device_of_the_eager_result(case, device):
    build = _CASES[case]
    eager = _tensors(build(device))
    before = _allocations()
    with phantom():
        ghosts = _tensors(build(device))
    assert _allocations() == before, "the phantom section allocated"
    assert len(ghosts) == len(eager)
    for ghost, real in zip(ghosts, eager, strict=True):
        assert ghost._impl.is_phantom
        assert not real._impl.is_phantom
        assert tuple(ghost.shape) == tuple(real.shape)
        assert ghost.stride() == real.stride()
        assert ghost.dtype == real.dtype
        assert ghost.device == real.device


def test_a_phantom_keeps_requires_grad_and_records_a_graph(device):
    with phantom():
        w = lucid.randn(4, 3, device=device, requires_grad=True)
        y = (lucid.ones(2, 4, device=device) @ w).sum()
    assert w._impl.is_phantom and y._impl.is_phantom
    assert w.requires_grad and w.is_leaf
    assert y.requires_grad and y.grad_fn is not None


def test_a_module_built_under_the_mode_holds_phantom_parameters(device):
    before = _allocations()
    with phantom():
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(device)
        out = model(lucid.ones(5, 16, device=device))
    assert _allocations() == before
    params = list(model.parameters())
    assert [tuple(p.shape) for p in params] == [(32, 16), (32,), (4, 32), (4,)]
    assert all(p._impl.is_phantom for p in params)
    assert tuple(out.shape) == (5, 4) and out._impl.is_phantom


# Each read, and the name its refusal gives: the read itself, on either
# device, not the generic refusal of an op outside the mode.
_READS: dict[str, tuple[Callable[[Tensor], object], str]] = {
    "item": (lambda t: t.item(), "item"),
    "tolist": (lambda t: t.tolist(), "to_bytes"),
    "bool": (lambda t: bool(t), "item"),
    "float": (lambda t: float(t), "item"),
    "to_string": (lambda t: t._impl.to_string(), "to_string"),
    "to_bytes": (lambda t: t._impl.to_bytes(), "to_bytes"),
    "data_ptr": (lambda t: t.data_ptr(), "data_ptr"),
}


@pytest.mark.parametrize("read", list(_READS))
def test_reading_a_phantom_s_values_raises(read, device):
    fn, name = _READS[read]
    with phantom():
        t = lucid.ones(1, device=device)
    with pytest.raises(_C_engine.LucidError, match=f"phantom tensor has no values: {name}"):
        fn(t)


def test_evaluating_a_phantom_is_a_no_op(device):
    with phantom():
        t = lucid.ones(2, device=device)
    _C_engine.eval_gpu(t._impl)
    _C_engine.eval_tensors([t._impl])
    _C_engine.eval_tensors_async([t._impl])
    t._impl.eval()
    assert t._impl.is_phantom


def test_reading_a_phantom_as_numpy_raises(device):
    pytest.importorskip("numpy")
    with phantom():
        t = lucid.ones(2, 2, device=device)
    with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
        t.numpy()


def test_a_phantom_prints_without_values():
    # The engine refuses to render the values; the printer says so.
    with phantom():
        t = lucid.ones(2)
    assert "unavailable" in repr(t)


def test_even_an_empty_phantom_has_no_values():
    with phantom():
        t = lucid.zeros(0)
    with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
        t.tolist()


def test_the_mode_belongs_to_the_thread_that_set_it():
    seen: dict[str, object] = {}

    def other() -> None:
        seen["mode"] = _C_engine.phantom_mode()
        seen["values"] = lucid.arange(3).tolist()

    with phantom():
        worker = threading.Thread(target=other)
        worker.start()
        worker.join()
        assert lucid.zeros(3)._impl.is_phantom
    assert seen == {"mode": False, "values": [0, 1, 2]}


def test_a_thread_in_the_mode_leaves_the_others_real():
    ready, release = threading.Event(), threading.Event()
    seen: dict[str, bool] = {}

    def other() -> None:
        with phantom():
            seen["phantom"] = lucid.zeros(2)._impl.is_phantom
            ready.set()
            release.wait(timeout=10)

    worker = threading.Thread(target=other)
    worker.start()
    assert ready.wait(timeout=10)
    try:
        assert lucid.ones(2).tolist() == [1.0, 1.0]
        assert not _C_engine.phantom_mode()
    finally:
        release.set()
        worker.join()
    assert seen == {"phantom": True}


def test_an_engine_error_inside_the_mode_leaves_it_as_it_was():
    with phantom():
        with pytest.raises(_C_engine.LucidError):
            lucid.ones(2).item()
        with pytest.raises(ValueError):
            lucid.ones(2, 3) + lucid.ones(4, 5)
        assert _C_engine.phantom_mode()
    assert not _C_engine.phantom_mode()


def test_the_mode_is_restored_when_the_body_raises():
    with pytest.raises(KeyError), phantom():
        lucid.zeros(3)
        raise KeyError("body")
    assert not _C_engine.phantom_mode()
    assert lucid.zeros(2).tolist() == [0.0, 0.0]


def test_a_real_target_refuses_a_phantom_s_values(device):
    real = lucid.zeros(3, device=device)
    with phantom():
        ghost = lucid.ones(3, device=device)
        with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
            real.add_(1)
    with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
        real.copy_(ghost)
    assert not real._impl.is_phantom
    assert real.tolist() == [0.0, 0.0, 0.0]


def test_a_phantom_target_takes_any_write(device):
    with phantom():
        ghost = lucid.zeros(3, device=device)
        ghost.add_(1)
        ghost.mul_(lucid.ones(3, device=device))
        ghost.zero_()
    real = lucid.ones(3, device=device)
    version = ghost._impl.version
    ghost.copy_(real)
    ghost.fill_(2.0)
    assert ghost._impl.is_phantom
    assert ghost._impl.version > version


# Outside the mode a phantom is inert: every op given one refuses where it
# reads the values or would make a new phantom, kernel bases and views alike,
# with the one typed message.
_OUTSIDE: dict[str, Callable[[Tensor, Tensor], object]] = {
    "binary": lambda g, r: g + r,
    "unary": lambda g, r: lucid.exp(g),
    "reduce": lambda g, r: g.sum(dim=0),
    "cast": lambda g, r: g.to(lucid.float16),
    "contiguous": lambda g, r: g.contiguous(),
    "copy": lambda g, r: g.clone(),
    "move": lambda g, r: g.to("cpu", copy=True),
    "transpose": lambda g, r: g.transpose(0, 1),
    "reshape": lambda g, r: g.reshape(-1),
    "slice": lambda g, r: g[0],
    "iterate": lambda g, r: list(g),
    "detach": lambda g, r: g.detach(),
    "matmul": lambda g, r: g @ r.reshape(3, 1),
    "backward": lambda g, r: g.backward(),
}


@pytest.mark.parametrize("op", list(_OUTSIDE))
def test_outside_the_mode_an_op_given_a_phantom_refuses(op, device):
    with phantom():
        ghost = lucid.ones(2, 3, device=device)
        if op == "backward":
            # The root of a graph built under the mode: Engine::backward refuses it.
            ghost = (lucid.ones(2, 3, device=device, requires_grad=True) * 2).sum()
    real = lucid.ones(3, device=device)
    with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
        _OUTSIDE[op](ghost, real)
    assert not _C_engine.phantom_mode()


def test_moving_a_phantom_relabels_it_under_the_mode(device):
    with phantom():
        ghost = lucid.ones(2, 2)
        target = _C_engine.GPU if device == "metal" else _C_engine.CPU
        moved = ghost._impl.transfer_to_device(target, True)
    assert moved.is_phantom and moved.shape == [2, 2]
    assert moved.device == target and moved.requires_grad
    with pytest.raises(_C_engine.LucidError, match="outside phantom mode"):
        ghost._impl.transfer_to_device(target, False)


def test_moving_a_real_tensor_under_the_mode_copies_nothing(device):
    real = lucid.ones(4)
    before = _allocations()
    with phantom():
        moved = real.to(device, copy=True)
    assert _allocations() == before
    assert moved._impl.is_phantom and moved.device == lucid.device(device)
    assert not real._impl.is_phantom and real.tolist() == [1.0] * 4


def test_a_real_tensor_is_not_a_phantom():
    assert not lucid.zeros(2)._impl.is_phantom
    assert not _C_engine.phantom_mode()


# ── a real tensor never becomes a phantom ────────────────────────────────────


@pytest.mark.parametrize("loss", ["scalar", "tensor", "unary"])
def test_a_real_leaf_refuses_a_phantom_gradient(loss, device):
    w = lucid.ones(3, device=device, requires_grad=True)
    with phantom():
        with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
            if loss == "scalar":
                (w * 2).sum().backward()
            elif loss == "tensor":
                (w * lucid.ones(3, device=device)).sum().backward()
            else:
                w.exp().sum().backward()
    assert w.grad is None


def test_a_real_gradient_refuses_a_phantom_one_added(device):
    w = lucid.ones(3, device=device, requires_grad=True)
    (w * 2).sum().backward()
    with phantom():
        with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
            (w * 2).sum().backward()
    assert not w.grad._impl.is_phantom
    assert w.grad.tolist() == [2.0, 2.0, 2.0]


def test_a_phantom_leaf_gets_a_phantom_gradient(device):
    with phantom():
        w = lucid.ones(3, device=device, requires_grad=True)
        (w * 2).sum().backward()
    assert w.grad is not None and w.grad._impl.is_phantom
    assert tuple(w.grad.shape) == (3,)


def test_assigning_a_phantom_gradient_to_a_real_tensor_refuses(device):
    w = lucid.ones(3, device=device, requires_grad=True)
    with phantom():
        ghost = lucid.ones(3, device=device)
        with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
            w.grad = ghost
    with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
        w.grad = ghost
    assert w.grad is None


def test_a_real_running_statistic_refuses_a_phantom_update(device):
    bn = nn.BatchNorm1d(3).to(device)
    with phantom():
        x = lucid.ones(4, 3, device=device)
        with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
            bn(x)
    assert not bn.running_mean._impl.is_phantom
    assert not bn.running_var._impl.is_phantom
    assert bn.running_mean.tolist() == [0.0, 0.0, 0.0]


def test_a_phantom_running_statistic_takes_the_update(device):
    with phantom():
        bn = nn.BatchNorm1d(3).to(device)
        out = bn(lucid.ones(4, 3, device=device))
    assert bn.running_mean._impl.is_phantom and out._impl.is_phantom


@pytest.mark.parametrize("optimizer", ["SGD", "Adam", "SparseAdam"])
def test_an_optimizer_step_under_the_mode_refuses_before_any_state_moves(optimizer, device):
    def make() -> tuple[nn.Parameter, optim.Optimizer]:
        w = nn.Parameter(lucid.ones(3, device=device))
        w.grad = lucid.ones(3, device=device)
        return w, getattr(optim, optimizer)([w], lr=0.1)

    w_ref, opt_ref = make()
    opt_ref.step()
    opt_ref.step()
    w, opt = make()
    with phantom():
        with pytest.raises(_C_engine.LucidError, match="optimizer step under phantom mode"):
            opt.step()
    opt.step()
    with phantom():
        with pytest.raises(_C_engine.LucidError, match="optimizer step under phantom mode"):
            opt.step()
    opt.step()
    assert not w._impl.is_phantom
    assert w.tolist() == pytest.approx(w_ref.tolist())


def test_an_optimizer_step_of_a_phantom_parameter_refuses(device):
    with phantom():
        w = nn.Parameter(lucid.ones(3, device=device))
        w.grad = lucid.ones(3, device=device)
    opt = optim.SGD([w], lr=0.1)
    with pytest.raises(_C_engine.LucidError, match="optimizer step of a phantom parameter"):
        opt.step()


@pytest.mark.parametrize(
    ("name", "kwargs", "grouped"),
    [
        ("SGD", {}, False),
        ("Adam", {}, False),
        ("RMSprop", {}, False),
        ("SGD", {}, True),
        ("Adam", {}, True),
        ("SGD", {"maximize": True}, False),
        ("RAdam", {"decoupled_weight_decay": True, "weight_decay": 0.1}, False),
        ("NAdam", {"decoupled_weight_decay": True, "weight_decay": 0.1}, False),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_an_engine_optimizer_step_over_a_phantom_parameter_moves_no_real_one(
    name, kwargs, grouped, device
):
    """The engine refuses a phantom parameter itself; a step that writes first
    (a group before it, a decoupled decay, ``maximize``) must still refuse
    before any real parameter moves, with the engine's own message."""
    real = nn.Parameter(lucid.ones(3, device=device))
    real.grad = lucid.ones(3, device=device)
    with phantom():
        bad = nn.Parameter(lucid.ones(3, device=device))
        bad_grad = lucid.ones(3, device=device)
    bad.grad = bad_grad
    params = [{"params": [real]}, {"params": [bad]}] if grouped else [real, bad]
    opt = getattr(optim, name)(params, lr=0.1, **kwargs)
    with pytest.raises(_C_engine.LucidError, match="optimizer step of a phantom parameter"):
        opt.step()
    assert real.tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "rebind",
    [
        "half",
        pytest.param("to", marks=pytest.mark.skipif(not metal_available(), reason="needs Metal")),
        "data",
    ],
)
def test_a_real_parameter_is_not_rebound_to_a_phantom(rebind):
    m = nn.Linear(3, 2)
    with phantom():
        with pytest.raises(_C_engine.LucidError, match="phantom tensor has no values"):
            if rebind == "half":
                m.half()
            elif rebind == "to":
                m.to("metal")
            else:
                m.weight.data = lucid.zeros(2, 3)
    assert not m.weight._impl.is_phantom


# ── a phantom gradient keeps its device ──────────────────────────────────────
#
# Autograd hands a gradient around as a bare Storage and wraps it as a tensor
# for a hook, a custom Function, a seed or a create_graph rebuild, reading the
# device off the storage (storage_device).  A phantom records the device it
# stands in for, so each path keeps the eager device.


class _Ident(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tensor:
        return x * 1

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> Tensor:
        _SEEN["function"] = grad.device
        return grad


_SEEN: dict[str, object] = {}


def _hook_sees(device: str) -> Tensor:
    w = lucid.ones(3, device=device, requires_grad=True)
    y = w * 2
    y.register_hook(lambda g: _SEEN.__setitem__("hook", g.device))
    y.sum().backward()
    return w


def _hook_returns(device: str) -> Tensor:
    w = lucid.ones(3, device=device, requires_grad=True)
    y = w * 2
    y.register_hook(lambda g: lucid.zeros(3, device=device))
    y.sum().backward()
    return w


def _function(device: str) -> Tensor:
    w = lucid.ones(3, device=device, requires_grad=True)
    _Ident.apply(w).sum().backward()
    return w


def _snapshot(device: str) -> Tensor:
    w = lucid.ones(4, device=device, requires_grad=True)
    y = w * 1
    y.index_add_(0, lucid.zeros(1, dtype=lucid.int64, device=device), lucid.ones(1, device=device))
    y.sum().backward()
    return w


def _create_graph(device: str) -> Tensor:
    w = lucid.ones(3, device=device, requires_grad=True)
    (g,) = lucid.autograd.grad((w * w).sum(), w, create_graph=True)
    g.sum().backward()
    return w


def _seeded(device: str) -> Tensor:
    w = lucid.ones(3, device=device, requires_grad=True)
    (w * 2).backward(lucid.ones(3, device=device))
    return w


_GRAD_PATHS: dict[str, Callable[[str], Tensor]] = {
    "hook sees": _hook_sees,
    "hook returns": _hook_returns,
    "function": _function,
    "snapshot": _snapshot,
    "create_graph": _create_graph,
    "seeded": _seeded,
}


@pytest.mark.parametrize("path", list(_GRAD_PATHS))
def test_a_phantom_gradient_keeps_its_device(path, device):
    _SEEN.clear()
    eager = _GRAD_PATHS[path](device)
    seen_eager = dict(_SEEN)
    _SEEN.clear()
    with phantom():
        ghost = _GRAD_PATHS[path](device)
    assert ghost.grad is not None and ghost.grad._impl.is_phantom
    assert ghost.grad.device == eager.grad.device == lucid.device(device)
    assert _SEEN == seen_eager


# ── phantom views follow their device's layout rule ──────────────────────────

_VIEWS: dict[str, Callable[[str], Tensor]] = {
    "slice": lambda d: lucid.ones(4, 6, device=d)[1:3, 2:5],
    "transpose": lambda d: lucid.ones(4, 6, device=d).T,
    "permute": lambda d: lucid.ones(4, 6, device=d).reshape(2, 2, 6).permute(2, 0, 1),
    "narrow": lambda d: lucid.ones(4, 6, device=d).narrow(1, 1, 3),
    "expand": lambda d: lucid.ones(3, 1, device=d).expand(3, 4),
    "diagonal": lambda d: lucid.diagonal(lucid.ones(4, 4, device=d)),
    "unfold": lambda d: lucid.ones(8, device=d).unfold(0, 3, 2),
    "split": lambda d: lucid.ones(4, 6, device=d).split(2, dim=1)[1],
}


@pytest.mark.parametrize("view", list(_VIEWS))
def test_a_phantom_view_has_the_eager_strides(view, device):
    eager = _VIEWS[view](device)
    with phantom():
        ghost = _VIEWS[view](device)
    assert ghost._impl.is_phantom
    assert (ghost.stride(), ghost.is_contiguous()) == (eager.stride(), eager.is_contiguous())


def test_a_view_refused_eagerly_is_refused_as_a_phantom(device):
    with pytest.raises(_C_engine.LucidError) as eager:
        lucid.ones(4, 6, device=device).T.view(-1)
    with phantom():
        with pytest.raises(_C_engine.LucidError) as ghost:
            lucid.ones(4, 6, device=device).T.view(-1)
    assert type(ghost.value) is type(eager.value)
