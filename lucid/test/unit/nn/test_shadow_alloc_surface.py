"""``shadow_alloc`` answers the same on the CPU and on Metal, across the op surface.

The defect these guard against is a class, not an op: a path that reads a
phantom's storage instead of answering with the shape it already computed.
Where a path exists on only one device, the other device's tests stay green.
``where`` read its Metal result's shape off an MLX array the phantom backend's
answer does not have (LCD-850), and with ``inner`` and a gradient seed that failed
163 public symbols on Metal alone: ``clamp``, ``lgamma``, ``cross_entropy``, every
``Embedding`` with a ``padding_idx``.

So instead of a list of ops, the sweep asks the audit for a working call of every
public op and runs it four ways, eager and phantom on each device.  The rules:

* no answer is the engine's generic "storage is not the kind this op reads": a
  phantom's values are refused by name (``refuse_phantom_read``) or not read;
* Metal answers whenever the CPU does and a real Metal tensor would, and the
  other way round but for :data:`_CPU_READS_VALUES`;
* what real Metal refuses for its dtype, the phantom refuses too: a shadow-built
  check is otherwise blind to float64 on Metal;
* the two phantoms agree on shape and dtype, and a phantom agrees with the eager
  result on its own device on shape, dtype and device.

The sweep runs in a subprocess: it calls every public op, and what that leaves
behind in the process should not reach the tests after it.
"""

import contextlib
import dataclasses
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import lucid
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.nn._shadow import is_shadow, shadow_alloc
from lucid.nn.quantized.mlx_linear import QuantizedLinearMLX
from lucid.test._fixtures.devices import metal_available
from lucid.test.audit import _probe, _specs, _surface

needs_metal = pytest.mark.skipif(not metal_available(), reason="compares the CPU with Metal")

#: Ops whose CPU path reads the tensor on the host where the Metal one does not, so
#: only the CPU refuses a phantom.  By design: ``binary_cross_entropy`` and
#: ``gaussian_nll_loss`` check values on the CPU alone, because a host read in every
#: loss call stalls every step (the docstring of ``_check_unit_interval``, in
#: ``lucid/nn/functional/_loss_checks.py``).  ``is_set_to`` compares buffer addresses
#: on the CPU and view families on Metal.
_CPU_READS_VALUES: dict[str, frozenset[str]] = {
    "forward": frozenset({"Tensor.is_set_to", "F.binary_cross_entropy", "F.gaussian_nll_loss"}),
    "backward": frozenset({"F.binary_cross_entropy", "F.gaussian_nll_loss"}),
    "second": frozenset({"F.binary_cross_entropy", "F.gaussian_nll_loss"}),
}

#: The fewest ops each question must have reached.  A sweep that stopped finding
#: working calls would otherwise pass by looking at nothing.
_FLOOR = {"forward": 1000, "backward": 500, "second": 700}

#: Seconds a sweep may take: one process per question, on the slowest machine.
_SWEEP_TIMEOUT = 600

_READS_A_STORAGE = "not the kind this op reads"

_METAL_DTYPE_REFUSAL = "is not supported on GPU"


@dataclasses.dataclass
class _Findings:
    """How many a sweep reached, and what each rule found among them."""

    swept: int = 0
    reads_a_storage: list[str] = dataclasses.field(default_factory=list)
    metal_refuses: list[str] = dataclasses.field(default_factory=list)
    blind_to_dtype: list[str] = dataclasses.field(default_factory=list)
    cpu_refuses: list[str] = dataclasses.field(default_factory=list)
    phantoms_differ: list[str] = dataclasses.field(default_factory=list)
    eager_differs: list[str] = dataclasses.field(default_factory=list)
    builds_differ: list[str] = dataclasses.field(default_factory=list)
    real_tensors: list[str] = dataclasses.field(default_factory=list)


def _narrowed(dtype: lucid.dtype) -> lucid.dtype:
    """The call's dtype on both devices: Metal holds no double precision."""
    if dtype == lucid.float64:
        return lucid.float32
    return lucid.complex64 if dtype == lucid.complex128 else dtype


def _rebuilt(value: object, make: Callable[[Tensor], Tensor]) -> object:
    if isinstance(value, Tensor):
        return make(value)
    if isinstance(value, list):
        return [_rebuilt(item, make) for item in value]
    if type(value) is tuple:
        return tuple(_rebuilt(item, make) for item in value)
    if isinstance(value, dict):
        return {name: _rebuilt(item, make) for name, item in value.items()}
    return value


def _real_on(device: str) -> Callable[[Tensor], Tensor]:
    def make(t: Tensor) -> Tensor:
        out = t.detach().to(_narrowed(t.dtype)).to(device, copy=True)
        return out.requires_grad_() if t.requires_grad else out

    return make


def _phantom_on(device: str) -> Callable[[Tensor], Tensor]:
    def make(t: Tensor) -> Tensor:
        out = lucid.zeros(tuple(t.shape), dtype=_narrowed(t.dtype), device=device)
        return out.requires_grad_() if t.requires_grad else out

    return make


def _tensors_in(out: object, depth: int = 0) -> Iterator[Tensor]:
    if isinstance(out, Tensor):
        yield out
    elif depth < 4 and isinstance(out, (list, tuple)):
        for item in out:
            yield from _tensors_in(item, depth + 1)
    elif depth < 4 and isinstance(out, dict):
        for item in out.values():
            yield from _tensors_in(item, depth + 1)


def _described(out: object) -> list[object]:
    return [[list(t.shape), str(t.dtype), t.device.type] for t in _tensors_in(out)]


def _gradient(x: Tensor, out: object) -> object:
    """``d out / d x``, or ``None`` for an output with no graph."""
    first = next((t for t in _tensors_in(out) if t.requires_grad), None)
    if first is None:
        return None
    (first.abs() if first.is_complex() else first).sum().backward()
    return x.grad


def _second_gradient(x: Tensor, out: object) -> object:
    """Both orders of ``d out / d x``, or ``None`` where the first has no graph."""
    first = next((t for t in _tensors_in(out) if t.requires_grad), None)
    if first is None:
        return None
    # Squared: under a plain sum a piecewise-linear op's gradient has no graph, so
    # the sweep would skip it, though its graph-mode backward has run.
    real = first.abs() if first.is_complex() else first
    grad = lucid.autograd.grad((real * real).sum(), x, create_graph=True, allow_unused=True)[0]
    if grad is None or not grad.requires_grad:
        return None
    return [grad, lucid.autograd.grad(grad.sum(), x, allow_unused=True)[0]]


_DIFFERENTIATE = {"backward": _gradient, "second": _second_gradient}


def _ask(
    fn: Callable[..., object],
    call: _specs.Call,
    device: str,
    shadow: bool,
    question: str,
) -> dict[str, object]:
    """One way of running ``call``: what it answered, or what it refused."""
    make = _phantom_on(device) if shadow else _real_on(device)
    block = shadow_alloc() if shadow else contextlib.nullcontext()
    try:
        with block, _probe.preserved_globals():
            lucid.set_default_device(device)
            args = [_rebuilt(a, make) for a in call.args]
            kwargs = {k: _rebuilt(v, make) for k, v in call.kwargs.items()}
            if question == "forward":
                return {"answer": _described(fn(*args, **kwargs))}
            x = args[call.primary]
            if not isinstance(x, Tensor) or not x.is_floating_point():
                return {"answer": None}
            args[call.primary] = x = x.detach().requires_grad_()
            grad = _DIFFERENTIATE[question](x, fn(*args, **kwargs))
            return {"answer": None if grad is None else _described(grad)}
    except Exception as exc:  # the sweep records every refusal, by message
        return {"refused": f"{type(exc).__name__}: {str(exc)[:200]}"}


def _working_call(
    fn: Callable[..., object], symbol: _surface.Symbol, question: str
) -> _specs.Call | None:
    """The first call of ``symbol`` the eager CPU answers ``question`` for."""
    for call in _specs.invocations(symbol.short, "moderate", symbol.qualname, fn):
        if question != "forward" and not 0 <= call.primary < len(call.args):
            continue
        if _ask(fn, call, "cpu", False, question).get("answer") is not None:
            return call
    return None


def _sweep_ops(question: str) -> _Findings:
    found = _Findings()
    for symbol in _surface.enumerate_surface():
        fn = _surface.resolve(symbol)
        if symbol.kind not in ("op", "method") or not symbol.inert or not callable(fn):
            continue
        if question != "forward" and symbol.short.endswith("_"):
            continue
        call = _working_call(fn, symbol, question)
        if call is None:
            continue
        found.swept += 1
        name = symbol.qualname
        on_cpu = _ask(fn, call, "cpu", True, question)
        on_metal = _ask(fn, call, "metal", True, question)
        real_metal = _ask(fn, call, "metal", False, question)
        if any(_READS_A_STORAGE in str(a.get("refused")) for a in (on_cpu, on_metal)):
            found.reads_a_storage.append(name)
        if _METAL_DTYPE_REFUSAL in str(real_metal.get("refused")) and "answer" in on_metal:
            found.blind_to_dtype.append(name)
        if "refused" in on_metal or "refused" in on_cpu:
            if "refused" not in on_cpu and "answer" in real_metal:
                found.metal_refuses.append(f"{name}: {on_metal['refused']}")
            elif "refused" not in on_metal:
                found.cpu_refuses.append(name)
            continue
        cpu_shapes = [t[:2] for t in on_cpu["answer"] or []]
        metal_shapes = [t[:2] for t in on_metal["answer"] or []]
        if cpu_shapes != metal_shapes:
            found.phantoms_differ.append(f"{name}: {cpu_shapes} on the CPU, {metal_shapes}")
        # A second gradient is not compared with the eager one: Metal's eager
        # frexp records no graph for its first gradient, and its phantom does.
        if question == "second":
            continue
        for device, phantom in (("cpu", on_cpu), ("metal", on_metal)):
            real = _ask(fn, call, device, False, question)
            if "answer" in real and real["answer"] != phantom["answer"]:
                found.eager_differs.append(
                    f"{name} on {device}: phantom {phantom['answer']}, eager {real['answer']}"
                )
    return found


def _described_module(module: lucid.nn.Module) -> list[list[object]]:
    named = list(module.named_parameters()) + list(module.named_buffers())
    return [[name, list(t.shape), str(t.dtype), is_shadow(t)] for name, t in named]


def _built(build: Callable[[], lucid.nn.Module], device: str) -> dict[str, object]:
    try:
        with shadow_alloc(), _probe.preserved_globals():
            lucid.set_default_device(device)
            return {"answer": _described_module(build())}
    except Exception as exc:  # the sweep records every refusal, by message
        return {"refused": f"{type(exc).__name__}: {str(exc)[:200]}"}


def _compared_builds(builds: dict[str, Callable[[], lucid.nn.Module]]) -> _Findings:
    found = _Findings()
    for name, build in builds.items():
        found.swept += 1
        on_cpu, on_metal = _built(build, "cpu"), _built(build, "metal")
        if "refused" in on_metal and "refused" not in on_cpu:
            found.builds_differ.append(f"{name}: Metal {on_metal['refused']}")
        elif on_cpu != on_metal:
            found.builds_differ.append(f"{name}: CPU {on_cpu}, Metal {on_metal}"[:400])
        if any(not entry[-1] for entry in on_metal.get("answer", [])):
            found.real_tensors.append(name)
    return found


def _sweep_modules() -> _Findings:
    from lucid.test.audit._axes import _try_construct

    return _compared_builds(
        {
            symbol.qualname: (lambda cls=symbol.obj: _try_construct(cls))
            for symbol in _surface.enumerate_surface()
            if symbol.kind == "module"
        }
    )


def _sweep_families() -> _Findings:
    """The first factory of each family: the whole zoo is half a minute per device."""
    from lucid.models._registry import _REGISTRY, create_model

    first: dict[str, str] = {}
    for name, entry in sorted(_REGISTRY.items()):
        first.setdefault(entry.family, name)
    return _compared_builds(
        {
            f"{family}/{name}": (lambda name=name: create_model(name))
            for family, name in first.items()
        }
    )


def _main(question: str) -> None:
    if question == "modules":
        found = _sweep_modules()
    elif question == "families":
        found = _sweep_families()
    else:
        found = _sweep_ops(question)
    print(json.dumps(dataclasses.asdict(found)))


def _sweep(question: str) -> _Findings:
    done = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-m",
            "lucid.test.unit.nn.test_shadow_alloc_surface",
            question,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=_SWEEP_TIMEOUT,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return _Findings(**json.loads(done.stdout.splitlines()[-1]))


@needs_metal
@pytest.mark.parametrize("question", ["forward", "backward", "second"])
def test_every_op_answers_the_same_on_both_devices(question: str) -> None:
    found = _sweep(question)
    assert found.swept >= _FLOOR[question], f"reached only {found.swept} ops"
    assert found.reads_a_storage == []
    assert found.metal_refuses == []
    assert found.blind_to_dtype == []
    assert set(found.cpu_refuses) == _CPU_READS_VALUES[question]
    assert found.phantoms_differ == []
    assert found.eager_differs == []


@needs_metal
def test_every_module_class_builds_the_same_under_a_metal_default() -> None:
    found = _sweep("modules")
    assert found.swept >= 300, f"reached only {found.swept} classes"
    assert found.builds_differ == []
    assert found.real_tensors == []


@needs_metal
def test_a_factory_of_every_family_builds_the_same_under_a_metal_default() -> None:
    found = _sweep("families")
    assert found.swept >= 50, f"reached only {found.swept} families"
    assert found.builds_differ == []
    assert found.real_tensors == []


#: ``PhantomStorage{...}``, ``PhantomStorage()``, and ``PhantomStorage p`` however it
#: is initialised or passed by value.  The definition in ``Storage.h`` is not one.
_BUILDS_A_PHANTOM = re.compile(
    r"(?<!struct )\bPhantomStorage\s*[({]|\bPhantomStorage\s+\w+\s*[;={(,)]"
)

#: What a ``.dtype`` edit has to start from: an extracted one, or a mutable reference
#: in a visitor.
_REACHES_INTO_A_PHANTOM = re.compile(r"\bget(_if)?<PhantomStorage>|(?<!const )\bPhantomStorage\s*&")


def test_every_phantom_is_made_where_the_dtype_rule_is() -> None:
    """A storage built or re-tagged by hand would stand in for a tensor Metal cannot hold.

    ``TensorImpl`` is the one place that reaches in: it sizes and tags what it adopts,
    and a dtype view of one.
    """
    engine = Path(lucid.__file__).parent / "_C"
    files = [
        p
        for p in engine.rglob("*")
        if p.suffix in (".h", ".cpp", ".mm") and "test" not in p.relative_to(engine).parts
    ]
    if not any(p.suffix == ".h" for p in files):
        pytest.skip("engine sources are not shipped with this install")
    code = {
        p.relative_to(engine).as_posix(): re.sub(r"//[^\n]*", "", p.read_text(errors="ignore"))
        for p in files
    }
    assert {name for name, text in code.items() if _BUILDS_A_PHANTOM.search(text)} == {
        "core/PhantomMode.h",
        "backend/PhantomBackend.h",
    }
    assert {name for name, text in code.items() if _REACHES_INTO_A_PHANTOM.search(text)} == {
        "core/TensorImpl.cpp"
    }


# ── the paths the sweep's single call per op does not take ───────────────────


def _phantom_and_eager(build: Callable[[], Tensor]) -> tuple[Tensor, Tensor]:
    with shadow_alloc():
        phantom = build()
    return phantom, build()


@pytest.mark.parametrize(
    "shape_a, shape_b, dims, expected",
    [
        ((2, 3, 4), (4, 5), 1, (2, 3, 5)),
        ((2, 3), (3, 2), ([1, 0], [0, 1]), ()),
        ((3,), (3,), 1, ()),
        ((2, 2), (2, 2), 0, (2, 2, 2, 2)),
    ],
)
def test_tensordot_answers_with_the_free_axes(shape_a, shape_b, dims, expected, device) -> None:
    def build() -> Tensor:
        return lucid.tensordot(
            lucid.ones(shape_a, device=device), lucid.ones(shape_b, device=device), dims
        )

    phantom, eager = _phantom_and_eager(build)
    assert is_shadow(phantom)
    assert tuple(phantom.shape) == tuple(eager.shape) == expected
    assert phantom.device == eager.device == lucid.device(device)


@pytest.mark.parametrize(
    "shape_a, shape_b, expected",
    [
        ((3,), (3,), ()),
        ((2, 3), (3,), (2,)),
        ((3,), (4, 3), (4,)),
        ((2, 3), (4, 3), (2, 4)),
        ((2, 0), (3, 0), (2, 3)),
    ],
)
def test_inner_answers_with_every_axis_but_the_last(shape_a, shape_b, expected, device) -> None:
    def build() -> Tensor:
        return lucid.linalg.inner(
            lucid.ones(shape_a, device=device), lucid.ones(shape_b, device=device)
        )

    phantom, eager = _phantom_and_eager(build)
    assert is_shadow(phantom)
    assert tuple(phantom.shape) == tuple(eager.shape) == expected
    assert phantom.device == eager.device == lucid.device(device)


@pytest.mark.parametrize(
    "cond, x, y, expected",
    [
        ((2, 3), (2, 3), (2, 3), (2, 3)),
        ((3,), (2, 1), (1,), (2, 3)),
        ((), (2, 3), (), (2, 3)),
        ((), (), (), ()),
        ((0,), (0,), (1,), (0,)),
    ],
)
def test_where_answers_with_the_broadcast_shape(cond, x, y, expected, device) -> None:
    def build() -> Tensor:
        return lucid.where(
            lucid.ones(cond, dtype=lucid.bool_, device=device),
            lucid.ones(x, device=device),
            lucid.ones(y, device=device),
        )

    phantom, eager = _phantom_and_eager(build)
    assert is_shadow(phantom)
    assert tuple(phantom.shape) == tuple(eager.shape) == expected
    assert phantom.device == eager.device == lucid.device(device)


def test_where_values_are_still_the_selected_ones(device) -> None:
    """The shape is no longer read off the array, so check that it describes it."""
    cond = lucid.tensor([[True], [False]], device=device)
    x = lucid.tensor([1.0, 2.0, 3.0], device=device)
    y = lucid.tensor(9.0, device=device)
    out = lucid.where(cond, x, y)
    assert tuple(out.shape) == (2, 3)
    assert out.tolist() == [[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]


def test_kthvalue_picks_the_kth_value_and_answers_as_a_phantom(device) -> None:
    x = lucid.tensor([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]], device=device)
    values = lucid.kthvalue(x, 2, dim=1)
    values = values[0] if isinstance(values, tuple) else values
    assert values.tolist() == [2.0, 5.0]
    with shadow_alloc():
        ghost = lucid.kthvalue(lucid.zeros(2, 3, device=device), 2, dim=1)
    ghost = ghost[0] if isinstance(ghost, tuple) else ghost
    assert is_shadow(ghost)
    assert tuple(ghost.shape) == (2,)
    assert ghost.device == lucid.device(device)


@pytest.mark.parametrize("op", [lucid.cummax, lucid.cummin], ids=["cummax", "cummin"])
def test_a_running_extreme_answers_both_orders_of_gradient(op, device) -> None:
    with shadow_alloc():
        x = lucid.zeros(2, 3, device=device).requires_grad_()
        out = op(x, 1)
        out = out[0] if isinstance(out, tuple) else out
        first = lucid.autograd.grad((out * out).sum(), x, create_graph=True)[0]
        second = lucid.autograd.grad(first.sum(), x, allow_unused=True)[0]
        y = lucid.zeros(2, 3, device=device).requires_grad_()
        out = op(y, 1)
        (out[0] if isinstance(out, tuple) else out).sum().backward()
    for grad in (first, second, y.grad):
        assert is_shadow(grad)
        assert tuple(grad.shape) == (2, 3)
        assert grad.device == lucid.device(device)


def test_the_gradient_of_a_determinant_is_a_phantom(device) -> None:
    with shadow_alloc():
        x = lucid.zeros(2, 3, 3, device=device).requires_grad_()
        lucid.linalg.det(x).sum().backward()
    assert is_shadow(x.grad)
    assert tuple(x.grad.shape) == (2, 3, 3)
    assert x.grad.device == lucid.device(device)


def test_a_gradient_seeded_by_a_view_at_an_offset_is_a_phantom(device) -> None:
    """A Metal gradient that is a view with no family is refused, for the array
    whose geometry the slot would lose.  A phantom view has no family and no array."""
    with shadow_alloc():
        x = lucid.zeros(8, device=device).requires_grad_()
        seed = lucid.eye(8, device=device)[3]
        (x * 2).backward(seed)
    assert is_shadow(x.grad)
    assert tuple(x.grad.shape) == (8,)


@needs_metal
@pytest.mark.parametrize(("src", "dst"), [("cpu", "metal"), ("metal", "cpu")])
def test_a_phantom_gradient_comes_back_across_a_device_move(src, dst) -> None:
    with shadow_alloc():
        x = lucid.zeros(2, 3, device=src).requires_grad_()
        y = x.to(dst)
        (y * 2).sum().backward()
    assert y.device == lucid.device(dst)
    assert is_shadow(x.grad)
    assert x.grad.device == lucid.device(src)
    assert tuple(x.grad.shape) == (2, 3)


@pytest.mark.parametrize(
    "mode, shape",
    [("bilinear", (1, 2, 4, 4)), ("bicubic", (1, 2, 4, 4)), ("trilinear", (1, 1, 2, 2, 2))],
    ids=["bilinear", "bicubic", "trilinear"],
)
def test_a_resample_answers_a_second_gradient(mode, shape, device) -> None:
    with shadow_alloc():
        x = lucid.zeros(*shape, device=device).requires_grad_()
        y = F.interpolate(x, scale_factor=2, mode=mode)
        first = lucid.autograd.grad((y * y).sum(), x, create_graph=True)[0]
        second = lucid.autograd.grad(first.sum(), x, allow_unused=True)[0]
    assert is_shadow(first) and is_shadow(second)
    assert tuple(first.shape) == tuple(second.shape) == shape
    assert second.device == lucid.device(device)


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_an_embedding_bag_answers_a_second_gradient(mode, device) -> None:
    with shadow_alloc():
        w = lucid.zeros(5, 3, device=device).requires_grad_()
        indices = lucid.zeros(4, dtype=lucid.int64, device=device)
        offsets = lucid.zeros(2, dtype=lucid.int64, device=device)
        y = F.embedding_bag(indices, w, offsets, mode=mode)
        first = lucid.autograd.grad((y * y).sum(), w, create_graph=True)[0]
        second = lucid.autograd.grad(first.sum(), w, allow_unused=True)[0]
    assert is_shadow(first) and is_shadow(second)
    assert tuple(first.shape) == tuple(second.shape) == (5, 3)
    assert second.device == lucid.device(device)


@needs_metal
@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_an_embedding_bag_s_second_gradient_runs_on_metal_too(mode) -> None:
    """Its weights were built in doubles and uploaded before the cast, which real
    Metal refuses; the phantom never got that far, so a shadow build hid it."""

    def run(device: str) -> tuple[list[float], list[float]]:
        w = lucid.tensor([[1.0, 2.0], [3.0, 0.0], [0.0, 5.0]], device=device).requires_grad_()
        indices = lucid.tensor([0, 1, 2, 1], dtype=lucid.int64, device=device)
        offsets = lucid.tensor([0, 2], dtype=lucid.int64, device=device)
        y = F.embedding_bag(indices, w, offsets, mode=mode)
        first = lucid.autograd.grad((y * y).sum(), w, create_graph=True)[0]
        second = lucid.autograd.grad(first.sum(), w)[0]
        return first.reshape(-1).tolist(), second.reshape(-1).tolist()

    on_cpu, on_metal = run("cpu"), run("metal")
    assert on_metal[0] == pytest.approx(on_cpu[0])
    assert on_metal[1] == pytest.approx(on_cpu[1])


@needs_metal
def test_a_metal_quantized_linear_answers_as_its_eager_self() -> None:
    def run() -> tuple[QuantizedLinearMLX, Tensor, Tensor]:
        float_layer = lucid.nn.Linear(64, 8, device="metal")
        q = QuantizedLinearMLX.from_float(float_layer, bits=4, group_size=32)
        x = lucid.ones(2, 64, device="metal").requires_grad_()
        y = q(x)
        y.sum().backward()
        return q, y, x

    real_q, real_y, real_x = run()
    with shadow_alloc():
        ghost_q, ghost_y, ghost_x = run()
    for ghost, real in (
        (ghost_q.packed_weight, real_q.packed_weight),
        (ghost_q.scales, real_q.scales),
        (ghost_q.biases, real_q.biases),
        (ghost_y, real_y),
        (ghost_x.grad, real_x.grad),
    ):
        assert is_shadow(ghost)
        assert tuple(ghost.shape) == tuple(real.shape)
        assert ghost.dtype == real.dtype
        assert ghost.device == real.device


if __name__ == "__main__":
    _main(sys.argv[1])
