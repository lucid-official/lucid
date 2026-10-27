"""An autocast op hands each input's gradient back in that input's dtype.

LCD-253 / LCD-318.  The unary, binary and reduction kernel bases cast an
input to the autocast dtype for the computation, and that cast was not
part of the graph: the op returned the input's gradient in the autocast
dtype.  A leaf survived — its gradient sink casts to the leaf's dtype —
but a view, slice or permute between the leaf and the op wrapped the
bfloat16 bytes with its own float32 dtype, and the CPU read raw bits:
``(x[:, 0] * ones(2)).sum()`` gave ``x.grad = [[1.0019, 0, ...], [0, ...]]``
(``1.0019`` is ``0x3F803F80``, two bfloat16 ones).  Metal hid the wrong
values because MLX promotes, but the gradient still reached the view as
bfloat16.

As in the reference, the cast is differentiable: the gradient each
operand receives has the operand's dtype, the leaf's values match the
reference under its own autocast, and a second derivative flows through.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

_DTYPES = {"bfloat16": lucid.bfloat16, "float16": lucid.float16}


@pytest.fixture(params=["cpu", "metal"])
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "metal" and not metal_available():
        pytest.skip("Metal not available on this host")
    return str(request.param)


# Tensors below are Lucid's or the reference's, so they are typed ``Any``.

# ── operand paths: how the autocast op's operand is reached from the leaf ──
#
# Each path is ``(leaf_shape(op_shape), operand(leaf))``.  Every path but
# "leaf" puts a non-leaf node between the leaf and the autocast op, which
# is where a gradient in the wrong dtype used to be misread.


def _swap_last(shape: tuple[int, ...]) -> tuple[int, ...]:
    return (*shape[:-2], shape[-1], shape[-2])


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


_Shape = tuple[int, ...]
_PATHS: dict[str, tuple[Callable[[_Shape], _Shape], Callable[[Any, _Shape], Any]]] = {
    "leaf": (lambda s: s, lambda x, s: x),
    "view": (lambda s: (_numel(s),), lambda x, s: x.reshape(*s)),
    "narrow": (lambda s: (s[0] + 1, *s[1:]), lambda x, s: x[1:]),
    "select": (lambda s: (2, *s), lambda x, s: x[1]),
    "transpose": (_swap_last, lambda x, s: x.transpose(-1, -2)),
    "contiguous": (_swap_last, lambda x, s: x.transpose(-1, -2).contiguous()),
}


# ── ops: one per kernel family that runs under an AMP policy ──
#
# Each op is ``(operand_shape, fn(functional, operand, consts))`` where
# ``consts`` are fixed float32 tensors of the same library.  Since the
# policy table (LCD-500) the element-wise and reduction families have no
# row and keep a float32 operand as it is; they check that the path
# without a cast hands the gradient back intact too.  matmul and conv are
# LOWER and cast it.


def _consts(lib: ModuleType, device: str, rng: np.random.Generator) -> dict[str, Any]:
    def t(*shape: int) -> Any:
        return lib.tensor(rng.standard_normal(shape).astype(np.float32), device=device)

    return {
        "w34": t(3, 4),
        "w45": t(4, 5),
        "k": t(2, 3, 3, 3),
        "idx": lib.tensor(np.array([[0, 2], [3, 1]], dtype=np.int64), device=device),
    }


_OPS: dict[str, tuple[_Shape, Callable[[ModuleType, Any, dict[str, Any]], Any]]] = {
    # BinaryKernel, no row.
    "binary": ((3, 4), lambda fn, a, c: a * c["w34"]),
    # UnaryKernel, no row.
    "unary": ((3, 4), lambda fn, a, c: a.cos()),
    # ReduceKernel over a binary.
    "reduce": ((3, 4), lambda fn, a, c: (a * c["w34"]).sum(dim=1)),
    # Matmul / conv / embedding cast through ``astype`` (or not at all).
    "matmul": ((3, 4), lambda fn, a, c: a @ c["w45"]),
    "conv": ((1, 3, 5, 5), lambda fn, a, c: fn.conv2d(a, c["k"])),
    "embedding": ((4, 3), lambda fn, a, c: fn.embedding(c["idx"], a)),
}


def _run_lucid(
    op: str, path: str, dtype: str, device: str, seed: int
) -> tuple[lucid.Tensor, list[lucid.Tensor], lucid.Tensor]:
    rng = np.random.default_rng(seed)
    op_shape, fn = _OPS[op]
    leaf_shape_of, operand_of = _PATHS[path]
    leaf_np = rng.standard_normal(leaf_shape_of(op_shape)).astype(np.float32)
    consts = _consts(lucid, device, rng)
    x = lucid.tensor(leaf_np, device=device).requires_grad_()
    operand = operand_of(x, op_shape)
    assert isinstance(operand, lucid.Tensor)
    seen: list[lucid.Tensor] = []
    if operand is not x:
        operand.register_hook(lambda g: seen.append(g))
    with lucid.amp.autocast(device, dtype=_DTYPES[dtype]):
        out = fn(F, operand, consts)
    assert isinstance(out, lucid.Tensor)
    weight = lucid.tensor(rng.standard_normal(out.shape).astype(np.float32), device=device)
    (out.float() * weight).sum().backward()
    return x, seen, operand


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("path", list(_PATHS))
@pytest.mark.parametrize("op", list(_OPS))
def test_operand_gradient_keeps_operand_dtype(op: str, path: str, dtype: str, device: str) -> None:
    x, seen, operand = _run_lucid(op, path, dtype, device, seed=0)
    assert x.grad is not None
    assert x.grad.dtype == lucid.float32
    assert all(g.dtype == operand.dtype == lucid.float32 for g in seen), [g.dtype for g in seen]
    assert len(seen) == (0 if path == "leaf" else 1)


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("path", list(_PATHS))
@pytest.mark.parametrize("op", list(_OPS))
def test_leaf_gradient_matches_reference(
    ref: ModuleType, op: str, path: str, dtype: str, device: str
) -> None:
    x, _, _ = _run_lucid(op, path, dtype, device, seed=1)

    rng = np.random.default_rng(1)
    op_shape, fn = _OPS[op]
    leaf_shape_of, operand_of = _PATHS[path]
    leaf_np = rng.standard_normal(leaf_shape_of(op_shape)).astype(np.float32)
    consts = _consts(ref, "cpu", rng)
    rx = ref.tensor(leaf_np, requires_grad=True)
    with ref.autocast(device_type="cpu", dtype=getattr(ref, dtype)):
        rout = fn(ref.nn.functional, operand_of(rx, op_shape), consts)
    weight = ref.tensor(rng.standard_normal(tuple(rout.shape)).astype(np.float32))
    (rout.float() * weight).sum().backward()

    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), rx.grad.numpy(), rtol=5e-2, atol=1e-1)


# ── a half input to an op that computes in float32 gets a half gradient ──


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("path", ["leaf", "transpose", "narrow"])
@pytest.mark.parametrize("op", ["exp", "mul", "prod"])
def test_half_operand_gradient_keeps_half_dtype(
    op: str, path: str, dtype: str, device: str
) -> None:
    """``exp`` is FP32 on Metal and ``prod`` FP32 on both devices: they cast
    a half operand *up*, and its gradient must come back down.  ``mul`` has
    no row and computes in the half dtype; its gradient is half as well."""
    half = _DTYPES[dtype]
    leaf_shape_of, operand_of = _PATHS[path]
    rng = np.random.default_rng(2)
    leaf_np = rng.standard_normal(leaf_shape_of((3, 4))).astype(np.float32)
    x = lucid.tensor(leaf_np, device=device).to(half).detach().requires_grad_()
    operand = operand_of(x, (3, 4))
    assert isinstance(operand, lucid.Tensor)
    seen: list[lucid.Tensor] = []
    if operand is not x:
        operand.register_hook(lambda g: seen.append(g))
    with lucid.amp.autocast(device, dtype=half):
        if op == "exp":
            out = operand.exp()
        elif op == "prod":
            out = operand.prod()
        else:
            out = operand * operand
    out.float().sum().backward()

    assert x.grad is not None and x.grad.dtype == half
    assert all(g.dtype == half for g in seen), [g.dtype for g in seen]
    used = x.detach().float().numpy()[_used(path)].astype(np.float64)
    want = {"exp": np.exp(used), "prod": np.prod(used) / used, "mul": 2 * used}[op]
    np.testing.assert_allclose(x.grad.float().numpy()[_used(path)], want, rtol=2e-2, atol=2e-2)


def _used(path: str) -> tuple[slice, ...]:
    """The leaf elements the operand reads (``narrow`` drops row 0)."""
    return (slice(1, None),) if path == "narrow" else (slice(None),)


# ── create_graph: the cast-back is differentiable ──


def test_second_derivative_through_an_autocast_cast(ref: ModuleType, device: str) -> None:
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((3, 4)).astype(np.float32)

    x = lucid.tensor(x_np, device=device).requires_grad_()
    with lucid.amp.autocast(device, dtype=lucid.bfloat16):
        y = (x.T.cos() * x.T).sum()
    (g,) = lucid.autograd.grad(y, x, create_graph=True)
    assert g.dtype == lucid.float32
    (g * g).sum().backward()

    rx = ref.tensor(x_np, requires_grad=True)
    with ref.autocast(device_type="cpu", dtype=ref.bfloat16):
        ry = (rx.T.cos() * rx.T).sum()
    (rg,) = ref.autograd.grad(ry, rx, create_graph=True)
    (rg * rg).sum().backward()

    assert x.grad is not None and x.grad.dtype == lucid.float32
    np.testing.assert_allclose(g.detach().numpy(), rg.detach().numpy(), rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(x.grad.numpy(), rx.grad.numpy(), rtol=5e-2, atol=1e-1)
