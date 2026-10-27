"""An in-place op never changes its target's dtype, under autocast or not (LCD-474).

The reference does not autocast an in-place op: ``x.mul_(y)`` computes in
``x``'s own dtype whatever scope it runs in.  Lucid ran the out-of-place
forward under the op's AMP policy and then gave ``x`` the result's dtype.
Under ``autocast(metal, float16)`` that made a float32 ``x`` float16 for
every ``Promote`` op (``mul_``, ``add_``, ``sqrt_``, the ``*=`` family), and
a float16 ``x`` float32 for every ``ForceFP32`` op (``exp_``, ``log_``,
``pow_``).  The composite in-place ops (``F.silu(x, inplace=True)``,
``index_add_`` and the rest) also computed under autocast.  They then either
refused their own result or, while autograd recorded, took its dtype.

Both owners now hold one rule.  The op runs with autocast off on the target's
device.  A result of another dtype is cast back when the in-place rule allows
it (``lucid._promotion.can_cast``) and refused otherwise.  The engine holds
the rule in ``inplace::run`` (``lucid/_C/ops/utils/InplaceGraph.h``), the
composite ops in ``_write_inplace`` (``lucid/_ops/composite/indexing.py``).

Running outside autocast, rather than computing under it and casting back,
is what the reference's numerics require.  A float32 product rounded through
float16 first keeps its dtype but not its value.  The float32-target checks
below compare with the reference to a few float32 ulps, about two orders of
magnitude under one float16 rounding.  The arithmetic ones compare exactly.
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.amp import autocast
from lucid.test._fixtures.devices import metal_available

_X = [0.1, 0.3, 0.55, 0.7, 0.9, 0.35]
_Y = [0.7, 1.3, 0.45, 2.1, 0.9, 1.7]

# (device, autocast dtype): the two scopes the reference runs AMP in.
_SCOPES = [
    pytest.param(
        "metal",
        lucid.float16,
        id="metal-f16",
        marks=pytest.mark.skipif(not metal_available(), reason="Metal unavailable"),
    ),
    pytest.param("cpu", lucid.bfloat16, id="cpu-bf16"),
]
_TARGETS = [pytest.param(lucid.float32, id="f32"), pytest.param(lucid.float16, id="f16")]
_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]

# ``(lib, x, y) -> x`` — one way to write into ``x`` in place, for either library.
Spelling = Callable[..., object]


def _method(name: str, *args: object) -> Spelling:
    return lambda lib, x, y: getattr(x, name)(*args)


def _with_y(name: str) -> Spelling:
    return lambda lib, x, y: getattr(x, name)(y)


def _augmented(op: str, rhs: str) -> Spelling:
    def run(lib: ModuleType, x: object, y: object) -> object:
        scope = {"x": x, "y": y}
        exec(f"x {op} {rhs}", scope)  # an augmented assignment is a statement
        return scope["x"]

    return run


def _index(lib: ModuleType, x: lucid.Tensor) -> object:
    return lib.tensor([0, 2], device=x.device)


def _mask(lib: ModuleType, x: lucid.Tensor) -> object:
    return lib.tensor([True, False, True, False, False, True], device=x.device)


def _functional(name: str, *args: object, **kwargs: object) -> Spelling:
    return lambda lib, x, y: getattr(lib.nn.functional, name)(x, *args, **kwargs)


_UNARY = [
    "abs_", "arccos_", "arcsin_", "arctan_", "ceil_", "cos_", "cosh_", "exp_",
    "floor_", "log2_", "log_", "neg_", "reciprocal_", "relu_", "round_",
    "sigmoid_", "sign_", "sin_", "sinh_", "sqrt_", "square_", "tan_", "tanh_",
    "zero_",
]  # fmt: skip
_BINARY = ["add_", "sub_", "mul_", "div_", "pow_", "copy_"]
_AUGMENTED = ["+=", "-=", "*=", "/=", "**="]

# name -> (Lucid's spelling, the reference's spelling of the same write).
_ENGINE: dict[str, tuple[Spelling, Spelling]] = {
    **{n: (_method(n), _method(n)) for n in _UNARY},
    **{n: (_with_y(n), _with_y(n)) for n in _BINARY},
    "cube_": (_method("cube_"), _method("pow_", 3)),
    "maximum_": (_with_y("maximum_"), lambda lib, x, y: x.copy_(lib.maximum(x, y))),
    "minimum_": (_with_y("minimum_"), lambda lib, x, y: x.copy_(lib.minimum(x, y))),
    "mul_(0.3)": (_method("mul_", 0.3),) * 2,
    "clamp_": (_method("clamp_", 0.3, 0.6),) * 2,
    "clamp_min_": (_method("clamp_min_", 0.3),) * 2,
    "clamp_max_": (_method("clamp_max_", 0.6),) * 2,
    "fill_": (_method("fill_", 0.3),) * 2,
    **{f"x {op} y": (_augmented(op, "y"),) * 2 for op in _AUGMENTED},
    **{f"x {op} 0.3": (_augmented(op, "0.3"),) * 2 for op in _AUGMENTED},
}

_COMPOSITE: dict[str, Spelling] = {
    "F.relu(inplace)": _functional("relu", inplace=True),
    "F.silu(inplace)": _functional("silu", inplace=True),
    "F.elu(inplace)": _functional("elu", inplace=True),
    "F.selu(inplace)": _functional("selu", inplace=True),
    "F.elu_": _functional("elu_"),
    "F.leaky_relu_": _functional("leaky_relu_"),
    "F.hardtanh_": _functional("hardtanh_", 0.2, 0.6),
    "index_add_": lambda lib, x, y: x.index_add_(0, _index(lib, x), y[:2]),
    "index_copy_": lambda lib, x, y: x.index_copy_(0, _index(lib, x), y[:2]),
    "index_fill_": lambda lib, x, y: x.index_fill_(0, _index(lib, x), 0.3),
    "index_put_": lambda lib, x, y: x.index_put_((_index(lib, x),), y[:2]),
    "masked_fill_": lambda lib, x, y: x.masked_fill_(_mask(lib, x), 0.3),
    "masked_scatter_": lambda lib, x, y: x.masked_scatter_(_mask(lib, x), y),
    "scatter_": lambda lib, x, y: x.scatter_(0, _index(lib, x), y[:2]),
    "scatter_add_": lambda lib, x, y: x.scatter_add_(0, _index(lib, x), y[:2]),
    "scatter_reduce_": lambda lib, x, y: x.scatter_reduce_(0, _index(lib, x), y[:2], "sum"),
}

_SPELLINGS: dict[str, tuple[Spelling, Spelling]] = {
    **_ENGINE,
    **{n: (f, f) for n, f in _COMPOSITE.items()},
}


def _lucid_pair(device: str, dtype: lucid.dtype) -> tuple[lucid.Tensor, lucid.Tensor]:
    return (
        lucid.tensor(_X, dtype=dtype, device=device),
        lucid.tensor(_Y, dtype=dtype, device=device),
    )


def _ref_device(ref: ModuleType, device: str) -> str:
    if device == "metal":
        if not ref.backends.mps.is_available():
            pytest.skip("MPS not available in reference framework")
        return "mps"
    return "cpu"


def _ref_dtype(ref: ModuleType, dtype: lucid.dtype) -> object:
    return getattr(ref, str(dtype).split(".")[-1])


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize(("device", "amp"), _SCOPES)
@pytest.mark.parametrize("spelling", sorted(_SPELLINGS))
def test_inplace_op_under_autocast_keeps_target_dtype(
    spelling: str, device: str, amp: lucid.dtype, target: lucid.dtype
) -> None:
    x, y = _lucid_pair(device, target)
    with autocast(device_type=device, dtype=amp):
        out = _SPELLINGS[spelling][0](lucid, x, y)
    assert out is x
    assert x.dtype == target


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize(("device", "amp"), _SCOPES)
@pytest.mark.parametrize("spelling", sorted(_SPELLINGS))
def test_inplace_op_under_autocast_matches_reference(
    ref: ModuleType, spelling: str, device: str, amp: lucid.dtype, target: lucid.dtype
) -> None:
    mine, theirs = _SPELLINGS[spelling]
    x, y = _lucid_pair(device, target)
    rdev = _ref_device(ref, device)
    rdt = _ref_dtype(ref, target)
    rx = ref.tensor(_X, dtype=rdt, device=rdev)
    ry = ref.tensor(_Y, dtype=rdt, device=rdev)
    with autocast(device_type=device, dtype=amp):
        mine(lucid, x, y)
    with ref.autocast(device_type=rdev, dtype=_ref_dtype(ref, amp)):
        rx = theirs(ref, rx, ry)
    assert str(x.dtype).split(".")[-1] == str(rx.dtype).split(".")[-1]
    # A few ulps of the target's own dtype.  One float16 rounding of a
    # float32 target is ~1e-4, far outside the float32 bound.
    rtol, atol = (2e-6, 1e-7) if target == lucid.float32 else (4e-3, 1e-3)
    np.testing.assert_allclose(x.float().numpy(), rx.float().cpu().numpy(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(("device", "amp"), _SCOPES)
@pytest.mark.parametrize("op", ["mul_", "add_", "sub_"])
def test_float32_arithmetic_under_autocast_is_exactly_float32(
    ref: ModuleType, op: str, device: str, amp: lucid.dtype
) -> None:
    """Correctly rounded IEEE arithmetic: the reference's bits, not a half rounding."""
    x, y = _lucid_pair(device, lucid.float32)
    rx = ref.tensor(_X, dtype=ref.float32)
    with autocast(device_type=device, dtype=amp):
        getattr(x, op)(y)
    getattr(rx, op)(ref.tensor(_Y, dtype=ref.float32))
    np.testing.assert_array_equal(x.numpy(), rx.numpy())


@pytest.mark.parametrize(("device", "amp"), _SCOPES)
@pytest.mark.parametrize(
    "spelling",
    ["mul_", "exp_", "sqrt_", "pow_", "x *= y", "F.silu(inplace)", "F.elu(inplace)", "index_add_"],
)
def test_inplace_op_under_autocast_differentiates_in_target_dtype(
    ref: ModuleType, spelling: str, device: str, amp: lucid.dtype
) -> None:
    mine, theirs = _SPELLINGS[spelling]
    rdev = _ref_device(ref, device)
    leaf = lucid.tensor(_X, device=device, requires_grad=True)
    rleaf = ref.tensor(_X, device=rdev, requires_grad=True)
    # The non-leaf target is made outside the scope, so the scope holds the
    # in-place op alone.
    z = leaf * 1.0
    rz = rleaf * 1.0
    with autocast(device_type=device, dtype=amp):
        mine(lucid, z, lucid.tensor(_Y, device=device))
    with ref.autocast(device_type=rdev, dtype=_ref_dtype(ref, amp)):
        rz = theirs(ref, rz, ref.tensor(_Y, device=rdev))
    z.sum().backward()
    rz.sum().backward()
    assert z.dtype == lucid.float32
    assert leaf.grad is not None
    assert leaf.grad.dtype == lucid.float32
    np.testing.assert_allclose(leaf.grad.numpy(), rleaf.grad.cpu().numpy(), rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [lucid.int64, lucid.int32, lucid.bool_], ids=str)
@pytest.mark.parametrize("op", ["exp_", "log_", "sqrt_", "sin_", "sigmoid_", "reciprocal_"])
def test_real_valued_inplace_op_refuses_integral_target(
    ref: ModuleType, op: str, dtype: lucid.dtype, device: str
) -> None:
    """The result is float and cannot be cast to the target: refused, target unchanged."""
    x = lucid.tensor([1, 2, 3], device=device).to(dtype)
    before = x.tolist()
    with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
        getattr(x, op)()
    assert x.dtype == dtype
    assert x.tolist() == before
    rx = ref.tensor([1, 2, 3]).to(_ref_dtype(ref, dtype))
    with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
        getattr(rx, op)()


_ENGINE_OPTIMIZERS: dict[str, dict[str, object]] = {
    "SGD": {"lr": 0.1, "momentum": 0.9},
    "Adam": {"lr": 0.1},
    "AdamW": {"lr": 0.1},
    "RMSprop": {"lr": 0.1},
    "Adagrad": {"lr": 0.1},
    "Adadelta": {"lr": 0.1},
    "Adamax": {"lr": 0.1},
    "NAdam": {"lr": 0.1},
    "RAdam": {"lr": 0.1},
    "ASGD": {"lr": 0.1},
    "Rprop": {"lr": 0.1},
}


def _stepped(name: str, device: str, amp: lucid.dtype | None) -> list[lucid.Tensor]:
    """One step of ``name`` from a fixed start with fixed gradients."""
    lucid.manual_seed(0)
    model = nn.Linear(4, 3).to(device)
    opt = getattr(optim, name)(model.parameters(), **_ENGINE_OPTIMIZERS[name])
    for p in model.parameters():
        p.grad = p.detach() * 0.3
    if amp is None:
        opt.step()
    else:
        with autocast(device_type=device, dtype=amp):
            opt.step()
    return list(model.parameters())


@pytest.mark.parametrize(("device", "amp"), _SCOPES)
@pytest.mark.parametrize("name", sorted(_ENGINE_OPTIMIZERS))
def test_optimizer_step_inside_autocast_keeps_float32_parameters(
    name: str, device: str, amp: lucid.dtype
) -> None:
    inside = _stepped(name, device, amp)
    outside = _stepped(name, device, None)
    for p, q in zip(inside, outside, strict=True):
        assert p.dtype == lucid.float32
        np.testing.assert_array_equal(p.detach().numpy(), q.detach().numpy())


@pytest.mark.parametrize(("device", "amp"), _SCOPES)
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        pytest.param("LBFGS", {"lr": 0.1}, id="LBFGS"),
        pytest.param(
            "NAdam",
            {"lr": 0.1, "weight_decay": 0.1, "decoupled_weight_decay": True},
            id="NAdam-decoupled",
        ),
    ],
)
def test_python_optimizer_step_inside_autocast_keeps_float32_parameters(
    name: str, kwargs: dict[str, object], device: str, amp: lucid.dtype
) -> None:
    lucid.manual_seed(0)
    model = nn.Linear(4, 3).to(device)
    data = lucid.randn(2, 4, device=device)
    opt = getattr(optim, name)(model.parameters(), **kwargs)

    def closure() -> lucid.Tensor:
        opt.zero_grad()
        loss = (model(data).float() ** 2).sum()
        loss.backward()
        return loss

    with autocast(device_type=device, dtype=amp):
        opt.step(closure)
    assert all(p.dtype == lucid.float32 for p in model.parameters())
