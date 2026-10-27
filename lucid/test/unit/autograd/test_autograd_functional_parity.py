"""Parity guard for ``lucid.autograd`` functional transforms (LCD-334 / #407).

Verifies ``jacobian``, ``hessian``, ``vjp``, ``jvp``, ``vhp``, and ``hvp``
against ``ref.autograd.functional`` across:
- ``0-D`` scalar, ``1-D``, ``2-D``, and ``3-D`` input and output shapes
- Single ``Tensor`` vs ``tuple[Tensor, ...]`` inputs and outputs
- Functions with unused inputs (returning zero tensors of matching shape,
  dtype, and device rather than ``None``)
- Boundary validation refusing non-``Tensor`` / non-``tuple`` containers.
"""

import math
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._tensor.tensor import Tensor

_SHAPES: list[tuple[int, ...]] = [(), (3,), (2, 3), (2, 1, 2)]
_TOL = 1e-10


def _ns(m: ModuleType) -> object:
    return lucid.autograd if m is lucid else getattr(m.autograd, "functional")


def _make_tensor(
    m: ModuleType, shape: tuple[int, ...], offset: float = 0.25, device: str = "cpu"
) -> Tensor:
    n = 1
    for s in shape:
        n *= s
    vals = [offset + 0.3 * i for i in range(n)]
    t: Tensor = m.tensor(vals, dtype=m.float64, device=device)
    return t.reshape(shape)


def _tree_spec(item: object) -> object:
    if isinstance(item, tuple):
        return tuple(_tree_spec(x) for x in item)
    assert hasattr(item, "shape") and hasattr(item, "dtype")
    shape = tuple(getattr(item, "shape"))
    dtype = str(getattr(item, "dtype")).split(".")[-1]
    return (shape, dtype)


def _tree_flat(item: object) -> list[float]:
    if isinstance(item, tuple):
        return [v for x in item for v in _tree_flat(x)]
    detached = getattr(item, "detach")().cpu().reshape(-1).tolist()
    return [float(v) for v in detached]


def _assert_parity(got: object, want: object, tol: float = _TOL) -> None:
    assert _tree_spec(got) == _tree_spec(want)
    g_vals = _tree_flat(got)
    w_vals = _tree_flat(want)
    assert len(g_vals) == len(w_vals)
    assert all(
        math.isclose(a, b, rel_tol=tol, abs_tol=tol) for a, b in zip(g_vals, w_vals, strict=True)
    ), (g_vals, w_vals)


@pytest.mark.parity
@pytest.mark.parametrize("in_shape", _SHAPES, ids=["in0d", "in1d", "in2d", "in3d"])
@pytest.mark.parametrize("out_shape", _SHAPES, ids=["out0d", "out1d", "out2d", "out3d"])
def test_jacobian_vjp_jvp_single_tensor_shapes(
    ref: ModuleType, in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> None:
    def make_fn(m: ModuleType) -> Callable[[Tensor], Tensor]:
        w = _make_tensor(m, out_shape, offset=0.5)

        def fn(x: Tensor) -> Tensor:
            base = (m.sin(x) + x * x).sum()
            return base * w

        return fn

    x_l = _make_tensor(lucid, in_shape)
    x_r = _make_tensor(ref, in_shape)
    v_out_l = _make_tensor(lucid, out_shape, offset=-0.4)
    v_out_r = _make_tensor(ref, out_shape, offset=-0.4)
    v_in_l = _make_tensor(lucid, in_shape, offset=0.6)
    v_in_r = _make_tensor(ref, in_shape, offset=0.6)

    jac_l = lucid.autograd.jacobian(make_fn(lucid), x_l)
    jac_r = ref.autograd.functional.jacobian(make_fn(ref), x_r)
    assert isinstance(jac_l, Tensor)
    assert jac_l.shape == out_shape + in_shape
    _assert_parity(jac_l, jac_r)

    vjp_l = lucid.autograd.vjp(make_fn(lucid), x_l, v_out_l)
    vjp_r = ref.autograd.functional.vjp(make_fn(ref), x_r, v_out_r)
    assert isinstance(vjp_l[1], Tensor)
    assert vjp_l[1].shape == in_shape
    _assert_parity(vjp_l, vjp_r)

    jvp_l = lucid.autograd.jvp(make_fn(lucid), x_l, v_in_l)
    jvp_r = ref.autograd.functional.jvp(make_fn(ref), x_r, v_in_r)
    assert isinstance(jvp_l[1], Tensor)
    assert jvp_l[1].shape == out_shape
    _assert_parity(jvp_l, jvp_r)


@pytest.mark.parity
@pytest.mark.parametrize("in_shape", _SHAPES, ids=["in0d", "in1d", "in2d", "in3d"])
def test_hessian_vhp_hvp_single_tensor_shapes(ref: ModuleType, in_shape: tuple[int, ...]) -> None:
    def scalar_fn(x: Tensor) -> Tensor:
        return (x * x * x).sum() + (x.sum() ** 2) * 0.5

    x_l = _make_tensor(lucid, in_shape)
    x_r = _make_tensor(ref, in_shape)
    v_l = _make_tensor(lucid, in_shape, offset=0.7)
    v_r = _make_tensor(ref, in_shape, offset=0.7)

    hess_l = lucid.autograd.hessian(scalar_fn, x_l)
    hess_r = ref.autograd.functional.hessian(scalar_fn, x_r)
    assert isinstance(hess_l, Tensor)
    assert hess_l.shape == in_shape + in_shape
    _assert_parity(hess_l, hess_r)

    vhp_l = lucid.autograd.vhp(scalar_fn, x_l, v_l)
    vhp_r = ref.autograd.functional.vhp(scalar_fn, x_r, v_r)
    assert isinstance(vhp_l[1], Tensor)
    assert vhp_l[1].shape == in_shape
    _assert_parity(vhp_l, vhp_r)

    hvp_l = lucid.autograd.hvp(scalar_fn, x_l, v_l)
    hvp_r = ref.autograd.functional.hvp(scalar_fn, x_r, v_r)
    assert isinstance(hvp_l[1], Tensor)
    assert hvp_l[1].shape == in_shape
    _assert_parity(hvp_l, hvp_r)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [((), (2,)), ((2,), (2, 3)), ((2, 2), (2, 1, 2))],
    ids=["0d-1d", "1d-2d", "2d-3d"],
)
def test_multi_tensor_tuple_inputs_and_outputs(
    ref: ModuleType, shape_a: tuple[int, ...], shape_b: tuple[int, ...]
) -> None:
    def multi_out(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (a * a + b.sum(), b * b * a.sum(), (a.sum() * b.sum()))

    def scalar_out(a: Tensor, b: Tensor) -> Tensor:
        return ((a**3).sum() * (b**2).sum()) + (a * a).sum() + (b**3).sum()

    a_l, b_l = _make_tensor(lucid, shape_a, 0.3), _make_tensor(lucid, shape_b, -0.5)
    a_r, b_r = _make_tensor(ref, shape_a, 0.3), _make_tensor(ref, shape_b, -0.5)
    va_l, vb_l = _make_tensor(lucid, shape_a, 0.4), _make_tensor(lucid, shape_b, 0.8)
    va_r, vb_r = _make_tensor(ref, shape_a, 0.4), _make_tensor(ref, shape_b, 0.8)
    vo_l = (va_l, vb_l, lucid.tensor(1.5, dtype=lucid.float64))
    vo_r = (va_r, vb_r, ref.tensor(1.5, dtype=ref.float64))

    jac_l = lucid.autograd.jacobian(multi_out, (a_l, b_l))
    jac_r = ref.autograd.functional.jacobian(multi_out, (a_r, b_r))
    _assert_parity(jac_l, jac_r)

    vjp_l = lucid.autograd.vjp(multi_out, (a_l, b_l), vo_l)
    vjp_r = ref.autograd.functional.vjp(multi_out, (a_r, b_r), vo_r)
    _assert_parity(vjp_l, vjp_r)

    jvp_l = lucid.autograd.jvp(multi_out, (a_l, b_l), (va_l, vb_l))
    jvp_r = ref.autograd.functional.jvp(multi_out, (a_r, b_r), (va_r, vb_r))
    _assert_parity(jvp_l, jvp_r)

    hess_l = lucid.autograd.hessian(scalar_out, (a_l, b_l))
    hess_r = ref.autograd.functional.hessian(scalar_out, (a_r, b_r))
    _assert_parity(hess_l, hess_r)

    vhp_l = lucid.autograd.vhp(scalar_out, (a_l, b_l), (va_l, vb_l))
    vhp_r = ref.autograd.functional.vhp(scalar_out, (a_r, b_r), (va_r, vb_r))
    _assert_parity(vhp_l, vhp_r)

    hvp_l = lucid.autograd.hvp(scalar_out, (a_l, b_l), (va_l, vb_l))
    hvp_r = ref.autograd.functional.hvp(scalar_out, (a_r, b_r), (va_r, vb_r))
    _assert_parity(hvp_l, hvp_r)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("shape_x", "shape_y"), [((), (3,)), ((2,), (2, 3))], ids=["0d-1d", "1d-2d"]
)
def test_unused_inputs_return_zeros_matching_reference(
    ref: ModuleType, shape_x: tuple[int, ...], shape_y: tuple[int, ...]
) -> None:
    def only_first_scalar(x: Tensor, y: Tensor) -> Tensor:
        del y  # Intentionally unused input to verify zero-fill semantics.
        return (x * x * x).sum()

    def only_first_tuple(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        del y  # Intentionally unused input to verify zero-fill semantics.
        return (x * x, (x * x * x).sum())

    x_l, y_l = _make_tensor(lucid, shape_x, 0.5), _make_tensor(lucid, shape_y, 1.1)
    x_r, y_r = _make_tensor(ref, shape_x, 0.5), _make_tensor(ref, shape_y, 1.1)
    vx_l, vy_l = _make_tensor(lucid, shape_x, 0.3), _make_tensor(lucid, shape_y, 0.7)
    vx_r, vy_r = _make_tensor(ref, shape_x, 0.3), _make_tensor(ref, shape_y, 0.7)

    _assert_parity(
        lucid.autograd.jacobian(only_first_tuple, (x_l, y_l)),
        ref.autograd.functional.jacobian(only_first_tuple, (x_r, y_r)),
    )
    _assert_parity(
        lucid.autograd.hessian(only_first_scalar, (x_l, y_l)),
        ref.autograd.functional.hessian(only_first_scalar, (x_r, y_r)),
    )
    _assert_parity(
        lucid.autograd.vjp(only_first_scalar, (x_l, y_l)),
        ref.autograd.functional.vjp(only_first_scalar, (x_r, y_r)),
    )
    _assert_parity(
        lucid.autograd.jvp(only_first_tuple, (x_l, y_l), (vx_l, vy_l)),
        ref.autograd.functional.jvp(only_first_tuple, (x_r, y_r), (vx_r, vy_r)),
    )
    _assert_parity(
        lucid.autograd.vhp(only_first_scalar, (x_l, y_l), (vx_l, vy_l)),
        ref.autograd.functional.vhp(only_first_scalar, (x_r, y_r), (vx_r, vy_r)),
    )
    _assert_parity(
        lucid.autograd.hvp(only_first_scalar, (x_l, y_l), (vx_l, vy_l)),
        ref.autograd.functional.hvp(only_first_scalar, (x_r, y_r), (vx_r, vy_r)),
    )


def test_unused_input_device_and_dtype_preserved(device: str) -> None:
    x = lucid.tensor([1.5, -0.5], dtype=lucid.float32, device=device)
    y = lucid.ones((2, 3), dtype=lucid.float32, device=device)

    def f(a: Tensor, b: Tensor) -> Tensor:
        del b  # Unused second input.
        return (a * a * a).sum()

    _, grads = lucid.autograd.vjp(f, (x, y))
    assert isinstance(grads, tuple) and len(grads) == 2
    assert grads[1] is not None
    assert grads[1].shape == (2, 3)
    assert grads[1].dtype == lucid.float32
    assert grads[1].device == x.device
    assert grads[1].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    jac = lucid.autograd.jacobian(f, (x, y))
    assert isinstance(jac, tuple) and len(jac) == 2
    assert isinstance(jac[1], Tensor)
    assert jac[1].shape == (2, 3)
    assert jac[1].device == x.device

    hess = lucid.autograd.hessian(f, (x, y))
    assert isinstance(hess, tuple) and len(hess) == 2
    assert hess[0][1].shape == (2, 2, 3)
    assert hess[1][0].shape == (2, 3, 2)
    assert hess[1][1].shape == (2, 3, 2, 3)
    assert hess[1][1].device == x.device


@pytest.mark.parity
@pytest.mark.parametrize("fn_name", ["jacobian", "hessian", "vjp", "jvp", "vhp", "hvp"])
def test_non_tensor_or_non_tuple_containers_raise_type_error(ref: ModuleType, fn_name: str) -> None:
    for m in (lucid, ref):
        ns = _ns(m)
        fn = getattr(ns, fn_name)
        x = m.tensor(2.0, dtype=m.float64)
        f = lambda a: (a * a).sum()
        with pytest.raises(TypeError, match="must be either a Tensor or a tuple"):
            fn(f, [x])
        with pytest.raises(TypeError, match="must be either a Tensor or a tuple"):
            fn(f, (x, 1))
        with pytest.raises(TypeError, match="must be either a Tensor or a tuple"):
            fn(lambda a: [a.sum()], x)
        if fn_name in ("vjp", "jvp", "vhp", "hvp"):
            with pytest.raises(TypeError, match="must be either a Tensor or a tuple"):
                fn(f, x, [x])
