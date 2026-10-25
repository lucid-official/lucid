"""Product gradients are defined at zero operands (LCD-289).

The gradient of a product with respect to one operand is the product of
the others.  ``scatter_prod``, ``prod`` and ``cumprod`` formed it as
``out / operand``, which is ``0 / 0`` — NaN — wherever the operand is
zero, where the reference gives the others' product (and 0 when another
zero is among them).  For ``x = [[4, 5, 0]]``, ``index = [[0, 0, 1]]``,
``src = [[0, 2, 0]]``, ``scatter_reduce(x, 1, index, src, "prod")`` gave
``x.grad = [0, 0, nan]`` and ``src.grad = [nan, 0, nan]`` against the
reference's ``[0, 0, 1]`` and ``[8, 0, 5]``.

Every case below runs on both devices against the reference, with the
zeros placed so the slices hold none, one, two and only zeros, and
``gradcheck`` confirms the float64 gradients against finite differences.
``prod``'s and ``cumprod``'s backward are themselves differentiable, so
``gradgradcheck`` runs on them too — for ``cumprod`` only on slices with
at most one zero, which is where its second derivative is exact (the
reference's is exact everywhere; matching it past a slice's second zero
needs a quadratic formula).
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.autograd import gradcheck, gradgradcheck
from lucid.test._fixtures.devices import metal_available

_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]
_TOL = 1e-5

# 2 x 4 inputs: across rows, columns and the whole tensor the slices hold
# no zero, one, two, and only zeros.
_X = {
    "none": [[2.0, 3.0, 0.5, -4.0], [1.5, -2.0, 3.0, 0.5]],
    "lone": [[2.0, 0.0, 0.5, -4.0], [1.5, -2.0, 3.0, 0.5]],
    "pair": [[2.0, 0.0, 0.5, 0.0], [1.5, -2.0, 3.0, 0.5]],
    "spread": [[0.0, 3.0, 0.5, 0.0], [0.0, -2.0, 0.0, 0.5]],
    "all": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
}

# (base, index, src) for scatter along dim 1.
_SCATTER = {
    "none": (
        [[4.0, 5.0, 2.0], [1.0, 3.0, 2.0]],
        [[0, 0, 1, 2], [2, 2, 2, 0]],
        [[2.0, 3.0, 0.5, 1.5], [2.0, -1.0, 0.5, 3.0]],
    ),
    "reported": ([[4.0, 5.0, 0.0]], [[0, 0, 1]], [[0.0, 2.0, 0.0]]),
    "lone_src": (
        [[4.0, 5.0, 2.0], [1.0, 3.0, 2.0]],
        [[0, 0, 1, 2], [2, 2, 2, 0]],
        [[0.0, 3.0, 0.5, 1.5], [2.0, -1.0, 0.0, 3.0]],
    ),
    "lone_base": (
        [[0.0, 5.0, 2.0], [1.0, 3.0, 0.0]],
        [[0, 0, 1, 2], [2, 2, 2, 0]],
        [[2.0, 3.0, 0.5, 1.5], [2.0, -1.0, 0.5, 3.0]],
    ),
    "base_and_src": (
        [[0.0, 5.0, 2.0], [1.0, 3.0, 0.0]],
        [[0, 0, 1, 2], [2, 2, 2, 0]],
        [[0.0, 3.0, 0.5, 1.5], [2.0, 0.0, 0.5, 3.0]],
    ),
    "src_pair": (
        [[4.0, 5.0, 2.0], [1.0, 3.0, 2.0]],
        [[0, 0, 1, 2], [2, 2, 2, 0]],
        [[0.0, 0.0, 0.5, 1.5], [0.0, -1.0, 0.0, 3.0]],
    ),
    "all": (
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0, 0, 1, 2], [2, 2, 2, 0]],
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ),
}


def _weights(shape: tuple[int, ...]) -> object:
    """Distinct upstream gradients, 1, 2, 3, ... in ``shape``."""
    if not shape:
        return 1.5
    n = 1
    for s in shape:
        n *= s
    flat: list[object] = [float(k + 1) for k in range(n)]
    for s in reversed(shape[1:]):
        flat = [flat[k : k + s] for k in range(0, len(flat), s)]
    return flat


def _close(got: object, want: object, tol: float = _TOL) -> None:
    if isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            _close(g, w, tol)
        return
    assert isinstance(got, float) and isinstance(want, float)
    assert got == pytest.approx(want, rel=tol, abs=tol), (got, want)


# op name -> (lucid call, reference call)
_REDUCE: dict[str, tuple[Callable[..., lucid.Tensor], Callable[..., object]]] = {
    "prod": (lambda t: t.prod(), lambda t: t.prod()),
    "prod_dim0": (lambda t: t.prod(0), lambda t: t.prod(0)),
    "prod_dim1_keepdim": (
        lambda t: t.prod(1, keepdim=True),
        lambda t: t.prod(1, keepdim=True),
    ),
    "cumprod_dim0": (lambda t: t.cumprod(0), lambda t: t.cumprod(0)),
    "cumprod_dim1": (lambda t: t.cumprod(1), lambda t: t.cumprod(1)),
}


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("op", list(_REDUCE))
@pytest.mark.parametrize("case", list(_X))
def test_reduction_gradient_matches_reference(
    ref: ModuleType, device: str, op: str, case: str
) -> None:
    ours, theirs = _REDUCE[op]
    xr = ref.tensor(_X[case], requires_grad=True)
    want = theirs(xr)
    want.backward(ref.tensor(_weights(tuple(want.shape))))

    xl = lucid.tensor(_X[case], device=device, requires_grad=True)
    got = ours(xl)
    got.backward(lucid.tensor(_weights(tuple(got.shape)), device=device))

    _close(got.tolist(), want.tolist())
    _close(xl.grad.tolist(), xr.grad.tolist())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("case", list(_SCATTER))
def test_scatter_prod_gradient_matches_reference(
    ref: ModuleType, device: str, include_self: bool, case: str
) -> None:
    base, index, src = _SCATTER[case]
    xr = ref.tensor(base, requires_grad=True)
    sr = ref.tensor(src, requires_grad=True)
    want = xr.scatter_reduce(
        1, ref.tensor(index), sr, "prod", include_self=include_self
    )
    want.backward(ref.tensor(_weights(tuple(want.shape))))

    xl = lucid.tensor(base, device=device, requires_grad=True)
    sl = lucid.tensor(src, device=device, requires_grad=True)
    got = lucid.scatter_reduce(
        xl, 1, lucid.tensor(index, device=device), sl, "prod", include_self=include_self
    )
    got.backward(lucid.tensor(_weights(tuple(got.shape)), device=device))

    _close(got.tolist(), want.tolist())
    _close(xl.grad.tolist(), xr.grad.tolist())
    _close(sl.grad.tolist(), sr.grad.tolist())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("op", list(_REDUCE))
@pytest.mark.parametrize("case", ["none", "lone", "pair"])
def test_bfloat16_reduction_matches_reference(
    ref: ModuleType, device: str, op: str, case: str
) -> None:
    """bfloat16 too — the CPU ``prod`` reduction had no bfloat16 kernel."""
    ours, theirs = _REDUCE[op]
    xr = ref.tensor(_X[case], dtype=ref.bfloat16, requires_grad=True)
    want = theirs(xr)
    want.backward(ref.tensor(_weights(tuple(want.shape)), dtype=ref.bfloat16))

    xl = lucid.tensor(_X[case]).to(lucid.bfloat16).to(device).detach().requires_grad_()
    got = ours(xl)
    got.backward(lucid.tensor(_weights(tuple(got.shape))).to(lucid.bfloat16).to(device))

    assert got.dtype == lucid.bfloat16
    assert xl.grad.dtype == lucid.bfloat16
    _close(got.float().tolist(), want.float().tolist(), tol=1e-2)
    _close(xl.grad.float().tolist(), xr.grad.float().tolist(), tol=1e-2)


def _f64(values: list[object]) -> lucid.Tensor:
    return lucid.tensor(values, dtype=lucid.float64, requires_grad=True)


def _weighted(fn: Callable[..., lucid.Tensor]) -> Callable[..., lucid.Tensor]:
    """``fn``'s output contracted with distinct weights, as a scalar."""

    def scalar(*args: lucid.Tensor) -> lucid.Tensor:
        out = fn(*args)
        w = lucid.tensor(_weights(tuple(out.shape)), dtype=lucid.float64)
        return (out * w).sum()

    return scalar


@pytest.mark.parametrize("op", list(_REDUCE))
@pytest.mark.parametrize("case", list(_X))
def test_reduction_gradcheck(op: str, case: str) -> None:
    assert gradcheck(_weighted(_REDUCE[op][0]), [_f64(_X[case])])


@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("case", list(_SCATTER))
def test_scatter_prod_gradcheck(include_self: bool, case: str) -> None:
    base, index, src = _SCATTER[case]
    idx = lucid.tensor(index)

    def fn(x: lucid.Tensor, s: lucid.Tensor) -> lucid.Tensor:
        return lucid.scatter_reduce(x, 1, idx, s, "prod", include_self=include_self)

    assert gradcheck(_weighted(fn), [_f64(base), _f64(src)])


def _gradgrad_params() -> list[object]:
    out = []
    for op in _REDUCE:
        out += [pytest.param(op, case, id=f"{op}-{case}") for case in _X]
    return out


@pytest.mark.parametrize(("op", "case"), _gradgrad_params())
def test_reduction_gradgradcheck(op: str, case: str) -> None:
    assert gradgradcheck(_weighted(_REDUCE[op][0]), [_f64(_X[case])])


@pytest.mark.parametrize("device", _DEVICES)
def test_expanded_zero_cumprod_second_derivative_matches_reference(
    ref: ModuleType, device: str
) -> None:
    ar = ref.tensor([0.0], requires_grad=True)
    (gr1,) = ref.autograd.grad(ar.expand(3).cumprod(0).sum(), [ar], create_graph=True)
    (gr2,) = ref.autograd.grad(gr1, [ar])

    al = lucid.tensor([0.0], device=device, requires_grad=True)
    (gl1,) = lucid.autograd.grad(
        lucid.cumprod(al.expand(3), 0).sum(), [al], create_graph=True
    )
    (gl2,) = lucid.autograd.grad(gl1, [al])
    assert gl1.tolist() == gr1.tolist() == [1.0]
    assert gl2.tolist() == gr2.tolist() == [2.0]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    ("shape", "op"),
    [
        ((0, 3), "prod_dim0"),
        ((2, 0), "prod_dim1_keepdim"),
        ((0, 3), "cumprod_dim0"),
        ((2, 0), "cumprod_dim1"),
        ((2, 0), "prod"),
    ],
)
def test_empty_input_gradient_matches_reference(
    ref: ModuleType, device: str, shape: tuple[int, int], op: str
) -> None:
    ours, theirs = _REDUCE[op]
    xr = ref.zeros(shape, requires_grad=True)
    want = theirs(xr)
    want.backward(ref.ones(want.shape))

    xl = lucid.zeros(*shape, device=device, requires_grad=True)
    got = ours(xl)
    got.backward(lucid.ones(*got.shape, device=device))

    assert tuple(xl.grad.shape) == tuple(xr.grad.shape)
    _close(xl.grad.tolist(), xr.grad.tolist())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("value", [0.0, 3.0])
def test_zero_dim_prod_passes_the_gradient(device: str, value: float) -> None:
    x = lucid.tensor(value, device=device, requires_grad=True)
    x.prod().backward()
    assert x.grad.item() == 1.0


@pytest.mark.parametrize("device", _DEVICES)
def test_3d_middle_axis_prod_and_int32_scatter_prod_gradients(
    ref: ModuleType, device: str
) -> None:
    vals = [
        [[2.0, 0.0], [3.0, -2.0], [0.5, 4.0]],
        [[0.0, 1.5], [0.0, 0.0], [2.0, -3.0]],
    ]
    xr = ref.tensor(vals, requires_grad=True)
    wr = ref.tensor(_weights((2, 2)))
    (xr.prod(dim=1) * wr).sum().backward()

    xl = lucid.tensor(vals, device=device, requires_grad=True)
    wl = lucid.tensor(_weights((2, 2)), device=device)
    (xl.prod(dim=1) * wl).sum().backward()
    _close(xl.grad.tolist(), xr.grad.tolist())

    base, index, src = _SCATTER["base_and_src"]
    br = ref.tensor(base, requires_grad=True)
    sr = ref.tensor(src, requires_grad=True)
    br.scatter_reduce(1, ref.tensor(index), sr, "prod").backward(
        ref.tensor(_weights((2, 3)))
    )

    bl = lucid.tensor(base, device=device, requires_grad=True)
    sl = lucid.tensor(src, device=device, requires_grad=True)
    idx32 = lucid.tensor(index, dtype=lucid.int32, device=device)
    bl.scatter_reduce(1, idx32, sl, "prod").backward(
        lucid.tensor(_weights((2, 3)), device=device)
    )
    _close(bl.grad.tolist(), br.grad.tolist())
    _close(sl.grad.tolist(), sr.grad.tolist())

    neg_idx = lucid.tensor([[-3, -3, -2, -1], [-1, -1, -1, -3]], device=device)
    bn = lucid.tensor(base, device=device, requires_grad=True)
    sn = lucid.tensor(src, device=device, requires_grad=True)
    bn.scatter_reduce(1, neg_idx, sn, "prod").backward(
        lucid.tensor(_weights((2, 3)), device=device)
    )
    _close(bn.grad.tolist(), br.grad.tolist())
    _close(sn.grad.tolist(), sr.grad.tolist())
