"""``scatter_reduce``: one rule for the positions ``index`` never names (LCD-282).

``amax`` / ``amin`` / ``prod`` kept ``input``'s value where no ``index``
entry landed; ``sum`` / ``mean`` with ``include_self=False`` answered 0
there, and ``input`` received no gradient at all.  Every reduction now
keeps ``input`` (value and gradient) at an untouched position, and with
``include_self=False`` takes nothing from ``input`` where ``index`` lands.

Each case runs every reduction both ways of ``include_self`` on both
devices, values and gradients, against the reference.  The data hold no
``input`` value tying an ``amax`` / ``amin`` result under
``include_self=False`` — Lucid splits that gradient differently on
purpose, pinned by :func:`test_self_tie_does_not_take_a_share`.
"""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]
_REDUCTIONS = ["sum", "mean", "prod", "amax", "amin"]
_TOL = {"float32": 1e-5, "float16": 2e-2}

# (dim, input, index, src)
_CASES = {
    "some_untouched": (
        1,
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        [[0, 0], [0, 1]],
        [[1.5, 2.5], [-0.5, 3.0]],
    ),
    "all_touched": (
        1,
        [[1.0, 2.0], [3.0, 4.0]],
        [[0, 1, 1], [1, 0, 0]],
        [[5.0, 6.0, 7.0], [8.0, 9.0, -1.0]],
    ),
    "duplicates": (
        1,
        [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
        [[2, 2, 2], [0, 0, 1]],
        [[0.5, -1.5, 2.5], [1.5, 2.0, -0.5]],
    ),
    "leading_dim": (
        0,
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[2, 2], [2, 0]],
        [[-1.0, 0.5], [2.0, 1.5]],
    ),
}


def _cases():
    for device in _DEVICES:
        for dtype in _TOL:
            for reduce in _REDUCTIONS:
                for include_self in (True, False):
                    for case in _CASES:
                        yield pytest.param(
                            device,
                            dtype,
                            reduce,
                            include_self,
                            case,
                            id=f"{device}-{dtype}-{reduce}-self{int(include_self)}-{case}",
                        )


def _close(got, want, dtype):
    np.testing.assert_allclose(
        got.astype(np.float64), want.astype(np.float64), rtol=_TOL[dtype], atol=_TOL[dtype]
    )


@pytest.mark.parametrize(("device", "dtype", "reduce", "include_self", "case"), list(_cases()))
def test_matches_reference(ref, device, dtype, reduce, include_self, case):
    dim, x, index, src = _CASES[case]
    x_np, src_np = np.array(x, dtype=dtype), np.array(src, dtype=dtype)
    index_np = np.array(index, dtype=np.int64)

    xr = ref.tensor(x_np, requires_grad=True)
    sr = ref.tensor(src_np, requires_grad=True)
    want = xr.scatter_reduce(dim, ref.tensor(index_np), sr, reduce, include_self=include_self)
    grad_np = (np.arange(want.numel()).reshape(want.shape) + 1).astype(dtype)
    want.backward(ref.tensor(grad_np))

    xl = lucid.tensor(x_np, device=device, requires_grad=True)
    sl = lucid.tensor(src_np, device=device, requires_grad=True)
    got = lucid.scatter_reduce(
        xl, dim, lucid.tensor(index_np, device=device), sl, reduce, include_self=include_self
    )
    got.backward(lucid.tensor(grad_np, device=device))

    assert got.dtype == xl.dtype
    _close(got.numpy(), want.detach().numpy(), dtype)
    _close(xl.grad.numpy(), xr.grad.numpy(), dtype)
    _close(sl.grad.numpy(), sr.grad.numpy(), dtype)


def _integer_cases():
    for device in _DEVICES:
        for dtype in ("int32", "int64"):
            for reduce in ("sum", "mean", "amax", "amin"):
                marks = ()
                # MLX scatters no int64 operand; the CPU runs every integer.
                if reduce in ("amax", "amin") and device == "metal" and dtype == "int64":
                    marks = (
                        pytest.mark.xfail(
                            raises=ValueError,
                            strict=True,
                            reason="no int64 scatter_amax/amin kernel on Metal",
                        ),
                    )
                for include_self in (True, False):
                    yield pytest.param(
                        device,
                        dtype,
                        reduce,
                        include_self,
                        marks=marks,
                        id=f"{device}-{dtype}-{reduce}-self{int(include_self)}",
                    )


@pytest.mark.parametrize(("device", "dtype", "reduce", "include_self"), list(_integer_cases()))
def test_integer_input(ref, device, dtype, reduce, include_self):
    # Negative totals tell a floored mean from a truncated one.
    x = np.array([[1, -2, 3, 4], [5, 6, -7, 8]], dtype=dtype)
    index = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int64)
    src = np.array([[100, 101, -9], [-50, 61, 2]], dtype=dtype)
    want = ref.tensor(x).scatter_reduce(
        1, ref.tensor(index), ref.tensor(src), reduce, include_self=include_self
    )
    got = lucid.scatter_reduce(
        lucid.tensor(x, device=device),
        1,
        lucid.tensor(index, device=device),
        lucid.tensor(src, device=device),
        reduce,
        include_self=include_self,
    )
    assert str(got.dtype) == f"lucid.{dtype}"
    np.testing.assert_array_equal(got.numpy(), want.numpy())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("reduce", ["amax", "amin"])
def test_self_tie_does_not_take_a_share(device, reduce):
    # input[0] equals the result there, but include_self=False leaves it out
    # of the reduction: the two tied src values split the gradient in half.
    # The reference counts input[0] as a third tie and then drops its share.
    x = lucid.tensor([[3.0, 1.0, 2.0]], device=device, requires_grad=True)
    src = lucid.tensor([[3.0, 3.0, 5.0]], device=device, requires_grad=True)
    index = lucid.tensor([[0, 0, 1]], device=device)
    out = lucid.scatter_reduce(x, 1, index, src, reduce, include_self=False)
    out.backward(lucid.tensor([[6.0, 4.0, 2.0]], device=device))
    np.testing.assert_array_equal(out.numpy(), [[3.0, 5.0, 2.0]])
    np.testing.assert_array_equal(x.grad.numpy(), [[0.0, 0.0, 2.0]])
    np.testing.assert_array_equal(src.grad.numpy(), [[3.0, 3.0, 4.0]])


def test_unknown_reduce_is_refused():
    x = lucid.zeros(2, 3)
    with pytest.raises(ValueError, match="unknown reduce"):
        lucid.scatter_reduce(x, 1, lucid.tensor([[0], [1]]), lucid.ones(2, 1), "median")
