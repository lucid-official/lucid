"""A half-precision BatchNorm in eval mode runs on the CPU (LCD-487).

The CPU's inference-mode BatchNorm kernel took float32 and float64 only, so
``model.half().eval()`` raised ``NotImplementedError`` on the CPU while Metal
and the reference ran it.  The global response norm beside it had the same
gap.  Both now widen the 16-bit floats to float32 at the door, as the
training kernel does (LCD-122): one float32 computation, one rounding on the
way out.  So the half forward *is* the float32 forward rounded once, and the
backward is off by at most the rounding of the statistic the forward saves
at the half's precision (``rstd``, ``Nx``).
"""

import functools
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.test._fixtures.devices import metal_available

_HALVES = {"float16": lucid.float16, "bfloat16": lucid.bfloat16}
#: Unit roundoff: half the distance from 1 to the next value.
_U = {"float16": 2.0**-11, "bfloat16": 2.0**-8}
_RANKS = {1: (6, 4, 5), 2: (2, 4, 3, 5), 3: (2, 4, 2, 3, 3)}
_DEVICES = [
    "cpu",
    pytest.param("metal", marks=pytest.mark.skipif(not metal_available(), reason="needs Metal")),
]
_BN_SLOTS = ("running_mean", "running_var", "weight", "bias")


def _on_grid(a: np.ndarray, dtype: str) -> np.ndarray:
    """``a`` rounded to the half, so float32 and the half see the same values."""
    return lucid.tensor(a.astype(np.float32)).to(_HALVES[dtype]).to(lucid.float32).numpy()


def _normal(seed: int, shape: tuple[int, ...], scale: float = 1.0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape) * scale


def _bn_inputs(rank: int, dtype: str) -> list[np.ndarray]:
    """x, running mean, running var, weight, bias, upstream gradient."""
    shape = _RANKS[rank]
    C = shape[1]
    arrays = [
        _normal(0, shape, 3.0),
        _normal(1, (C,)),
        np.abs(_normal(2, (C,))) + 0.25,
        _normal(3, (C,)),
        _normal(4, (C,)),
        _normal(5, shape),
    ]
    return [_on_grid(a, dtype) for a in arrays]


def _grn_inputs(dtype: str) -> list[np.ndarray]:
    """x, gamma, beta, upstream gradient.  x is large enough that a float16
    sum of squares would pass 65504."""
    shape = (2, 4, 3, 5)
    arrays = [_normal(0, shape, 40.0), _normal(1, (4,)), _normal(2, (4,)), _normal(3, shape)]
    return [_on_grid(a, dtype) for a in arrays]


def _run_bn(arrays: list[np.ndarray], dtype: object, device: str) -> list[np.ndarray]:
    x, rm, rv, w, b = (lucid.tensor(a, dtype=dtype, device=device) for a in arrays[:5])
    for t in (x, rm, w, b):
        t.requires_grad_()
    out = F.batch_norm(x, rm, rv, w, b, training=False, eps=1e-5)
    out.backward(lucid.tensor(arrays[5], dtype=dtype, device=device))
    return _collect(dtype, out, x, rm, w, b)


def _run_grn(arrays: list[np.ndarray], dtype: object, device: str) -> list[np.ndarray]:
    x, g, b = (lucid.tensor(a, dtype=dtype, device=device, requires_grad=True) for a in arrays[:3])
    out = _wrap(_C_engine.nn.global_response_norm(_unwrap(x), _unwrap(g), _unwrap(b), 1e-6))
    out.backward(lucid.tensor(arrays[3], dtype=dtype, device=device))
    return _collect(dtype, out, x, g, b)


def _collect(dtype: object, out: lucid.Tensor, *leaves: lucid.Tensor) -> list[np.ndarray]:
    """The output and every leaf's gradient, each checked to be in ``dtype``."""
    found = [out]
    for leaf in leaves:
        assert leaf.grad is not None
        found.append(leaf.grad)
    assert all(t.dtype == dtype for t in found)
    return [t.detach().to("cpu").to(lucid.float32).numpy() for t in found]


_Inputs = Callable[[str], list[np.ndarray]]
_Runner = Callable[[list[np.ndarray], object, str], list[np.ndarray]]
_KERNELS: dict[str, tuple[_Inputs, _Runner]] = {
    **{
        f"batch_norm{rank}d_eval": (functools.partial(_bn_inputs, rank), _run_bn) for rank in _RANKS
    },
    "global_response_norm": (_grn_inputs, _run_grn),
}


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", list(_HALVES))
@pytest.mark.parametrize("kernel", list(_KERNELS))
def test_the_half_kernel_is_the_float32_kernel_rounded_once(
    kernel: str, dtype: str, device: str
) -> None:
    make, run = _KERNELS[kernel]
    arrays = make(dtype)
    got = run(arrays, _HALVES[dtype], device)
    want = run(arrays, lucid.float32, device)
    np.testing.assert_array_equal(got[0], _on_grid(want[0], dtype), err_msg="output")
    # The backward reads the saved statistic at the half's precision, so it
    # may carry that rounding as well as its own.
    for i, (g, w) in enumerate(zip(got[1:], want[1:], strict=True), start=1):
        scale = max(1.0, float(np.abs(w).max()))
        np.testing.assert_allclose(
            g, w, rtol=2.5 * _U[dtype], atol=4 * _U[dtype] * scale, err_msg=f"gradient {i}"
        )


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", list(_HALVES))
@pytest.mark.parametrize("rank", list(_RANKS))
def test_half_batch_norm_eval_matches_the_reference(
    rank: int, dtype: str, device: str, ref: ModuleType
) -> None:
    x, *stats, grad = _bn_inputs(rank, dtype)
    C = x.shape[1]
    bn = getattr(nn, f"BatchNorm{rank}d")(C).eval()
    rbn = getattr(ref.nn, f"BatchNorm{rank}d")(C).eval()
    with lucid.no_grad():
        for name, value in zip(_BN_SLOTS, stats, strict=True):
            getattr(bn, name).copy_(lucid.tensor(value))
    with ref.no_grad():
        for name, value in zip(_BN_SLOTS, stats, strict=True):
            getattr(rbn, name).copy_(ref.tensor(value))
    bn = bn.to(device=device, dtype=_HALVES[dtype])
    rbn = rbn.to(getattr(ref, dtype))

    lx = lucid.tensor(x).to(_HALVES[dtype]).to(device).requires_grad_()
    out = bn(lx)
    out.backward(lucid.tensor(grad).to(_HALVES[dtype]).to(device))
    rx = ref.tensor(x).to(getattr(ref, dtype)).requires_grad_()
    rout = rbn(rx)
    rout.backward(ref.tensor(grad).to(getattr(ref, dtype)))

    got = [out, lx.grad, bn.weight.grad, bn.bias.grad]
    want = [rout, rx.grad, rbn.weight.grad, rbn.bias.grad]
    u = _U[dtype]
    names = ("output", "input grad", "weight grad", "bias grad")
    for what, g, r in zip(names, got, want, strict=True):
        assert g is not None and g.dtype == _HALVES[dtype], what
        g_np = g.detach().to("cpu").to(lucid.float32).numpy()
        r_np = r.detach().float().numpy()
        # One step of the half's grid: the reference rounds once, both CPU and
        # Metal at most twice (the saved rstd).
        np.testing.assert_allclose(g_np, r_np, rtol=2 * u, atol=1e-6, err_msg=what)
