"""Dtype contract for distance, similarity, and normalization ops (LCD-252).

* ``cdist``, ``F.pdist``, and ``F.cosine_similarity`` (plus
  ``nn.CosineSimilarity``) are defined on real floating-point vectors only.
  Metal previously ran their real formulas on complex inputs and returned
  complex values, while the CPU failed later inside ``matmul`` or
  ``lp_normalize``.  Each entry point now refuses complex and discrete
  inputs at the boundary with ``DtypeMismatch`` (both ``TypeError`` and
  ``RuntimeError``), matching the reference.
* ``F.pairwise_distance`` (plus ``nn.PairwiseDistance``) and ``F.normalize``
  are defined on complex inputs as well as real ones via the real magnitude
  :math:`|x_i|`.  Single-axis ``F.normalize`` on CPU previously failed in
  ``lp_normalize_forward: dtype``, and on Metal its backward ignored the
  Wirtinger real projection; routing complex ``lp_normalize`` through
  ``abs`` and the real :math:`L_p` norm matches the reference on both
  devices in forward and backward.
"""

import math
from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

_COMPLEX_X = [[1 + 1j, 2 + 0j], [0 + 1j, 3 - 1j]]
_COMPLEX_Y = [[2 - 1j, 0 + 1j], [1 + 0j, 1 + 2j]]
_REAL_X = [[1.0, 2.0], [-0.5, 3.0]]
_REAL_Y = [[2.0, -1.0], [1.0, 1.5]]
_GRAD_OUT = [[0.5 - 1.0j, -0.25 + 0.5j], [1.5 + 0.2j, -0.8 - 0.4j]]


def _skip_metal_64bit(device: str, dtype: str) -> None:
    if device == "metal" and dtype in ("float64", "complex128"):
        pytest.skip(f"Metal does not support {dtype}")


def _ref_ops(
    m: ModuleType,
) -> dict[str, Callable[[object, object], object]]:
    functional = m.nn.functional if m is not lucid else F
    mod = m.nn if m is not lucid else nn
    return {
        "cdist-p2": lambda x, y: m.cdist(x, y, p=2.0),
        "cdist-p1": lambda x, y: m.cdist(x, y, p=1.0),
        "cdist-p0": lambda x, y: m.cdist(x, y, p=0.0),
        "cdist-pinf": lambda x, y: m.cdist(x, y, p=math.inf),
        "pdist": lambda x, _y: functional.pdist(x, p=2.0),
        "cosine_similarity": lambda x, y: functional.cosine_similarity(x, y, dim=1),
        "CosineSimilarity": lambda x, y: mod.CosineSimilarity(dim=1)(x, y),
    }


_REAL_ONLY_IDS = list(_ref_ops(lucid))


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("op_id", _REAL_ONLY_IDS)
def test_distance_and_cosine_refuse_complex_inputs(
    ref: ModuleType, device: str, dtype: str, op_id: str
) -> None:
    _skip_metal_64bit(device, dtype)
    rx = ref.tensor(_COMPLEX_X, dtype=getattr(ref, dtype))
    ry = ref.tensor(_COMPLEX_Y, dtype=getattr(ref, dtype))
    with pytest.raises((RuntimeError, TypeError)):
        _ref_ops(ref)[op_id](rx, ry)

    lx = lucid.tensor(_COMPLEX_X, dtype=getattr(lucid, dtype), device=device)
    ly = lucid.tensor(_COMPLEX_Y, dtype=getattr(lucid, dtype), device=device)
    expected_op = (
        "cosine_similarity"
        if op_id in ("cosine_similarity", "CosineSimilarity")
        else op_id.split("-")[0]
    )
    with pytest.raises(
        _C_engine.DtypeMismatch,
        match=rf"^{expected_op}: expected a floating point tensor",
    ) as exc_info:
        _ref_ops(lucid)[op_id](lx, ly)
    assert isinstance(exc_info.value, TypeError)
    assert isinstance(exc_info.value, RuntimeError)


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["complex64", "int64", "bool"])
@pytest.mark.parametrize("n_rows", [0, 1])
def test_pdist_short_circuit_rows_still_refuse_non_floating_dtype(
    ref: ModuleType, device: str, dtype: str, n_rows: int
) -> None:
    rx = ref.zeros((n_rows, 2), dtype=getattr(ref, dtype))
    with pytest.raises((RuntimeError, TypeError)):
        ref.nn.functional.pdist(rx)

    lx = lucid.zeros((n_rows, 2), dtype=getattr(lucid, dtype), device=device)
    with pytest.raises(
        _C_engine.DtypeMismatch,
        match=r"^pdist: expected a floating point tensor",
    ):
        F.pdist(lx)


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("op_id", _REAL_ONLY_IDS)
def test_distance_and_cosine_accept_real_floating_inputs(
    ref: ModuleType, device: str, dtype: str, op_id: str
) -> None:
    _skip_metal_64bit(device, dtype)
    rx = ref.tensor(_REAL_X, dtype=getattr(ref, dtype))
    ry = ref.tensor(_REAL_Y, dtype=getattr(ref, dtype))
    want = _ref_ops(ref)[op_id](rx, ry)

    lx = lucid.tensor(_REAL_X, dtype=getattr(lucid, dtype), device=device)
    ly = lucid.tensor(_REAL_Y, dtype=getattr(lucid, dtype), device=device)
    got = _ref_ops(lucid)[op_id](lx, ly)
    assert isinstance(got, lucid.Tensor)
    assert str(got.dtype).rsplit(".", 1)[-1] == dtype
    np.testing.assert_allclose(got.numpy(), want.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("p", [2.0, 1.0, math.inf, 1.5], ids=str)
@pytest.mark.parametrize("via_module", [False, True], ids=["functional", "module"])
def test_pairwise_distance_real_and_complex_parity(
    ref: ModuleType, device: str, dtype: str, p: float, via_module: bool
) -> None:
    _skip_metal_64bit(device, dtype)
    is_cplx = dtype.startswith("complex")
    raw_x = _COMPLEX_X if is_cplx else _REAL_X
    raw_y = _COMPLEX_Y if is_cplx else _REAL_Y

    rx = ref.tensor(raw_x, dtype=getattr(ref, dtype), requires_grad=True)
    ry = ref.tensor(raw_y, dtype=getattr(ref, dtype), requires_grad=True)
    if via_module:
        want = ref.nn.PairwiseDistance(p=p)(rx, ry)
    else:
        want = ref.nn.functional.pairwise_distance(rx, ry, p=p)
    want.sum().backward()

    lx = lucid.tensor(raw_x, dtype=getattr(lucid, dtype), device=device, requires_grad=True)
    ly = lucid.tensor(raw_y, dtype=getattr(lucid, dtype), device=device, requires_grad=True)
    if via_module:
        got = nn.PairwiseDistance(p=p)(lx, ly)
    else:
        got = F.pairwise_distance(lx, ly, p=p)
    got.sum().backward()

    expected_real_dt = {"complex64": "float32", "complex128": "float64"}.get(dtype, dtype)
    assert str(got.dtype).rsplit(".", 1)[-1] == expected_real_dt
    np.testing.assert_allclose(got.numpy(), want.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)
    assert lx.grad is not None and ly.grad is not None
    np.testing.assert_allclose(
        lx.grad.numpy(), rx.grad.detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        ly.grad.numpy(), ry.grad.detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("p", [2.0, 1.0, math.inf, 3.0], ids=str)
@pytest.mark.parametrize("dim", [1, 0, -1, (0, 1), None], ids=str)
def test_normalize_real_and_complex_parity_forward_and_backward(
    ref: ModuleType,
    device: str,
    dtype: str,
    p: float,
    dim: int | tuple[int, int] | None,
) -> None:
    _skip_metal_64bit(device, dtype)
    is_cplx = dtype.startswith("complex")
    raw_x = _COMPLEX_X if is_cplx else _REAL_X
    raw_g = _GRAD_OUT if is_cplx else _REAL_Y

    rx = ref.tensor(raw_x, dtype=getattr(ref, dtype), requires_grad=True)
    rg = ref.tensor(raw_g, dtype=getattr(ref, dtype))
    ref_dim: int | tuple[int, int] = (0, 1) if dim is None else dim
    want = ref.nn.functional.normalize(rx, p=p, dim=ref_dim)
    if is_cplx:
        ref.real(want * rg.conj()).sum().backward()
    else:
        (want * rg).sum().backward()

    lx = lucid.tensor(raw_x, dtype=getattr(lucid, dtype), device=device, requires_grad=True)
    lg = lucid.tensor(raw_g, dtype=getattr(lucid, dtype), device=device)
    got = F.normalize(lx, p=p, dim=dim)
    if is_cplx:
        lucid.real(got * lucid.conj(lg)).sum().backward()
    else:
        (got * lg).sum().backward()

    assert str(got.dtype).rsplit(".", 1)[-1] == dtype
    np.testing.assert_allclose(got.numpy(), want.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)
    assert lx.grad is not None
    np.testing.assert_allclose(
        lx.grad.numpy(), rx.grad.detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parity
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_normalize_complex_zero_row_and_zero_entry_parity(
    ref: ModuleType, device: str, dtype: str
) -> None:
    _skip_metal_64bit(device, dtype)
    data = [[0 + 0j, 3 + 4j], [0 + 0j, 0 + 0j]]
    rx = ref.tensor(data, dtype=getattr(ref, dtype), requires_grad=True)
    want = ref.nn.functional.normalize(rx, p=2.0, dim=1)
    ref.real(want).sum().backward()

    lx = lucid.tensor(data, dtype=getattr(lucid, dtype), device=device, requires_grad=True)
    got = F.normalize(lx, p=2.0, dim=1)
    lucid.real(got).sum().backward()

    np.testing.assert_allclose(got.numpy(), want.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)
    assert lx.grad is not None
    np.testing.assert_allclose(
        lx.grad.numpy(), rx.grad.detach().cpu().numpy(), rtol=1e-5, atol=1e-5
    )
