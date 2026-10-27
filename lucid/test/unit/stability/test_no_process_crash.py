"""Inputs that used to kill the interpreter now raise or answer.

The cases here ended the Python process rather than raising — before the
fix, 15 of them on the CPU and 44 on Metal — or sit on the same call sites
as one that did.  No in-process test could see such a failure: the test
runner died with it.  So each case runs in a child interpreter and the
parent asserts the child survived.

Four mechanisms were behind them:

* **Accelerate's BLAS error handler exits.**  A row-major GEMM over an
  empty extent naturally passes that extent as a leading dimension, and
  BLAS requires every leading dimension to be at least 1.  Accelerate
  prints ``BLAS error: Parameter number 14 passed to cblas_sgemm had an
  invalid value`` and calls ``exit(255)``.  An attention with no keys was
  enough.  The BLAS wrappers now answer empty extents themselves.

* **MLX's rank-5 convolution segfaults on an empty operand** — batch 0, no
  output channels, a zero spatial extent.  The Metal convolution entry
  points now answer those cases without calling MLX.

* **MLX's CPU-stream linalg throws from a worker thread.**  A failing
  LAPACK call inside an MLX primitive (``getrf`` on a singular matrix,
  ``gesvdx`` or ``geev`` on a NaN) is a C++ exception on the stream's
  worker thread, where nothing catches it, so the process ends in
  ``std::terminate`` — SIGABRT, whichever later call forced the
  evaluation.  The Metal linalg kernels now keep such inputs away from
  those routines: a singular matrix raises ``LucidError`` as it does on
  the CPU, a NaN matrix gets the reference's NaN decomposition.

* **MLX segfaults on ``contiguous()`` of some empty results** — a matmul
  with an empty extent, an attention with no queries.  The Metal backend
  wraps nearly every result as ``contiguous(op(...))``, so an empty matmul
  gradient was enough.  Empty results are now replaced by fresh empty
  arrays where the backend takes them over.

The sweep around the reported cases puts an empty extent on every GEMM and
convolution call site (matmul and its fused-add relatives, linear,
attention, the recurrent layers, conv and transposed conv of every rank,
the ODE combination kernels) and every linalg op, forward and backward, on
both devices.

How it runs: one child interpreter per device runs every case in order,
printing a marker before and after each.  A child that dies is blamed on the
case it had begun and is restarted after it, so one crash does not hide the
next.  Cases marked ``SURVIVE`` only have to come back — a Python exception
is acceptable there because the reference raises too, or because the
answer is pinned elsewhere; ``OK`` cases assert their result in the child.
"""

import json
import os
import signal
import subprocess
import sys

import pytest

pytestmark = pytest.mark.stability

OK = "ok"
SURVIVE = "survive"

# ── the cases ─────────────────────────────────────────────────────────────────
#
# Each is (name, code, expectation).  ``code`` runs in the child with the
# helpers from ``_PRELUDE`` in scope and ``D`` naming the device.  It must
# force evaluation (``N(...)``) — Metal is lazy, and a lazy failure would
# otherwise surface after the case has been reported as done.

_REPORTED = [
    # CHA-125
    (
        "sdpa_no_keys",
        "y = F.scaled_dot_product_attention(R(2, 4, 3, 8), R(2, 4, 0, 8), R(2, 4, 0, 8))\n"
        "zeros(y, (2, 4, 3, 8))",
        OK,
    ),
    (
        "conv3d_batch0_reported",
        "y = F.conv3d(Z(0, 4, 5, 6, 4), Z(6, 4, 3, 3, 3))\nshape(y, (0, 6, 3, 4, 2))",
        OK,
    ),
    (
        "conv_transpose3d_batch0_reported",
        "y = F.conv_transpose3d(Z(0, 4, 5, 6, 4), Z(4, 6, 3, 3, 3))\nshape(y, (0, 6, 7, 8, 6))",
        OK,
    ),
    (
        "Conv3d_module_batch0",
        "m = nn.Conv3d(4, 6, 3).to(D)\nshape(m(Z(0, 4, 5, 6, 4)), (0, 6, 3, 4, 2))",
        OK,
    ),
    # CHA-140 — singular / NaN through MLX's CPU-stream linalg.
    ("inv_singular", "singular(lambda: L.inv(SING))", OK),
    ("inv_ex_singular", "inv, info = L.inv_ex(SING)\nN(inv)\nassert int(info.item()) != 0", OK),
    ("solve_singular", "singular(lambda: L.solve(SING, A([[1.0], [2.0]])))", OK),
    (
        "solve_ex_singular",
        "x, info = L.solve_ex(SING, A([[1.0], [2.0]]))\nN(x)\nassert int(info.item()) != 0",
        OK,
    ),
    ("matrix_power_negative_singular", "singular(lambda: L.matrix_power(SING, -1))", OK),
    ("cond_p1_singular", "N(raises_or_value(lambda: L.cond(SING, 1)))", SURVIVE),
    (
        "det_backward_singular",
        "x = A([[1.0, 2.0], [2.0, 4.0]], grad=True)\n"
        "N(raises_or_value(lambda: (L.det(x).backward(), x.grad)[1]))",
        SURVIVE,
    ),
    ("eig_nan", "nonfinite(lambda: L.eig(NANSQ))", OK),
    ("eigvals_nan", "nonfinite(lambda: L.eigvals(NANSQ))", OK),
    ("eig_inf", "nonfinite(lambda: L.eig(INFSQ))", OK),
    # CHA-41 — NaN through the SVD.
    ("svdvals_nan", "allnan(L.svdvals(NAN23), (2,))", OK),
    ("svdvals_inf", "allnan(L.svdvals(INF23), (2,))", OK),
    (
        "svd_nan",
        "u, s, vh = L.svd(NAN23)\nallnan(u, (2, 2))\nallnan(s, (2,))\nallnan(vh, (3, 3))",
        OK,
    ),
    ("norm_nuc_nan", "allnan(L.norm(NAN23, ord='nuc'), ())", OK),
    ("norm_2_nan", "allnan(L.norm(NAN23, ord=2), ())", OK),
    ("norm_minus2_nan", "allnan(L.norm(NAN23, ord=-2), ())", OK),
    ("matrix_norm_2_nan", "allnan(L.matrix_norm(NAN23, ord=2), ())", OK),
    ("cond_nan", "allnan(L.cond(NANSQ), ())", OK),
    ("matrix_rank_nan", "assert int(N(L.matrix_rank(NAN23))) == 0", OK),
    ("pinv_nan", "allnan(L.pinv(NAN23), (3, 2))", OK),
    (
        "svdvals_batch_one_nan",
        "s = N(L.svdvals(lucid.stack([lucid.eye(2, device=D), C(NAN, 2, 2)])))\n"
        "assert (s[0] == 1).all() and np.isnan(s[1]).all(), s",
        OK,
    ),
    (
        "pinv_batch_one_nan",
        "p = N(L.pinv(lucid.stack([lucid.eye(2, device=D), C(NAN, 2, 2)])))\n"
        "assert np.allclose(p[0], np.eye(2)) and np.isnan(p[1]).all(), p",
        OK,
    ),
    (
        "inv_batch_one_singular",
        "singular(lambda: L.inv(lucid.stack([lucid.eye(2, device=D), Z(2, 2)])))",
        OK,
    ),
]

# The singular / NaN fixes must not have changed a well-posed answer.
_STILL_RIGHT = [
    (
        "inv_values",
        "a = A([[4.0, 7.0, 2.0], [3.0, 6.0, 1.0], [2.0, 5.0, 3.0]])\n"
        "close(L.inv(a), np.linalg.inv(N(a).astype(np.float64)))",
        OK,
    ),
    (
        "inv_batched_pivoting",
        "a = R(5, 4, 4)\nclose(L.inv(a), np.linalg.inv(N(a).astype(np.float64)), 1e-3)",
        OK,
    ),
    (
        "solve_matrix_rhs",
        "a, b = R(3, 4, 4), R(3, 4, 2)\n"
        "close(L.solve(a, b), np.linalg.solve(N(a).astype(np.float64), N(b)), 1e-3)",
        OK,
    ),
    (
        "solve_vector_rhs",
        "close(L.solve(A([[2.0, 0.0], [0.0, 4.0]]), A([2.0, 4.0])), np.array([1.0, 1.0]))",
        OK,
    ),
    (
        "matrix_power_minus2",
        "a = A([[2.0, 1.0], [1.0, 3.0]])\n"
        "close(L.matrix_power(a, -2), np.linalg.matrix_power(np.linalg.inv(N(a)), 2))",
        OK,
    ),
    ("inv_nan_propagates", "assert np.isnan(N(L.inv(A([[NAN, 1.0], [1.0, 1.0]])))).all()", OK),
    (
        "svdvals_values",
        "a = R(3, 5)\nclose(L.svdvals(a), np.linalg.svd(N(a).astype(np.float64), compute_uv=False), 1e-4)",
        OK,
    ),
    (
        "pinv_values",
        "a = R(3, 5)\nclose(L.pinv(a), np.linalg.pinv(N(a).astype(np.float64)), 1e-3)",
        OK,
    ),
    (
        "inv_backward",
        "a = A([[4.0, 7.0], [2.0, 6.0]], grad=True)\nL.inv(a).sum().backward()\n"
        "inv = np.linalg.inv(N(a).astype(np.float64))\n"
        "close(a.grad, -(inv.T @ np.ones((2, 2)) @ inv.T), 1e-4)",
        OK,
    ),
]

# ── GEMM call sites: an empty M, N or K on every one ─────────────────────────

_GEMM = [
    ("matmul_M0", "shape(lucid.matmul(R(0, 3), R(3, 4)), (0, 4))", OK),
    ("matmul_N0", "shape(lucid.matmul(R(3, 4), R(4, 0)), (3, 0))", OK),
    ("matmul_K0", "zeros(lucid.matmul(R(3, 0), R(0, 4)), (3, 4))", OK),
    ("matmul_batch0", "shape(lucid.matmul(R(0, 3, 4), R(0, 4, 5)), (0, 3, 5))", OK),
    ("matmul_K0_broadcast", "zeros(lucid.matmul(R(2, 3, 0), R(0, 5)), (2, 3, 5))", OK),
    ("matmul_vector_K0", "zeros(lucid.matmul(R(3, 0), R(0)), (3,))", OK),
    ("mm_K0", "zeros(lucid.mm(R(3, 0), R(0, 4)), (3, 4))", OK),
    ("bmm_K0", "zeros(lucid.bmm(R(2, 3, 0), R(2, 0, 4)), (2, 3, 4))", OK),
    ("mv_K0", "zeros(lucid.mv(R(3, 0), R(0)), (3,))", OK),
    ("tensordot_K0", "zeros(lucid.tensordot(R(3, 0), R(0, 4), dims=1), (3, 4))", OK),
    ("einsum_K0", "zeros(lucid.einops.einsum('ik,kj->ij', R(3, 0), R(0, 4)), (3, 4))", OK),
    ("kron_empty", "shape(lucid.kron(R(0, 2), R(3, 3)), (0, 6))", OK),
    # The fused-add forms over an empty contraction leave ``beta * input``.
    # ``beta == 0`` over a NaN input only has to survive here: the reference
    # gives zeros, Lucid's composite still gives ``0 * NaN``.  (The BLAS
    # wrappers themselves overwrite with zeros — test_blas_empty_extents.cpp.)
    ("addmm_K0_beta1", "full(lucid.addmm(C(1.0, 3, 4), R(3, 0), R(0, 4)), (3, 4), 1.0)", OK),
    (
        "addmm_K0_beta2",
        "full(lucid.addmm(C(1.0, 3, 4), R(3, 0), R(0, 4), beta=2.0), (3, 4), 2.0)",
        OK,
    ),
    ("addmm_K0_beta0_nan", "N(lucid.addmm(C(NAN, 3, 4), R(3, 0), R(0, 4), beta=0.0))", SURVIVE),
    ("addmv_K0", "full(lucid.addmv(C(1.0, 3), R(3, 0), R(0)), (3,), 1.0)", OK),
    (
        "baddbmm_K0",
        "full(lucid.baddbmm(C(1.0, 2, 3, 4), R(2, 3, 0), R(2, 0, 4)), (2, 3, 4), 1.0)",
        OK,
    ),
    ("addbmm_K0", "full(lucid.addbmm(C(1.0, 3, 4), R(2, 3, 0), R(2, 0, 4)), (3, 4), 1.0)", OK),
    (
        "matmul_K0_backward",
        "a, b = R(3, 0, grad=True), R(0, 4, grad=True)\nlucid.matmul(a, b).sum().backward()\n"
        "shape(a.grad, (3, 0))\nshape(b.grad, (0, 4))",
        OK,
    ),
    (
        "matmul_M0_backward",
        "a, b = R(0, 3, grad=True), R(3, 4, grad=True)\nlucid.matmul(a, b).sum().backward()\n"
        "shape(a.grad, (0, 3))\nzeros(b.grad, (3, 4))",
        OK,
    ),
    (
        "matmul_N0_backward",
        "a, b = R(3, 4, grad=True), R(4, 0, grad=True)\nlucid.matmul(a, b).sum().backward()\n"
        "zeros(a.grad, (3, 4))\nshape(b.grad, (4, 0))",
        OK,
    ),
    ("linear_batch0", "shape(F.linear(R(0, 3), R(4, 3), R(4)), (0, 4))", OK),
    ("linear_in0", "full(F.linear(R(2, 0), R(4, 0), C(1.0, 4)), (2, 4), 1.0)", OK),
    ("linear_out0", "shape(F.linear(R(2, 3), R(0, 3)), (2, 0))", OK),
    (
        "linear_batch0_backward",
        "x, w = R(0, 3, grad=True), R(4, 3, grad=True)\nF.linear(x, w).sum().backward()\n"
        "shape(x.grad, (0, 3))\nzeros(w.grad, (4, 3))",
        OK,
    ),
    ("bilinear_batch0", "shape(F.bilinear(R(0, 3), R(0, 2), R(4, 3, 2)), (0, 4))", OK),
    ("bilinear_in0", "N(F.bilinear(R(2, 0), R(2, 2), R(4, 0, 2)))", SURVIVE),
    ("matrix_power_batch0", "shape(L.matrix_power(R(0, 3, 3), 3), (0, 3, 3))", OK),
    ("matrix_power_batch0_negative", "shape(L.matrix_power(R(0, 3, 3), -2), (0, 3, 3))", OK),
    ("multi_dot_K0", "zeros(L.multi_dot([R(3, 0), R(0, 4), R(4, 2)]), (3, 2))", OK),
]

_ATTENTION = [
    (
        "sdpa_no_queries",
        "shape(F.scaled_dot_product_attention(R(2, 0, 8), R(2, 5, 8), R(2, 5, 8)), (2, 0, 8))",
        OK,
    ),
    (
        "sdpa_batch0",
        "shape(F.scaled_dot_product_attention(R(0, 3, 8), R(0, 5, 8), R(0, 5, 8)), (0, 3, 8))",
        OK,
    ),
    (
        "sdpa_value_width0",
        "shape(F.scaled_dot_product_attention(R(2, 3, 8), R(2, 5, 8), R(2, 5, 0)), (2, 3, 0))",
        OK,
    ),
    (
        "sdpa_no_queries_4d",
        "shape(F.scaled_dot_product_attention(R(2, 4, 0, 8), R(2, 4, 5, 8), R(2, 4, 5, 8)), (2, 4, 0, 8))",
        OK,
    ),
    (
        "sdpa_batch0_backward",
        "q, k, v = R(0, 3, 8, grad=True), R(0, 5, 8, grad=True), R(0, 5, 8, grad=True)\n"
        "F.scaled_dot_product_attention(q, k, v).sum().backward()\n"
        "shape(q.grad, (0, 3, 8))\nshape(k.grad, (0, 5, 8))\nshape(v.grad, (0, 5, 8))",
        OK,
    ),
    (
        "sdpa_value_width0_backward",
        "q, k, v = R(2, 3, 8, grad=True), R(2, 5, 8, grad=True), R(2, 5, 0, grad=True)\n"
        "F.scaled_dot_product_attention(q, k, v).sum().backward()\n"
        "zeros(q.grad, (2, 3, 8))\nzeros(k.grad, (2, 5, 8))\nshape(v.grad, (2, 5, 0))",
        OK,
    ),
    (
        "sdpa_no_keys_causal",
        "zeros(F.scaled_dot_product_attention(R(2, 3, 8), R(2, 0, 8), R(2, 0, 8), is_causal=True), (2, 3, 8))",
        SURVIVE,
    ),
    (
        "sdpa_no_keys_backward",
        "q, k, v = R(2, 3, 8, grad=True), R(2, 0, 8, grad=True), R(2, 0, 8, grad=True)\n"
        "F.scaled_dot_product_attention(q, k, v).sum().backward()\n"
        "zeros(q.grad, (2, 3, 8))\nshape(k.grad, (2, 0, 8))\nshape(v.grad, (2, 0, 8))",
        OK,
    ),
    (
        "sdpa_no_queries_backward",
        "q, k, v = R(2, 0, 8, grad=True), R(2, 5, 8, grad=True), R(2, 5, 8, grad=True)\n"
        "F.scaled_dot_product_attention(q, k, v).sum().backward()\n"
        "shape(q.grad, (2, 0, 8))\nzeros(k.grad, (2, 5, 8))\nzeros(v.grad, (2, 5, 8))",
        OK,
    ),
    (
        "sdpa_no_keys_float64",
        "y = F.scaled_dot_product_attention(R(2, 3, 8, f64=True), R(2, 0, 8, f64=True), R(2, 0, 8, f64=True))\n"
        "zeros(y, (2, 3, 8))",
        OK,
    ),
    (
        "mha_empty_sequence",
        "m = nn.MultiheadAttention(8, 2, batch_first=True).to(D)\n"
        "shape(m(R(2, 0, 8), R(2, 0, 8), R(2, 0, 8))[0], (2, 0, 8))",
        OK,
    ),
    (
        "mha_no_keys",
        "m = nn.MultiheadAttention(8, 2, batch_first=True).to(D)\n"
        "N(m(R(2, 3, 8), R(2, 0, 8), R(2, 0, 8))[0])",
        SURVIVE,
    ),
]

_RECURRENT = []
for _cls in ("LSTM", "GRU", "RNN"):
    _RECURRENT += [
        (
            f"{_cls}_batch0",
            f"m = nn.{_cls}(4, 5, batch_first=True).to(D)\nshape(m(R(0, 3, 4))[0], (0, 3, 5))",
            OK,
        ),
        (
            f"{_cls}_batch0_backward",
            f"m = nn.{_cls}(4, 5, batch_first=True).to(D)\nx = R(0, 3, 4, grad=True)\n"
            "m(x)[0].sum().backward()\n"
            "for p in m.parameters():\n"
            "    assert p.grad is None or not np.abs(N(p.grad)).any()",
            SURVIVE,
        ),
        (
            f"{_cls}_seq0",
            f"m = nn.{_cls}(4, 5, batch_first=True).to(D)\nN(m(R(2, 0, 4))[0])",
            SURVIVE,
        ),
    ]
_RECURRENT += [
    (
        "LSTM_proj_batch0",
        "m = nn.LSTM(4, 5, batch_first=True, proj_size=3).to(D)\nshape(m(R(0, 3, 4))[0], (0, 3, 3))",
        OK,
    ),
    ("LSTMCell_batch0", "shape(nn.LSTMCell(4, 5).to(D)(R(0, 4))[0], (0, 5))", OK),
    ("GRUCell_batch0", "shape(nn.GRUCell(4, 5).to(D)(R(0, 4)), (0, 5))", OK),
]

_CONV = []
for _d in (1, 2, 3):
    _sp = ", ".join(["5"] * _d)
    _k = ", ".join(["3"] * _d)
    _o = ", ".join(["3"] * _d)
    _t = ", ".join(["7"] * _d)
    _CONV += [
        (
            f"conv{_d}d_batch0",
            f"shape(F.conv{_d}d(R(0, 4, {_sp}), R(6, 4, {_k})), (0, 6, {_o}))",
            OK,
        ),
        (
            f"conv{_d}d_in_channels0",
            f"full(F.conv{_d}d(R(2, 0, {_sp}), R(6, 0, {_k}), C(1.0, 6)), (2, 6, {_o}), 1.0)",
            SURVIVE,
        ),
        (f"conv{_d}d_out_channels0", f"N(F.conv{_d}d(R(2, 4, {_sp}), R(0, 4, {_k})))", SURVIVE),
        (
            f"conv{_d}d_batch0_grouped",
            f"shape(F.conv{_d}d(R(0, 4, {_sp}), R(6, 2, {_k}), groups=2), (0, 6, {_o}))",
            OK,
        ),
        (
            f"conv{_d}d_batch0_pointwise",
            f"shape(F.conv{_d}d(R(0, 4, {_sp}), R(6, 4, {', '.join(['1'] * _d)})), (0, 6, {_sp}))",
            OK,
        ),
        (
            f"conv{_d}d_batch0_backward",
            f"x, w, b = R(0, 4, {_sp}, grad=True), R(6, 4, {_k}, grad=True), R(6, grad=True)\n"
            f"F.conv{_d}d(x, w, b).sum().backward()\n"
            f"shape(x.grad, (0, 4, {_sp}))\nzeros(w.grad, (6, 4, {_k}))\nzeros(b.grad, (6,))",
            OK,
        ),
        (
            f"conv{_d}d_in_channels0_backward",
            f"x, w, b = R(2, 0, {_sp}, grad=True), R(6, 0, {_k}, grad=True), R(6, grad=True)\n"
            f"F.conv{_d}d(x, w, b).sum().backward()\n"
            f"shape(x.grad, (2, 0, {_sp}))\nshape(w.grad, (6, 0, {_k}))",
            SURVIVE,
        ),
        (
            f"conv{_d}d_zero_extent",
            f"N(F.conv{_d}d(R(2, 4, 0{', 5' * (_d - 1)}), R(6, 4, {_k}), padding=1))",
            SURVIVE,
        ),
        (
            f"conv_transpose{_d}d_batch0",
            f"shape(F.conv_transpose{_d}d(R(0, 4, {_sp}), R(4, 6, {_k})), (0, 6, {_t}))",
            OK,
        ),
        (
            f"conv_transpose{_d}d_out_channels0",
            f"N(F.conv_transpose{_d}d(R(2, 4, {_sp}), R(4, 0, {_k})))",
            SURVIVE,
        ),
        (
            f"conv_transpose{_d}d_in_channels0",
            f"N(F.conv_transpose{_d}d(R(2, 0, {_sp}), R(0, 6, {_k}), C(1.0, 6)))",
            SURVIVE,
        ),
        (
            f"conv_transpose{_d}d_batch0_grouped",
            f"shape(F.conv_transpose{_d}d(R(0, 4, {_sp}), R(4, 3, {_k}), groups=2), (0, 6, {_t}))",
            OK,
        ),
        (
            f"conv_transpose{_d}d_batch0_backward",
            f"x, w, b = R(0, 4, {_sp}, grad=True), R(4, 6, {_k}, grad=True), R(6, grad=True)\n"
            f"F.conv_transpose{_d}d(x, w, b).sum().backward()\n"
            f"shape(x.grad, (0, 4, {_sp}))\nzeros(w.grad, (4, 6, {_k}))\nzeros(b.grad, (6,))",
            OK,
        ),
    ]
_CONV += [
    ("unfold_batch0", "shape(F.unfold(R(0, 3, 5, 5), 2), (0, 12, 16))", OK),
    ("fold_batch0", "shape(F.fold(R(0, 12, 16), (5, 5), 2), (0, 3, 5, 5))", OK),
]

# The fused Runge-Kutta kernels accumulate through ``axpy``.
_ODE = [
    (
        "rk_combine_empty_state",
        "from lucid.diffeq import _fused\n"
        "shape(_fused.combine(Z(0), [Z(0), Z(0)], [0.5, 0.5], 0.1), (0,))",
        OK,
    ),
    (
        "rk_error_ratio_empty_state",
        "from lucid.diffeq import _fused\n"
        "_fused.error_ratio(Z(0), Z(0), [Z(0), Z(0)], [0.5, -0.5], 0.1, 1e-3, 1e-6)",
        SURVIVE,
    ),
    (
        "odeint_empty_state",
        "import lucid.diffeq as de\n"
        "N(de.odeint(lambda t, y: -y, Z(0), lucid.tensor([0.0, 1.0], device=D)))",
        SURVIVE,
    ),
]

# Every linalg op on an empty matrix and on an empty batch.
_LINALG_EMPTY = []
for _shape in ("(0, 0)", "(0, 3, 3)"):
    _tag = "0x0" if _shape == "(0, 0)" else "batch0"
    _sq = f"R{_shape}"
    _LINALG_EMPTY += [
        (f"inv_{_tag}", f"N(L.inv({_sq}))", OK),
        (f"det_{_tag}", f"N(L.det({_sq}))", OK),
        (f"slogdet_{_tag}", f"N(L.slogdet({_sq})[1])", OK),
        (f"cholesky_{_tag}", f"N(L.cholesky({_sq}))", OK),
        (f"eig_{_tag}", f"N(L.eig({_sq})[0])", OK),
        (f"eigvals_{_tag}", f"N(L.eigvals({_sq}))", OK),
        (f"eigh_{_tag}", f"N(L.eigh({_sq})[0])", OK),
        (f"eigvalsh_{_tag}", f"N(L.eigvalsh({_sq}))", OK),
        (f"svd_{_tag}", f"N(L.svd({_sq})[1])", OK),
        (f"svdvals_{_tag}", f"N(L.svdvals({_sq}))", OK),
        (f"qr_{_tag}", f"N(L.qr({_sq})[1])", OK),
        (f"pinv_{_tag}", f"N(L.pinv({_sq}))", OK),
        (f"matrix_rank_{_tag}", f"N(L.matrix_rank({_sq}))", OK),
        (f"matrix_exp_{_tag}", f"N(L.matrix_exp({_sq}))", OK),
        (f"norm_nuc_{_tag}", f"N(L.matrix_norm({_sq}, ord='nuc'))", SURVIVE),
        (f"norm_2_{_tag}", f"N(L.matrix_norm({_sq}, ord=2))", SURVIVE),
        (f"cond_{_tag}", f"N(L.cond({_sq}))", SURVIVE),
        (f"lu_factor_{_tag}", f"N(L.lu_factor({_sq})[0])", OK),
        (f"lu_{_tag}", f"N(L.lu({_sq})[2])", SURVIVE),
        (f"ldl_factor_{_tag}", f"N(L.ldl_factor({_sq})[0])", SURVIVE),
    ]
_LINALG_EMPTY += [
    (
        "svd_0x3",
        "u, s, vh = L.svd(R(0, 3))\nshape(u, (0, 0))\nshape(s, (0,))\nshape(vh, (3, 3))",
        OK,
    ),
    (
        "svd_3x0",
        "u, s, vh = L.svd(R(3, 0))\nshape(u, (3, 3))\nshape(s, (0,))\nshape(vh, (0, 0))",
        OK,
    ),
    ("qr_0x3", "N(L.qr(R(0, 3))[1])", OK),
    ("pinv_0x3", "shape(L.pinv(R(0, 3)), (3, 0))", OK),
    ("pinv_3x0", "shape(L.pinv(R(3, 0)), (0, 3))", OK),
    ("solve_batch0", "shape(L.solve(R(0, 3, 3), R(0, 3, 2)), (0, 3, 2))", OK),
    ("solve_no_rhs", "shape(L.solve(R(3, 3), R(3, 0)), (3, 0))", OK),
    ("solve_0x0", "shape(L.solve(R(0, 0), R(0, 2)), (0, 2))", OK),
    ("lstsq_batch0", "N(L.lstsq(R(0, 3, 3), R(0, 3, 2))[0])", SURVIVE),
    ("lstsq_no_rhs", "N(L.lstsq(R(3, 3), R(3, 0))[0])", SURVIVE),
    ("lstsq_0x3", "N(L.lstsq(R(0, 3), R(0, 2))[0])", SURVIVE),
    ("solve_triangular_no_rhs", "shape(L.solve_triangular(EYE3, R(3, 0), upper=True), (3, 0))", OK),
    (
        "solve_triangular_batch0",
        "shape(L.solve_triangular(R(0, 3, 3), R(0, 3, 2), upper=True), (0, 3, 2))",
        OK,
    ),
    (
        "lu_solve_no_rhs",
        "lu, piv = L.lu_factor(EYE3 * 2)\nshape(L.lu_solve(lu, piv, R(3, 0)), (3, 0))",
        OK,
    ),
    ("householder_product_0x0", "N(L.householder_product(R(0, 0), R(0)))", SURVIVE),
    ("vector_norm_empty", "N(L.vector_norm(R(0)))", OK),
    ("norm_empty_matrix", "N(L.norm(R(0, 3)))", OK),
]

CASES = _REPORTED + _STILL_RIGHT + _GEMM + _ATTENTION + _RECURRENT + _CONV + _ODE + _LINALG_EMPTY

# ── the child ─────────────────────────────────────────────────────────────────

_PRELUDE = """
import sys
import numpy as np
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.linalg as L

D = sys.argv[1]
NAN = float("nan")
INF = float("inf")
lucid.manual_seed(0)


def R(*shape, grad=False, f64=False):
    dt = lucid.float64 if f64 and D == "cpu" else lucid.float32
    return lucid.randn(*shape, dtype=dt, device=D, requires_grad=grad)


def Z(*shape):
    return lucid.zeros(*shape, device=D)


def C(value, *shape):
    return lucid.full(shape, value, device=D)


def A(rows, grad=False):
    return lucid.tensor(rows, device=D, requires_grad=grad)


def N(t):
    # ``.numpy()`` forces Metal's lazy graph; a failure has to happen here,
    # inside the case, not at some later case's evaluation.
    return np.asarray(t.numpy())


def shape(t, want):
    got = tuple(N(t).shape)
    assert got == tuple(want), f"shape {got}, expected {tuple(want)}"


def full(t, want, value):
    shape(t, want)
    arr = N(t)
    assert (arr == value).all(), f"expected all {value}, got {arr.ravel()[:6]}"


def zeros(t, want):
    full(t, want, 0.0)


def allnan(t, want):
    shape(t, want)
    arr = N(t)
    assert np.isnan(arr).all(), f"expected all NaN, got {arr.ravel()[:6]}"


def close(t, want, tol=1e-5):
    got = N(t).astype(np.float64)
    assert got.shape == want.shape, f"shape {got.shape}, expected {want.shape}"
    assert np.allclose(got, want, rtol=tol, atol=tol), f"{got.ravel()[:6]} vs {want.ravel()[:6]}"


def _raise_with(fn, needle):
    try:
        out = fn()
        for t in out if isinstance(out, (tuple, list)) else [out]:
            N(t)
    except RuntimeError as e:
        assert needle in str(e), f"wrong error: {e}"
        return
    raise AssertionError("did not raise")


def singular(fn):
    # The CPU stream's error, word for word apart from ``info``.
    _raise_with(fn, "because the input matrix is singular")


def nonfinite(fn):
    _raise_with(fn, "infs or NaNs")


def raises_or_value(fn):
    try:
        return fn()
    except RuntimeError:
        return lucid.zeros(1)


SING = A([[1.0, 2.0], [2.0, 4.0]])
NANSQ = A([[1.0, NAN], [2.0, 4.0]])
INFSQ = A([[1.0, INF], [2.0, 4.0]])
NAN23 = A([[1.0, NAN, 2.0], [3.0, 4.0, 5.0]])
INF23 = A([[1.0, INF, 2.0], [3.0, 4.0, 5.0]])
EYE3 = lucid.eye(3, device=D)
"""

_CHILD = _PRELUDE + """
import json
import traceback

_cases = json.loads(sys.stdin.read())
_scope = dict(globals())
for _name, _code in _cases:
    print("@@BEGIN " + _name, flush=True)
    try:
        exec(_code, dict(_scope))
        _verdict = ["ok", ""]
    except AssertionError as _e:
        _verdict = ["assert", str(_e) or traceback.format_exc(limit=2)]
    except Exception as _e:
        _verdict = ["raised", type(_e).__name__ + ": " + str(_e)[:300]]
    print("@@END " + _name + " " + json.dumps(_verdict), flush=True)
"""

_CHILD_TIMEOUT_S = 600


def _describe_exit(code: int) -> str:
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    if code == 255:
        return "exit 255 (Accelerate's BLAS error handler exits with 255)"
    return f"exit {code}"


def _run_cases(device: str) -> dict[str, list[str]]:
    """Run every case for ``device``; a dead child is restarted past its case."""
    results: dict[str, list[str]] = {}
    pending = [(name, code) for name, code, _ in CASES]
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    while pending:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, device],
            input=json.dumps(pending),
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT_S,
            env=env,
        )
        current = None
        for line in done.stdout.splitlines():
            if line.startswith("@@BEGIN "):
                current = line[len("@@BEGIN ") :]
            elif line.startswith("@@END "):
                name, _, verdict = line[len("@@END ") :].partition(" ")
                results[name] = json.loads(verdict)
                current = None
        if done.returncode == 0:
            break
        tail = done.stderr.strip()[-600:]
        if current is None:
            # Died outside any case — the prelude itself.  Blame everything left.
            for name, _ in pending:
                results.setdefault(name, ["crash", f"{_describe_exit(done.returncode)}: {tail}"])
            break
        results[current] = ["crash", f"{_describe_exit(done.returncode)}: {tail}"]
        pending = [(n, c) for n, c in pending if n not in results]
    return results


_RESULTS: dict[str, dict[str, list[str]]] = {}


def _result(device: str, name: str) -> list[str]:
    if device not in _RESULTS:
        _RESULTS[device] = _run_cases(device)
    return _RESULTS[device].get(name, ["missing", "the child never reached this case"])


@pytest.mark.parametrize("name,expect", [(n, e) for n, _, e in CASES], ids=[n for n, _, _ in CASES])
def test_the_process_survives(device: str, name: str, expect: str) -> None:
    status, detail = _result(device, name)
    assert status != "crash", f"{name} on {device} killed the interpreter — {detail}"
    if expect == OK:
        assert status == "ok", f"{name} on {device}: {status}: {detail}"
    else:
        assert status in ("ok", "raised"), f"{name} on {device}: {status}: {detail}"


def test_case_names_are_unique() -> None:
    names = [n for n, _, _ in CASES]
    assert len(names) == len(set(names))


if __name__ == "__main__":
    # ``python -m pytest`` is the entry point; this guard only keeps a direct
    # ``python test_no_process_crash.py`` from spawning children at import.
    sys.exit(pytest.main([__file__, "-q"]))
