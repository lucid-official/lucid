"""OpInfo catalog for multi-axis reductions, prefix scans, and indexing/view ops."""

from collections.abc import Sequence
import numpy as np

import lucid
import lucid.linalg as llinalg
from lucid.test.parity.opinfo._core import OpInfo, SampleInput, index_arg, mask_arg, tensor_arg


def _reduction_multi_axis_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(2026)
    x_2d = rng.uniform(-1.8, 2.2, size=(3, 4))
    x_3d = rng.uniform(-1.5, 1.9, size=(2, 3, 2))
    return (
        SampleInput("all_dims", (tensor_arg(x_2d),)),
        SampleInput("dim0_drop", (tensor_arg(x_2d),), {"dim": 0, "keepdim": False}),
        SampleInput("dim_neg1_keep", (tensor_arg(x_2d),), {"dim": -1, "keepdim": True}),
        SampleInput("tuple_dims_drop", (tensor_arg(x_3d),), {"dim": (0, 2), "keepdim": False}),
        SampleInput("tuple_dims_keep", (tensor_arg(x_3d),), {"dim": (0, 2), "keepdim": True}),
    )


def _single_axis_reduction_samples() -> Sequence[SampleInput]:
    x_2d = np.array([[0.4, -1.2, 2.1, 0.9], [1.7, 0.3, -0.8, 1.4], [-0.5, 1.9, 0.6, -1.1]])
    x_3d = np.arange(1, 13, dtype=np.float64).reshape(2, 3, 2) * 0.25 - 1.5
    return (
        SampleInput("dim0_drop", (tensor_arg(x_2d),), {"dim": 0, "keepdim": False}),
        SampleInput("dim_neg1_keep", (tensor_arg(x_2d),), {"dim": -1, "keepdim": True}),
        SampleInput("dim1_3d", (tensor_arg(x_3d),), {"dim": 1, "keepdim": False}),
    )


def _var_std_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(303)
    x_2d = rng.uniform(-2.0, 2.0, size=(3, 4))
    x_3d = rng.uniform(-1.5, 1.5, size=(2, 3, 2))
    return (
        SampleInput("correction1_all", (tensor_arg(x_2d),), {"correction": 1}),
        SampleInput("correction0_dim0", (tensor_arg(x_2d),), {"dim": 0, "correction": 0}),
        SampleInput(
            "correction1_neg1_keep",
            (tensor_arg(x_2d),),
            {"dim": -1, "correction": 1, "keepdim": True},
        ),
        SampleInput(
            "tuple_dims_correction1",
            (tensor_arg(x_3d),),
            {"dim": (0, 2), "correction": 1, "keepdim": False},
        ),
    )


def _scan_samples() -> Sequence[SampleInput]:
    x_2d = np.array([[0.5, 1.2, -0.7, 0.9], [1.1, -0.4, 0.8, 1.3], [-0.6, 0.7, 1.4, -0.5]])
    x_3d = np.linspace(0.4, 1.6, num=12, dtype=np.float64).reshape(2, 3, 2)
    return (
        SampleInput("dim0_2d", (tensor_arg(x_2d),), {"dim": 0}),
        SampleInput("dim_neg1_2d", (tensor_arg(x_2d),), {"dim": -1}),
        SampleInput("dim1_3d", (tensor_arg(x_3d),), {"dim": 1}),
    )


def _vector_norm_samples() -> Sequence[SampleInput]:
    x_2d = np.array([[0.6, -1.2, 1.5], [-0.9, 0.8, 1.1], [1.4, -0.5, 0.7]], dtype=np.float64)
    return (
        SampleInput("l2_all", (tensor_arg(x_2d),), {"ord": 2.0}),
        SampleInput("l2_dim1_keep", (tensor_arg(x_2d),), {"ord": 2.0, "dim": -1, "keepdim": True}),
        SampleInput("l1_dim0", (tensor_arg(x_2d),), {"ord": 1.0, "dim": 0}),
        SampleInput("linf_dim1", (tensor_arg(x_2d),), {"ord": float("inf"), "dim": 1}),
    )


def _matrix_norm_samples() -> Sequence[SampleInput]:
    x_2d = np.array([[1.2, -0.5, 0.8], [0.4, 1.5, -0.7], [-0.9, 0.6, 1.1]], dtype=np.float64)
    x_3d = np.stack([x_2d, x_2d * 0.7 + 0.2], axis=0)
    return (
        SampleInput("fro_2d", (tensor_arg(x_2d),), {"ord": "fro"}),
        SampleInput("fro_3d_keep", (tensor_arg(x_3d),), {"ord": "fro", "keepdim": True}),
        SampleInput(
            "ord1_2d",
            (tensor_arg(x_2d),),
            {"ord": 1, " supports_backward": False} if False else {"ord": 1},
            supports_backward=False,
            supports_gradcheck=False,
        ),
        SampleInput(
            "ordinf_2d",
            (tensor_arg(x_2d),),
            {"ord": float("inf")},
            supports_backward=False,
            supports_gradcheck=False,
        ),
    )


def _index_select_samples() -> Sequence[SampleInput]:
    x = np.arange(1, 13, dtype=np.float64).reshape(3, 4) * 0.3 - 1.2
    idx0 = np.array([2, 0, 2, 1], dtype=np.int64)
    idx1 = np.array([3, 1, 0], dtype=np.int64)
    return (
        SampleInput("dim0_repeats", (tensor_arg(x), 0, index_arg(idx0))),
        SampleInput("dim1", (tensor_arg(x), 1, index_arg(idx1))),
    )


def _gather_samples() -> Sequence[SampleInput]:
    x = np.arange(1, 13, dtype=np.float64).reshape(3, 4) * 0.25 - 1.0
    idx_d0 = np.array([[2, 0, 1, 2], [0, 2, 0, 1]], dtype=np.int64)
    idx_d1 = np.array([[3, 1], [0, 2], [2, 3]], dtype=np.int64)
    return (
        SampleInput("dim0", (tensor_arg(x), 0, index_arg(idx_d0))),
        SampleInput("dim1", (tensor_arg(x), 1, index_arg(idx_d1))),
    )


def _scatter_samples() -> Sequence[SampleInput]:
    base = np.linspace(-1.0, 1.0, num=12, dtype=np.float64).reshape(3, 4)
    idx_d0 = np.array([[2, 0, 1, 2], [0, 1, 2, 0]], dtype=np.int64)
    src_d0 = np.linspace(0.5, 2.5, num=8, dtype=np.float64).reshape(2, 4)
    idx_d1 = np.array([[3, 1], [0, 2], [2, 0]], dtype=np.int64)
    src_d1 = np.linspace(-1.5, 1.5, num=6, dtype=np.float64).reshape(3, 2)
    return (
        SampleInput("dim0_unique", (tensor_arg(base), 0, index_arg(idx_d0), tensor_arg(src_d0))),
        SampleInput("dim1_unique", (tensor_arg(base), 1, index_arg(idx_d1), tensor_arg(src_d1))),
    )


def _index_copy_samples() -> Sequence[SampleInput]:
    base = np.linspace(-1.0, 1.0, num=12, dtype=np.float64).reshape(3, 4)
    idx0 = np.array([2, 0], dtype=np.int64)
    src0 = np.linspace(0.5, 2.0, num=8, dtype=np.float64).reshape(2, 4)
    idx1 = np.array([3, 1], dtype=np.int64)
    src1 = np.linspace(-1.2, 1.2, num=6, dtype=np.float64).reshape(3, 2)
    return (
        SampleInput("dim0", (tensor_arg(base), 0, index_arg(idx0), tensor_arg(src0))),
        SampleInput("dim1", (tensor_arg(base), 1, index_arg(idx1), tensor_arg(src1))),
    )


def _take_samples() -> Sequence[SampleInput]:
    x = np.linspace(-1.5, 2.0, num=12, dtype=np.float64).reshape(3, 4)
    idx = np.array([[11, 0, 5], [3, 8, 2]], dtype=np.int64)
    return (SampleInput("2d_idx", (tensor_arg(x), index_arg(idx)), supports_strided=False),)


def _masked_fill_samples() -> Sequence[SampleInput]:
    x = np.linspace(-1.2, 1.8, num=12, dtype=np.float64).reshape(3, 4)
    mask_same = np.array(
        [[True, False, False, True], [False, True, False, False], [True, False, True, False]]
    )
    mask_bcast = np.array([[True], [False], [True]])
    return (
        SampleInput("same_mask", (tensor_arg(x), mask_arg(mask_same), -2.5)),
        SampleInput("bcast_mask", (tensor_arg(x), mask_arg(mask_bcast), 0.75)),
    )


def _tri_samples() -> Sequence[SampleInput]:
    x_2d = np.linspace(-1.5, 2.0, num=12, dtype=np.float64).reshape(3, 4)
    x_3d = np.linspace(-1.0, 1.5, num=18, dtype=np.float64).reshape(2, 3, 3)
    return (
        SampleInput("diag0_2d", (tensor_arg(x_2d),), {"diagonal": 0}),
        SampleInput("diag_pos1_2d", (tensor_arg(x_2d),), {"diagonal": 1}),
        SampleInput("diag_neg1_3d", (tensor_arg(x_3d),), {"diagonal": -1}),
    )


def _diagonal_samples() -> Sequence[SampleInput]:
    x_3d = np.linspace(-1.5, 2.0, num=24, dtype=np.float64).reshape(2, 3, 4)
    return (
        SampleInput("offset0_last2", (tensor_arg(x_3d),), {"offset": 0, "dim1": -2, "dim2": -1}),
        SampleInput("offset1_last2", (tensor_arg(x_3d),), {"offset": 1, "dim1": 1, "dim2": 2}),
        SampleInput("offset_neg1_01", (tensor_arg(x_3d),), {"offset": -1, "dim1": 0, "dim2": 1}),
    )


def _roll_flip_samples() -> Sequence[SampleInput]:
    x_2d = np.linspace(-1.5, 2.0, num=12, dtype=np.float64).reshape(3, 4)
    x_3d = np.linspace(-1.0, 1.5, num=12, dtype=np.float64).reshape(2, 3, 2)
    return (SampleInput("2d", (tensor_arg(x_2d),)), SampleInput("3d", (tensor_arg(x_3d),)))


REDUCTION_INDEX_OPINFOS: tuple[OpInfo, ...] = (
    OpInfo(
        name="sum",
        family="reduction",
        lucid_fn=lucid.sum,
        ref_fn=lambda r, x, **kw: r.sum(x, **kw),
        sample_inputs_fn=_reduction_multi_axis_samples,
    ),
    OpInfo(
        name="mean",
        family="reduction",
        lucid_fn=lucid.mean,
        ref_fn=lambda r, x, **kw: r.mean(x, **kw),
        sample_inputs_fn=_reduction_multi_axis_samples,
    ),
    OpInfo(
        name="prod",
        family="reduction",
        lucid_fn=lucid.prod,
        ref_fn=lambda r, x, **kw: r.prod(x, **kw),
        sample_inputs_fn=_single_axis_reduction_samples,
    ),
    OpInfo(
        name="amax",
        family="reduction",
        lucid_fn=lucid.amax,
        ref_fn=lambda r, x, **kw: r.amax(x, **kw),
        sample_inputs_fn=_reduction_multi_axis_samples,
    ),
    OpInfo(
        name="amin",
        family="reduction",
        lucid_fn=lucid.amin,
        ref_fn=lambda r, x, **kw: r.amin(x, **kw),
        sample_inputs_fn=_reduction_multi_axis_samples,
    ),
    OpInfo(
        name="logsumexp",
        family="reduction",
        lucid_fn=lucid.logsumexp,
        ref_fn=lambda r, x, **kw: r.logsumexp(x, **kw),
        sample_inputs_fn=_single_axis_reduction_samples,
    ),
    OpInfo(
        name="var",
        family="reduction",
        lucid_fn=lucid.var,
        ref_fn=lambda r, x, **kw: r.var(x, **kw),
        sample_inputs_fn=_var_std_samples,
    ),
    OpInfo(
        name="std",
        family="reduction",
        lucid_fn=lucid.std,
        ref_fn=lambda r, x, **kw: r.std(x, **kw),
        sample_inputs_fn=_var_std_samples,
    ),
    OpInfo(
        name="nansum",
        family="reduction",
        lucid_fn=lucid.nansum,
        ref_fn=lambda r, x, **kw: r.nansum(x, **kw),
        sample_inputs_fn=_reduction_multi_axis_samples,
    ),
    OpInfo(
        name="nanmean",
        family="reduction",
        lucid_fn=lucid.nanmean,
        ref_fn=lambda r, x, **kw: r.nanmean(x, **kw),
        sample_inputs_fn=_reduction_multi_axis_samples,
    ),
    OpInfo(
        name="cumsum",
        family="scan",
        lucid_fn=lucid.cumsum,
        ref_fn=lambda r, x, **kw: r.cumsum(x, **kw),
        sample_inputs_fn=_scan_samples,
    ),
    OpInfo(
        name="cumprod",
        family="scan",
        lucid_fn=lucid.cumprod,
        ref_fn=lambda r, x, **kw: r.cumprod(x, **kw),
        sample_inputs_fn=_scan_samples,
    ),
    OpInfo(
        name="linalg.vector_norm",
        family="reduction",
        lucid_fn=llinalg.vector_norm,
        ref_fn=lambda r, x, **kw: r.linalg.vector_norm(x, **kw),
        sample_inputs_fn=_vector_norm_samples,
    ),
    OpInfo(
        name="linalg.matrix_norm",
        family="reduction",
        lucid_fn=llinalg.matrix_norm,
        ref_fn=lambda r, x, **kw: r.linalg.matrix_norm(x, **kw),
        sample_inputs_fn=_matrix_norm_samples,
    ),
    OpInfo(
        name="index_select",
        family="indexing",
        lucid_fn=lucid.index_select,
        ref_fn=lambda r, x, d, i: r.index_select(x, d, i),
        sample_inputs_fn=_index_select_samples,
    ),
    OpInfo(
        name="gather",
        family="indexing",
        lucid_fn=lucid.gather,
        ref_fn=lambda r, x, d, i: r.gather(x, d, i),
        sample_inputs_fn=_gather_samples,
    ),
    OpInfo(
        name="scatter",
        family="indexing",
        lucid_fn=lucid.scatter,
        ref_fn=lambda r, x, d, i, s: r.scatter(x, d, i, s),
        sample_inputs_fn=_scatter_samples,
    ),
    OpInfo(
        name="scatter_add",
        family="indexing",
        lucid_fn=lucid.scatter_add,
        ref_fn=lambda r, x, d, i, s: r.scatter_add(x, d, i, s),
        sample_inputs_fn=_scatter_samples,
    ),
    OpInfo(
        name="index_copy",
        family="indexing",
        lucid_fn=lucid.index_copy,
        ref_fn=lambda r, x, d, i, s: r.index_copy(x, d, i, s),
        sample_inputs_fn=_index_copy_samples,
    ),
    OpInfo(
        name="take",
        family="indexing",
        lucid_fn=lucid.take,
        ref_fn=lambda r, x, i: r.take(x, i),
        sample_inputs_fn=_take_samples,
        supports_strided=False,
    ),
    OpInfo(
        name="masked_fill",
        family="indexing",
        lucid_fn=lucid.masked_fill,
        ref_fn=lambda r, x, m, v: r.masked_fill(x, m, v),
        sample_inputs_fn=_masked_fill_samples,
    ),
    OpInfo(
        name="tril",
        family="view_layout",
        lucid_fn=lucid.tril,
        ref_fn=lambda r, x, **kw: r.tril(x, **kw),
        sample_inputs_fn=_tri_samples,
    ),
    OpInfo(
        name="triu",
        family="view_layout",
        lucid_fn=lucid.triu,
        ref_fn=lambda r, x, **kw: r.triu(x, **kw),
        sample_inputs_fn=_tri_samples,
    ),
    OpInfo(
        name="diagonal",
        family="view_layout",
        lucid_fn=lucid.diagonal,
        ref_fn=lambda r, x, **kw: r.diagonal(x, **kw),
        sample_inputs_fn=_diagonal_samples,
    ),
    OpInfo(
        name="roll",
        family="view_layout",
        lucid_fn=lambda x: lucid.roll(x, shifts=(1, -1), dims=(0, 1)),
        ref_fn=lambda r, x: r.roll(x, shifts=(1, -1), dims=(0, 1)),
        sample_inputs_fn=_roll_flip_samples,
    ),
    OpInfo(
        name="flip",
        family="view_layout",
        lucid_fn=lambda x: lucid.flip(x, dims=(0, -1)),
        ref_fn=lambda r, x: r.flip(x, dims=(0, -1)),
        sample_inputs_fn=_roll_flip_samples,
    ),
    OpInfo(
        name="rot90",
        family="view_layout",
        lucid_fn=lambda x: lucid.rot90(x, k=1, dims=(0, 1)),
        ref_fn=lambda r, x: r.rot90(x, k=1, dims=(0, 1)),
        sample_inputs_fn=_roll_flip_samples,
    ),
)
