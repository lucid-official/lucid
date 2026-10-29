"""OpInfo catalog for linear algebra contractions/decompositions and ``nn.functional``."""

from collections.abc import Callable, Sequence
import numpy as np

import lucid
import lucid.linalg as llinalg
import lucid.nn.functional as F
from lucid.test.parity.opinfo._core import OpInfo, SampleInput, index_arg, tensor_arg


def _matmul_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(77)
    a_2d = rng.uniform(-1.5, 1.5, size=(3, 4))
    b_2d = rng.uniform(-1.5, 1.5, size=(4, 2))
    a_3d = rng.uniform(-1.2, 1.2, size=(2, 3, 4))
    b_3d = rng.uniform(-1.2, 1.2, size=(2, 4, 2))
    b_bcast = rng.uniform(-1.2, 1.2, size=(4, 2))
    return (
        SampleInput("2d_x_2d", (tensor_arg(a_2d), tensor_arg(b_2d))),
        SampleInput("3d_x_3d", (tensor_arg(a_3d), tensor_arg(b_3d))),
        SampleInput("3d_x_2d_bcast", (tensor_arg(a_3d), tensor_arg(b_bcast))),
    )


def _mm_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(78)
    return (
        SampleInput(
            "3x4_4x2",
            (
                tensor_arg(rng.uniform(-1.5, 1.5, size=(3, 4))),
                tensor_arg(rng.uniform(-1.5, 1.5, size=(4, 2))),
            ),
        ),
    )


def _bmm_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(79)
    return (
        SampleInput(
            "2x3x4_2x4x2",
            (
                tensor_arg(rng.uniform(-1.5, 1.5, size=(2, 3, 4))),
                tensor_arg(rng.uniform(-1.5, 1.5, size=(2, 4, 2))),
            ),
        ),
    )


def _addmm_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(80)
    bias = rng.uniform(-1.0, 1.0, size=(3, 2))
    bias_1d = rng.uniform(-1.0, 1.0, size=(2,))
    m1 = rng.uniform(-1.2, 1.2, size=(3, 4))
    m2 = rng.uniform(-1.2, 1.2, size=(4, 2))
    return (
        SampleInput("default_ab", (tensor_arg(bias), tensor_arg(m1), tensor_arg(m2))),
        SampleInput(
            "scaled_bcast_bias",
            (tensor_arg(bias_1d), tensor_arg(m1), tensor_arg(m2)),
            {"beta": 0.5, "alpha": 1.25},
        ),
    )


def _baddbmm_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(81)
    inp = rng.uniform(-1.0, 1.0, size=(2, 3, 2))
    b1 = rng.uniform(-1.2, 1.2, size=(2, 3, 4))
    b2 = rng.uniform(-1.2, 1.2, size=(2, 4, 2))
    return (
        SampleInput("default", (tensor_arg(inp), tensor_arg(b1), tensor_arg(b2))),
        SampleInput(
            "scaled",
            (tensor_arg(inp), tensor_arg(b1), tensor_arg(b2)),
            {"beta": 0.75, "alpha": -0.5},
        ),
    )


def _einsum_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(82)
    a = rng.uniform(-1.2, 1.2, size=(2, 3, 4))
    b = rng.uniform(-1.2, 1.2, size=(2, 5, 4))
    m = rng.uniform(-1.5, 1.5, size=(3, 3))
    return (
        SampleInput("bmm_transpose", ("bij,bkj->bik", tensor_arg(a), tensor_arg(b))),
        SampleInput("trace_like", ("ii->", tensor_arg(m))),
    )


def _tensordot_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(83)
    a = rng.uniform(-1.2, 1.2, size=(2, 3, 4))
    b = rng.uniform(-1.2, 1.2, size=(3, 4, 2))
    return (
        SampleInput("dims_int2", (tensor_arg(a), tensor_arg(b)), {"dims": 2}),
        SampleInput("dims_axes", (tensor_arg(a), tensor_arg(b)), {"dims": ([1, 2], [0, 1])}),
    )


def _inner_samples() -> Sequence[SampleInput]:
    v1 = np.array([0.8, -1.2, 1.5], dtype=np.float64)
    v2 = np.array([-0.5, 1.1, 0.7], dtype=np.float64)
    m1 = np.array([[0.5, -1.0, 1.2], [1.1, 0.4, -0.8]], dtype=np.float64)
    return (
        SampleInput("1d_pair", (tensor_arg(v1), tensor_arg(v2))),
        SampleInput("2d_x_1d", (tensor_arg(m1), tensor_arg(v1))),
    )


def _outer_kron_samples() -> Sequence[SampleInput]:
    v1 = np.array([0.8, -1.2, 1.5], dtype=np.float64)
    v2 = np.array([-0.5, 1.1, 0.7, -0.9], dtype=np.float64)
    return (SampleInput("1d_pair", (tensor_arg(v1), tensor_arg(v2))),)


def _spd_matrix_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(84)
    m = rng.uniform(-0.8, 0.8, size=(3, 3))
    spd_2d = m @ m.T + 2.5 * np.eye(3)
    m_b = rng.uniform(-0.8, 0.8, size=(2, 3, 3))
    spd_3d = np.matmul(m_b, np.swapaxes(m_b, -1, -2)) + 2.5 * np.eye(3)
    return (
        SampleInput("2d_spd", (tensor_arg(spd_2d),)),
        SampleInput("3d_batched_spd", (tensor_arg(spd_3d),)),
    )


def _solve_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(85)
    a_2d = rng.uniform(-0.6, 0.6, size=(3, 3)) + 3.0 * np.eye(3)
    b_2d = rng.uniform(-1.5, 1.5, size=(3, 2))
    a_3d = rng.uniform(-0.6, 0.6, size=(2, 3, 3)) + 3.0 * np.eye(3)
    b_3d = rng.uniform(-1.5, 1.5, size=(2, 3, 2))
    return (
        SampleInput("2d_rhs", (tensor_arg(a_2d), tensor_arg(b_2d))),
        SampleInput("3d_batched_rhs", (tensor_arg(a_3d), tensor_arg(b_3d))),
    )


def _solve_triangular_samples() -> Sequence[SampleInput]:
    upper = np.array([[2.2, -0.6, 0.5], [0.0, 1.8, -0.4], [0.0, 0.0, 2.5]], dtype=np.float64)
    lower = upper.T.copy()
    b = np.array([[1.0, -0.5], [0.8, 1.2], [-0.7, 0.9]], dtype=np.float64)
    return (
        SampleInput("upper", (tensor_arg(upper), tensor_arg(b)), {"upper": True}),
        SampleInput("lower", (tensor_arg(lower), tensor_arg(b)), {"upper": False}),
    )


def _activation_samples() -> Sequence[SampleInput]:
    x_2d = np.array([[-2.2, -0.7, -0.15, 0.25], [0.6, 1.3, -1.4, 2.1]], dtype=np.float64)
    x_3d = np.linspace(-1.8, 1.9, num=12, dtype=np.float64).reshape(2, 3, 2)
    return (SampleInput("2d", (tensor_arg(x_2d),)), SampleInput("3d", (tensor_arg(x_3d),)))


def _softmax_samples() -> Sequence[SampleInput]:
    x_2d = np.array([[0.5, -1.2, 1.8, 0.1], [-0.7, 1.4, 0.3, -1.5]], dtype=np.float64)
    x_3d = np.linspace(-1.5, 1.5, num=12, dtype=np.float64).reshape(2, 3, 2)
    return (
        SampleInput("dim_neg1_2d", (tensor_arg(x_2d),), {"dim": -1}),
        SampleInput("dim0_2d", (tensor_arg(x_2d),), {"dim": 0}),
        SampleInput("dim1_3d", (tensor_arg(x_3d),), {"dim": 1}),
    )


def _layer_norm_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(90)
    x = rng.uniform(-1.5, 1.5, size=(2, 3, 4))
    w = rng.uniform(0.5, 1.5, size=(4,))
    b = rng.uniform(-0.5, 0.5, size=(4,))
    return (
        SampleInput(
            "affine_1d", (tensor_arg(x), (4,)), {"weight": tensor_arg(w), "bias": tensor_arg(b)}
        ),
        SampleInput("no_affine_2d", (tensor_arg(x), (3, 4))),
    )


def _rms_norm_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(91)
    x = rng.uniform(-1.5, 1.5, size=(2, 3, 4))
    w = rng.uniform(0.5, 1.5, size=(4,))
    return (
        SampleInput("weighted", (tensor_arg(x), (4,)), {"weight": tensor_arg(w), "eps": 1e-5}),
        SampleInput("unweighted", (tensor_arg(x), (4,)), {"eps": 1e-5}),
    )


def _group_norm_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(92)
    x = rng.uniform(-1.5, 1.5, size=(2, 4, 3, 3))
    w = rng.uniform(0.5, 1.5, size=(4,))
    b = rng.uniform(-0.5, 0.5, size=(4,))
    return (
        SampleInput(
            "g2_affine", (tensor_arg(x), 2), {"weight": tensor_arg(w), "bias": tensor_arg(b)}
        ),
        SampleInput("g2_no_affine", (tensor_arg(x), 2)),
    )


def _linear_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(93)
    x_2d = rng.uniform(-1.5, 1.5, size=(3, 4))
    x_3d = rng.uniform(-1.5, 1.5, size=(2, 3, 4))
    w = rng.uniform(-1.0, 1.0, size=(5, 4))
    b = rng.uniform(-0.5, 0.5, size=(5,))
    return (
        SampleInput("2d_with_bias", (tensor_arg(x_2d), tensor_arg(w), tensor_arg(b))),
        SampleInput("3d_no_bias", (tensor_arg(x_3d), tensor_arg(w))),
    )


def _conv2d_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(94)
    x = rng.uniform(-1.2, 1.2, size=(2, 2, 5, 5))
    w = rng.uniform(-0.8, 0.8, size=(3, 2, 3, 3))
    b = rng.uniform(-0.4, 0.4, size=(3,))
    w_g2 = rng.uniform(-0.8, 0.8, size=(2, 1, 3, 3))
    return (
        SampleInput(
            "stride1_pad1_bias",
            (tensor_arg(x), tensor_arg(w), tensor_arg(b)),
            {"stride": 1, "padding": 1},
        ),
        SampleInput(
            "stride2_dilation2_groups2",
            (tensor_arg(x), tensor_arg(w_g2)),
            {"stride": 1, "padding": 1, "dilation": 2, "groups": 2},
        ),
    )


def _pool2d_samples() -> Sequence[SampleInput]:
    x = np.arange(1, 33, dtype=np.float64).reshape(1, 2, 4, 4) * 0.15 - 1.2
    return (
        SampleInput("k2_s2", (tensor_arg(x), 2), {"stride": 2}),
        SampleInput("k2_s1_p1", (tensor_arg(x), 2), {"stride": 1, "padding": 1}),
    )


def _interpolate_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(95)
    x = rng.uniform(-1.2, 1.2, size=(1, 2, 3, 3))
    return (
        SampleInput("nearest_size", (tensor_arg(x),), {"size": (5, 5), "mode": "nearest"}),
        SampleInput(
            "bilinear_align_true",
            (tensor_arg(x),),
            {"size": (5, 5), "mode": "bilinear", "align_corners": True},
        ),
        SampleInput(
            "bilinear_align_false",
            (tensor_arg(x),),
            {"size": (4, 6), "mode": "bilinear", "align_corners": False},
        ),
    )


def _grid_sample_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(96)
    x = rng.uniform(-1.2, 1.2, size=(1, 2, 4, 4))
    grid = rng.uniform(-0.75, 0.75, size=(1, 3, 3, 2))
    return (
        SampleInput(
            "bilinear_zeros_align_true",
            (tensor_arg(x), tensor_arg(grid)),
            {"mode": "bilinear", "padding_mode": "zeros", "align_corners": True},
        ),
        SampleInput(
            "bilinear_border_align_false",
            (tensor_arg(x), tensor_arg(grid)),
            {"mode": "bilinear", "padding_mode": "border", "align_corners": False},
        ),
    )


def _sdpa_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(97)
    q = rng.uniform(-1.0, 1.0, size=(1, 2, 4, 4))
    k = rng.uniform(-1.0, 1.0, size=(1, 2, 4, 4))
    v = rng.uniform(-1.0, 1.0, size=(1, 2, 4, 4))
    mask = rng.uniform(-0.5, 0.5, size=(1, 1, 4, 4))
    return (
        SampleInput("non_causal", (tensor_arg(q), tensor_arg(k), tensor_arg(v))),
        SampleInput("causal", (tensor_arg(q), tensor_arg(k), tensor_arg(v)), {"is_causal": True}),
        SampleInput(
            "float_attn_mask",
            (tensor_arg(q), tensor_arg(k), tensor_arg(v)),
            {"attn_mask": tensor_arg(mask)},
        ),
    )


def _regression_loss_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(98)
    pred = rng.uniform(-1.5, 1.5, size=(3, 4))
    target = rng.uniform(-1.5, 1.5, size=(3, 4))
    return (
        SampleInput("mean", (tensor_arg(pred), tensor_arg(target)), {"reduction": "mean"}),
        SampleInput("sum", (tensor_arg(pred), tensor_arg(target)), {"reduction": "sum"}),
        SampleInput("none", (tensor_arg(pred), tensor_arg(target)), {"reduction": "none"}),
    )


def _cross_entropy_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(99)
    logits = rng.uniform(-1.5, 1.5, size=(4, 5))
    targets = np.array([2, 0, 4, 1], dtype=np.int64)
    return (
        SampleInput("mean", (tensor_arg(logits), index_arg(targets)), {"reduction": "mean"}),
        SampleInput(
            "label_smoothing",
            (tensor_arg(logits), index_arg(targets)),
            {"reduction": "mean", "label_smoothing": 0.1},
        ),
        SampleInput("none", (tensor_arg(logits), index_arg(targets)), {"reduction": "none"}),
    )


_ACTIVATION_SPECS: tuple[tuple[str, Callable[..., object], Callable[..., object]], ...] = (
    ("F.relu", F.relu, lambda r, x: r.nn.functional.relu(x)),
    (
        "F.leaky_relu",
        lambda x: F.leaky_relu(x, 0.1),
        lambda r, x: r.nn.functional.leaky_relu(x, 0.1),
    ),
    ("F.elu", F.elu, lambda r, x: r.nn.functional.elu(x)),
    ("F.selu", F.selu, lambda r, x: r.nn.functional.selu(x)),
    ("F.celu", F.celu, lambda r, x: r.nn.functional.celu(x)),
    ("F.gelu", F.gelu, lambda r, x: r.nn.functional.gelu(x)),
    ("F.silu", F.silu, lambda r, x: r.nn.functional.silu(x)),
    ("F.mish", F.mish, lambda r, x: r.nn.functional.mish(x)),
    ("F.softplus", F.softplus, lambda r, x: r.nn.functional.softplus(x)),
    ("F.softsign", F.softsign, lambda r, x: r.nn.functional.softsign(x)),
    ("F.hardtanh", F.hardtanh, lambda r, x: r.nn.functional.hardtanh(x)),
    ("F.hardswish", F.hardswish, lambda r, x: r.nn.functional.hardswish(x)),
    ("F.hardsigmoid", F.hardsigmoid, lambda r, x: r.nn.functional.hardsigmoid(x)),
)

LINALG_NN_OPINFOS: tuple[OpInfo, ...] = (
    OpInfo(
        name="matmul",
        family="linalg",
        lucid_fn=lucid.matmul,
        ref_fn=lambda r, a, b: r.matmul(a, b),
        sample_inputs_fn=_matmul_samples,
    ),
    OpInfo(
        name="mm",
        family="linalg",
        lucid_fn=lucid.mm,
        ref_fn=lambda r, a, b: r.mm(a, b),
        sample_inputs_fn=_mm_samples,
    ),
    OpInfo(
        name="bmm",
        family="linalg",
        lucid_fn=lucid.bmm,
        ref_fn=lambda r, a, b: r.bmm(a, b),
        sample_inputs_fn=_bmm_samples,
    ),
    OpInfo(
        name="addmm",
        family="linalg",
        lucid_fn=lucid.addmm,
        ref_fn=lambda r, inp, m1, m2, **kw: r.addmm(inp, m1, m2, **kw),
        sample_inputs_fn=_addmm_samples,
    ),
    OpInfo(
        name="baddbmm",
        family="linalg",
        lucid_fn=lucid.baddbmm,
        ref_fn=lambda r, inp, b1, b2, **kw: r.baddbmm(inp, b1, b2, **kw),
        sample_inputs_fn=_baddbmm_samples,
    ),
    OpInfo(
        name="einops.einsum",
        family="linalg",
        lucid_fn=lucid.einops.einsum,
        ref_fn=lambda r, eq, *ops: r.einsum(eq, *ops),
        sample_inputs_fn=_einsum_samples,
    ),
    OpInfo(
        name="tensordot",
        family="linalg",
        lucid_fn=lucid.tensordot,
        ref_fn=lambda r, a, b, **kw: r.tensordot(a, b, **kw),
        sample_inputs_fn=_tensordot_samples,
    ),
    OpInfo(
        name="inner",
        family="linalg",
        lucid_fn=lucid._ops.inner,
        ref_fn=lambda r, a, b: r.inner(a, b),
        sample_inputs_fn=_inner_samples,
    ),
    OpInfo(
        name="outer",
        family="linalg",
        lucid_fn=lucid._ops.outer,
        ref_fn=lambda r, a, b: r.outer(a, b),
        sample_inputs_fn=_outer_kron_samples,
    ),
    OpInfo(
        name="kron",
        family="linalg",
        lucid_fn=lucid.kron,
        ref_fn=lambda r, a, b: r.kron(a, b),
        sample_inputs_fn=_outer_kron_samples,
    ),
    OpInfo(
        name="linalg.inv",
        family="linalg",
        lucid_fn=llinalg.inv,
        ref_fn=lambda r, a: r.linalg.inv(a),
        sample_inputs_fn=_spd_matrix_samples,
        supports_half=False,
    ),
    OpInfo(
        name="linalg.cholesky",
        family="linalg",
        lucid_fn=llinalg.cholesky,
        ref_fn=lambda r, a: r.linalg.cholesky(a),
        sample_inputs_fn=_spd_matrix_samples,
        supports_half=False,
        supports_gradcheck=False,
    ),
    OpInfo(
        name="linalg.pinv",
        family="linalg",
        lucid_fn=llinalg.pinv,
        ref_fn=lambda r, a: r.linalg.pinv(a),
        sample_inputs_fn=_spd_matrix_samples,
        supports_half=False,
    ),
    OpInfo(
        name="linalg.slogdet",
        family="linalg",
        lucid_fn=lambda a: llinalg.slogdet(a).logabsdet,
        ref_fn=lambda r, a: r.linalg.slogdet(a).logabsdet,
        sample_inputs_fn=_spd_matrix_samples,
        supports_half=False,
    ),
    OpInfo(
        name="linalg.solve",
        family="linalg",
        lucid_fn=llinalg.solve,
        ref_fn=lambda r, a, b: r.linalg.solve(a, b),
        sample_inputs_fn=_solve_samples,
        supports_half=False,
    ),
    OpInfo(
        name="linalg.solve_triangular",
        family="linalg",
        lucid_fn=llinalg.solve_triangular,
        ref_fn=lambda r, a, b, **kw: r.linalg.solve_triangular(a, b, **kw),
        sample_inputs_fn=_solve_triangular_samples,
        supports_half=False,
        supports_strided=False,
    ),
    *(
        OpInfo(
            name=name,
            family="nn_activation",
            lucid_fn=l_fn,
            ref_fn=r_fn,
            sample_inputs_fn=_activation_samples,
        )
        for name, l_fn, r_fn in _ACTIVATION_SPECS
    ),
    OpInfo(
        name="F.softmax",
        family="nn_activation",
        lucid_fn=F.softmax,
        ref_fn=lambda r, x, **kw: r.nn.functional.softmax(x, **kw),
        sample_inputs_fn=_softmax_samples,
    ),
    OpInfo(
        name="F.log_softmax",
        family="nn_activation",
        lucid_fn=F.log_softmax,
        ref_fn=lambda r, x, **kw: r.nn.functional.log_softmax(x, **kw),
        sample_inputs_fn=_softmax_samples,
    ),
    OpInfo(
        name="F.layer_norm",
        family="nn_norm",
        lucid_fn=F.layer_norm,
        ref_fn=lambda r, x, s, **kw: r.nn.functional.layer_norm(x, s, **kw),
        sample_inputs_fn=_layer_norm_samples,
    ),
    OpInfo(
        name="F.rms_norm",
        family="nn_norm",
        lucid_fn=F.rms_norm,
        ref_fn=lambda r, x, s, **kw: r.nn.functional.rms_norm(x, s, **kw),
        sample_inputs_fn=_rms_norm_samples,
    ),
    OpInfo(
        name="F.group_norm",
        family="nn_norm",
        lucid_fn=F.group_norm,
        ref_fn=lambda r, x, g, **kw: r.nn.functional.group_norm(x, g, **kw),
        sample_inputs_fn=_group_norm_samples,
    ),
    OpInfo(
        name="F.linear",
        family="nn_layer",
        lucid_fn=F.linear,
        ref_fn=lambda r, *a, **kw: r.nn.functional.linear(*a, **kw),
        sample_inputs_fn=_linear_samples,
    ),
    OpInfo(
        name="F.conv2d",
        family="nn_layer",
        lucid_fn=F.conv2d,
        ref_fn=lambda r, *a, **kw: r.nn.functional.conv2d(*a, **kw),
        sample_inputs_fn=_conv2d_samples,
        atol=2e-4,
        bwd_atol=5e-4,
    ),
    OpInfo(
        name="F.avg_pool2d",
        family="nn_pool",
        lucid_fn=F.avg_pool2d,
        ref_fn=lambda r, x, k, **kw: r.nn.functional.avg_pool2d(x, k, **kw),
        sample_inputs_fn=_pool2d_samples,
    ),
    OpInfo(
        name="F.max_pool2d",
        family="nn_pool",
        lucid_fn=F.max_pool2d,
        ref_fn=lambda r, x, k, **kw: r.nn.functional.max_pool2d(x, k, **kw),
        sample_inputs_fn=_pool2d_samples,
    ),
    OpInfo(
        name="F.interpolate",
        family="nn_spatial",
        lucid_fn=F.interpolate,
        ref_fn=lambda r, x, **kw: r.nn.functional.interpolate(x, **kw),
        sample_inputs_fn=_interpolate_samples,
    ),
    OpInfo(
        name="F.grid_sample",
        family="nn_spatial",
        lucid_fn=F.grid_sample,
        ref_fn=lambda r, x, g, **kw: r.nn.functional.grid_sample(x, g, **kw),
        sample_inputs_fn=_grid_sample_samples,
        atol=2e-4,
        bwd_atol=5e-4,
    ),
    OpInfo(
        name="F.scaled_dot_product_attention",
        family="nn_attention",
        lucid_fn=F.scaled_dot_product_attention,
        ref_fn=lambda r, q, k, v, **kw: r.nn.functional.scaled_dot_product_attention(q, k, v, **kw),
        sample_inputs_fn=_sdpa_samples,
        atol=2e-4,
        bwd_atol=5e-4,
    ),
    OpInfo(
        name="F.mse_loss",
        family="nn_loss",
        lucid_fn=F.mse_loss,
        ref_fn=lambda r, x, y, **kw: r.nn.functional.mse_loss(x, y, **kw),
        sample_inputs_fn=_regression_loss_samples,
    ),
    OpInfo(
        name="F.l1_loss",
        family="nn_loss",
        lucid_fn=F.l1_loss,
        ref_fn=lambda r, x, y, **kw: r.nn.functional.l1_loss(x, y, **kw),
        sample_inputs_fn=_regression_loss_samples,
    ),
    OpInfo(
        name="F.smooth_l1_loss",
        family="nn_loss",
        lucid_fn=F.smooth_l1_loss,
        ref_fn=lambda r, x, y, **kw: r.nn.functional.smooth_l1_loss(x, y, **kw),
        sample_inputs_fn=_regression_loss_samples,
    ),
    OpInfo(
        name="F.huber_loss",
        family="nn_loss",
        lucid_fn=F.huber_loss,
        ref_fn=lambda r, x, y, **kw: r.nn.functional.huber_loss(x, y, **kw),
        sample_inputs_fn=_regression_loss_samples,
    ),
    OpInfo(
        name="F.binary_cross_entropy_with_logits",
        family="nn_loss",
        lucid_fn=F.binary_cross_entropy_with_logits,
        ref_fn=lambda r, x, y, **kw: r.nn.functional.binary_cross_entropy_with_logits(x, y, **kw),
        sample_inputs_fn=_regression_loss_samples,
    ),
    OpInfo(
        name="F.cross_entropy",
        family="nn_loss",
        lucid_fn=F.cross_entropy,
        ref_fn=lambda r, x, y, **kw: r.nn.functional.cross_entropy(x, y, **kw),
        sample_inputs_fn=_cross_entropy_samples,
    ),
)
