"""OpInfo catalog for FFT/spectral ops, extended linalg, and extended ``nn.functional``."""

from collections.abc import Sequence
import numpy as np

import lucid.fft as lfft
import lucid.linalg as llinalg
import lucid.nn.functional as F
from lucid.test.parity.opinfo._core import OpInfo, SampleInput, index_arg, tensor_arg


def _fft_1d_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(501)
    x = rng.uniform(-1.5, 1.5, size=(3, 8))
    return (
        SampleInput("default_norm", (tensor_arg(x),), {"dim": -1, "norm": "backward"}),
        SampleInput("ortho_norm", (tensor_arg(x),), {"dim": -1, "norm": "ortho"}),
        SampleInput("forward_norm_n6", (tensor_arg(x),), {"n": 6, "dim": -1, "norm": "forward"}),
    )


def _fft_2d_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(502)
    x = rng.uniform(-1.2, 1.2, size=(2, 4, 6))
    return (
        SampleInput("default_last2", (tensor_arg(x),), {"dim": (-2, -1), "norm": "backward"}),
        SampleInput("ortho_last2", (tensor_arg(x),), {"dim": (-2, -1), "norm": "ortho"}),
    )


def _fftshift_samples() -> Sequence[SampleInput]:
    x_2d = np.linspace(-1.5, 1.5, num=12, dtype=np.float64).reshape(3, 4)
    x_3d = np.linspace(-1.0, 2.0, num=24, dtype=np.float64).reshape(2, 3, 4)
    return (
        SampleInput("all_axes_2d", (tensor_arg(x_2d),)),
        SampleInput("dim_neg1_3d", (tensor_arg(x_3d),), {"dim": -1}),
        SampleInput("dim_tuple_3d", (tensor_arg(x_3d),), {"dim": (1, 2)}),
    )


def _eigvalsh_samples() -> Sequence[SampleInput]:
    # Well-separated eigenvalues so eigenvectors and eigenvalue derivatives are unique.
    a = np.array([[2.5, 0.3, -0.2], [0.3, 5.2, 0.4], [-0.2, 0.4, 8.7]], dtype=np.float64)
    b = np.stack([a, a * 1.1 + np.diag([0.5, 1.2, 2.1])], axis=0)
    return (
        SampleInput("2d_lower", (tensor_arg(a),), {"UPLO": "L"}),
        SampleInput("2d_upper", (tensor_arg(a),), {"UPLO": "U"}),
        SampleInput("3d_batched", (tensor_arg(b),), {"UPLO": "L"}),
    )


def _svdvals_samples() -> Sequence[SampleInput]:
    # Distinct singular values for well-conditioned forward/backward parity.
    a = np.array(
        [[3.2, 0.2, -0.1], [0.1, 1.8, 0.3], [-0.2, 0.1, 0.7], [0.3, -0.2, 0.4]], dtype=np.float64
    )
    return (
        SampleInput("tall_4x3", (tensor_arg(a),)),
        SampleInput("wide_3x4", (tensor_arg(a.T.copy()),)),
    )


def _matrix_power_samples() -> Sequence[SampleInput]:
    a = np.array([[1.5, 0.2, -0.1], [-0.1, 1.3, 0.2], [0.1, -0.2, 1.4]], dtype=np.float64)
    return (
        SampleInput("n0", (tensor_arg(a), 0), supports_backward=False, supports_gradcheck=False),
        SampleInput("n2", (tensor_arg(a), 2)),
        SampleInput("n3", (tensor_arg(a), 3)),
        SampleInput("n_neg1", (tensor_arg(a), -1)),
    )


def _matrix_exp_samples() -> Sequence[SampleInput]:
    a_2d = np.array([[0.3, -0.2, 0.1], [0.15, -0.25, 0.2], [-0.1, 0.3, 0.2]], dtype=np.float64)
    a_3d = np.stack([a_2d, a_2d * 0.6], axis=0)
    return (SampleInput("2d", (tensor_arg(a_2d),)), SampleInput("3d_batched", (tensor_arg(a_3d),)))


def _cross_vecdot_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(503)
    a = rng.uniform(-1.5, 1.5, size=(2, 3))
    b = rng.uniform(-1.5, 1.5, size=(2, 3))
    a_3d = rng.uniform(-1.2, 1.2, size=(2, 3, 2))
    b_3d = rng.uniform(-1.2, 1.2, size=(2, 3, 2))
    return (
        SampleInput("dim_neg1_2d", (tensor_arg(a), tensor_arg(b)), {"dim": -1}),
        SampleInput("dim1_3d", (tensor_arg(a_3d), tensor_arg(b_3d)), {"dim": 1}),
    )


def _conv1d_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(504)
    x = rng.uniform(-1.2, 1.2, size=(2, 2, 7))
    w = rng.uniform(-0.8, 0.8, size=(3, 2, 3))
    b = rng.uniform(-0.4, 0.4, size=(3,))
    w_g2 = rng.uniform(-0.8, 0.8, size=(2, 1, 3))
    return (
        SampleInput(
            "s1_p1_bias", (tensor_arg(x), tensor_arg(w), tensor_arg(b)), {"stride": 1, "padding": 1}
        ),
        SampleInput(
            "s2_d2_g2",
            (tensor_arg(x), tensor_arg(w_g2)),
            {"stride": 2, "padding": 1, "dilation": 2, "groups": 2},
        ),
    )


def _conv_transpose2d_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(505)
    x = rng.uniform(-1.0, 1.0, size=(1, 2, 3, 3))
    w = rng.uniform(-0.7, 0.7, size=(2, 3, 2, 2))
    b = rng.uniform(-0.3, 0.3, size=(3,))
    return (
        SampleInput(
            "s1_bias", (tensor_arg(x), tensor_arg(w), tensor_arg(b)), {"stride": 1, "padding": 0}
        ),
        SampleInput("s2_p1", (tensor_arg(x), tensor_arg(w)), {"stride": 2, "padding": 1}),
    )


def _bilinear_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(506)
    x1 = rng.uniform(-1.2, 1.2, size=(3, 2))
    x2 = rng.uniform(-1.2, 1.2, size=(3, 3))
    w = rng.uniform(-0.8, 0.8, size=(4, 2, 3))
    b = rng.uniform(-0.4, 0.4, size=(4,))
    return (
        SampleInput("with_bias", (tensor_arg(x1), tensor_arg(x2), tensor_arg(w), tensor_arg(b))),
        SampleInput("no_bias", (tensor_arg(x1), tensor_arg(x2), tensor_arg(w))),
    )


def _adaptive_avg_pool2d_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(507)
    x = rng.uniform(-1.5, 1.5, size=(1, 2, 5, 6))
    return (
        SampleInput("out_1x1", (tensor_arg(x), (1, 1))),
        SampleInput("out_2x3", (tensor_arg(x), (2, 3))),
    )


def _pad_samples() -> Sequence[SampleInput]:
    x = np.linspace(-1.2, 1.5, num=18, dtype=np.float64).reshape(1, 2, 3, 3)
    return (
        SampleInput("constant", (tensor_arg(x), (1, 1, 1, 1)), {"mode": "constant", "value": 0.25}),
        SampleInput("reflect", (tensor_arg(x), (1, 2, 1, 1)), {"mode": "reflect"}),
        SampleInput("replicate", (tensor_arg(x), (1, 1, 2, 0)), {"mode": "replicate"}),
        SampleInput("circular", (tensor_arg(x), (1, 1, 1, 1)), {"mode": "circular"}),
    )


def _pixel_shuffle_samples() -> Sequence[SampleInput]:
    x = np.linspace(-1.5, 1.5, num=32, dtype=np.float64).reshape(1, 8, 2, 2)
    return (SampleInput("r2", (tensor_arg(x), 2)),)


def _pixel_unshuffle_samples() -> Sequence[SampleInput]:
    x = np.linspace(-1.5, 1.5, num=32, dtype=np.float64).reshape(1, 2, 4, 4)
    return (SampleInput("r2", (tensor_arg(x), 2)),)


def _cosine_distance_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(508)
    x1 = rng.uniform(0.3, 1.8, size=(3, 4))
    x2 = rng.uniform(0.3, 1.8, size=(3, 4))
    return (
        SampleInput("dim_neg1", (tensor_arg(x1), tensor_arg(x2)), {"dim": -1}),
        SampleInput("dim0", (tensor_arg(x1), tensor_arg(x2)), {"dim": 0}),
    )


def _normalize_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(509)
    x = rng.uniform(0.3, 1.8, size=(3, 4))
    return (
        SampleInput("p2_dim1", (tensor_arg(x),), {"p": 2.0, "dim": 1}),
        SampleInput("p1_dim0", (tensor_arg(x),), {"p": 1.0, "dim": 0}),
    )


def _kl_div_samples() -> Sequence[SampleInput]:
    log_p = np.array([[-1.2, -0.8, -1.5], [-0.5, -1.4, -1.9]], dtype=np.float64)
    q = np.array([[0.3, 0.5, 0.2], [0.6, 0.25, 0.15]], dtype=np.float64)
    log_q = np.log(q)
    return (
        SampleInput(
            "prob_target_batchmean",
            (tensor_arg(log_p), tensor_arg(q, differentiable=False)),
            {"reduction": "batchmean", "log_target": False},
        ),
        SampleInput(
            "log_target_sum",
            (tensor_arg(log_p), tensor_arg(log_q, differentiable=False)),
            {"reduction": "sum", "log_target": True},
        ),
    )


def _nll_loss_samples() -> Sequence[SampleInput]:
    log_probs = np.array(
        [[-0.8, -1.2, -2.1, -1.5], [-1.9, -0.5, -1.7, -2.0], [-1.1, -1.6, -0.7, -1.8]],
        dtype=np.float64,
    )
    target = np.array([0, 1, 2], dtype=np.int64)
    return (
        SampleInput("mean", (tensor_arg(log_probs), index_arg(target)), {"reduction": "mean"}),
        SampleInput("sum", (tensor_arg(log_probs), index_arg(target)), {"reduction": "sum"}),
        SampleInput("none", (tensor_arg(log_probs), index_arg(target)), {"reduction": "none"}),
    )


def _margin_ranking_samples() -> Sequence[SampleInput]:
    x1 = np.array([0.8, -0.5, 1.2, -0.9], dtype=np.float64)
    x2 = np.array([-0.2, 0.4, 0.1, -0.1], dtype=np.float64)
    y = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    return (
        SampleInput(
            "mean_m05",
            (tensor_arg(x1), tensor_arg(x2), tensor_arg(y, differentiable=False)),
            {"margin": 0.5, "reduction": "mean"},
        ),
        SampleInput(
            "sum_m02",
            (tensor_arg(x1), tensor_arg(x2), tensor_arg(y, differentiable=False)),
            {"margin": 0.2, "reduction": "sum"},
        ),
    )


def _triplet_margin_samples() -> Sequence[SampleInput]:
    rng = np.random.default_rng(510)
    a = rng.uniform(-1.0, 1.0, size=(3, 4))
    p = a + rng.uniform(0.1, 0.3, size=(3, 4))
    n = a - rng.uniform(0.4, 0.8, size=(3, 4))
    return (
        SampleInput(
            "mean_p2",
            (tensor_arg(a), tensor_arg(p), tensor_arg(n)),
            {"margin": 1.0, "p": 2.0, "reduction": "mean"},
        ),
        SampleInput(
            "sum_swap",
            (tensor_arg(a), tensor_arg(p), tensor_arg(n)),
            {"margin": 0.8, "p": 2.0, "swap": True, "reduction": "sum"},
        ),
    )


FFT_EXT_OPINFOS: tuple[OpInfo, ...] = (
    OpInfo(
        name="fft.fft_abs",
        family="fft",
        lucid_fn=lambda x, **kw: lfft.fft(x, **kw).abs(),
        ref_fn=lambda r, x, **kw: r.fft.fft(x, **kw).abs(),
        sample_inputs_fn=_fft_1d_samples,
        supports_half=False,
    ),
    OpInfo(
        name="fft.rfft_abs",
        family="fft",
        lucid_fn=lambda x, **kw: lfft.rfft(x, **kw).abs(),
        ref_fn=lambda r, x, **kw: r.fft.rfft(x, **kw).abs(),
        sample_inputs_fn=_fft_1d_samples,
        supports_half=False,
    ),
    OpInfo(
        name="fft.irfft",
        family="fft",
        lucid_fn=lambda x, **kw: lfft.irfft(lfft.rfft(x, **kw), **kw),
        ref_fn=lambda r, x, **kw: r.fft.irfft(r.fft.rfft(x, **kw), **kw),
        sample_inputs_fn=_fft_1d_samples,
        supports_half=False,
    ),
    OpInfo(
        name="fft.rfft2_abs",
        family="fft",
        lucid_fn=lambda x, **kw: lfft.rfft2(x, **kw).abs(),
        ref_fn=lambda r, x, **kw: r.fft.rfft2(x, **kw).abs(),
        sample_inputs_fn=_fft_2d_samples,
        supports_half=False,
    ),
    OpInfo(
        name="fft.fftshift",
        family="fft",
        lucid_fn=lfft.fftshift,
        ref_fn=lambda r, x, **kw: r.fft.fftshift(x, **kw),
        sample_inputs_fn=_fftshift_samples,
    ),
    OpInfo(
        name="fft.ifftshift",
        family="fft",
        lucid_fn=lfft.ifftshift,
        ref_fn=lambda r, x, **kw: r.fft.ifftshift(x, **kw),
        sample_inputs_fn=_fftshift_samples,
    ),
    OpInfo(
        name="linalg.eigvalsh",
        family="linalg",
        lucid_fn=llinalg.eigvalsh,
        ref_fn=lambda r, x, **kw: r.linalg.eigvalsh(x, **kw),
        sample_inputs_fn=_eigvalsh_samples,
        supports_half=False,
        supports_gradcheck=False,
    ),
    OpInfo(
        name="linalg.svdvals",
        family="linalg",
        lucid_fn=llinalg.svdvals,
        ref_fn=lambda r, x: r.linalg.svdvals(x),
        sample_inputs_fn=_svdvals_samples,
        supports_half=False,
    ),
    OpInfo(
        name="linalg.matrix_power",
        family="linalg",
        lucid_fn=llinalg.matrix_power,
        ref_fn=lambda r, x, n: r.linalg.matrix_power(x, n),
        sample_inputs_fn=_matrix_power_samples,
        supports_half=False,
    ),
    OpInfo(
        name="linalg.matrix_exp",
        family="linalg",
        lucid_fn=llinalg.matrix_exp,
        ref_fn=lambda r, x: r.linalg.matrix_exp(x),
        sample_inputs_fn=_matrix_exp_samples,
        supports_half=False,
    ),
    OpInfo(
        name="linalg.cross",
        family="linalg",
        lucid_fn=llinalg.cross,
        ref_fn=lambda r, a, b, **kw: r.linalg.cross(a, b, **kw),
        sample_inputs_fn=_cross_vecdot_samples,
    ),
    OpInfo(
        name="linalg.vecdot",
        family="linalg",
        lucid_fn=llinalg.vecdot,
        ref_fn=lambda r, a, b, **kw: r.linalg.vecdot(a, b, **kw),
        sample_inputs_fn=_cross_vecdot_samples,
    ),
    OpInfo(
        name="F.conv1d",
        family="nn_layer",
        lucid_fn=F.conv1d,
        ref_fn=lambda r, *a, **kw: r.nn.functional.conv1d(*a, **kw),
        sample_inputs_fn=_conv1d_samples,
        atol=2e-4,
        bwd_atol=5e-4,
    ),
    OpInfo(
        name="F.conv_transpose2d",
        family="nn_layer",
        lucid_fn=F.conv_transpose2d,
        ref_fn=lambda r, *a, **kw: r.nn.functional.conv_transpose2d(*a, **kw),
        sample_inputs_fn=_conv_transpose2d_samples,
        atol=2e-4,
        bwd_atol=5e-4,
    ),
    OpInfo(
        name="F.bilinear",
        family="nn_layer",
        lucid_fn=F.bilinear,
        ref_fn=lambda r, *a, **kw: r.nn.functional.bilinear(*a, **kw),
        sample_inputs_fn=_bilinear_samples,
    ),
    OpInfo(
        name="F.adaptive_avg_pool2d",
        family="nn_pool",
        lucid_fn=F.adaptive_avg_pool2d,
        ref_fn=lambda r, x, s: r.nn.functional.adaptive_avg_pool2d(x, s),
        sample_inputs_fn=_adaptive_avg_pool2d_samples,
    ),
    OpInfo(
        name="F.pad",
        family="nn_spatial",
        lucid_fn=F.pad,
        ref_fn=lambda r, x, p, **kw: r.nn.functional.pad(x, p, **kw),
        sample_inputs_fn=_pad_samples,
    ),
    OpInfo(
        name="F.pixel_shuffle",
        family="nn_spatial",
        lucid_fn=F.pixel_shuffle,
        ref_fn=lambda r, x, f: r.nn.functional.pixel_shuffle(x, f),
        sample_inputs_fn=_pixel_shuffle_samples,
    ),
    OpInfo(
        name="F.pixel_unshuffle",
        family="nn_spatial",
        lucid_fn=F.pixel_unshuffle,
        ref_fn=lambda r, x, f: r.nn.functional.pixel_unshuffle(x, f),
        sample_inputs_fn=_pixel_unshuffle_samples,
    ),
    OpInfo(
        name="F.cosine_similarity",
        family="nn_layer",
        lucid_fn=F.cosine_similarity,
        ref_fn=lambda r, x1, x2, **kw: r.nn.functional.cosine_similarity(x1, x2, **kw),
        sample_inputs_fn=_cosine_distance_samples,
    ),
    OpInfo(
        name="F.pairwise_distance",
        family="nn_layer",
        lucid_fn=lambda x1, x2, **kw: F.pairwise_distance(x1, x2),
        ref_fn=lambda r, x1, x2, **kw: r.nn.functional.pairwise_distance(x1, x2),
        sample_inputs_fn=_cosine_distance_samples,
    ),
    OpInfo(
        name="F.normalize",
        family="nn_norm",
        lucid_fn=F.normalize,
        ref_fn=lambda r, x, **kw: r.nn.functional.normalize(x, **kw),
        sample_inputs_fn=_normalize_samples,
    ),
    OpInfo(
        name="F.kl_div",
        family="nn_loss",
        lucid_fn=F.kl_div,
        ref_fn=lambda r, x, y, **kw: r.nn.functional.kl_div(x, y, **kw),
        sample_inputs_fn=_kl_div_samples,
    ),
    OpInfo(
        name="F.nll_loss",
        family="nn_loss",
        lucid_fn=F.nll_loss,
        ref_fn=lambda r, x, y, **kw: r.nn.functional.nll_loss(x, y, **kw),
        sample_inputs_fn=_nll_loss_samples,
    ),
    OpInfo(
        name="F.margin_ranking_loss",
        family="nn_loss",
        lucid_fn=F.margin_ranking_loss,
        ref_fn=lambda r, x1, x2, y, **kw: r.nn.functional.margin_ranking_loss(x1, x2, y, **kw),
        sample_inputs_fn=_margin_ranking_samples,
    ),
    OpInfo(
        name="F.triplet_margin_loss",
        family="nn_loss",
        lucid_fn=F.triplet_margin_loss,
        ref_fn=lambda r, anchor, pos, neg, **kw: r.nn.functional.triplet_margin_loss(
            anchor, pos, neg, **kw
        ),
        sample_inputs_fn=_triplet_margin_samples,
    ),
)
