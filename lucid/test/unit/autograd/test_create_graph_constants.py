"""A create_graph gradient requires grad only when it depends on the input (LCD-422).

The engine stood a zero-derivative node behind every gradient that came back
detached, so ``grad((a * 2).sum(), a, create_graph=True)[0]`` — the constant
2 — reported ``requires_grad=True``, and ``strict`` could not tell a linear
function from a curved one.  The reference keeps a gradient in the graph when
its backward is an op of the input, even one whose own derivative is zero
(relu, hardtanh, max pooling), and leaves a constant detached.  The zero
derivative is now attached by those formulas, not by the engine.

A Python ``Function`` said "no gradient" with an empty storage, and a CPU
gradient of zero elements owns no memory: its input's ``.grad`` stayed
``None`` where the reference gives an empty tensor.
"""

import pytest

import lucid
import lucid.nn.functional as F
from lucid.autograd import Function

_VALUES = [
    -1.5,
    0.5,
    2.0,
    7.0,
    -0.3,
    3.2,
    1.1,
    -2.4,
    0.7,
    5.5,
    -4.1,
    2.9,
    0.2,
    -0.9,
    6.3,
    1.7,
]


def _values(out):
    return out[0] if isinstance(out, tuple) else out


def _functional(lib):
    return lib.autograd if lib is lucid else lib.autograd.functional


class _Env:
    """What a case needs from either library: the module, its functional
    namespace, and tensors on the device under test."""

    def __init__(self, lib, device):
        self.L = lib
        self.F = F if lib is lucid else lib.nn.functional
        self._kw = {"device": device} if lib is lucid else {}

    def t(self, values):
        return self.L.tensor(values, **self._kw)

    def zeros(self, *shape):
        return self.L.zeros(*shape, **self._kw)


# name -> call(env, x) on a 16-element x.  Every op whose create_graph
# gradient the reference keeps in the graph, every one it leaves detached,
# and the constants in between: what the engine attaches is held to it.
_CASES = {
    # constants: the gradient is detached
    "constant_sum": lambda e, x: x,
    "linear": lambda e, x: x * 2,
    "affine": lambda e, x: x * 3 + 1,
    "mean": lambda e, x: x.mean(),
    "div": lambda e, x: x / 2,
    "neg": lambda e, x: -x,
    "cat": lambda e, x: e.L.cat([x, x]),
    "flip": lambda e, x: e.L.flip(x, [0]),
    "roll": lambda e, x: e.L.roll(x, 2, 0),
    "cumsum": lambda e, x: e.L.cumsum(x, 0),
    "getitem_adv": lambda e, x: x[[0, 2, 2]],
    "index_select": lambda e, x: e.L.index_select(x, 0, e.t([0, 3, 3])),
    "gather": lambda e, x: e.L.gather(x.reshape(4, 4), 1, e.t([[0, 2], [1, 3], [0, 0], [3, 3]])),
    "masked_select": lambda e, x: e.L.masked_select(x, x > 0),
    "masked_fill": lambda e, x: x.masked_fill(x < 0, 0.0),
    "where": lambda e, x: e.L.where(x > 0, x, e.zeros(16)),
    "maximum": lambda e, x: e.L.maximum(x, e.zeros(16)),
    "minimum": lambda e, x: e.L.minimum(x, e.zeros(16)),
    "sort": lambda e, x: _values(e.L.sort(x))[:3],
    "topk": lambda e, x: _values(e.L.topk(x, 3)),
    "kthvalue": lambda e, x: _values(e.L.kthvalue(x, 3)),
    "cummax": lambda e, x: _values(e.L.cummax(x, 0)),
    "max_all": lambda e, x: x.max(),
    "max_dim": lambda e, x: _values(x.reshape(4, 4).max(1)),
    "min_dim": lambda e, x: _values(x.reshape(4, 4).min(1)),
    "amax": lambda e, x: e.L.amax(x.reshape(4, 4), 1),
    "round": lambda e, x: e.L.round(x),
    "floor": lambda e, x: e.L.floor(x),
    "trunc": lambda e, x: e.L.trunc(x),
    "sign": lambda e, x: e.L.sign(x),
    "frac": lambda e, x: e.L.frac(x),
    "fmod": lambda e, x: e.L.fmod(x, 2.0),
    "remainder": lambda e, x: e.L.remainder(x, 2.0),
    "clamp_min": lambda e, x: e.L.clamp(x, min=0.0),
    "clamp_max": lambda e, x: e.L.clamp(x, max=1.0),
    "embedding": lambda e, x: e.F.embedding(e.t([0, 3, 1]), x.reshape(4, 4)),
    "scatter_add": lambda e, x: e.zeros(4).scatter_add(0, e.t([0, 1, 1, 3] * 4), x),
    "index_put": lambda e, x: e.zeros(4).index_put((e.t([0, 2]),), x[:2]),
    "unfold": lambda e, x: e.F.unfold(x.reshape(1, 1, 4, 4), 2),
    "fold": lambda e, x: e.F.fold(x.reshape(1, 4, 4), (3, 3), 2),
    "pixel_shuffle": lambda e, x: e.F.pixel_shuffle(x.reshape(1, 4, 2, 2), 2),
    "pixel_unshuffle": lambda e, x: e.F.pixel_unshuffle(x.reshape(1, 1, 4, 4), 2),
    "pad_constant": lambda e, x: e.F.pad(x.reshape(1, 1, 4, 4), (1, 1, 1, 1)),
    "pad_circular": lambda e, x: e.F.pad(x.reshape(1, 1, 4, 4), (1, 1, 1, 1), mode="circular"),
    "interp_nearest": lambda e, x: e.F.interpolate(
        x.reshape(1, 1, 4, 4), scale_factor=2, mode="nearest"
    ),
    "interp_bilinear": lambda e, x: e.F.interpolate(
        x.reshape(1, 1, 4, 4), scale_factor=2, mode="bilinear", align_corners=False
    ),
    "interp_bicubic": lambda e, x: e.F.interpolate(
        x.reshape(1, 1, 4, 4), scale_factor=2, mode="bicubic", align_corners=False
    ),
    "interp_linear": lambda e, x: e.F.interpolate(
        x.reshape(1, 1, 16), scale_factor=2, mode="linear", align_corners=False
    ),
    "linear_layer": lambda e, x: e.F.linear(x.reshape(4, 4), e.t([[1.0, 2, 3, 4], [0, 1, 0, 1]])),
    "matmul": lambda e, x: x.reshape(4, 4) @ e.t([[1.0], [2.0], [3.0], [4.0]]),
    "hinge": lambda e, x: e.F.hinge_embedding_loss(x, e.t([1.0, -1] * 8)),
    "margin_rank": lambda e, x: e.F.margin_ranking_loss(x, e.zeros(16), e.t([1.0, -1] * 8)),
    # a backward that reads the input: kept in the graph
    "relu": lambda e, x: e.F.relu(x),
    "relu6": lambda e, x: e.F.relu6(x),
    "leaky_relu": lambda e, x: e.F.leaky_relu(x, 0.1),
    "hardtanh": lambda e, x: e.F.hardtanh(x),
    "hardtanh_range": lambda e, x: e.F.hardtanh(x, -2.0, 2.0),
    "hardsigmoid": lambda e, x: e.F.hardsigmoid(x),
    "hardswish": lambda e, x: e.F.hardswish(x),
    "elu": lambda e, x: e.F.elu(x),
    "selu": lambda e, x: e.F.selu(x),
    "celu": lambda e, x: e.F.celu(x),
    "gelu": lambda e, x: e.F.gelu(x),
    "silu": lambda e, x: e.F.silu(x),
    "softplus": lambda e, x: e.F.softplus(x, 1.0, 1.0),
    "softsign": lambda e, x: e.F.softsign(x),
    "logsigmoid": lambda e, x: e.F.logsigmoid(x),
    "tanhshrink": lambda e, x: e.F.tanhshrink(x),
    "glu": lambda e, x: e.F.glu(x, 0),
    "prelu": lambda e, x: e.F.prelu(x.reshape(1, 16), e.t([0.25])),
    "abs": lambda e, x: e.L.abs(x),
    "clamp": lambda e, x: e.L.clamp(x, -1.0, 3.0),
    "quadratic": lambda e, x: x * x,
    "exp": lambda e, x: e.L.exp(x),
    "prod": lambda e, x: x.prod(),
    "cumprod": lambda e, x: e.L.cumprod(x, 0),
    "std": lambda e, x: x.std(),
    "softmax": lambda e, x: e.F.softmax(x, 0),
    "logsumexp": lambda e, x: e.L.logsumexp(x, 0),
    "norm_1": lambda e, x: e.L.linalg.norm(x, 1),
    "norm_inf": lambda e, x: e.L.linalg.norm(x, float("inf")),
    "cdist": lambda e, x: e.L.cdist(x.reshape(4, 4), e.t([[0.0] * 4])),
    "cosine_sim": lambda e, x: e.F.cosine_similarity(x.reshape(4, 4), e.t([[1.0, 2, 3, 4]] * 4)),
    "max_pool1d": lambda e, x: e.F.max_pool1d(x.reshape(1, 1, 16), 2),
    "max_pool2d": lambda e, x: e.F.max_pool2d(x.reshape(1, 1, 4, 4), 2),
    "max_pool3d": lambda e, x: e.F.max_pool3d(x.reshape(1, 1, 2, 2, 4), 2),
    "adaptive_max_pool1d": lambda e, x: e.F.adaptive_max_pool1d(x.reshape(1, 1, 16), 4),
    "adaptive_max_pool2d": lambda e, x: e.F.adaptive_max_pool2d(x.reshape(1, 1, 4, 4), 2),
    "avg_pool1d": lambda e, x: e.F.avg_pool1d(x.reshape(1, 1, 16), 2),
    "avg_pool2d": lambda e, x: e.F.avg_pool2d(x.reshape(1, 1, 4, 4), 2),
    "avg_pool3d": lambda e, x: e.F.avg_pool3d(x.reshape(1, 1, 2, 2, 4), 2),
    "adaptive_avg_pool1d": lambda e, x: e.F.adaptive_avg_pool1d(x.reshape(1, 1, 16), 4),
    "adaptive_avg_pool2d": lambda e, x: e.F.adaptive_avg_pool2d(x.reshape(1, 1, 4, 4), 2),
    "lp_pool1d": lambda e, x: e.F.lp_pool1d(x.reshape(1, 1, 16), 2.0, 2),
    "interp_area": lambda e, x: e.F.interpolate(x.reshape(1, 1, 4, 4), size=(2, 2), mode="area"),
    "conv1d": lambda e, x: e.F.conv1d(x.reshape(1, 1, 16), e.t([[[1.0, -2.0, 0.5]]])),
    "conv2d": lambda e, x: e.F.conv2d(x.reshape(1, 1, 4, 4), e.t([[[[1.0, 2.0], [3.0, 4.0]]]])),
    "conv_transpose2d": lambda e, x: e.F.conv_transpose2d(
        x.reshape(1, 1, 4, 4), e.t([[[[1.0, 2.0], [3.0, 4.0]]]])
    ),
    "grid_sample_nearest": lambda e, x: e.F.grid_sample(
        x.reshape(1, 1, 4, 4),
        e.t([[[[0.1, 0.2], [-0.5, 0.3]]]]),
        mode="nearest",
        align_corners=False,
    ),
    "grid_sample_bilinear": lambda e, x: e.F.grid_sample(
        x.reshape(1, 1, 4, 4),
        e.t([[[[0.1, 0.2], [-0.5, 0.3]]]]),
        mode="bilinear",
        align_corners=False,
    ),
    "interp_nearest3d": lambda e, x: e.F.interpolate(
        x.reshape(1, 1, 2, 2, 4), scale_factor=2, mode="nearest"
    ),
    # LCD-436: eval dropout is the identity and p=0 training dropout a
    # pass-through, so their gradient is a constant; eval batch_norm is
    # affine in x with frozen statistics, and the reference still keeps its
    # gradient in the graph (derivative zero).
    "dropout_eval": lambda e, x: e.F.dropout(x, 0.5, training=False),
    "dropout_p0": lambda e, x: e.F.dropout(x, 0.0, training=True),
    "dropout2d_eval": lambda e, x: e.F.dropout2d(x.reshape(1, 4, 2, 2), 0.5, training=False),
    "alpha_dropout_eval": lambda e, x: e.F.alpha_dropout(x, 0.5, training=False),
    "batch_norm_eval": lambda e, x: e.F.batch_norm(
        x.reshape(4, 4),
        e.t([0.5, -1.0, 0.0, 2.0]),
        e.t([1.0, 0.5, 2.0, 4.0]),
        training=False,
    ),
    "batch_norm_eval_affine": lambda e, x: e.F.batch_norm(
        x.reshape(2, 4, 2),
        e.t([0.5, -1.0, 0.0, 2.0]),
        e.t([1.0, 0.5, 2.0, 4.0]),
        e.t([2.0, -1.0, 0.5, 1.0]),
        e.t([0.1, 0.2, 0.3, 0.4]),
        training=False,
    ),
    "batch_norm_eval_squared": lambda e, x: (
        e.F.batch_norm(
            x.reshape(4, 4),
            e.t([0.5, -1.0, 0.0, 2.0]),
            e.t([1.0, 0.5, 2.0, 4.0]),
            training=False,
        )
        ** 2
    ),
    "hypot": lambda e, x: e.L.hypot(x, e.t([1.0, -0.25, 3.0, 5.0] * 4)),
    "layer_norm": lambda e, x: e.F.layer_norm(x.reshape(4, 4), (4,)),
    "group_norm": lambda e, x: e.F.group_norm(x.reshape(1, 4, 4), 2),
    "local_response_norm": lambda e, x: e.F.local_response_norm(x.reshape(1, 4, 4), 2),
    "l1_loss": lambda e, x: e.F.l1_loss(x, e.zeros(16)),
    "mse_loss": lambda e, x: e.F.mse_loss(x, e.zeros(16)),
    "smooth_l1_loss": lambda e, x: e.F.smooth_l1_loss(x, e.zeros(16)),
    "huber_loss": lambda e, x: e.F.huber_loss(x, e.zeros(16)),
    "huber_loss_delta": lambda e, x: e.F.huber_loss(x, e.zeros(16), delta=0.5),
    "cross_entropy": lambda e, x: e.F.cross_entropy(x.reshape(4, 4), e.t([0, 1, 2, 3])),
    "cross_entropy_smooth": lambda e, x: e.F.cross_entropy(
        x.reshape(4, 4), e.t([0, 1, 2, 3]), label_smoothing=0.1
    ),
    "multi_margin_loss": lambda e, x: e.F.multi_margin_loss(x.reshape(4, 4), e.t([0, 1, 2, 3])),
    "soft_margin_loss": lambda e, x: e.F.soft_margin_loss(x, e.t([1.0, -1] * 8)),
    "kl_div": lambda e, x: e.F.kl_div(x, e.t([0.0625] * 16), reduction="sum"),
    "poisson_nll_loss": lambda e, x: e.F.poisson_nll_loss(x, e.t([1.0] * 16)),
    "bce_with_logits": lambda e, x: e.F.binary_cross_entropy_with_logits(x, e.t([1.0] * 16)),
    "triplet_margin_loss": lambda e, x: e.F.triplet_margin_loss(
        x.reshape(4, 4), e.zeros(4, 4), e.t([[1.0] * 4] * 4)
    ),
    # Python composites that keep the input in the graph (LCD-437)
    "hardshrink": lambda e, x: e.F.hardshrink(x, 0.5),
    "softshrink": lambda e, x: e.F.softshrink(x, 0.5),
    "threshold": lambda e, x: e.F.threshold(x, 0.5, -1.0),
    "rrelu_eval": lambda e, x: e.F.rrelu(x, training=False),
    "pad_reflect": lambda e, x: e.F.pad(x.reshape(1, 1, 4, 4), (1, 1, 1, 1), mode="reflect"),
    "pad_replicate": lambda e, x: e.F.pad(x.reshape(1, 1, 4, 4), (1, 1, 1, 1), mode="replicate"),
    "pad_reflect1d": lambda e, x: e.F.pad(x.reshape(1, 1, 16), (2, 2), mode="reflect"),
    "pad_replicate1d": lambda e, x: e.F.pad(x.reshape(1, 1, 16), (2, 2), mode="replicate"),
    "nll_loss": lambda e, x: e.F.nll_loss(x.reshape(4, 4), e.t([0, 1, 2, 3])),
    "nll_loss_2d": lambda e, x: e.F.nll_loss(x.reshape(1, 4, 2, 2), e.t([[[0, 1], [2, 3]]])),
}

# Lucid's clamp backward attaches a zero derivative so composites over clamp
# stay differentiable twice; the losses built on clamp inherit that.
_CLAMP_FAMILY = {"clamp", "clamp_min", "clamp_max", "hinge", "margin_rank"}


def _first_and_second(env, call):
    x = env.t(_VALUES).requires_grad_(True)
    (g1,) = env.L.autograd.grad(call(env, x).sum(), x, create_graph=True)
    try:
        (g2,) = env.L.autograd.grad(g1.sum(), x)
    except RuntimeError:  # no double backward, or a constant
        return g1, None
    return g1, g2


@pytest.mark.parametrize("name", list(_CASES))
def test_create_graph_gradient_matches_reference(ref, device, name):
    call = _CASES[name]
    want1, want2 = _first_and_second(_Env(ref, device), call)
    got1, got2 = _first_and_second(_Env(lucid, device), call)
    expected = True if name in _CLAMP_FAMILY else want1.requires_grad
    assert got1.requires_grad == expected
    if want2 is None:
        # The reference has no second derivative here (a constant, or no
        # double backward); Lucid may have one, but never for a constant.
        if got2 is not None and not expected:
            pytest.fail("a constant gradient came back differentiable")
        return
    assert got2 is not None, "the reference differentiates this gradient; Lucid refused"
    assert got2.reshape(-1).tolist() == pytest.approx(
        (want2 + 0).reshape(-1).tolist(), rel=1e-3, abs=1e-3
    )


def test_clamp_keeps_a_zero_derivative(device):
    """``hardtanh`` is ``clamp`` in Lucid, and the reference keeps
    hardtanh's gradient in the graph; ``clamp`` keeps it too (the reference's
    clamp detaches it).  Recorded so a change here is a decision."""
    x = lucid.tensor(_VALUES, device=device, requires_grad=True)
    (g,) = lucid.autograd.grad(x.clamp(-1, 1).sum(), x, create_graph=True)
    assert g.requires_grad
    (h,) = lucid.autograd.grad(g.sum(), x)
    assert h.tolist() == [0.0] * len(_VALUES)


@pytest.mark.parametrize(
    "kind,fn",
    [
        ("constant", lambda a: a.sum() * 0 + 3.0),
        ("linear", lambda a: (a * 2).sum()),
        ("quadratic", lambda a: (a * a).sum()),
    ],
    ids=["constant", "linear", "quadratic"],
)
def test_hessian_strict_matches_reference(ref, device, kind, fn):
    def run(lib, x):
        try:
            return _functional(lib).hessian(fn, x, strict=True), None
        except RuntimeError as err:
            return None, err

    want, want_err = run(ref, ref.tensor([1.0, 2.0], dtype=ref.float64))
    got, got_err = run(lucid, lucid.tensor([1.0, 2.0], dtype=lucid.float32, device=device))
    assert (got_err is None) == (want_err is None), (kind, got_err, want_err)
    if want_err is None:
        assert got.flatten().tolist() == pytest.approx(want.flatten().tolist())


@pytest.mark.parametrize(
    "kind,fn",
    [
        ("linear", lambda a: a * 2),
        ("affine", lambda a: a * 3 + 1),
        ("quadratic", lambda a: a * a),
    ],
    ids=["linear", "affine", "quadratic"],
)
def test_jacobian_strict_create_graph_matches_reference(ref, device, kind, fn):
    def run(lib, x):
        try:
            return (
                _functional(lib).jacobian(fn, x, create_graph=True, strict=True),
                None,
            )
        except RuntimeError as err:
            return None, err

    want, want_err = run(ref, ref.tensor([1.0, 2.0], dtype=ref.float64))
    got, got_err = run(lucid, lucid.tensor([1.0, 2.0], dtype=lucid.float32, device=device))
    assert (got_err is None) == (want_err is None), (kind, got_err, want_err)
    if want_err is None:
        assert got.flatten().tolist() == pytest.approx(want.flatten().tolist())
        assert got.requires_grad == want.requires_grad


# ── a Python Function over an empty input ────────────────────────────────────


class _Identity(Function):
    @staticmethod
    def forward(ctx, x):
        return x * 1

    @staticmethod
    def backward(ctx, g):
        return g


class _FirstOnly(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x * 1

    @staticmethod
    def backward(ctx, g):
        return g, None


@pytest.mark.parametrize("shape", [(0,), (0, 0), (0, 3), (2, 0, 4)], ids=str)
def test_function_gives_an_empty_input_an_empty_gradient(device, shape):
    x = lucid.zeros(*shape, device=device, requires_grad=True)
    _Identity.apply(x).sum().backward()
    assert x.grad is not None
    assert tuple(x.grad.shape) == shape
    assert x.grad.device == x.device
    # Accumulates into what is there.
    _Identity.apply(x).sum().backward()
    assert tuple(x.grad.shape) == shape
    (g,) = lucid.autograd.grad(_Identity.apply(x).sum(), x)
    assert tuple(g.shape) == shape


@pytest.mark.parametrize("shape", [(0, 3), (3,)], ids=str)
def test_function_none_still_means_no_gradient(device, shape):
    x = lucid.zeros(*shape, device=device, requires_grad=True)
    y = lucid.zeros(*shape, device=device, requires_grad=True)
    _FirstOnly.apply(x, y).sum().backward()
    assert x.grad is not None and tuple(x.grad.shape) == shape
    assert y.grad is None
