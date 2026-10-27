"""Logical per-sample shape and reduction parity tests for ``lucid.func.vmap`` (LCD-254).

Verifies that ``vmap`` (in default ``strategy="auto"``) presents the wrapped
function with logical per-sample shapes rather than physical batched shapes:
1. Per-sample ``vmap(grad(...))`` and forward ``vmap(...)`` parity against
   ``strategy="isolated"`` and ``ref.func.vmap`` across
   ``{cross_entropy, nll_loss, mse_loss, binary_cross_entropy, kl_div}`` ×
   ``{mean, sum, none}`` × ``{cpu, metal}``.
2. Mixed ``in_dims=(0, None)`` and ``(None, 0, 0)`` with ``unsqueeze(0)`` and
   same-shape-validating losses (``binary_cross_entropy``,
   ``binary_cross_entropy_with_logits``).
3. Shape-observing functions (``x.shape``, ``x.ndim``, ``x.numel()``,
   unqualified ``x.mean()``, ``x.sum()``) and nested transforms
   (``grad_and_value``, ``vjp``, ``jvp``) inside ``vmap``.
"""

from types import ModuleType
from typing import Callable

import numpy as np
import pytest

import lucid
import lucid.func as func
import lucid.nn.functional as F
from lucid.test._helpers.compare import assert_close

_LOSS_NAMES = ("cross_entropy", "nll_loss", "mse_loss", "binary_cross_entropy", "kl_div")
_REDUCTIONS = ("mean", "sum", "none")


def _make_loss_pair(
    loss_name: str, reduction: str, ref: ModuleType
) -> tuple[
    Callable[[lucid.Tensor, lucid.Tensor, lucid.Tensor], lucid.Tensor],
    Callable[[object, object, object], object],
    bool,
]:
    """Return ``(lucid_loss_fn, ref_loss_fn, uses_class_indices)`` for scalar per-sample grad."""
    rf = ref.nn.functional

    if loss_name == "cross_entropy":

        def l_fn(w: lucid.Tensor, x: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
            return F.cross_entropy(x @ w, t, reduction=reduction)

        def r_fn(w: object, x: object, t: object) -> object:
            return rf.cross_entropy(x @ w, t, reduction=reduction)

        return l_fn, r_fn, True

    if loss_name == "nll_loss":

        def l_fn(w: lucid.Tensor, x: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
            return F.nll_loss(F.log_softmax(x @ w, dim=-1), t, reduction=reduction)

        def r_fn(w: object, x: object, t: object) -> object:
            return rf.nll_loss(rf.log_softmax(x @ w, dim=-1), t, reduction=reduction)

        return l_fn, r_fn, True

    if loss_name == "mse_loss":

        def l_fn(w: lucid.Tensor, x: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
            out = F.mse_loss(x @ w, t, reduction=reduction)
            return out.mean() if reduction == "none" else out

        def r_fn(w: object, x: object, t: object) -> object:
            out = rf.mse_loss(x @ w, t, reduction=reduction)
            return out.mean() if reduction == "none" else out

        return l_fn, r_fn, False

    if loss_name == "binary_cross_entropy":

        def l_fn(w: lucid.Tensor, x: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
            out = F.binary_cross_entropy(lucid.sigmoid(x @ w), t, reduction=reduction)
            return out.mean() if reduction == "none" else out

        def r_fn(w: object, x: object, t: object) -> object:
            out = rf.binary_cross_entropy(ref.sigmoid(x @ w), t, reduction=reduction)
            return out.mean() if reduction == "none" else out

        return l_fn, r_fn, False

    def l_kl(w: lucid.Tensor, x: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
        out = F.kl_div(F.log_softmax(x @ w, dim=-1), t, reduction=reduction)
        return out.mean() if reduction == "none" else out

    def r_kl(w: object, x: object, t: object) -> object:
        out = rf.kl_div(rf.log_softmax(x @ w, dim=-1), t, reduction=reduction)
        return out.mean() if reduction == "none" else out

    return l_kl, r_kl, False


@pytest.mark.parity
@pytest.mark.parametrize("loss_name", _LOSS_NAMES)
@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_vmap_grad_per_sample_losses_parity(
    ref: ModuleType, device: str, loss_name: str, reduction: str
) -> None:
    """Per-sample ``vmap(grad(...))`` matches ``strategy='isolated'`` and ``ref``."""
    rng = np.random.default_rng(42)
    b_sz, d_in, n_cls = 4, 5, 3
    w_np = rng.standard_normal((d_in, n_cls)).astype(np.float32)
    x_np = rng.standard_normal((b_sz, d_in)).astype(np.float32)

    l_fn, r_fn, is_cls = _make_loss_pair(loss_name, reduction, ref)
    if is_cls:
        t_np = rng.integers(0, n_cls, size=(b_sz,)).astype(np.int64)
        t_l = lucid.tensor(t_np, dtype=lucid.int64, device=device)
        t_r = ref.tensor(t_np, dtype=ref.int64)
    else:
        t_np = rng.uniform(0.15, 0.85, size=(b_sz, n_cls)).astype(np.float32)
        t_l = lucid.tensor(t_np, device=device)
        t_r = ref.tensor(t_np)

    w_l = lucid.tensor(w_np, device=device)
    x_l = lucid.tensor(x_np, device=device)
    w_r = ref.tensor(w_np)
    x_r = ref.tensor(x_np)

    grad_l = func.grad(l_fn, argnums=(0, 1))
    grad_r = ref.func.grad(r_fn, argnums=(0, 1))

    gw_auto, gx_auto = func.vmap(grad_l, in_dims=(None, 0, 0))(w_l, x_l, t_l)
    gw_iso, gx_iso = func.vmap(grad_l, in_dims=(None, 0, 0), strategy="isolated")(w_l, x_l, t_l)
    gw_ref, gx_ref = ref.func.vmap(grad_r, in_dims=(None, 0, 0))(w_r, x_r, t_r)

    assert tuple(gw_auto.shape) == (b_sz, d_in, n_cls)
    assert tuple(gx_auto.shape) == (b_sz, d_in)
    assert_close(gw_auto, gw_iso, atol=1e-5)
    assert_close(gx_auto, gx_iso, atol=1e-5)
    assert_close(gw_auto, gw_ref, atol=1e-5)
    assert_close(gx_auto, gx_ref, atol=1e-5)


def _make_forward_loss_pair(
    loss_name: str, reduction: str, ref: ModuleType
) -> tuple[
    Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor], Callable[[object, object], object], bool
]:
    rf = ref.nn.functional
    if loss_name == "cross_entropy":
        return (
            lambda x, t: F.cross_entropy(x, t, reduction=reduction),
            lambda x, t: rf.cross_entropy(x, t, reduction=reduction),
            True,
        )
    if loss_name == "nll_loss":
        return (
            lambda x, t: F.nll_loss(F.log_softmax(x, dim=-1), t, reduction=reduction),
            lambda x, t: rf.nll_loss(rf.log_softmax(x, dim=-1), t, reduction=reduction),
            True,
        )
    if loss_name == "mse_loss":
        return (
            lambda x, t: F.mse_loss(x, t, reduction=reduction),
            lambda x, t: rf.mse_loss(x, t, reduction=reduction),
            False,
        )
    if loss_name == "binary_cross_entropy":
        return (
            lambda x, t: F.binary_cross_entropy(lucid.sigmoid(x), t, reduction=reduction),
            lambda x, t: rf.binary_cross_entropy(ref.sigmoid(x), t, reduction=reduction),
            False,
        )
    return (
        lambda x, t: F.kl_div(F.log_softmax(x, dim=-1), t, reduction=reduction),
        lambda x, t: rf.kl_div(rf.log_softmax(x, dim=-1), t, reduction=reduction),
        False,
    )


@pytest.mark.parity
@pytest.mark.parametrize("loss_name", _LOSS_NAMES)
@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_vmap_forward_losses_parity(
    ref: ModuleType, device: str, loss_name: str, reduction: str
) -> None:
    """Forward ``vmap(loss_fn)`` reduces per sample, not across the batch axis."""
    rng = np.random.default_rng(7)
    b_sz, n_cls = 4, 3
    logits_np = rng.standard_normal((b_sz, n_cls)).astype(np.float32)

    f_l, f_r, is_cls = _make_forward_loss_pair(loss_name, reduction, ref)
    if is_cls:
        t_np = rng.integers(0, n_cls, size=(b_sz,)).astype(np.int64)
        t_l = lucid.tensor(t_np, dtype=lucid.int64, device=device)
        t_r = ref.tensor(t_np, dtype=ref.int64)
    else:
        t_np = rng.uniform(0.15, 0.85, size=(b_sz, n_cls)).astype(np.float32)
        t_l = lucid.tensor(t_np, device=device)
        t_r = ref.tensor(t_np)

    x_l = lucid.tensor(logits_np, device=device)
    x_r = ref.tensor(logits_np)

    out_auto = func.vmap(f_l)(x_l, t_l)
    out_iso = func.vmap(f_l, strategy="isolated")(x_l, t_l)
    out_ref = ref.func.vmap(f_r)(x_r, t_r)

    assert tuple(out_auto.shape) == tuple(out_ref.shape)
    assert_close(out_auto, out_iso, atol=1e-5)
    assert_close(out_auto, out_ref, atol=1e-5)


@pytest.mark.parity
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_vmap_grad_cross_entropy_with_unsqueeze0(
    ref: ModuleType, device: str, reduction: str
) -> None:
    """LCD-254 repro: ``unsqueeze(0)`` inside ``vmap(grad(...))`` sees 1-D per-sample inputs."""
    rng = np.random.default_rng(11)
    b_sz, d_in, n_cls = 4, 5, 3
    w_np = rng.standard_normal((d_in, n_cls)).astype(np.float32)
    x_np = rng.standard_normal((b_sz, d_in)).astype(np.float32)
    t_np = rng.integers(0, n_cls, size=(b_sz,)).astype(np.int64)

    w_l = lucid.tensor(w_np, device=device)
    x_l = lucid.tensor(x_np, device=device)
    t_l = lucid.tensor(t_np, dtype=lucid.int64, device=device)

    w_r = ref.tensor(w_np)
    x_r = ref.tensor(x_np)
    t_r = ref.tensor(t_np, dtype=ref.int64)

    def loss_l(w: lucid.Tensor, x: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
        return F.cross_entropy((x @ w).unsqueeze(0), t.unsqueeze(0), reduction=reduction)

    def loss_r(w: object, x: object, t: object) -> object:
        return ref.nn.functional.cross_entropy(
            (x @ w).unsqueeze(0), t.unsqueeze(0), reduction=reduction
        )

    gw_l, gx_l = func.vmap(func.grad(loss_l, argnums=(0, 1)), in_dims=(None, 0, 0))(w_l, x_l, t_l)
    gw_r, gx_r = ref.func.vmap(ref.func.grad(loss_r, argnums=(0, 1)), in_dims=(None, 0, 0))(
        w_r, x_r, t_r
    )

    assert_close(gw_l, gw_r, atol=1e-5)
    assert_close(gx_l, gx_r, atol=1e-5)


@pytest.mark.parity
@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_vmap_mixed_in_dims_unbatched_target_bce(
    ref: ModuleType, device: str, reduction: str
) -> None:
    """LCD-254 repro: ``in_dims=(0, None)`` presents matching per-sample target shapes to BCE."""
    rng = np.random.default_rng(19)
    b_sz, d_ = 5, 4
    p_np = rng.uniform(0.1, 0.9, size=(b_sz, d_)).astype(np.float32)
    y_np = rng.uniform(0.1, 0.9, size=(d_,)).astype(np.float32)

    p_l = lucid.tensor(p_np, device=device)
    y_l = lucid.tensor(y_np, device=device)
    p_r = ref.tensor(p_np)
    y_r = ref.tensor(y_np)

    out_l = func.vmap(
        lambda p, t: F.binary_cross_entropy(p, t, reduction=reduction), in_dims=(0, None)
    )(p_l, y_l)
    out_r = ref.func.vmap(
        lambda p, t: ref.nn.functional.binary_cross_entropy(p, t, reduction=reduction),
        in_dims=(0, None),
    )(p_r, y_r)
    assert_close(out_l, out_r, atol=1e-5)

    out_logits_l = func.vmap(
        lambda x, t: F.binary_cross_entropy_with_logits(x, t, reduction=reduction),
        in_dims=(0, None),
    )(p_l, y_l)
    out_logits_r = ref.func.vmap(
        lambda x, t: ref.nn.functional.binary_cross_entropy_with_logits(x, t, reduction=reduction),
        in_dims=(0, None),
    )(p_r, y_r)
    assert_close(out_logits_l, out_logits_r, atol=1e-5)


@pytest.mark.parity
def test_vmap_shape_observing_and_unqualified_reductions(ref: ModuleType, device: str) -> None:
    """Inside ``vmap``, ``x.shape``, ``x.ndim``, ``x.numel()``, ``x.mean()``, ``x.sum()`` see logical shapes."""
    rng = np.random.default_rng(23)
    x_np = rng.standard_normal((4, 3, 2)).astype(np.float32)
    x_l = lucid.tensor(x_np, device=device)
    x_r = ref.tensor(x_np)

    seen_shapes: list[tuple[int, ...]] = []
    seen_ndims: list[int] = []
    seen_numels: list[int] = []

    def observe(x: lucid.Tensor) -> lucid.Tensor:
        seen_shapes.append(tuple(x.shape))
        seen_ndims.append(x.ndim)
        seen_numels.append(x.numel())
        return lucid.stack([x.mean(), x.sum()])

    out_l = func.vmap(observe)(x_l)
    out_r = ref.func.vmap(lambda x: ref.stack([x.mean(), x.sum()]))(x_r)

    assert seen_shapes == [(3, 2)] * 4
    assert seen_ndims == [2] * 4
    assert seen_numels == [6] * 4
    assert tuple(out_l.shape) == (4, 2)
    assert_close(out_l, out_r, atol=1e-5)


@pytest.mark.parity
def test_vmap_composes_with_grad_and_value_vjp_jvp(ref: ModuleType, device: str) -> None:
    """``vmap`` over ``grad_and_value``, ``vjp``, and ``jvp`` operates on logical per-sample shapes."""
    rng = np.random.default_rng(31)
    x_np = rng.standard_normal((4, 3)).astype(np.float32)
    y_np = rng.standard_normal((4, 3)).astype(np.float32)

    x_l = lucid.tensor(x_np, device=device)
    y_l = lucid.tensor(y_np, device=device)
    x_r = ref.tensor(x_np)
    y_r = ref.tensor(y_np)

    gv_l = func.grad_and_value(lambda a, b: F.mse_loss(a, b, reduction="mean"))
    gv_r = ref.func.grad_and_value(lambda a, b: ref.nn.functional.mse_loss(a, b, reduction="mean"))
    g_l, v_l = func.vmap(gv_l)(x_l, y_l)
    g_r, v_r = ref.func.vmap(gv_r)(x_r, y_r)
    assert_close(g_l, g_r, atol=1e-5)
    assert_close(v_l, v_r, atol=1e-5)

    def vjp_step(a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
        out, vjp_fn = func.vjp(lambda z: F.mse_loss(z, b, reduction="mean"), a)
        (g,) = vjp_fn(lucid.ones_like(out))
        assert g is not None
        return g

    g_vjp = func.vmap(vjp_step)(x_l, y_l)
    assert_close(g_vjp, g_r, atol=1e-5)

    def jvp_step(a: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
        _, dy = func.jvp(lambda z: z.mean(), (a,), (t,))
        return dy

    dy_l = func.vmap(jvp_step)(x_l, y_l)
    dy_r = ref.func.vmap(lambda a, t: ref.func.jvp(lambda z: z.mean(), (a,), (t,))[1])(x_r, y_r)
    assert_close(dy_l, dy_r, atol=1e-5)
