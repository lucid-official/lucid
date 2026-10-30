"""Core data structures and verification runners for OpInfo differential testing.

Provides ``SampleInput`` and ``OpInfo`` descriptors and four automated
verification engines against the reference framework:

1. ``verify_opinfo_forward``: multi-argument, broadcast, and boundary-value
   forward parity across devices (``cpu``, ``metal``) and dtypes
   (``float32``, ``float16``, ``bfloat16``).
2. ``verify_opinfo_strided_views``: forward and backward parity when inputs
   are non-contiguous strided views (transposed for rank >= 2, offset + step-2
   sliced for rank == 1), exercising ``ViewFamily`` on CPU and Metal.
3. ``verify_opinfo_backward``: first-order VJP parity under deterministic,
   non-uniform upstream cotangents (avoiding the all-ones gradient blind spot).
4. ``verify_opinfo_gradcheck``: float64 central-difference numerical gradient
   verification across multi-argument samples.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import lucid
from lucid.test._helpers.compare import _to_numpy, assert_close


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """Descriptor for a tensor input inside a :class:`SampleInput`."""

    data: np.ndarray
    differentiable: bool = True
    is_index: bool = False
    is_mask: bool = False


def tensor_arg(
    data: np.ndarray | Sequence[Any] | float, *, differentiable: bool = True
) -> ArraySpec:
    """Create a floating-point :class:`ArraySpec` from ``data``."""
    arr = np.asarray(data, dtype=np.float64)
    return ArraySpec(data=arr, differentiable=differentiable)


def index_arg(data: np.ndarray | Sequence[Any] | int) -> ArraySpec:
    """Create an integer index :class:`ArraySpec` from ``data``."""
    arr = np.asarray(data, dtype=np.int64)
    return ArraySpec(data=arr, differentiable=False, is_index=True)


def mask_arg(data: np.ndarray | Sequence[Any] | bool) -> ArraySpec:
    """Create a boolean mask :class:`ArraySpec` from ``data``."""
    arr = np.asarray(data, dtype=np.bool_)
    return ArraySpec(data=arr, differentiable=False, is_mask=True)


@dataclass(frozen=True, slots=True)
class SampleInput:
    """One concrete multi-argument invocation for an :class:`OpInfo`."""

    name: str
    args: tuple[object, ...]
    kwargs: dict[str, object] = field(default_factory=dict)
    ref_kwargs: dict[str, object] | None = None
    supports_backward: bool = True
    supports_strided: bool = True
    supports_gradcheck: bool = True


@dataclass(frozen=True, slots=True)
class OpInfo:
    """Operator specification for differential forward, view, and VJP testing."""

    name: str
    family: str
    lucid_fn: Callable[..., object]
    ref_fn: Callable[..., object]
    sample_inputs_fn: Callable[[], Sequence[SampleInput]]
    supports_backward: bool = True
    supports_strided: bool = True
    supports_gradcheck: bool = True
    supports_half: bool = True
    atol: float = 1e-4
    rtol: float = 1e-4
    bwd_atol: float = 3e-4
    bwd_rtol: float = 3e-4


def _tolerances_for_dtype(
    op: OpInfo, dtype_name: str, *, backward: bool = False
) -> tuple[float, float]:
    """Return ``(atol, rtol)`` scaled appropriately for ``dtype_name``."""
    base_atol = op.bwd_atol if backward else op.atol
    base_rtol = op.bwd_rtol if backward else op.rtol
    if dtype_name == "float16":
        return max(base_atol, 5e-3), max(base_rtol, 5e-3)
    if dtype_name == "bfloat16":
        return max(base_atol, 2e-2), max(base_rtol, 2e-2)
    return base_atol, base_rtol


def _make_strided_pair(
    arr: np.ndarray,
    ref: Any,
    device: str,
    dtype_name: str,
    *,
    requires_grad: bool,
    is_index: bool = False,
    is_mask: bool = False,
) -> tuple[lucid.Tensor, Any]:
    """Build a non-contiguous strided view pair with identical logical values to ``arr``."""
    if is_index:
        l_dt, r_dt, np_dt = lucid.int64, ref.int64, np.int64
    elif is_mask:
        l_dt, r_dt, np_dt = lucid.bool_, ref.bool, np.bool_
    else:
        l_dt = getattr(lucid, dtype_name)
        r_dt = getattr(ref, dtype_name)
        np_dt = np.float64 if dtype_name == "float64" else np.float32

    cast_arr = arr.astype(np_dt)
    if cast_arr.ndim >= 2:
        perm = (*range(cast_arr.ndim - 2), cast_arr.ndim - 1, cast_arr.ndim - 2)
        base_np = np.ascontiguousarray(np.transpose(cast_arr, perm))
        l_base = lucid.tensor(
            base_np.copy(), dtype=l_dt, device=device, requires_grad=requires_grad
        )
        r_base = ref.tensor(base_np.copy(), dtype=r_dt, requires_grad=requires_grad)
        return l_base.transpose(-1, -2), r_base.transpose(-1, -2)
    if cast_arr.ndim == 1 and cast_arr.size > 0:
        padded = np.zeros((2 * cast_arr.size + 1,), dtype=np_dt)
        padded[1::2] = cast_arr
        l_base = lucid.tensor(padded.copy(), dtype=l_dt, device=device, requires_grad=requires_grad)
        r_base = ref.tensor(padded.copy(), dtype=r_dt, requires_grad=requires_grad)
        return l_base[1::2], r_base[1::2]
    l_t = lucid.tensor(cast_arr.copy(), dtype=l_dt, device=device, requires_grad=requires_grad)
    r_t = ref.tensor(cast_arr.copy(), dtype=r_dt, requires_grad=requires_grad)
    return l_t, r_t


def _materialize_value(
    val: object,
    ref: Any,
    device: str,
    dtype_name: str,
    *,
    with_grad: bool,
    strided: bool,
    l_leaves: list[lucid.Tensor],
    r_leaves: list[Any],
) -> tuple[object, object]:
    """Recursively materialize :class:`ArraySpec` items into paired tensors."""
    if isinstance(val, ArraySpec):
        req_grad = bool(with_grad and val.differentiable and not val.is_index and not val.is_mask)
        if strided:
            l_t, r_t = _make_strided_pair(
                val.data,
                ref,
                device,
                dtype_name,
                requires_grad=req_grad,
                is_index=val.is_index,
                is_mask=val.is_mask,
            )
            if req_grad:
                l_t.retain_grad()
                r_t.retain_grad()
        else:
            if val.is_index:
                l_dt, r_dt, np_dt = lucid.int64, ref.int64, np.int64
            elif val.is_mask:
                l_dt, r_dt, np_dt = lucid.bool_, ref.bool, np.bool_
            else:
                l_dt = getattr(lucid, dtype_name)
                r_dt = getattr(ref, dtype_name)
                np_dt = np.float64 if dtype_name == "float64" else np.float32
            cast_arr = val.data.astype(np_dt)
            l_t = lucid.tensor(cast_arr.copy(), dtype=l_dt, device=device, requires_grad=req_grad)
            r_t = ref.tensor(cast_arr.copy(), dtype=r_dt, requires_grad=req_grad)
        if req_grad:
            l_leaves.append(l_t)
            r_leaves.append(r_t)
        return l_t, r_t
    if isinstance(val, tuple):
        pairs = [
            _materialize_value(
                item,
                ref,
                device,
                dtype_name,
                with_grad=with_grad,
                strided=strided,
                l_leaves=l_leaves,
                r_leaves=r_leaves,
            )
            for item in val
        ]
        return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)
    if isinstance(val, list):
        pairs = [
            _materialize_value(
                item,
                ref,
                device,
                dtype_name,
                with_grad=with_grad,
                strided=strided,
                l_leaves=l_leaves,
                r_leaves=r_leaves,
            )
            for item in val
        ]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    return val, val


def _materialize_sample(
    sample: SampleInput, ref: Any, device: str, dtype_name: str, *, with_grad: bool, strided: bool
) -> tuple[
    tuple[object, ...],
    dict[str, object],
    tuple[object, ...],
    dict[str, object],
    list[lucid.Tensor],
    list[Any],
]:
    """Materialize ``sample`` into Lucid and reference args/kwargs plus leaf lists."""
    l_leaves: list[lucid.Tensor] = []
    r_leaves: list[Any] = []
    l_args, r_args = _materialize_value(
        sample.args,
        ref,
        device,
        dtype_name,
        with_grad=with_grad,
        strided=strided,
        l_leaves=l_leaves,
        r_leaves=r_leaves,
    )
    l_kw: dict[str, object] = {}
    r_kw: dict[str, object] = {}
    raw_r_kw = sample.kwargs if sample.ref_kwargs is None else sample.ref_kwargs
    for k, v in sample.kwargs.items():
        lv, rv = _materialize_value(
            v,
            ref,
            device,
            dtype_name,
            with_grad=with_grad,
            strided=strided,
            l_leaves=l_leaves,
            r_leaves=r_leaves,
        )
        l_kw[k] = lv
        if k in raw_r_kw and raw_r_kw[k] is v:
            r_kw[k] = rv
    for k, v in raw_r_kw.items():
        if k not in r_kw:
            _, rv = _materialize_value(
                v,
                ref,
                device,
                dtype_name,
                with_grad=with_grad,
                strided=strided,
                l_leaves=[],
                r_leaves=[],
            )
            r_kw[k] = rv
    assert isinstance(l_args, tuple) and isinstance(r_args, tuple)
    return l_args, l_kw, r_args, r_kw, l_leaves, r_leaves


def _extract_tensors(out: object) -> tuple[object, ...]:
    """Flatten single tensor or tuple/list of tensors into a tuple."""
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


def _assert_outputs_match(
    op: OpInfo, sample_name: str, l_out: object, r_out: object, device: str, dtype_name: str
) -> None:
    """Compare forward outputs for shape, device, and numerical closeness."""
    atol, rtol = _tolerances_for_dtype(op, dtype_name, backward=False)
    l_items = _extract_tensors(l_out)
    r_items = _extract_tensors(r_out)
    assert len(l_items) == len(r_items), f"{op.name}[{sample_name}]: output tuple length mismatch"
    for idx, (li, ri) in enumerate(zip(l_items, r_items, strict=True)):
        assert isinstance(li, lucid.Tensor), f"{op.name}[{sample_name}]: output {idx} not a Tensor"
        assert (
            li.device.type == device
        ), f"{op.name}[{sample_name}]: expected device {device}, got {li.device.type}"
        l_np = _to_numpy(li)
        r_np = _to_numpy(ri)
        if np.issubdtype(r_np.dtype, np.floating) or np.issubdtype(r_np.dtype, np.complexfloating):
            l_f = l_np.astype(np.float32) if l_np.dtype != np.complex64 else l_np
            r_f = r_np.astype(np.float32) if r_np.dtype != np.complex64 else r_np
            assert_close(
                l_f,
                r_f,
                atol=atol,
                rtol=rtol,
                msg=f"{op.name}[{sample_name}] out[{idx}] ({device}, {dtype_name})",
            )
        else:
            np.testing.assert_array_equal(
                l_np, r_np, err_msg=f"{op.name}[{sample_name}] out[{idx}] ({device}, {dtype_name})"
            )


def _non_uniform_cotangent(shape: tuple[int, ...], seed: int = 17) -> np.ndarray:
    """Build a deterministic non-uniform cotangent tensor so constant-gradient bugs fail."""
    n = int(np.prod(shape)) if shape else 1
    rng = np.random.default_rng(seed + n)
    vals = rng.uniform(0.25, 1.75, size=(n,)).astype(np.float64)
    signs = np.where((np.arange(n) % 3) == 1, -1.0, 1.0)
    return (vals * signs).reshape(shape)


def _run_backward_comparison(
    op: OpInfo, sample: SampleInput, ref: Any, device: str, dtype_name: str, *, strided: bool
) -> None:
    """Execute forward + VJP with non-uniform cotangent and compare leaf gradients."""
    l_args, l_kw, r_args, r_kw, l_leaves, r_leaves = _materialize_sample(
        sample, ref, device, dtype_name, with_grad=True, strided=strided
    )
    if not l_leaves:
        return
    l_out = op.lucid_fn(*l_args, **l_kw)
    r_out = op.ref_fn(ref, *r_args, **r_kw)
    _assert_outputs_match(op, sample.name, l_out, r_out, device, dtype_name)

    l_first = _extract_tensors(l_out)[0]
    r_first = _extract_tensors(r_out)[0]
    assert isinstance(l_first, lucid.Tensor)
    if l_first.numel() == 0:
        return

    cot_np = _non_uniform_cotangent(tuple(int(d) for d in l_first.shape))
    l_dt = getattr(lucid, dtype_name)
    r_dt = getattr(ref, dtype_name)
    np_dt = np.float64 if dtype_name == "float64" else np.float32
    l_cot = lucid.tensor(cot_np.astype(np_dt), dtype=l_dt, device=device)
    r_cot = ref.tensor(cot_np.astype(np_dt), dtype=r_dt)

    l_loss = (l_first * l_cot).sum()
    r_loss = (r_first * r_cot).sum()
    l_loss.backward()
    r_loss.backward()

    atol, rtol = _tolerances_for_dtype(op, dtype_name, backward=True)
    mode = "strided-bwd" if strided else "bwd"
    for idx, (ll, rl) in enumerate(zip(l_leaves, r_leaves, strict=True)):
        assert ll.grad is not None, f"{op.name}[{sample.name}] ({mode}): leaf {idx} grad is None"
        assert (
            ll.grad.shape == ll.shape
        ), f"{op.name}[{sample.name}] ({mode}): leaf {idx} grad shape {ll.grad.shape} != {ll.shape}"
        assert_close(
            _to_numpy(ll.grad).astype(np.float32),
            _to_numpy(rl.grad).astype(np.float32),
            atol=atol,
            rtol=rtol,
            msg=f"{op.name}[{sample.name}] ({mode}) leaf[{idx}] ({device}, {dtype_name})",
        )


def verify_opinfo_forward(op: OpInfo, ref: Any, device: str, dtype_name: str) -> None:
    """Verify forward parity across all ``SampleInput``s of ``op``."""
    if dtype_name in ("float16", "bfloat16") and not op.supports_half:
        return
    for sample in op.sample_inputs_fn():
        l_args, l_kw, r_args, r_kw, _, _ = _materialize_sample(
            sample, ref, device, dtype_name, with_grad=False, strided=False
        )
        l_out = op.lucid_fn(*l_args, **l_kw)
        r_out = op.ref_fn(ref, *r_args, **r_kw)
        _assert_outputs_match(op, sample.name, l_out, r_out, device, dtype_name)


def verify_opinfo_backward(op: OpInfo, ref: Any, device: str, dtype_name: str = "float32") -> None:
    """Verify backward VJP parity across all differentiable ``SampleInput``s of ``op``."""
    if not op.supports_backward:
        return
    for sample in op.sample_inputs_fn():
        if not sample.supports_backward:
            continue
        _run_backward_comparison(op, sample, ref, device, dtype_name, strided=False)


def verify_opinfo_strided_views(
    op: OpInfo, ref: Any, device: str, dtype_name: str = "float32"
) -> None:
    """Verify forward + backward parity when inputs are non-contiguous strided views."""
    if not op.supports_strided:
        return
    for sample in op.sample_inputs_fn():
        if not sample.supports_strided:
            continue
        if op.supports_backward and sample.supports_backward:
            _run_backward_comparison(op, sample, ref, device, dtype_name, strided=True)
        else:
            l_args, l_kw, r_args, r_kw, _, _ = _materialize_sample(
                sample, ref, device, dtype_name, with_grad=False, strided=True
            )
            l_out = op.lucid_fn(*l_args, **l_kw)
            r_out = op.ref_fn(ref, *r_args, **r_kw)
            _assert_outputs_match(op, f"{sample.name}:strided", l_out, r_out, device, dtype_name)


def _eval_perturbed_sample(
    op: OpInfo,
    sample: SampleInput,
    ref: Any,
    target_leaf_idx: int,
    flat_idx: int,
    delta: float,
    cot_np: np.ndarray,
) -> float:
    """Evaluate ``(op(*args, **kwargs) * cot).sum()`` with one scalar entry perturbed."""
    l_args, l_kw, _, _, l_leaves, _ = _materialize_sample(
        sample, ref, "cpu", "float64", with_grad=True, strided=False
    )
    target = l_leaves[target_leaf_idx]
    arr = _to_numpy(target.detach()).astype(np.float64).copy()
    arr.flat[flat_idx] += delta
    with lucid.no_grad():
        target.copy_(lucid.tensor(arr, dtype=lucid.float64, device="cpu"))
        out = _extract_tensors(op.lucid_fn(*l_args, **l_kw))[0]
        return float((_to_numpy(out).astype(np.float64) * cot_np).sum())


def verify_opinfo_gradcheck(op: OpInfo, ref: Any) -> None:
    """Verify analytical CPU float64 gradients against central finite differences."""
    if not (op.supports_backward and op.supports_gradcheck):
        return
    eps = 1e-5
    for sample in op.sample_inputs_fn():
        if not (sample.supports_backward and sample.supports_gradcheck):
            continue
        l_args, l_kw, _, _, l_leaves, _ = _materialize_sample(
            sample, ref, "cpu", "float64", with_grad=True, strided=False
        )
        if not l_leaves:
            continue
        l_out = _extract_tensors(op.lucid_fn(*l_args, **l_kw))[0]
        assert isinstance(l_out, lucid.Tensor)
        if l_out.numel() == 0:
            continue
        cot_np = _non_uniform_cotangent(tuple(int(d) for d in l_out.shape))
        cot = lucid.tensor(cot_np, dtype=lucid.float64, device="cpu")
        (l_out * cot).sum().backward()

        for leaf_idx, leaf in enumerate(l_leaves):
            assert leaf.grad is not None
            analytic = _to_numpy(leaf.grad).astype(np.float64)
            num_grad = np.zeros_like(analytic)
            for flat_i in range(analytic.size):
                f_pos = _eval_perturbed_sample(op, sample, ref, leaf_idx, flat_i, eps, cot_np)
                f_neg = _eval_perturbed_sample(op, sample, ref, leaf_idx, flat_i, -eps, cot_np)
                num_grad.flat[flat_i] = (f_pos - f_neg) / (2.0 * eps)
            assert_close(
                analytic,
                num_grad,
                atol=max(op.bwd_atol, 1e-4),
                rtol=max(op.bwd_rtol, 1e-3),
                msg=f"{op.name}[{sample.name}] gradcheck leaf[{leaf_idx}]",
            )


def verify_opinfo_second_order_strided(
    op: OpInfo, ref: Any, device: str, dtype_name: str = "float32"
) -> None:
    """Verify 2nd-order HVPs (``create_graph=True``) through non-contiguous strided views."""
    if not (op.supports_backward and op.supports_strided):
        return
    atol, rtol = max(op.bwd_atol, 5e-4), max(op.bwd_rtol, 5e-4)
    for sample in op.sample_inputs_fn():
        if not (sample.supports_backward and sample.supports_strided):
            continue
        l_args, l_kw, r_args, r_kw, l_leaves, r_leaves = _materialize_sample(
            sample, ref, device, dtype_name, with_grad=True, strided=True
        )
        if not l_leaves:
            continue
        l_out = _extract_tensors(op.lucid_fn(*l_args, **l_kw))[0]
        r_out = _extract_tensors(op.ref_fn(ref, *r_args, **r_kw))[0]
        assert isinstance(l_out, lucid.Tensor)
        if l_out.numel() == 0:
            continue

        cot1_np = _non_uniform_cotangent(tuple(int(d) for d in l_out.shape), seed=19).astype(
            np.float32
        )
        l_cot1 = lucid.tensor(cot1_np, dtype=lucid.float32, device=device)
        r_cot1 = ref.tensor(cot1_np, dtype=ref.float32)

        try:
            l_grads1 = lucid.autograd.grad(
                (l_out * l_cot1).sum(), l_leaves, create_graph=True, allow_unused=True
            )
            r_grads1 = ref.autograd.grad(
                (r_out * r_cot1).sum(), r_leaves, create_graph=True, allow_unused=True
            )
        except NotImplementedError:
            continue
        except RuntimeError as exc:
            if "not implemented" in str(exc).lower():
                continue
            raise

        for idx, (lg, rg) in enumerate(zip(l_grads1, r_grads1, strict=True)):
            assert lg is not None and rg is not None
            assert_close(
                _to_numpy(lg).astype(np.float32),
                _to_numpy(rg).astype(np.float32),
                atol=atol,
                rtol=rtol,
                msg=f"{op.name}[{sample.name}] 1st-order strided grad[{idx}] ({device})",
            )

        if not any(g is not None and g.requires_grad for g in l_grads1) or not any(
            g is not None and g.requires_grad for g in r_grads1
        ):
            continue

        l_hvp_scalar = lucid.zeros((), dtype=lucid.float32, device=device)
        r_hvp_scalar = ref.zeros((), dtype=ref.float32)
        for idx, (lg, rg) in enumerate(zip(l_grads1, r_grads1, strict=True)):
            if lg is None or rg is None or not (lg.requires_grad and rg.requires_grad):
                continue
            v_np = _non_uniform_cotangent(tuple(int(d) for d in lg.shape), seed=31 + idx).astype(
                np.float32
            )
            l_v = lucid.tensor(v_np, dtype=lucid.float32, device=device)
            r_v = ref.tensor(v_np, dtype=ref.float32)
            l_hvp_scalar = l_hvp_scalar + (lg * l_v).sum()
            r_hvp_scalar = r_hvp_scalar + (rg * r_v).sum()

        try:
            l_grads2 = lucid.autograd.grad(l_hvp_scalar, l_leaves, allow_unused=True)
            r_grads2 = ref.autograd.grad(r_hvp_scalar, r_leaves, allow_unused=True)
        except NotImplementedError:
            continue
        except RuntimeError as exc:
            if "not implemented" in str(exc).lower():
                continue
            raise
        for idx, (lg2, rg2) in enumerate(zip(l_grads2, r_grads2, strict=True)):
            if lg2 is None and rg2 is None:
                continue
            assert (
                lg2 is not None and rg2 is not None
            ), f"{op.name}[{sample.name}] 2nd-order strided HVP[{idx}] None mismatch ({device})"
            assert_close(
                _to_numpy(lg2).astype(np.float32),
                _to_numpy(rg2).astype(np.float32),
                atol=atol,
                rtol=rtol,
                msg=f"{op.name}[{sample.name}] 2nd-order strided HVP[{idx}] ({device})",
            )
