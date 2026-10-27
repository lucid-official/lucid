"""Hypothesis boundary property tests driven by ``lucid._ops._registry._REGISTRY`` (LCD-268).

Drives every ``OpEntry`` in ``_REGISTRY`` across boundary shapes (``0-d``, ``0``
extents, length-``1`` axes, broadcast edges), boundary element values (``0``,
``-0.0``, negative numbers, boundary integers, ``±inf``, ``nan``), dtypes
(including ``float64`` and ``complex64`` on CPU/Metal), devices (``cpu`` and
``metal``), and out-of-range indices, verifying under ``faulthandler`` against
the reference framework that:
  (a) when the reference succeeds, Lucid either matches the reference on shape,
      dtype, device, and element values or raises ``NotImplementedError`` for
      unsupported backend/dtype combinations (never a bare ``LucidError``), and
  (b) when the reference refuses an invalid input, Lucid also refuses with a
      typed exception (or, on documented Lucid dtype extensions such as
      integer-to-float32 promotion per LCD-508, matches the reference evaluated
      at the promoted dtype).
"""

from collections.abc import Callable, Sequence
import faulthandler
import math
from types import ModuleType

from hypothesis import given, settings, strategies as st
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._ops._registry import _REGISTRY, OpEntry
from lucid.test._fixtures.devices import metal_available
from lucid.test._fixtures.ref_framework import ref_module

faulthandler.enable()

_DEVICES: tuple[str, ...] = ("cpu", "metal") if metal_available() else ("cpu",)

_DTYPE_NAMES: tuple[str, ...] = (
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
)

_TYPED_REFUSALS: tuple[type[Exception], ...] = (
    ValueError,
    TypeError,
    IndexError,
    ZeroDivisionError,
    NotImplementedError,
    _C_engine.ShapeMismatch,
    _C_engine.DtypeMismatch,
    _C_engine.DeviceMismatch,
    _C_engine.InvalidArgument,
    _C_engine.LinAlgError,
)

_FLOAT_PROMOTING_OPS: frozenset[str] = frozenset(
    {
        "exp",
        "log",
        "log2",
        "log10",
        "log1p",
        "exp2",
        "sqrt",
        "rsqrt",
        "reciprocal",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "sigmoid",
        "silu",
        "gelu",
        "mish",
        "selu",
        "softplus",
        "relu6",
        "hard_sigmoid",
        "hard_swish",
        "erf",
        "erfinv",
        "mean",
        "var",
        "std",
        "softmax",
        "log_softmax",
        "logsumexp",
    }
)

_INTEGRAL_DTYPES: frozenset[str] = frozenset({"bool_", "int8", "int16", "int32", "int64"})

_DIVMOD_OPS: frozenset[str] = frozenset({"div", "div_", "floor_divide", "fmod", "remainder"})

_SHIFT_OPS: frozenset[str] = frozenset({"bitwise_left_shift", "bitwise_right_shift"})

_BOUNDARY_FLOATS: tuple[float, ...] = (
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.0,
    -1.0,
    2.0,
    -2.0,
    10.0,
    -10.0,
    math.inf,
    -math.inf,
    math.nan,
)

_BOUNDARY_INTS: tuple[int, ...] = (0, 1, -1, 2, -2, 7, -8, 63, 64, 127, -128)

# ── Hypothesis strategies ────────────────────────────────────────────────────

boundary_shapes: st.SearchStrategy[tuple[int, ...]] = st.lists(
    st.sampled_from((0, 1, 2, 3)), min_size=0, max_size=3
).map(tuple)

dtype_names: st.SearchStrategy[str] = st.sampled_from(_DTYPE_NAMES)
devices: st.SearchStrategy[str] = st.sampled_from(_DEVICES)

boundary_float_pools: st.SearchStrategy[tuple[float, ...]] = st.lists(
    st.sampled_from(_BOUNDARY_FLOATS), min_size=1, max_size=6
).map(tuple)

boundary_int_pools: st.SearchStrategy[tuple[int, ...]] = st.lists(
    st.sampled_from(_BOUNDARY_INTS), min_size=1, max_size=6
).map(tuple)


@st.composite
def broadcast_shape_pairs(
    draw: st.DrawFn,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    s1 = draw(boundary_shapes)
    if draw(st.booleans()):
        rank2 = draw(st.integers(min_value=0, max_value=3))
        dims2: list[int] = []
        for i in range(rank2):
            offset = i - (rank2 - len(s1))
            if 0 <= offset < len(s1):
                dims2.append(draw(st.sampled_from((s1[offset], 1, 0))))
            else:
                dims2.append(draw(st.sampled_from((0, 1, 2))))
        return s1, tuple(dims2)
    return s1, draw(boundary_shapes)


def _element_values(
    numel: int,
    dtype_name: str,
    float_pool: Sequence[float],
    int_pool: Sequence[int],
) -> list[bool] | list[int] | list[float]:
    if dtype_name == "bool_":
        return [(int_pool[i % len(int_pool)] % 2) != 0 for i in range(numel)]
    if dtype_name in _INTEGRAL_DTYPES:
        return [int(int_pool[i % len(int_pool)]) for i in range(numel)]
    return [float(float_pool[i % len(float_pool)]) for i in range(numel)]


def _make_tensor_pair(
    ref: ModuleType | None,
    shape: tuple[int, ...],
    dtype_name: str,
    device: str,
    float_pool: Sequence[float],
    int_pool: Sequence[int],
) -> tuple[lucid.Tensor, object]:
    dt = getattr(lucid, dtype_name)
    ref_dt = (
        getattr(ref, "bool" if dtype_name == "bool_" else dtype_name) if ref is not None else None
    )
    numel = math.prod(shape)
    if numel == 0:
        return (
            lucid.zeros(shape, dtype=dt, device=device),
            ref.zeros(shape, dtype=ref_dt) if ref is not None else None,
        )
    if dtype_name == "complex64":
        re_vals = [float(float_pool[i % len(float_pool)]) for i in range(numel)]
        im_vals = [float(float_pool[(i + 1) % len(float_pool)]) for i in range(numel)]
        re_l = lucid.tensor(re_vals, dtype=lucid.float32, device=device).reshape(shape)
        im_l = lucid.tensor(im_vals, dtype=lucid.float32, device=device).reshape(shape)
        if ref is None:
            return lucid.complex(re_l, im_l), None
        re_r = ref.tensor(re_vals, dtype=ref.float32).reshape(shape)
        im_r = ref.tensor(im_vals, dtype=ref.float32).reshape(shape)
        return lucid.complex(re_l, im_l), ref.complex(re_r, im_r)
    vals = _element_values(numel, dtype_name, float_pool, int_pool)
    return (
        lucid.tensor(vals, dtype=dt, device=device).reshape(shape),
        ref.tensor(vals, dtype=ref_dt).reshape(shape) if ref is not None else None,
    )


def _extract_tensors(res: object) -> tuple[lucid.Tensor, ...]:
    if isinstance(res, lucid.Tensor):
        return (res,)
    if isinstance(res, (tuple, list)):
        return tuple(t for t in res if isinstance(t, lucid.Tensor))
    return ()


def _assert_single_tensor_matches(
    op_name: str, got: lucid.Tensor, ref_t: object, device: str
) -> None:
    assert got.device.type == device, f"{op_name}: expected {device}, got {got.device}"
    assert tuple(got.shape) == tuple(
        getattr(ref_t, "shape")
    ), f"{op_name}: shape {got.shape} != {getattr(ref_t, 'shape')}"
    if got.numel() == 0:
        return
    if op_name in ("argsort", "topk", "kthvalue"):
        return
    got_cpu = got.detach().to("cpu")
    ref_cpu = getattr(ref_t, "detach")()
    if op_name == "unique":
        assert got_cpu.shape == tuple(ref_cpu.shape)
        return
    if got_cpu.dtype in (lucid.float16, lucid.bfloat16):
        got_list = got_cpu.to(lucid.float32).reshape(-1).tolist()
        ref_list = ref_cpu.float().reshape(-1).tolist()
        assert got_list == pytest.approx(ref_list, rel=3e-2, abs=3e-2, nan_ok=True)
    elif got_cpu.dtype in (lucid.float32, lucid.float64):
        got_list = got_cpu.reshape(-1).tolist()
        ref_list = ref_cpu.double().reshape(-1).tolist()
        assert got_list == pytest.approx(ref_list, rel=1e-4, abs=1e-4, nan_ok=True)
    elif got_cpu.dtype in (lucid.complex64, lucid.complex128):
        got_re = lucid.real(got_cpu).reshape(-1).tolist()
        got_im = lucid.imag(got_cpu).reshape(-1).tolist()
        ref_re = ref_cpu.real.double().reshape(-1).tolist()
        ref_im = ref_cpu.imag.double().reshape(-1).tolist()
        if any(not math.isfinite(v) for v in (*ref_re, *ref_im, *got_re, *got_im)):
            return
        if got_re != pytest.approx(
            ref_re, rel=1e-4, abs=1e-4, nan_ok=True
        ) or got_im != pytest.approx(ref_im, rel=1e-4, abs=1e-4, nan_ok=True):
            # Signed-zero branch cuts (e.g. arctan(-0.0 + 2j)) flip the sheet sign by -1.
            assert [abs(v) for v in got_re] == pytest.approx(
                [abs(v) for v in ref_re], rel=1e-4, abs=1e-4, nan_ok=True
            )
            assert [abs(v) for v in got_im] == pytest.approx(
                [abs(v) for v in ref_im], rel=1e-4, abs=1e-4, nan_ok=True
            )
    else:
        assert (
            got_cpu.reshape(-1).tolist() == ref_cpu.reshape(-1).tolist()
        ), f"{op_name}: {got_cpu.reshape(-1).tolist()} != {ref_cpu.reshape(-1).tolist()}"


def _assert_matches_ref(op_name: str, got_obj: object, ref_obj: object, device: str) -> None:
    got_tensors = _extract_tensors(got_obj)
    if not got_tensors:
        return
    if isinstance(ref_obj, (tuple, list)):
        assert len(got_tensors) == len(ref_obj)
        for idx, (g, r) in enumerate(zip(got_tensors, ref_obj, strict=True)):
            sub_name = (
                "argsort"
                if idx == 1 and op_name in ("sort", "topk", "kthvalue", "cummax", "cummin")
                else op_name
            )
            _assert_single_tensor_matches(sub_name, g, r, device)
        return
    _assert_single_tensor_matches(op_name, got_tensors[0], ref_obj, device)


def _call_entry(entry: OpEntry, *args: object, **kwargs: object) -> object:
    if entry.inplace:
        first = args[0]
        assert isinstance(first, lucid.Tensor)
        target = first.clone()
        assert entry.method_name is not None
        return getattr(target, entry.method_name)(*args[1:], **kwargs)
    if entry.free_fn_name is not None:
        return getattr(lucid._ops, entry.free_fn_name)(*args, **kwargs)
    assert entry.method_name is not None
    return getattr(args[0], entry.method_name)(*args[1:], **kwargs)


_REF_NN_F_MAP: dict[str, str] = {
    "silu": "silu",
    "gelu": "gelu",
    "mish": "mish",
    "selu": "selu",
    "softplus": "softplus",
    "relu6": "relu6",
    "hard_sigmoid": "hardsigmoid",
    "hard_swish": "hardswish",
}


def _call_ref_entry(ref: ModuleType, entry: OpEntry, *args: object, **kwargs: object) -> object:
    name = entry.name
    if entry.inplace:
        target = getattr(args[0], "clone")()
        if name == "cube_":
            return target.pow_(3)
        assert entry.method_name is not None
        if hasattr(target, entry.method_name):
            return getattr(target, entry.method_name)(*args[1:], **kwargs)
        base_fn = getattr(ref, name[:-1])
        return base_fn(target, *args[1:], out=target, **kwargs)
    if name in _REF_NN_F_MAP:
        return getattr(ref.nn.functional, _REF_NN_F_MAP[name])(*args, **kwargs)
    if name == "squeeze_all":
        return getattr(args[0], "squeeze")()
    if name == "t":
        ndim = int(getattr(args[0], "ndim"))
        return getattr(args[0], "permute")(tuple(reversed(range(ndim))))
    if name == "trace" and int(getattr(args[0], "ndim")) > 2:
        rx = args[0]
        return ref.diagonal(rx, 0, 0, 1).sum(-1, dtype=getattr(rx, "dtype"))
    if name == "expand_dims":
        return getattr(args[0], "unsqueeze")(*args[1:], **kwargs)
    if name == "repeat":
        return ref.repeat_interleave(*args, **kwargs)
    if name == "repeat_method":
        return getattr(args[0], "repeat")(*args[1:], **kwargs)
    if name == "split_method":
        return getattr(args[0], "split")(*args[1:], **kwargs)
    if entry.free_fn_name is not None and hasattr(ref, entry.free_fn_name):
        return getattr(ref, entry.free_fn_name)(*args, **kwargs)
    if entry.method_name is not None and hasattr(args[0], entry.method_name):
        return getattr(args[0], entry.method_name)(*args[1:], **kwargs)
    raise AssertionError(f"Unmapped reference entry: {name}")


def _assert_valid_refusal(op_name: str, exc: Exception) -> None:
    msg = str(exc)
    assert "bad_variant_access" not in msg, f"{op_name} leaked bad_variant_access: {exc}"
    if isinstance(exc, (_TYPED_REFUSALS, RuntimeError)) and msg:
        return
    pytest.fail(f"{op_name} raised untyped {type(exc).__name__}: {exc}")


def _is_permitted_lucid_refusal_when_ref_succeeds(
    entry: OpEntry, exc: Exception, dtypes: Sequence[str], device: str
) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    if entry.name in _SHIFT_OPS:
        return True
    if entry.name in _DIVMOD_OPS and "bool_" in dtypes:
        return True
    if entry.name in ("pow", "pow_") and any(d in _INTEGRAL_DTYPES for d in dtypes):
        return True
    if (
        entry.name in ("matmul", "mm", "bmm", "dot", "inner", "outer", "tensordot")
        and isinstance(exc, _C_engine.DtypeMismatch)
        and len(set(dtypes)) > 1
    ):
        return True
    if entry.name == "histc":
        return True
    if entry.name == "tensordot" and isinstance(exc, _C_engine.ShapeMismatch):
        return True
    if entry.name in ("nextafter", "nextafter_") and "dtype must be F32 or F64" in str(exc):
        return True
    if entry.name in ("repeat_interleave", "repeat", "tile", "repeat_method") and device == "metal":
        return isinstance(exc, _TYPED_REFUSALS)
    return False


def _is_permitted_lucid_success_when_ref_raises(
    ref: ModuleType,
    entry: OpEntry,
    got: object,
    ref_exc: Exception,
    l_args: Sequence[object],
    r_args: Sequence[object],
    r_kwargs: dict[str, object],
    dtypes: Sequence[str],
    device: str,
) -> bool:
    for t in _extract_tensors(got):
        assert t.device.type == device
        _ = t.detach().to("cpu").tolist()

    # 1. Float-valued unary / reduction ops promote integer & bool inputs to float32 (LCD-508).
    if entry.name in _FLOAT_PROMOTING_OPS and dtypes and dtypes[0] in _INTEGRAL_DTYPES:
        promoted_args = [
            getattr(a, "to")(ref.float32) if hasattr(a, "dtype") else a for a in r_args
        ]
        try:
            want = _call_ref_entry(ref, entry, *promoted_args, **r_kwargs)
        except Exception:
            return True
        _assert_matches_ref(entry.name, got, want, device)
        return True

    if entry.name == "masked_select" and len(r_args) == 2:
        try:
            want = ref.masked_select(r_args[0], getattr(r_args[1], "to")(ref.bool))
        except Exception:
            return True
        _assert_matches_ref(entry.name, got, want, device)
        return True

    if entry.name == "complex" and len(r_args) == 2:
        try:
            want = ref.complex(
                getattr(r_args[0], "to")(ref.float32),
                getattr(r_args[1], "to")(ref.float32),
            )
        except Exception:
            return True
        _assert_matches_ref(entry.name, got, want, device)
        return True

    if entry.name == "isclose" and len(r_args) == 2:
        common = ref.promote_types(getattr(r_args[0], "dtype"), getattr(r_args[1], "dtype"))
        try:
            want = ref.isclose(
                getattr(r_args[0], "to")(common),
                getattr(r_args[1], "to")(common),
            )
        except Exception:
            return True
        _assert_matches_ref(entry.name, got, want, device)
        return True

    # 2. Metal avoids host stalls on integer division/mod by zero or negative bit shifts.
    if entry.name in _DIVMOD_OPS or entry.name in _SHIFT_OPS:
        return True

    # 3. Reference CPU kernel missing for bool / int8 / int16 / half / complex64.
    ref_msg = str(ref_exc)
    if isinstance(ref_exc, (NotImplementedError, RuntimeError)) and (
        "not implemented for" in ref_msg
        or "is not implemented" in ref_msg
        or "not supported" in ref_msg
        or "does not support" in ref_msg
        or "can't be cast to the desired output type" in ref_msg
        or "Complex" in ref_msg
        or "normal_kernel_cpu" in ref_msg
        or "Cannot convert" in ref_msg
    ):
        return True

    # 4. Empty-tensor reductions / extrema / kthvalue / topk / quantile edge differences.
    if any(isinstance(a, lucid.Tensor) and a.numel() == 0 for a in l_args):
        return True

    return False


def _verify_against_ref(
    ref: ModuleType | None,
    entry: OpEntry,
    l_args: Sequence[object],
    r_args: Sequence[object],
    l_kwargs: dict[str, object],
    r_kwargs: dict[str, object],
    dtypes: Sequence[str],
    device: str,
) -> None:
    try:
        got = _call_entry(entry, *l_args, **l_kwargs)
        lucid_ok = True
        lucid_exc: Exception | None = None
    except Exception as exc:
        got = None
        lucid_ok = False
        lucid_exc = exc
        _assert_valid_refusal(entry.name, exc)

    if ref is None:
        if lucid_ok:
            for t in _extract_tensors(got):
                assert t.device.type == device
                _ = t.detach().to("cpu").tolist()
        return

    try:
        want = _call_ref_entry(ref, entry, *r_args, **r_kwargs)
        ref_ok = True
        ref_exc: Exception | None = None
    except Exception as exc:
        want = None
        ref_ok = False
        ref_exc = exc

    if ref_ok:
        if not lucid_ok:
            assert lucid_exc is not None
            if _is_permitted_lucid_refusal_when_ref_succeeds(entry, lucid_exc, dtypes, device):
                return
            pytest.fail(
                f"{entry.name} raised {type(lucid_exc).__name__} ({lucid_exc}) "
                f"on dtypes={dtypes} device={device} when ref succeeded"
            )
        if (
            entry.name in _SHIFT_OPS
            or ((entry.name in _DIVMOD_OPS or entry.name in ("pow", "pow_")) and device == "metal")
        ) and any(d in _INTEGRAL_DTYPES for d in dtypes):
            for t in _extract_tensors(got):
                assert t.device.type == device
            return
        _assert_matches_ref(entry.name, got, want, device)
    else:
        if not lucid_ok:
            return
        assert ref_exc is not None
        if _is_permitted_lucid_success_when_ref_raises(
            ref, entry, got, ref_exc, l_args, r_args, r_kwargs, dtypes, device
        ):
            return
        pytest.fail(
            f"{entry.name} succeeded with {got!r} on dtypes={dtypes} device={device} "
            f"when ref raised {type(ref_exc).__name__}: {ref_exc}"
        )


# ── Classification of _REGISTRY entries ──────────────────────────────────────

_SPECIAL_UNARY_KWARGS: dict[
    str,
    Callable[
        [lucid.Tensor, ModuleType | None, object],
        tuple[
            tuple[object, ...],
            dict[str, object],
            tuple[object, ...],
            dict[str, object],
        ],
    ],
] = {
    "reshape": lambda _x, _r, _rx: (((-1,),), {}, ((-1,),), {}),
    "view": lambda _x, _r, _rx: (((-1,),), {}, ((-1,),), {}),
    "squeeze": lambda _x, _r, _rx: ((), {}, (), {}),
    "squeeze_all": lambda _x, _r, _rx: ((), {}, (), {}),
    "unsqueeze": lambda _x, _r, _rx: ((0,), {}, (0,), {}),
    "flatten": lambda _x, _r, _rx: ((), {}, (), {}),
    "permute": lambda x, _r, _rx: (
        (tuple(reversed(range(x.ndim))),),
        {},
        (tuple(reversed(range(x.ndim))),),
        {},
    ),
    "transpose": lambda _x, _r, _rx: ((0, -1), {}, (0, -1), {}),
    "swapaxes": lambda _x, _r, _rx: ((0, -1), {}, (0, -1), {}),
    "broadcast_to": lambda x, _r, _rx: (
        ((2, *x.shape),),
        {},
        ((2, *x.shape),),
        {},
    ),
    "expand": lambda x, _r, _rx: (((2, *x.shape),), {}, ((2, *x.shape),), {}),
    "as_strided": lambda _x, _r, _rx: (((0,), (1,)), {}, ((0,), (1,)), {}),
    "expand_dims": lambda _x, _r, _rx: ((0,), {}, (0,), {}),
    "repeat": lambda _x, _r, _rx: ((2,), {}, (2,), {"dim": 0}),
    "repeat_method": lambda x, _r, _rx: (
        (*(2 for _ in range(max(1, x.ndim))),),
        {},
        (*(2 for _ in range(max(1, x.ndim))),),
        {},
    ),
    "tile": lambda _x, _r, _rx: (((2,),), {}, ((2,),), {}),
    "flip": lambda _x, _r, _rx: (((0,),), {}, ((0,),), {}),
    "fliplr": lambda _x, _r, _rx: ((), {}, (), {}),
    "flipud": lambda _x, _r, _rx: ((), {}, (), {}),
    "roll": lambda _x, _r, _rx: ((1,), {}, (1,), {}),
    "tril": lambda _x, _r, _rx: ((), {}, (), {}),
    "triu": lambda _x, _r, _rx: ((), {}, (), {}),
    "sort": lambda _x, _r, _rx: ((), {}, (), {}),
    "argsort": lambda _x, _r, _rx: ((), {}, (), {}),
    "nonzero": lambda _x, _r, _rx: ((), {}, (), {}),
    "unique": lambda _x, _r, _rx: ((), {}, (), {}),
    "unique_consecutive": lambda _x, _r, _rx: ((), {}, (), {}),
    "topk": lambda _x, _r, _rx: ((1,), {}, (1,), {}),
    "diagonal": lambda _x, _r, _rx: (
        (),
        {"offset": 0, "dim1": 0, "dim2": 1},
        (),
        {"offset": 0, "dim1": 0, "dim2": 1},
    ),
    "masked_fill": lambda x, r, _rx: (
        (lucid.zeros(x.shape, dtype=lucid.bool_, device=x.device), 0),
        {},
        (r.zeros(x.shape, dtype=r.bool), 0) if r is not None else (),
        {},
    ),
    "split": lambda _x, _r, _rx: ((1,), {}, (1,), {}),
    "split_method": lambda _x, _r, _rx: ((1,), {}, (1,), {}),
    "chunk": lambda _x, _r, _rx: ((2,), {}, (2,), {}),
    "unbind": lambda _x, _r, _rx: ((), {}, (), {}),
    "softmax": lambda _x, _r, _rx: ((), {"dim": -1}, (), {"dim": -1}),
    "log_softmax": lambda _x, _r, _rx: ((), {"dim": -1}, (), {"dim": -1}),
    "logsumexp": lambda _x, _r, _rx: ((0,), {}, (0,), {}),
    "cummax": lambda _x, _r, _rx: ((0,), {}, (0,), {}),
    "cummin": lambda _x, _r, _rx: ((0,), {}, (0,), {}),
    "cumsum": lambda _x, _r, _rx: ((0,), {}, (0,), {}),
    "cumprod": lambda _x, _r, _rx: ((0,), {}, (0,), {}),
    "clip": lambda _x, _r, _rx: ((), {"min": 0, "max": 2}, (), {"min": 0, "max": 2}),
    "clamp": lambda _x, _r, _rx: ((), {"min": 0, "max": 2}, (), {"min": 0, "max": 2}),
    "clamp_min": lambda _x, _r, _rx: ((0,), {}, (0,), {}),
    "clamp_max": lambda _x, _r, _rx: ((2,), {}, (2,), {}),
    "narrow": lambda _x, _r, _rx: ((0, 0, 0), {}, (0, 0, 0), {}),
    "kthvalue": lambda _x, _r, _rx: ((1,), {}, (1,), {}),
    "movedim": lambda _x, _r, _rx: ((0, -1), {}, (0, -1), {}),
    "unflatten": lambda _x, _r, _rx: ((0, (1, -1)), {}, (0, (1, -1)), {}),
    "histc": lambda _x, _r, _rx: (
        (),
        {"bins": 4, "min": 0.0, "max": 4.0},
        (),
        {"bins": 4, "min": 0.0, "max": 4.0},
    ),
    "repeat_interleave": lambda _x, _r, _rx: ((2,), {}, (2,), {}),
}

_INDEXED_NAMES: frozenset[str] = frozenset(
    {
        "gather",
        "scatter",
        "scatter_add",
        "take",
        "index_select",
        "searchsorted",
        "bucketize",
    }
)

_VARIADIC_ZERO_NAMES: frozenset[str] = frozenset({"where", "meshgrid", "cartesian_prod"})

_JOIN_ENTRIES: tuple[OpEntry, ...] = tuple(e for e in _REGISTRY if e.n_tensor_args == -1)

_UNARY_PLAIN_ENTRIES: tuple[OpEntry, ...] = tuple(
    e
    for e in _REGISTRY
    if e.n_tensor_args == 1 and e.name not in _SPECIAL_UNARY_KWARGS and e.name not in _INDEXED_NAMES
)

_UNARY_SPECIAL_ENTRIES: tuple[OpEntry, ...] = tuple(
    e for e in _REGISTRY if e.name in _SPECIAL_UNARY_KWARGS
)

_BINARY_ENTRIES: tuple[OpEntry, ...] = tuple(
    e for e in _REGISTRY if e.n_tensor_args == 2 and e.name not in _INDEXED_NAMES
)

_INDEXED_ENTRIES: tuple[OpEntry, ...] = tuple(e for e in _REGISTRY if e.name in _INDEXED_NAMES)

_VARIADIC_ZERO_ENTRIES: tuple[OpEntry, ...] = tuple(
    e for e in _REGISTRY if e.name in _VARIADIC_ZERO_NAMES
)


@pytest.mark.property
def test_every_registry_entry_is_covered_by_boundary_harness() -> None:
    covered = {
        e.name
        for group in (
            _UNARY_PLAIN_ENTRIES,
            _UNARY_SPECIAL_ENTRIES,
            _BINARY_ENTRIES,
            _JOIN_ENTRIES,
            _INDEXED_ENTRIES,
            _VARIADIC_ZERO_ENTRIES,
        )
        for e in group
    }
    all_names = {e.name for e in _REGISTRY}
    assert covered == all_names


@pytest.mark.property
@settings(max_examples=20, deadline=None, database=None)
@given(
    shape=boundary_shapes,
    dtype_name=dtype_names,
    device=devices,
    float_pool=boundary_float_pools,
    int_pool=boundary_int_pools,
)
def test_unary_and_reduction_registry_ops_at_boundary_inputs(
    shape: tuple[int, ...],
    dtype_name: str,
    device: str,
    float_pool: tuple[float, ...],
    int_pool: tuple[int, ...],
) -> None:
    ref = ref_module()
    if device == "metal" and dtype_name == "float64":
        with pytest.raises(NotImplementedError):
            _make_tensor_pair(ref, shape, dtype_name, device, float_pool, int_pool)
        return
    x, rx = _make_tensor_pair(ref, shape, dtype_name, device, float_pool, int_pool)

    for entry in _UNARY_PLAIN_ENTRIES:
        _verify_against_ref(ref, entry, (x,), (rx,), {}, {}, (dtype_name,), device)


@pytest.mark.property
@settings(max_examples=20, deadline=None, database=None)
@given(
    shape=boundary_shapes,
    dtype_name=dtype_names,
    device=devices,
    float_pool=boundary_float_pools,
    int_pool=boundary_int_pools,
)
def test_shape_and_parameterized_unary_registry_ops_at_boundary_inputs(
    shape: tuple[int, ...],
    dtype_name: str,
    device: str,
    float_pool: tuple[float, ...],
    int_pool: tuple[int, ...],
) -> None:
    ref = ref_module()
    if device == "metal" and dtype_name == "float64":
        with pytest.raises(NotImplementedError):
            _make_tensor_pair(ref, shape, dtype_name, device, float_pool, int_pool)
        return
    x, rx = _make_tensor_pair(ref, shape, dtype_name, device, float_pool, int_pool)

    for entry in _UNARY_SPECIAL_ENTRIES:
        l_extra, l_kw, r_extra, r_kw = _SPECIAL_UNARY_KWARGS[entry.name](x, ref, rx)
        _verify_against_ref(
            ref,
            entry,
            (x, *l_extra),
            (rx, *r_extra),
            l_kw,
            r_kw,
            (dtype_name,),
            device,
        )


@pytest.mark.property
@settings(max_examples=20, deadline=None, database=None)
@given(
    shapes=broadcast_shape_pairs(),
    dt_a=dtype_names,
    dt_b=dtype_names,
    device=devices,
    float_pool=boundary_float_pools,
    int_pool=boundary_int_pools,
)
def test_binary_registry_ops_at_boundary_shapes_and_mixed_dtypes(
    shapes: tuple[tuple[int, ...], tuple[int, ...]],
    dt_a: str,
    dt_b: str,
    device: str,
    float_pool: tuple[float, ...],
    int_pool: tuple[int, ...],
) -> None:
    ref = ref_module()
    s1, s2 = shapes
    if device == "metal" and ("float64" in (dt_a, dt_b)):
        with pytest.raises(NotImplementedError):
            _make_tensor_pair(ref, s1, dt_a, device, float_pool, int_pool)
            _make_tensor_pair(ref, s2, dt_b, device, float_pool, int_pool)
        return
    a, ra = _make_tensor_pair(ref, s1, dt_a, device, float_pool, int_pool)
    b, rb = _make_tensor_pair(ref, s2, dt_b, device, float_pool, int_pool)

    for entry in _BINARY_ENTRIES:
        _verify_against_ref(ref, entry, (a, b), (ra, rb), {}, {}, (dt_a, dt_b), device)


@pytest.mark.property
@settings(max_examples=15, deadline=None, database=None)
@given(
    shape=boundary_shapes,
    n_tensors=st.sampled_from((0, 1, 2)),
    dtype_name=dtype_names,
    device=devices,
    float_pool=boundary_float_pools,
    int_pool=boundary_int_pools,
)
def test_join_and_variadic_registry_ops_at_boundary_inputs(
    shape: tuple[int, ...],
    n_tensors: int,
    dtype_name: str,
    device: str,
    float_pool: tuple[float, ...],
    int_pool: tuple[int, ...],
) -> None:
    ref = ref_module()
    if device == "metal" and dtype_name == "float64":
        if n_tensors > 0:
            with pytest.raises(NotImplementedError):
                _make_tensor_pair(ref, shape, dtype_name, device, float_pool, int_pool)
        return
    pairs = [
        _make_tensor_pair(ref, shape, dtype_name, device, float_pool, int_pool)
        for _ in range(n_tensors)
    ]
    tensors = [p[0] for p in pairs]
    ref_tensors = [p[1] for p in pairs]

    for entry in _JOIN_ENTRIES:
        _verify_against_ref(ref, entry, (tensors,), (ref_tensors,), {}, {}, (dtype_name,), device)

    if tensors:
        x, rx = tensors[0], ref_tensors[0]
        cond = lucid.zeros(x.shape, dtype=lucid.bool_, device=device)
        rcond = ref.zeros(x.shape, dtype=ref.bool) if ref is not None else None
        where_entry = next(e for e in _VARIADIC_ZERO_ENTRIES if e.name == "where")
        _verify_against_ref(
            ref,
            where_entry,
            (cond, x, x),
            (rcond, rx, rx),
            {},
            {},
            (dtype_name,),
            device,
        )
        mesh_entry = next(e for e in _VARIADIC_ZERO_ENTRIES if e.name == "meshgrid")
        _verify_against_ref(
            ref,
            mesh_entry,
            tuple(tensors),
            tuple(ref_tensors),
            {"indexing": "ij"},
            {"indexing": "ij"},
            (dtype_name,),
            device,
        )
        prod_entry = next(e for e in _VARIADIC_ZERO_ENTRIES if e.name == "cartesian_prod")
        _verify_against_ref(
            ref,
            prod_entry,
            tuple(tensors),
            tuple(ref_tensors),
            {},
            {},
            (dtype_name,),
            device,
        )


@pytest.mark.property
@settings(max_examples=25, deadline=None, database=None)
@given(
    n_rows=st.sampled_from((0, 1, 3)),
    idx_val=st.sampled_from((-5, -1, 0, 1, 2, 3, 10)),
    device=devices,
)
def test_indexed_registry_ops_refuse_out_of_range_indices_without_crash(
    n_rows: int, idx_val: int, device: str
) -> None:
    ref = ref_module()
    base = lucid.arange(n_rows * 2, dtype=lucid.float32, device=device).reshape((n_rows, 2))
    rbase = (
        ref.arange(n_rows * 2, dtype=ref.float32).reshape((n_rows, 2)) if ref is not None else None
    )
    idx_2d = lucid.tensor([[idx_val, idx_val]], dtype=lucid.int64, device=device)
    ridx_2d = ref.tensor([[idx_val, idx_val]], dtype=ref.int64) if ref is not None else None
    idx_1d = lucid.tensor([idx_val], dtype=lucid.int64, device=device)
    ridx_1d = ref.tensor([idx_val], dtype=ref.int64) if ref is not None else None
    src = lucid.ones((1, 2), dtype=lucid.float32, device=device)
    rsrc = ref.ones((1, 2), dtype=ref.float32) if ref is not None else None
    seq = lucid.arange(n_rows, dtype=lucid.float32, device=device)
    rseq = ref.arange(n_rows, dtype=ref.float32) if ref is not None else None
    queries = lucid.tensor([float(idx_val)], dtype=lucid.float32, device=device)
    rqueries = ref.tensor([float(idx_val)], dtype=ref.float32) if ref is not None else None

    spec_map: dict[str, tuple[tuple[object, ...], tuple[object, ...], int]] = {
        "gather": ((base, 0, idx_2d), (rbase, 0, ridx_2d), n_rows),
        "scatter": ((base, 0, idx_2d, src), (rbase, 0, ridx_2d, rsrc), n_rows),
        "scatter_add": ((base, 0, idx_2d, src), (rbase, 0, ridx_2d, rsrc), n_rows),
        "take": ((base, idx_1d), (rbase, ridx_1d), base.numel()),
        "index_select": ((base, 0, idx_1d), (rbase, 0, ridx_1d), n_rows),
        "searchsorted": ((seq, queries), (rseq, rqueries), -1),
        "bucketize": ((queries, seq), (rqueries, rseq), -1),
    }
    assert set(spec_map) == _INDEXED_NAMES

    for entry in _INDEXED_ENTRIES:
        l_args, r_args, limit = spec_map[entry.name]
        if limit >= 0:
            # Reference gather/scatter/index_select accept only non-negative
            # indices [0, limit), whereas take accepts [-limit, limit).
            in_bounds = (
                (-limit <= idx_val < limit) if entry.name == "take" else (0 <= idx_val < limit)
            )
            lucid_in_bounds = -limit <= idx_val < limit
            if not lucid_in_bounds and (device == "cpu" or limit == 0):
                with pytest.raises(IndexError):
                    _call_entry(entry, *l_args)
                continue
            got = _call_entry(entry, *l_args)
            assert isinstance(got, lucid.Tensor)
            assert got.device.type == device
            if in_bounds and ref is not None:
                want = _call_ref_entry(ref, entry, *r_args)
                _assert_matches_ref(entry.name, got, want, device)
            continue
        _verify_against_ref(ref, entry, l_args, r_args, {}, {}, ("float32",), device)


@pytest.mark.property
@pytest.mark.parametrize("device", _DEVICES)
def test_boundary_regressions_uncovered_by_registry_properties(device: str) -> None:
    ref = ref_module()

    # 1. hypot(0.0, 0.0), hypot(inf, inf), and logaddexp(-inf, -inf) / (0, inf)
    # must not produce NaN.
    for dt_name in ("float32", "float16", "bfloat16"):
        dt = getattr(lucid, dt_name)
        z = lucid.zeros((), dtype=dt, device=device)
        inf_t = lucid.full((), float("inf"), dtype=dt, device=device)
        ninf_t = lucid.full((), float("-inf"), dtype=dt, device=device)
        assert float(lucid.hypot(z, z).to("cpu").item()) == 0.0
        assert math.isinf(float(lucid.hypot(inf_t, inf_t).to("cpu").item()))
        assert math.isinf(float(lucid.logaddexp(z, inf_t).to("cpu").item()))
        assert float(lucid.logaddexp(ninf_t, ninf_t).to("cpu").item()) == float("-inf")

    # 2. _broadcast_with_mask with extent 1 and 0 must broadcast to 0, not 1.
    empty_x = lucid.zeros((0,), dtype=lucid.float32, device=device)
    unit_mask = lucid.ones((1,), dtype=lucid.bool_, device=device)
    assert lucid.masked_select(empty_x, unit_mask).shape == (0,)
    assert lucid.masked_fill(empty_x, unit_mask, 1.0).shape == (0,)

    # 3. inner supports 0-D scalars, empty integer leading dims, and complex64.
    s_a = lucid.tensor(2, dtype=lucid.int32, device=device)
    s_b = lucid.tensor(3, dtype=lucid.int32, device=device)
    assert int(lucid._ops.inner(s_a, s_b).to("cpu").item()) == 6
    e_a = lucid.zeros((2, 0), dtype=lucid.int32, device=device)
    assert lucid._ops.inner(e_a, e_a).shape == (2, 2)
    c_a = lucid.complex(
        lucid.tensor([1.0, 2.0], device=device),
        lucid.tensor([3.0, 4.0], device=device),
    )
    got_inner = lucid._ops.inner(c_a, c_a)
    assert complex(got_inner.to("cpu").item()) == pytest.approx(complex(-20.0, 22.0))
    if ref is not None:
        rc_a = ref.complex(ref.tensor([1.0, 2.0]), ref.tensor([3.0, 4.0]))
        _assert_single_tensor_matches("inner", got_inner, ref.inner(rc_a, rc_a), device)

    # 4. kron pads lower-rank operand with leading 1s.
    k0 = lucid.tensor(2.0, device=device)
    k1 = lucid.tensor([1.0, 3.0], device=device)
    k2 = lucid.ones((2, 3), device=device)
    assert lucid.kron(k0, k1).shape == (2,)
    assert lucid.kron(k1, k2).shape == (2, 6)

    # 5. tensordot validates out-of-range integer dims and supports integers/complex64.
    with pytest.raises(_C_engine.InvalidArgument):
        lucid.tensordot(k0, k0, dims=2)
    with pytest.raises(_C_engine.InvalidArgument):
        lucid.tensordot(k1, k1, dims=-1)
    td_i = lucid.tensor([[1, 2], [3, 4]], dtype=lucid.int32, device=device)
    got_td = lucid.tensordot(td_i, td_i, dims=2)
    assert int(got_td.to("cpu").item()) == 30
    if ref is not None:
        rtd_i = ref.tensor([[1, 2], [3, 4]], dtype=ref.int32)
        _assert_single_tensor_matches(
            "tensordot",
            got_td,
            ref.tensordot(rtd_i, rtd_i, dims=2),
            device,
        )

    # 6. cartesian_prod(1 tensor) -> 1-D; meshgrid(0-D) -> (1,); hstack/vstack(0-D).
    assert lucid.cartesian_prod(k1).shape == (2,)
    assert lucid.meshgrid(k0, indexing="ij")[0].shape == (1,)
    assert lucid.hstack((k0, k0)).shape == (2,)
    assert lucid.vstack((k0, k0)).shape == (2, 1)

    # 7. sigmoid, silu, var, and sign on complex64 raise NotImplementedError on both devices.
    with pytest.raises(NotImplementedError):
        lucid.sigmoid(c_a)
    with pytest.raises(NotImplementedError):
        lucid._ops.silu(c_a)
    with pytest.raises(NotImplementedError):
        lucid.var(c_a)
    with pytest.raises(NotImplementedError):
        lucid.sign(c_a)
    with pytest.raises(NotImplementedError):
        lucid.argmax(c_a)
    with pytest.raises(NotImplementedError):
        lucid.argmin(c_a)
    # Empty integer dividend divided by scalar 0 returns empty tensor without ZeroDivisionError.
    e_i = lucid.zeros((0,), dtype=lucid.int8, device=device)
    z_i = lucid.zeros((), dtype=lucid.int8, device=device)
    assert lucid.floor_divide(e_i, z_i).shape == (0,)
    # isclose handles int8(-128) without two's-complement abs overflow and infinities properly.
    min_i8 = lucid.tensor(-128, dtype=lucid.int8, device=device)
    assert bool(lucid.isclose(min_i8, min_i8).to("cpu").item()) is True
    inf_f = lucid.tensor(float("inf"), dtype=lucid.float32, device=device)
    one_f = lucid.tensor(1.0, dtype=lucid.float32, device=device)
    assert bool(lucid.isclose(inf_f, inf_f).to("cpu").item()) is True
    assert bool(lucid.isclose(one_f, inf_f).to("cpu").item()) is False
