"""Op × dtype replay matrix for ``lucid.compile``.

Every case is a small function of one tensor.  For each dtype the case is
run eagerly on Metal, compiled, traced on one input and replayed on a
second, and the replay is compared with eager on that second input.  Four
outcomes, of which only the first is quiet:

* ``ok``        — compiled, and the replay matches eager;
* ``wrong``     — compiled, and the replay does not match (a frozen value,
  an emitter with other semantics, an output read back as the wrong dtype);
* ``eager``     — the signature fell back to eager (the answer is right, the
  compile is lost);
* ``error``     — the compiled call raised where eager answers.

A declared dtype eager itself refuses is not run at all: :data:`EAGER_REJECTS`
lists every such pair with the exception eager raises on each device, and
``test_op_eager_rejects`` holds eager to that list.  A refusal that is not
listed is ``rejected``, which fails.  float32 is never listed: if eager
float32 raises, the recipe is wrong, not the op (``recipe``).

Reading goes through ``.numpy()`` on the Metal tensor, not ``.to("cpu")``:
the two take different paths out of a compiled output's storage, and the
first is the one that once reported every integer output as float32.
"""

import contextlib
import io
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

import lucid
import lucid.nn.functional as F

from lucid._C import engine as _C_engine
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

DEV = COMPILE_DEVICE
DTYPES = {
    "f32": lucid.float32,
    "i64": lucid.int64,
    "i32": lucid.int32,
    "bool": lucid.bool_,
}


@dataclass(frozen=True)
class Case:
    name: str
    fn: Callable[[lucid.Tensor], object]
    dtypes: tuple[str, ...] = ("f32", "i64", "i32", "bool")
    shape: tuple[int, ...] = (4, 6)
    kind: str = "normal"  # normal | pos | unit | nat
    random: bool = False  # values differ by design; compare shape + dtype only


@dataclass
class Outcome:
    status: str
    detail: str = ""
    reached: set[str] = field(default_factory=set)


def make_input(kind: str, dtype: str, shape: tuple[int, ...], seed: int) -> lucid.Tensor:
    lucid.manual_seed(seed)
    if dtype == "bool":
        return (lucid.randint(0, 2, shape) == 1).to(DEV)
    if dtype == "f32":
        if kind == "pos":
            t = lucid.rand(*shape) * 4 + 0.1
        elif kind == "unit":
            t = lucid.rand(*shape) * 1.8 - 0.9
        elif kind == "nat":
            t = lucid.randint(0, 9, shape).to(lucid.float32)
        else:
            t = lucid.randn(*shape)
        return t.to(DEV)
    lo, hi = {"pos": (1, 9), "unit": (-1, 2), "nat": (0, 9)}.get(kind, (-9, 9))
    return lucid.randint(lo, hi, shape).to(DTYPES[dtype]).to(DEV)


def _leaves(out: object) -> list[lucid.Tensor]:
    if isinstance(out, lucid.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for o in out for t in _leaves(o)]
    return []


def _compare(got: object, want: object, random: bool) -> str:
    g, w = _leaves(got), _leaves(want)
    if len(g) != len(w):
        return f"{len(g)} outputs, want {len(w)}"
    for i, (a, b) in enumerate(zip(g, w)):
        an, bn = a.numpy(), b.numpy()
        if an.dtype != bn.dtype:
            return f"out[{i}] dtype {an.dtype}, want {bn.dtype}"
        if an.shape != bn.shape:
            return f"out[{i}] shape {an.shape}, want {bn.shape}"
        if random:
            continue
        if np.issubdtype(bn.dtype, np.floating):
            if not np.allclose(an, bn, rtol=1e-4, atol=1e-4, equal_nan=True):
                d = np.nanmax(np.abs(an.astype(np.float64) - bn.astype(np.float64)))
                return f"out[{i}] max|diff| {d:.3g}"
        elif not np.array_equal(an, bn):
            return f"out[{i}] differs: {an.ravel()[:6].tolist()} vs {bn.ravel()[:6].tolist()}"
    return ""


def _reached(fn: Callable[[lucid.Tensor], object], x: lucid.Tensor) -> set[str]:
    from lucid.autograd._grad_mode import no_grad
    from lucid.compile import _tracing

    with no_grad():
        with _tracing() as tracer:
            fn(x)
    return {n.name for n in tracer.graph.ops}


def run(case: Case, dtype: str) -> Outcome:
    import os

    os.environ["LUCID_COMPILE_VERBOSE"] = "1"  # the fallback reason is printed only then
    x1 = make_input(case.kind, dtype, case.shape, 1)
    x2 = make_input(case.kind, dtype, case.shape, 2)
    try:
        want = case.fn(x2)
    except Exception as e:  # noqa: BLE001 — eager decides which dtypes are valid
        if dtype == "f32":
            return Outcome("recipe", f"eager float32 raised {type(e).__name__}: {e}")
        return Outcome("rejected", f"{type(e).__name__}: {str(e)[:160]}")
    if not _leaves(want):
        return Outcome("recipe", "case returns no tensor")

    compiled = lucid.compile(case.fn)
    err, old = io.StringIO(), sys.stderr
    sys.stderr = err
    try:
        compiled(x1)
        got = compiled(x2)
    except Exception as e:  # noqa: BLE001
        return Outcome("error", f"{type(e).__name__}: {str(e)[:160]}")
    finally:
        sys.stderr = old
    reached = _reached(case.fn, x1)
    if compiled.cache_info()["eager_only"]:
        why = [ln for ln in err.getvalue().splitlines() if "eager fallback" in ln]
        detail = why[-1].split("eager fallback: ")[-1] if why else ""
        return Outcome("eager", detail[:160], reached)
    bad = _compare(got, want, case.random)
    return Outcome("wrong" if bad else "ok", bad, reached)


# ── The matrix ─────────────────────────────────────────────────────────────


# Constants are built once and reused, never inside a trace: a tensor made
# on the CPU and moved to Metal mid-function makes the trace mixed-device,
# and every case using it would "fall back" for a reason that is the
# recipe's, not the op's.
_CONSTS: dict[tuple[object, ...], lucid.Tensor] = {}
_ACTIVE: list[str] = [DEV]


@contextlib.contextmanager
def on_device(device: str) -> Iterator[None]:
    """Build the recipes' constants on ``device`` for the duration.

    The matrices run on the compile device; the numpy oracle runs the same
    recipes on the CPU as well, and a Metal weight beside a CPU input is a
    device mismatch rather than an answer.
    """
    _ACTIVE.append(device)
    try:
        yield
    finally:
        _ACTIVE.pop()


def _idx(*vals: int) -> lucid.Tensor:
    key = ("idx", _ACTIVE[-1], *vals)
    if key not in _CONSTS:
        _CONSTS[key] = lucid.tensor(list(vals)).to(lucid.int64).to(_ACTIVE[-1])
    return _CONSTS[key]


def _w(*shape: int, seed: int = 7) -> lucid.Tensor:
    key = ("w", _ACTIVE[-1], seed, *shape)
    if key not in _CONSTS:
        state = lucid.get_rng_state()
        lucid.manual_seed(seed)
        _CONSTS[key] = (lucid.randn(*shape) * 0.3).to(_ACTIVE[-1])
        lucid.set_rng_state(state)
    return _CONSTS[key]


def _eye(n: int) -> lucid.Tensor:
    key = ("eye", _ACTIVE[-1], n)
    if key not in _CONSTS:
        _CONSTS[key] = lucid.eye(n).to(_ACTIVE[-1])
    return _CONSTS[key]


def _cases() -> list[Case]:
    C: list[Case] = []

    def add(name: str, fn: Callable[[lucid.Tensor], object], **kw: object) -> None:
        C.append(Case(name, fn, **kw))  # type: ignore[arg-type]

    # ── unary, any dtype
    for name, f in {
        "abs": lucid.abs,
        "neg": lucid.neg,
        "sign": lucid.sign,
        "square": lucid.square,
        "round": lucid.round,
        "floor": lucid.floor,
        "ceil": lucid.ceil,
        "trunc": lucid.trunc,
        "relu": lucid.relu,
        "isnan": lucid.isnan,
        "isinf": lucid.isinf,
        "isfinite": lucid.isfinite,
        "logical_not": lucid.logical_not,
        "nan_to_num": lucid.nan_to_num,
        "clamp": lambda t: lucid.clamp(t, -2, 2),
        "clip": lambda t: lucid.clip(t, -1, 3),
        "pow_scalar": lambda t: t**2,
        "pow_scalar3": lambda t: t**3,
        "rpow_scalar": lambda t: 2.0**t,
        "astype_f32": lambda t: t.to(lucid.float32),
        "astype_i64": lambda t: t.to(lucid.int64),
        "astype_bool": lambda t: t.to(lucid.bool_),
    }.items():
        add(name, f)

    # ── unary float math
    for name, f, kind in [
        ("exp", lucid.exp, "normal"),
        ("log", lucid.log, "pos"),
        ("log2", lucid.log2, "pos"),
        ("sqrt", lucid.sqrt, "pos"),
        ("rsqrt", lucid.rsqrt, "pos"),
        ("reciprocal", lucid.reciprocal, "pos"),
        ("sin", lucid.sin, "normal"),
        ("cos", lucid.cos, "normal"),
        ("tan", lucid.tan, "unit"),
        ("arcsin", lucid.arcsin, "unit"),
        ("arccos", lucid.arccos, "unit"),
        ("arctan", lucid.arctan, "normal"),
        ("sinh", lucid.sinh, "normal"),
        ("cosh", lucid.cosh, "normal"),
        ("tanh", lucid.tanh, "normal"),
        ("erf", lucid.erf, "normal"),
        ("erfinv", lucid.erfinv, "unit"),
        ("erfinv_edge", lambda t: lucid.erfinv(t.clamp(-1, 1).round()), "normal"),
        ("sigmoid", lucid.sigmoid, "normal"),
        ("cbrt", lambda t: t ** (1.0 / 3.0), "pos"),
    ]:
        add(name, f, kind=kind)

    # ── binary
    for name, f in {
        "add": lambda t: t + t.flip([0]),
        "sub": lambda t: t - t.flip([0]),
        "mul": lambda t: t * t.flip([0]),
        "div": lambda t: t / (t.flip([0]).abs() + 1),
        "floordiv": lambda t: t // (t.flip([0]).abs() + 1),
        "remainder": lambda t: t % 4,
        "fmod": lambda t: lucid.fmod(t, 4),
        "pow": lambda t: lucid.pow(t.abs() + 1, t.flip([0]).abs().clamp(max=2)),
        "maximum": lambda t: lucid.maximum(t, t.flip([0])),
        "minimum": lambda t: lucid.minimum(t, t.flip([0])),
        "eq": lambda t: t == t.flip([0]),
        "ne": lambda t: t != t.flip([0]),
        "lt": lambda t: t < t.flip([0]),
        "le": lambda t: t <= t.flip([0]),
        "gt": lambda t: t > t.flip([0]),
        "ge": lambda t: t >= t.flip([0]),
        "logical_and": lambda t: lucid.logical_and(t, t.flip([0])),
        "logical_or": lambda t: lucid.logical_or(t, t.flip([0])),
        "logical_xor": lambda t: lucid.logical_xor(t, t.flip([0])),
        "where": lambda t: lucid.where(t > 0, t, t.flip([0])),
        "masked_fill": lambda t: lucid.masked_fill(t, t > 1, 0),
        "atan2": lambda t: lucid.atan2(t, t.flip([0]) + 0.5),
        "hypot": lambda t: lucid.hypot(t, t.flip([0])),
        "xlogy": lambda t: lucid.xlogy(t, t.flip([0]).abs() + 1),
        "logaddexp": lambda t: lucid.logaddexp(t, t.flip([0])),
        "nextafter": lambda t: lucid.nextafter(t, t.flip([0])),
        "isclose": lambda t: lucid.isclose(t, t.flip([0])),
    }.items():
        add(name, f)
    # Not bool: the reference framework rejects ``lerp`` on bools; Lucid's
    # eager answers with bool arithmetic that is not worth pinning down.
    add("lerp", lambda t: lucid.lerp(t, t.flip([0]), 0.25), dtypes=("f32", "i64", "i32"))
    add("bitwise_and", lambda t: t & t.flip([0]), dtypes=("i64", "i32", "bool"))
    add("invert", lambda t: ~t, dtypes=("i64", "i32", "bool"))
    add("bitwise_or", lambda t: t | t.flip([0]), dtypes=("i64", "i32", "bool"))
    add("bitwise_xor", lambda t: t ^ t.flip([0]), dtypes=("i64", "i32", "bool"))
    add(
        "shift_left",
        lambda t: t << t.flip([0]).abs().clamp(max=3),
        dtypes=("i64", "i32"),
    )
    add(
        "shift_right",
        lambda t: t >> t.flip([0]).abs().clamp(max=3),
        dtypes=("i64", "i32"),
    )
    add("shift_right_scalar", lambda t: t >> 1, dtypes=("i64", "i32"))

    # ── reductions
    for name, f in {
        "sum": lambda t: t.sum(),
        "sum_dim": lambda t: t.sum(dim=1),
        "sum_keep": lambda t: t.sum(dim=0, keepdim=True),
        "mean": lambda t: t.mean(dim=1),
        "prod": lambda t: t.prod(dim=0),
        "max_all": lambda t: t.max(),
        "max_dim": lambda t: lucid.max(t, dim=0),
        "min_dim": lambda t: lucid.min(t, dim=1),
        "argmax": lambda t: lucid.argmax(t, dim=1),
        "argmin": lambda t: lucid.argmin(t, dim=0),
        "var": lambda t: lucid.var(t, dim=1),
        "std": lambda t: lucid.std(t, dim=0),
        "logsumexp": lambda t: lucid.logsumexp(t, dim=1),
        "cumsum": lambda t: lucid.cumsum(t, 0),
        "cumprod": lambda t: lucid.cumprod(t, 1),
        "cummax": lambda t: lucid.cummax(t, 0),
        "cummin": lambda t: lucid.cummin(t, 1),
        "all": lambda t: lucid.all(t),
        "any": lambda t: lucid.any(t),
        "norm": lambda t: lucid.linalg.norm(t),
        "vector_norm": lambda t: lucid.linalg.vector_norm(t, dim=1),
        "nansum": lambda t: lucid.nansum(t),
        "trace": lambda t: lucid.trace(t),
    }.items():
        add(name, f)

    # ── sort family
    for name, f in {
        "sort": lambda t: lucid.sort(t, dim=-1),
        "argsort": lambda t: lucid.argsort(t, dim=-1),
        "topk_values": lambda t: lucid.topk(t, 2, dim=-1)[0],
        "kthvalue": lambda t: lucid.kthvalue(t, 2, dim=0),
    }.items():
        add(name, f)

    # Which of two equal values ``topk`` reports first is unspecified, and
    # integers and bools tie constantly — indices are checked on floats.
    add("topk", lambda t: lucid.topk(t, 2, dim=-1), dtypes=("f32",))

    # ── shape / layout / indexing
    for name, f in {
        "reshape": lambda t: t.reshape(6, 4) * 1,
        "view": lambda t: t.view(2, 12) * 1,
        "flatten": lambda t: lucid.flatten(t) * 1,
        "unsqueeze": lambda t: t.unsqueeze(1) * 1,
        "squeeze": lambda t: t.unsqueeze(0).squeeze(0) * 1,
        "permute": lambda t: t.permute(1, 0) * 1,
        "transpose_contig": lambda t: t.mT.contiguous(),
        "broadcast_to": lambda t: lucid.broadcast_to(t[0], (3, 6)) * 1,
        "tile": lambda t: lucid.tile(t, (2, 1)),
        "repeat": lambda t: t.repeat(1, 2),
        "repeat_interleave": lambda t: lucid.repeat_interleave(t, 2, dim=0),
        "cat": lambda t: lucid.cat([t, t.flip([1])], dim=1),
        "stack": lambda t: lucid.stack([t, t.flip([0])]),
        "split": lambda t: lucid.split(t, 2, dim=0),
        "chunk": lambda t: lucid.chunk(t, 3, dim=1),
        "unbind": lambda t: lucid.unbind(t, 1),
        "flip": lambda t: lucid.flip(t, [0, 1]),
        "roll": lambda t: lucid.roll(t, [1, -2], [0, 1]),
        "tril": lambda t: lucid.tril(t),
        "triu_k1": lambda t: lucid.triu(t, 1),
        "tril_km1": lambda t: lucid.tril(t, -1),
        "diagonal": lambda t: lucid.diagonal(t),
        "pad": lambda t: F.pad(t, (1, 2)),
        "slice": lambda t: t[1:3, ::2] * 1,
        "narrow": lambda t: lucid.narrow(t, 1, 2, 3) * 1,
        "gather": lambda t: lucid.gather(t, 1, _idx(0, 5, 2, 1, 3, 4).reshape(1, 6).expand(4, 6)),
        "index_select": lambda t: lucid.index_select(t, 0, _idx(3, 0, 0, 2)),
        "fancy_index": lambda t: t[_idx(2, 0, 3)],
        "take_along_dim": lambda t: lucid.take_along_dim(t, lucid.argsort(t, dim=1), 1),
        "scatter_add": lambda t: lucid.scatter_add(
            lucid.zeros_like(t), 1, _idx(0, 1, 0, 2, 5, 5).reshape(1, 6).expand(4, 6), t
        ),
        "scatter": lambda t: lucid.scatter(
            lucid.zeros_like(t), 1, _idx(5, 4, 3, 2, 1, 0).reshape(1, 6).expand(4, 6), t
        ),
        "unfold_dim": lambda t: t.unfold(1, 3, 2) * 1,
        "one_hot": lambda t: F.one_hot(t.abs().clamp(max=4), 5),
        "meshgrid": lambda t: lucid.meshgrid(t[0], t[1], indexing="ij"),
        # Unequal lengths, or the xy axis swap would be invisible.
        "meshgrid_xy": lambda t: lucid.meshgrid(t[0][:3], t[1], indexing="xy"),
    }.items():
        add(name, f)

    # ── linear algebra
    add("matmul", lambda t: t @ t.mT, dtypes=("f32", "i64", "i32"))
    add("matmul_batched", lambda t: t @ t.mT, shape=(2, 3, 4), dtypes=("f32",))
    add("dot", lambda t: lucid.linalg.dot(t[0], t[1]))
    add("inner", lambda t: lucid.linalg.inner(t, t))
    add("outer", lambda t: lucid.linalg.outer(t[0], t[1]))
    add("tensordot", lambda t: lucid.tensordot(t, _w(6, 3), dims=1), dtypes=("f32",))
    add("matrix_power", lambda t: lucid.linalg.matrix_power(t, 3), shape=(4, 4))
    add(
        "det",
        lambda t: lucid.linalg.det(t + 3 * _eye(3)),
        shape=(3, 3),
        dtypes=("f32",),
    )
    add(
        "inv",
        lambda t: lucid.linalg.inv(t + 3 * _eye(2)),
        shape=(2, 2),
        dtypes=("f32",),
    )
    add("einsum", lambda t: lucid.einops.einsum("ij,kj->ik", t, t), dtypes=("f32",))

    # ── nn.functional (float)
    fo = ("f32",)
    add("linear", lambda t: F.linear(t, _w(3, 6), _w(3)), dtypes=fo)
    add("bilinear", lambda t: F.bilinear(t, t, _w(2, 6, 6), _w(2)), dtypes=fo)
    add("softmax", lambda t: F.softmax(t, dim=-1), dtypes=fo)
    add("log_softmax", lambda t: F.log_softmax(t, dim=-1), dtypes=fo)
    for act in (
        "relu6",
        "elu",
        "selu",
        "silu",
        "mish",
        "softplus",
        "hardsigmoid",
        "hardswish",
        "gelu",
    ):
        add(act, (lambda a: lambda t: getattr(F, a)(t))(act), dtypes=fo)
    add("gelu_tanh", lambda t: F.gelu(t, approximate="tanh"), dtypes=fo)
    add("leaky_relu", lambda t: F.leaky_relu(t, 0.1), dtypes=fo)
    add("layer_norm", lambda t: F.layer_norm(t, (6,), _w(6), _w(6)), dtypes=fo)
    add("rms_norm", lambda t: F.rms_norm(t, (6,), _w(6)), dtypes=fo)
    add(
        "group_norm",
        lambda t: F.group_norm(t, 2, _w(4), _w(4)),
        shape=(2, 4, 5),
        dtypes=fo,
    )
    add(
        "batch_norm_eval",
        lambda t: F.batch_norm(t, _w(3).abs(), _w(3).abs() + 1, _w(3), _w(3), False),
        shape=(4, 3, 5),
        dtypes=fo,
    )
    # Batch statistics — the matrix had only the eval form, so the train-mode
    # VJP was never checked on its own, and a 5-D one aborted the process
    # (BatchNorm3d: MPSGraph's fused reduction supports axes 0–3 only).
    add(
        "batch_norm_train",
        lambda t: F.batch_norm(t, None, None, _w(3), _w(3, seed=8), True),
        shape=(4, 3, 5),
        dtypes=fo,
    )
    add(
        "batch_norm3d_train",
        lambda t: F.batch_norm(t, None, None, _w(2), _w(2, seed=8), True),
        shape=(2, 2, 3, 3, 3),
        dtypes=fo,
    )
    # A grid built inside the traced function — a stub until Mask2Former's
    # point sampling needed it.
    add(
        "linspace_scale",
        lambda t: t * lucid.linspace(-1.0, 2.0, 6, device=t.device),
        dtypes=fo,
    )
    add("normalize", lambda t: F.normalize(t, dim=1), dtypes=fo)
    add(
        "conv1d",
        lambda t: F.conv1d(t, _w(4, 3, 3), _w(4), padding=1),
        shape=(2, 3, 8),
        dtypes=fo,
    )
    add(
        "conv2d",
        lambda t: F.conv2d(t, _w(4, 3, 3, 3), _w(4), stride=2),
        shape=(2, 3, 8, 8),
        dtypes=fo,
    )
    add(
        "conv3d",
        lambda t: F.conv3d(t, _w(2, 3, 2, 2, 2), None),
        shape=(1, 3, 4, 4, 4),
        dtypes=fo,
    )
    add(
        "conv_transpose1d",
        lambda t: F.conv_transpose1d(t, _w(3, 2, 3), None, stride=2),
        shape=(2, 3, 5),
        dtypes=fo,
    )
    add(
        "conv_transpose2d",
        lambda t: F.conv_transpose2d(t, _w(3, 2, 3, 3), None, stride=2),
        shape=(1, 3, 4, 4),
        dtypes=fo,
    )
    add(
        "conv_transpose3d",
        lambda t: F.conv_transpose3d(t, _w(3, 2, 2, 2, 2), None),
        shape=(1, 3, 3, 3, 3),
        dtypes=fo,
    )
    add("avg_pool2d", lambda t: F.avg_pool2d(t, 2), shape=(2, 3, 8, 8), dtypes=fo)
    add("max_pool2d", lambda t: F.max_pool2d(t, 2), shape=(2, 3, 8, 8), dtypes=fo)
    # A pad MPSGraph folds into the pool: global arg-max indices then counted
    # in the unpadded tensor — YOLOv3-tiny's backbone came back 87 % off.
    add(
        "pad_max_pool2d",
        lambda t: F.max_pool2d(F.pad(t, (0, 1, 0, 1), value=-1e4), 2, 1),
        shape=(2, 3, 5, 5),
        dtypes=fo,
    )
    add(
        "pad_max_pool3d",
        lambda t: F.max_pool3d(F.pad(t, (1, 0, 0, 1, 1, 0), value=-1e4), 2, 1),
        shape=(1, 2, 3, 3, 3),
        dtypes=fo,
    )
    # Every pooling rank, padded, strided, with the divisor options — and each
    # also read transposed, since MPSGraph's max-pool gradient kernel went
    # wrong exactly there.  None of these had a case, and none of 1-D / 3-D
    # pooling or interpolation had a VJP.
    add(
        "avg_pool2d_ceil_nopad",
        lambda t: F.avg_pool2d(t, 3, 2, ceil_mode=True, count_include_pad=False),
        shape=(2, 3, 8, 8),
        dtypes=fo,
    )
    add(
        "avg_pool2d_t",
        lambda t: F.avg_pool2d(t, 2).permute(0, 1, 3, 2),
        shape=(2, 3, 8, 8),
        dtypes=fo,
    )
    add(
        "max_pool2d_t",
        lambda t: F.max_pool2d(t, 2).permute(0, 1, 3, 2),
        shape=(2, 3, 8, 8),
        dtypes=fo,
    )
    add(
        "max_pool1d",
        lambda t: F.max_pool1d(t, 3, 2, padding=1),
        shape=(2, 3, 9),
        dtypes=fo,
    )
    add(
        "avg_pool1d",
        lambda t: F.avg_pool1d(t, 3, 2, padding=1),
        shape=(2, 3, 9),
        dtypes=fo,
    )
    add(
        "avg_pool1d_nopad",
        lambda t: F.avg_pool1d(t, 3, 2, padding=1, count_include_pad=False),
        shape=(2, 3, 9),
        dtypes=fo,
    )
    add("max_pool3d", lambda t: F.max_pool3d(t, 2), shape=(1, 2, 4, 4, 4), dtypes=fo)
    add(
        "avg_pool3d",
        lambda t: F.avg_pool3d(t, 3, 2, padding=1),
        shape=(1, 2, 5, 5, 5),
        dtypes=fo,
    )
    add(
        "interp_nearest",
        lambda t: F.interpolate(t, scale_factor=2, mode="nearest"),
        shape=(2, 3, 4, 4),
        dtypes=fo,
    )
    add(
        "interp_nearest_size",
        lambda t: F.interpolate(t, size=(3, 7), mode="nearest"),
        shape=(2, 3, 4, 5),
        dtypes=fo,
    )
    add(
        "interp_bilinear",
        lambda t: F.interpolate(t, size=(7, 5), mode="bilinear", align_corners=False),
        shape=(2, 3, 4, 3),
        dtypes=fo,
    )
    add(
        "interp_bilinear_ac",
        lambda t: F.interpolate(t, size=(7, 5), mode="bilinear", align_corners=True),
        shape=(2, 3, 4, 3),
        dtypes=fo,
    )
    add(
        "interp_trilinear",
        lambda t: F.interpolate(t, size=(3, 5, 4), mode="trilinear", align_corners=False),
        shape=(1, 2, 2, 3, 3),
        dtypes=fo,
    )
    # grid_sample, fed the image (the input gradient) and fed the grid (the
    # grid gradient), each against a fixed partner.  The grid reaches past
    # [-1, 1], so zero padding drops corners and border padding clamps.
    for suffix, gs_kw in (
        ("", {}),
        ("_border_ac", {"padding_mode": "border", "align_corners": True}),
        ("_nearest", {"mode": "nearest"}),
    ):
        add(
            f"grid_sample_x{suffix}",
            lambda t, kw=gs_kw: F.grid_sample(t, _w(2, 3, 4, 2, seed=11) * 2.5, **kw),
            shape=(2, 3, 5, 4),
            dtypes=fo,
        )
        add(
            f"grid_sample_grid{suffix}",
            lambda t, kw=gs_kw: F.grid_sample(_w(2, 3, 5, 4, seed=13), t * 0.2, **kw),
            shape=(2, 3, 4, 2),
            dtypes=fo,
        )
    add(
        "interp_trilinear_ac",
        lambda t: F.interpolate(t, size=(3, 5, 4), mode="trilinear", align_corners=True),
        shape=(1, 2, 2, 3, 3),
        dtypes=fo,
    )
    add(
        "interp_nearest3d",
        lambda t: F.interpolate(t, size=(3, 5, 4), mode="nearest"),
        shape=(1, 2, 2, 3, 3),
        dtypes=fo,
    )
    add(
        "interp_bilinear_t",
        lambda t: F.interpolate(t, scale_factor=2, mode="bilinear").permute(0, 1, 3, 2),
        shape=(2, 3, 4, 3),
        dtypes=fo,
    )
    add(
        "embedding",
        lambda t: F.embedding(t.abs().clamp(max=4), _w(5, 3)),
        dtypes=("i64", "i32"),
    )
    add("cross_entropy", lambda t: F.cross_entropy(t, _idx(0, 5, 2, 1)), dtypes=fo)
    add(
        "nll_loss",
        lambda t: F.nll_loss(F.log_softmax(t, dim=1), _idx(0, 5, 2, 1)),
        dtypes=fo,
    )
    add("mse_loss", lambda t: F.mse_loss(t, t.flip([0])), dtypes=fo)
    add("huber_loss", lambda t: F.huber_loss(t, t.flip([0])), dtypes=fo)
    add(
        "bce",
        lambda t: F.binary_cross_entropy(lucid.sigmoid(t), (t.flip([0]) > 0).to(lucid.float32)),
        dtypes=fo,
    )
    add(
        "bce_logits",
        lambda t: F.binary_cross_entropy_with_logits(t, (t.flip([0]) > 0).to(lucid.float32)),
        dtypes=fo,
    )
    add(
        "sdpa",
        lambda t: F.scaled_dot_product_attention(t, t.flip([2]), t * 0.5),
        shape=(2, 2, 5, 4),
        dtypes=fo,
    )
    add(
        "sdpa_causal",
        lambda t: F.scaled_dot_product_attention(t, t.flip([2]), t * 0.5, is_causal=True),
        shape=(2, 2, 5, 4),
        dtypes=fo,
    )
    add("dropout_eval", lambda t: F.dropout(t, 0.5, training=False), dtypes=fo)

    # ── random — values differ by design
    add(
        "dropout_train",
        lambda t: F.dropout(t, 0.5, training=True),
        dtypes=fo,
        random=True,
    )
    add("rand_like", lambda t: t + lucid.rand_like(t), dtypes=fo, random=True)
    add("randn_like", lambda t: t + lucid.randn_like(t), dtypes=fo, random=True)
    return C


CASES = _cases()
CASE_BY_NAME = {c.name: c for c in CASES}

#: (case, dtype) → why eager is the right answer there.  A fallback that is
#: not listed fails the matrix, so a new one is seen the day it appears.
_INT_MATMUL = "MPSGraph's matmul aborts on integers, so the emitter declines"
_BOOL_SUM = "MPSGraph cannot sum bool exactly (_DataMovement.h), so it declines"
_NO_OPS = "a cast to the dtype the tensor already has records no op"
EXPECTED_EAGER: dict[tuple[str, str], str] = {
    ("astype_f32", "f32"): _NO_OPS,
    ("astype_i64", "i64"): _NO_OPS,
    ("astype_bool", "bool"): _NO_OPS,
    ("scatter_add", "bool"): _BOOL_SUM,
    ("matmul", "i64"): _INT_MATMUL,
    ("matmul", "i32"): _INT_MATMUL,
    ("matrix_power", "i64"): _INT_MATMUL,
    ("matrix_power", "i32"): _INT_MATMUL,
    ("matrix_power", "bool"): _INT_MATMUL,
    # An integer inner product runs as ``A @ B^T``: no kernel takes integers.
    ("inner", "i64"): _INT_MATMUL,
    ("inner", "i32"): _INT_MATMUL,
}


@dataclass(frozen=True)
class Refusal:
    """The exception eager raises for one (case, dtype), on each device.

    ``None`` for a device means eager answers there, and the pair is run on
    that device like any other.
    """

    cpu: type[Exception] | None
    metal: type[Exception] | None
    why: str

    def on(self, device: str) -> type[Exception] | None:
        return self.cpu if device == "cpu" else self.metal


# The engine's own classes, both ``LucidError`` subclasses — the engine's
# ``NotImplementedError`` is not the builtin.
_NOT_IMPL = _C_engine.NotImplementedError


def _both(exc: type[Exception], why: str) -> Refusal:
    return Refusal(exc, exc, why)


def _each(name: str, dtypes: tuple[str, ...], r: Refusal) -> dict[tuple[str, str], Refusal]:
    return {(name, d): r for d in dtypes}


_INT = ("i64", "i32")
_NON_FLOAT = ("i64", "i32", "bool")

#: (case, dtype) → what eager raises instead of answering, for every declared
#: dtype eager refuses.  The matrices do not generate these pairs;
#: ``test_op_eager_rejects`` holds eager to the table both ways — each entry
#: must still raise exactly its exception type on each device (one that now
#: answers, or raises something else, is stale), and no pair outside the table
#: may raise at all.  float32 is never here: an eager float32 that raises is a
#: broken recipe.
#:
#: Measured 2026-09-25.  Not every entry is behaviour worth keeping: the last
#: group are inputs the reference framework answers, kept on record here
#: until the ops are fixed.
EAGER_REJECTS: dict[tuple[str, str], Refusal] = {
    # The reference framework refuses every one of these as well.
    ("neg", "bool"): _both(_C_engine.LucidError, "negating a bool; use ~"),
    # ``sub`` of two bools, refused at the public entry point.  ``scatter``
    # used to subtract bool masks inside the engine; it overwrites now
    # (CHA-158).
    ("sub", "bool"): _both(_C_engine.LucidError, "subtracting bools; use ^"),
    **_each("lerp", _INT, _both(_NOT_IMPL, "lerp takes floating tensors only")),
    **_each(
        "nextafter",
        _NON_FLOAT,
        # Every other clean refusal here is a subclass; this one is the base.
        _both(_C_engine.LucidError, "nextafter takes float32 / float64 only"),
    ),
    ("argmax", "bool"): _both(_NOT_IMPL, "no arg-reduction of bools"),
    ("argmin", "bool"): _both(_NOT_IMPL, "no arg-reduction of bools"),
    ("trace", "bool"): _both(_NOT_IMPL, "no trace of a bool matrix"),
    ("topk_values", "bool"): _both(_NOT_IMPL, "topk does not order bools"),
    ("kthvalue", "bool"): _both(_NOT_IMPL, "kthvalue does not order bools"),
    ("inner", "bool"): _both(_NOT_IMPL, "no inner product of bools"),
    **_each("hypot", _NON_FLOAT, _both(_NOT_IMPL, "hypot takes floating tensors")),
    **_each("logaddexp", _NON_FLOAT, _both(_NOT_IMPL, "logaddexp takes floating tensors")),
    # A norm has no integer form; the reference refuses every order (LCD-384).
    **_each("norm", _NON_FLOAT, _both(_C_engine.DtypeMismatch, "a norm takes floats")),
    **_each(
        "vector_norm",
        _NON_FLOAT,
        _both(_C_engine.DtypeMismatch, "a norm takes floats"),
    ),
}


def refusal(name: str, dtype: str, device: str) -> type[Exception] | None:
    """What eager raises for ``name`` on ``dtype`` on ``device``; ``None`` if it
    answers."""
    r = EAGER_REJECTS.get((name, dtype))
    return None if r is None else r.on(device)


def all_params() -> list[tuple[str, str]]:
    """Every (case, dtype) the replay runs: the declared dtypes eager answers."""
    return [(c.name, d) for c in CASES for d in c.dtypes if refusal(c.name, d, DEV) is None]


def _main() -> None:  # pragma: no cover — manual triage
    only = set(sys.argv[1:])
    counts: dict[str, int] = {}
    reached: set[str] = set()
    for name, dtype in all_params():
        if only and name not in only:
            continue
        print(f"RUN {name} {dtype}", file=sys.stderr, flush=True)
        o = run(CASE_BY_NAME[name], dtype)
        reached |= o.reached
        counts[o.status] = counts.get(o.status, 0) + 1
        if o.status != "ok":
            print(f"{o.status:7s} {name:18s} {dtype:5s} {o.detail}", flush=True)
    print("counts", counts)
    print("reached", len(reached), "op names")


if __name__ == "__main__":  # pragma: no cover
    _main()
