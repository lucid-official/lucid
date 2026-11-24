"""A loss ranks its operands' dtypes where the reference does, and norms what it broadcast (LCD-1074).

The regression losses (``mse_loss``, ``l1_loss``, ``smooth_l1_loss``, ``huber_loss``) broadcast
input and target to one shape *before* the reference ranks their dtypes, so a 0-d operand is a
dimensioned one by then.  Lucid ranked the operands as they arrived, and a float32 scalar beside
a float16 vector came back float16.  Three more rank a 0-d operand too early:

- ``gaussian_nll_loss`` aligns ``var`` first (a 0-d one beside a vector is unsqueezed to rank 1),
  so the three dtypes meet after the alignment;
- ``cosine_similarity`` normalised each operand over ``dim`` *before* broadcasting, so a 0-d
  operand was always +-1 (``tensor(1.342)`` where the reference gives ``0.9487``), a 0-d operand
  beside a 2-D one with ``dim=1`` raised, and the dtype was ranked by rank;
- ``binary_cross_entropy_with_logits`` returns the *target's* dtype (the reference writes into a
  tensor of that dtype), and ``binary_cross_entropy`` refuses a target of another dtype.

The owner is :func:`lucid.nn.functional._loss_checks._loss_dtype` with the ``Tiers`` of
:mod:`lucid._promotion`, chosen per loss (``_RANKED_AFTER_BROADCAST``), and
``cosine_similarity`` broadcasts before it normalises.

The table sweeps every loss that has two or three tensor operands beside each other, plus the
two distance functions.  A cell is one loss at one operand-shape combination, every dtype pair
(three for the three-operand losses) and every reduction.  Under ``-m parity`` the verdict,
dtype, shape, values and gradients of each cell are the reference's, on each device.  The
reference-free tests below pin the rule and the cases the issue named.
"""

import itertools
import math
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.amp import autocast
from lucid.nn.functional._loss_checks import _OPERAND_DTYPES, _RANKED_AFTER_BROADCAST, _SAME_DTYPE
from lucid.test._fixtures.devices import device_supports, metal_available

Shape = tuple[int, ...]

_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]
_DTYPES = ("float16", "bfloat16", "float32", "float64")
_ALL_REDUCTIONS = ("none", "mean", "sum")

#: ``(low, high, kind)`` of the values each operand role takes.
_DOMAINS: dict[str, tuple[float, float, str]] = {
    "x": (-2.0, 2.0, "real"),
    "p": (0.1, 0.9, "real"),
    "pos": (0.5, 2.5, "real"),
    "lp": (-3.0, -0.2, "real"),
    "sign": (-1.0, 1.0, "sign"),
    "bin": (0.0, 1.0, "bin"),
}


@dataclass(frozen=True)
class Spec:
    """One function of the table: its operands, the shape combinations to sweep, its options."""

    name: str
    roles: tuple[str, ...]
    shapes: tuple[tuple[Shape, ...], ...]
    options: dict[str, object] = field(default_factory=dict)
    #: ``None`` for a function with no ``reduction`` argument.
    reductions: tuple[str, ...] | None = _ALL_REDUCTIONS
    #: Whether each operand takes a gradient in the reference.
    grads: tuple[bool, ...] = (True, True)


_PAIR_SHAPES: tuple[tuple[Shape, ...], ...] = (
    ((), (2,)),
    ((2,), ()),
    ((), (2, 3)),
    ((2, 3), ()),
    ((), ()),
    ((2, 1), (3,)),
)
_SAME_SHAPES: tuple[tuple[Shape, ...], ...] = (((2, 3), (2, 3)), ((), ()))
_TRIPLET = (((3, 4), (3, 4), (3, 4)), ((4,), (4,), (4,)), ((1, 4), (3, 4), (3, 4)))

SPECS: tuple[Spec, ...] = (
    # Ranked after broadcasting: the regression losses.
    Spec("mse_loss", ("x", "x"), _PAIR_SHAPES),
    Spec("l1_loss", ("x", "x"), _PAIR_SHAPES),
    Spec("smooth_l1_loss", ("x", "x"), _PAIR_SHAPES, {"beta": 0.7}),
    Spec("huber_loss", ("x", "x"), _PAIR_SHAPES, {"delta": 0.8}),
    # Ranked as they arrive: they must not be moved with the rest.
    Spec("hinge_embedding_loss", ("x", "sign"), _PAIR_SHAPES, grads=(True, False)),
    Spec("kl_div", ("lp", "p"), _PAIR_SHAPES, reductions=("none", "sum", "batchmean")),
    Spec("poisson_nll_loss", ("pos", "pos"), _PAIR_SHAPES),
    Spec(
        "soft_margin_loss",
        ("x", "sign"),
        (((2,), ()), ((), ()), ((2, 3), (2, 3))),
        grads=(True, False),
    ),
    Spec("binary_cross_entropy", ("p", "bin"), _SAME_SHAPES),
    Spec("binary_cross_entropy_with_logits", ("x", "bin"), _SAME_SHAPES),
    Spec(
        "gaussian_nll_loss",
        ("x", "x", "pos"),
        (((3,), (), ()), ((3,), (3,), ()), ((), (), ()), ((3, 4), (3, 4), (3, 4))),
        grads=(True, True, True),
    ),
    Spec(
        "margin_ranking_loss",
        ("x", "x", "sign"),
        (((3,), (3,), (3,)), ((), (), ()), ((), (3,), (3,)), ((3,), (), (3,))),
        grads=(True, True, False),
    ),
    Spec(
        "cosine_embedding_loss",
        ("x", "x", "sign"),
        (((3, 4), (3, 4), (3,)), ((4,), (4,), ())),
        grads=(True, True, False),
    ),
    Spec("triplet_margin_loss", ("x", "x", "x"), _TRIPLET, {"swap": True}, grads=(True,) * 3),
    Spec("triplet_margin_with_distance_loss", ("x", "x", "x"), _TRIPLET, grads=(True,) * 3),
    # The distance functions: no reduction.
    Spec(
        "cosine_similarity",
        ("x", "x"),
        (((), (2,)), ((2,), ()), ((), ())),
        {"dim": 0},
        reductions=None,
    ),
    Spec(
        "cosine_similarity",
        ("x", "x"),
        (((), (2, 3)), ((2, 3), ()), ((2, 3), (3,)), ((2, 1), (1, 3))),
        {"dim": 1},
        reductions=None,
    ),
    Spec(
        "cosine_similarity",
        ("x", "x"),
        (((2, 3, 4), (4,)), ((), (2, 3, 4)), ((2, 3, 4), ())),
        {"dim": -1, "eps": 1e-6},
        reductions=None,
    ),
    Spec("pairwise_distance", ("x", "x"), _PAIR_SHAPES, reductions=None),
    Spec(
        "pairwise_distance",
        ("x", "x"),
        (((2, 3, 4), (4,)), ((), (2, 3, 4))),
        {"p": 1.5, "keepdim": True},
        reductions=None,
    ),
)

#: The losses with no operand that meets another by broadcasting or promotion: class indices,
#: or one sequence.  Their operand dtypes are the table of ``test_loss_operand_dtypes.py``.
_EXEMPT = frozenset(
    {
        "cross_entropy",
        "nll_loss",
        "multi_margin_loss",
        "multilabel_margin_loss",
        "ctc_loss",
        # The reference refuses a 0-d input here and Lucid takes it (see the xfail below).
        "multilabel_soft_margin_loss",
    }
)

_CELLS = [
    pytest.param(spec, shapes, id=f"{spec.name}{sorted(spec.options.items())}-{i}")
    for spec in SPECS
    for i, shapes in enumerate(spec.shapes)
]


def _values(role: str, shape: Shape, seed: int) -> object:
    """Deterministic data of ``shape`` in role ``role``'s domain, as nested lists."""
    low, high, kind = _DOMAINS[role]
    rnd = random.Random(seed)

    def build(dims: Shape) -> object:
        if dims:
            return [build(dims[1:]) for _ in range(dims[0])]
        v = rnd.uniform(low, high)
        if kind == "sign":
            return 1.0 if v >= 0.0 else -1.0
        if kind == "bin":
            return 1.0 if v >= 0.5 else 0.0
        return round(v, 3)

    return build(shape)


def _flat(x: object) -> list[float]:
    if isinstance(x, list):
        return [v for item in x for v in _flat(item)]
    assert isinstance(x, (int, float))
    return [float(x)]


def _name(dtype: object) -> str:
    """A dtype's bare name, ``float32`` for either framework's."""
    return str(dtype).rsplit(".", 1)[-1]


def _tol(*dtypes: str) -> float:
    if any("16" in d for d in dtypes):
        return 6e-2
    return 1e-6 if all(d == "float64" for d in dtypes) else 1e-4


def _close(got: list[float], want: list[float], tol: float) -> bool:
    return len(got) == len(want) and all(
        (math.isnan(g) and math.isnan(w)) or abs(g - w) <= tol * max(1.0, abs(w))
        for g, w in zip(got, want, strict=True)
    )


def _call(
    module: object, spec: Spec, tensors: Sequence[object], reduction: str | None
) -> object | Exception:
    """The function on ``tensors``, or the refusal it raised."""
    options = dict(spec.options)
    if reduction is not None:
        options["reduction"] = reduction
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return getattr(module, spec.name)(*tensors, **options)
    except (RuntimeError, TypeError, ValueError, NotImplementedError, IndexError) as exc:
        return exc


def _backward(out: object) -> Exception | None:
    try:
        out.sum().backward()  # type: ignore[attr-defined]  # a tensor of either framework
    except (RuntimeError, TypeError, ValueError) as exc:
        return exc
    return None


def _cell_problem(
    ref: ModuleType,
    spec: Spec,
    shapes: tuple[Shape, ...],
    dtypes: tuple[str, ...],
    reduction: str | None,
    device: str,
) -> str | None:
    """How one cell differs from the reference, or ``None``."""
    data = [_values(r, s, i + 1) for i, (r, s) in enumerate(zip(spec.roles, shapes, strict=True))]
    want_ops = [
        ref.tensor(v, dtype=getattr(ref, d)).requires_grad_(g)
        for v, d, g in zip(data, dtypes, spec.grads, strict=True)
    ]
    got_ops = [
        lucid.tensor(v, dtype=getattr(lucid, d), device=device).requires_grad_(g)
        for v, d, g in zip(data, dtypes, spec.grads, strict=True)
    ]
    want = _call(ref.nn.functional, spec, want_ops, reduction)
    got = _call(F, spec, got_ops, reduction)
    if isinstance(want, Exception) != isinstance(got, Exception):
        return f"verdict: reference {_describe(want)}, lucid {_describe(got)}"
    if isinstance(want, Exception) or isinstance(got, Exception):
        return None
    if _name(got.dtype) != _name(want.dtype):  # type: ignore[attr-defined]
        return f"dtype: reference {want.dtype}, lucid {got.dtype}"  # type: ignore[attr-defined]
    if tuple(got.shape) != tuple(want.shape):  # type: ignore[attr-defined]
        return f"shape: reference {tuple(want.shape)}, lucid {tuple(got.shape)}"  # type: ignore[attr-defined]
    tol = _tol(*dtypes)
    got_v, want_v = _flat(got.tolist()), _flat(want.detach().tolist())  # type: ignore[attr-defined]
    if not _close(got_v, want_v, tol):
        return f"value: reference {want_v[:3]}, lucid {got_v[:3]}"
    ref_error, lucid_error = _backward(want), _backward(got)
    if lucid_error is not None:
        return f"lucid backward raised {lucid_error!r}"
    if ref_error is not None:
        # The reference's own backward refuses a mixed pair for ``huber_loss``; Lucid answers.
        return None
    for i, (w, g) in enumerate(zip(want_ops, got_ops, strict=True)):
        if w.grad is None or g.grad is None:
            if (w.grad is None) != (g.grad is None):
                return f"grad[{i}] presence"
            continue
        if _name(g.grad.dtype) != _name(w.grad.dtype):
            return f"grad[{i}] dtype: reference {w.grad.dtype}, lucid {g.grad.dtype}"
        if tuple(g.grad.shape) != tuple(w.grad.shape):
            return f"grad[{i}] shape: reference {tuple(w.grad.shape)}, lucid {tuple(g.grad.shape)}"
        if not _close(_flat(g.grad.tolist()), _flat(w.grad.tolist()), tol):
            return f"grad[{i}] value: reference {_flat(w.grad.tolist())[:3]}"
    return None


def _holds(device: str, *dtypes: str) -> bool:
    return all(device_supports(device, getattr(lucid, d)) for d in dtypes)


def _skip_unless_held(device: str, *dtypes: str) -> None:
    if not _holds(device, *dtypes):
        pytest.skip(f"{device} cannot hold {' and '.join(dtypes)}")


def _describe(outcome: object) -> str:
    return "refuses" if isinstance(outcome, Exception) else "answers"


# ── against the reference ─────────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("spec", "shapes"), _CELLS)
def test_a_cell_is_the_references(
    ref: ModuleType, spec: Spec, shapes: tuple[Shape, ...], device: str
) -> None:
    """Every dtype combination and reduction of one loss at one shape combination."""
    problems: list[str] = []
    for dtypes in itertools.product(_DTYPES, repeat=len(spec.roles)):
        if not _holds(device, *dtypes):
            continue
        for reduction in spec.reductions or (None,):
            problem = _cell_problem(ref, spec, shapes, dtypes, reduction, device)
            if problem is not None:
                problems.append(f"{dtypes} {reduction}: {problem}")
    assert not problems, f"{len(problems)} cell(s) differ:\n" + "\n".join(problems[:12])


def test_every_loss_is_swept_or_exempt() -> None:
    """A new loss has to be in the table, or named as one that has no operand to broadcast."""
    swept = {spec.name for spec in SPECS}
    assert set(_OPERAND_DTYPES) == (swept & set(_OPERAND_DTYPES)) | _EXEMPT
    assert swept - set(_OPERAND_DTYPES) == {"cosine_similarity", "pairwise_distance"}


def test_the_losses_ranked_after_broadcasting_are_the_regression_ones() -> None:
    assert _RANKED_AFTER_BROADCAST == {"mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss"}
    assert {"binary_cross_entropy"} == _SAME_DTYPE


# ── the rule, without the reference ───────────────────────────────────────


def _scalar_and_vector(dx: str, dt: str, device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    return (
        lucid.tensor(0.5, dtype=getattr(lucid, dx), device=device),
        lucid.tensor([1.0, 2.0], dtype=getattr(lucid, dt), device=device),
    )


_PAIRS = [
    pytest.param(dx, dt, id=f"{dx}-{dt}")
    for dx, dt in itertools.product(_DTYPES, repeat=2)
    if dx != dt
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("dx", "dt"), _PAIRS)
@pytest.mark.parametrize("name", ["mse_loss", "l1_loss", "smooth_l1_loss"])
def test_a_0d_operand_meets_a_vector_at_its_own_width(
    name: str, dx: str, dt: str, device: str
) -> None:
    """Both operands are dimensioned once broadcast, so the loss is in their promoted dtype
    whichever of them is the 0-d one: float32 for a float32 scalar beside a float16 vector."""
    _skip_unless_held(device, dx, dt)
    want = lucid.promote_types(getattr(lucid, dx), getattr(lucid, dt))
    scalar, vector = _scalar_and_vector(dx, dt, device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, t in ((scalar, vector), (vector, scalar)):
            assert getattr(F, name)(x, t).dtype == want
            assert getattr(F, name)(x, t, reduction="none").dtype == want


@pytest.mark.parametrize("device", _DEVICES)
def test_the_loss_modules_forward_the_rule(device: str) -> None:
    scalar, vector = _scalar_and_vector("float32", "float16", device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for module in (nn.MSELoss(), nn.L1Loss(), nn.SmoothL1Loss()):
            assert module(scalar, vector).dtype == lucid.float32


@pytest.mark.parametrize("device", _DEVICES)
def test_a_0d_variance_is_a_vector_when_the_dtypes_meet(device: str) -> None:
    """``gaussian_nll_loss`` unsqueezes a 0-d variance beside a vector before the dtypes are
    ranked: bfloat16 beside float16 is float32, not float16."""
    x = lucid.tensor([0.1, 0.4, 0.9], dtype=lucid.float16, device=device)
    target = lucid.tensor([0.0, 0.5, 1.0], dtype=lucid.float16, device=device)
    var = lucid.tensor(0.8, dtype=lucid.bfloat16, device=device)
    assert F.gaussian_nll_loss(x, target, var).dtype == lucid.float32


# ── cosine similarity ─────────────────────────────────────────────────────


@pytest.mark.parametrize("device", _DEVICES)
def test_cosine_similarity_of_a_scalar_is_the_vector_s_angle(device: str) -> None:
    """The issue's case: the scalar is repeated along ``dim`` first, so it is the angle between
    ``[0.5, 0.5]`` and ``[1, 2]`` — not the sum of two unit-vector products."""
    a = lucid.tensor(0.5, device=device)
    b = lucid.tensor([1.0, 2.0], device=device)
    want = (0.5 * 1.0 + 0.5 * 2.0) / (math.sqrt(0.5) * math.sqrt(5.0))
    assert abs(F.cosine_similarity(a, b, dim=0).item() - want) < 1e-6
    assert abs(F.cosine_similarity(b, a, dim=0).item() - want) < 1e-6


_BROADCAST_CASES = [
    pytest.param((), (2, 3), 1, id="scalar-matrix-dim1"),
    pytest.param((2, 3), (), 1, id="matrix-scalar-dim1"),
    pytest.param((), (2, 3, 4), -1, id="scalar-3d-last"),
    pytest.param((2, 3), (3,), 0, id="matrix-vector-dim0"),
    pytest.param((2, 1, 4), (3, 4), 1, id="stretched-dim1"),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("sx", "st", "dim"), _BROADCAST_CASES)
def test_cosine_similarity_is_the_similarity_of_the_broadcast_operands(
    sx: Shape, st: Shape, dim: int, device: str
) -> None:
    """Broadcasting the operands by hand first changes neither the value nor the gradients."""
    shape = lucid.broadcast_shapes(sx, st)
    a = lucid.tensor(_values("x", sx, 1), device=device, requires_grad=True)
    b = lucid.tensor(_values("x", st, 2), device=device, requires_grad=True)
    out = F.cosine_similarity(a, b, dim=dim)
    a2 = a.detach().requires_grad_(True)
    b2 = b.detach().requires_grad_(True)
    want = F.cosine_similarity(a2.broadcast_to(shape), b2.broadcast_to(shape), dim=dim)
    assert tuple(out.shape) == tuple(want.shape)
    assert _close(_flat(out.tolist()), _flat(want.tolist()), 1e-6)
    out.sum().backward()
    want.sum().backward()
    assert a.grad is not None and b.grad is not None
    assert a2.grad is not None and b2.grad is not None
    assert _close(_flat(a.grad.tolist()), _flat(a2.grad.tolist()), 1e-5)
    assert _close(_flat(b.grad.tolist()), _flat(b2.grad.tolist()), 1e-5)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("dx", "dt"), _PAIRS)
def test_cosine_similarity_dtype_ranks_the_expanded_operands(dx: str, dt: str, device: str) -> None:
    _skip_unless_held(device, dx, dt)
    want = lucid.promote_types(getattr(lucid, dx), getattr(lucid, dt))
    scalar, vector = _scalar_and_vector(dx, dt, device)
    assert F.cosine_similarity(scalar, vector, dim=0).dtype == want
    assert F.cosine_similarity(vector, scalar, dim=0).dtype == want


# ── binary cross-entropy ──────────────────────────────────────────────────

_FLOAT_PAIRS = [
    pytest.param(dx, dt, id=f"{dx}-{dt}") for dx, dt in itertools.product(_DTYPES, repeat=2)
]


def _bce_operands(dx: str, dt: str, device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    return (
        lucid.tensor([0.2, 0.7, 0.9], dtype=getattr(lucid, dx), device=device),
        lucid.tensor([0.0, 1.0, 1.0], dtype=getattr(lucid, dt), device=device),
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("dx", "dt"), _FLOAT_PAIRS)
def test_binary_cross_entropy_refuses_a_target_of_another_dtype(
    dx: str, dt: str, device: str
) -> None:
    _skip_unless_held(device, dx, dt)
    p, y = _bce_operands(dx, dt, device)
    if dx == dt:
        assert F.binary_cross_entropy(p, y).dtype == p.dtype
        return
    with pytest.raises(_C_engine.DtypeMismatch, match="binary_cross_entropy"):
        F.binary_cross_entropy(p, y)
    with pytest.raises(RuntimeError, match=f"lucid.{dx}.*lucid.{dt}"):
        nn.BCELoss()(p, y)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(("dx", "dt"), _FLOAT_PAIRS)
def test_binary_cross_entropy_with_logits_is_in_the_target_s_dtype(
    dx: str, dt: str, device: str
) -> None:
    """The reference writes the loss into a tensor of the target's dtype; the gradient
    comes back in the input's."""
    _skip_unless_held(device, dx, dt)
    x, y = _bce_operands(dx, dt, device)
    x.requires_grad_(True)
    for reduction in _ALL_REDUCTIONS:
        out = F.binary_cross_entropy_with_logits(x, y, reduction=reduction)
        assert out.dtype == y.dtype
    out.backward()
    assert x.grad is not None and x.grad.dtype == x.dtype


@pytest.mark.parametrize("device", _DEVICES)
def test_a_mixed_pair_is_the_autocast_policy_s_not_a_refusal(device: str) -> None:
    """Under autocast the row decides the operands' dtypes, and the Metal row keeps a float16
    sigmoid beside the float32 target: that is no mixture the caller made."""
    half = lucid.float16 if device == "metal" else lucid.bfloat16
    p = lucid.tensor([0.2, 0.7, 0.9], dtype=half, device=device)
    y = lucid.tensor([0.0, 1.0, 1.0], dtype=lucid.float32, device=device)
    with pytest.raises(_C_engine.DtypeMismatch):
        F.binary_cross_entropy(p, y)
    with autocast(device_type=device, dtype=half):
        out = F.binary_cross_entropy(p, y)
    assert math.isfinite(out.item())


# ── a different class, held until it is fixed ─────────────────────────────


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LCD-1074 FOUND: the reference raises IndexError for a 0-d input to "
        "multilabel_soft_margin_loss (it takes the mean over dim -1); Lucid answers"
    ),
)
def test_multilabel_soft_margin_loss_refuses_a_0d_input() -> None:
    with pytest.raises(IndexError):
        F.multilabel_soft_margin_loss(lucid.tensor(0.5), lucid.tensor(1.0))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LCD-1074 FOUND: the reference casts an integer operand beside a floating one for "
        "cosine_similarity (common dtype float32); Lucid refuses any integer operand"
    ),
)
def test_cosine_similarity_takes_an_integer_beside_a_float() -> None:
    a = lucid.tensor([1, 2, 3])
    b = lucid.tensor([1.0, 0.0, 2.0])
    assert abs(F.cosine_similarity(a, b, dim=0).item() - 0.83666) < 1e-4
