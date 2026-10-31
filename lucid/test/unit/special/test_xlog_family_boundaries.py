"""``xlogy``, ``special.xlog1py`` and ``special.entr`` at ``0 · log`` (CHA-230).

The defect class: ``x · log(y)`` and ``x · log1p(y)`` are 0 wherever
``x == 0``, and the backward never forms ``0 · inf`` there, whatever ``y``
is.  With ``x != 0`` the plain definition applies, so ``xlogy(2, 0)`` is
``-inf``.  Before, the public ``xlogy`` masked the *product*: it returned 0
for ``xlogy(2, 0)``, and its gradient was NaN at ``x == 0`` with ``y`` infinite
or negative.  ``xlog1py`` had a guard that changed nothing.  Meanwhile
``lucid.distributions`` kept correct private copies of its own.

There is now one implementation,
``lucid._ops.composite.elementwise._x_times_log``.  It guards the operand,
not the product.  ``xlogy``, ``xlog1py``, ``entr``, ``kl_div`` and the
distributions' log-densities all reach it.  The grid
{xlogy, xlog1py} × x ∈ {0, 2, -1} × y ∈ {0, -1, inf, nan, 0.5} is checked on
CPU and Metal, value and both gradients, in two ways:

* against the documented convention written out in :func:`_spec` (always
  runs);
* under ``-m parity``, against the reference.  Four cells differ on purpose
  (``_DEVIATIONS``), and the test pins the reference's value there too, so
  the docstrings that name it stay true.
"""

import ast
import math
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.special
from lucid.test._fixtures.devices import metal_available

_INF = math.inf
_NAN = math.nan

_FUNCS: dict[str, Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]] = {
    "xlogy": lucid.xlogy,
    "xlog1py": lucid.special.xlog1py,
}
_XS = (0.0, 2.0, -1.0)
_YS = (0.0, -1.0, _INF, _NAN, 0.5)

#: (function, x, y) -> (component, the reference's value there, why Lucid differs).
_DEVIATIONS: dict[tuple[str, float, float], tuple[str, float, str]] = {
    ("xlogy", 0.0, 0.0): (
        "dy",
        _NAN,
        "the reference's d/dy is 0/0; the convention keeps the backward finite",
    ),
    ("xlog1py", 0.0, -1.0): (
        "dy",
        _NAN,
        "the reference's d/dy is 0/0; the convention keeps the backward finite",
    ),
    ("xlogy", 0.0, _INF): (
        "dx",
        _INF,
        "a composite cannot give a value of 0 the gradient +inf without 0*inf",
    ),
    ("xlog1py", 0.0, _INF): (
        "dx",
        _INF,
        "a composite cannot give a value of 0 the gradient +inf without 0*inf",
    ),
}


def _grid() -> list[object]:
    return [
        pytest.param(name, x, y, id=f"{name}-x{x}-y{y}")
        for name in _FUNCS
        for x in _XS
        for y in _YS
    ]


def _log(name: str, y: float) -> float:
    """The plain ``log(y)`` or ``log1p(y)``, extended to the reals and NaN."""
    u = y if name == "xlogy" else y + 1.0
    if math.isnan(u) or u < 0.0:
        return _NAN
    if u == 0.0:
        return -_INF
    if math.isinf(u):
        return _INF
    return math.log(y) if name == "xlogy" else math.log1p(y)


def _spec(name: str, x: float, y: float) -> tuple[float, float, float]:
    """``(value, d/dx, d/dy)`` as the docstrings define them."""
    if math.isnan(y):
        return _NAN, _NAN, _NAN
    pole = 0.0 if name == "xlogy" else -1.0
    if x == 0.0 and (y <= pole or math.isinf(y)):
        return 0.0, 0.0, 0.0
    lg = _log(name, y)
    u = y - pole  # the plain log's argument; 0 here means x != 0
    value = 0.0 if x == 0.0 else x * lg
    dy = math.copysign(_INF, x) if u == 0.0 else x / u
    return value, lg, dy


def _same(got: float, want: float) -> bool:
    if math.isnan(want):
        return math.isnan(got)
    if math.isinf(want):
        return got == want
    return math.isclose(got, want, rel_tol=1e-5, abs_tol=1e-6)


def _run(name: str, x: float, y: float, device: str) -> tuple[float, float, float]:
    xt = lucid.tensor(x, requires_grad=True, device=device)
    yt = lucid.tensor(y, requires_grad=True, device=device)
    out = _FUNCS[name](xt, yt)
    out.backward()
    assert xt.grad is not None and yt.grad is not None
    return out.item(), xt.grad.item(), yt.grad.item()


def _run_ref(R: ModuleType, name: str, x: float, y: float) -> tuple[float, float, float]:
    xt = R.tensor(x, requires_grad=True)
    yt = R.tensor(y, requires_grad=True)
    fn = R.xlogy if name == "xlogy" else R.special.xlog1py
    out = fn(xt, yt)
    out.backward()
    return out.item(), xt.grad.item(), yt.grad.item()


_PARTS = ("value", "dx", "dy")


@pytest.mark.parametrize(("name", "x", "y"), _grid())
def test_the_grid_follows_the_convention(name: str, x: float, y: float, device: str) -> None:
    got = _run(name, x, y, device)
    want = _spec(name, x, y)
    for part, g, w in zip(_PARTS, got, want):
        assert _same(g, w), f"{name}({x}, {y}) {part} on {device}: {g} != {w}"


@pytest.mark.parity
@pytest.mark.parametrize(("name", "x", "y"), _grid())
def test_the_grid_matches_the_reference(
    name: str, x: float, y: float, device: str, ref: ModuleType
) -> None:
    got = _run(name, x, y, device)
    want = _run_ref(ref, name, x, y)
    deviation = _DEVIATIONS.get((name, x, y))
    for i, part in enumerate(_PARTS):
        if deviation is not None and deviation[0] == part:
            _, ref_value, why = deviation
            assert _same(want[i], ref_value), (
                f"the reference changed at {name}({x}, {y}) {part}: {want[i]} — "
                "update the docstrings and this table"
            )
            assert _same(got[i], _spec(name, x, y)[i]), why
            continue
        assert _same(got[i], want[i]), f"{name}({x}, {y}) {part} on {device}: {got[i]} != {want[i]}"


# ── The class, beyond the grid ───────────────────────────────────────────────

_EDGE_YS = [0.0, -0.0, -1.0, -2.0, -_INF, _INF, 1e-30, 0.5, 1e30]


@pytest.mark.parametrize("name", list(_FUNCS))
@pytest.mark.parametrize("x0", [0.0, -0.0])
def test_x_zero_gives_zero_and_finite_gradients_for_every_y(
    name: str, x0: float, device: str
) -> None:
    """The one place the convention speaks: ``x == 0`` (either sign of zero)."""
    xt = lucid.tensor([x0] * len(_EDGE_YS), requires_grad=True, device=device)
    yt = lucid.tensor(_EDGE_YS, requires_grad=True, device=device)
    out = _FUNCS[name](xt, yt)
    out.sum().backward()
    assert xt.grad is not None and yt.grad is not None
    assert all(v == 0.0 for v in out.tolist()), out.tolist()
    assert all(math.isfinite(g) for g in xt.grad.tolist()), xt.grad.tolist()
    assert all(math.isfinite(g) for g in yt.grad.tolist()), yt.grad.tolist()


@pytest.mark.parametrize("name", list(_FUNCS))
def test_broadcast_gradients_reduce_to_each_operand(name: str, device: str) -> None:
    xs = [[0.0], [2.0], [-1.0]]
    ys = [0.5, 3.0]
    xt = lucid.tensor(xs, requires_grad=True, device=device)
    yt = lucid.tensor(ys, requires_grad=True, device=device)
    _FUNCS[name](xt, yt).sum().backward()
    assert xt.grad is not None and yt.grad is not None
    assert xt.grad.shape == (3, 1) and yt.grad.shape == (2,)
    want_dx = [sum(_spec(name, r[0], y)[1] for y in ys) for r in xs]
    want_dy = [sum(_spec(name, r[0], y)[2] for r in xs) for y in ys]
    got_dx = [r[0] for r in xt.grad.tolist()]
    assert all(_same(g, w) for g, w in zip(got_dx, want_dx)), (got_dx, want_dx)
    got_dy = yt.grad.tolist()
    assert all(_same(g, w) for g, w in zip(got_dy, want_dy)), (got_dy, want_dy)


@pytest.mark.skipif(not metal_available(), reason="needs Metal")
def test_a_python_scalar_follows_the_tensor_operand_to_metal() -> None:
    """A scalar used to become a CPU tensor and raise DeviceMismatch."""
    y = lucid.tensor([0.0, 0.5], device="metal")
    left = lucid.xlogy(2.0, y)
    right = lucid.xlogy(y, 3.0)
    assert left.device.type == "metal" and right.device.type == "metal"
    assert left.tolist()[0] == -_INF
    assert _same(right.tolist()[1], 0.5 * math.log(3.0))


def test_two_python_scalars() -> None:
    assert lucid.xlogy(2.0, 0.0).item() == -_INF
    assert lucid.xlogy(0.0, 0.0).item() == 0.0


@pytest.mark.parametrize("name", list(_FUNCS))
def test_integer_operands_are_taken_in_the_default_float(name: str) -> None:
    """``xlog1py`` used to raise DtypeMismatch where ``xlogy`` promoted."""
    out = _FUNCS[name](lucid.tensor([0, 2]), lucid.tensor([0, 3]))
    assert out.dtype == lucid.get_default_dtype()
    assert _same(out.tolist()[1], _spec(name, 2.0, 3.0)[0])


@pytest.mark.parametrize("name", list(_FUNCS))
def test_half_precision_keeps_its_dtype(name: str, device: str) -> None:
    x = lucid.tensor([0.0, 2.0], dtype=lucid.float16, device=device)
    y = lucid.tensor([0.0, 3.0], dtype=lucid.float16, device=device)
    out = _FUNCS[name](x, y)
    assert out.dtype == lucid.float16
    assert out.tolist()[0] == 0.0


# ── entr = -xlogy(x, x) ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("x", "value", "grad"),
    [
        # The value takes the 0 log 0 convention; d/dx is the one-sided +inf (LCD-772).
        (0.0, 0.0, _INF),
        (0.5, -0.5 * math.log(0.5), -(1.0 + math.log(0.5))),
        (1.0, 0.0, -1.0),
        (-1.0, -_INF, _NAN),
        (_NAN, _NAN, _NAN),
    ],
)
def test_entr_takes_the_convention_from_xlogy(
    x: float, value: float, grad: float, device: str
) -> None:
    xt = lucid.tensor(x, requires_grad=True, device=device)
    out = lucid.special.entr(xt)
    out.backward()
    assert xt.grad is not None
    assert _same(out.item(), value), out.item()
    assert _same(xt.grad.item(), grad), xt.grad.item()


def test_entr_of_zero_is_positive_zero() -> None:
    assert math.copysign(1.0, lucid.special.entr(lucid.tensor(0.0)).item()) == 1.0


# ── One owner ────────────────────────────────────────────────────────────────


def test_no_second_copy_of_the_convention() -> None:
    """``distributions`` kept its own ``_xlogy`` / ``_xlog1py`` beside the public ops.

    Any function named like an ``x · log`` helper outside the two public
    definitions is a second source of truth.
    """
    root = Path(lucid.__file__).parent
    allowed = {("_ops/composite/elementwise.py", "xlogy"), ("special/__init__.py", "xlog1py")}
    found: list[tuple[str, str]] = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("test/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.lstrip("_") in ("xlogy", "xlog1py"):
                found.append((rel, node.name))
    assert sorted(set(found) - allowed) == [], found
