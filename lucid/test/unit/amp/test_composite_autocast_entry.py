"""An ``nn.functional`` composite takes its autocast dtype at its entry (LCD-475).

A composite built from other ops ran each of them under the live autocast,
so each applied its own row of the policy table mid-composite:

- ``local_response_norm`` of a rank-4+ input kept a half input's dtype under
  a CPU autocast, where the reference pools that rank with ``avg_pool3d``
  (FP32 on the CPU) and answers float32;
- ``grid_sample`` applied its row only inside the 4-D kernel: the reflection
  that runs before the kernel computed its coordinates at the grid's half
  precision, and the composite paths (5-D input, bicubic) applied no row, so
  a half input under a CPU autocast sampled in half and a half input beside
  a float32 grid was refused as a dtype mismatch;
- ``interpolate``'s resampling products ran ``matmul``'s LOWER row and turned
  a float32 image half precision;
- ``normalize`` over several dims took ``vector_norm``'s FP32 Metal row,
  which the reference's ``Tensor.norm`` does not have;
- ``embedding``'s ``max_norm`` write (an in-place op, which autocast does not
  touch) computed the rows in float32 and refused to copy them into a float16
  table;
- ``_keep_in_graph`` summed its zero under ``sum``'s FP32 Metal row, and the
  raw ``add`` refused the float32 zero beside a float16 output: a
  ``reflect`` / ``replicate`` pad, or ``nll_loss(reduction="none")``, of a
  half input that requires grad raised under a Metal autocast.

Each case runs under ``autocast(metal, float16)`` and ``autocast(cpu,
bfloat16)`` with a float32 and a half input: no error (the backward too),
the values of the same call outside autocast on the inputs at the dtype the
row gives, and the dtype the reference gives (``-m parity`` checks the table
against the reference).  The rows still wrong are strict xfails naming
LCD-614 — they must still run and match their values; only the dtype is
pinned.

A composite with a row enters through ``_autocast_entry``, one without
through ``_autocast_suspended``; the last tests check that every name handed
to ``_autocast_entry`` or ``amp_cast_inputs`` is a row of the table, so a
misspelt row cannot quietly become "keep".
"""

import ast
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.amp import autocast
from lucid.nn.functional import _pool_core
from lucid.test._fixtures.devices import metal_available

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet

_METAL = pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
_SETTINGS = [("metal", "float16"), ("cpu", "bfloat16")]
_INPUTS = ["float32", "half"]
# Eight units in the last place of the half dtype.
_TOL = {"float16": 8e-3, "bfloat16": 6e-2}
# The same ops on the same values: only the dtype's own rounding differs.
_EXACT = 1e-6


class _Make:
    """Inputs for one call: the same values in either library.

    ``dtype`` is the dtype the values are rounded to; ``compute``, when
    given, the dtype they are then held in — a half input as a row that
    computes in float32 sees it.
    """

    def __init__(
        self,
        lib: ModuleType,
        device: str,
        dtype: object,
        *,
        grad: bool = False,
        compute: object | None = None,
    ) -> None:
        self.lib, self.device, self.dtype = lib, device, dtype
        self.grad, self.compute = grad, compute
        self.made: list[Any] = []

    def _held(self, values: np.ndarray) -> Any:
        t = self.lib.tensor(values.astype(np.float32), device=self.device)
        t = t.to(self.dtype)
        return t if self.compute is None else t.to(self.compute)

    def __call__(self, *shape: int, seed: int = 0, scale: float = 1.0) -> Any:
        """An operand: it requires grad when the call is differentiated."""
        values = np.random.default_rng(seed).standard_normal(shape) * scale
        t = self._held(values).requires_grad_(self.grad)
        self.made.append(t)
        return t

    def unit(self, *shape: int) -> Any:
        """A target in ``(0.1, 0.9)``."""
        return self._held(np.random.default_rng(7).random(shape) * 0.8 + 0.1)

    def signs(self, n: int) -> Any:
        """A ``±1`` target."""
        return self._held(np.array([1.0, -1.0] * (n // 2)))

    def ints(self, values: list[int]) -> Any:
        return self.lib.tensor(values, device=self.device)


_Fn = Callable[[Any, _Make], Any]

_LOSS_NONE = (
    "LCD-614: the loss applies its FP32 row only through its reduction, so "
    "reduction='none' keeps a half input half (the reference: float32)"
)
_INNER_ROW = (
    "LCD-614: a part's FP32 Metal row applies inside (vector_norm, softplus); "
    "the reference keeps float16"
)


@dataclass(frozen=True)
class _Case:
    fn: _Fn
    # What a half input comes out as, on each device: ``"half"`` (the
    # autocast dtype, i.e. kept) or ``"float32"``.  A float32 input stays
    # float32 in every case here.
    metal: str
    cpu: str
    devices: tuple[str, ...] = ("metal", "cpu")
    # Whether the operands require grad (a backward then runs as well).
    grad: bool = True
    # Devices where a half input still comes out in another dtype than the
    # reference's, and why — pinned as strict xfails.
    pinned: tuple[str, ...] = ()
    reason: str = ""


def _grid(m: _Make, *shape: int, scale: float = 0.5) -> Any:
    return m(*shape, seed=1, scale=scale)


def _loss(fn: _Fn, pinned: tuple[str, ...]) -> _Case:
    return _Case(fn, "float32", "float32", pinned=pinned, reason=_LOSS_NONE)


_BOTH = ("metal", "cpu")

_CASES: dict[str, _Case] = {
    # The reference pools a rank-3 window with avg_pool2d (no row) and a
    # deeper one with avg_pool3d (FP32 on the CPU); pow is FP32 on Metal.
    "local_response_norm_3d": _Case(
        lambda F, m: F.local_response_norm(m(2, 5, 6), 3), "float32", "half"
    ),
    "local_response_norm_4d": _Case(
        lambda F, m: F.local_response_norm(m(2, 5, 3, 4), 3), "float32", "float32"
    ),
    "local_response_norm_5d": _Case(
        lambda F, m: F.local_response_norm(m(1, 5, 2, 3, 2), 2), "float32", "float32"
    ),
    # grid_sample: WIDEST on Metal (one dtype stays), FP32 on the CPU.
    "grid_sample_reflection": _Case(
        lambda F, m: F.grid_sample(
            m(1, 2, 3, 3),
            _grid(m, 1, 2, 2, 2, scale=1.5),
            padding_mode="reflection",
            align_corners=False,
        ),
        "half",
        "float32",
    ),
    "grid_sample_reflection_nearest": _Case(
        lambda F, m: F.grid_sample(
            m(1, 2, 3, 3),
            _grid(m, 1, 2, 2, 2, scale=1.5),
            mode="nearest",
            padding_mode="reflection",
            align_corners=True,
        ),
        "half",
        "float32",
    ),
    "grid_sample_5d": _Case(
        lambda F, m: F.grid_sample(
            m(1, 2, 2, 3, 3), _grid(m, 1, 2, 2, 2, 3), align_corners=False
        ),
        "half",
        "float32",
    ),
    "grid_sample_5d_nearest": _Case(
        lambda F, m: F.grid_sample(
            m(1, 2, 2, 3, 3),
            _grid(m, 1, 2, 2, 2, 3),
            mode="nearest",
            align_corners=True,
        ),
        "half",
        "float32",
    ),
    "grid_sample_bicubic": _Case(
        lambda F, m: F.grid_sample(
            m(1, 2, 3, 3), _grid(m, 1, 2, 2, 2), mode="bicubic", align_corners=False
        ),
        "half",
        "float32",
    ),
    # The resampling composite: no row, as the reference's kernels have none.
    "interpolate_linear": _Case(
        lambda F, m: F.interpolate(m(1, 2, 5), scale_factor=1.5, mode="linear"),
        "half",
        "half",
    ),
    "interpolate_bicubic": _Case(
        lambda F, m: F.interpolate(m(1, 1, 3, 3), scale_factor=2, mode="bicubic"),
        "half",
        "half",
    ),
    "interpolate_bilinear_fractional": _Case(
        lambda F, m: F.interpolate(m(1, 1, 3, 3), scale_factor=1.7, mode="bilinear"),
        "half",
        "half",
    ),
    "interpolate_bilinear_antialias": _Case(
        lambda F, m: F.interpolate(
            m(1, 1, 8, 8), size=(3, 3), mode="bilinear", antialias=True
        ),
        "half",
        "half",
    ),
    "interpolate_bicubic_antialias": _Case(
        lambda F, m: F.interpolate(
            m(1, 1, 8, 8), size=(3, 3), mode="bicubic", antialias=True
        ),
        "half",
        "half",
    ),
    "normalize_dims": _Case(
        lambda F, m: F.normalize(m(2, 3, 4), dim=(1, 2)), "half", "half"
    ),
    "normalize_all": _Case(
        lambda F, m: F.normalize(m(2, 3, 4), dim=(0, 1, 2)), "half", "half"
    ),
    "embedding_max_norm": _Case(
        lambda F, m: F.embedding(m.ints([0, 2, 1, 2]), m(4, 3), max_norm=0.5),
        "half",
        "half",
        grad=False,
    ),
    # _keep_in_graph sums its zero when the output's shape differs.  The CPU
    # half case keeps its dtype where the reference runs float32: the
    # documented ``pad`` departure (LCD-500 E2, AmpPolicyTable.def).
    "pad_reflect": _Case(
        lambda F, m: F.pad(m(1, 2, 5), (1, 2), mode="reflect"),
        "half",
        "half",
        devices=("metal",),
    ),
    "pad_replicate": _Case(
        lambda F, m: F.pad(m(1, 2, 5), (1, 2), mode="replicate"),
        "half",
        "half",
        devices=("metal",),
    ),
    # Same class, not fixed here (LCD-614).  nll_loss's "none" also runs
    # _keep_in_graph, whose Metal crash is fixed: the xfail only takes an
    # AssertionError, so the forward, backward and values must still pass.
    "nll_loss_none": _loss(
        lambda F, m: F.nll_loss(m(3, 4), m.ints([0, 1, 3]), reduction="none"),
        _BOTH,
    ),
    "l1_loss_none": _loss(
        lambda F, m: F.l1_loss(m(4, 4), m(4, 4, seed=1), reduction="none"), _BOTH
    ),
    "binary_cross_entropy_with_logits_none": _loss(
        lambda F, m: F.binary_cross_entropy_with_logits(
            m(4, 4), m.unit(4, 4), reduction="none"
        ),
        _BOTH,
    ),
    "margin_ranking_loss_none": _loss(
        lambda F, m: F.margin_ranking_loss(
            m(4), m(4, seed=1), m.signs(4), reduction="none"
        ),
        _BOTH,
    ),
    "hinge_embedding_loss_none": _loss(
        lambda F, m: F.hinge_embedding_loss(
            m(4, 4), m.signs(16).reshape(4, 4), reduction="none"
        ),
        _BOTH,
    ),
    "cross_entropy_none": _loss(
        lambda F, m: F.cross_entropy(m(3, 4), m.ints([0, 1, 3]), reduction="none"),
        ("cpu",),
    ),
    "cosine_embedding_loss_none": _loss(
        lambda F, m: F.cosine_embedding_loss(
            m(4, 4), m(4, 4, seed=1), m.signs(4), reduction="none"
        ),
        ("cpu",),
    ),
    "triplet_margin_loss_none": _loss(
        lambda F, m: F.triplet_margin_loss(
            m(3, 4), m(3, 4, seed=1), m(3, 4, seed=2), reduction="none"
        ),
        ("cpu",),
    ),
    "multi_margin_loss_none": _loss(
        lambda F, m: F.multi_margin_loss(m(3, 4), m.ints([0, 1, 3]), reduction="none"),
        ("cpu",),
    ),
    "pairwise_distance": _Case(
        lambda F, m: F.pairwise_distance(m(3, 4), m(3, 4, seed=1)),
        "half",
        "half",
        pinned=("metal",),
        reason=_INNER_ROW,
    ),
    "logsigmoid": _Case(
        lambda F, m: F.logsigmoid(m(4, 4, scale=4.0)),
        "half",
        "half",
        pinned=("metal",),
        reason=_INNER_ROW,
    ),
}


def _params(*, pin: bool) -> list[ParameterSet]:
    """Every (case, setting, input) a case is checked on; ``pin`` marks the
    LCD-614 rows as strict xfails."""
    params = []
    for name, case in sorted(_CASES.items()):
        for device, autocast_dt in _SETTINGS:
            if device not in case.devices:
                continue
            for input_dt in _INPUTS:
                marks = [_METAL] if device == "metal" else []
                if pin and input_dt == "half" and device in case.pinned:
                    marks.append(
                        pytest.mark.xfail(
                            strict=True, raises=AssertionError, reason=case.reason
                        )
                    )
                params.append(
                    pytest.param(
                        name,
                        device,
                        autocast_dt,
                        input_dt,
                        marks=marks,
                        id=f"{name}-{device}-{autocast_dt}-{input_dt}",
                    )
                )
    return params


def _dtype_name(dtype: object) -> str:
    return str(dtype).split(".")[-1]


def _input_dtype(lib: ModuleType, autocast_dt: str, input_dt: str) -> object:
    return getattr(lib, autocast_dt if input_dt == "half" else "float32")


def _want(case: _Case, device: str, autocast_dt: str, input_dt: str) -> str:
    if input_dt == "float32":
        return "float32"
    policy = case.metal if device == "metal" else case.cpu
    return autocast_dt if policy == "half" else "float32"


def _under_autocast(
    case: _Case, device: str, autocast_dt: str, input_dt: str, *, grad: bool
) -> tuple[Any, list[Any]]:
    """The case's output under the autocast, and the operands it made."""
    make = _Make(lucid, device, _input_dtype(lucid, autocast_dt, input_dt), grad=grad)
    with autocast(device_type=device, dtype=getattr(lucid, autocast_dt)):
        return case.fn(lucid.nn.functional, make), make.made


@pytest.mark.parametrize(
    ("name", "device", "autocast_dt", "input_dt"), _params(pin=False)
)
def test_a_composite_under_autocast_runs_and_keeps_its_values(
    name: str, device: str, autocast_dt: str, input_dt: str
) -> None:
    """Every cell, the LCD-614 ones included: forward, backward and values."""
    case = _CASES[name]
    out, made = _under_autocast(case, device, autocast_dt, input_dt, grad=case.grad)
    if case.grad:
        out.float().sum().backward()
        for t in made:
            assert t.grad is not None and t.grad.dtype == t.dtype, name

    # Outside autocast, on the same values held at the dtype the output has.
    dtype = _input_dtype(lucid, autocast_dt, input_dt)
    with lucid.no_grad():
        want = case.fn(
            lucid.nn.functional, _Make(lucid, device, dtype, compute=out.dtype)
        )
    # A row applied at the entry runs the same ops as the call outside;
    # the LCD-614 rows do not apply theirs there, so they round in between.
    tol = _TOL[autocast_dt] if case.pinned else _EXACT
    np.testing.assert_allclose(
        out.detach().float().numpy(),
        want.float().numpy(),
        rtol=tol,
        atol=tol,
        err_msg=name,
    )


@pytest.mark.parametrize(
    ("name", "device", "autocast_dt", "input_dt"), _params(pin=True)
)
def test_a_composite_under_autocast_answers_in_its_entry_dtype(
    name: str, device: str, autocast_dt: str, input_dt: str
) -> None:
    """The dtype alone, so a pinned LCD-614 cell hides nothing else."""
    case = _CASES[name]
    out, _ = _under_autocast(case, device, autocast_dt, input_dt, grad=False)
    assert _dtype_name(out.dtype) == _want(case, device, autocast_dt, input_dt)


_MIXED = {
    "5d": ((1, 2, 2, 3, 3), (1, 2, 2, 2, 3), "bilinear", "zeros"),
    "bicubic": ((1, 2, 3, 3), (1, 2, 2, 2), "bicubic", "zeros"),
    "reflection": ((1, 2, 3, 3), (1, 2, 2, 2), "bilinear", "reflection"),
    "reflection_nearest": ((1, 2, 3, 3), (1, 2, 2, 2), "nearest", "reflection"),
    "4d": ((1, 2, 3, 3), (1, 2, 2, 2), "bilinear", "zeros"),
}


@pytest.mark.parametrize("other", ["float32", "int64"])
@pytest.mark.parametrize("half_operand", ["input", "grid"])
@pytest.mark.parametrize("path", sorted(_MIXED))
@pytest.mark.parametrize(
    ("device", "autocast_dt"),
    [pytest.param(*s, marks=[_METAL] if s[0] == "metal" else []) for s in _SETTINGS],
)
def test_grid_sample_runs_a_half_operand_beside_another_in_float32(
    device: str, autocast_dt: str, path: str, half_operand: str, other: str
) -> None:
    """Every path, as the row says: the composites refused a float32 pair,
    the reflection mirrored a half grid before the kernel widened it, and an
    integer operand is promoted to float32 before the row, as the kernel's
    schema promotes it."""
    shape, grid_shape, mode, padding = _MIXED[path]
    half = getattr(lucid, autocast_dt)
    x_dt, grid_dt = (half, getattr(lucid, other))
    if half_operand == "grid":
        x_dt, grid_dt = grid_dt, x_dt
    x = _Make(lucid, device, x_dt)(*shape)
    grid = _Make(lucid, device, grid_dt)(*grid_shape, seed=1, scale=1.5)
    F = lucid.nn.functional
    kwargs = {"mode": mode, "padding_mode": padding, "align_corners": False}
    with autocast(device_type=device, dtype=half):
        out = F.grid_sample(x, grid, **kwargs)
    assert out.dtype == lucid.float32
    want = F.grid_sample(x.float(), grid.float(), **kwargs)
    np.testing.assert_allclose(out.numpy(), want.numpy(), rtol=_EXACT, atol=_EXACT)


def test_outside_autocast_grid_sample_still_refuses_a_mixed_pair() -> None:
    x = lucid.randn(1, 1, 2, 3, 3).to(lucid.bfloat16)
    with pytest.raises(TypeError, match="same dtype"):
        lucid.nn.functional.grid_sample(x, lucid.randn(1, 2, 2, 2, 3))


@pytest.mark.parametrize(
    ("device", "autocast_dt"),
    [pytest.param(*s, marks=[_METAL] if s[0] == "metal" else []) for s in _SETTINGS],
)
def test_embedding_max_norm_renormalises_a_half_table_in_place(
    device: str, autocast_dt: str
) -> None:
    half = getattr(lucid, autocast_dt)
    weight = (lucid.randn(4, 3, device=device) * 3).to(half)
    with autocast(device_type=device, dtype=half):
        lucid.nn.functional.embedding(
            lucid.tensor([0, 2], device=device), weight, max_norm=1.0
        )
    assert weight.dtype == half
    norms = lucid.linalg.vector_norm(weight.float(), dim=1)
    assert float(norms[0].item()) <= 1.0 + _TOL[autocast_dt]
    assert float(norms[2].item()) <= 1.0 + _TOL[autocast_dt]


# ── parity: the dtype the reference's autocast gives ────────────────────────


def _ref_device(ref: ModuleType, device: str) -> str:
    if device == "metal":
        if not ref.backends.mps.is_available():
            pytest.skip("the reference's MPS backend is unavailable")
        return "mps"
    return "cpu"


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "device", "autocast_dt", "input_dt"), _params(pin=False)
)
def test_the_dtype_matches_the_reference(
    ref: ModuleType, name: str, device: str, autocast_dt: str, input_dt: str
) -> None:
    case = _CASES[name]
    ref_device = _ref_device(ref, device)
    make = _Make(ref, ref_device, _input_dtype(ref, autocast_dt, input_dt))
    try:
        with ref.autocast(device_type=ref_device, dtype=getattr(ref, autocast_dt)):
            out = case.fn(ref.nn.functional, make)
    except NotImplementedError:
        # No bfloat16 CPU antialiasing kernel; no MPS multi_margin_loss.
        pytest.skip(f"the reference has no {ref_device} kernel for {name}")
    assert _dtype_name(out.dtype) == _want(case, device, autocast_dt, input_dt)


# ── every name an entry is handed is a row ──────────────────────────────────

_ENTRY_FUNCTIONS = frozenset({"_autocast_entry", "amp_cast_inputs"})
# Call sites whose name is built at run time, with every name each passes.
_COMPUTED_NAMES: dict[tuple[str, str], frozenset[str]] = {
    ("lucid/fft/_transforms.py", "_transform"): frozenset(
        {"fftn", "ifftn", "rfftn", "irfftn"}
    ),
    ("lucid/nn/functional/_pool_core.py", "_pool_autocast"): _pool_core._POOL_ROWS,
}
# The entry itself, which hands on its caller's name.
_FORWARDING = frozenset({("lucid/_ops/_autocast.py", "_autocast_entry")})


def _rows() -> frozenset[str]:
    return frozenset(name for name, _, _ in _C_engine.amp_policy_table())


class _EntryCalls(ast.NodeVisitor):
    """Each call to an entry function: its file, enclosing function and name."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.scope = ["<module>"]
        self.calls: list[tuple[str, str, int, ast.expr]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        called = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
        if called in _ENTRY_FUNCTIONS and node.args:
            self.calls.append((self.path, self.scope[-1], node.lineno, node.args[0]))
        self.generic_visit(node)


def _entry_calls() -> Iterator[tuple[str, str, int, ast.expr]]:
    root = Path(lucid.__file__).parent
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent).as_posix()
        text = path.read_text(encoding="utf-8")
        if rel.startswith("lucid/test/") or not any(
            f in text for f in _ENTRY_FUNCTIONS
        ):
            continue
        visitor = _EntryCalls(rel)
        visitor.visit(ast.parse(text))
        yield from visitor.calls


def test_every_literal_entry_name_is_a_row() -> None:
    rows = _rows()
    calls = list(_entry_calls())
    literal = [
        (f"{path}:{line}", arg.value)
        for path, _, line, arg in calls
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]
    # The scan reads the call sites, not nothing.
    assert len(literal) > 40
    assert [site for site in literal if site[1] not in rows] == []


def test_every_computed_entry_name_is_listed_and_a_row() -> None:
    rows = _rows()
    computed = {
        (path, func)
        for path, func, _, arg in _entry_calls()
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str))
    }
    assert computed == set(_COMPUTED_NAMES) | _FORWARDING
    for site, names in _COMPUTED_NAMES.items():
        assert names <= rows, (site, sorted(names - rows))


def test_the_pool_rows_are_the_pools_the_table_has() -> None:
    """``_pool_autocast`` keeps a pool that has no row; its set must be the
    table's, or a pool with one would quietly keep."""
    pools = {
        *(f"adaptive_{k}_pool{n}d" for k in ("avg", "max") for n in (1, 2, 3)),
        *(f"fractional_max_pool{n}d" for n in (2, 3)),
        *(f"max_unpool{n}d" for n in (1, 2, 3)),
    }
    assert {p for p in pools if p in _rows()} == _pool_core._POOL_ROWS
