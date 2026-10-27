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
  half input that requires grad raised under a Metal autocast;
- the losses applied their FP32 row only through their reduction, so with
  ``reduction="none"`` a half input stayed half; and the reduction widened
  a loss with no row (``multilabel_soft_margin_loss``) that the reference
  keeps (LCD-614).

Each case runs under ``autocast(metal, float16)`` and ``autocast(cpu,
bfloat16)`` with a float32 and a half input: no error (the backward too),
the values of the same call outside autocast on the inputs at the dtype the
row gives, and the dtype the reference gives (``-m parity`` checks the table
against the reference).  Every loss is in the table at every reduction.
``pairwise_distance`` and ``logsigmoid`` have no row, and their parts'
FP32 Metal rows (``vector_norm``, ``softplus``) widened them (LCD-614).

A composite with a row enters through ``_autocast_entry`` (a loss through
``_amp_op``, the same entry around its whole call), one without through
``_autocast_suspended``; the last tests check that every name handed to an
entry is a row of the table, so a misspelt row cannot quietly become "keep",
and that each loss enters through its own row or, with none, decomposes as
the reference's does.
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

    def unit(self, *shape: int, seed: int = 7) -> Any:
        """A target in ``(0.1, 0.9)``."""
        return self._held(np.random.default_rng(seed).random(shape) * 0.8 + 0.1)

    def prob(self, *shape: int) -> Any:
        """A probability operand in ``(0.1, 0.9)`` that, like ``__call__``'s,
        requires grad when the call is differentiated."""
        t = self.unit(*shape, seed=11).requires_grad_(self.grad)
        self.made.append(t)
        return t

    def pos(self, *shape: int) -> Any:
        """A positive operand in ``(0.5, 1.5)`` — a variance, a weight."""
        return self._held(np.random.default_rng(3).random(shape) + 0.5)

    def signs(self, n: int) -> Any:
        """A ``±1`` target."""
        return self._held(np.array([1.0, -1.0] * (n // 2)))

    def ints(self, values: list[int]) -> Any:
        return self.lib.tensor(values, device=self.device)


_Fn = Callable[[Any, _Make], Any]


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
    # No row: its parts apply theirs, as the reference's do, so it rounds
    # to half between them and its values are the call outside autocast's
    # to within half precision only.
    decomposed: bool = False
    # Operands of two dtypes.  The reference's MPS backend aborts the
    # process on those, so the parity check reads its CPU only.
    mixed: bool = False


def _grid(m: _Make, *shape: int, scale: float = 0.5) -> Any:
    return m(*shape, seed=1, scale=scale)


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
        lambda F, m: F.grid_sample(m(1, 2, 2, 3, 3), _grid(m, 1, 2, 2, 2, 3), align_corners=False),
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
        lambda F, m: F.interpolate(m(1, 1, 8, 8), size=(3, 3), mode="bilinear", antialias=True),
        "half",
        "half",
    ),
    "interpolate_bicubic_antialias": _Case(
        lambda F, m: F.interpolate(m(1, 1, 8, 8), size=(3, 3), mode="bicubic", antialias=True),
        "half",
        "half",
    ),
    "normalize_dims": _Case(lambda F, m: F.normalize(m(2, 3, 4), dim=(1, 2)), "half", "half"),
    "normalize_all": _Case(lambda F, m: F.normalize(m(2, 3, 4), dim=(0, 1, 2)), "half", "half"),
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
    # No row, as the reference's norm and log_sigmoid have none: the FP32
    # Metal rows of vector_norm and softplus inside must not apply.
    "pairwise_distance": _Case(
        lambda F, m: F.pairwise_distance(m(3, 4), m(3, 4, seed=1)), "half", "half"
    ),
    "pairwise_distance_float32_operand": _Case(
        lambda F, m: F.pairwise_distance(m(3, 4), m(3, 4, seed=1).float()),
        "float32",
        "float32",
        mixed=True,
    ),
    "logsigmoid": _Case(lambda F, m: F.logsigmoid(m(4, 4, scale=4.0)), "half", "half"),
}


# ── the losses, at every reduction (LCD-614) ────────────────────────────────

_LossFn = Callable[[Any, _Make, str], Any]
_REDUCTIONS = ("none", "mean", "sum")
_F32 = "float32"


@dataclass(frozen=True)
class _Loss:
    """One loss call, and what a half input comes out as at each reduction.

    ``metal`` / ``cpu`` is ``"half"`` or ``"float32"``, or one of the two
    per reduction.  A loss with a row is FP32 on both devices except
    ``bce_loss`` (KEEP on Metal) and ``gaussian_nll_loss`` (KEEP on the
    CPU); one without decomposes, as the reference's does.
    """

    fn: _LossFn
    metal: str | dict[str, str] = _F32
    cpu: str | dict[str, str] = _F32
    decomposed: bool = False
    mixed: bool = False


def _at(spec: str | dict[str, str], reduction: str) -> str:
    return spec if isinstance(spec, str) else spec[reduction]


def _ce_target(m: _Make) -> Any:
    return m.ints([0, 1, 3])


#: Keyed by (the function, a variant); every loss in ``F.__all__`` is here.
_LOSSES: dict[tuple[str, str], _Loss] = {
    ("mse_loss", ""): _Loss(lambda F, m, r: F.mse_loss(m(4, 4), m(4, 4, seed=1), reduction=r)),
    # Weighted, the reference's mse_loss and l1_loss are composites: ``pow``
    # is FP32 on MPS, ``abs`` and ``mul`` have no row, ``sum`` is FP32 on MPS.
    ("mse_loss", "weighted"): _Loss(
        lambda F, m, r: F.mse_loss(m(4, 4), m(4, 4, seed=1), reduction=r, weight=m.pos(4, 4)),
        cpu="half",
        decomposed=True,
    ),
    ("l1_loss", ""): _Loss(lambda F, m, r: F.l1_loss(m(4, 4), m(4, 4, seed=1), reduction=r)),
    ("l1_loss", "weighted"): _Loss(
        lambda F, m, r: F.l1_loss(m(4, 4), m(4, 4, seed=1), reduction=r, weight=m.pos(4, 4)),
        metal={"none": "half", "mean": _F32, "sum": _F32},
        cpu="half",
        decomposed=True,
    ),
    ("l1_loss", "float32_target"): _Loss(
        lambda F, m, r: F.l1_loss(m(4, 4), m(4, 4, seed=1).float(), reduction=r),
        mixed=True,
    ),
    ("smooth_l1_loss", ""): _Loss(
        lambda F, m, r: F.smooth_l1_loss(m(4, 4), m(4, 4, seed=1), reduction=r, beta=0.5)
    ),
    ("huber_loss", ""): _Loss(lambda F, m, r: F.huber_loss(m(4, 4), m(4, 4, seed=1), reduction=r)),
    ("huber_loss", "weighted"): _Loss(
        lambda F, m, r: F.huber_loss(m(4, 4), m(4, 4, seed=1), reduction=r, weight=m.pos(4, 4))
    ),
    ("cross_entropy", ""): _Loss(
        lambda F, m, r: F.cross_entropy(m(3, 4), _ce_target(m), reduction=r)
    ),
    ("cross_entropy", "weighted_smoothed"): _Loss(
        lambda F, m, r: F.cross_entropy(
            m(3, 4), _ce_target(m), m.pos(4), reduction=r, label_smoothing=0.1
        )
    ),
    ("cross_entropy", "probabilities"): _Loss(
        lambda F, m, r: F.cross_entropy(m(3, 4), m.unit(3, 4), reduction=r)
    ),
    ("cross_entropy", "float32_weight"): _Loss(
        lambda F, m, r: F.cross_entropy(m(3, 4), _ce_target(m), m.pos(4).float(), reduction=r),
        mixed=True,
    ),
    ("nll_loss", ""): _Loss(lambda F, m, r: F.nll_loss(m(3, 4), _ce_target(m), reduction=r)),
    ("nll_loss", "weighted"): _Loss(
        lambda F, m, r: F.nll_loss(m(3, 4), _ce_target(m), m.pos(4), reduction=r)
    ),
    ("binary_cross_entropy", ""): _Loss(
        lambda F, m, r: F.binary_cross_entropy(m.prob(4, 4), m.unit(4, 4), reduction=r),
        metal="half",
    ),
    ("binary_cross_entropy", "weighted"): _Loss(
        lambda F, m, r: F.binary_cross_entropy(
            m.prob(4, 4), m.unit(4, 4), m.pos(4, 4), reduction=r
        ),
        metal="half",
    ),
    ("binary_cross_entropy_with_logits", ""): _Loss(
        lambda F, m, r: F.binary_cross_entropy_with_logits(m(4, 4), m.unit(4, 4), reduction=r)
    ),
    ("binary_cross_entropy_with_logits", "weighted"): _Loss(
        lambda F, m, r: F.binary_cross_entropy_with_logits(
            m(4, 4), m.unit(4, 4), m.pos(4, 4), reduction=r, pos_weight=m.pos(4)
        )
    ),
    ("binary_cross_entropy_with_logits", "float32_pos_weight"): _Loss(
        lambda F, m, r: F.binary_cross_entropy_with_logits(
            m(4, 4), m.unit(4, 4), reduction=r, pos_weight=m.pos(4).float()
        ),
        mixed=True,
    ),
    ("kl_div", ""): _Loss(lambda F, m, r: F.kl_div(m(4, 4), m.unit(4, 4), reduction=r)),
    ("kl_div", "log_target"): _Loss(
        lambda F, m, r: F.kl_div(m(4, 4), m(4, 4, seed=1), reduction=r, log_target=True)
    ),
    ("triplet_margin_loss", ""): _Loss(
        lambda F, m, r: F.triplet_margin_loss(
            m(3, 4), m(3, 4, seed=1), m(3, 4, seed=2), reduction=r
        )
    ),
    # The reference's distance is ``aten::norm``, which has no MPS row, and
    # its ``sum`` is FP32 on MPS; the CPU keeps throughout.
    ("triplet_margin_with_distance_loss", ""): _Loss(
        lambda F, m, r: F.triplet_margin_with_distance_loss(
            m(3, 4), m(3, 4, seed=1), m(3, 4, seed=2), reduction=r
        ),
        metal={"none": "half", "mean": "half", "sum": _F32},
        cpu="half",
        decomposed=True,
    ),
    ("cosine_embedding_loss", ""): _Loss(
        lambda F, m, r: F.cosine_embedding_loss(m(4, 4), m(4, 4, seed=1), m.signs(4), reduction=r)
    ),
    ("margin_ranking_loss", ""): _Loss(
        lambda F, m, r: F.margin_ranking_loss(m(4), m(4, seed=1), m.signs(4), reduction=r)
    ),
    ("hinge_embedding_loss", ""): _Loss(
        lambda F, m, r: F.hinge_embedding_loss(m(4, 4), m.signs(16).reshape(4, 4), reduction=r)
    ),
    ("poisson_nll_loss", ""): _Loss(
        lambda F, m, r: F.poisson_nll_loss(m(4, 4), m.unit(4, 4), reduction=r)
    ),
    ("gaussian_nll_loss", ""): _Loss(
        lambda F, m, r: F.gaussian_nll_loss(m(4, 4), m(4, 4, seed=1), m.pos(4, 4), reduction=r),
        cpu="half",
    ),
    # Lucid's Metal row is FP32 too; the reference has no MPS kernel.
    ("ctc_loss", ""): _Loss(
        lambda F, m, r: F.ctc_loss(
            m(5, 2, 4),
            m.ints([[1, 2], [3, 1]]),
            m.ints([5, 5]),
            m.ints([2, 2]),
            reduction=r,
        )
    ),
    ("multi_margin_loss", ""): _Loss(
        lambda F, m, r: F.multi_margin_loss(m(3, 4), _ce_target(m), reduction=r)
    ),
    ("multi_margin_loss", "weighted"): _Loss(
        lambda F, m, r: F.multi_margin_loss(m(3, 4), _ce_target(m), weight=m.pos(4), reduction=r)
    ),
    ("multilabel_margin_loss", ""): _Loss(
        lambda F, m, r: F.multilabel_margin_loss(
            m(3, 4),
            m.ints([[0, 2, -1, 0], [1, -1, 0, 0], [3, 0, -1, 0]]),
            reduction=r,
        )
    ),
    ("soft_margin_loss", ""): _Loss(
        lambda F, m, r: F.soft_margin_loss(m(4, 4), m.signs(16).reshape(4, 4), reduction=r)
    ),
    # No row; the reference's parts are logsigmoid and a sum, FP32 on MPS.
    ("multilabel_soft_margin_loss", ""): _Loss(
        lambda F, m, r: F.multilabel_soft_margin_loss(m(3, 4), m.unit(3, 4), reduction=r),
        cpu="half",
        decomposed=True,
    ),
    ("multilabel_soft_margin_loss", "weighted"): _Loss(
        lambda F, m, r: F.multilabel_soft_margin_loss(m(3, 4), m.unit(3, 4), m.pos(4), reduction=r),
        cpu="half",
        decomposed=True,
    ),
}


def _bound(fn: _LossFn, reduction: str) -> _Fn:
    return lambda F, m: fn(F, m, reduction)


for (_name, _variant), _loss in _LOSSES.items():
    for _r in _REDUCTIONS:
        _CASES["_".join(filter(None, (_name, _variant, _r)))] = _Case(
            _bound(_loss.fn, _r),
            _at(_loss.metal, _r),
            _at(_loss.cpu, _r),
            decomposed=_loss.decomposed,
            mixed=_loss.mixed,
        )


def _params() -> list[ParameterSet]:
    """Every (case, setting, input) a case is checked on."""
    params = []
    for name, case in sorted(_CASES.items()):
        for device, autocast_dt in _SETTINGS:
            if device not in case.devices:
                continue
            for input_dt in _INPUTS:
                marks = [_METAL] if device == "metal" else []
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


@pytest.mark.parametrize(("name", "device", "autocast_dt", "input_dt"), _params())
def test_a_composite_under_autocast_runs_and_keeps_its_values(
    name: str, device: str, autocast_dt: str, input_dt: str
) -> None:
    """Every cell: forward, backward and values."""
    case = _CASES[name]
    out, made = _under_autocast(case, device, autocast_dt, input_dt, grad=case.grad)
    if case.grad:
        out.float().sum().backward()
        for t in made:
            assert t.grad is not None and t.grad.dtype == t.dtype, name

    # Outside autocast, on the same values held at the dtype the output has.
    dtype = _input_dtype(lucid, autocast_dt, input_dt)
    with lucid.no_grad():
        want = case.fn(lucid.nn.functional, _Make(lucid, device, dtype, compute=out.dtype))
    # A row applied at the entry runs the same ops as the call outside; a
    # decomposed case's parts apply theirs, so it rounds between them.
    tol = _TOL[autocast_dt] if case.decomposed else _EXACT
    np.testing.assert_allclose(
        out.detach().float().numpy(),
        want.float().numpy(),
        rtol=tol,
        atol=tol,
        err_msg=name,
    )


@pytest.mark.parametrize(("name", "device", "autocast_dt", "input_dt"), _params())
def test_a_composite_under_autocast_answers_in_its_entry_dtype(
    name: str, device: str, autocast_dt: str, input_dt: str
) -> None:
    """The dtype alone, against the static table — no reference needed."""
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
        lucid.nn.functional.embedding(lucid.tensor([0, 2], device=device), weight, max_norm=1.0)
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
@pytest.mark.parametrize(("name", "device", "autocast_dt", "input_dt"), _params())
def test_the_dtype_matches_the_reference(
    ref: ModuleType, name: str, device: str, autocast_dt: str, input_dt: str
) -> None:
    case = _CASES[name]
    if case.mixed and device == "metal":
        pytest.skip("the reference's MPS backend aborts the process on mixed dtypes")
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

_ENTRY_FUNCTIONS = frozenset({"_autocast_entry", "amp_cast_inputs", "_amp_op"})
# Call sites whose name is built at run time, with every name each passes.
_COMPUTED_NAMES: dict[tuple[str, str], frozenset[str]] = {
    ("lucid/fft/_transforms.py", "_transform"): frozenset({"fftn", "ifftn", "rfftn", "irfftn"}),
    ("lucid/nn/functional/_pool_core.py", "_pool_autocast"): _pool_core._POOL_ROWS,
}
# The entries themselves, which hand on their caller's name.
_FORWARDING = frozenset(
    {
        ("lucid/_ops/_autocast.py", "_autocast_entry"),
        ("lucid/_ops/_autocast.py", "at_entry"),
    }
)


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
        if rel.startswith("lucid/test/") or not any(f in text for f in _ENTRY_FUNCTIONS):
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


# ── each loss enters through its own row, or decomposes ─────────────────────

#: A loss whose row is not its function's name.
_LOSS_ROWS = {
    "cross_entropy": "cross_entropy_loss",
    "binary_cross_entropy": "bce_loss",
    "binary_cross_entropy_with_logits": "bce_with_logits",
}
#: Losses whose row applies inside rather than around the call: their
#: weighted form is a composite in the reference, so only the unweighted one
#: has the row — ``mse_loss`` and ``huber_loss`` through their fused kernel's
#: own entry, ``l1_loss`` through ``_autocast_entry`` on that branch.
_ROW_INSIDE = frozenset({"mse_loss", "huber_loss", "l1_loss"})


def _public_losses() -> set[str]:
    from lucid.nn.functional import loss as module

    F = lucid.nn.functional
    return {n for n in F.__all__ if getattr(F, n).__module__ == module.__name__}


def test_every_loss_is_in_the_table() -> None:
    """A new loss is checked at every reduction once it is listed here."""
    assert {name for name, _ in _LOSSES} == _public_losses()


def test_a_loss_enters_through_its_row_and_only_one_with_a_row_does() -> None:
    """``_amp_op`` with its row on every loss that has one (but those in
    ``_ROW_INSIDE``); none on a loss without, whose parts apply their rows as
    the reference's composite's do."""
    from lucid.nn.functional import loss as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    decorated: dict[str, str | None] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _public_losses():
            rows = [
                d.args[0].value
                for d in node.decorator_list
                if isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id == "_amp_op"
                and isinstance(d.args[0], ast.Constant)
            ]
            decorated[node.name] = rows[0] if rows else None
    assert decorated.keys() == _public_losses()
    table = _rows()
    for name, row in sorted(decorated.items()):
        want = _LOSS_ROWS.get(name, name)
        assert want in table or name not in _ROW_INSIDE, name
        entered = want in table and name not in _ROW_INSIDE
        assert row == (want if entered else None), name
