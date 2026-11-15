"""The ops that refuse a bool tensor, pinned op by op against the reference (LCD-1013).

The reference's CPU turns a bool operand away from a long list of ops where
Lucid used to promote the mask silently (``mask.mean()``, ``mask.abs()``,
``softmax(mask)``).  :data:`lucid._ops._bool_refusal.BOOL_REFUSED` is the one
table of those ops; this file is the test of it:

* every row here names the op it exercises, and a final test fails when an op
  of the table has no row -- a refusal cannot be added to the table and left
  unpinned, and an op cannot be dropped from it unnoticed;
* every row is replayed on the reference (``-m parity``), so a row that drifts
  from the reference's answer -- the verdict *or* the error class -- fails;
* a refusal is the same on both devices, whichever spelling reached it (free
  function, ``Tensor`` method, in-place twin, ``abs()``, ``nn.functional``,
  an ``nn`` module) and whether the call is traced by ``lucid.compile``.

Kinds, named after how the reference refuses:

``kernel``
    a ``NotImplementedError`` (a kernel with no bool instantiation).
``guard``
    a plain ``RuntimeError`` that is *not* a ``NotImplementedError`` (a check
    on the input dtype).
``refused``
    some ``RuntimeError``: the operand is refused for a reason of its own
    (a bool ``bins`` where the reference wants the input's dtype).
``accepted``
    answered, and answered with the reference's values: the spelling the
    reference offers in place of the refused one.

Every Lucid refusal is a ``RuntimeError``, as the reference's are.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import pytest

import lucid
from lucid._ops._bool_refusal import BOOL_REFUSED

Kind = Literal["kernel", "guard", "refused", "accepted"]

# ``fw`` is the framework module under test, ``t`` a bool vector, ``m`` a bool
# matrix.  Both frameworks spell every call below the same way; the extra
# tensors a call needs are made from ``t`` (``t[:1]``, ``t.long()``).
_Call = Callable[[ModuleType, object, object], object]


@dataclass(frozen=True)
class _Row:
    """One call: the op of ``BOOL_REFUSED`` it pins (``None`` for an accepted
    spelling), how the reference answers it, and the call."""

    op: str | None
    kind: Kind
    call: _Call
    #: False for a spelling only Lucid has, which the reference cannot replay.
    parity: bool = True


def _kernel(op: str, call: _Call, *, parity: bool = True) -> _Row:
    return _Row(op, "kernel", call, parity)


def _guard(op: str, call: _Call) -> _Row:
    return _Row(op, "guard", call)


def _refused(op: str, call: _Call) -> _Row:
    return _Row(op, "refused", call)


def _accepted(call: _Call) -> _Row:
    return _Row(None, "accepted", call)


_TABLE: dict[str, _Row] = {
    # ── elementwise ──────────────────────────────────────────────────────────
    "abs": _kernel("abs", lambda fw, t, m: fw.abs(t)),
    "abs-method": _kernel("abs", lambda fw, t, m: t.abs()),
    "abs-builtin": _kernel("abs", lambda fw, t, m: abs(t)),
    "abs-inplace": _kernel("abs", lambda fw, t, m: t.clone().abs_()),
    "absolute": _kernel("abs", lambda fw, t, m: fw.absolute(t)),
    "ceil": _kernel("ceil", lambda fw, t, m: fw.ceil(t)),
    "ceil-method": _kernel("ceil", lambda fw, t, m: t.ceil()),
    "ceil-inplace": _kernel("ceil", lambda fw, t, m: t.clone().ceil_()),
    "floor": _kernel("floor", lambda fw, t, m: fw.floor(t)),
    "floor-method": _kernel("floor", lambda fw, t, m: t.floor()),
    "floor-inplace": _kernel("floor", lambda fw, t, m: t.clone().floor_()),
    "round": _kernel("round", lambda fw, t, m: fw.round(t)),
    "round-method": _kernel("round", lambda fw, t, m: t.round()),
    "round-decimals": _kernel("round", lambda fw, t, m: fw.round(t, decimals=1)),
    "round-inplace": _kernel("round", lambda fw, t, m: t.clone().round_()),
    "trunc": _kernel("trunc", lambda fw, t, m: fw.trunc(t)),
    "trunc-method": _kernel("trunc", lambda fw, t, m: t.trunc()),
    "trunc-inplace": _kernel("trunc", lambda fw, t, m: t.clone().trunc_()),
    "fix": _kernel("trunc", lambda fw, t, m: fw.fix(t)),
    "fix-method": _kernel("trunc", lambda fw, t, m: t.fix()),
    "frac": _kernel("frac", lambda fw, t, m: fw.frac(t)),
    "frac-method": _kernel("frac", lambda fw, t, m: t.frac()),
    "frac-inplace": _kernel("frac", lambda fw, t, m: t.clone().frac_()),
    "frexp": _guard("frexp", lambda fw, t, m: fw.frexp(t)),
    "frexp-method": _guard("frexp", lambda fw, t, m: t.frexp()),
    "polar": _guard("polar", lambda fw, t, m: fw.polar(t, t)),
    "polar-first": _guard("polar", lambda fw, t, m: fw.polar(t, t.float())),
    "polar-second": _guard("polar", lambda fw, t, m: fw.polar(t.float(), t)),
    # ── activations ──────────────────────────────────────────────────────────
    "relu": _guard("relu", lambda fw, t, m: fw.relu(t)),
    "relu-method": _guard("relu", lambda fw, t, m: t.relu()),
    "relu-inplace": _guard("relu", lambda fw, t, m: t.clone().relu_()),
    "relu-functional": _guard("relu", lambda fw, t, m: fw.nn.functional.relu(t)),
    "relu-functional-inplace": _guard(
        "relu", lambda fw, t, m: fw.nn.functional.relu(t.clone(), inplace=True)
    ),
    "relu-module": _guard("relu", lambda fw, t, m: fw.nn.ReLU()(t)),
    "selu": _kernel("selu", lambda fw, t, m: fw.selu(t)),
    "selu-method": _kernel("selu", lambda fw, t, m: t.selu(), parity=False),
    "selu-functional": _kernel("selu", lambda fw, t, m: fw.nn.functional.selu(t)),
    "selu-module": _kernel("selu", lambda fw, t, m: fw.nn.SELU()(t)),
    "softmax": _kernel("softmax", lambda fw, t, m: fw.softmax(t, 0)),
    "softmax-dim": _kernel("softmax", lambda fw, t, m: fw.softmax(m, -1)),
    "softmax-method": _kernel("softmax", lambda fw, t, m: t.softmax(0)),
    "softmax-functional": _kernel("softmax", lambda fw, t, m: fw.nn.functional.softmax(t, 0)),
    "softmax-module": _kernel("softmax", lambda fw, t, m: fw.nn.Softmax(0)(t)),
    "log_softmax": _kernel("log_softmax", lambda fw, t, m: fw.log_softmax(t, 0)),
    "log_softmax-method": _kernel("log_softmax", lambda fw, t, m: t.log_softmax(0)),
    "log_softmax-functional": _kernel(
        "log_softmax", lambda fw, t, m: fw.nn.functional.log_softmax(t, 0)
    ),
    "log_softmax-module": _kernel("log_softmax", lambda fw, t, m: fw.nn.LogSoftmax(0)(t)),
    # ── moments ──────────────────────────────────────────────────────────────
    "mean": _guard("mean", lambda fw, t, m: fw.mean(t)),
    "mean-dim": _guard("mean", lambda fw, t, m: fw.mean(m, dim=1)),
    "mean-method": _guard("mean", lambda fw, t, m: t.mean()),
    "std": _guard("std", lambda fw, t, m: fw.std(t)),
    "std-dim": _guard("std", lambda fw, t, m: fw.std(m, dim=1)),
    "std-method": _guard("std", lambda fw, t, m: t.std()),
    "var": _guard("var", lambda fw, t, m: fw.var(t)),
    "var-correction": _guard("var", lambda fw, t, m: fw.var(t, correction=0)),
    "var-method": _guard("var", lambda fw, t, m: t.var()),
    "std_mean": _guard("std_mean", lambda fw, t, m: fw.std_mean(t)),
    "std_mean-dim": _guard("std_mean", lambda fw, t, m: fw.std_mean(m, dim=1)),
    "var_mean": _guard("var_mean", lambda fw, t, m: fw.var_mean(t)),
    "var_mean-dim": _guard("var_mean", lambda fw, t, m: fw.var_mean(m, dim=1)),
    "nanmean": _guard("nanmean", lambda fw, t, m: fw.nanmean(t)),
    "nanmean-method": _guard("nanmean", lambda fw, t, m: t.nanmean()),
    # ``nanmean`` has no spelling that takes a bool, ``mean`` does (below).
    "nanmean-dtype": _guard("nanmean", lambda fw, t, m: fw.nanmean(t, dtype=fw.float32)),
    "cov": _guard("cov", lambda fw, t, m: fw.cov(m)),
    "corrcoef": _guard("corrcoef", lambda fw, t, m: fw.corrcoef(m)),
    "gradient": _guard("gradient", lambda fw, t, m: fw.gradient(t)),
    "gradient-spacing": _guard("gradient", lambda fw, t, m: fw.gradient(t, spacing=2.0)),
    "gradient-dim": _guard("gradient", lambda fw, t, m: fw.gradient(m, dim=1)),
    "histogram": _kernel("histogram", lambda fw, t, m: fw.histogram(t, 2)),
    "histogram-range": _kernel("histogram", lambda fw, t, m: fw.histogram(t, 2, range=(0.0, 1.0))),
    "histogram-bins": _refused("histogram", lambda fw, t, m: fw.histogram(t.float(), t)),
    "histogram-weight": _refused(
        "histogram", lambda fw, t, m: fw.histogram(t.float(), 2, weight=t)
    ),
    "poisson": _kernel("poisson", lambda fw, t, m: fw.poisson(t)),
    # ── order statistics, search and membership ──────────────────────────────
    "quantile": _guard("quantile", lambda fw, t, m: fw.quantile(t, 0.5)),
    "quantile-dim": _guard("quantile", lambda fw, t, m: fw.quantile(m, 0.5, dim=1)),
    "quantile-method": _guard("quantile", lambda fw, t, m: t.quantile(0.5)),
    "nanquantile": _guard("nanquantile", lambda fw, t, m: fw.nanquantile(t, 0.5)),
    "nanquantile-method": _guard("nanquantile", lambda fw, t, m: t.nanquantile(0.5)),
    "isin": _guard("isin", lambda fw, t, m: fw.isin(t, t)),
    "isin-elements": _guard("isin", lambda fw, t, m: fw.isin(t, t.long())),
    "isin-test-elements": _guard("isin", lambda fw, t, m: fw.isin(t.long(), t)),
    # A sequence and its values meet at their common dtype: both bool is the
    # one pairing the reference has no kernel for.
    "searchsorted": _kernel("searchsorted", lambda fw, t, m: fw.searchsorted(t, t[:1])),
    "bucketize": _kernel("bucketize", lambda fw, t, m: fw.bucketize(t, t)),
    # ``clamp`` refuses a bool *number* bound on a bool tensor: a number bound
    # that is an int or a float promotes the answer, and a tensor bound is
    # answered by ``minimum`` / ``maximum``.
    "clamp": _kernel("clamp", lambda fw, t, m: fw.clamp(t, False, True)),
    "clamp-min": _kernel("clamp", lambda fw, t, m: fw.clamp(t, min=False)),
    "clamp-max": _kernel("clamp", lambda fw, t, m: fw.clamp(t, max=True)),
    "clamp-method": _kernel("clamp", lambda fw, t, m: t.clamp(False, True)),
    "clip": _kernel("clamp", lambda fw, t, m: fw.clip(t, False, True)),
    "clamp_min": _kernel("clamp", lambda fw, t, m: fw.clamp_min(t, True)),
    "clamp_max": _kernel("clamp", lambda fw, t, m: fw.clamp_max(t, True)),
    # ── what the reference answers, and so does Lucid ────────────────────────
    "mean-dtype": _accepted(lambda fw, t, m: fw.mean(t, dtype=fw.float32)),
    "mean-dim-dtype": _accepted(lambda fw, t, m: fw.mean(m, dim=1, dtype=fw.float32)),
    "softmax-dtype": _accepted(lambda fw, t, m: fw.softmax(t, 0, dtype=fw.float32)),
    "log_softmax-dtype": _accepted(lambda fw, t, m: fw.log_softmax(t, 0, dtype=fw.float32)),
    "softmax-functional-dtype": _accepted(
        lambda fw, t, m: fw.nn.functional.softmax(t, 0, dtype=fw.float32)
    ),
    "clamp-int-bounds": _accepted(lambda fw, t, m: fw.clamp(t, 0, 1)),
    "clamp-float-bound": _accepted(lambda fw, t, m: fw.clamp(t, min=0.5)),
    "clamp-tensor-bound": _accepted(lambda fw, t, m: fw.clamp(t, min=t)),
    "sum": _accepted(lambda fw, t, m: fw.sum(t)),
    "cov-float": _accepted(lambda fw, t, m: fw.cov(m.float())),
    "abs-int": _accepted(lambda fw, t, m: fw.abs(t.long())),
    "relu-int": _accepted(lambda fw, t, m: fw.relu(t.long())),
}

_VECTOR = [True, False, True, True]
_MATRIX = [[True, False, True], [False, False, True]]


def _flat(result: object) -> list[float]:
    """``result``'s values as one flat list of floats, whichever framework made it."""
    if isinstance(result, (tuple, list)):
        return [v for part in result for v in _flat(part)]
    tolist = getattr(result, "tolist", None)
    return _flat(tolist()) if callable(tolist) else [float(result)]  # type: ignore[arg-type]


def _refusals() -> dict[str, _Row]:
    return {name: row for name, row in _TABLE.items() if row.kind != "accepted"}


def _error_class_matches(kind: Kind, error: BaseException) -> bool:
    """Whether ``error`` is of the class ``kind`` names."""
    if not isinstance(error, RuntimeError):
        return False
    if kind == "kernel":
        return isinstance(error, NotImplementedError)
    if kind == "guard":
        return not isinstance(error, NotImplementedError)
    return True


@pytest.mark.parametrize("name", list(_refusals()))
def test_refusal_is_the_same_on_every_device(device: str, name: str) -> None:
    row = _TABLE[name]
    t = lucid.tensor(_VECTOR, device=device)
    m = lucid.tensor(_MATRIX, device=device)
    with pytest.raises(RuntimeError) as caught:
        row.call(lucid, t, m)
    assert _error_class_matches(row.kind, caught.value), (row.kind, type(caught.value))
    # The message names the op and the cast to make, not a kernel.
    assert "does not accept" in str(caught.value) and "cast to" in str(caught.value)


@pytest.mark.parametrize("name", [n for n, row in _TABLE.items() if row.kind == "accepted"])
def test_the_spelling_the_reference_offers_still_answers(device: str, name: str) -> None:
    t = lucid.tensor(_VECTOR, device=device)
    m = lucid.tensor(_MATRIX, device=device)
    assert _TABLE[name].call(lucid, t, m) is not None


@pytest.mark.parity
@pytest.mark.parametrize("name", [n for n, row in _TABLE.items() if row.parity])
def test_table_row_is_what_the_reference_does_on_cpu(ref: ModuleType, name: str) -> None:
    row = _TABLE[name]
    t = ref.tensor(_VECTOR)
    m = ref.tensor(_MATRIX)
    if row.kind == "accepted":
        row.call(ref, t, m)
        return
    with pytest.raises(RuntimeError) as caught:
        row.call(ref, t, m)
    assert _error_class_matches(row.kind, caught.value), (row.kind, type(caught.value))


@pytest.mark.parity
@pytest.mark.parametrize("name", [n for n, row in _TABLE.items() if row.kind == "accepted"])
def test_accepted_rows_return_the_reference_values(ref: ModuleType, device: str, name: str) -> None:
    call = _TABLE[name].call
    got = call(lucid, lucid.tensor(_VECTOR, device=device), lucid.tensor(_MATRIX, device=device))
    want = call(ref, ref.tensor(_VECTOR), ref.tensor(_MATRIX))
    assert _flat(got) == pytest.approx(_flat(want))
    if hasattr(got, "dtype"):
        assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]  # type: ignore[attr-defined]


def test_every_op_of_the_table_is_pinned() -> None:
    pinned = {row.op for row in _TABLE.values() if row.op is not None}
    assert pinned == set(BOOL_REFUSED), (
        f"missing a row: {sorted(set(BOOL_REFUSED) - pinned)}; "
        f"not in the table: {sorted(pinned - set(BOOL_REFUSED))}"
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda t: lucid.mean(t),
        lambda t: lucid.abs(t),
        lambda t: lucid.softmax(t, 0),
        lambda t: lucid.relu(t),
        lambda t: lucid.quantile(t, 0.5),
        lambda t: lucid.clamp(t, False, True),
    ],
    ids=["mean", "abs", "softmax", "relu", "quantile", "clamp"],
)
def test_a_compiled_bool_op_refuses_as_eager_does(
    device: str, build: Callable[[lucid.Tensor], lucid.Tensor]
) -> None:
    # The trace runs the eager entry, so the refusal fires while the function
    # is traced, before any graph holds the op.
    x = (lucid.randint(0, 2, (4, 6)) == 1).to(device)
    with pytest.raises(RuntimeError, match="does not accept"):
        lucid.compile(build)(x)


def test_a_non_tensor_still_gets_the_engines_own_error() -> None:
    # The guard looks only at tensors: a number or a list falls through to the
    # engine op, whose ``TypeError`` is the right answer for them.
    for bad in (3.0, [1.0], None):
        with pytest.raises(TypeError):
            lucid.abs(bad)  # type: ignore[arg-type]
