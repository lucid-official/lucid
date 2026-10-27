"""The dtypes eager refuses, held to :data:`._op_matrix.EAGER_REJECTS`.

The replay matrix and the numpy oracle run a case only on the declared dtypes
eager answers; the others are listed with the exception eager raises on each
device.  The list is held both ways here.  A listed pair must still raise
exactly that exception type, not a subclass or a parent: one that answers now,
or raises something else, is a stale entry.  And no declared pair outside the
list may raise, on any device — including the pairs neither matrix runs on
the CPU.
"""

import pytest

import lucid
import lucid.test.unit.compile._op_matrix as M

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


DEVICES = ["cpu"] + ([COMPILE_DEVICE] if _metal_ok() else [])


def _eager_error(name: str, dtype: str, device: str) -> Exception | None:
    """Run the case eagerly on the input the matrices use; what it raised."""
    case = M.CASE_BY_NAME[name]
    x = M.make_input(case.kind, dtype, case.shape, 2)
    with M.on_device(device):
        try:
            case.fn(x.to(device))
        except Exception as e:  # noqa: BLE001 — the exception is the answer
            return e
    return None


def _listed() -> list[tuple[str, str, str]]:
    return [
        (name, dtype, dev)
        for name, dtype in sorted(M.EAGER_REJECTS)
        for dev in DEVICES
        if M.refusal(name, dtype, dev) is not None
    ]


def test_every_entry_is_a_declared_non_float_dtype() -> None:
    bad = []
    for (name, dtype), r in M.EAGER_REJECTS.items():
        case = M.CASE_BY_NAME.get(name)
        if case is None:
            bad.append(f"{name}: no such case")
        elif dtype not in case.dtypes:
            bad.append(f"{name} {dtype}: not a declared dtype of the case")
        elif dtype == "f32":
            bad.append(f"{name} f32: a float32 refusal is a broken recipe")
        elif r.cpu is None and r.metal is None:
            bad.append(f"{name} {dtype}: refused on no device — drop the entry")
    assert not bad, bad


@pytest.mark.parametrize(("name", "dtype", "device"), _listed(), ids=str)
def test_listed_refusal_still_raises(name: str, dtype: str, device: str) -> None:
    want = M.refusal(name, dtype, device)
    assert want is not None
    err = _eager_error(name, dtype, device)
    if err is None:
        pytest.fail(
            f"eager answers {name} on {dtype} on {device} now; drop the "
            "entry from EAGER_REJECTS so the matrices run it"
        )
    assert type(err) is want, (
        f"raises {type(err).__name__}: {str(err)[:160]} — EAGER_REJECTS says " f"{want.__name__}"
    )


def test_no_unlisted_refusal() -> None:
    """Every declared pair that is not listed answers, on every device."""
    unlisted = [
        f"{c.name} {d} {dev}: {type(err).__name__}: {str(err)[:120]}"
        for c in M.CASES
        for d in c.dtypes
        for dev in DEVICES
        if M.refusal(c.name, d, dev) is None and (err := _eager_error(c.name, d, dev)) is not None
    ]
    assert not unlisted, "eager refuses, not in EAGER_REJECTS:\n" + "\n".join(unlisted)
