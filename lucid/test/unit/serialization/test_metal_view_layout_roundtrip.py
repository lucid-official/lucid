"""A loaded Metal view reads the storage where the live one did (LCD-513).

``lucid.save`` places each Metal view in the storage it shares with its base
by the view's ``stride()`` and ``storage_offset()``, which the engine's view
family reports for a Metal view as for a CPU one.  It used to replay the
view's links on a CPU probe instead, by the CPU's rules, which restart the
storage at a leading slice (LCD-517): ``y[1:].as_strided((2,), (1,), 6)``
reads ``y[1, :2]`` live and came back reading ``y[2, :2]``.  And a view whose
base is gone was placed in the freed base's storage, which the saver then
wrote out whole, zeros and all, for the few elements anyone could read.
"""

import io
from collections.abc import Callable

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

type _View = Callable[[lucid.Tensor], lucid.Tensor]

#: Views of a live (4, 6) base, the windows among them in the storage's
#: elements, as the reference counts them.
VIEWS: dict[str, _View] = {
    "row": lambda y: y[1],
    "rows": lambda y: y[1:],
    "row-tail": lambda y: y[2, 3:],
    "column": lambda y: y[:, 2],
    "transpose-rows": lambda y: y.t()[1:4],
    "window-past-a-slice": lambda y: y[1:].as_strided((2,), (1,), 6),
    "window-of-a-row": lambda y: y[2].as_strided((2, 2), (1, 2)),
    "window-of-a-transpose": lambda y: y.t().as_strided((3,), (7,), 1),
    "unbind-row": lambda y: y.unbind(0)[3],
    "split-rows": lambda y: y.split(2)[1],
    "diagonal": lambda y: y.diagonal(1),
}


def _round_trip(obj: object) -> object:
    buffer = io.BytesIO()
    lucid.save(obj, buffer)
    buffer.seek(0)
    return lucid.load(buffer)


@pytest.mark.parametrize("make", list(VIEWS.values()), ids=list(VIEWS))
def test_a_loaded_view_writes_where_the_live_one_does(make: _View) -> None:
    y = lucid.arange(24.0, device="metal").reshape(4, 6)
    view = make(y)
    loaded = _round_trip({"y": y, "view": view})
    assert isinstance(loaded, dict)
    ly, lview = loaded["y"], loaded["view"]
    assert lview.tolist() == view.tolist()
    assert lview.storage_offset() == view.storage_offset()
    view.add_(100.0)
    lview.add_(100.0)
    assert ly.tolist() == y.tolist()


def test_a_view_whose_base_is_gone_saves_only_what_it_reads() -> None:
    # Nothing holds the product: its 10 000 elements are no storage anyone
    # can read, so a column and a view of it save the column's 100.
    column = (lucid.arange(10_000.0, device="metal").reshape(100, 100) * 1.0)[:, 3]
    tail = column[1:]
    buffer = io.BytesIO()
    lucid.save({"column": column, "tail": tail}, buffer)
    assert buffer.tell() < 4 * 10_000 // 4
    buffer.seek(0)
    loaded = lucid.load(buffer)
    assert isinstance(loaded, dict)
    assert loaded["column"].tolist() == column.tolist()
    assert loaded["tail"].tolist() == tail.tolist()
    loaded["tail"].add_(1.0)
    assert loaded["column"].tolist()[1] == column.tolist()[1] + 1.0
