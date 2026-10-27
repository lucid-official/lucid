"""A Metal view reads, writes and reports its base's storage as the CPU does.

Defects of one owner, the Python layer that stood in for the storage a Metal
tensor did not share with its views.  The engine's view family has since
replaced it: a Metal view is a member of its base's family, as a CPU view is.

* **Which keys are views (LCD-512).**  Every basic key was linked back to its
  base, a negative step included, so ``x[::-1].add_(10)`` wrote ``x`` on
  Metal while the CPU, which gathers a reversed slice into a copy, left it
  alone; and ``x[::-1].view(-1)`` was refused.  A NumPy-style integer
  (anything with ``__index__``) was not linked at all, though the CPU reads
  it in place.  A key is now linked exactly when the CPU reads it in place,
  and one that goes on past a reversed slice views the copy, as on the CPU.

* **Where a view sits in the storage (LCD-513).**  ``as_strided`` read the
  view's own packed elements, so ``y.t().as_strided((2,), (1,))`` gave
  ``[0, 3]`` on Metal and ``[0, 1]`` on the CPU and in the reference — a
  silent wrong answer — and ``stride()``, ``storage_offset()`` and
  ``is_contiguous()`` described the packed copy.  They now describe the
  view's place in its base's storage, as the reference does, and the window
  is read from there.

* **What writes reach.**  A write through a window read past its view left
  the view stale, to write its old values back later; it is refused while
  the view lives, where the CPU and the reference write it (LCD-535).  A
  reshape the CPU copies shares its input's buffer on Metal, so writes cross
  between them (LCD-535, which needs the engine to detach a family member).
  A write into a view of an overlapping view was refused on Metal only.

The reference is the contract for the layout.  Where the CPU disagrees with
it — a leading slice restarts the CPU's storage (LCD-517) — the CPU cell is
a strict xfail.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

type _View = Callable[[Any], Any]


def _grid(module: Any, device: str) -> Any:
    return module.arange(24.0, device=device).reshape(4, 6)


# ── Which keys are views (LCD-512) ───────────────────────────────────────────


class _Index:
    """An integer of another type, as a NumPy integer is."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


KEYS: dict[str, object] = {
    "reversed": slice(None, None, -1),
    "reversed-stepped": slice(None, None, -2),
    "reversed-range": slice(3, 0, -1),
    "reversed-empty": slice(1, 3, -1),
    "inner-reversed": (slice(None), slice(None, None, -1)),
    "reversed-then-int": (slice(None, None, -1), 2),
    "reversed-then-slice": (slice(None, None, -1), slice(1, 4)),
    "int-then-reversed": (1, slice(None, None, -1)),
    "ellipsis-reversed": (Ellipsis, slice(None, None, -1)),
    "reversed-ellipsis-int": (slice(None, None, -1), Ellipsis, 0),
    "none-reversed": (None, slice(None, None, -1)),
    "reversed-then-none": (slice(None, None, -1), None, 1),
    "two-reversed": (slice(None, None, -1), slice(4, None, -2)),
    "index-scalar": _Index(1),
    "index-scalar-bounds": (slice(None), slice(_Index(1), _Index(5))),
    "index-scalar-step": (slice(None), slice(None, None, _Index(2))),
    "index-scalar-reversed": (slice(None), slice(None, None, _Index(-1))),
    "int": 1,
    "stepped": (slice(None), slice(1, None, 2)),
    "none": (None, 1),
    "bool": True,
    "list": [2, 0],
}


def _written(device: str, key: object, follow: _View) -> list[object]:
    x = _grid(lucid, device)
    view = follow(x[key])
    if view.numel():
        view.add_(100.0)
    return x.tolist()


FOLLOWS: dict[str, _View] = {
    "itself": lambda v: v,
    "an element of it": lambda v: v[(0,) * v.ndim] if v.numel() else v,
    "a reshape of it": lambda v: v.reshape(-1),
    "its transpose": lambda v: v.mT if v.ndim >= 2 else v,
    "a slice of it": lambda v: v[1:] if v.ndim else v,
}


@pytest.mark.parametrize("follow", list(FOLLOWS), ids=list(FOLLOWS))
@pytest.mark.parametrize("key", list(KEYS), ids=list(KEYS))
def test_a_write_through_a_key_reaches_the_base_as_on_the_cpu(key: str, follow: str) -> None:
    want = _written("cpu", KEYS[key], FOLLOWS[follow])
    assert _written("metal", KEYS[key], FOLLOWS[follow]) == want


def _viewed(device: str, key: object) -> object:
    try:
        return _grid(lucid, device)[key].view(-1).tolist()
    except RuntimeError:
        return "refused"


@pytest.mark.parametrize("key", list(KEYS), ids=list(KEYS))
def test_view_of_a_key_is_refused_only_where_the_cpu_refuses_it(key: str) -> None:
    assert _viewed("metal", KEYS[key]) == _viewed("cpu", KEYS[key])


@pytest.mark.parametrize("key", ["reversed-then-int", "reversed-then-slice", "reversed-then-none"])
def test_a_key_past_a_reversed_slice_views_the_copy_as_on_the_cpu(key: str) -> None:
    # The CPU gathers what the key has read so far and views that copy with
    # the rest of the key: the view's layout, and a window of it, are the
    # copy's — and the copy stays alive with the view.
    def seen(device: str) -> tuple[object, ...]:
        v = _grid(lucid, device)[KEYS[key]]
        window = v.as_strided((3,), (1,), 0).tolist()
        return v.stride(), v.storage_offset(), v.is_contiguous(), window

    assert seen("metal") == seen("cpu")


def test_a_reversed_slice_takes_writes_and_assignments_alone() -> None:
    x = lucid.arange(4.0, device="metal")
    reversed_ = x[::-1]
    reversed_.mul_(0.0)
    reversed_[0] = 7.0
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert reversed_.tolist() == [7.0, 0.0, 0.0, 0.0]


# ── Where a view sits in the storage (LCD-513) ───────────────────────────────

#: Views of ``_grid`` — shape (4, 6) — and whether the CPU's storage restarts.
LAYOUTS: dict[str, tuple[_View, bool]] = {
    "row": (lambda t: t[1], True),
    "rows": (lambda t: t[1:], True),
    "stepped-rows": (lambda t: t[1::2], True),
    "row-tail": (lambda t: t[1, 2:], True),
    "row-of-rows": (lambda t: t[1:][1], True),
    "column": (lambda t: t[:, 1], False),
    "columns": (lambda t: t[:, 2:5], False),
    "stepped-columns": (lambda t: t[:, 1::2], False),
    "narrow-rows": (lambda t: t.narrow(0, 1, 2), True),
    "narrow-columns": (lambda t: t.narrow(1, 2, 3), False),
    "unbind-rows": (lambda t: t.unbind(0)[2], True),
    "unbind-columns": (lambda t: t.unbind(1)[3], False),
    "split-rows": (lambda t: t.split(3)[1], True),
    "split-sections": (lambda t: t.split([1, 3, 2], 1)[2], False),
    "chunk-rows": (lambda t: t.chunk(3)[1], True),
    "chunk-columns": (lambda t: t.chunk(4, 1)[2], False),
    "transpose": (lambda t: t.t(), False),
    "transpose-row": (lambda t: t.t()[1], False),
    "transposed-columns": (lambda t: t.t()[2:], False),
    "diagonal-above": (lambda t: t.diagonal(2), False),
    "diagonal-below": (lambda t: t.diagonal(-1), False),
    "unfold": (lambda t: t.unfold(1, 3, 2), False),
    "expand": (lambda t: t[:, :1].expand(4, 3), False),
    "new-axis": (lambda t: t[None, :, 2], False),
    "permute": (lambda t: t.reshape(2, 2, 6).permute(2, 0, 1)[1:], False),
    "detach": (lambda t: t.t().detach()[1], False),
}


def _layout_cases() -> list[Any]:
    cases = []
    for name in LAYOUTS:
        for device in ("cpu", "metal"):
            cases.append(pytest.param(name, device, id=f"{name}-{device}"))
    return cases


@pytest.mark.parity
@pytest.mark.parametrize(("name", "device"), _layout_cases())
def test_a_view_reports_the_references_layout(name: str, device: str, ref: Any) -> None:
    make = LAYOUTS[name][0]
    ours, theirs = make(_grid(lucid, device)), make(_grid(ref, "cpu"))

    def moving(t: Any) -> tuple[int, ...]:
        return tuple(s for n, s in zip(t.shape, t.stride(), strict=True) if n != 1)

    assert ours.tolist() == theirs.tolist()
    assert moving(ours) == moving(theirs)
    assert ours.storage_offset() == theirs.storage_offset()
    assert ours.is_contiguous() == theirs.is_contiguous()


class _Window(NamedTuple):
    view: _View
    args: tuple[object, ...]
    restarts: bool  # the CPU's storage restarts at the view (LCD-517)
    outside: bool  # the window reads an element the view does not hold


#: Windows onto views of ``_grid``: the view, then ``as_strided``'s arguments.
WINDOWS: dict[str, _Window] = {
    "transpose": _Window(lambda t: t.t(), ((2,), (1,)), False, False),
    "transpose-offset": _Window(lambda t: t.t(), ((3, 2), (2, 7), 1), False, False),
    "column": _Window(lambda t: t[:, 1], ((2,), (1,)), False, True),
    "column-whole": _Window(lambda t: t[:, 1], ((4,), (6,)), False, False),
    "row-from-the-start": _Window(lambda t: t[1], ((2,), (1,), 0), True, True),
    "row-reaches-all": _Window(lambda t: t[0], ((24,), (1,)), True, True),
    "row-own": _Window(lambda t: t[2], ((3,), (2,)), False, False),
    "diagonal": _Window(lambda t: t.diagonal(), ((4,), (7,)), False, False),
    "diagonal-past-it": _Window(lambda t: t.diagonal(1), ((3,), (5,)), False, True),
    "unfold": _Window(lambda t: t.unfold(1, 2, 2), ((3,), (2,)), False, False),
    "expand": _Window(lambda t: t[:1].expand(3, 6), ((2, 2), (6, 1)), True, True),
    "permute": _Window(
        lambda t: t.reshape(2, 3, 4).permute(2, 0, 1), ((2, 3), (4, 1)), False, False
    ),
    "detach": _Window(lambda t: t.t().detach(), ((2,), (1,)), False, False),
    "window-of-a-window": _Window(
        lambda t: t.t().as_strided((3, 4), (6, 1), 1), ((2,), (2,)), False, False
    ),
    "stepped": _Window(lambda t: t[:, ::2], ((2, 3), (6, 2)), False, False),
}


def _window_cases() -> list[Any]:
    cases = []
    for name in WINDOWS:
        for device in ("cpu", "metal"):
            cases.append(pytest.param(name, device, id=f"{name}-{device}"))
    return cases


def _reads_outside(make: _View, args: tuple[object, ...], ref: Any) -> bool:
    """Whether the window reads an element of the storage the view does not hold."""
    index = ref.arange(24).reshape(4, 6)
    view = make(index)
    held = set(view.reshape(-1).tolist())
    return not set(view.as_strided(*args).reshape(-1).tolist()) <= held


@pytest.mark.parity
@pytest.mark.parametrize("name", list(WINDOWS), ids=list(WINDOWS))
def test_the_table_says_which_windows_read_outside_their_view(name: str, ref: Any) -> None:
    window = WINDOWS[name]
    assert _reads_outside(window.view, window.args, ref) == window.outside


@pytest.mark.parity
@pytest.mark.parametrize(("name", "device"), _window_cases())
def test_a_window_reads_and_writes_the_references_elements(
    name: str, device: str, ref: Any
) -> None:
    window = WINDOWS[name]
    ours, theirs = _grid(lucid, device), _grid(ref, "cpu")
    view, their_view = window.view(ours), window.view(theirs)
    mine, want = view.as_strided(*window.args), their_view.as_strided(*window.args)
    assert mine.tolist() == want.tolist()
    mine.add_(100.0)
    want.add_(100.0)
    assert view.tolist() == their_view.tolist()
    assert ours.tolist() == theirs.tolist()


@pytest.mark.parametrize("name", [n for n, w in WINDOWS.items() if w.outside])
def test_a_write_into_a_window_past_a_live_view_reaches_the_storage(name: str) -> None:
    window = WINDOWS[name]
    ours_m, ours_c = _grid(lucid, "metal"), _grid(lucid, "cpu")
    view_m, view_c = window.view(ours_m), window.view(ours_c)
    view_m.as_strided(*window.args).add_(100.0)
    view_c.as_strided(*window.args).add_(100.0)
    assert view_m.tolist() == view_c.tolist()
    assert ours_m.tolist() == ours_c.tolist()


def test_a_window_past_a_view_nothing_holds_writes_its_base() -> None:
    y = _grid(lucid, "metal")
    y[:, 1].as_strided((2,), (1,), 1).fill_(-7.0)  # the column is gone at once
    assert y[0].tolist() == [0.0, -7.0, -7.0, 3.0, 4.0, 5.0]


def test_the_free_function_reads_the_window_the_method_does() -> None:
    y = _grid(lucid, "metal")
    assert lucid.as_strided(y.t(), (2,), (1,)).tolist() == [0.0, 1.0]
    assert lucid.as_strided(y.t(), size=(2,), stride=(6,), storage_offset=1).tolist() == [1.0, 7.0]


def test_a_write_through_a_window_reaches_the_view_it_was_taken_from() -> None:
    y = _grid(lucid, "metal")
    column = y[:, 1]
    column.as_strided((2,), (6,)).add_(100.0)
    assert column.tolist() == [101.0, 107.0, 13.0, 19.0]
    assert y[:, 1].tolist() == [101.0, 107.0, 13.0, 19.0]


# ── The idioms built from a view's own strides ───────────────────────────────

SOURCES: dict[str, _View] = {
    "transposed": lambda t: t.t(),
    "permuted": lambda t: t.reshape(2, 3, 4).permute(1, 2, 0),
    "sliced": lambda t: t[1:, ::2],
    "narrowed": lambda t: t.narrow(1, 1, 4),
    "row": lambda t: t[2],
    "diagonal": lambda t: t.diagonal(),
    "expanded": lambda t: t[:1].expand(2, 6),
}


@pytest.mark.parametrize("source", list(SOURCES), ids=list(SOURCES))
def test_a_window_of_its_own_strides_is_the_view(source: str) -> None:
    for device in ("cpu", "metal"):
        v = SOURCES[source](_grid(lucid, device))
        assert v.as_strided(v.shape, v.stride()).tolist() == v.tolist(), device


def _sliding(v: Any, width: int) -> Any:
    flat = v.reshape(-1) if v.ndim != 1 else v
    n = flat.shape[0]
    return flat.as_strided((n - width + 1, width), (flat.stride(0), flat.stride(0)))


ONE_D: dict[str, _View] = {
    "column": lambda t: t[:, 2],
    "transposed-row": lambda t: t.t()[3],
    "stepped": lambda t: t[1, ::2],
    "narrowed": lambda t: t.narrow(1, 1, 4)[2],
    "diagonal": lambda t: t.diagonal(1),
    "permuted": lambda t: t.reshape(2, 3, 4).permute(2, 1, 0)[1, 2],
}


@pytest.mark.parity
@pytest.mark.parametrize("source", list(ONE_D), ids=list(ONE_D))
def test_sliding_windows_from_a_views_stride_match_the_reference(source: str, ref: Any) -> None:
    want = _sliding(ONE_D[source](_grid(ref, "cpu")), 2).tolist()
    for device in ("cpu", "metal"):
        assert _sliding(ONE_D[source](_grid(lucid, device)), 2).tolist() == want, device


@pytest.mark.parity
def test_a_sliding_window_over_rows_of_a_transposed_view(ref: Any) -> None:
    def windows(t: Any) -> Any:
        v = t.t()  # (6, 4), strides (1, 6)
        return v.as_strided((5, 2, 4), (v.stride(0), v.stride(0), v.stride(1)))

    want = windows(_grid(ref, "cpu")).tolist()
    for device in ("cpu", "metal"):
        assert windows(_grid(lucid, device)).tolist() == want, device


# ── Gradients, and storage that is gone ──────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize(
    "make",
    [lambda t: t.t(), lambda t: t[:, ::2], lambda t: t[1:3]],
    ids=["transposed", "stepped", "rows"],
)
def test_a_gradient_through_a_window_of_a_view_is_the_references(make: _View, ref: Any) -> None:
    def grad(module: Any, device: str) -> list[object]:
        w = module.arange(24.0, device=device).reshape(4, 6).requires_grad_()
        v = make(w)
        size = (2, 2)
        stride = (v.stride(0), v.stride(1))
        (v.as_strided(size, stride) * module.tensor([1.0, 2.0], device=device)).sum().backward()
        assert w.grad is not None
        return w.grad.tolist()

    assert grad(lucid, "metal") == grad(ref, "cpu")


def test_a_gradient_through_elements_outside_the_view_is_refused() -> None:
    w = lucid.arange(24.0, device="metal").reshape(4, 6).requires_grad_()
    with pytest.raises(NotImplementedError, match="every element of the window"):
        w[:, 1].as_strided((2,), (1,))
    with lucid.no_grad():
        assert w[:, 1].as_strided((2,), (1,)).tolist() == [1.0, 2.0]


def test_a_window_of_a_view_whose_base_is_gone() -> None:
    # The view's engine family root holds the whole allocation even after the
    # temporary ``arange`` is gone, matching CPU and the reference.
    tail = lucid.arange(10.0, device="metal")[2:]
    assert tail.storage_offset() == 2
    assert tail.as_strided((3,), (2,)).tolist() == [2.0, 4.0, 6.0]
    every_other = lucid.arange(10.0, device="metal")[::2]
    assert _sliding(every_other, 2).tolist() == [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]
    assert tail.as_strided((3,), (1,), 0).tolist() == [0.0, 1.0, 2.0]


def test_a_window_past_the_storage_is_refused_as_the_engine_refuses_it() -> None:
    y = _grid(lucid, "metal")
    with pytest.raises(RuntimeError, match="past the end of the storage"):
        y.t().as_strided((5,), (6,))
    with pytest.raises(ValueError, match="non-negative"):
        y.t().as_strided((2,), (-1,))
    with pytest.raises(RuntimeError, match="same length"):
        y.t().as_strided((2,), (1, 1))


def test_a_view_of_a_reshape_of_a_temporary_holds_the_whole_storage() -> None:
    # The card's case: ``y`` holds every element of the freed ``arange``.
    y = lucid.arange(6.0, device="metal").reshape(2, 3)
    assert y.t().as_strided((2,), (1,)).tolist() == [0.0, 1.0]
    assert y._impl.family_id() != 0


# ── A reshape the CPU copies is a tensor of its own ──────────────────────────

#: Reshapes of the transpose ``u`` of a (2, 3) base, which no strides read in
#: place: the CPU copies, and the engine on Metal shares ``u``'s buffer.
COPYING: dict[str, _View] = {
    "reshape": lambda u: u.reshape(-1),
    "flatten": lambda u: u.flatten(),
    "ravel": lambda u: u.ravel(),
    "reshape_as": lambda u: u.reshape_as(u.flatten()),
    "unflatten": lambda u: u.reshape(-1).unflatten(0, (2, 3)),
}

THEN: dict[str, Callable[[Any, Any], object]] = {
    "write the copy": lambda u, f: f.add_(100.0),
    "write the view": lambda u, f: u.add_(100.0),
    "write a view of the copy": lambda u, f: f.view(-1)[1:3].add_(100.0),
    "write a view of the view": lambda u, f: u[1].add_(100.0),
    "assign into the copy": lambda u, f: f.__setitem__(0, -1.0),
}


@pytest.mark.parametrize("then", list(THEN), ids=list(THEN))
@pytest.mark.parametrize("op", list(COPYING), ids=list(COPYING))
def test_a_copying_reshape_of_a_view_is_a_tensor_of_its_own(op: str, then: str) -> None:
    def run(device: str) -> tuple[object, ...]:
        x = lucid.arange(6.0, device=device).reshape(2, 3)
        u = x.t()
        f = COPYING[op](u)
        THEN[then](u, f)
        u.mul_(1.0)  # ``u`` writes its values back to ``x``
        return x.tolist(), u.tolist(), f.tolist()

    assert run("metal") == run("cpu")


def test_a_copying_reshape_runs_one_op_and_view_refuses_it() -> None:
    # The merge-heads reshape of every attention layer copies nothing.
    u = lucid.arange(6.0, device="metal").reshape(2, 3).t()
    with lucid.profiler.profile() as prof:
        u.reshape(-1)
    assert [event.name for event in prof.events()] == ["reshape"]
    with pytest.raises(RuntimeError, match="use .reshape"):
        u.view(-1)


# What autograd knows of a copying reshape of a Metal view is the CPU's: a
# leaf of its own, its gradient, a retained gradient, whatever is read or
# written around it.


def _base(device: str, requires_grad: bool = False) -> lucid.Tensor:
    x = lucid.arange(6.0, device=device).reshape(2, 3)
    return x.requires_grad_() if requires_grad else x


def _leaf_after(around: Callable[[lucid.Tensor, str], object]) -> Callable[[str], object]:
    def run(device: str) -> object:
        r = _base(device).t().reshape(-1).requires_grad_()
        loss = (r * 3.0).sum()
        around(r, device)
        loss.backward()
        assert r.grad is not None
        return r.is_leaf, r.requires_grad, r.grad.tolist()

    return run


def _retained(device: str) -> object:
    x = _base(device, requires_grad=True)
    r = x.t().reshape(-1)
    r.retain_grad()
    loss = (r * r).sum()
    _ = r[0]
    lucid.zeros(4, 4, device=device).view(-1).add_(1.0)
    loss.backward()
    assert r.grad is not None and x.grad is not None
    return r.grad.tolist(), x.grad.tolist()


def _stepped(device: str) -> object:
    r = _base(device).t().reshape(-1).requires_grad_()
    optimizer = lucid.optim.SGD([r], lr=1.0)
    (r * 2.0).sum().backward()
    optimizer.step()
    assert r.grad is not None
    return r.is_leaf, r.tolist(), r.grad.tolist()


def _parameter(device: str) -> object:
    p = lucid.nn.Parameter(_base(device).t().reshape(-1))
    (p * 2.0).sum().backward()
    assert p.grad is not None
    return p.is_leaf, p.requires_grad, p.grad.tolist()


AUTOGRAD: dict[str, Callable[[str], object]] = {
    "leaf": _leaf_after(lambda r, d: None),
    "leaf after a view of it": _leaf_after(lambda r, d: r[1:]),
    "leaf after its .data": _leaf_after(lambda r, d: r.data),
    "leaf after its .grad is read": _leaf_after(lambda r, d: r.grad),
    "leaf after a write elsewhere": _leaf_after(
        lambda r, d: lucid.zeros(4, 4, device=d)[0].add_(1.0)
    ),
    "retained gradient": _retained,
    "optimizer step": _stepped,
    "Parameter of it": _parameter,
}


@pytest.mark.parametrize("case", list(AUTOGRAD), ids=list(AUTOGRAD))
def test_a_copying_reshape_keeps_its_place_in_autograd(case: str) -> None:
    assert AUTOGRAD[case]("metal") == AUTOGRAD[case]("cpu")


# ── A write into a view of a view whose elements overlap ─────────────────────

#: An overlapping view of ``_grid`` and a view of it whose own elements do not
#: overlap, which the reference writes through.
OVERLAPPED: dict[str, tuple[_View, _View]] = {
    "unfold-window": (lambda y: y[0].unfold(0, 3, 1), lambda u: u[1]),
    "unfold-row-windows": (lambda y: y.unfold(1, 2, 1), lambda u: u[1, 2]),
    "expand-row": (lambda y: y[:1].expand(3, 6), lambda e: e[1]),
    "expand-window": (lambda y: y[:1].expand(3, 6), lambda e: e.as_strided((2,), (1,))),
    "expand-part-of-a-row": (lambda y: y[1:2].expand(3, 6), lambda e: e[2, 1:4]),
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(OVERLAPPED), ids=list(OVERLAPPED))
def test_a_write_into_a_view_of_an_overlapping_view_is_the_references(name: str, ref: Any) -> None:
    outer, inner = OVERLAPPED[name]

    def run(module: Any, device: str) -> tuple[object, ...]:
        y = _grid(module, device)
        overlapping = outer(y)
        inner(overlapping).add_(100.0)
        return y.tolist(), overlapping.tolist()

    want = run(ref, "cpu")
    for device in ("cpu", "metal"):
        assert run(lucid, device) == want, device


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_a_write_into_overlapping_elements_is_still_refused(device: str) -> None:
    y = _grid(lucid, device)
    with pytest.raises(RuntimeError, match="overlap"):
        y[:1].expand(3, 6).add_(1.0)
    y[:1].expand(3, 6).fill_(-1.0)  # one value for all of them is no ambiguity
    assert y[0].tolist() == [-1.0] * 6


# ── as_strided's arguments, checked as the reference checks them ─────────────

BAD_WINDOWS: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {
    "float size": (((2.0,), (1,)), {}),
    "float stride": (((2,), (1.0,)), {}),
    "float offset": (((2,), (1,), 1.0), {}),
    "bool size": (((True,), (1,)), {}),
    "bool stride": (((2,), (True,)), {}),
    "bool offset": (((2,), (1,), True), {}),
    "no stride": (((2,),), {}),
    "fourth argument": (((2,), (1,), 0, 1), {}),
    "unknown keyword": (((2,), (1,)), {"offset": 1}),
    "size twice": (((2,), (1,)), {"size": (2,)}),
}


@pytest.mark.parametrize("args", list(BAD_WINDOWS), ids=list(BAD_WINDOWS))
@pytest.mark.parametrize(
    "make",
    [lambda d: lucid.arange(6.0, device=d), lambda d: _grid(lucid, d).t()],
    ids=["tensor", "view"],
)
@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_a_bad_window_is_refused_before_anything_is_read(
    device: str, make: Callable[[str], lucid.Tensor], args: str
) -> None:
    positional, named = BAD_WINDOWS[args]
    with pytest.raises(TypeError):
        make(device).as_strided(*positional, **named)


def test_a_slice_step_that_is_no_integer_is_refused_in_the_cpus_words() -> None:
    def message(device: str) -> str:
        with pytest.raises(TypeError) as caught:
            lucid.arange(6.0, device=device)[::1.5]
        return str(caught.value)

    assert message("metal") == message("cpu")
