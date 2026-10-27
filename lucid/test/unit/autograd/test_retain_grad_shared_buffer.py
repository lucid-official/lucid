"""``retain_grad`` on a tensor read more than once (CHA-111).

Eager backward stored the first gradient reaching a retaining tensor as
the very buffer it also kept pending for that tensor's node, and added
every later arrival into both — in place.  On the CPU the two adds hit
one buffer, so ``y.retain_grad()`` on a ``y`` read twice gave
``y.grad`` and everything behind it (``x.grad``) the second gradient
twice: ``[4, 4, 4]`` where the reference gives ``[3, 3, 3]``.  Metal adds
into a new array, so it was right by accident.

Graph-mode backward (``create_graph=True``) had no retain step at all and
left ``y.grad`` at ``None``; it now gets the gradient with its graph, as a
leaf does.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.test._helpers.compare import assert_close


def _compare(got: dict[str, object], want: dict[str, object]) -> None:
    assert got.keys() == want.keys()
    for key, w in want.items():
        g = got[key]
        if isinstance(w, bool) or w is None:
            assert g == w, key
        else:
            assert_close(g, w, msg=key)


def _x(lib: ModuleType, device: str, dtype: str) -> object:
    return lib.tensor(
        [1.0, -2.0, 0.5], dtype=getattr(lib, dtype), device=device, requires_grad=True
    )


# ── scenarios: run on Lucid and on the reference ──────────────────────────


def _read_twice(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    x = _x(lib, device, dtype)
    y = x * 1.5
    y.retain_grad()
    (y.sum() + (2 * y).sum()).backward()
    return {"y": y.grad, "x": x.grad}


def _read_three_times(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    x = _x(lib, device, dtype)
    y = x * x
    y.retain_grad()
    (y.sum() + (2 * y).sum() + (y * y).sum()).backward()
    return {"y": y.grad, "x": x.grad}


def _two_passes(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    x = _x(lib, device, dtype)
    y = x * 1.5
    y.retain_grad()
    loss = y.sum() + (2 * y).sum()
    loss.backward(retain_graph=True)
    loss.backward()
    return {"y": y.grad, "x": x.grad}


def _create_graph(lib: ModuleType, device: str, dtype: str) -> dict[str, object]:
    x = _x(lib, device, dtype)
    y = x * 1.5
    y.retain_grad()
    ((y * y).sum() + (2 * y).sum()).backward(create_graph=True)
    assert y.grad is not None
    y_grad = y.grad.detach().clone()
    carries_graph = y.grad.grad_fn is not None
    # Read y.grad first: the reference also adds into a retained gradient
    # during autograd.grad, which Lucid's grad() leaves alone.
    (h,) = lib.autograd.grad(y.grad.sum(), x, retain_graph=True)
    return {"y": y_grad, "x": x.grad, "carries_graph": carries_graph, "h": h}


_SCENARIOS: dict[str, Callable[[ModuleType, str, str], dict[str, object]]] = {
    "read-twice": _read_twice,
    "read-three-times": _read_three_times,
    "two-passes": _two_passes,
    "create-graph": _create_graph,
}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_matches_the_reference(name: str, dtype: str, ref: ModuleType, device: str) -> None:
    if device == "metal" and dtype == "float64":
        pytest.skip("Metal holds no float64")
    _compare(_SCENARIOS[name](lucid, device, dtype), _SCENARIOS[name](ref, "cpu", dtype))


# ── without the reference ──────────────────────────────────────────────────


def test_the_reported_case(device: str) -> None:
    x = lucid.ones(3, requires_grad=True, device=device)
    y = x * 1
    y.retain_grad()
    (y.sum() + (2 * y).sum()).backward()
    assert y.grad is not None and x.grad is not None
    assert y.grad.tolist() == [3.0, 3.0, 3.0]
    assert x.grad.tolist() == [3.0, 3.0, 3.0]


def test_single_requires_grad_operand_and_shared_gradient_isolation(device: str) -> None:
    x = lucid.tensor([2.0, -3.0, 4.0], requires_grad=True, device=device)
    c = lucid.tensor(5.0, device=device)
    g = lucid.tensor([1.0, 2.0, -1.0], device=device)

    (x + c).backward(g)
    assert x.grad is not None and c.grad is None
    assert x.grad.tolist() == [1.0, 2.0, -1.0]
    g.add_(10.0)
    assert x.grad.tolist() == [1.0, 2.0, -1.0]

    x.grad = None
    y = lucid.tensor([1.0, 1.0, 1.0], requires_grad=True, device=device)
    (x + y).sum().backward()
    assert x.grad is not None and y.grad is not None
    x.grad.add_(7.0)
    assert y.grad.tolist() == [1.0, 1.0, 1.0]

    x.grad = None
    ((x * c) - (c / x) + (x**c) + lucid.fmod(x, c) + lucid.remainder(x, c)).sum().backward()
    assert x.grad is not None and c.grad is None

    a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=device)
    b = lucid.tensor([[2.0, 0.0], [1.0, 3.0]], device=device)
    cond = lucid.tensor([[True, False], [False, True]], device=device)
    lucid.where(cond, a @ b, b @ a).sum().backward()
    assert a.grad is not None and b.grad is None
    assert a.grad.tolist() == [[3.0, 3.0], [3.0, 3.0]]
