"""``view`` of a strided tensor answers as the reference does.

A strided tensor's axes fall into chunks — runs whose strides step through
their elements in order.  ``view`` (and ``reshape``, which copies only where
``view`` refuses) reads the input in place as any shape whose axes multiply
out to exactly those chunks: it merges axes within a chunk and splits any
axis, and refuses a shape that would need one axis to span two chunks
(LCD-390).  Until then it took a strided input only when the new shape
inserted or dropped size-1 axes, refusing ``x[..., 1:][..., ::2].view(6, 4)``
which the reference views.

Each layout here is a member of a base's view family with an explicit
geometry, so the Metal half exercises the engine's strided Metal views too.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.test._fixtures.devices import metal_available

DEVICES = ["cpu"] + (["metal"] if metal_available() else [])

# name: (base numel, shape, stride, offset), in elements.
LAYOUTS: dict[str, tuple[int, tuple[int, ...], tuple[int, ...], int]] = {
    # x = arange(54).reshape(2, 3, 9)
    "sliced": (54, (2, 3, 8), (27, 9, 1), 1),  # x[..., 1:]
    "step_sliced": (54, (2, 3, 4), (27, 9, 2), 1),  # x[..., 1:][..., ::2]
    "row_offset": (12, (2, 3), (3, 1), 3),  # arange(12).reshape(4, 3)[1:3]
    "transposed": (12, (4, 3), (1, 4), 0),  # arange(12).reshape(3, 4).t()
    "expanded_outer": (3, (2, 3), (0, 1), 0),  # arange(3).expand(2, 3)
    "expanded_inner": (3, (3, 4), (1, 0), 0),  # arange(3)[:, None].expand(3, 4)
    "expanded_scalar": (1, (2, 3), (0, 0), 0),
    # A size-1 axis with a stride of its own — it moves nothing.
    "unit_axis": (24, (3, 1, 4), (8, 5, 2), 0),
    "empty": (6, (2, 0, 3), (3, 3, 1), 0),
    "empty_offset": (12, (0, 3), (5, 1), 4),
    "scalar": (6, (), (), 4),
}

TARGETS: dict[str, list[tuple[int, ...]]] = {
    "sliced": [(6, 8), (6, 2, 4), (2, 3, 2, 4), (-1, 8), (2, 24), (48,), (16, 3)],
    "step_sliced": [
        (6, 4),
        (6, 2, 2),
        (2, 3, 2, 2),
        (-1, 4),
        (6, -1),
        (2, 1, 3, 4, 1),
        (24,),
        (3, 8),
        (2, 12),
        (12, 2),
        (-1,),
    ],
    "row_offset": [(6,), (3, 2), (-1,), (1, 2, 3, 1)],
    "transposed": [(2, 2, 3), (4, 3, 1), (1, 4, 3), (12,), (3, 4), (4, 3, 1, 1), (2, 6)],
    "expanded_outer": [(2, 1, 3), (2, 3, 1), (1, 2, 3), (6,), (3, 2)],
    "expanded_inner": [(3, 2, 2), (3, 4, 1), (12,), (6, 2), (-1, 2, 2)],
    "expanded_scalar": [(6,), (3, 2), (6, 1), (1, 1, 6)],
    "unit_axis": [(12,), (3, 4), (6, 2), (2, 6), (3, 1, 2, 2), (-1,)],
    "empty": [(0,), (3, 0, 2), (0, 6), (2, 0, 3), (-1,), (6, 0)],
    "empty_offset": [(0,), (3, 0), (0, 1, 3), (-1, 3)],
    "scalar": [(), (1,), (1, 1), (-1,)],
}

CASES = [(layout, target) for layout, targets in TARGETS.items() for target in targets]
IDS = [f"{layout}-{'x'.join(map(str, t)) or 'scalar'}" for layout, t in CASES]


def _lucid_layout(name: str, device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    n, shape, stride, offset = LAYOUTS[name]
    base = lucid.arange(float(n)).to(device)
    impl = _C_engine.TensorImpl._make_view(base._impl, list(shape), list(stride), offset)
    return base, _wrap(impl)


def _ref_layout(ref: Any, name: str) -> tuple[Any, Any]:
    n, shape, stride, offset = LAYOUTS[name]
    base = ref.arange(float(n), dtype=ref.float32)
    return base, base.as_strided(shape, stride, offset)


def _try(fn: Callable[[], Any]) -> tuple[Any, Exception | None]:
    try:
        return fn(), None
    except RuntimeError as exc:
        return None, exc


def _numel(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape)) if shape else 1


@pytest.mark.parity
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("layout", "target"), CASES, ids=IDS)
def test_view_answers_as_the_reference(
    ref: Any, layout: str, target: tuple[int, ...], device: str
) -> None:
    rbase, rv = _ref_layout(ref, layout)
    want, refused = _try(lambda: rv.view(target))
    base, v = _lucid_layout(layout, device)
    if refused is not None:
        with pytest.raises(RuntimeError, match="use .reshape"):
            v.view(*target)
        # ``reshape`` takes it anyway, by copying.
        got = v.reshape(*target)
        np.testing.assert_array_equal(got.numpy(), rv.reshape(target).numpy())
        return

    got = v.view(*target)
    assert got.shape == tuple(want.shape)
    np.testing.assert_array_equal(got.numpy(), want.numpy())
    assert got.stride() == tuple(want.stride())
    # ``reshape`` is the same view.
    assert v.reshape(*target).stride() == got.stride()

    # A later write to the base shows through the view...
    with lucid.no_grad():
        base.add_(100.0)
    rbase.add_(100.0)
    np.testing.assert_array_equal(got.numpy(), want.numpy())

    # ...and, where no two of its elements share a slot, a write through the
    # view reaches the base.
    if 0 not in LAYOUTS[layout][2] or _numel(LAYOUTS[layout][1]) <= 1:
        with lucid.no_grad():
            got.mul_(-1.0)
        want.mul_(-1.0)
        np.testing.assert_array_equal(base.numpy(), rbase.numpy())


def _strided(x: lucid.Tensor | Any) -> Any:
    return x[..., 1:][..., ::2]


@pytest.mark.parity
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("target", [(6, 4), (6, 2, 2), (2, 3, 2, 2)])
def test_gradient_flows_through_a_merged_or_split_view(
    ref: Any, target: tuple[int, ...], device: str
) -> None:
    data = np.random.default_rng(0).standard_normal((2, 3, 9)).astype(np.float32)
    weight = np.random.default_rng(1).standard_normal(target).astype(np.float32)

    x = lucid.tensor(data, device=device, requires_grad=True)
    w = _strided(x * 1.0).view(*target)
    (w**3 * lucid.tensor(weight, device=device)).sum().backward()

    rx = ref.tensor(data, requires_grad=True)
    rw = _strided(rx * 1.0).view(target)
    (rw**3 * ref.tensor(weight)).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), rx.grad.numpy(), rtol=1e-5, atol=1e-6)

    # Second order through the view.
    x2 = lucid.tensor(data, device=device, requires_grad=True)
    (g,) = lucid.autograd.grad((_strided(x2).view(*target) ** 3).sum(), x2, create_graph=True)
    g.sum().backward()
    rx2 = ref.tensor(data, requires_grad=True)
    (rg,) = ref.autograd.grad((_strided(rx2).view(target) ** 3).sum(), rx2, create_graph=True)
    rg.sum().backward()
    np.testing.assert_allclose(x2.grad.numpy(), rx2.grad.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("device", DEVICES)
def test_in_place_write_through_a_merged_view_reaches_the_base(ref: Any, device: str) -> None:
    data = np.random.default_rng(2).standard_normal((2, 3, 9)).astype(np.float32)

    x = lucid.tensor(data, device=device, requires_grad=True)
    y = x * 1.0
    _strided(y).view(6, 4).mul_(2.0)
    (y**2).sum().backward()

    rx = ref.tensor(data, requires_grad=True)
    ry = rx * 1.0
    _strided(ry).view(6, 4).mul_(2.0)
    (ry**2).sum().backward()

    np.testing.assert_allclose(y.detach().numpy(), ry.detach().numpy(), rtol=1e-6)
    np.testing.assert_allclose(x.grad.numpy(), rx.grad.numpy(), rtol=1e-5, atol=1e-6)
