"""A view that covers its whole source is still a tensor of its own.

``x[:]``, ``x[...]``, ``x[()]``, ``x[0:n]``, ``x.narrow(0, 0, n)``,
``x.squeeze(d)`` on a dim that is not size 1 and ``lucid.Tensor(x)`` used to
hand back ``x``'s own ``TensorImpl`` under a second Python object.  While
``requires_grad`` changes swapped the impl that went unnoticed; once they
flip the flag in place (so an optimizer holding the parameter keeps
updating it) the two objects were visibly one tensor:

* ``x[:].requires_grad_(True)`` turned ``x``'s flag on, and a saliency loop
  that takes ``X[:n]`` each iteration accumulated every iteration's
  gradient into one ``.grad``;
* ``p[:].requires_grad = False`` froze the parameter ``p``.

The reference framework returns a view: the same storage, a separate
tensor, and — when the source requires grad — a non-leaf in its graph.
These pin that, and that the view still reads and writes the source.
"""

from typing import Any, Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn

_WHOLE: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "x[:]": lambda x: x[:],
    "x[...]": lambda x: x[...],
    "x[()]": lambda x: x[()],
    "x[0:4]": lambda x: x[0:4],
    "x[-4:]": lambda x: x[-4:],
    "x[::1]": lambda x: x[::1],
    "x[:, 0:3]": lambda x: x[:, 0:3],
    "x.narrow(0, 0, 4)": lambda x: x.narrow(0, 0, 4),
    "x.squeeze(0)": lambda x: x.squeeze(0),
    "x.squeeze()": lambda x: x.squeeze(),
    "x.squeeze([0, 1])": lambda x: x.squeeze([0, 1]),
    "lucid.squeeze(x, 0)": lambda x: lucid.squeeze(x, 0),
}
_IDS = list(_WHOLE)


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("op", _IDS, ids=_IDS)
def test_flag_on_the_view_stays_on_the_view(op: str, device: str) -> None:
    x = lucid.ones(4, 3, device=device)
    y = _WHOLE[op](x)
    assert y._impl is not x._impl
    y.requires_grad_(True)
    assert not x.requires_grad
    (y * 2.0).sum().backward()
    assert x.grad is None
    np.testing.assert_allclose(y.grad.numpy(), np.full((4, 3), 2.0))


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("op", _IDS, ids=_IDS)
def test_view_of_a_parameter_is_in_its_graph(op: str, device: str) -> None:
    p = nn.Parameter(lucid.ones(4, 3, device=device))
    v = _WHOLE[op](p)
    assert not v.is_leaf and v.requires_grad
    for flag in (True, False):
        with pytest.raises(RuntimeError, match="leaf"):
            v.requires_grad = flag
    with pytest.raises(RuntimeError, match="leaf"):
        v.requires_grad_(False)
    assert p.requires_grad
    (v * 3.0).sum().backward()
    np.testing.assert_allclose(p.grad.numpy(), np.full((4, 3), 3.0))


@pytest.mark.parametrize("op", _IDS, ids=_IDS)
def test_view_still_shares_storage(op: str) -> None:
    x = lucid.zeros(4, 3)
    y = _WHOLE[op](x)
    y.add_(1.0)
    assert float(x.sum().item()) == 12.0
    x.add_(1.0)
    assert float(y.sum().item()) == 24.0


@pytest.mark.parametrize(
    "op",
    [o for o in _IDS if not o.startswith("lucid.")],
    ids=[o for o in _IDS if not o.startswith("lucid.")],
)
def test_metal_view_writes_reach_the_source(op: str) -> None:
    x = lucid.zeros(4, 3, device="metal")
    y = _WHOLE[op](x)
    y.add_(1.0)
    y[0] = 5.0
    assert float(x.sum().item()) == 24.0


@pytest.mark.parametrize("requires_grad", [False, True])
def test_tensor_constructor_is_a_tensor_of_its_own(requires_grad: bool) -> None:
    x = lucid.ones(3, requires_grad=requires_grad)
    y = lucid.Tensor(x, requires_grad=requires_grad)
    assert y._impl is not x._impl
    if requires_grad:
        # In x's graph, as the reference's Tensor(x) is.
        assert not y.is_leaf
        (y * 2.0).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0])
    else:
        # Same storage, separate flag.
        y.add_(1.0)
        assert float(x.sum().item()) == 6.0
        y.requires_grad_(True)
        assert not x.requires_grad


def _saliency(lib: Any, data: Any, model: Any) -> tuple[list[float], bool]:
    out = []
    for _ in range(3):
        x = data[:4]  # every row: a whole view
        x.requires_grad_(True)
        model(x).sum().backward()
        out.append(float(x.grad.sum().item()))
    return out, bool(data.requires_grad)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_saliency_loop_gradient_is_per_iteration(device: str) -> None:
    model = nn.Linear(3, 1, bias=False).to(device)
    model.weight = nn.Parameter(lucid.ones(1, 3, device=device))
    grads, data_flag = _saliency(lucid, lucid.ones(4, 3, device=device), model)
    assert grads == [12.0, 12.0, 12.0]
    assert not data_flag


@pytest.mark.parity
def test_saliency_loop_follows_the_reference(ref: Any) -> None:
    model = nn.Linear(3, 1, bias=False)
    model.weight = nn.Parameter(lucid.ones(1, 3))
    ref_model = ref.nn.Linear(3, 1, bias=False)
    with ref.no_grad():
        ref_model.weight.fill_(1.0)
    assert _saliency(lucid, lucid.ones(4, 3), model) == _saliency(ref, ref.ones(4, 3), ref_model)
