"""An in-place write through a free-function view of a Metal tensor reaches it.

``x.squeeze(0).add_(1)`` wrote through on Metal, but ``lucid.squeeze(x,
0).add_(1)`` did not: only the Tensor methods linked their Metal results to
the tensor they view, in a Python layer since replaced by the engine's view
family, and the free functions generated from the same registry entries went
straight to the engine.  The
same was true of ``lucid.narrow``, ``lucid.transpose``, ``lucid.split`` and
the rest, of ``lucid.Tensor(x)``, and of the methods ``ravel``,
``broadcast_to``, ``as_strided``, ``squeeze_all`` and ``expand_dims``, which
were never in the table at all.  None raised; the CPU wrote through.

Every case is held to what the CPU does with the same code.
"""

import inspect
from collections.abc import Callable

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

View = Callable[[lucid.Tensor], lucid.Tensor]

#: Every free-function spelling of a view op, with the shape of the tensor it
#: is taken from.  Registry-generated ones first, then the hand-written ones
#: built on them, then the constructor.
FREE_VIEWS: dict[str, tuple[tuple[int, ...], View]] = {
    "view": ((2, 3, 4), lambda x: lucid.view(x, (6, 4))),
    "reshape": ((2, 3, 4), lambda x: lucid.reshape(x, (4, 6))),
    "flatten": ((2, 3, 4), lambda x: lucid.flatten(x)),
    "unflatten": ((2, 3, 4), lambda x: lucid.unflatten(x, 2, (2, 2))),
    "ravel": ((2, 3, 4), lambda x: lucid.ravel(x)),
    "squeeze": ((1, 3, 4), lambda x: lucid.squeeze(x, 0)),
    "squeeze (all)": ((3, 1, 4), lambda x: lucid.squeeze(x)),
    "unsqueeze": ((2, 3, 4), lambda x: lucid.unsqueeze(x, 1)),
    "transpose": ((2, 3, 4), lambda x: lucid.transpose(x, 0, 2)),
    "permute": ((2, 3, 4), lambda x: lucid.permute(x, (2, 0, 1))),
    "movedim": ((2, 3, 4), lambda x: lucid.movedim(x, 0, 2)),
    "narrow": ((2, 3, 4), lambda x: lucid.narrow(x, 2, 1, 2)),
    "diagonal": ((2, 3, 4), lambda x: lucid.diagonal(x, 1, 1, 2)),
    "expand": ((2, 3, 4), lambda x: lucid.expand(x, (2, 3, 4))),
    "broadcast_to": ((2, 3, 4), lambda x: lucid.broadcast_to(x, (2, 3, 4))),
    "as_strided": ((2, 3, 4), lambda x: lucid.as_strided(x, (3, 4), (4, 1), 4)),
    "detach": ((2, 3, 4), lambda x: lucid.detach(x)),
    "contiguous": ((2, 3, 4), lambda x: lucid.contiguous(x)),
    "swapaxes": ((2, 3, 4), lambda x: lucid.swapaxes(x, 1, 2)),
    "swapdims": ((2, 3, 4), lambda x: lucid.swapdims(x, 0, 1)),
    "moveaxis": ((2, 3, 4), lambda x: lucid.moveaxis(x, 2, 0)),
    "adjoint": ((2, 3, 4), lambda x: lucid.adjoint(x)),
    "t": ((3, 4), lambda x: lucid.t(x)),
    "atleast_3d": ((3, 4), lambda x: lucid.atleast_3d(x)),
    "Tensor(x)": ((2, 3, 4), lambda x: lucid.Tensor(x)),
}

#: The method-only view ops that were missing from the table.
METHOD_VIEWS: dict[str, tuple[tuple[int, ...], View]] = {
    "ravel": ((2, 3, 4), lambda x: x.ravel()),
    "squeeze_all": ((1, 3, 1, 4), lambda x: x.squeeze_all()),
    "expand_dims": ((2, 3, 4), lambda x: x.expand_dims(1)),
    "broadcast_to": ((2, 3, 4), lambda x: x.broadcast_to((2, 3, 4))),
    "as_strided": ((2, 3, 4), lambda x: x.as_strided((2, 6), (6, 1), 6)),
}


def _src(v: lucid.Tensor) -> lucid.Tensor:
    """A source whose every element is distinct, to catch a misplaced write."""
    n = v.numel()
    return (-1.0 - lucid.arange(float(n))).reshape(tuple(v.shape)).to(v.device)


def _setitem(v: lucid.Tensor) -> None:
    v[0] = -7.0


WRITES: dict[str, Callable[[lucid.Tensor], object]] = {
    "add_": lambda v: v.add_(100.0),
    "zero_": lambda v: v.zero_(),
    "__setitem__": _setitem,
    "copy_": lambda v: v.copy_(_src(v)),
}


def _base(shape: tuple[int, ...], dev: str) -> lucid.Tensor:
    n = 1
    for s in shape:
        n *= s
    return lucid.arange(float(n)).reshape(shape).to(dev)


def _after(view: View, write: Callable[[lucid.Tensor], object], shape: tuple[int, ...]):
    out = {}
    for dev in ("cpu", "metal"):
        x = _base(shape, dev)
        write(view(x))
        out[dev] = x.to("cpu").tolist()
    return out


@pytest.mark.parametrize("write", list(WRITES))
@pytest.mark.parametrize("name", list(FREE_VIEWS))
def test_a_write_through_a_free_function_view_matches_the_cpu(
    name: str, write: str
) -> None:
    shape, view = FREE_VIEWS[name]
    out = _after(view, WRITES[write], shape)
    assert out["cpu"] != _base(shape, "cpu").tolist()  # the CPU wrote at all
    assert out["metal"] == out["cpu"]


@pytest.mark.parametrize("write", list(WRITES))
@pytest.mark.parametrize("name", list(METHOD_VIEWS))
def test_a_write_through_a_method_view_matches_the_cpu(name: str, write: str) -> None:
    shape, view = METHOD_VIEWS[name]
    out = _after(view, WRITES[write], shape)
    assert out["cpu"] != _base(shape, "cpu").tolist()
    assert out["metal"] == out["cpu"]


PIECES: dict[str, tuple[tuple[int, ...], Callable[[lucid.Tensor], object]]] = {
    "split (size)": ((2, 3, 4), lambda x: lucid.split(x, 3, 2)),
    "split (sizes)": ((2, 3, 4), lambda x: lucid.split(x, [1, 3], 2)),
    # The method names the sizes ``split_size``; the free function's
    # keywords reach the replay by position (LCD-410).
    "split (keywords)": (
        (2, 3, 4),
        lambda x: lucid.split(x, split_size_or_sections=3, dim=2),
    ),
    "chunk": ((2, 3, 4), lambda x: lucid.chunk(x, 2, 2)),
    "unbind": ((2, 3, 4), lambda x: lucid.unbind(x, 1)),
    "hsplit": ((2, 3, 4), lambda x: lucid.hsplit(x, 3)),
    "vsplit": ((2, 3, 4), lambda x: lucid.vsplit(x, 2)),
    "dsplit": ((2, 3, 4), lambda x: lucid.dsplit(x, [1, 3])),
    "tensor_split": ((2, 3, 4), lambda x: lucid.tensor_split(x, 2, 2)),
}


@pytest.mark.parametrize("name", list(PIECES))
def test_every_piece_of_a_free_function_split_writes_back(name: str) -> None:
    shape, cut = PIECES[name]
    out = {}
    for dev in ("cpu", "metal"):
        x = _base(shape, dev)
        pieces = cut(x)
        assert isinstance(pieces, (tuple, list))
        for i, piece in enumerate(pieces):
            if i % 2:
                piece.mul_(-(i + 1.0))
            else:
                piece.copy_(_src(piece))
        out[dev] = x.to("cpu").tolist()
    assert out["cpu"] != _base(shape, "cpu").tolist()
    assert out["metal"] == out["cpu"]


CHAINS: dict[str, tuple[tuple[int, ...], View]] = {
    "squeeze of transpose": (
        (1, 3, 4),
        lambda x: lucid.squeeze(lucid.transpose(x, 1, 2), 0),
    ),
    "narrow of permute": (
        (2, 3, 4),
        lambda x: lucid.narrow(lucid.permute(x, (2, 0, 1)), 0, 1, 2),
    ),
    "free of method": ((2, 3, 4), lambda x: lucid.narrow(x.transpose(0, 1), 0, 1, 2)),
    "method of free": ((2, 3, 4), lambda x: lucid.movedim(x, 0, 2)[1].t()),
    "piece of a view": ((2, 3, 4), lambda x: lucid.split(lucid.view(x, (4, 6)), 2)[1]),
    "Tensor of a view": ((2, 3, 4), lambda x: lucid.Tensor(lucid.narrow(x, 1, 1, 2))),
}


@pytest.mark.parametrize("write", list(WRITES))
@pytest.mark.parametrize("name", list(CHAINS))
def test_a_chain_of_views_writes_all_the_way_up(name: str, write: str) -> None:
    shape, view = CHAINS[name]
    out = _after(view, WRITES[write], shape)
    assert out["cpu"] != _base(shape, "cpu").tolist()
    assert out["metal"] == out["cpu"]


GRAD_VIEWS: dict[str, tuple[View, View]] = {
    "squeeze": (lambda w: lucid.squeeze(w, 0), lambda w: w.squeeze(0)),
    "transpose": (lambda w: lucid.transpose(w, 1, 2), lambda w: w.transpose(1, 2)),
    "narrow": (lambda w: lucid.narrow(w, 2, 0, 2), lambda w: w.narrow(2, 0, 2)),
    "reshape": (lambda w: lucid.reshape(w, (12,)), lambda w: w.reshape(12)),
    "split piece": (lambda w: lucid.split(w, 2, 2)[1], lambda w: w.split(2, 2)[1]),
    "unbind piece": (lambda w: lucid.unbind(w, 1)[0], lambda w: w.unbind(1)[0]),
}


@pytest.mark.parametrize("spelling", ["free", "method"])
@pytest.mark.parametrize("name", list(GRAD_VIEWS))
def test_a_view_of_a_leaf_that_requires_grad_is_refused_as_on_the_cpu(
    name: str, spelling: str
) -> None:
    view = GRAD_VIEWS[name][0 if spelling == "free" else 1]
    for dev in ("cpu", "metal"):
        w = lucid.zeros(1, 3, 4).to(dev).requires_grad_()
        with pytest.raises(RuntimeError, match="leaf tensor that requires grad"):
            view(w).add_(1.0)
        assert w.to("cpu").tolist() == lucid.zeros(1, 3, 4).tolist()
    seen = {}
    for dev in ("cpu", "metal"):
        w = lucid.zeros(1, 3, 4).to(dev).requires_grad_()
        with lucid.no_grad():
            view(w).add_(1.0)
        seen[dev] = w.to("cpu").tolist()
    assert seen["cpu"] != lucid.zeros(1, 3, 4).tolist()
    assert seen["metal"] == seen["cpu"]


def test_tensor_of_a_leaf_that_requires_grad_is_refused_as_on_the_cpu() -> None:
    for dev in ("cpu", "metal"):
        w = lucid.zeros(2, 3).to(dev).requires_grad_()
        with pytest.raises(RuntimeError, match="leaf tensor that requires grad"):
            lucid.Tensor(w).add_(1.0)
        with lucid.no_grad():
            lucid.Tensor(w).add_(1.0)
        assert w.to("cpu").tolist() == [[1.0] * 3] * 2


def test_a_free_function_detach_is_writable_under_autograd() -> None:
    for dev in ("cpu", "metal"):
        w = lucid.zeros(3).to(dev).requires_grad_()
        lucid.detach(w).add_(2.0)
        assert w.to("cpu").tolist() == [2.0, 2.0, 2.0]


@pytest.mark.parametrize(
    "view",
    [
        lambda x: lucid.expand(x, (2, 3, 4)),
        lambda x: lucid.broadcast_to(x, (2, 3, 4)),
        lambda x: lucid.as_strided(x, (3, 3), (1, 1)),
    ],
    ids=["expand", "broadcast_to", "as_strided"],
)
# zero_ is a fill, which every element can take at once: it goes through
# (test_fill_into_overlapping_views.py).
@pytest.mark.parametrize("write", [w for w in WRITES if w != "zero_"])
def test_a_write_into_overlapping_elements_is_refused_as_on_the_cpu(
    view: View, write: str
) -> None:
    for dev in ("cpu", "metal"):
        x = lucid.zeros(1, 3, 4).to(dev)
        with pytest.raises(RuntimeError, match="overlap"):
            WRITES[write](view(x))
        assert x.to("cpu").tolist() == lucid.zeros(1, 3, 4).tolist()


def test_gradients_through_free_function_view_writes_match_the_cpu() -> None:
    x0 = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)

    def f(x: lucid.Tensor) -> lucid.Tensor:
        y = x * 2.0
        lucid.narrow(y, 1, 1, 2).mul_(3.0)
        lucid.transpose(y, 0, 1)[3].add_(x[:, 0] * 5.0)
        lucid.split(y, 2, 1)[0].sub_(1.5)
        lucid.reshape(y, (12,))[4:8].mul_(0.5)
        return (y * y).sum()

    grads = {}
    for dev in ("cpu", "metal"):
        x = lucid.tensor(x0).to(dev).requires_grad_()
        f(x).backward()
        grads[dev] = x.grad.numpy()
    np.testing.assert_allclose(grads["metal"], grads["cpu"], rtol=1e-6, atol=1e-6)


_VIEW_OPS = frozenset(
    {
        "as_strided",
        "broadcast_to",
        "chunk",
        "contiguous",
        "detach",
        "diagonal",
        "expand",
        "flatten",
        "movedim",
        "narrow",
        "permute",
        "ravel",
        "reshape",
        "split",
        "squeeze",
        "transpose",
        "unbind",
        "unfold",
        "unsqueeze",
        "view",
    }
)


def test_every_free_function_of_a_linked_view_op_is_covered_here() -> None:
    from lucid._ops._registry import _REGISTRY

    shared = {
        e.free_fn_name
        for e in _REGISTRY
        if e.free_fn_name is not None and e.free_fn_name == e.method_name
    }
    covered = {case.split(" ")[0] for case in (*FREE_VIEWS, *PIECES)}
    missing = _VIEW_OPS & shared - covered
    assert not missing, f"no case for lucid.{', lucid.'.join(sorted(missing))}"


def test_a_cpu_view_is_not_linked() -> None:
    for dev in ("cpu", "metal"):
        x = lucid.zeros(1, 3, 4, device=dev)
        for v in (lucid.squeeze(x, 0), lucid.transpose(x, 1, 2), lucid.Tensor(x)):
            assert "_metal_view" not in v.__dict__
            assert v._impl.family_id() == x._impl.family_id() != 0
        for piece in lucid.split(x, 2, 2):
            assert "_metal_view" not in piece.__dict__
            assert piece._impl.family_id() == x._impl.family_id() != 0


def test_contiguous_returns_self_when_already_contiguous() -> None:
    for dev in ("cpu", "metal"):
        x = lucid.arange(12.0, device=dev).reshape(3, 4)
        assert x.contiguous() is x
        assert lucid.contiguous(x) is x
        t = x.t()
        assert not t.is_contiguous()
        c = t.contiguous()
        assert c is not t
        assert c.is_contiguous()
        assert c.contiguous() is c


def test_the_free_functions_keep_their_signatures() -> None:
    assert list(inspect.signature(lucid.narrow).parameters) == [
        "input",
        "dim",
        "start",
        "length",
    ]
    assert lucid.squeeze.__name__ == "squeeze"
    assert lucid.squeeze.__doc__ == inspect.unwrap(lucid.squeeze).__doc__
    assert "data" in inspect.signature(lucid.Tensor).parameters
