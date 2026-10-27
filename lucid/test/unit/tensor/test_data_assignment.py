"""``t.data = value`` swaps the storage under a tensor and keeps the tensor.

There was no setter, so ``param.data = t`` — the reference idiom for
re-initialising a weight or a hand-written update — raised "property 'data'
of 'Parameter' object has no setter" (LCD-478).  The reference keeps the
object, its ``requires_grad``, a module's registration of it and its
gradient, takes ``value``'s storage without a copy, and records no graph.
``Module._apply`` (``.to()`` / ``.double()``) swaps through the same setter.
"""

import pytest

import lucid
import lucid.nn as nn


def _holders(device):
    """A plain tensor that requires grad, and a module's Parameter."""
    module = nn.Linear(2, 3).to(device)
    return [
        ("tensor", lucid.ones(3, 2, device=device, requires_grad=True), None),
        ("parameter", module.weight, module),
    ]


@pytest.mark.parametrize("which", ["tensor", "parameter"])
def test_assignment_swaps_the_data_and_keeps_the_tensor(device, which):
    _, t, module = next(h for h in _holders(device) if h[0] == which)
    kind, before = type(t), id(t)
    value = lucid.arange(8.0, device=device).reshape(4, 2)
    t.data = value
    assert id(t) == before and type(t) is kind
    assert t.shape == (4, 2) and t.tolist() == value.tolist()
    assert t.requires_grad and t.is_leaf and t.grad_fn is None
    assert t.device == value.device
    if module is not None:
        assert any(p is t for p in module.parameters())
        assert module.weight is t


def test_the_tensor_still_trains_after_a_swap(device):
    module = nn.Linear(2, 1).to(device)
    module.weight.data = lucid.zeros(1, 2, device=device)
    module(lucid.ones(1, 2, device=device)).sum().backward()
    assert module.weight.grad is not None
    assert module.weight.grad.tolist() == [[1.0, 1.0]]


def test_an_optimizer_steps_the_new_data(device):
    module = nn.Linear(2, 1).to(device)
    opt = lucid.optim.SGD(module.parameters(), lr=0.5)
    module.weight.data = lucid.zeros(1, 2, device=device)
    module(lucid.ones(1, 2, device=device)).sum().backward()
    opt.step()
    assert module.weight.tolist() == [[-0.5, -0.5]]


def test_a_gradient_that_still_fits_is_kept(device):
    t = lucid.ones(3, device=device, requires_grad=True)
    (t * 2.0).sum().backward()
    t.data = lucid.zeros(3, device=device)
    assert t.grad is not None and t.grad.tolist() == [2.0, 2.0, 2.0]
    t.data = t.data - 0.5 * t.grad  # the hand-written update
    assert t.tolist() == [-1.0, -1.0, -1.0] and t.grad is not None


def test_a_gradient_of_the_old_shape_is_dropped(device):
    t = lucid.ones(3, device=device, requires_grad=True)
    (t * 2.0).sum().backward()
    t.data = lucid.zeros(2, 2, device=device)
    assert t.grad is None


def test_the_value_keeps_its_own_flag_and_graph(device):
    src = lucid.ones(2, device=device, requires_grad=True)
    value = src * 5.0
    t = lucid.zeros(2, device=device)
    t.data = value
    assert not t.requires_grad and t.grad_fn is None
    assert value.requires_grad and value.grad_fn is not None


def test_the_storage_is_shared_with_the_value_and_not_with_old_views(device):
    value = lucid.zeros(2, device=device)
    t = lucid.ones(3, device=device)
    view = t[:2]
    t.data = value
    value.add_(5.0)
    assert t.tolist() == [5.0, 5.0]
    assert view.tolist() == [1.0, 1.0]
    view.add_(10.0)
    assert view.tolist() == [11.0, 11.0]
    assert t.tolist() == [5.0, 5.0]
    assert value.tolist() == [5.0, 5.0]


@pytest.mark.parametrize(
    "take",
    [lambda y: y[0], lambda y: y[:, 0], lambda y: y.view(-1), lambda y: y.T],
    ids=["row", "col", "flat", "transpose"],
)
def test_setting_data_on_a_view_detaches_it_from_its_old_base(device, take):
    y = lucid.zeros(2, 3, device=device)
    w = take(y)
    replacement = lucid.ones(w.shape, device=device)
    w.data = replacement
    w.add_(2.0)
    assert y.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert w.tolist() == lucid.full(w.shape, 3.0).tolist()
    assert replacement.tolist() == w.tolist()
    assert w.stride() == replacement.stride()
    assert w.storage_offset() == 0
    assert w.is_contiguous() is True


def test_setting_data_to_another_view_retargets_writes_to_the_new_base(device):
    y = lucid.zeros(2, 3, device=device)
    z = lucid.zeros(3, 2, device=device)
    w = y[0]
    v = z[:, 0]
    w.data = v
    assert w.stride() == (2,)
    assert w.is_contiguous() is False
    w.add_(5.0)
    assert y.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert z.tolist() == [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]]
    assert v.tolist() == [5.0, 5.0, 5.0]


def test_module_apply_and_load_state_dict_on_a_view_parameter(device):
    base = lucid.zeros(2, 3, device=device)
    module = nn.Linear(3, 1, bias=False).to(device)
    module.weight = nn.Parameter(base[:1])
    with lucid.no_grad():
        module.weight.add_(2.0)
    assert base.tolist() == [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]

    module.load_state_dict({"weight": lucid.full((1, 3), 4.0, device=device)})
    assert base.tolist() == [[4.0, 4.0, 4.0], [0.0, 0.0, 0.0]]

    module._apply(lambda t: lucid.ones_like(t))
    with lucid.no_grad():
        module.weight.add_(10.0)
    assert module.weight.tolist() == [[11.0, 11.0, 11.0]]
    assert base.tolist() == [[4.0, 4.0, 4.0], [0.0, 0.0, 0.0]]

    base2 = lucid.zeros(2, 3, device=device)
    module.weight.data = base2[1:]
    module.load_state_dict({"weight": lucid.full((1, 3), 7.0, device=device)}, assign=True)
    with lucid.no_grad():
        module.weight.add_(1.0)
    assert module.weight.tolist() == [[8.0, 8.0, 8.0]]
    assert base2.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


@pytest.mark.parametrize(
    "take", [lambda y: y[0], lambda y: y[:, 0], lambda y: y.view(-1)], ids=["row", "col", "flat"]
)
def test_detach_in_place_on_a_view_matches_cpu_and_metal(device, take):
    # A detached view shares storage with its base, and writes through it are
    # untracked by autograd on both a leaf and a computed tensor.
    leaf = lucid.zeros(2, 3, device=device, requires_grad=True)
    v_leaf = take(leaf)
    v_leaf.detach_()
    assert not v_leaf.requires_grad and v_leaf.grad_fn is None
    v_leaf.add_(3.0)

    x = lucid.ones(2, 3, device=device, requires_grad=True)
    y = x * 2.0
    v = take(y)
    v.detach_()
    assert not v.requires_grad and v.grad_fn is None
    v.add_(10.0)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.tolist() == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]

    # Detaching the base itself in-place leaves the base writable even when an
    # older view was taken while it required grad.
    b = lucid.zeros(2, 3, device=device, requires_grad=True)
    _held = b[:, 0]
    b.detach_()
    b.add_(1.0)
    assert b.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_an_integer_value_on_a_tensor_without_grad_is_taken():
    t = lucid.ones(2)
    t.data = lucid.zeros(2, dtype=lucid.int64)
    assert t.dtype == lucid.int64


@pytest.mark.parametrize(
    "case,error",
    [
        ("not a tensor", TypeError),
        ("integer into a tensor that requires grad", RuntimeError),
        ("not a leaf", RuntimeError),
    ],
)
def test_assignment_refusals_leave_the_tensor_alone(case, error):
    leaf = lucid.ones(2, requires_grad=True)
    t = leaf * 2.0 if case == "not a leaf" else leaf
    value = {
        "not a tensor": 3,
        "integer into a tensor that requires grad": lucid.zeros(2, dtype=lucid.int64),
        "not a leaf": lucid.zeros(2),
    }[case]
    impl = t._impl
    with pytest.raises(error):
        t.data = value
    assert t._impl is impl


@pytest.mark.parity
def test_assignment_matches_the_reference(ref):
    import numpy as np

    def run(lib, tensor_of):
        module = lib.nn.Linear(2, 2)
        w = module.weight
        module(tensor_of(np.ones((1, 2)))).sum().backward()
        w.data = tensor_of(np.arange(6.0).reshape(3, 2))
        grad = None if w.grad is None else w.grad.shape
        return (
            module.weight is w,
            type(w).__name__,
            w.requires_grad,
            tuple(w.shape),
            w.grad_fn is None,
            any(p is w for p in module.parameters()),
            np.asarray(w.detach().numpy()).tolist(),
            grad,
        )

    mine = run(lucid, lambda a: lucid.tensor(a.astype(np.float32)))
    theirs = run(ref, lambda a: ref.tensor(a.astype(np.float32)))
    # The reference keeps a gradient of the old shape; Lucid's gradient
    # always matches its tensor, so that one is dropped.
    assert mine[:-1] == theirs[:-1]
    assert theirs[-1] is not None and mine[-1] is None


@pytest.mark.parametrize("via_local", [False, True], ids=["direct", "local-handle"])
def test_setting_data_on_a_gradient_updates_the_owner_slot(device, via_local):
    p = nn.Parameter(lucid.tensor([1.0, 2.0, 3.0], device=device))
    (p * 2.0).sum().backward()
    replacement = lucid.tensor([7.0, 8.0, 9.0], device=device)
    if via_local:
        g = p.grad
        assert g is not None
        g.data = replacement
        assert g.tolist() == [7.0, 8.0, 9.0]
    else:
        assert p.grad is not None
        p.grad.data = replacement
    assert p.grad is not None
    assert p.grad.tolist() == [7.0, 8.0, 9.0]

    opt = lucid.optim.SGD([p], lr=0.5)
    opt.step()
    assert p.tolist() == [-2.5, -2.0, -1.5]


@pytest.mark.parity
@pytest.mark.parametrize("via_local", [False, True], ids=["direct", "local-handle"])
def test_gradient_data_assignment_matches_the_reference(ref, via_local):
    import numpy as np

    def run(lib, tensor_of):
        p = lib.nn.Parameter(tensor_of(np.array([1.0, 2.0, 3.0])))
        (p * 2.0).sum().backward()
        replacement = tensor_of(np.array([7.0, 8.0, 9.0]))
        if via_local:
            g = p.grad
            g.data = replacement
        else:
            p.grad.data = replacement
        opt = lib.optim.SGD([p], lr=0.5)
        opt.step()
        return (
            np.asarray(p.grad.detach().numpy()).tolist(),
            np.asarray(p.detach().numpy()).tolist(),
        )

    mine = run(lucid, lambda a: lucid.tensor(a.astype(np.float32)))
    theirs = run(ref, lambda a: ref.tensor(a.astype(np.float32)))
    assert mine == theirs
