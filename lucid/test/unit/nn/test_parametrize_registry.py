"""A parametrized tensor's originals are part of the module tree (LCD-624).

``register_parametrization`` used to keep its container in a plain dict
and delete the leaf parameter, so the trainable original was in no
registry: ``parameters()`` and ``state_dict()`` did not list it, an
optimizer built from the module never moved it, and a checkpoint dropped
the weight.  Training through a parametrization was a silent no-op.

Each check here is one way the module tree is read — the names, an
optimizer step, a checkpoint, a device move — so a tensor that falls out
of the tree fails at least one of them.
"""

import copy
import pickle

import pytest

import lucid
import lucid.nn as nn
from lucid.nn.utils import parametrize


class _Symmetric(nn.Module):
    def forward(self, X: lucid.Tensor) -> lucid.Tensor:
        return 0.5 * (X + X.mT)


class _Scale(nn.Module):
    def forward(self, X: lucid.Tensor) -> lucid.Tensor:
        return 3.0 * X


class _Halved(nn.Module):
    """A parametrization with a right inverse, so assignment works."""

    def forward(self, X: lucid.Tensor) -> lucid.Tensor:
        return 2.0 * X

    def right_inverse(self, W: lucid.Tensor) -> lucid.Tensor:
        return W / 2.0


class _Sum(nn.Module):
    """Two originals: ``right_inverse`` splits the tensor in halves."""

    def forward(self, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
        return a + b

    def right_inverse(self, W: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        return W * 0.5, W * 0.5


class _WithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buf", lucid.ones(3))


def _names(module):
    return (
        [n for n, _ in module.named_parameters()],
        [n for n, _ in module.named_buffers()],
        list(module.state_dict()),
    )


def _steps(nn_):
    """The parametrizations above, written against either framework's ``nn``."""

    class Sym(nn_.Module):
        def forward(self, X):
            return 0.5 * (X + X.mT)

    class Scale(nn_.Module):
        def forward(self, X):
            return 3.0 * X

    class Sum(nn_.Module):
        def forward(self, a, b):
            return a + b

        def right_inverse(self, W):
            return W * 0.5, W * 0.5

    return Sym, Scale, Sum


def _register(case, nn_, P, m):
    Sym, Scale, Sum = _steps(nn_)
    plan = {
        "one": [("weight", Sym)],
        "stacked": [("weight", Sym), ("weight", Scale)],
        "two originals": [("weight", Sum)],
        "weight and bias": [("weight", Scale), ("bias", Scale)],
    }[case]
    for name, step in plan:
        P.register_parametrization(m, name, step())


CASES = ["one", "stacked", "two originals", "weight and bias"]


@pytest.mark.parity
@pytest.mark.parametrize("case", CASES)
def test_the_names_match_the_reference(ref, case):
    theirs = ref.nn.Linear(4, 4)
    ours = nn.Linear(4, 4)
    _register(case, ref.nn, ref.nn.utils.parametrize, theirs)
    _register(case, nn, parametrize, ours)
    assert _names(ours) == _names(theirs)
    assert type(ours).__name__ == type(theirs).__name__
    assert [n for n, _ in ours.named_modules()] == [n for n, _ in theirs.named_modules()]


@pytest.mark.parity
def test_a_parametrized_buffer_stays_a_buffer(ref):
    class TheirBuffer(ref.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buf", ref.ones(3))

    theirs, ours = TheirBuffer(), _WithBuffer()
    ref.nn.utils.parametrize.register_parametrization(theirs, "buf", _steps(ref.nn)[1]())
    parametrize.register_parametrization(ours, "buf", _Scale())
    assert _names(ours) == _names(theirs)


def test_the_names_without_the_reference() -> None:
    # The same contract, held when the parity tier is not installed.
    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", _Symmetric())
    assert _names(m)[0] == ["bias", "parametrizations.weight.original"]
    assert _names(m)[2] == ["bias", "parametrizations.weight.original"]
    parametrize.register_parametrization(m, "weight", _Scale())
    assert _names(m)[0] == ["bias", "parametrizations.weight.original"]


def test_an_optimizer_step_moves_the_original_and_the_read(device: str) -> None:
    m = nn.Linear(4, 4).to(device)
    parametrize.register_parametrization(m, "weight", _Symmetric())
    original = m.parametrizations.weight.original
    opt = lucid.optim.SGD(m.parameters(), lr=0.1)
    assert any(p is original for p in m.parameters())

    before = original.detach().clone()
    m(lucid.randn(2, 4, device=device)).sum().backward()
    opt.step()

    assert not bool(lucid.equal(original, before))
    # No forward since the step: the read is computed from the moved original.
    expected = 0.5 * (original + original.mT)
    assert bool(lucid.allclose(m.weight, expected))


def test_an_optimizer_built_before_registration_still_trains() -> None:
    m = nn.Linear(4, 4)
    weight = m.weight
    opt = lucid.optim.SGD(m.parameters(), lr=0.1)
    parametrize.register_parametrization(m, "weight", _Symmetric())
    assert m.parametrizations.weight.original is weight
    before = weight.detach().clone()
    m(lucid.randn(2, 4)).sum().backward()
    opt.step()
    assert not bool(lucid.equal(weight, before))


def test_zero_grad_and_device_moves_reach_the_original(device: str) -> None:
    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", _Symmetric())
    m.to(device)
    assert m.parametrizations.weight.original.device.type == device
    m(lucid.randn(2, 4, device=device)).sum().backward()
    assert m.parametrizations.weight.original.grad is not None
    m.zero_grad()
    assert m.parametrizations.weight.original.grad is None


@pytest.mark.parametrize("make", [_Symmetric, _Sum], ids=["one", "two originals"])
def test_a_checkpoint_round_trips_into_a_fresh_parametrized_module(make: type[nn.Module]) -> None:
    src = nn.Linear(4, 4)
    parametrize.register_parametrization(src, "weight", make())
    dst = nn.Linear(4, 4)
    parametrize.register_parametrization(dst, "weight", make())
    assert not bool(lucid.equal(src.weight, dst.weight))

    result = dst.load_state_dict(src.state_dict())

    assert list(result.missing_keys) == []
    assert list(result.unexpected_keys) == []
    assert bool(lucid.equal(src.weight, dst.weight))


def test_removing_with_the_parametrized_value_keeps_the_object() -> None:
    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", _Scale())
    original = m.parametrizations.weight.original
    current = m.weight.detach().clone()

    parametrize.remove_parametrizations(m, "weight")

    assert type(m) is nn.Linear
    assert not hasattr(m, "parametrizations")
    assert m.weight is original
    assert isinstance(m.weight, nn.Parameter)
    assert bool(lucid.equal(m.weight, current))
    assert [n for n, _ in m.named_parameters()] == ["bias", "weight"]


def test_removing_without_the_parametrized_value_restores_the_original() -> None:
    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", _Scale())
    raw = m.parametrizations.weight.original.detach().clone()

    parametrize.remove_parametrizations(m, "weight", leave_parametrized=False)

    assert type(m) is nn.Linear
    assert bool(lucid.equal(m.weight, raw))


def test_removing_one_of_two_keeps_the_other() -> None:
    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", _Scale())
    parametrize.register_parametrization(m, "bias", _Scale())
    parametrize.remove_parametrizations(m, "weight")
    assert parametrize.is_parametrized(m, "bias")
    assert parametrize.type_before_parametrizations(m) is nn.Linear
    assert [n for n, _ in m.named_parameters()] == ["weight", "parametrizations.bias.original"]


def test_two_originals_cannot_be_left_unparametrized() -> None:
    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", _Sum())
    with pytest.raises(ValueError, match="sequence of tensors"):
        parametrize.remove_parametrizations(m, "weight", leave_parametrized=False)
    parametrize.remove_parametrizations(m, "weight")
    assert [n for n, _ in m.named_parameters()] == ["bias", "weight"]


def test_removing_what_was_never_registered_is_refused() -> None:
    with pytest.raises(ValueError, match="does not have a parametrization"):
        parametrize.remove_parametrizations(nn.Linear(2, 2), "weight")


def test_cached_computes_once_per_context() -> None:
    calls = []

    class Counting(nn.Module):
        def forward(self, X: lucid.Tensor) -> lucid.Tensor:
            calls.append(1)
            return X * 1.0

    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", Counting())
    calls.clear()
    with parametrize.cached():
        first = m.weight
        with parametrize.cached():
            assert m.weight is first
        assert m.weight is first  # the inner exit keeps the outer cache
    assert len(calls) == 1
    assert m.weight is not first
    assert len(calls) == 2


def test_assigning_a_tensor_writes_the_originals_through_the_inverse() -> None:
    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", _Halved())
    original = m.parametrizations.weight.original
    m.weight = lucid.ones(4, 4)
    assert m.parametrizations.weight.original is original
    assert bool(lucid.allclose(original, lucid.full((4, 4), 0.5)))
    assert bool(lucid.allclose(m.weight, lucid.ones(4, 4)))


def test_assigning_without_an_inverse_is_refused() -> None:
    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", _Scale())
    with pytest.raises(RuntimeError, match="right_inverse"):
        m.weight = lucid.ones(4, 4)


def test_a_parameter_cannot_be_hidden_under_the_property() -> None:
    # It would sit in ``_parameters`` while the property answered for the
    # name: trained by an optimizer, read by nothing.
    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", _Scale())
    with pytest.raises(KeyError, match="already exists"):
        m.weight = nn.Parameter(lucid.ones(4, 4))
    assert _names(m)[0] == ["bias", "parametrizations.weight.original"]


def test_a_refused_registration_leaves_the_module_unchanged() -> None:
    class Widen(nn.Module):
        def forward(self, X: lucid.Tensor) -> lucid.Tensor:
            return lucid.cat([X, X], 1)

        def right_inverse(self, W: lucid.Tensor) -> lucid.Tensor:
            return W / 2.0

    m = nn.Linear(4, 4)
    weight = m.weight
    values = weight.detach().clone()
    with pytest.raises(ValueError, match="shape"):
        parametrize.register_parametrization(m, "weight", Widen())
    assert type(m) is nn.Linear
    assert m.weight is weight
    assert bool(lucid.equal(weight, values))
    assert _names(m)[0] == ["weight", "bias"]


def test_a_missing_name_is_refused() -> None:
    with pytest.raises(ValueError, match="does not have a parameter"):
        parametrize.register_parametrization(nn.Linear(2, 2), "nope", _Scale())


def test_a_right_inverse_sets_the_original_so_the_read_is_unchanged() -> None:
    m = nn.Linear(4, 4)
    values = m.weight.detach().clone()
    parametrize.register_parametrization(m, "weight", _Halved())
    assert bool(lucid.allclose(m.weight, values))
    assert bool(lucid.allclose(m.parametrizations.weight.original, values / 2.0))


def test_deepcopy_copies_and_pickle_is_refused() -> None:
    m = nn.Linear(4, 4)
    parametrize.register_parametrization(m, "weight", _Symmetric())
    twin = copy.deepcopy(m)
    assert type(twin) is type(m)
    assert bool(lucid.equal(twin.weight, m.weight))
    assert twin.parametrizations.weight.original is not (m.parametrizations.weight.original)
    with pytest.raises(RuntimeError, match="state_dict"):
        pickle.dumps(m)


def test_each_module_gets_its_own_class() -> None:
    a, b = nn.Linear(2, 2), nn.Linear(2, 2)
    parametrize.register_parametrization(a, "weight", _Scale())
    parametrize.register_parametrization(b, "weight", _Scale())
    assert type(a) is not type(b)
    assert isinstance(a, nn.Linear)
    # The shared class carries no property: a fresh Linear is untouched.
    fresh = nn.Linear(2, 2)
    assert isinstance(fresh.weight, nn.Parameter)
    assert parametrize.type_before_parametrizations(fresh) is nn.Linear


def test_the_parametrization_follows_the_module_training_mode() -> None:
    m = nn.Linear(2, 2).eval()
    step = _Scale()
    parametrize.register_parametrization(m, "weight", step)
    assert not step.training
    m.train()
    assert step.training
