"""``functional_call`` and ``stack_module_state``.

A module run with the parameters and buffers of a dict has to give what a
module holding them gives, leave the module as it was — also when its
forward raises — treat tied weights the way the reference framework does,
and be differentiable and batchable in those tensors.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.func as func
import lucid.nn as nn
from lucid import Tensor
from lucid.nn._swap import TensorSwap


def _close(got: object, want: object, tol: float = 1e-5) -> None:
    assert isinstance(got, Tensor) and isinstance(want, Tensor)
    assert tuple(got.shape) == tuple(want.shape), (tuple(got.shape), tuple(want.shape))
    assert float((got - want).abs().max().item()) < tol, (got, want)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())


class Tied(nn.Module):
    """``a`` and ``b`` share one weight."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 2, bias=False)
        self.b = nn.Linear(2, 2, bias=False)
        self.b.weight = self.a.weight

    def forward(self, x: Tensor) -> Tensor:
        return self.a(x) + self.b(x) * 10.0


class Boom(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        self.fc(x)
        raise ZeroDivisionError("forward failed")


def _batch(device: str, rows: int = 5, cols: int = 3) -> Tensor:
    return lucid.arange(float(rows * cols), device=device).reshape(rows, cols) / 7.0


def _fill(model: nn.Module, device: str, scale: float) -> dict[str, Tensor]:
    """Fresh tensors for every parameter of ``model``: ``scale`` times a ramp."""
    return {
        name: (lucid.arange(float(p.numel()), device=device).reshape(p.shape) * scale) / p.numel()
        for name, p in model.named_parameters()
    }


def _mlp(device: str) -> MLP:
    return MLP().to(device)


# ── what it computes ─────────────────────────────────────────────────────────


def test_functional_call_matches_the_closed_form_of_a_two_layer_mlp(device: str) -> None:
    model = _mlp(device)
    params = _fill(model, device, 1.0)
    x = _batch(device)
    got = func.functional_call(model, params, (x,))
    hidden = (x @ params["fc1.weight"].T + params["fc1.bias"]).relu()
    _close(got, hidden @ params["fc2.weight"].T + params["fc2.bias"])


def test_functional_call_gives_what_a_module_holding_the_tensors_gives(device: str) -> None:
    model, twin = _mlp(device), _mlp(device)
    params = _fill(model, device, 2.0)
    twin.load_state_dict(params)
    x = _batch(device)
    _close(func.functional_call(model, params, (x,)), twin(x))


def test_functional_call_leaves_the_module_as_it_was(device: str) -> None:
    model = _mlp(device)
    x = _batch(device)
    before = {name: p for name, p in model.named_parameters()}
    values = {name: p.tolist() for name, p in before.items()}
    wanted = model(x)
    func.functional_call(model, _fill(model, device, 3.0), (x,))
    after = dict(model.named_parameters())
    assert list(after) == list(before)
    for name, p in after.items():
        assert p is before[name]
        assert p.tolist() == values[name]
    _close(model(x), wanted)


def test_functional_call_restores_the_module_when_forward_raises(device: str) -> None:
    model = Boom().to(device)
    original = {name: p for name, p in model.named_parameters()}
    values = {name: p.tolist() for name, p in original.items()}
    replacement = {name: lucid.zeros_like(p) for name, p in original.items()}
    with pytest.raises(ZeroDivisionError):
        func.functional_call(model, replacement, (lucid.ones(1, 2, device=device),))
    for name, p in model.named_parameters():
        assert p is original[name] and p.tolist() == values[name]


def test_the_swap_is_undone_when_the_tensors_do_not_fit(device: str) -> None:
    model = _mlp(device)
    original = dict(model.named_parameters())
    wrong = {"fc1.weight": lucid.ones(5, 5, device=device)}
    with pytest.raises(Exception):  # noqa: B017, PT011 - the engine's own shape error
        func.functional_call(model, wrong, (_batch(device),))
    assert all(p is original[name] for name, p in model.named_parameters())


@pytest.mark.parametrize("how", ["tuple", "single", "keyword"])
def test_the_arguments_are_unpacked_as_a_tuple_or_taken_as_one(device: str, how: str) -> None:
    model = nn.Linear(3, 2).to(device)
    w, b = lucid.ones(2, 3, device=device), lucid.zeros(2, device=device)
    x = _batch(device)
    params = {"weight": w, "bias": b}
    calls: dict[str, Callable[[], object]] = {
        "tuple": lambda: func.functional_call(model, params, (x,)),
        "single": lambda: func.functional_call(model, params, x),
        "keyword": lambda: func.functional_call(model, params, (), {"input": x}),
    }
    _close(calls[how](), x @ w.T + b)


def test_a_module_without_arguments(device: str) -> None:
    class Constant(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("value", lucid.zeros(2))

        def forward(self) -> Tensor:
            return self.value * 2.0

    m = Constant().to(device)
    got = func.functional_call(m, {"value": lucid.ones(2, device=device)})
    _close(got, lucid.full((2,), 2.0, device=device))


def test_dicts_in_a_sequence_are_merged(device: str) -> None:
    model = nn.Linear(3, 2).to(device)
    w, b = lucid.ones(2, 3, device=device), lucid.full((2,), 5.0, device=device)
    x = _batch(device)
    _close(func.functional_call(model, ({"weight": w}, {"bias": b}), (x,)), x @ w.T + b)


def test_a_name_in_two_dicts_is_ambiguous(device: str) -> None:
    model = nn.Linear(3, 2).to(device)
    w = lucid.ones(2, 3, device=device)
    with pytest.raises(ValueError, match=r"\['weight'\] appeared in multiple dictionaries"):
        func.functional_call(model, ({"weight": w}, {"weight": w}), (_batch(device),))


@pytest.mark.parametrize("value", [3.0, [1.0], None, "w"], ids=str)
def test_a_value_that_is_not_a_tensor_is_refused(device: str, value: object) -> None:
    model = nn.Linear(3, 2).to(device)
    with pytest.raises(TypeError, match="is not an instance of Tensor"):
        func.functional_call(model, {"weight": value}, (_batch(device),))  # type: ignore[dict-item]


# ── names: missing, unexpected, strict ──────────────────────────────────────


def test_a_name_left_out_keeps_the_modules_own_tensor(device: str) -> None:
    model = nn.Linear(3, 2).to(device)
    w = lucid.ones(2, 3, device=device)
    x = _batch(device)
    _close(func.functional_call(model, {"weight": w}, (x,)), x @ w.T + model.bias)


def test_a_name_the_module_does_not_have_is_ignored_unless_strict(device: str) -> None:
    model = nn.Linear(3, 2).to(device)
    given = {"weight": lucid.ones(2, 3, device=device), "bias": lucid.zeros(2, device=device)}
    x = _batch(device)
    want = func.functional_call(model, given, (x,))
    _close(func.functional_call(model, {**given, "nope": given["bias"]}, (x,)), want)
    _close(func.functional_call(model, {**given, "fc.nope": given["bias"]}, (x,)), want)


_STRICT: dict[str, tuple[Callable[[Tensor, Tensor], dict[str, Tensor]], str]] = {
    "missing": (lambda w, b: {"weight": w}, r"Missing key\(s\): 'bias'\."),
    "unexpected": (
        lambda w, b: {"weight": w, "bias": b, "extra": b},
        r"Unexpected key\(s\): 'extra'\.",
    ),
    "both": (
        lambda w, b: {"weight": w, "extra": b},
        r"Unexpected key\(s\): 'extra'\.\n\tMissing key\(s\): 'bias'\.",
    ),
}


@pytest.mark.parametrize("case", list(_STRICT))
def test_strict_requires_exactly_the_names_of_the_module(device: str, case: str) -> None:
    model = nn.Linear(3, 2).to(device)
    build, message = _STRICT[case]
    given = build(lucid.ones(2, 3, device=device), lucid.zeros(2, device=device))
    with pytest.raises(RuntimeError, match=message):
        func.functional_call(model, given, (_batch(device),), strict=True)
    func.functional_call(model, given, (_batch(device),))  # not strict: fine


def test_strict_accepts_exactly_the_names_of_the_module(device: str) -> None:
    model = _mlp(device)
    params = _fill(model, device, 1.0)
    func.functional_call(model, params, (_batch(device),), strict=True)


# ── tied weights ─────────────────────────────────────────────────────────────


def _tied(device: str) -> tuple[Tied, Tensor]:
    return Tied().to(device), lucid.arange(4.0, device=device).reshape(2, 2) / 3.0 + 1.0


def test_a_tied_weight_given_once_replaces_every_name_it_is_tied_to(device: str) -> None:
    model, x = _tied(device)
    w = lucid.full((2, 2), 2.0, device=device)
    _close(func.functional_call(model, {"a.weight": w}, (x,)), x @ w.T * 11.0)


def test_without_tie_weights_only_the_name_given_is_replaced(device: str) -> None:
    model, x = _tied(device)
    w = lucid.full((2, 2), 2.0, device=device)
    got = func.functional_call(model, {"a.weight": w}, (x,), tie_weights=False)
    _close(got, x @ w.T + (x @ model.b.weight.T) * 10.0)


def test_different_tensors_for_tied_names_are_refused(device: str) -> None:
    model, x = _tied(device)
    w2, w3 = lucid.full((2, 2), 2.0, device=device), lucid.full((2, 2), 3.0, device=device)
    with pytest.raises(ValueError, match=r"multiple values for keys \['a.weight', 'b.weight'\]"):
        func.functional_call(model, {"a.weight": w2, "b.weight": w3}, (x,))


def test_the_same_tensor_for_tied_names_is_fine(device: str) -> None:
    model, x = _tied(device)
    w = lucid.full((2, 2), 2.0, device=device)
    _close(func.functional_call(model, {"a.weight": w, "b.weight": w}, (x,)), x @ w.T * 11.0)


def test_tied_names_may_differ_when_the_weights_are_not_tied(device: str) -> None:
    model, x = _tied(device)
    w2, w3 = lucid.full((2, 2), 2.0, device=device), lucid.full((2, 2), 3.0, device=device)
    got = func.functional_call(model, {"a.weight": w2, "b.weight": w3}, (x,), tie_weights=False)
    _close(got, x @ w2.T + (x @ w3.T) * 10.0)


def test_strict_counts_a_tied_name_as_named_only_when_weights_are_tied(device: str) -> None:
    model, x = _tied(device)
    w = lucid.full((2, 2), 2.0, device=device)
    func.functional_call(model, {"a.weight": w}, (x,), strict=True)
    with pytest.raises(RuntimeError, match=r"Missing key\(s\): 'b.weight'\."):
        func.functional_call(model, {"a.weight": w}, (x,), strict=True, tie_weights=False)


def test_tied_weights_are_still_tied_afterwards(device: str) -> None:
    model, x = _tied(device)
    func.functional_call(model, {"a.weight": lucid.ones(2, 2, device=device)}, (x,))
    assert model.a.weight is model.b.weight


def test_a_module_used_under_two_names_is_swapped_once(device: str) -> None:
    class Twice(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(2, 2, bias=False)
            self.again = self.shared

        def forward(self, x: Tensor) -> Tensor:
            return self.shared(x) + self.again(x)

    model = Twice().to(device)
    original = model.shared.weight
    w = lucid.full((2, 2), 2.0, device=device)
    x = _batch(device, 3, 2)
    _close(func.functional_call(model, {"shared.weight": w}, (x,)), x @ w.T * 2.0)
    assert model.shared.weight is original


# ── buffers ──────────────────────────────────────────────────────────────────


def test_buffers_are_replaced_like_parameters(device: str) -> None:
    bn = nn.BatchNorm1d(3).to(device).eval()
    mean = lucid.full((3,), 2.0, device=device)
    var = lucid.full((3,), 4.0, device=device)
    x = _batch(device)
    got = func.functional_call(bn, {"running_mean": mean, "running_var": var}, (x,))
    _close(got, (x - 2.0) / (4.0 + bn.eps) ** 0.5, tol=1e-4)
    assert float(bn.running_mean.abs().sum().item()) == 0.0


def test_a_running_statistic_is_updated_in_the_tensor_that_was_given(device: str) -> None:
    bn = nn.BatchNorm1d(3, momentum=0.5).to(device).train()
    mean = lucid.zeros(3, device=device)
    var = lucid.ones(3, device=device)
    x = _batch(device)
    func.functional_call(bn, {"running_mean": mean, "running_var": var}, (x,))
    _close(mean, x.mean(dim=0) * 0.5, tol=1e-5)
    assert float(bn.running_mean.abs().sum().item()) == 0.0


# ── differentiable and batchable in the tensors ──────────────────────────────


def test_grad_with_respect_to_the_parameters_is_the_closed_form(device: str) -> None:
    model = nn.Linear(3, 2).to(device)
    x = _batch(device)
    params = {"weight": lucid.ones(2, 3, device=device), "bias": lucid.zeros(2, device=device)}
    grads = func.grad(lambda p: func.functional_call(model, p, (x,)).sum())(params)  # type: ignore[union-attr]
    assert isinstance(grads, dict)
    _close(grads["weight"], x.sum(dim=0).reshape(1, 3).expand(2, 3))
    _close(grads["bias"], lucid.full((2,), float(x.shape[0]), device=device))


def test_grad_leaves_the_modules_own_parameters_untouched(device: str) -> None:
    model = nn.Linear(3, 2).to(device)
    func.grad(lambda p: func.functional_call(model, p, (_batch(device),)).sum())(  # type: ignore[union-attr]
        {"weight": model.weight.detach(), "bias": model.bias.detach()}
    )
    assert model.weight.grad is None and model.bias.grad is None


def test_the_output_is_attached_to_the_graph_of_the_given_tensors(device: str) -> None:
    model = nn.Linear(3, 2).to(device)
    w = lucid.ones(2, 3, device=device).requires_grad_(True)
    out = func.functional_call(model, {"weight": w}, (_batch(device),))
    assert isinstance(out, Tensor)
    out.sum().backward()
    assert w.grad is not None and model.weight.grad is None


def test_an_ensemble_by_vmap_over_stacked_state_equals_the_models_one_by_one(device: str) -> None:
    models = [_mlp(device) for _ in range(3)]
    params, buffers = func.stack_module_state(models)
    x = _batch(device)
    got = func.vmap(
        lambda p, b, inp: func.functional_call(models[0], (p, b), (inp,)), in_dims=(0, 0, None)
    )(params, buffers, x)
    _close(got, lucid.stack([m(x) for m in models]), tol=1e-4)


def test_per_sample_gradients_of_a_module(device: str) -> None:
    model = nn.Linear(3, 1).to(device)
    params = {"weight": model.weight.detach(), "bias": model.bias.detach()}
    x = _batch(device)

    def loss(p: dict[str, Tensor], row: Tensor) -> Tensor:
        out = func.functional_call(model, p, (row.reshape(1, 3),))
        assert isinstance(out, Tensor)
        return (out**2).sum()

    got = func.vmap(func.grad(loss), in_dims=(None, 0))(params, x)
    assert isinstance(got, dict)
    out = x @ params["weight"].T + params["bias"]
    _close(got["weight"], (2.0 * out).reshape(-1, 1, 1) * x.reshape(-1, 1, 3), tol=1e-4)
    _close(got["bias"], 2.0 * out)


# ── TensorSwap on its own ────────────────────────────────────────────────────


def test_the_swap_is_a_context_manager_that_restores_on_exit(device: str) -> None:
    model = nn.Linear(2, 2).to(device)
    original = model.weight
    w = lucid.ones(2, 2, device=device)
    with TensorSwap(model, {"weight": w}):
        assert model.weight._impl is w._impl
    assert model.weight is original


def test_the_swap_restores_after_an_exception_in_the_block(device: str) -> None:
    model = nn.Linear(2, 2).to(device)
    original = model.weight
    with pytest.raises(KeyError):
        with TensorSwap(model, {"weight": lucid.ones(2, 2, device=device)}):
            raise KeyError("inside")
    assert model.weight is original


def test_a_swap_that_fails_half_way_puts_back_what_it_had_changed(device: str) -> None:
    model = nn.Linear(2, 2).to(device)
    weight, bias = model.weight, model.bias
    broken = {"weight": lucid.ones(2, 2, device=device), "bias": "not a tensor"}
    with pytest.raises(AttributeError):
        with TensorSwap(model, broken):  # type: ignore[arg-type]
            raise AssertionError("not reached")
    assert model.weight is weight and model.bias is bias


def test_swaps_nest_and_unwind_in_order(device: str) -> None:
    model = nn.Linear(2, 2).to(device)
    original = model.weight
    one, two = lucid.ones(2, 2, device=device), lucid.zeros(2, 2, device=device)
    with TensorSwap(model, {"weight": one}):
        with TensorSwap(model, {"weight": two}):
            assert model.weight._impl is two._impl
        assert model.weight._impl is one._impl
    assert model.weight is original


def test_the_swap_changes_the_parameters_only_while_it_is_in(device: str) -> None:
    model = nn.Linear(2, 2).to(device)
    w = lucid.ones(2, 2, device=device)
    swap = TensorSwap(model, {"weight": w})
    assert model.weight._impl is not w._impl  # nothing happens until it is entered
    with swap:
        assert model.weight._impl is w._impl


def test_a_swap_of_a_nested_module_reaches_its_slot(device: str) -> None:
    model = _mlp(device)
    w = lucid.zeros(2, 4, device=device)
    with TensorSwap(model, {"fc2.weight": w}):
        assert model.fc2.weight._impl is w._impl


# ── stack_module_state ───────────────────────────────────────────────────────


def test_stack_module_state_stacks_every_parameter_along_a_new_first_axis(device: str) -> None:
    models = [_mlp(device) for _ in range(4)]
    params, buffers = func.stack_module_state(models)
    assert list(params) == [name for name, _ in models[0].named_parameters()]
    assert buffers == {}
    for name, stacked in params.items():
        assert tuple(stacked.shape) == (4, *dict(models[0].named_parameters())[name].shape)
        for i, m in enumerate(models):
            _close(stacked[i], dict(m.named_parameters())[name])


def test_stacked_parameters_are_fresh_leaves_that_require_grad(device: str) -> None:
    models = [nn.Linear(2, 2).to(device) for _ in range(2)]
    params, _ = func.stack_module_state(models)
    for stacked in params.values():
        assert stacked.requires_grad and stacked._impl.is_leaf


def test_stack_module_state_stacks_the_buffers_too(device: str) -> None:
    models = [nn.BatchNorm1d(3).to(device) for _ in range(2)]
    models[1].running_mean.copy_(lucid.ones(3, device=device))
    params, buffers = func.stack_module_state(models)
    assert set(buffers) == {"running_mean", "running_var", "num_batches_tracked"}
    assert tuple(buffers["running_mean"].shape) == (2, 3)
    assert buffers["running_mean"].tolist() == [[0.0] * 3, [1.0] * 3]
    assert not buffers["running_mean"].requires_grad
    assert set(params) == {"weight", "bias"}


def test_stack_module_state_of_one_model(device: str) -> None:
    (params, _) = func.stack_module_state([nn.Linear(2, 3).to(device)])
    assert tuple(params["weight"].shape) == (1, 3, 2)


def test_stack_module_state_does_not_change_the_models(device: str) -> None:
    models = [nn.Linear(2, 2).to(device) for _ in range(2)]
    originals = [m.weight for m in models]
    func.stack_module_state(models)
    assert all(m.weight is o for m, o in zip(models, originals, strict=True))


_STACK_REFUSED: dict[str, Callable[[str], list[nn.Module]]] = {
    "empty": lambda d: [],
    "classes": lambda d: [nn.Linear(2, 2).to(d), nn.Identity()],
    "modes": lambda d: [nn.Linear(2, 2).to(d), nn.Linear(2, 2).to(d).eval()],
}


@pytest.mark.parametrize("case", list(_STACK_REFUSED))
def test_stack_module_state_refuses_models_that_do_not_agree(device: str, case: str) -> None:
    with pytest.raises(RuntimeError, match="stack_module_state: Expected"):
        func.stack_module_state(_STACK_REFUSED[case](device))


def test_stack_module_state_refuses_models_that_disagree_on_requires_grad(device: str) -> None:
    a, b = nn.Linear(2, 2).to(device), nn.Linear(2, 2).to(device)
    b.weight.requires_grad_(False)
    with pytest.raises(
        RuntimeError, match="weight from each model to have the same .requires_grad"
    ):
        func.stack_module_state([a, b])


# ── the reference framework ──────────────────────────────────────────────────


def _linear_pair(m: ModuleType) -> tuple[object, dict[str, object]]:
    layer = m.nn.Linear(3, 2)
    params = {"weight": m.ones(2, 3), "bias": m.full((2,), 0.5)}
    return layer, params


def _call_cases(m: ModuleType) -> dict[str, Callable[[], object]]:
    x = (m.arange(15.0).reshape(5, 3)) / 7.0
    layer, params = _linear_pair(m)
    tied = m.nn.Module()
    tied.a = m.nn.Linear(2, 2, bias=False)
    tied.b = m.nn.Linear(2, 2, bias=False)
    tied.b.weight = tied.a.weight
    tied.forward = lambda t: tied.a(t) + tied.b(t) * 10.0  # type: ignore[method-assign]
    t = m.full((2, 2), 2.0)
    xt = m.arange(4.0).reshape(2, 2) / 3.0 + 1.0
    return {
        "linear": lambda: m.func.functional_call(layer, params, (x,)),
        "single-argument": lambda: m.func.functional_call(layer, params, x),
        "partial": lambda: m.func.functional_call(layer, {"weight": params["weight"]}, (x,)).shape,
        "merged": lambda: m.func.functional_call(
            layer, ({"weight": params["weight"]}, {"bias": params["bias"]}), (x,)
        ),
        "tied": lambda: m.func.functional_call(tied, {"a.weight": t}, (xt,)),
        "untied": lambda: m.func.functional_call(
            tied, {"a.weight": t, "b.weight": t * 2.0}, (xt,), tie_weights=False
        ),
    }


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_call_cases(lucid)))
def test_functional_call_matches_the_reference(ref: ModuleType, name: str) -> None:
    got, want = _call_cases(lucid)[name](), _call_cases(ref)[name]()
    if isinstance(want, tuple):
        assert tuple(got) == tuple(want)  # type: ignore[arg-type]
    else:
        assert got.shape == want.shape  # type: ignore[attr-defined]


def _refusals(m: ModuleType) -> dict[str, Callable[[], object]]:
    layer, params = _linear_pair(m)
    x = m.ones(5, 3)
    tied = m.nn.Module()
    tied.a = m.nn.Linear(2, 2, bias=False)
    tied.b = m.nn.Linear(2, 2, bias=False)
    tied.b.weight = tied.a.weight
    w = m.ones(2, 2)
    return {
        "duplicate": lambda: m.func.functional_call(layer, (params, params), (x,)),
        "not-a-tensor": lambda: m.func.functional_call(layer, {"weight": 3.0}, (x,)),
        "strict-missing": lambda: m.func.functional_call(
            layer, {"weight": params["weight"]}, (x,), strict=True
        ),
        "strict-unexpected": lambda: m.func.functional_call(
            layer, {**params, "extra": params["bias"]}, (x,), strict=True
        ),
        "tied-conflict": lambda: m.func.functional_call(
            tied, {"a.weight": w, "b.weight": w * 2.0}, (w,)
        ),
        "stack-empty": lambda: m.func.stack_module_state([]),
        "stack-classes": lambda: m.func.stack_module_state([layer, m.nn.Identity()]),
    }


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_refusals(lucid)))
def test_functional_call_refuses_what_the_reference_refuses_with_the_same_class(
    ref: ModuleType, name: str
) -> None:
    classes = []
    for m in (lucid, ref):
        with pytest.raises(Exception) as error:  # noqa: PT011 - the class is what is compared
            _refusals(m)[name]()
        classes.append(error.type)
    assert issubclass(classes[0], classes[1]), classes


@pytest.mark.parity
def test_stack_module_state_matches_the_reference(ref: ModuleType) -> None:
    got = func.stack_module_state([nn.BatchNorm1d(3), nn.BatchNorm1d(3)])
    want = ref.func.stack_module_state([ref.nn.BatchNorm1d(3), ref.nn.BatchNorm1d(3)])
    for g, w in zip(got, want, strict=True):
        assert list(g) == list(w)
        assert {k: tuple(v.shape) for k, v in g.items()} == {
            k: tuple(v.shape) for k, v in w.items()
        }
        assert {k: v.requires_grad for k, v in g.items()} == {
            k: v.requires_grad for k, v in w.items()
        }
