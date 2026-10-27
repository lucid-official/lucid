"""Every ``inplace=`` in ``nn.functional`` and ``nn`` writes ``x`` (LCD-235).

The activations and dropouts accepted ``inplace=True`` and returned a fresh
tensor, leaving ``x`` as it was — ``F.relu(x, inplace=True)``,
``nn.ReLU(inplace=True)`` and eleven more.  They now share one route
(``lucid.nn.functional._inplace._in_place``), and these tests hold every
function and module that takes the flag to the reference's in-place
semantics: ``x`` itself is returned and holds the out-of-place result; a
leaf that requires grad is refused while autograd records; the gradient
through a non-leaf is the op's own; and a view's base sees the write on
both devices — on Metal, where a view is a value of its own, too.

The completeness checks make a new ``inplace=`` parameter fail here until
it is added to the tables.
"""

import inspect
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

# name -> (call given the functional namespace, input shape).  The random
# ones run at ``p=1`` in training (every element dropped: deterministic, and
# still a write) or in eval (``rrelu``'s midpoint slope).
_Call = Callable[[ModuleType, object, bool], object]
_FUNCTIONALS: dict[str, tuple[_Call, tuple[int, ...]]] = {
    "relu": (lambda f, x, ip: f.relu(x, inplace=ip), (2, 3, 4)),
    "leaky_relu": (lambda f, x, ip: f.leaky_relu(x, 0.2, inplace=ip), (2, 3, 4)),
    "elu": (lambda f, x, ip: f.elu(x, 0.7, inplace=ip), (2, 3, 4)),
    "selu": (lambda f, x, ip: f.selu(x, inplace=ip), (2, 3, 4)),
    "silu": (lambda f, x, ip: f.silu(x, inplace=ip), (2, 3, 4)),
    "mish": (lambda f, x, ip: f.mish(x, inplace=ip), (2, 3, 4)),
    "hardswish": (lambda f, x, ip: f.hardswish(x, inplace=ip), (2, 3, 4)),
    "hardsigmoid": (lambda f, x, ip: f.hardsigmoid(x, inplace=ip), (2, 3, 4)),
    "relu6": (lambda f, x, ip: f.relu6(x, inplace=ip), (2, 3, 4)),
    "celu": (lambda f, x, ip: f.celu(x, 0.5, inplace=ip), (2, 3, 4)),
    "hardtanh": (lambda f, x, ip: f.hardtanh(x, -0.5, 0.25, inplace=ip), (2, 3, 4)),
    "threshold": (lambda f, x, ip: f.threshold(x, 0.1, -3.0, inplace=ip), (2, 3, 4)),
    "rrelu": (lambda f, x, ip: f.rrelu(x, 0.1, 0.3, False, inplace=ip), (2, 3, 4)),
    "dropout": (lambda f, x, ip: f.dropout(x, 1.0, True, inplace=ip), (2, 3, 4)),
    "dropout1d": (lambda f, x, ip: f.dropout1d(x, 1.0, True, inplace=ip), (2, 3, 4)),
    "dropout2d": (
        lambda f, x, ip: f.dropout2d(x, 1.0, True, inplace=ip),
        (2, 3, 2, 2),
    ),
    "dropout3d": (
        lambda f, x, ip: f.dropout3d(x, 1.0, True, inplace=ip),
        (1, 2, 2, 2, 2),
    ),
    "alpha_dropout": (
        lambda f, x, ip: f.alpha_dropout(x, 1.0, True, inplace=ip),
        (2, 3, 4),
    ),
    "feature_alpha_dropout": (
        lambda f, x, ip: f.feature_alpha_dropout(x, 1.0, True, inplace=ip),
        (2, 3, 4),
    ),
}

# Every module with an ``inplace`` constructor argument -> its other arguments.
_MODULES: dict[str, dict[str, float]] = {
    "ReLU": {},
    "LeakyReLU": {"negative_slope": 0.2},
    "ELU": {"alpha": 0.7},
    "SELU": {},
    "SiLU": {},
    "Mish": {},
    "Hardswish": {},
    "Hardsigmoid": {},
    "ReLU6": {},
    "CELU": {"alpha": 0.5},
    "Hardtanh": {"min_val": -0.5, "max_val": 0.25},
    "Threshold": {"threshold": 0.1, "value": -3.0},
    "RReLU": {"lower": 0.1, "upper": 0.3},
    "Dropout": {"p": 1.0},
    "Dropout1d": {"p": 1.0},
    "Dropout2d": {"p": 1.0},
    "Dropout3d": {"p": 1.0},
    "AlphaDropout": {"p": 1.0},
    "FeatureAlphaDropout": {"p": 1.0},
}


def _values(shape: tuple[int, ...]) -> list[float]:
    """Both signs, both sides of every threshold above, and no exact ties."""
    n = 1
    for d in shape:
        n *= d
    return [(-1.0) ** i * (0.05 + 0.37 * i) for i in range(n)]


def _nested(shape: tuple[int, ...]) -> object:
    flat: list[object] = list(_values(shape))
    for d in reversed(shape[1:]):
        flat = [flat[i : i + d] for i in range(0, len(flat), d)]
    return flat


def _input(shape: tuple[int, ...], device: str, **kw: bool) -> lucid.Tensor:
    """A leaf of ``shape`` — built nested, since a reshape would not be one."""
    return lucid.tensor(_nested(shape), device=device, **kw)


def _close(a: object, b: object) -> bool:
    flat_a = lucid.tensor(a).reshape(-1).tolist()
    flat_b = lucid.tensor(b).reshape(-1).tolist()
    return all(abs(p - q) < 1e-5 for p, q in zip(flat_a, flat_b, strict=True))


# ── the tables are the whole surface ─────────────────────────────────────────


def test_every_functional_with_inplace_is_covered() -> None:
    taking = {
        name
        for name, fn in vars(F).items()
        if callable(fn)
        and not name.startswith("_")
        and not name.endswith("_")
        and "inplace" in _parameters(fn)
    }
    assert taking == set(_FUNCTIONALS)


def test_every_module_with_inplace_is_covered() -> None:
    taking = {
        name
        for name, cls in vars(nn).items()
        if isinstance(cls, type)
        and issubclass(cls, nn.Module)
        and "inplace" in _parameters(cls.__init__)
    }
    assert taking == set(_MODULES)


def _parameters(fn: Callable[..., object]) -> set[str]:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):  # a builtin without a signature takes no flag
        return set()


# ── the functionals ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("name", list(_FUNCTIONALS))
def test_result_is_x_exactly_when_inplace(name: str, inplace: bool, device: str) -> None:
    call, shape = _FUNCTIONALS[name]
    want = call(F, _input(shape, device), False)
    x = _input(shape, device)
    out = call(F, x, inplace)
    assert _close(out, want)
    if inplace:
        assert out is x
    else:
        assert out is not x
        assert _close(x, _input(shape, device)), "out-of-place must not write x"


@pytest.mark.parametrize("name", list(_FUNCTIONALS))
def test_a_leaf_that_requires_grad_is_refused(name: str, device: str) -> None:
    call, shape = _FUNCTIONALS[name]
    x = _input(shape, device, requires_grad=True)
    with pytest.raises(RuntimeError, match="leaf"):
        call(F, x, True)
    assert _close(x.detach(), _input(shape, device)), "a refused write changed x"
    # Outside autograd the leaf is written and stays a trainable leaf.
    with lucid.no_grad():
        assert call(F, x, True) is x
    assert x.is_leaf and x.requires_grad


@pytest.mark.parametrize("name", list(_FUNCTIONALS))
def test_a_view_of_a_leaf_that_requires_grad_is_refused(name: str, device: str) -> None:
    call, shape = _FUNCTIONALS[name]
    base = _input(shape, device, requires_grad=True)
    with pytest.raises(RuntimeError, match="leaf"):
        call(F, base[0], True)
    assert _close(base.detach(), _input(shape, device))


@pytest.mark.parametrize("name", list(_FUNCTIONALS))
def test_a_views_base_sees_the_write(name: str, device: str) -> None:
    call, shape = _FUNCTIONALS[name]
    whole = call(F, _input(shape, device), False)

    # Every element through a rearranging view.
    base = _input(shape, device)
    call(F, base.transpose(0, 1), True)
    assert _close(base, whole)

    # Part of the base through an indexing view; the rest is untouched.
    base = _input(shape, device)
    call(F, base[0], True)
    assert _close(base[0], call(F, _input(shape, device)[0], False))
    assert _close(base[1:], _input(shape, device)[1:])


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_FUNCTIONALS))
def test_gradient_through_a_non_leaf_matches_the_reference(
    name: str, device: str, ref: ModuleType
) -> None:
    call, shape = _FUNCTIONALS[name]
    weights = [0.5 + 0.25 * i for i in range(len(_values(shape)))]

    a = _input(shape, device, requires_grad=True)
    hidden = a * 1.5
    assert call(F, hidden, True) is hidden
    (hidden * lucid.tensor(weights, device=device).reshape(shape)).sum().backward()

    ra = ref.tensor(_values(shape), requires_grad=True)
    ref_hidden = ra.reshape(shape) * 1.5
    call(ref.nn.functional, ref_hidden, True)
    (ref_hidden * ref.tensor(weights).reshape(shape)).sum().backward()

    assert _close(hidden.detach(), ref_hidden.detach().tolist())
    assert _close(a.grad, ra.grad.reshape(shape).tolist())


# ── the modules ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("name", list(_MODULES))
def test_module_passes_inplace_through(name: str, inplace: bool, device: str) -> None:
    module = getattr(nn, name)(**_MODULES[name], inplace=inplace)
    # Dropout at p=1 drops everything only in training; RReLU's slope is
    # deterministic only in eval.
    module.train(name != "RReLU")
    assert module.inplace is inplace
    shape = (1, 2, 2, 2, 2) if name == "Dropout3d" else (2, 3, 2, 2)
    want = module(_input(shape, device))
    x = _input(shape, device)
    out = module(x)
    assert _close(out, want)
    assert (out is x) is inplace
