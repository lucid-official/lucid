"""A method that *is* a free function gets the free function's verdict.

``lucid._tensor._free_fn_methods.FREE_FN_METHODS`` are ``lucid.<name>``
with ``self`` first.  When LCD-547 added them, the audit's qualname-keyed
tables — where the reference is finite, why an op has no derivative —
had no row for any ``Tensor.<name>``, and CI went red on eight cells
whose answer was already on file under the free function: ``x.acosh()``
NaN below 1, ``x.heaviside(v)`` a step function.
"""

import inspect

import pytest

import lucid
from lucid._tensor._free_fn_methods import FREE_FN_METHODS
from lucid._tensor.tensor import Tensor
from lucid.test.audit import _surface
from lucid.test.audit._axes import Context, GradientAxis
from lucid.test.audit._axes_stability import StabilityAxis
from lucid.test.audit._result import Status


def _pair(name: str) -> tuple[_surface.Symbol, _surface.Symbol]:
    """The method and the free function, built directly rather than found
    by a walk of the whole surface."""
    method = _surface.Symbol(
        f"Tensor.{name}", "tensor", "method", inspect.getattr_static(Tensor, name)
    )
    free = _surface.Symbol(f"lucid.{name}", "lucid", "op", getattr(lucid, name))
    return method, free


@pytest.mark.parametrize("name", FREE_FN_METHODS)
def test_every_twin_is_filed_under_its_free_function(name: str) -> None:
    assert _surface.recorded_as(f"Tensor.{name}") == f"lucid.{name}"
    assert callable(getattr(lucid, name))


@pytest.mark.parametrize("qualname", ["Tensor.acos", "Tensor.exp_", "lucid.acosh"])
def test_a_spelling_with_knowledge_of_its_own_keeps_it(qualname: str) -> None:
    # ``Tensor.acos`` is a registry method, measured on its own.
    assert _surface.recorded_as(qualname) == qualname


@pytest.mark.parametrize(
    "axis", [StabilityAxis(), GradientAxis()], ids=["stability", "grad"]
)
@pytest.mark.parametrize("name", ["acosh", "logit", "gcd", "heaviside", "expm1"])
def test_a_twin_gets_the_free_functions_verdict(
    axis: StabilityAxis | GradientAxis, name: str
) -> None:
    method, free = _pair(name)
    ctx = Context(metal=False)
    got = axis.run(method, ctx)
    want = axis.run(free, ctx)
    assert (got.status, got.detail) == (want.status, want.detail)
    # Equal is not enough: ``lucid.gcd`` failed the gradient axis exactly
    # as its twin did while it truncated the float probe; it refuses one
    # now, and the axis files that as unsupported.
    assert got.status != Status.FAIL, got.detail
