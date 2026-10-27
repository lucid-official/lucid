"""A graph-mode comparison must observe imaginary gradients, not just real ones."""

import numpy as np
import pytest

import lucid
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Context, CreateGraphAxis
from lucid.test.audit._result import Status


def test_complex_contraction_is_real_and_uses_both_lanes() -> None:
    values = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    weights = np.array([0.5, -0.25])
    result = _probe.contract(lucid.tensor(values), weights)
    imaginary_weights = _probe.rng(_probe.SEED_A + 1).standard_normal(2)
    assert not result.is_complex()
    assert float(result) == pytest.approx(values.real @ weights + values.imag @ imaginary_weights)


@pytest.mark.parametrize("corrupt", [False, True])
def test_creategraph_observes_imaginary_gradient_errors(
    monkeypatch: pytest.MonkeyPatch,
    corrupt: bool,
) -> None:
    original = lucid.tensor([1 + 2j, 3 - 4j], dtype=lucid.complex128)
    call = _specs.Call([original])
    seen = []

    def identity(x: lucid.Tensor) -> lucid.Tensor:
        seen.append(x.numpy())
        return x

    if corrupt:
        true_grad = lucid.autograd.grad

        def bad_grad(*args: object, **kwargs: object) -> tuple[lucid.Tensor]:
            (g,) = true_grad(*args, **kwargs)
            return (lucid.complex(lucid.real(g), lucid.imag(g) + 1),)

        monkeypatch.setattr(lucid.autograd, "grad", bad_grad)
    axis = CreateGraphAxis()
    monkeypatch.setattr(axis, "_working_call", lambda *args: (call, "moderate", None))
    finding = axis.run(_surface.Symbol("lucid.identity", "lucid", "op", identity), Context())
    assert finding.status == (Status.FAIL if corrupt else Status.PASS), finding.detail
    for observed in seen:
        np.testing.assert_array_equal(observed, original.numpy())
