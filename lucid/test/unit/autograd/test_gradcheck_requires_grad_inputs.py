"""``gradcheck`` differentiates only the inputs that require grad.

It made every input a leaf requiring grad and differentiated it, so an
index among the inputs raised "only Tensors of floating point and complex
dtype can require gradients", and a constant had its gradient checked as if
the caller had asked for it (LCD-486).  The reference treats an input
without ``requires_grad`` as a constant, passes it through unchanged, and
refuses a call where no input requires grad.  ``gradgradcheck`` reads the
inputs the same way.
"""

import pytest

import lucid
from lucid.autograd import gradcheck, gradgradcheck


def _x(lib):
    return lib.tensor([0.3, -1.2, 0.7], dtype=lib.float64, requires_grad=True)


_CASES = {
    "integer index": (
        lambda x, i: (x.gather(0, i) ** 3).sum(),
        lambda lib: lib.tensor([0, 2]),
    ),
    "float constant": (
        lambda x, w: (x * x * w).sum(),
        lambda lib: lib.tensor([2.0, -1.0, 0.5], dtype=lib.float64),
    ),
    "bool mask": (
        lambda x, m: (x * x * m).sum(),
        lambda lib: lib.tensor([True, False, True]),
    ),
}


@pytest.mark.parametrize("check", [gradcheck, gradgradcheck], ids=lambda f: f.__name__)
@pytest.mark.parametrize("case", list(_CASES))
def test_an_input_without_grad_is_a_constant(check, case):
    fn, constant = _CASES[case]
    c = constant(lucid)
    assert check(fn, [_x(lucid), c])
    assert not c.requires_grad and c.grad is None


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_CASES))
def test_the_reference_checks_the_same_calls(ref, case):
    fn, constant = _CASES[case]
    assert ref.autograd.gradcheck(fn, (_x(ref), constant(ref)))


class _WrongSecondGradient(lucid.autograd.Function):
    """``x * y`` with a deliberately wrong gradient for ``y``."""

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        return grad * y, grad * x * 0.0


def test_a_wrong_gradient_of_a_constant_is_not_checked():
    y = lucid.tensor([1.5, -2.0, 0.25], dtype=lucid.float64)
    assert gradcheck(lambda x, y: _WrongSecondGradient.apply(x, y).sum(), [_x(lucid), y])
    with pytest.raises(AssertionError, match="input 1"):
        gradcheck(
            lambda x, y: _WrongSecondGradient.apply(x, y).sum(),
            [_x(lucid), y.requires_grad_()],
        )


@pytest.mark.parametrize("check", [gradcheck, gradgradcheck], ids=lambda f: f.__name__)
def test_a_call_where_nothing_requires_grad_is_refused(check):
    with pytest.raises(ValueError, match="at least one input tensor to require gradient"):
        check(lambda x: (x * x).sum(), [_x(lucid).detach()])


@pytest.mark.parity
def test_the_reference_refuses_it_too(ref):
    with pytest.raises(ValueError, match="at least one input tensor to require gradient"):
        ref.autograd.gradcheck(lambda x: (x * x).sum(), (_x(ref).detach(),))
