"""Regression tests: the in-place activations dropped their own derivative.

Found 2026-08-02 by sweeping every differentiable op's analytic gradient
against central finite differences in float64.

All seven ``F.*_`` variants ended in ``x.copy_(f(x))``, and
:meth:`Tensor.copy_` documents that it does **not** extend the autograd
graph.  The values were right and the gradients were not: the returned
tensor kept whatever ``grad_fn`` ``x`` already carried, so the backward
pass behaved as though no activation had been applied.  ``F.elu_`` on
``[-1, 0.5, 2]`` returned ``[1, 1, 1]`` where ELU's derivative is
``[0.368, 1, 1]``.

Nothing inside Lucid called these at the time — the ``inplace=`` flag of
the out-of-place functions and of ``nn.ReLU`` was accepted and ignored — so
the reach was user code, silently.  Since LCD-235 the flag routes through
the same in-place path as these (``test_inplace_flag.py``).

The invariant asserted here is deliberately **not** a hand-derived
derivative: it is that the in-place form agrees with the out-of-place
form.  Writing the derivatives out by hand is how the first version of
this check "failed" ``hardtanh_`` at ``x = min_val``, where Lucid's
choice of one-sided convention is its own and is consistent between the
two spellings.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

# Spans both saturating regions and the exact clamp boundary, which is
# where a convention difference between the two spellings would show.
_PROBE = [-2.0, -1.0, -0.3, 0.0, 0.5, 2.0]

_PAIRS = [
    ("relu", F.relu, F.relu_),
    ("elu", F.elu, F.elu_),
    ("selu", F.selu, F.selu_),
    ("leaky_relu", F.leaky_relu, F.leaky_relu_),
    ("hardtanh", F.hardtanh, F.hardtanh_),
    ("threshold", lambda t: F.threshold(t, 0.0, 0.0), lambda t: F.threshold_(t, 0.0, 0.0)),
]


def _grad_through(fn):
    """Gradient reaching a leaf through ``fn`` applied to a non-leaf."""
    x = lucid.tensor(np.array(_PROBE, dtype=np.float64), dtype=lucid.float64)
    x.requires_grad_(True)
    hidden = x * 1.0
    fn(hidden).sum().backward()
    return x.grad.numpy()


def _value_of(fn):
    hidden = lucid.tensor(np.array(_PROBE, dtype=np.float64), dtype=lucid.float64) * 1.0
    out = fn(hidden)
    return out.numpy(), hidden.numpy(), out is hidden


@pytest.mark.parametrize("name,out_of_place,in_place", _PAIRS)
def test_in_place_gradient_matches_out_of_place(name, out_of_place, in_place):
    """The defect: this returned the pre-activation gradient for all seven."""
    assert np.allclose(_grad_through(out_of_place), _grad_through(in_place)), name


@pytest.mark.parametrize("name,out_of_place,in_place", _PAIRS)
def test_in_place_value_matches_out_of_place(name, out_of_place, in_place):
    got, mutated, aliased = _value_of(in_place)
    want, untouched, _ = _value_of(out_of_place)
    assert np.allclose(got, want), name
    # It is called in-place: the caller's tensor holds the result and is
    # the object returned.
    assert np.allclose(mutated, want), name
    assert aliased, name
    assert np.allclose(untouched, _PROBE), f"{name}: out-of-place must not mutate"


def test_gradient_is_not_the_identity() -> None:
    """Guard the instrument.

    Every check above compares two spellings.  If both were broken the
    same way they would agree and prove nothing, so pin one concrete
    derivative: ELU's negative branch is ``exp(x)``, which is neither 1
    nor 0 and so cannot be confused with a dropped or zeroed gradient.
    """
    got = _grad_through(F.elu_)
    want = np.where(np.array(_PROBE) > 0, 1.0, np.exp(_PROBE))
    assert np.allclose(got, want, atol=1e-9)
    assert not np.allclose(got, np.ones_like(got))


def test_rrelu_in_place_is_differentiable_in_eval() -> None:
    """``rrelu`` draws a slope in training, so only eval mode is comparable."""
    grad_out = _grad_through(lambda t: F.rrelu(t, training=False))
    grad_in = _grad_through(lambda t: F.rrelu_(t, training=False))
    assert np.allclose(grad_out, grad_in)
    assert not np.allclose(grad_in, np.ones_like(grad_in))


def test_relu_in_place_keeps_its_native_kernel_without_a_graph() -> None:
    """The no-graph branch must still mutate and still return the same object.

    ``relu_`` is the only one with a fused engine kernel, and it is kept
    for the case it exists for — inference, where there is no graph to
    extend and the temporary is pure waste.
    """
    x = lucid.tensor([-1.0, 0.5, 2.0])
    with lucid.no_grad():
        out = F.relu_(x)
    assert out is x
    assert np.allclose(x.numpy(), [0.0, 0.5, 2.0])


def test_module_form_gives_relus_derivative_either_way() -> None:
    """``nn.ReLU`` differentiates the same with and without ``inplace``.

    ``inplace=True`` was once ignored, which is why the defect above stayed
    invisible; now that it writes in place it must keep the derivative.
    """
    for inplace in (False, True):
        x = lucid.tensor(np.array(_PROBE, dtype=np.float64), dtype=lucid.float64)
        x.requires_grad_(True)
        hidden = x * 1.0
        nn.ReLU(inplace=inplace)(hidden).sum().backward()
        assert np.allclose(x.grad.numpy(), (np.array(_PROBE) > 0).astype(float))
