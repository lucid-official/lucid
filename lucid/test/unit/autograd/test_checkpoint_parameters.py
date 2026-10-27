"""Activation checkpointing trains what it wraps, and replays its randomness.

``lucid.utils.checkpoint`` kept its own copy of the checkpoint Function,
whose backward took gradients with respect to the explicit inputs only:
every parameter inside the checkpointed function received no gradient,
so training through it silently left those weights frozen.  It now routes
through :func:`lucid.autograd.checkpoint`, which backpropagates through
the recomputed segment into every leaf it reaches.

Both entry points also recomputed with a fresh random state —
``preserve_rng_state`` was accepted and ignored — so a dropout inside the
segment was differentiated through a different mask than the forward used.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.autograd import checkpoint as autograd_checkpoint
from lucid.test._fixtures.devices import metal_available
from lucid.utils.checkpoint import checkpoint as utils_checkpoint

DEVICES = ["cpu"] + (["metal"] if metal_available() else [])


def test_parameters_inside_the_segment_receive_gradients() -> None:
    lin = nn.Linear(4, 4)
    x = lucid.randn(2, 4, requires_grad=True)
    utils_checkpoint(lin, x).sum().backward()
    assert x.grad is not None and lin.weight.grad is not None

    reference = nn.Linear(4, 4)
    reference.load_state_dict(lin.state_dict())
    reference(x.detach()).sum().backward()
    np.testing.assert_allclose(lin.weight.grad.numpy(), reference.weight.grad.numpy())


def test_a_segment_fed_only_constants_still_trains_its_parameters() -> None:
    lin = nn.Linear(4, 4)
    utils_checkpoint(lin, lucid.randn(2, 4)).sum().backward()
    assert lin.weight.grad is not None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "entry",
    [
        lambda f, x: utils_checkpoint(f, x),
        lambda f, x: autograd_checkpoint(f, x, use_reentrant=True),
        lambda f, x: autograd_checkpoint(f, x, use_reentrant=False),
    ],
    ids=["utils", "autograd reentrant", "autograd non-reentrant"],
)
def test_dropout_in_the_segment_is_replayed(device: str, entry) -> None:  # type: ignore[no-untyped-def]
    lin = nn.Linear(16, 16).to(device)
    drop = nn.Dropout(0.5)
    x = lucid.randn(8, 16, device=device, requires_grad=True)

    def segment(t: lucid.Tensor) -> lucid.Tensor:
        return drop(lin(t))

    lucid.manual_seed(0)
    entry(segment, x).sum().backward()
    checkpointed = (x.grad.numpy().copy(), lin.weight.grad.numpy().copy())
    x.grad = None
    lin.weight.grad = None
    lucid.manual_seed(0)
    segment(x).sum().backward()
    np.testing.assert_allclose(checkpointed[0], x.grad.numpy(), rtol=1e-6)
    np.testing.assert_allclose(checkpointed[1], lin.weight.grad.numpy(), rtol=1e-6)


def test_a_segment_returning_several_tensors() -> None:
    a = lucid.randn(3, requires_grad=True)
    first, second = utils_checkpoint(lambda t: (t * 2, t**2), a)
    (first.sum() + second.sum()).backward()
    np.testing.assert_allclose(a.grad.numpy(), 2 + 2 * a.detach().numpy(), rtol=1e-6)


@pytest.mark.parametrize("use_reentrant", [True, False])
@pytest.mark.parametrize("none_first", [False, True], ids=["tensor-none", "none-tensor"])
def test_a_segment_returning_none_beside_its_tensor(use_reentrant: bool, none_first: bool) -> None:
    """A ``None`` among the outputs takes no gradient, wherever it sits.

    ``backward`` gets one gradient per tensor output; pairing them with
    every output raised once ``zip`` turned strict, and with the ``None``
    first it had always read ``requires_grad`` off it.
    """
    a = lucid.randn(3, requires_grad=True)

    def segment(t: lucid.Tensor) -> tuple[lucid.Tensor | None, ...]:
        return (None, t * 2) if none_first else (t * 2, None)

    out = autograd_checkpoint(segment, a, use_reentrant=use_reentrant)
    assert isinstance(out, tuple)
    tensor = out[1] if none_first else out[0]
    assert isinstance(tensor, lucid.Tensor)
    tensor.sum().backward()
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.numpy(), np.full(3, 2.0))
