"""Every loss refuses a cross-device operand pair at its boundary, naming
itself in the message, instead of letting a mismatched pair reach an op
deep inside that names something else (LCD-638).

``mse_loss(cpu, metal)`` used to raise a phantom-storage ``LucidError``
out of its fused kernel's dispatch; ``l1_loss`` and ``smooth_l1_loss``
raised ``DeviceMismatch`` naming the op that touched the mismatched pair
last (``sub``, ``huber_loss``) rather than the loss the caller called.
The check now lives in one place, ``_check_device``
(``lucid.nn.functional._loss_checks``), reached by every public loss
either through ``_operands`` / ``_pair_operands``, or — for the losses
that bypass those two — through one of the other boundary validators
every loss already called unconditionally (``_check_dtypes``,
``_check_same_rank``, ``_check_class_weight``), so a loss never needs
its own, separate device check.

The losses are found the same way ``test_loss_boundary_validation`` finds
them — ``LOSSES`` and ``VALID`` there, built by introspecting
``nn.functional`` — so a new loss is swept automatically rather than
needing a row here.  ``ctc_loss`` is excluded: its ``targets`` and
``*_lengths`` are deliberately moved onto ``log_probs``'s device rather
than refused (see its docstring), so it has no second operand this rule
applies to.
"""

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.nn.functional import loss as _loss_module
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.nn.test_loss_boundary_validation import (
    C,
    CLASS_WEIGHT,
    N,
    LOSSES,
    VALID,
    _inputs,
    losses_with,
)

pytestmark = pytest.mark.skipif(not metal_available(), reason="needs Metal")

#: ``ctc_loss`` moves its second operand onto ``log_probs``'s device by
#: design (see its docstring) rather than refusing a cross-device pair.
_NO_DEVICE_CHECK = frozenset({"ctc_loss"})

CROSS_DEVICE_LOSSES: list[str] = sorted(set(LOSSES) - _NO_DEVICE_CHECK)
_DEVICE_PAIRS = [("cpu", "metal"), ("metal", "cpu")]


def _tensor_kwargs(kwargs: dict[str, object]) -> list[str]:
    """The names of ``kwargs`` holding a ``Tensor`` — the ones a
    cross-device call can disagree on."""
    return [k for k, v in kwargs.items() if isinstance(v, lucid.Tensor)]


def _weight_for(name: str, device: str) -> lucid.Tensor:
    """A ``weight`` of the shape loss ``name`` takes, on ``device``: one
    entry per class for the class-index losses, the input's own shape
    for the rest."""
    return (
        lucid.ones(C, device=device)
        if name in CLASS_WEIGHT
        else lucid.ones(N, C, device=device)
    )


@pytest.mark.parametrize("base,other", _DEVICE_PAIRS)
@pytest.mark.parametrize("name", CROSS_DEVICE_LOSSES)
def test_cross_device_operand_raises_device_mismatch(
    name: str, base: str, other: str
) -> None:
    """Every tensor operand of loss ``name``, moved alone to ``other``'s
    device while its siblings stay on ``base``, is refused before the
    loss computes anything — a ``DeviceMismatch`` naming ``name``."""
    base_kwargs = VALID[name](_inputs(base))
    other_kwargs = VALID[name](_inputs(other))
    swap_names = _tensor_kwargs(base_kwargs)
    assert swap_names, f"{name}: no tensor operand to mismatch"
    fn = getattr(_loss_module, name)
    for swapped in swap_names:
        kwargs = dict(base_kwargs)
        kwargs[swapped] = other_kwargs[swapped]
        with pytest.raises(_C_engine.DeviceMismatch, match=name):
            fn(**kwargs)


@pytest.mark.parametrize("base,other", _DEVICE_PAIRS)
@pytest.mark.parametrize("name", sorted(losses_with("weight")))
def test_cross_device_weight_raises_device_mismatch(
    name: str, base: str, other: str
) -> None:
    """A ``weight`` on ``other``'s device, beside operands on ``base``,
    is refused the same way as any other cross-device operand pair."""
    kwargs = VALID[name](_inputs(base))
    kwargs["weight"] = _weight_for(name, other)
    fn = getattr(_loss_module, name)
    with pytest.raises(_C_engine.DeviceMismatch, match=name):
        fn(**kwargs)
