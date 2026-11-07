"""Which colour transforms keep an input's graph, and which declare they cannot.

A float image that requires grad has to come back either carrying a graph
or from a transform that is on record as non-differentiable.  ``equalize``
was neither: it quantises to 256 bins and looks the result up, so its
output is piecewise constant, yet it returned a plain tensor and the audit
could only call that a lost graph.  The audit's ``NOT_DIFFERENTIABLE``
table is where the declaration lives; this file ties it to what the
functions actually do, so a transform cannot drift to either side of it.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.utils.transforms.functional as F
from lucid import Tensor
from lucid.test.audit._axes import NOT_DIFFERENTIABLE

_PREFIX = "lucid.utils.transforms.functional."

# (name, call).  Every transform in the module that bins, thresholds or
# stretches pixel intensities, so the class is covered as a whole.
_SWEPT: list[tuple[str, Callable[[Tensor], Tensor]]] = [
    ("equalize", F.equalize),
    ("posterize", lambda img: F.posterize(img, 4)),
    ("autocontrast", F.autocontrast),
    ("solarize", lambda img: F.solarize(img, 0.5)),
    ("invert", F.invert),
    ("clahe", F.clahe),
    ("adjust_sharpness", lambda img: F.adjust_sharpness(img, 1.5)),
]


def _image(batched: bool = False) -> Tensor:
    shape = (2, 3, 16, 16) if batched else (3, 16, 16)
    return lucid.rand(*shape, requires_grad=True)


@pytest.mark.parametrize("name,call", _SWEPT, ids=[name for name, _ in _SWEPT])
def test_graph_matches_the_recorded_contract(name: str, call: Callable[[Tensor], Tensor]) -> None:
    declared = (_PREFIX + name) in NOT_DIFFERENTIABLE
    out = call(_image())
    assert out.requires_grad is (not declared), (
        f"{name}: output requires_grad={out.requires_grad} but the audit records it as "
        f"{'non-differentiable' if declared else 'differentiable'}"
    )


@pytest.mark.parametrize("name,call", _SWEPT, ids=[name for name, _ in _SWEPT])
def test_differentiable_transforms_reach_the_input(
    name: str, call: Callable[[Tensor], Tensor]
) -> None:
    if (_PREFIX + name) in NOT_DIFFERENTIABLE:
        pytest.skip("declared non-differentiable")
    x = _image()
    call(x).sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("clip_limit", [None, 2.0], ids=["plain", "clipped"])
@pytest.mark.parametrize("batched", [False, True], ids=["chw", "bchw"])
def test_equalize_never_carries_a_graph(clip_limit: float | None, batched: bool) -> None:
    assert not F.equalize(_image(batched), clip_limit).requires_grad


def test_equalize_with_a_flat_channel_carries_no_graph() -> None:
    """A flat channel passes through unchanged; it must not hand back a
    gradient for that channel alone while its neighbours get none."""
    base = lucid.rand(3, 16, 16, requires_grad=True)
    flat = lucid.cat([base[:1] * 0.0 + 0.5, base[1:]], 0)
    assert flat.requires_grad
    assert not F.equalize(flat).requires_grad
