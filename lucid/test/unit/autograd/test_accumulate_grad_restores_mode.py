"""Accumulating a ``.grad`` hands back the grad-mode flag it was given (LCD-1042).

Adding a gradient that tracks autograd into ``.grad`` switches grad mode on for
the add and back after.  It saved the flag through ``GradMode::is_enabled()``,
which is false while inference mode is on, and restored it through
``set_enabled``, which writes the raw flag: a thread with the raw flag on and
inference mode on came out of the accumulation with the raw flag off, and grad
mode stayed off once inference mode ended.  The scopes of ``lucid.inference_mode``
restore the raw flag on exit and hide it; the engine binding does not.  The add is
under ``GradModeGuard`` now, which restores exactly the raw flag it saw, and also
when the add throws.
"""

from collections.abc import Iterator

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap


@pytest.fixture
def engine_flags_restored() -> Iterator[None]:
    """Whatever a test does to the two grad-mode flags, the next one starts from the default."""
    yield
    _C_engine.set_inference_enabled(False)
    _C_engine.set_grad_enabled(True)


def _leaf_holding_a_tracked_grad(device: str) -> lucid.Tensor:
    leaf = lucid.randn(3, device=device).requires_grad_(True)
    (leaf * leaf).sum().backward(create_graph=True)
    assert leaf.grad is not None and leaf.grad.requires_grad
    return leaf


def _tracked(device: str, n: int = 3) -> lucid.Tensor:
    """A gradient that tracks autograd, as ``backward(create_graph=True)`` hands one in."""
    return lucid.randn(n, device=device).requires_grad_(True) * 2.0


@pytest.mark.usefixtures("engine_flags_restored")
@pytest.mark.parametrize("path", ["merge", "first"])
def test_accumulating_under_inference_leaves_the_raw_flag_on(device: str, path: str) -> None:
    if path == "merge":
        leaf = _leaf_holding_a_tracked_grad(device)
    else:
        leaf = lucid.randn(3, device=device).requires_grad_(True)
    g = _tracked(device)
    assert _C_engine.raw_grad_enabled()

    _C_engine.set_inference_enabled(True)
    _C_engine._accumulate_grad(_unwrap(leaf), _unwrap(g))
    assert _C_engine.raw_grad_enabled(), "the accumulation wrote the raw grad-mode flag off"

    _C_engine.set_inference_enabled(False)
    assert lucid.is_grad_enabled()


@pytest.mark.usefixtures("engine_flags_restored")
def test_a_throwing_accumulation_leaves_the_raw_flag_on(device: str) -> None:
    leaf = _leaf_holding_a_tracked_grad(device)
    wrong_shape = _tracked(device, n=2)

    _C_engine.set_inference_enabled(True)
    with pytest.raises(_C_engine.LucidError):
        _C_engine._accumulate_grad(_unwrap(leaf), _unwrap(wrong_shape))
    assert _C_engine.raw_grad_enabled()

    _C_engine.set_inference_enabled(False)
    assert lucid.is_grad_enabled()


@pytest.mark.usefixtures("engine_flags_restored")
@pytest.mark.parametrize("path", ["merge", "first"])
def test_accumulating_under_no_grad_leaves_the_raw_flag_off(device: str, path: str) -> None:
    """The add is switched on for a tracked gradient, and the caller's ``no_grad`` comes back."""
    if path == "merge":
        leaf = _leaf_holding_a_tracked_grad(device)
    else:
        leaf = lucid.randn(3, device=device).requires_grad_(True)
    g = _tracked(device)

    with lucid.no_grad():
        _C_engine._accumulate_grad(_unwrap(leaf), _unwrap(g))
        assert not _C_engine.raw_grad_enabled()
    assert lucid.is_grad_enabled()


@pytest.mark.usefixtures("engine_flags_restored")
def test_the_accumulated_grad_still_tracks_autograd(device: str) -> None:
    """The guard changes what is restored, not what the add records."""
    leaf = _leaf_holding_a_tracked_grad(device)
    before = leaf.grad.detach().clone() if leaf.grad is not None else None
    g = _tracked(device)
    expected = before + g.detach() if before is not None else None

    _C_engine._accumulate_grad(_unwrap(leaf), _unwrap(g))

    assert leaf.grad is not None and expected is not None
    assert leaf.grad.requires_grad
    assert lucid.allclose(leaf.grad.detach(), expected)
