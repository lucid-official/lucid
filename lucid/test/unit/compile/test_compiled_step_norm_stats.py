"""``compiled_step`` applies a norm's tracked running statistics (LCD-877).

An InstanceNorm folds each batch into its buffers with an in-place write, a
BatchNorm hands the new statistics out of the kernel: both ride the executable
and are written back after the run (``carried_writes``), as in ``make_step`` and
the module entry.  ``compiled_step`` has no eager path, so a norm whose update
cannot be carried is refused with a message that says so, never run with the
statistics frozen.  The step is compared with the eager one: loss, gradients,
running statistics and counter after several steps.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import compiled_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

STEPS = 4
FEATURES = 3

type Layer = Callable[[], nn.Module]


def _norms() -> dict[str, tuple[Layer, tuple[int, ...]]]:
    """``name -> (the norm, the input shape it takes)``."""
    track = {"track_running_stats": True}
    return {
        "instance1d": (lambda: nn.InstanceNorm1d(FEATURES, affine=False, **track), (4, 3, 6)),
        "instance1d-affine": (lambda: nn.InstanceNorm1d(FEATURES, affine=True, **track), (4, 3, 6)),
        "instance1d-momentum": (
            lambda: nn.InstanceNorm1d(FEATURES, momentum=0.3, affine=True, **track),
            (4, 3, 6),
        ),
        "instance1d-none": (
            lambda: nn.InstanceNorm1d(FEATURES, momentum=None, affine=True, **track),
            (4, 3, 6),
        ),
        "instance2d": (lambda: nn.InstanceNorm2d(FEATURES, affine=False, **track), (4, 3, 5, 5)),
        "instance2d-affine": (
            lambda: nn.InstanceNorm2d(FEATURES, affine=True, **track),
            (4, 3, 5, 5),
        ),
        "batch1d": (lambda: nn.BatchNorm1d(FEATURES), (4, 3, 6)),
        "batch1d-momentum": (lambda: nn.BatchNorm1d(FEATURES, momentum=0.3), (4, 3, 6)),
        "batch1d-no-affine": (lambda: nn.BatchNorm1d(FEATURES, affine=False), (4, 3, 6)),
        "batch2d": (lambda: nn.BatchNorm2d(FEATURES), (4, 3, 5, 5)),
        "batch2d-momentum": (lambda: nn.BatchNorm2d(FEATURES, momentum=0.05), (4, 3, 5, 5)),
        "batch2d-no-affine": (lambda: nn.BatchNorm2d(FEATURES, affine=False), (4, 3, 5, 5)),
    }


NORMS = _norms()


def _model(name: str) -> nn.Module:
    layer, _ = NORMS[name]
    lucid.manual_seed(0)
    spatial = len(NORMS[name][1]) - 2
    conv = nn.Conv1d(FEATURES, FEATURES, 1) if spatial == 1 else nn.Conv2d(FEATURES, FEATURES, 1)
    return nn.Sequential(conv, layer()).to(COMPILE_DEVICE)


def _loss(out: lucid.Tensor) -> lucid.Tensor:
    return out.square().mean()


def _batches(name: str) -> list[lucid.Tensor]:
    lucid.manual_seed(1)
    shape = NORMS[name][1]
    return [lucid.randn(*shape, device=COMPILE_DEVICE) * (1.0 + 0.5 * i) for i in range(STEPS)]


def _close(got: lucid.Tensor, want: lucid.Tensor, atol: float = 1e-4) -> bool:
    return bool(lucid.allclose(got.detach(), want.detach(), atol=atol, rtol=1e-4))


def _grads(model: nn.Module) -> list[lucid.Tensor]:
    return [p.grad for p in model.parameters() if p.grad is not None]


def _buffers(model: nn.Module) -> dict[str, lucid.Tensor]:
    return dict(model.named_buffers())


@pytest.mark.parametrize("name", list(NORMS))
def test_running_statistics_counters_loss_and_grads_match_eager(name: str) -> None:
    model, eager = _model(name), _model(name)
    eager.load_state_dict(model.state_dict())
    for x in _batches(name):
        model.zero_grad()
        eager.zero_grad()
        got = compiled_step(model, x, _loss)
        want = _loss(eager(x))
        want.backward()
        assert _close(got, want)
        for g, w in zip(_grads(model), _grads(eager), strict=True):
            assert _close(g, w)
        for key, buf in _buffers(eager).items():
            assert _close(_buffers(model)[key], buf), key


@pytest.mark.parametrize("name", [n for n in NORMS if n != "instance1d-none"])
def test_the_statistics_advance_from_their_first_values(name: str) -> None:
    """Pins the defect itself: the call refused, or the buffers were never written.

    An InstanceNorm with ``momentum=None`` folds nothing in, so it is left out.
    """
    model = _model(name)
    before = {key: buf.clone() for key, buf in _buffers(model).items()}
    compiled_step(model, _batches(name)[0], _loss)
    after = _buffers(model)
    moved = [key for key in before if not _close(before[key], after[key], atol=1e-6)]
    assert moved, "no running statistic moved"
    if "num_batches_tracked" in before and "batch" in name:
        assert int(after["num_batches_tracked"].item()) == 1


@pytest.mark.parametrize("name", ["instance1d-affine", "batch1d", "batch2d"])
def test_a_held_buffer_sees_the_in_place_update(name: str) -> None:
    """The write goes into the buffer's own storage: a reference taken earlier moves."""
    model = _model(name)
    held = model[1].running_mean
    before = held.clone()
    compiled_step(model, _batches(name)[0], _loss)
    assert not _close(held, before, atol=1e-6)


@pytest.mark.parametrize("name", ["instance1d-affine", "batch1d", "batch2d-momentum"])
def test_an_evaluating_norm_reads_its_buffers_and_leaves_them(name: str) -> None:
    model, eager = _model(name), _model(name)
    eager.load_state_dict(model.state_dict())
    model.eval()
    eager.eval()
    before = {key: buf.clone() for key, buf in _buffers(model).items()}
    for x in _batches(name)[:2]:
        model.zero_grad()
        eager.zero_grad()
        got = compiled_step(model, x, _loss)
        want = _loss(eager(x))
        want.backward()
        assert _close(got, want)
        for g, w in zip(_grads(model), _grads(eager), strict=True):
            assert _close(g, w)
    for key, buf in before.items():
        assert _close(_buffers(model)[key], buf, atol=0.0), key


@pytest.mark.parametrize("shape", [(4, 3, 6), (4, 3, 5, 5)], ids=["1d", "2d"])
def test_a_cumulative_batch_norm_in_training_is_refused_with_its_buffers_untouched(
    shape: tuple[int, ...],
) -> None:
    lucid.manual_seed(0)
    norm = (
        nn.BatchNorm1d(FEATURES, momentum=None)
        if len(shape) == 3
        else nn.BatchNorm2d(FEATURES, momentum=None)
    )
    model = nn.Sequential(nn.Identity(), norm).to(COMPILE_DEVICE).train()
    before = {key: buf.clone() for key, buf in _buffers(model).items()}
    x = lucid.randn(*shape, device=COMPILE_DEVICE, requires_grad=True)
    with pytest.raises(RuntimeError, match=r"momentum=None.*training mode.*no eager fallback"):
        compiled_step(model, x, _loss)
    for key, buf in before.items():
        assert _close(_buffers(model)[key], buf, atol=0.0), key


def test_a_cumulative_batch_norm_that_is_evaluating_compiles() -> None:
    model = nn.Sequential(nn.Conv1d(FEATURES, FEATURES, 1), nn.BatchNorm1d(FEATURES, momentum=None))
    model = model.to(COMPILE_DEVICE).eval()
    x = lucid.randn(4, FEATURES, 6, device=COMPILE_DEVICE)
    got = compiled_step(model, x, _loss)
    assert _close(got, _loss(model(x)))


class _StatisticsOfATensorTheCallMade(nn.Module):
    """A functional batch norm whose statistics are built inside the call."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(FEATURES, FEATURES)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        mean = lucid.zeros(FEATURES, device=x.device)
        var = lucid.ones(FEATURES, device=x.device)
        return F.batch_norm(self.lin(x), mean, var, training=True)


def test_statistics_with_no_tensor_to_go_to_are_refused_not_dropped() -> None:
    """Compiled they would vanish, and there is no eager call to run instead."""
    model = _StatisticsOfATensorTheCallMade().to(COMPILE_DEVICE)
    x = lucid.randn(4, FEATURES, device=COMPILE_DEVICE)
    with pytest.raises(RuntimeError, match=r"compiled_step: .*no eager fallback"):
        compiled_step(model, x, _loss)
