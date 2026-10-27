"""The profiler records backward work (LCD-156).

``OpScope`` instruments the ops a forward dispatches, and the engine ran the
nodes of a backward pass without a word: ``(x @ x).sum().backward()``
recorded three events, none of them backward, and a backward twice as slow
as the reference's could not be attributed.  A pass is now one event,
``"backward"``, and every node it runs one more, named after the node
(``"MulBackward"``, ``"AccumulateGrad"``) and marked as backward work — what
``OpEvent.backward`` tells apart from the ops the step dispatched.
"""

import pytest

import lucid
from lucid.profiler import Profiler


def _names(prof: Profiler) -> list[str]:
    return [ev.name for ev in prof.events()]


def _backward_flags(prof: Profiler) -> dict[str, bool]:
    return {ev.name: ev.backward for ev in prof.events()}


def test_a_backward_pass_and_its_nodes_are_events(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True, device=device)
    y = (x * x + x.sin()).sum()
    with lucid.profiler.profile() as prof:
        y.backward()
    names = _names(prof)
    for node in ("backward", "SumBackward", "AddBackward", "MulBackward", "SinBackward"):
        assert node in names, names
    assert "AccumulateGrad" in names
    # The pass encloses its nodes: it ends, and is recorded, last.
    assert names[-1] == "backward"
    assert {"MulBackward"} <= {s.name for s in prof.key_averages()}


def test_backward_events_are_marked_and_forward_ops_are_not(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    with lucid.profiler.profile() as prof:
        y = (x * x).sum()
        y.backward()
    flags = _backward_flags(prof)
    assert flags["mul"] is False and flags["sum"] is False
    assert flags["backward"] is True and flags["MulBackward"] is True


def test_autograd_grad_records_its_nodes(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    y = (x * x * x).sum()
    with lucid.profiler.profile() as prof:
        (g,) = lucid.autograd.grad(y, x)
    assert g is not None
    names = _names(prof)
    assert "backward" in names and "MulBackward" in names, names
    # grad() writes no .grad: the leaf's AccumulateGrad never runs.
    assert "AccumulateGrad" not in names


def test_a_create_graph_pass_records_its_nodes_and_the_ops_they_dispatch(device: str) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device)
    y = (x * x).sum()
    with lucid.profiler.profile() as prof:
        (g,) = lucid.autograd.grad(y, x, create_graph=True)
    flags = _backward_flags(prof)
    assert flags["MulBackward"] is True
    # The gradient is built from ops, which are events of their own.
    assert flags["mul"] is False


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_a_node_event_carries_the_gradient_it_ran_on(shape: tuple[int, ...], device: str) -> None:
    x = lucid.ones(*shape, requires_grad=True, device=device)
    with lucid.profiler.profile() as prof:
        (x.exp()).sum().backward()
    (event,) = [ev for ev in prof.events() if ev.name == "ExpBackward"]
    # An eager gradient carries no shape: the event's is its element count.
    assert event.shape == [x.numel()]


def test_nothing_is_recorded_without_a_profiler(device: str) -> None:
    x = lucid.tensor([1.0, 2.0], requires_grad=True, device=device)
    (x * x).sum().backward()
    with lucid.profiler.profile() as prof:
        pass
    assert _names(prof) == []
