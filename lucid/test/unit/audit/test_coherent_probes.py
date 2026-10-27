"""Probes whose operands constrain each other stay coherent (LCD-367).

Each case here was a cell the audit stopped answering when the op began
refusing an incoherent probe: a solve whose matrix was resized without
its right-hand side, a packed sequence whose lengths outran its batch, a
``grad`` handed two unrelated leaves.  The refusals are correct; these
tests pin the probes that ask the question properly instead.
"""

import functools

import numpy as np
import pytest

import lucid
from lucid.test.audit import _specs, _surface
from lucid.test.audit._axes import NOT_DIFFERENTIABLE, Context, EdgeAxis, GradientAxis, _try_forward
from lucid.test.audit._axes_stability import ContiguityAxis
from lucid.test.audit._result import Status


@functools.cache
def _symbols() -> dict[str, _surface.Symbol]:
    return {s.qualname: s for s in _surface.enumerate_surface()}


def _symbol(qualname: str) -> _surface.Symbol:
    return _symbols()[qualname]


def _first_call(qualname: str) -> tuple[object, _specs.Call]:
    """The first candidate that runs, as the axes choose it."""
    symbol = _symbol(qualname)
    fn = _surface.resolve(symbol)
    for call in _specs.invocations(symbol.short, "moderate", qualname, fn):
        try:
            fn(*call.args, **call.kwargs)
        except Exception:  # noqa: BLE001 - the axes skip a refused candidate too
            continue
        return fn, call
    raise AssertionError(f"no candidate for {qualname} runs")


_SOLVES = (
    "lucid.linalg.solve",
    "lucid.linalg.solve_ex",
    "lucid.linalg.solve_triangular",
    "lucid.linalg.lu_solve",
    "lucid.linalg.ldl_solve",
)


@pytest.mark.parametrize("qualname", _SOLVES)
def test_a_resized_solve_keeps_its_sides_and_pivots_coherent(qualname: str) -> None:
    fn, call = _first_call(qualname)
    shape = call.base.shape
    for array in (np.zeros((0, *shape[1:])), np.full((1,) * len(shape), 0.7)):
        resized = call.resized(array)
        matrix, rhs = resized.args[0], resized.args[-1]
        assert matrix.shape[-1] == matrix.shape[-2], matrix.shape
        assert tuple(rhs.shape[: matrix.ndim - 1]) == tuple(matrix.shape[:-1])
        if len(resized.args) == 3:
            pivots = resized.args[1].numpy()
            assert pivots.shape == tuple(matrix.shape[:-1])
            np.testing.assert_array_equal(
                pivots, np.broadcast_to(np.arange(1, matrix.shape[-1] + 1), pivots.shape)
            )
        fn(*resized.args, **call.kwargs)


#: An in-place clamp cannot grow its receiver, so its tensor bounds shrink
#: with it (LCD-532, once LCD-519 made ``clamp_`` refuse as ``add_`` does).
_CLAMPS_IN_PLACE = ("Tensor.clamp_", "Tensor.clamp_min_", "Tensor.clamp_max_", "Tensor.clip_")


@pytest.mark.parametrize(
    "qualname", (*_SOLVES, "lucid.nn.utils.pack_padded_sequence", "F.ctc_loss", *_CLAMPS_IN_PLACE)
)
def test_the_edge_axis_answers_ops_with_joint_operands(qualname: str) -> None:
    finding = EdgeAxis().run(_symbol(qualname), Context())
    assert finding.status == Status.PASS, finding.detail


def test_a_resized_padded_batch_takes_lengths_of_its_own_batch() -> None:
    fn, call = _first_call("lucid.nn.utils.pack_padded_sequence")
    shape = call.base.shape

    single = call.resized(np.full((1,) * len(shape), 0.7))
    assert list(single.args[1]) == [1]
    fn(*single.args, **call.kwargs)

    # An empty time axis admits no valid length, so the empty problem is
    # an empty batch — which the op refuses, as the reference does.
    empty = call.resized(np.zeros((0, *shape[1:])))
    assert empty.args[0].shape[0] == shape[0] and empty.args[0].shape[1] == 0
    assert list(empty.args[1]) == []
    with pytest.raises(ValueError):
        fn(*empty.args, **call.kwargs)


def test_a_resized_ctc_problem_keeps_its_targets_and_lengths_coherent() -> None:
    """LCD-275: the edge axis once emptied ``log_probs`` alone, so every ctc
    call it built was incoherent and the cell was dropped from coverage."""
    fn, call = _first_call("F.ctc_loss")
    shape = call.base.shape

    single = call.resized(np.full((1,) * len(shape), 0.7))
    assert tuple(single.args[1].shape) == (1, 0)  # C = 1 holds only the blank
    assert single.args[2].numpy().tolist() == [1]
    assert single.args[3].numpy().tolist() == [0]
    fn(*single.args, **call.kwargs)

    resized = call.resized(np.full((4, 3, 3), 0.1))
    targets = resized.args[1].numpy()
    assert targets.shape == (3, 2) and targets.min() >= 1 and targets.max() <= 2
    assert resized.args[2].numpy().tolist() == [4, 4, 4]
    assert resized.args[3].numpy().tolist() == [2, 2, 2]
    fn(*resized.args, **call.kwargs)

    # An empty T is still a coherent problem: every length is 0.
    empty = call.resized(np.zeros((0, *shape[1:])))
    assert empty.args[2].numpy().tolist() == [0] * shape[1]
    assert empty.args[3].numpy().tolist() == [0] * shape[1]


@pytest.mark.parametrize("qualname", ["Tensor.view", "lucid.view"])
def test_view_is_probed_with_a_shape_a_strided_slice_can_express(qualname: str) -> None:
    finding = ContiguityAxis().run(_symbol(qualname), Context())
    assert finding.status == Status.PASS, finding.detail
    assert "vacuous" not in finding.detail


@pytest.mark.parametrize(
    "qualname",
    ["lucid.autograd.grad", "lucid.autograd.vjp", "lucid.autograd.jvp", "Tensor.retain_grad"],
)
def test_graph_taking_callables_have_an_invocation_that_runs(qualname: str) -> None:
    _first_call(qualname)


@pytest.mark.parametrize("cls", [lucid.nn.BCELoss, lucid.nn.GaussianNLLLoss])
def test_loss_modules_with_a_value_domain_forward(cls: type) -> None:
    out, why = _try_forward(cls())
    assert out is not None, why


def test_detach_is_not_reported_for_carrying_no_graph() -> None:
    finding = GradientAxis().run(_symbol("lucid.detach"), Context())
    assert finding.status == Status.UNSUPPORTED, finding.detail


def test_every_free_function_twin_of_a_non_differentiable_method_is_listed() -> None:
    # One op, two spellings: ``lucid.detach`` reported a defect for
    # months while ``Tensor.detach`` sat in the table.
    surface = _symbols()
    missing = [
        f"lucid.{key.removeprefix('Tensor.')}"
        for key in NOT_DIFFERENTIABLE
        if key.startswith("Tensor.")
        and f"lucid.{key.removeprefix('Tensor.')}" in surface
        and f"lucid.{key.removeprefix('Tensor.')}" not in NOT_DIFFERENTIABLE
    ]
    assert not missing, missing
