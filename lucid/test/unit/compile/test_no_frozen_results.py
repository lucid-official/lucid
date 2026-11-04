"""A compiled model must answer its input, not the one it was traced with.

An op that never records its trace I/O leaves the node with the empty
input list ``on_op_enter`` seeded, and its result never becomes a traced
tensor. The builder then drops the node as dead — correctly, by its own
rule — and the *consumer* meets the result as a fresh external feed,
bound once, at trace time. The compiled model reports success, agrees
with eager on the first call, and returns that same answer forever.

That is the worst shape a defect can take here: no crash, no fallback,
no warning. ``fft.fftn`` was off by 20 on the second call.

The test is the property rather than the mechanism: compile, call with
two inputs that eager distinguishes, and require the output to move.
Whether an op compiles or falls back is not the point — either is
honest. Answering the previous question is not.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.linalg as la
import lucid.nn as nn
from lucid._C import engine as _C_engine


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

# Every one of these used to freeze, except the last two, which are
# controls: one op that compiles and one that falls back, both of which
# have to keep tracking their input.
CASES = [
    ("fftn", lambda t: lucid.real(lucid.fft.fftn(t)), (2, 4)),
    ("ifftn", lambda t: lucid.real(lucid.fft.ifftn(lucid.complex(t, t))), (2, 4)),
    ("rfftn", lambda t: lucid.real(lucid.fft.rfftn(t)), (2, 4)),
    ("irfftn", lambda t: lucid.fft.irfftn(lucid.complex(t, t)), (2, 4)),
    ("nonzero", lambda t: lucid.nonzero(t > 0)[0], (8,)),
    ("histogram2d", lambda t: lucid.histogram2d(t[:, 0], t[:, 1], bins=4)[0], (16, 2)),
    ("histogramdd", lambda t: lucid.histogramdd(t, bins=[4, 4])[0], (16, 2)),
    ("relu-control", lambda t: t.relu(), (2, 4)),
    ("svd-control", lambda t: lucid.linalg.svd(t)[1], (4, 4)),
]


class _Apply(nn.Module):
    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)  # type: ignore[operator]


@pytest.mark.parametrize(("name", "fn", "shape"), CASES, ids=[c[0] for c in CASES])
def test_the_second_call_answers_the_second_input(name, fn, shape):
    lucid.manual_seed(0)
    model = _Apply(fn).eval()
    first = lucid.randn(*shape)
    second = lucid.randn(*shape) * 7 + 3

    want_first = model(first)
    want_second = model(second)
    spread = float((want_first - want_second).abs().max().item())
    assert spread > 1e-6, "the probe itself is blind — eager gives the same answer"

    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got_first = compiled(first.to("metal")).to("cpu")
    got_second = compiled(second.to("metal")).to("cpu")

    moved = float((got_first - got_second).abs().max().item())
    assert moved > 1e-6, (
        f"{name}: the compiled model returned the same values for two different "
        "inputs — its result was frozen into the graph at trace time"
    )
    scale = max(float(want_second.abs().max().item()), 1e-30)
    assert float((got_second - want_second).abs().max().item()) / scale < 1e-5


def test_a_pure_python_composite_that_reads_values_runs_eager():
    """A composite that reads an element with ``.item()`` while it runs.

    Reading an element during tracing used to take the trace-time value, so
    whatever was computed from it was baked in as a constant and every call
    answered the first.  A host read of a traced value now marks the trace
    unsupported: the call runs eager and tracks its input.  (This was
    ``lucid.histogram`` until it became tensor operations, API-05 — see
    ``test_a_histogram_compiles_into_the_graph``.)
    """
    lucid.manual_seed(0)
    model = _Apply(lambda t: t / t.abs().max().item()).eval()
    first = lucid.randn(16)
    second = lucid.randn(16) * 7 + 3
    want_second = model(second)
    assert float((model(first) - want_second).abs().max().item()) > 1e-6

    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    compiled(first.to("metal"))
    got_second = compiled(second.to("metal")).to("cpu")
    assert float((got_second - want_second).abs().max().item()) <= 1e-6
    assert compiled.cache_info()["eager_only"]


def test_a_histogram_compiles_into_the_graph():
    """``lucid.histogram`` reads no value back to the host (API-05): its
    range is the input's extremes as tensors, so it traces, and the
    compiled graph still follows its input."""
    lucid.manual_seed(0)
    model = _Apply(lambda t: lucid.histogram(t, bins=4)[0]).eval()
    first = lucid.randn(16)
    second = lucid.randn(16) * 7 + 3
    want_second = model(second)
    assert float((model(first) - want_second).abs().max().item()) > 1e-6

    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    compiled(first.to("metal"))
    got_second = compiled(second.to("metal")).to("cpu")
    assert float((got_second - want_second).abs().max().item()) <= 1e-6


# ── Engine ops that handed back their result without recording the trace
# I/O (LCD-836): ``masked_select``, the LU and LDL factorisations, and
# ``householder_product``. They had no ``OpScopeFull`` and no ``on_op_io``, so
# not even a node was left behind for the builder to refuse; the result was an
# anonymous tensor, and whatever consumed it bound it as a feed at trace time.
#
# The property is the one above, held at the three places an op can stand in a
# function: first, after another op, and before another op. The position
# matters because the result reaches the builder by different paths: as the
# function's own output, or through a consumer that binds it as a feed. A
# function whose only op is the unrecorded one is refused for another reason
# ("recorded no operations"), which hides the defect. Every output of a
# multi-output op has to be recorded, so each output is probed on its own,
# ``info`` included. The checked forms (``lu_factor``, ``lu``, ``ldl_factor``)
# are pinned too: they fell back only because they read ``info`` on the host,
# which says nothing about the op.

Fn = Callable[[lucid.Tensor], lucid.Tensor]
Pair = tuple[lucid.Tensor, lucid.Tensor]


def _random_pair(*shape: int) -> Pair:
    lucid.manual_seed(0)
    return lucid.randn(*shape), lucid.randn(*shape) * 7 + 3


def _square() -> Pair:
    return _random_pair(4, 4)


def _tall() -> Pair:
    return _random_pair(5, 3)


def _symmetric_positive_definite() -> Pair:
    a, b = _random_pair(4, 4)
    eye = lucid.eye(4)
    return a @ a.mT + 4 * eye, b @ b.mT + 4 * eye


def _lu_pivots_differ() -> Pair:
    # The largest entry of the first column is the first pivot; it is in the
    # last row of one matrix and the first row of the other.
    rows = [[1.0, 2.0, 3.0, 4.0], [2.0, 5.0, 1.0, 3.0], [3.0, 1.0, 4.0, 2.0], [4.0, 3.0, 2.0, 5.0]]
    return lucid.tensor(rows), lucid.tensor(rows[::-1])


def _ldl_pivots_differ() -> Pair:
    # Only 1x1 blocks against only 2x2 blocks: Bunch-Kaufman pivots of
    # opposite sign.
    swaps = [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    return lucid.diag(lucid.tensor([1.0, 2.0, 3.0, 4.0])), lucid.tensor(swaps)


def _lu_info_differs() -> Pair:
    # ``getrf`` reports a zero pivot in ``info`` and still factors.
    return lucid.tensor([[1.0, 2.0], [2.0, 4.0]]), lucid.tensor([[1.0, 2.0], [3.0, 4.0]])


def _ldl_info_differs() -> Pair:
    return lucid.zeros(3, 3), lucid.eye(3) * 2


def _every_other(x: lucid.Tensor) -> lucid.Tensor:
    """A mask the trace makes itself, so only the data it selects from is live."""
    return lucid.arange(int(x.numel()), device=x.device).reshape(x.shape) % 2 == 0


def _reflectors(h: lucid.Tensor) -> lucid.Tensor:
    return lucid.tensor([0.5, 1.2, 0.8], device=h.device)


# (name, the op as a function of one tensor, the two inputs it is probed with)
_OPS: list[tuple[str, Fn, Callable[[], Pair]]] = [
    ("masked_select-const-mask", lambda x: lucid.masked_select(x, _every_other(x)), _square),
    ("masked_select-live-mask", lambda x: lucid.masked_select(x, x.abs() >= 0), _square),
    ("lu_factor_ex-LU", lambda x: la.lu_factor_ex(x).LU, _square),
    ("lu_factor_ex-pivots", lambda x: la.lu_factor_ex(x).pivots, _lu_pivots_differ),
    ("lu_factor_ex-info", lambda x: la.lu_factor_ex(x).info, _lu_info_differs),
    ("lu_factor-LU", lambda x: la.lu_factor(x).LU, _square),
    ("lu_factor-pivots", lambda x: la.lu_factor(x).pivots, _lu_pivots_differ),
    ("lu-U", lambda x: la.lu(x).U, _square),
    ("lu-P", lambda x: la.lu(x).P, _lu_pivots_differ),
    ("ldl_factor_ex-LD", lambda x: la.ldl_factor_ex(x).LD, _symmetric_positive_definite),
    ("ldl_factor_ex-pivots", lambda x: la.ldl_factor_ex(x).pivots, _ldl_pivots_differ),
    ("ldl_factor_ex-info", lambda x: la.ldl_factor_ex(x).info, _ldl_info_differs),
    ("ldl_factor-LD", lambda x: la.ldl_factor(x).LD, _symmetric_positive_definite),
    ("ldl_factor-pivots", lambda x: la.ldl_factor(x).pivots, _ldl_pivots_differ),
    ("householder_product", lambda h: la.householder_product(h, _reflectors(h)), _tall),
    ("householder_product-live-tau", lambda h: la.householder_product(h, h.abs().sum(0)), _tall),
]

_POSITIONS: dict[str, Callable[[Fn], Fn]] = {
    "first": lambda op: op,
    "after": lambda op: lambda x: op(x * 2),
    "before": lambda op: lambda x: op(x) * 2,
}

_CASES = [
    pytest.param(place(op), pair, id=f"{name}-{where}")
    for name, op, pair in _OPS
    for where, place in _POSITIONS.items()
]


def _gap(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


@pytest.mark.parametrize(("fn", "pair"), _CASES)
def test_an_unemitted_op_answers_the_second_input(fn: Fn, pair: Callable[[], Pair]) -> None:
    first, second = (t.to("metal") for t in pair())
    with lucid.no_grad():
        want_first, want_second = fn(first), fn(second)
        assert _gap(want_first, want_second) > 1e-6, "the probe is blind: eager gives one answer"

        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile(fn)
        got_first = compiled(first)
        got_second = compiled(second)

    assert _gap(got_first, got_second) > 1e-6, (
        "the compiled function returned the same values for two different inputs: "
        "its result was frozen into the graph at trace time"
    )
    assert _gap(got_second, want_second) <= 1e-5 * max(float(want_second.abs().max().item()), 1.0)
