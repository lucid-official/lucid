"""A trace stays clean and keeps recording after an op inside it throws (LCD-1042, LCD-1045).

``dropout_stateful`` runs its nested ``dropout`` with the tracer detached, so the
graph holds one ``dropout_stateful`` node and no ``dropout`` under it.  That
nested draw throws for a bool or integer input and under determinism without a
generator; the detach used to be a bare ``set_current_tracer(nullptr)`` ...
``set_current_tracer(saved)`` pair, and the throw skipped the second half.  The
tracer stayed unset and the rest of the trace recorded nothing.

Even with ``TracerDetach`` restoring the tracer, any op that throws after its
``OpScopeFull`` opened (``concatenate``, ``gather``, ``dropout_stateful``, …)
used to leave a dangling placeholder node (``inputs=[]``) in the trace, causing
compiled callers that catch the refusal and continue to fall back to eager
(LCD-1045).  Unwinding ``OpScopeFull`` before ``on_op_io`` now pops that pending
node so a refused op leaves the trace as if it had never been called.
"""

import warnings
from collections.abc import Callable, Iterator

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.compile import _tracing
from lucid.test._helpers.dtype_refusal import engine_refusal


def _dropout_refusal(dtype: lucid.dtype) -> type[RuntimeError]:
    """The class ``dropout``'s schema raises for ``dtype``: a bool and every integer alike."""
    refusal = engine_refusal("dropout", dtype)
    assert refusal is not None, f"dropout accepts {dtype} now"
    return refusal.error


#: The ways the nested draw refuses: the input is no float (a bool or an integer, turned away by
#: the schema of ``dropout``), or determinism is on and no generator was given.
_REFUSALS = [
    *(
        pytest.param(
            dtype, False, _dropout_refusal(dtype), id=f"{str(dtype).removeprefix('lucid.')}-input"
        )
        for dtype in (lucid.bool_, lucid.int8, lucid.int16, lucid.int32, lucid.int64)
    ),
    pytest.param(lucid.float32, True, _C_engine.LucidError, id="deterministic-without-generator"),
]


@pytest.fixture
def deterministic_restored() -> Iterator[None]:
    """Leave the process-wide determinism flag as the test found it."""
    before = lucid.are_deterministic_algorithms_enabled()
    yield
    lucid.use_deterministic_algorithms(before)


def _state(device: str) -> lucid.Tensor:
    return lucid.zeros((2,), dtype=lucid.int32, device=device)


@pytest.mark.usefixtures("deterministic_restored")
@pytest.mark.parametrize(("dtype", "deterministic", "error"), _REFUSALS)
def test_a_throwing_dropout_stateful_leaves_the_tracer_installed(
    device: str, dtype: lucid.dtype, deterministic: bool, error: type[Exception]
) -> None:
    x = lucid.ones((4,), dtype=dtype, device=device)
    state = _state(device)
    probe = lucid.randn(3, device=device)
    with _tracing() as tracer:
        lucid.use_deterministic_algorithms(deterministic)
        with pytest.raises(error):
            _C_engine.nn.dropout_stateful(_unwrap(x), _unwrap(state), 0.5, True)
        lucid.use_deterministic_algorithms(False)

        assert _C_engine.compile.current_tracer() is tracer
        assert tracer.op_count() == 0, "the refused op left a dangling node in the trace"
        lucid.relu(probe)
        assert tracer.op_count() == 1, "the trace stopped recording after the throw"
    assert _C_engine.compile.current_tracer() is None


@pytest.mark.usefixtures("deterministic_restored")
def test_a_throw_does_not_install_a_tracer_that_was_not_there(device: str) -> None:
    """Outside a trace the detach is a no-op both ways: nothing is installed afterwards."""
    x = lucid.ones((4,), dtype=lucid.bool_, device=device)
    assert _C_engine.compile.current_tracer() is None
    with pytest.raises(_dropout_refusal(lucid.bool_)):
        _C_engine.nn.dropout_stateful(_unwrap(x), _unwrap(_state(device)), 0.5, True)
    assert _C_engine.compile.current_tracer() is None


def test_dropout_stateful_records_one_node_and_keeps_the_tracer(device: str) -> None:
    """The success path: one ``dropout_stateful`` node, no nested ``dropout``, tracer kept."""
    x = lucid.randn(4, 4, device=device)
    state = _state(device)
    with _tracing() as tracer:
        _C_engine.nn.dropout_stateful(_unwrap(x), _unwrap(state), 0.5, True)
        assert _C_engine.compile.current_tracer() is tracer
        lucid.relu(x)
    names = [node.name for node in tracer.graph.ops]
    assert names.count("dropout_stateful") == 1
    assert "dropout" not in names
    assert "relu" in names


def _make_refused_cat(device: str) -> Callable[[], None]:
    a = lucid.ones((2, 3), device=device)
    b = lucid.ones((2, 4), device=device)

    def _run() -> None:
        lucid.cat([a, b], dim=0)

    return _run


def _make_refused_gather_rank(device: str) -> Callable[[], None]:
    src = lucid.ones((2, 3), device=device)
    idx = lucid.tensor([0, 1], dtype=lucid.int64, device=device)

    def _run() -> None:
        lucid.gather(src, 1, idx)

    return _run


def _make_refused_gather_empty_axis(device: str) -> Callable[[], None]:
    src = lucid.zeros((2, 0), device=device)
    idx = lucid.zeros((2, 1), dtype=lucid.int64, device=device)

    def _run() -> None:
        lucid.gather(src, 1, idx)

    return _run


def _make_refused_dropout_stateful(device: str) -> Callable[[], None]:
    x = lucid.ones((4,), dtype=lucid.bool_, device=device)
    st = _state(device)

    def _run() -> None:
        _C_engine.nn.dropout_stateful(_unwrap(x), _unwrap(st), 0.5, True)

    return _run


@pytest.mark.parametrize(
    ("make_trigger", "expected_error"),
    [
        pytest.param(_make_refused_cat, (RuntimeError, ValueError), id="cat-shape-mismatch"),
        pytest.param(
            _make_refused_gather_rank, (RuntimeError, ValueError), id="gather-rank-mismatch"
        ),
        pytest.param(_make_refused_gather_empty_axis, IndexError, id="gather-empty-axis"),
        pytest.param(_make_refused_dropout_stateful, RuntimeError, id="dropout-stateful-bool"),
    ],
)
def test_refused_op_leaves_op_count_unchanged(
    device: str,
    make_trigger: Callable[[str], Callable[[], None]],
    expected_error: type[Exception] | tuple[type[Exception], ...],
) -> None:
    """A throwing op pops its pending node, both on an empty trace and after earlier ops."""
    seed = lucid.randn(3, device=device)
    trigger = make_trigger(device)
    with _tracing() as tracer:
        lucid.relu(seed)
        before = tracer.op_count()
        assert before == 1
        with pytest.raises(expected_error):
            trigger()
        assert tracer.op_count() == before
        lucid.neg(seed)
        assert tracer.op_count() == before + 1
    assert [node.name for node in tracer.graph.ops] == ["relu", "neg"]


def test_refused_gather_oob_cpu_leaves_op_count_unchanged() -> None:
    """CPU gather validates index bounds inside the kernel after OpScopeFull opened."""
    src = lucid.ones((2, 3), device="cpu")
    idx = lucid.tensor([[0, 99]], dtype=lucid.int64, device="cpu")
    with _tracing() as tracer:
        with pytest.raises(IndexError):
            lucid.gather(src, 1, idx)
        assert tracer.op_count() == 0


@pytest.mark.parametrize(
    ("make_trigger", "expected_error"),
    [
        pytest.param(_make_refused_cat, (RuntimeError, ValueError), id="cat-shape-mismatch"),
        pytest.param(
            _make_refused_gather_rank, (RuntimeError, ValueError), id="gather-rank-mismatch"
        ),
        pytest.param(_make_refused_gather_empty_axis, IndexError, id="gather-empty-axis"),
        pytest.param(_make_refused_dropout_stateful, RuntimeError, id="dropout-stateful-bool"),
    ],
)
def test_compiled_catch_and_continue_compiles_without_fallback(
    make_trigger: Callable[[str], Callable[[], None]],
    expected_error: type[Exception] | tuple[type[Exception], ...],
) -> None:
    """A compiled function that catches a refused op compiles (entries=1) and replays."""
    trigger = make_trigger("metal")
    caught_refusal = 0

    def fn(x: lucid.Tensor) -> lucid.Tensor:
        nonlocal caught_refusal
        try:
            trigger()
        except expected_error:
            caught_refusal += 1
        return lucid.relu(x) + 1.0

    compiled = lucid.compile(fn)
    x1 = lucid.randn((4, 4), device="metal")
    x2 = lucid.randn((4, 4), device="metal")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y1 = compiled(x1)
        y2 = compiled(x2)

    assert caught_refusal == 1, "expected the trace call to hit and catch the refusal"
    assert caught == [], [str(w.message) for w in caught]
    info = compiled.cache_info()
    assert info["entries"] == 1
    assert len(info["eager_only"]) == 0
    assert lucid.allclose(y1, fn(x1), atol=1e-5, rtol=1e-5)
    assert lucid.allclose(y2, fn(x2), atol=1e-5, rtol=1e-5)
