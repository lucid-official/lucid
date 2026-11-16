"""A trace goes on recording after an op inside it throws (LCD-1042).

``dropout_stateful`` runs its nested ``dropout`` with the tracer detached, so the
graph holds one ``dropout_stateful`` node and no ``dropout`` under it.  That
nested draw throws for a bool or integer input and under determinism without a
generator; the detach used to be a bare ``set_current_tracer(nullptr)`` ...
``set_current_tracer(saved)`` pair, and the throw skipped the second half.  The
tracer stayed unset and the rest of the trace recorded nothing.

The detach is the engine's ``TracerDetach`` guard now.  These tests hold the
contract: whatever the nested call does, the tracer that was installed is the
tracer installed afterwards, and the next op is recorded into it.
"""

from collections.abc import Iterator

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.compile import _tracing

#: The three ways the nested draw refuses: the input is no float (bool and int64 are
#: each refused by a different check), or determinism is on and no generator was given.
_REFUSALS = [
    pytest.param(lucid.bool_, False, TypeError, id="bool-input"),
    pytest.param(lucid.int64, False, NotImplementedError, id="int64-input"),
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
        recorded = tracer.op_count()
        lucid.relu(probe)
        assert tracer.op_count() == recorded + 1, "the trace stopped recording after the throw"
    assert _C_engine.compile.current_tracer() is None


@pytest.mark.usefixtures("deterministic_restored")
def test_a_throw_does_not_install_a_tracer_that_was_not_there(device: str) -> None:
    """Outside a trace the detach is a no-op both ways: nothing is installed afterwards."""
    x = lucid.ones((4,), dtype=lucid.bool_, device=device)
    assert _C_engine.compile.current_tracer() is None
    with pytest.raises(TypeError):
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
