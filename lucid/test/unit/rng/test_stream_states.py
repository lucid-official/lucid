"""Every stream's position can be read and put back without a tensor op.

A compile that records a call drawing random values advances the default
stream and the call's own ``lucid.Generator``; the first real run then
drew past what eager would have (LCD-607).  ``_stream_states`` reads the
position of the default stream and of every live ``Generator`` from their
own fields — safe inside a recording — and ``_restore_stream_states``
returns them there.
"""

import gc

import lucid
from lucid._factories._generator import _LIVE, _restore_stream_states, _stream_states


def test_restoring_returns_every_stream_to_its_start(device: str) -> None:
    lucid.manual_seed(5)
    g = lucid.Generator(7)
    _ = lucid.rand(3, generator=g, device=device)
    states = _stream_states()
    want_default = lucid.randn(4, device=device).tolist()
    want_own = lucid.randn(4, generator=g, device=device).tolist()
    _ = lucid.rand(100, device=device)
    _ = lucid.rand(100, generator=g, device=device)
    _restore_stream_states(states)
    assert lucid.randn(4, device=device).tolist() == want_default
    assert lucid.randn(4, generator=g, device=device).tolist() == want_own


def test_a_reseeded_stream_gets_its_key_back() -> None:
    lucid.manual_seed(3)
    g = lucid.Generator(11)
    states = _stream_states()
    lucid.manual_seed(99)
    g.manual_seed(42)
    _restore_stream_states(states)
    assert lucid.initial_seed() == 3 and g.initial_seed() == 11
    assert int(g.counter) == 0


def test_reading_runs_no_tensor_op() -> None:
    g = lucid.Generator(1)
    with lucid.compile._tracing() as tracer:
        _restore_stream_states(_stream_states())
    assert len(tracer.graph.ops) == 0
    del g


def test_a_generator_made_since_is_left_alone() -> None:
    states = _stream_states()
    late = lucid.Generator(9)
    _ = lucid.rand(5, generator=late)
    counter = int(late.counter)
    _restore_stream_states(states)
    assert int(late.counter) == counter


def test_the_registry_holds_no_generator_alive() -> None:
    g = lucid.Generator(2)
    assert g in _LIVE
    before = len(_LIVE)
    del g
    gc.collect()
    assert len(_LIVE) == before - 1
