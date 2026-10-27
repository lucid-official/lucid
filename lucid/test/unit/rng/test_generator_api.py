"""``lucid.Generator`` carries the reference's methods (LCD-374).

It was the bare engine class, so ``Generator().manual_seed(0)`` raised
``AttributeError``.  Each method is held to the reference's contract —
what it returns, how it reads a seed — and a captured state replays the
same draws on both devices.
"""

from types import ModuleType

import pytest

import lucid


@pytest.mark.parametrize("seed", [0, 7, 2**63 - 1, 2**63, 2**64 - 1, -1, -(2**63)])
def test_manual_seed_and_initial_seed(ref: ModuleType, seed: int) -> None:
    g = lucid.Generator()
    r = ref.Generator()
    assert g.manual_seed(seed) is g
    assert r.manual_seed(seed) is r
    assert g.initial_seed() == r.initial_seed()
    assert lucid.Generator(seed).initial_seed() == r.initial_seed()


@pytest.mark.parametrize("seed", [1.5, 2**64, -(2**63) - 1])
def test_a_seed_the_reference_refuses_is_refused(ref: ModuleType, seed: int) -> None:
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        ref.Generator().manual_seed(seed)
    with pytest.raises((ValueError, TypeError)):
        lucid.Generator().manual_seed(seed)
    with pytest.raises((ValueError, TypeError)):
        lucid.Generator(seed)


def test_set_state_returns_the_generator(ref: ModuleType) -> None:
    g, r = lucid.Generator(), ref.Generator()
    assert g.set_state(g.get_state()) is g
    assert r.set_state(r.get_state()) is r


@pytest.mark.parametrize("seed", [3, 2**64 - 5])
def test_a_captured_state_replays_the_same_draws(device: str, seed: int) -> None:
    g = lucid.Generator(seed)
    _ = lucid.rand(5, generator=g, device=device)
    state = g.get_state()
    assert state.dtype == lucid.int64 and state.shape == (2,)
    first = lucid.randn(4, generator=g, device=device).tolist()
    again = lucid.randn(4, generator=g.set_state(state), device=device).tolist()
    assert first == again
    assert g.initial_seed() == seed


def test_manual_seed_rewinds_the_stream(device: str) -> None:
    g = lucid.Generator().manual_seed(11)
    first = lucid.rand(3, generator=g, device=device).tolist()
    assert lucid.rand(3, generator=g.manual_seed(11), device=device).tolist() == first


def test_a_generator_leaves_the_default_stream_alone() -> None:
    lucid.manual_seed(0)
    want = lucid.rand(3).tolist()
    lucid.manual_seed(0)
    _ = lucid.rand(10, generator=lucid.Generator().manual_seed(5))
    assert lucid.rand(3).tolist() == want


def test_the_default_state_round_trips_a_wide_seed() -> None:
    lucid.manual_seed(2**64 - 3)
    state = lucid.get_rng_state()
    want = lucid.rand(3).tolist()
    lucid.set_rng_state(state)
    assert lucid.rand(3).tolist() == want
    assert lucid.initial_seed() == 2**64 - 3


def test_seed_reseeds_from_entropy_as_the_reference_does() -> None:
    # ``seed`` read the key as a property until LCD-410; it is the
    # reference's method now, and ``initial_seed()`` reads the key.
    g = lucid.Generator(9)
    assert g.initial_seed() == 9
    _ = lucid.rand(4, generator=g)
    new = g.seed()
    assert isinstance(new, int) and 0 <= new < 2**63
    assert g.initial_seed() == new
    assert g.counter == 0
    want = lucid.rand(3, generator=g).tolist()
    assert lucid.rand(3, generator=lucid.Generator(new)).tolist() == want
    assert g.seed() != new  # 2**-63 to collide


def test_counter_stays_readable_and_writable() -> None:
    g = lucid.Generator(4)
    g.counter = 6
    assert g.counter == 6
    g.manual_seed(4)
    assert g.counter == 0


@pytest.mark.parametrize("seed", [True, 2.0, 2.5, -1])
def test_the_global_manual_seed_reads_its_seed_through_int(ref: ModuleType, seed: int) -> None:
    # The reference's global manual_seed calls int() on its seed; its
    # Generator.manual_seed does not, and neither does Lucid's.
    ref.manual_seed(seed)
    lucid.manual_seed(seed)
    assert lucid.initial_seed() == ref.initial_seed()


def test_generator_device_parameter_and_property(ref: ModuleType) -> None:
    assert lucid.Generator().device == lucid.device("cpu")
    assert lucid.Generator(device="cpu").device == lucid.device("cpu")
    assert lucid.Generator("cpu").device == lucid.device("cpu")
    assert lucid.Generator("metal").device == lucid.device("metal")
    assert lucid.Generator(7, device="metal").device == lucid.device("metal")
    assert lucid.Generator(7, device="metal").initial_seed() == 7
    assert ref.Generator(device="cpu").device.type == "cpu"
    with pytest.raises(TypeError, match="multiple values for argument 'device'"):
        lucid.Generator("cpu", device="metal")
    with pytest.raises(ValueError, match="Unknown device"):
        lucid.Generator(device="bogus")
