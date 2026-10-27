"""RNG: ``manual_seed`` / ``Generator`` / state save+restore / reproducibility."""

import numpy as np
import pytest

import lucid


class TestManualSeed:
    def test_reproduces_rand(self) -> None:
        lucid.manual_seed(42)
        a = lucid.rand(8).numpy()
        lucid.manual_seed(42)
        b = lucid.rand(8).numpy()
        np.testing.assert_array_equal(a, b)

    def test_reproduces_randn(self) -> None:
        lucid.manual_seed(7)
        a = lucid.randn(16).numpy()
        lucid.manual_seed(7)
        b = lucid.randn(16).numpy()
        np.testing.assert_array_equal(a, b)

    def test_initial_seed_returns_set_seed(self) -> None:
        lucid.manual_seed(99)
        assert lucid.initial_seed() == 99


class TestSeedFn:
    def test_returns_int(self) -> None:
        s = lucid.seed()
        assert isinstance(s, int)

    def test_seeds_default_generator(self) -> None:
        s = lucid.seed()
        assert lucid.initial_seed() == s

    def test_subsequent_seeds_differ(self) -> None:
        # Drawn from os.urandom — collision probability is ≈ 0.
        a = lucid.seed()
        b = lucid.seed()
        assert a != b


class TestRngState:
    def test_state_shape(self) -> None:
        st = lucid.get_rng_state()
        assert st.shape == (2,)
        assert st.dtype == lucid.int64

    def test_round_trip_reproduces_samples(self) -> None:
        lucid.manual_seed(123)
        _ = lucid.rand(4)  # advance counter.
        state = lucid.get_rng_state()
        first = lucid.rand(8).numpy()

        lucid.set_rng_state(state)
        second = lucid.rand(8).numpy()
        np.testing.assert_array_equal(first, second)

    def test_set_rng_state_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            lucid.set_rng_state(lucid.tensor([1, 2, 3], dtype=lucid.int64))


class TestGenerator:
    def test_construct(self) -> None:
        g = lucid.Generator(seed=10)
        assert g.initial_seed() == 10
        assert g.counter == 0

    def test_set_seed_resets_counter(self) -> None:
        g = lucid.Generator(seed=10)
        # Manually push the counter, then reseeding should reset it.
        g.counter = 5
        g.set_seed(20)
        assert g.initial_seed() == 20
        assert g.counter == 0

    def test_isolated_streams(self) -> None:
        g1 = lucid.Generator(seed=1)
        g2 = lucid.Generator(seed=2)
        a = lucid.rand(4, generator=g1).numpy()
        b = lucid.rand(4, generator=g2).numpy()
        assert not (a == b).all()

    def test_explicit_generator_does_not_affect_default(self) -> None:
        lucid.manual_seed(0)
        baseline = lucid.rand(4).numpy()

        # Drawing through an explicit Generator must not advance the
        # default stream.
        lucid.manual_seed(0)
        g = lucid.Generator(seed=999)
        _ = lucid.rand(100, generator=g)
        same = lucid.rand(4).numpy()
        np.testing.assert_array_equal(baseline, same)

    def test_no_default_override_binding(self) -> None:
        assert not hasattr(lucid._C_engine, "_set_default_override")
