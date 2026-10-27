"""Algebraic identities — properties any correct numeric op must obey.

These are property-style tests: rather than checking against a fixed
reference value, they assert that ``a + b == b + a`` etc. on random
inputs.  When one side drifts the test fails on every device, which is
the quickest signal that an op miscompiles.
"""

import numpy as np

import lucid
from lucid.test._helpers.compare import assert_close


def _rand(shape: tuple[int, ...], device: str, seed: int = 0) -> lucid.Tensor:
    rng = np.random.default_rng(seed)
    return lucid.tensor(rng.uniform(-1.0, 1.0, size=shape).astype(np.float32), device=device)


# ── + and × algebra ─────────────────────────────────────────────────────


class TestAddInvariants:
    def test_commutativity(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        b = _rand((4, 5), device, seed=1)
        assert_close((a + b), (b + a), atol=0.0)

    def test_associativity(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        b = _rand((4, 5), device, seed=1)
        c = _rand((4, 5), device, seed=2)
        assert_close(((a + b) + c), (a + (b + c)), atol=1e-5)

    def test_zero_identity(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        z = lucid.zeros(4, 5, device=device)
        assert_close((a + z), a, atol=0.0)


class TestMulInvariants:
    def test_commutativity(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        b = _rand((4, 5), device, seed=1)
        assert_close((a * b), (b * a), atol=0.0)

    def test_one_identity(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        one = lucid.ones(4, 5, device=device)
        assert_close((a * one), a, atol=0.0)

    def test_zero_annihilator(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        z = lucid.zeros(4, 5, device=device)
        assert_close((a * z), z, atol=0.0)

    def test_distributivity(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        b = _rand((4, 5), device, seed=1)
        c = _rand((4, 5), device, seed=2)
        assert_close((a * (b + c)), (a * b + a * c), atol=1e-5)


# ── matmul algebra ──────────────────────────────────────────────────────


class TestMatmulInvariants:
    def test_associativity(self, device: str) -> None:
        a = _rand((3, 4), device, seed=0)
        b = _rand((4, 5), device, seed=1)
        c = _rand((5, 2), device, seed=2)
        assert_close((a @ b) @ c, a @ (b @ c), atol=1e-4)

    def test_identity(self, device: str) -> None:
        a = _rand((4, 4), device, seed=0)
        eye = lucid.eye(4, device=device)
        assert_close(a @ eye, a, atol=1e-5)
        assert_close(eye @ a, a, atol=1e-5)

    def test_transpose_swap(self, device: str) -> None:
        # (A B)^T = B^T A^T
        a = _rand((3, 4), device, seed=0)
        b = _rand((4, 2), device, seed=1)
        lhs = (a @ b).mT
        rhs = b.mT @ a.mT
        assert_close(lhs, rhs, atol=1e-5)


# ── exp / log inverses ──────────────────────────────────────────────────


class TestExpLogInvariants:
    def test_log_exp(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        assert_close(a.exp().log(), a, atol=1e-5)

    def test_exp_log(self, device: str) -> None:
        # log defined for positives only.
        a = lucid.tensor(
            np.random.default_rng(0).uniform(0.1, 5.0, size=(4, 5)).astype(np.float32),
            device=device,
        )
        assert_close(a.log().exp(), a, atol=1e-4)


# ── reshape / transpose involutions ────────────────────────────────────


class TestShapeInvariants:
    def test_double_negate(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        assert_close(-(-a), a, atol=0.0)

    def test_double_transpose(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        assert_close(a.mT.mT, a, atol=0.0)

    def test_reshape_round_trip(self, device: str) -> None:
        a = _rand((4, 6), device, seed=0)
        b = a.reshape(8, 3).reshape(4, 6)
        assert_close(b, a, atol=0.0)


# ── sum / mean linearity ─────────────────────────────────────────────────


class TestReductionLinearity:
    def test_sum_linearity(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        b = _rand((4, 5), device, seed=1)
        lhs = lucid.sum(a + b).item()
        rhs = lucid.sum(a).item() + lucid.sum(b).item()
        assert abs(lhs - rhs) < 1e-4

    def test_sum_scaling(self, device: str) -> None:
        a = _rand((4, 5), device, seed=0)
        lhs = lucid.sum(a * 3.0).item()
        rhs = 3.0 * lucid.sum(a).item()
        assert abs(lhs - rhs) < 1e-4
