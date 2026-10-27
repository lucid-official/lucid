"""Indexing / scatter / gather ops."""

import numpy as np
import pytest

import lucid


class TestGather:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
        idx = lucid.tensor([[0, 2], [1, 0]], dtype=lucid.int64, device=device)
        out = lucid.gather(x, 1, idx).numpy()
        np.testing.assert_array_equal(out, [[1.0, 3.0], [5.0, 4.0]])


class TestScatter:
    def test_scatter(self, device: str) -> None:
        base = lucid.zeros(3, 4, device=device)
        idx = lucid.tensor([[0, 1], [2, 1], [0, 2]], dtype=lucid.int32, device=device)
        src = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        out = lucid.scatter(base, dim=1, index=idx, src=src).numpy()
        expected = np.zeros((3, 4))
        for r in range(3):
            for c in range(2):
                col = [[0, 1], [2, 1], [0, 2]][r][c]
                val = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]][r][c]
                expected[r, col] = val
        np.testing.assert_array_equal(out, expected)

    def test_scatter_add(self, device: str) -> None:
        base = lucid.zeros(4, device=device)
        idx = lucid.tensor([0, 0, 2], dtype=lucid.int32, device=device)
        src = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = base.scatter_add(0, idx, src).numpy()
        np.testing.assert_array_equal(out, [3.0, 0.0, 3.0, 0.0])


class TestIndexSelect:
    def test_dim0(self, device: str) -> None:
        x = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        idx = lucid.tensor([0, 2], dtype=lucid.int64, device=device)
        out = lucid.index_select(x, 0, idx).numpy()
        np.testing.assert_array_equal(out, [[1.0, 2.0], [5.0, 6.0]])


class TestMaskedSelect:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        mask = lucid.tensor([True, False, True, False], dtype=lucid.bool_, device=device)
        out = lucid.masked_select(x, mask).numpy()
        np.testing.assert_array_equal(out, [1.0, 3.0])


class TestMaskedFill:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        mask = lucid.tensor([True, False, True, False], dtype=lucid.bool_, device=device)
        out = lucid.masked_fill(x, mask, -1.0).numpy()
        np.testing.assert_array_equal(out, [-1.0, 2.0, -1.0, 4.0])

    @pytest.mark.parametrize("value", [float("-inf"), float("inf"), float("nan")])
    @pytest.mark.parametrize("dtype", [lucid.int64, lucid.int32, lucid.bool_])
    def test_a_non_finite_value_into_integers_is_refused(
        self, device: str, dtype: lucid.dtype, value: float
    ) -> None:
        # An integer tensor has no infinity or NaN; the engine's cast wrote
        # INT64_MIN for -inf with nothing to say so.
        x = lucid.tensor([1, 0, 1], dtype=dtype, device=device)
        mask = lucid.tensor([True, False, True], device=device)
        with pytest.raises(RuntimeError, match="masked_fill"):
            x.masked_fill(mask, value)
        with pytest.raises(RuntimeError, match="masked_fill"):
            lucid.masked_fill(x, mask, value)

    def test_a_finite_value_into_integers_is_cast(self, device: str) -> None:
        x = lucid.tensor([1, 2, 3], dtype=lucid.int64, device=device)
        mask = lucid.tensor([True, False, True], device=device)
        out = x.masked_fill(mask, -7.0)
        assert out.dtype == lucid.int64
        assert out.tolist() == [-7, 2, -7]

    def test_a_non_finite_value_into_floats_is_kept(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0], device=device)
        out = x.masked_fill(lucid.tensor([True, False], device=device), -np.inf)
        assert out.tolist() == [float("-inf"), 2.0]


class TestPut:
    def test_overwrite(self, device: str) -> None:
        x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        idx = lucid.tensor([0, 3], dtype=lucid.int64, device=device)
        src = lucid.tensor([10.0, 20.0], device=device)
        out = lucid.put(x, idx, src).numpy()
        np.testing.assert_array_equal(out, [[10.0, 2.0], [3.0, 20.0]])

    def test_accumulate(self, device: str) -> None:
        x = lucid.zeros(4, device=device)
        idx = lucid.tensor([0, 0, 1], dtype=lucid.int64, device=device)
        src = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = lucid.put(x, idx, src, accumulate=True).numpy()
        np.testing.assert_array_equal(out, [3.0, 3.0, 0.0, 0.0])


class TestIndexPut:
    def test_2d_overwrite(self, device: str) -> None:
        x = lucid.zeros(3, 4, device=device)
        out = lucid.index_put(
            x,
            (
                lucid.tensor([0, 1], dtype=lucid.int64, device=device),
                lucid.tensor([2, 3], dtype=lucid.int64, device=device),
            ),
            lucid.tensor([10.0, 20.0], device=device),
        ).numpy()
        expected = np.zeros((3, 4))
        expected[0, 2] = 10.0
        expected[1, 3] = 20.0
        np.testing.assert_array_equal(out, expected)

    def test_inplace(self, device: str) -> None:
        x = lucid.zeros(3, 4, device=device)
        ret = lucid.index_put_(
            x,
            (
                lucid.tensor([0], dtype=lucid.int64, device=device),
                lucid.tensor([1], dtype=lucid.int64, device=device),
            ),
            lucid.tensor([5.0], device=device),
        )
        assert ret is x
        assert x.numpy()[0, 1] == 5.0

    def test_inplace_under_no_grad_keeps_a_parameter_trainable(self) -> None:
        # The result is swapped in as the destination's impl; under no_grad
        # that impl does not require grad, so a Parameter written this way
        # silently stopped training.
        import lucid.nn as nn

        p = nn.Parameter(lucid.zeros(3, 4))
        with lucid.no_grad():
            lucid.index_put_(
                p,
                (
                    lucid.tensor([0], dtype=lucid.int64),
                    lucid.tensor([1], dtype=lucid.int64),
                ),
                lucid.tensor([5.0]),
            )
        assert p.requires_grad and p.is_leaf
        assert p.detach().numpy()[0, 1] == 5.0


class TestNonzero:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 0.0, 2.0, 0.0, 3.0], device=device)
        out = lucid.nonzero(t).numpy()
        # ``nonzero`` returns coords as a (n, ndim) tensor.
        np.testing.assert_array_equal(out.flatten(), [0, 2, 4])


class TestUnique:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 2.0, 3.0, 1.0], device=device)
        out = lucid.unique(t).numpy()
        np.testing.assert_array_equal(np.sort(out), [1.0, 2.0, 3.0])


class TestWhere:
    def test_basic(self, device: str) -> None:
        cond = lucid.tensor([True, False, True], dtype=lucid.bool_, device=device)
        a = lucid.tensor([1.0, 2.0, 3.0], device=device)
        b = lucid.tensor([10.0, 20.0, 30.0], device=device)
        out = lucid.where(cond, a, b).numpy()
        np.testing.assert_array_equal(out, [1.0, 20.0, 3.0])

    def test_mixed_dtype_branches_meet_at_the_common_dtype(self, device: str) -> None:
        # The engine selects between one dtype only, so an int64 branch
        # beside a float32 one raised DtypeMismatch.
        cond = lucid.tensor([True, False, True], device=device)
        ints = lucid.tensor([1, 2, 3], dtype=lucid.int64, device=device)
        floats = lucid.tensor([0.5, 1.5, 2.5], device=device)
        out = lucid.where(cond, ints, floats)
        assert out.dtype == lucid.float32
        assert out.tolist() == [1.0, 1.5, 3.0]
        assert lucid.where(cond, floats, ints).tolist() == [0.5, 2.0, 2.5]
        assert lucid.where(cond, ints, 0.5).tolist() == [1.0, 0.5, 3.0]
        # The method form takes the same path; it used to build the scalar
        # at the tensor's own dtype and truncate 0.5 to 0.
        assert ints.where(cond, 0.5).tolist() == [1.0, 0.5, 3.0]
        assert ints.where(cond, floats).dtype == lucid.float32

    def test_a_scalar_branch_never_widens_the_width(self, device: str) -> None:
        cond = lucid.tensor([True, False], device=device)
        half = lucid.tensor([1.0, 2.0], dtype=lucid.float16, device=device)
        assert lucid.where(cond, half, 0.5).dtype == lucid.float16
        ints = lucid.tensor([1, 2], dtype=lucid.int32, device=device)
        assert lucid.where(cond, ints, 7).dtype == lucid.int32
        wider = lucid.tensor([0.5, 0.5], device=device)
        assert lucid.where(cond, half, wider).dtype == lucid.float32


class TestTake:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        idx = lucid.tensor([0, 3], dtype=lucid.int64, device=device)
        # ``take`` indexes into the flattened view.
        out = lucid.take(x, idx).numpy()
        np.testing.assert_array_equal(out, [1.0, 4.0])
