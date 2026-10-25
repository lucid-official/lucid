"""Shape-manipulation ops — reshape / permute / cat / stack / split / pad / ..."""

import numpy as np
import pytest

import lucid


class TestReshape:
    def test_basic(self, device: str) -> None:
        t = lucid.arange(0.0, 12.0, 1.0, device=device)
        out = t.reshape(3, 4)
        assert out.shape == (3, 4)

    def test_inferred_dim(self, device: str) -> None:
        t = lucid.arange(0.0, 12.0, 1.0, device=device)
        out = t.reshape(3, -1)
        assert out.shape == (3, 4)


class TestView:
    def test_compatible(self, device: str) -> None:
        t = lucid.arange(0.0, 8.0, 1.0, device=device)
        out = t.view(2, 4)
        assert out.shape == (2, 4)


class TestPermute:
    def test_swap(self, device: str) -> None:
        t = lucid.zeros(2, 3, 4, device=device)
        out = t.permute(2, 0, 1)
        assert out.shape == (4, 2, 3)


class TestTranspose:
    def test_2d_default(self, device: str) -> None:
        # ``transpose`` takes no args — flips a 2-D tensor.
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        np.testing.assert_array_equal(t.transpose().numpy(), [[1.0, 3.0], [2.0, 4.0]])

    def test_swapaxes(self, device: str) -> None:
        # ``swapaxes`` is the explicit-axis form.
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        np.testing.assert_array_equal(
            lucid.swapaxes(t, 0, 1).numpy(), [[1.0, 3.0], [2.0, 4.0]]
        )

    def test_mT_property(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        np.testing.assert_array_equal(t.mT.numpy(), [[1.0, 3.0], [2.0, 4.0]])


class TestUnsqueezeSqueeze:
    def test_unsqueeze(self, device: str) -> None:
        t = lucid.zeros(3, 4, device=device)
        out = t.unsqueeze(0)
        assert out.shape == (1, 3, 4)

    def test_squeeze(self, device: str) -> None:
        t = lucid.zeros(1, 3, 1, device=device)
        out = t.squeeze()
        assert out.shape == (3,)


class TestFlattenUnflatten:
    def test_flatten(self, device: str) -> None:
        t = lucid.zeros(2, 3, 4, device=device)
        out = t.flatten()
        assert out.shape == (24,)


class TestExpand:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([[1.0]], device=device)
        out = t.expand(3, 4).numpy()
        np.testing.assert_array_equal(out, np.ones((3, 4)))


class TestCatStack:
    def test_cat_dim0(self, device: str) -> None:
        a = lucid.tensor([[1.0, 2.0]], device=device)
        b = lucid.tensor([[3.0, 4.0]], device=device)
        out = lucid.cat([a, b], dim=0).numpy()
        np.testing.assert_array_equal(out, [[1.0, 2.0], [3.0, 4.0]])

    def test_cat_dim1(self, device: str) -> None:
        a = lucid.tensor([[1.0, 2.0]], device=device)
        b = lucid.tensor([[3.0, 4.0]], device=device)
        out = lucid.cat([a, b], dim=1).numpy()
        np.testing.assert_array_equal(out, [[1.0, 2.0, 3.0, 4.0]])

    def test_stack(self, device: str) -> None:
        a = lucid.tensor([1.0, 2.0], device=device)
        b = lucid.tensor([3.0, 4.0], device=device)
        out = lucid.stack([a, b], dim=0)
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(out.numpy(), [[1.0, 2.0], [3.0, 4.0]])

    @pytest.mark.parametrize(
        "join",
        [
            "cat",
            "concat",
            "stack",
            "hstack",
            "vstack",
            "dstack",
            "column_stack",
            "row_stack",
        ],
    )
    def test_mixed_dtypes_join_at_the_common_dtype(
        self, device: str, join: str
    ) -> None:
        # The engine joins one dtype only, so an int64 beside a float32
        # raised DtypeMismatch; the reference joins them in float32.
        ints = lucid.tensor([1, 2], dtype=lucid.int64, device=device)
        floats = lucid.tensor([0.5, 1.5], device=device)
        fn = getattr(lucid, join)
        out = fn([ints, floats])
        assert out.dtype == lucid.float32
        np.testing.assert_array_equal(out.numpy(), fn([ints.float(), floats]).numpy())

    def test_join_promotion_orders_kind_then_width(self, device: str) -> None:
        flag = lucid.tensor([True], device=device)
        narrow = lucid.tensor([2], dtype=lucid.int32, device=device)
        wide = lucid.tensor([3], dtype=lucid.int64, device=device)
        assert lucid.cat([flag, narrow]).dtype == lucid.int32
        assert lucid.cat([narrow, wide]).dtype == lucid.int64
        assert lucid.cat([flag, narrow, wide]).tolist() == [1, 2, 3]
        assert lucid.cat([narrow, narrow]).dtype == lucid.int32

    def test_a_promoted_join_still_carries_gradient(self, device: str) -> None:
        ints = lucid.tensor([1, 2], dtype=lucid.int64, device=device)
        floats = lucid.tensor([0.5, 1.5], device=device, requires_grad=True)
        lucid.cat([ints, floats]).sum().backward()
        assert floats.grad is not None
        assert floats.grad.tolist() == [1.0, 1.0]


class TestSplitChunk:
    def test_split_size(self, device: str) -> None:
        t = lucid.arange(0.0, 6.0, 1.0, device=device)
        parts = lucid.split(t, 2, dim=0)
        assert len(parts) == 3
        assert parts[0].shape == (2,)

    def test_chunk(self, device: str) -> None:
        t = lucid.arange(0.0, 6.0, 1.0, device=device)
        parts = lucid.chunk(t, 3, dim=0)
        assert len(parts) == 3


class TestUnbind:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        rows = lucid.unbind(t, dim=0)
        assert len(rows) == 2
        np.testing.assert_array_equal(rows[0].numpy(), [1.0, 2.0])


class TestNarrow:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([0.0, 1.0, 2.0, 3.0, 4.0], device=device)
        out = t.narrow(0, 1, 3).numpy()
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


class TestRollFlip:
    def test_roll(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        out = lucid.roll(t, [1], [0]).numpy()
        np.testing.assert_array_equal(out, [4.0, 1.0, 2.0, 3.0])

    def test_flip(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = lucid.flip(t, dims=[0]).numpy()
        np.testing.assert_array_equal(out, [3.0, 2.0, 1.0])

    def test_fliplr(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        out = lucid.fliplr(t).numpy()
        np.testing.assert_array_equal(out, [[2.0, 1.0], [4.0, 3.0]])

    def test_flipud(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        out = lucid.flipud(t).numpy()
        np.testing.assert_array_equal(out, [[3.0, 4.0], [1.0, 2.0]])


class TestTrilTriu:
    def test_tril(self, device: str) -> None:
        t = lucid.ones(3, 3, device=device)
        out = lucid.tril(t).numpy()
        expected = np.tril(np.ones((3, 3)))
        np.testing.assert_array_equal(out, expected)

    def test_triu(self, device: str) -> None:
        t = lucid.ones(3, 3, device=device)
        out = lucid.triu(t).numpy()
        expected = np.triu(np.ones((3, 3)))
        np.testing.assert_array_equal(out, expected)


class TestTile:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0], device=device)
        out = lucid.tile(t, (2,)).numpy()
        np.testing.assert_array_equal(out, [1.0, 2.0, 1.0, 2.0])


class TestSortArgsort:
    def test_sort_ascending(self, device: str) -> None:
        t = lucid.tensor([3.0, 1.0, 4.0, 1.0, 5.0], device=device)
        out = lucid.sort(t, dim=0)
        np.testing.assert_array_equal(out.values.numpy(), [1.0, 1.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(out.indices.numpy(), [1, 3, 0, 2, 4])

    def test_argsort(self, device: str) -> None:
        t = lucid.tensor([3.0, 1.0, 4.0], device=device)
        out = lucid.argsort(t, dim=0).numpy()
        np.testing.assert_array_equal(out, [1, 0, 2])


class TestTopK:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 5.0, 3.0, 4.0, 2.0], device=device)
        out = lucid.topk(t, 3)
        # ``topk`` may return a tensor or a (values, indices) tuple
        # depending on engine surface.
        vals = out[0] if isinstance(out, tuple) else out
        assert sorted(vals.numpy().tolist()) == [3.0, 4.0, 5.0]


class TestRepeatInterleave:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = lucid.repeat_interleave(t, 2, dim=0).numpy()
        np.testing.assert_array_equal(out, [1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
