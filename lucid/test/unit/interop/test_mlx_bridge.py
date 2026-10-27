"""Zero-copy DLPack between Lucid's Metal tensors and ``mlx.core``.

Lucid's GPU storage *is* an ``mlx::core::array``, and both frameworks sit
on the same unified memory, yet handing a tensor from one to the other
used to cost a GPU download and a re-upload: the DLPack export went
through NumPy, which can only describe host memory, so a Metal tensor was
announced as ``kDLCPU`` and copied to make that true.

MLX already spoke the other dialect — it tags its own capsules
``kDLMetal`` and its importer shares pages — so what was missing was on
Lucid's side.

Sharing is asserted by *writing through one side and reading the other*.
Equal values would also hold for a copy; only a write proves one
allocation. The lifetime tests matter just as much: a capsule outliving
the tensor it came from is the failure mode that yields a use-after-free
rather than a wrong number, and it is invisible to a test that keeps
every reference alive.
"""

import gc

import numpy as np
import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

mx = pytest.importorskip("mlx.core", reason="the bridge's whole purpose is MLX")

pytestmark = pytest.mark.skipif(
    not metal_available(), reason="the Metal dialect needs a Metal device"
)


def _host_view(arr: object) -> np.ndarray:  # type: ignore[type-arg]
    """A writable NumPy view over an MLX array's own pages."""
    mx.eval(arr)
    return np.asarray(memoryview(arr))


class TestLucidToMlx:
    def test_mlx_reads_lucid_pages(self) -> None:
        t = lucid.arange(6, dtype=lucid.float32).reshape(2, 3).to("metal")

        a = mx.from_dlpack(t)

        assert a.shape == (2, 3)
        np.testing.assert_array_equal(_host_view(a), t.numpy())

    def test_a_write_through_mlx_is_visible_in_lucid(self) -> None:
        # Equal values prove nothing; a write proves one allocation.
        t = lucid.arange(6, dtype=lucid.float32).reshape(2, 3).to("metal")

        _host_view(mx.from_dlpack(t))[0, 0] = 99.0

        assert t.numpy()[0, 0] == 99.0

    def test_the_capsule_outlives_the_tensor(self) -> None:
        """The export holds the MLX array, not just the buffer address."""
        t = lucid.arange(6, dtype=lucid.float32).reshape(2, 3).to("metal")
        a = mx.from_dlpack(t)

        del t
        gc.collect()

        np.testing.assert_array_equal(_host_view(a).ravel(), np.arange(6, dtype=np.float32))

    @pytest.mark.parametrize(
        "dtype",
        [lucid.float32, lucid.float16, lucid.int32, lucid.int64, lucid.bool_],
    )
    def test_dtypes_survive(self, dtype: object) -> None:
        t = lucid.ones(2, 3, dtype=dtype).to("metal")

        a = mx.from_dlpack(t)

        assert a.shape == (2, 3)
        assert float(_host_view(a).sum()) == 6.0


class TestMlxToLucid:
    def test_lucid_adopts_an_mlx_array_on_metal(self) -> None:
        a = mx.arange(6, dtype=mx.float32).reshape(2, 3)

        t = lucid.from_dlpack(a)

        assert t.is_metal
        assert t.shape == (2, 3)
        np.testing.assert_array_equal(t.numpy(), _host_view(a))

    def test_a_write_through_mlx_is_visible_in_the_adopted_tensor(self) -> None:
        a = mx.arange(6, dtype=mx.float32).reshape(2, 3)
        t = lucid.from_dlpack(a)

        _host_view(a)[0, 0] = -42.0

        assert t.numpy()[0, 0] == -42.0

    def test_the_tensor_outlives_the_mlx_array(self) -> None:
        """The import owns the producer's block until its storage dies."""
        a = mx.arange(6, dtype=mx.float32).reshape(2, 3)
        t = lucid.from_dlpack(a)

        del a
        gc.collect()

        np.testing.assert_array_equal(t.numpy().ravel(), np.arange(6, dtype=np.float32))

    def test_round_trip_returns_the_same_pages(self) -> None:
        t = lucid.arange(6, dtype=lucid.float32).reshape(2, 3).to("metal")

        back = lucid.from_dlpack(mx.from_dlpack(t))

        assert back.is_metal
        _host_view(mx.from_dlpack(back))[0, 0] = 7.0
        assert t.numpy()[0, 0] == 7.0


class TestRefusals:
    """Each of these would otherwise be a wrong answer, not an error."""

    def test_a_host_tensor_has_no_metal_capsule(self) -> None:
        with pytest.raises(ValueError, match="on the CPU"):
            _C_engine.to_dlpack_metal(lucid.ones(2, 3)._impl)

    def test_float64_never_gets_far_enough_to_export(self) -> None:
        # The exporter refuses float64, but that branch is unreachable in
        # practice: Metal has no double, so the tensor cannot be moved
        # there at all.  Pinned so the two refusals stay consistent —
        # whichever fires, it is never a capsule over absent storage.
        # The engine raises its own ``NotImplementedError``, which is not
        # the builtin of that name — catching it needs the engine's class.
        with pytest.raises(_C_engine.NotImplementedError, match="float64"):
            lucid.ones(2, dtype=lucid.float64).to("metal")

        with pytest.raises(ValueError, match="on the CPU"):
            _C_engine.to_dlpack_metal(lucid.ones(2, dtype=lucid.float64)._impl)

    def test_a_capsule_is_consumed_once(self) -> None:
        t = lucid.ones(2, 3).to("metal")
        capsule = t.__dlpack__()

        _C_engine.from_dlpack_metal(capsule)

        with pytest.raises(ValueError, match="consumed once"):
            _C_engine.from_dlpack_metal(capsule)

    def test_a_host_capsule_is_not_adopted_as_metal(self) -> None:
        host = lucid.ones(2, 3).__dlpack__()

        with pytest.raises(ValueError, match="device type"):
            _C_engine.from_dlpack_metal(host)


class TestACapsuleThatBeginsPastItsBuffer:
    """A DLPack tensor's first element is at ``data + byte_offset``.

    The import built its array from ``data`` alone and never read the
    offset, so a capsule that begins later was adopted as though it began
    at the buffer — every element shifted by ``byte_offset / itemsize``,
    with no error. The packed-strides check above does not catch it: a
    leading-axis slice of a contiguous buffer is still row-major packed
    and merely starts further in.

    It is reachable from any producer that slices, ``mlx.core`` included:
    ``mx.arange(12)[4:]`` hands over a capsule with ``byte_offset`` 16.

    Refused rather than honoured, because honouring it needs a slice this
    layer cannot express — ``buffer_to_array`` takes the same position for
    the same reason, and this path was bypassing it. A refusal is
    recoverable; a tensor quietly reading the wrong four elements is not.
    """

    def test_an_offset_slice_is_refused(self) -> None:
        import mlx.core as mx

        whole = mx.arange(12, dtype=mx.float32)
        mx.eval(whole)

        with pytest.raises(ValueError, match="byte_offset"):
            lucid.from_dlpack(whole[4:])

    def test_the_message_names_the_offset(self) -> None:
        """So the reader can tell how far off it would have been."""
        import mlx.core as mx

        whole = mx.arange(12, dtype=mx.float32)
        mx.eval(whole)

        with pytest.raises(ValueError) as refusal:
            lucid.from_dlpack(whole[4:])
        assert "16" in str(refusal.value)  # four float32 elements in

    def test_lucids_own_slice_round_trips(self) -> None:
        """Lucid's export of a slice used to claim four times its offset.

        MLX keeps an array's offset in bytes and the export multiplied it by
        the item size again, so ``x[3:]`` of float32 claimed byte_offset 48
        and the import refused it — while reading that number from a capsule
        it had already released.  A slice is now exported from a buffer of
        its own.
        """
        x = lucid.arange(12.0).to("metal")
        back = lucid.from_dlpack(x[3:])
        assert back.tolist() == [float(v) for v in range(3, 12)]
        # And MLX, which honours byte_offset, read 36 bytes too far in.
        np.testing.assert_array_equal(
            _host_view(mx.from_dlpack(x[3:])), np.arange(3, 12, dtype=np.float32)
        )

    def test_a_capsule_that_starts_at_its_buffer_is_still_adopted(self) -> None:
        """As narrow as the defect: the ordinary case is untouched."""
        import mlx.core as mx

        whole = mx.arange(12, dtype=mx.float32)
        mx.eval(whole)

        adopted = lucid.from_dlpack(whole)
        assert [float(v) for v in adopted.cpu().reshape(-1)] == list(range(12))

    def test_a_lucid_tensor_still_round_trips(self) -> None:
        t = lucid.arange(6, dtype=lucid.float32).reshape(2, 3).to("metal")
        back = lucid.from_dlpack(t)
        assert [float(v) for v in back.cpu().reshape(-1)] == list(range(6))
