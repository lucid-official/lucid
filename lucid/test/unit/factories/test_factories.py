"""Tensor factory functions — value, shape, dtype, device coverage."""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import device_supports, devices_supporting
from lucid.test._helpers.compare import assert_close, assert_equal_int

# ── deterministic factories ──────────────────────────────────────────────


class TestZeros:
    def test_shape(self, device: str) -> None:
        t = lucid.zeros(3, 4, device=device)
        assert t.shape == (3, 4)

    def test_values(self, device: str, float_dtype: lucid.dtype) -> None:
        t = lucid.zeros(2, 3, dtype=float_dtype, device=device)
        assert_equal_int(t, np.zeros((2, 3)))

    def test_dtype_propagates(self, device: str) -> None:
        # Metal's refusal of float64 is asserted in test_metal_dtype_support.
        dtypes = (lucid.float32, lucid.float64, lucid.int32, lucid.int64)
        for dt in [d for d in dtypes if device_supports(device, d)]:
            t = lucid.zeros(2, dtype=dt, device=device)
            assert t.dtype == dt


class TestOnes:
    def test_values(self, device: str, float_dtype: lucid.dtype) -> None:
        t = lucid.ones(2, 3, dtype=float_dtype, device=device)
        assert_close(t, np.ones((2, 3)))

    def test_zero_dim(self, device: str) -> None:
        t = lucid.ones(device=device)
        assert t.shape == ()
        assert_close(t, np.array(1.0))


class TestEmpty:
    def test_shape_only(self, device: str) -> None:
        t = lucid.empty(4, 5, device=device)
        assert t.shape == (4, 5)

    @pytest.mark.parametrize("device", devices_supporting(lucid.float64))
    def test_dtype(self, device: str) -> None:
        t = lucid.empty(3, dtype=lucid.float64, device=device)
        assert t.dtype == lucid.float64


class TestFull:
    def test_known_value(self, device: str, float_dtype: lucid.dtype) -> None:
        t = lucid.full((2, 3), 7.5, dtype=float_dtype, device=device)
        assert_close(t, np.full((2, 3), 7.5))

    def test_int_value(self, device: str) -> None:
        t = lucid.full((4,), -3, dtype=lucid.int64, device=device)
        assert_equal_int(t, np.full(4, -3, dtype=np.int64))


class TestEye:
    def test_square_identity(self, device: str, float_dtype: lucid.dtype) -> None:
        t = lucid.eye(4, dtype=float_dtype, device=device)
        assert_close(t, np.eye(4))

    def test_rectangular(self, device: str) -> None:
        t = lucid.eye(3, 5, device=device)
        expected = np.eye(3, 5)
        assert_close(t, expected)


class TestArange:
    def test_basic(self, device: str) -> None:
        t = lucid.arange(0.0, 5.0, 1.0, device=device)
        assert_close(t, np.arange(0.0, 5.0, 1.0))

    def test_negative_step(self, device: str) -> None:
        t = lucid.arange(5, -1, -1, dtype=lucid.int32, device=device)
        assert_equal_int(t, np.arange(5, -1, -1, dtype=np.int32))

    def test_float_step(self, device: str) -> None:
        t = lucid.arange(0.0, 1.0, 0.25, device=device)
        assert_close(t, np.arange(0.0, 1.0, 0.25))


class TestArangeDtype:
    """Integer arguments give int64; one float among them gives the default float.

    The reference framework's rule.  Before it, ``arange(5)`` was always
    the default float, so every index built from it needed a cast.
    """

    def test_integers_give_int64(self, device: str) -> None:
        t = lucid.arange(0, 5, 1, device=device)
        assert t.dtype == lucid.int64
        assert t.tolist() == [0, 1, 2, 3, 4]

    def test_the_one_argument_form_keeps_int64(self, device: str) -> None:
        t = lucid.arange(5, device=device)
        assert t.dtype == lucid.int64
        assert t.tolist() == [0, 1, 2, 3, 4]

    @pytest.mark.parametrize(
        "args",
        [(5.0,), (0, 5.0), (0, 5, 0.5), (0.0, 5, 1)],
        ids=["one-argument", "end", "step", "start"],
    )
    def test_one_float_gives_the_default_float(self, device: str, args: tuple[float, ...]) -> None:
        t = lucid.arange(*args, device=device)
        assert t.dtype == lucid.get_default_dtype()
        assert_close(t, np.arange(*args))

    def test_the_default_float_is_the_one_in_force(self) -> None:
        previous = lucid.get_default_dtype()
        lucid.set_default_dtype(lucid.float64)
        try:
            assert lucid.arange(0, 1, 0.5).dtype == lucid.float64
            assert lucid.arange(3).dtype == lucid.int64
        finally:
            lucid.set_default_dtype(previous)

    @pytest.mark.parametrize(
        "dtype", [lucid.float32, lucid.int32, lucid.int64], ids=["float32", "int32", "int64"]
    )
    def test_an_explicit_dtype_wins(self, device: str, dtype: lucid.dtype) -> None:
        assert lucid.arange(4, dtype=dtype, device=device).dtype == dtype
        assert lucid.arange(0.0, 4.0, dtype=dtype, device=device).dtype == dtype

    def test_negative_step(self, device: str) -> None:
        t = lucid.arange(5, 0, -2, device=device)
        assert t.dtype == lucid.int64
        assert t.tolist() == [5, 3, 1]

    @pytest.mark.parametrize(
        ("args", "integral"),
        [((0,), True), ((3, 3), True), ((0.0,), False)],
        ids=["zero", "equal-bounds", "float-zero"],
    )
    def test_an_empty_range_keeps_its_dtype(
        self, device: str, args: tuple[float, ...], integral: bool
    ) -> None:
        t = lucid.arange(*args, device=device)
        assert t.shape == (0,)
        assert t.dtype == (lucid.int64 if integral else lucid.get_default_dtype())

    @pytest.mark.parametrize(
        ("make", "integral"),
        [
            (lambda: True, True),
            (lambda: np.int64(4), True),
            (lambda: lucid.tensor(4), True),
            (lambda: lucid.tensor(4, dtype=lucid.int32), True),
            (lambda: np.float32(4.0), False),
            (lambda: lucid.tensor(4.0), False),
        ],
        ids=["bool", "numpy-int", "int64-tensor", "int32-tensor", "numpy-float", "float-tensor"],
    )
    def test_other_scalars_count_as_what_they_hold(self, make: object, integral: bool) -> None:
        bound = make()  # type: ignore[operator]
        expected = lucid.int64 if integral else lucid.get_default_dtype()
        assert lucid.arange(bound).dtype == expected

    def test_integers_past_two_to_the_53_stay_exact(self, device: str) -> None:
        """The engine fills in double precision, which would repeat values here."""
        start = 2**53 + 1
        up = lucid.arange(start, start + 4, device=device)
        assert up.dtype == lucid.int64
        assert up.tolist() == [start, start + 1, start + 2, start + 3]
        down = lucid.arange(-start, -start - 6, -2, device=device)
        assert down.tolist() == [-start, -start - 2, -start - 4]

    def test_a_range_past_int64_is_refused(self) -> None:
        with pytest.raises(OverflowError):
            lucid.arange(2**63 - 1, 2**63 + 1)


class TestLinspace:
    def test_endpoint(self, device: str, float_dtype: lucid.dtype) -> None:
        t = lucid.linspace(0.0, 1.0, 5, dtype=float_dtype, device=device)
        assert_close(t, np.linspace(0.0, 1.0, 5))

    def test_single_point(self, device: str) -> None:
        t = lucid.linspace(3.0, 7.0, 1, device=device)
        assert t.shape == (1,)


class TestLogspace:
    def test_known(self, device: str, float_dtype: lucid.dtype) -> None:
        t = lucid.logspace(0.0, 2.0, 3, dtype=float_dtype, device=device)
        # Default base is 10 — [1, 10, 100].
        assert_close(t, np.logspace(0.0, 2.0, 3), atol=1e-4)


class TestZerosLike:
    def test_shape_dtype_device(self, device: str) -> None:
        src = lucid.tensor([1.0, 2.0, 3.0], device=device)
        t = lucid.zeros_like(src)
        assert t.shape == src.shape
        assert t.dtype == src.dtype
        assert_close(t, np.zeros(3))


class TestOnesLike:
    def test_shape_dtype_device(self, device: str) -> None:
        src = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        t = lucid.ones_like(src)
        assert t.shape == src.shape
        assert_close(t, np.ones((2, 2)))


class TestFullLike:
    def test_value(self, device: str) -> None:
        src = lucid.zeros(3, 4, device=device)
        t = lucid.full_like(src, 9.0)
        assert_close(t, np.full((3, 4), 9.0))


# ── random factories ─────────────────────────────────────────────────────


class TestRand:
    def test_in_unit_interval(self, device: str, float_dtype: lucid.dtype) -> None:
        t = lucid.rand(64, dtype=float_dtype, device=device)
        arr = t.numpy()
        assert (arr >= 0).all() and (arr < 1).all()

    def test_shape(self, device: str) -> None:
        assert lucid.rand(2, 3, 4, device=device).shape == (2, 3, 4)

    def test_reproducible_with_manual_seed(self, device: str) -> None:
        lucid.manual_seed(0)
        a = lucid.rand(8, device=device).numpy()
        lucid.manual_seed(0)
        b = lucid.rand(8, device=device).numpy()
        np.testing.assert_array_equal(a, b)


class TestRandn:
    def test_shape(self, device: str) -> None:
        assert lucid.randn(5, 3, device=device).shape == (5, 3)

    def test_distribution_loose(self, device: str) -> None:
        lucid.manual_seed(0)
        arr = lucid.randn(10_000, device=device).numpy()
        assert abs(arr.mean()) < 0.1
        assert abs(arr.std() - 1.0) < 0.1


class TestRandint:
    def test_in_range(self, device: str) -> None:
        t = lucid.randint(low=0, high=10, size=(64,), device=device)
        arr = t.numpy()
        assert (arr >= 0).all() and (arr < 10).all()

    def test_dtype_int64_default(self, device: str) -> None:
        t = lucid.randint(low=0, high=2, size=(4,), device=device)
        assert t.dtype == lucid.int64


class TestNormal:
    def test_shape(self, device: str) -> None:
        t = lucid.normal(mean=0.0, std=1.0, size=(3, 4), device=device)
        assert t.shape == (3, 4)

    def test_loose_moments(self, device: str) -> None:
        lucid.manual_seed(0)
        arr = lucid.normal(mean=2.0, std=0.5, size=(10_000,), device=device).numpy()
        assert abs(arr.mean() - 2.0) < 0.05
        assert abs(arr.std() - 0.5) < 0.05


class TestBernoulli:
    def test_binary(self, device: str) -> None:
        t = lucid.bernoulli(lucid.full((100,), 0.5, device=device))
        arr = t.numpy()
        assert set(arr.flatten().tolist()) <= {0.0, 1.0}


class TestRandperm:
    def test_is_permutation(self, device: str) -> None:
        n = 16
        t = lucid.randperm(n, device=device)
        arr = sorted(t.numpy().tolist())
        assert arr == list(range(n))


class TestRandLike:
    def test_inherits_metadata(self, device: str) -> None:
        src = lucid.zeros(3, 4, device=device)
        t = lucid.rand_like(src)
        assert t.shape == src.shape
        assert t.device == src.device


class TestRandnLike:
    def test_inherits_metadata(self, device: str) -> None:
        src = lucid.zeros(2, 5, device=device)
        t = lucid.randn_like(src)
        assert t.shape == src.shape


# ── tensor() / as_tensor() / from_numpy() ───────────────────────────────


class TestTensorFactory:
    def test_from_list(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        assert_close(t, np.array([1.0, 2.0, 3.0]))

    def test_from_nested_list(self, device: str) -> None:
        t = lucid.tensor([[1, 2], [3, 4]], dtype=lucid.int32, device=device)
        assert_equal_int(t, np.array([[1, 2], [3, 4]], dtype=np.int32))

    @pytest.mark.parametrize("device", devices_supporting(lucid.float64))
    def test_dtype_override(self, device: str) -> None:
        t = lucid.tensor([1, 2, 3], dtype=lucid.float64, device=device)
        assert t.dtype == lucid.float64

    def test_requires_grad_flag(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
        assert t.requires_grad

    def test_a_tensor_source_is_copied(self, device: str) -> None:
        # Documented to copy; a Tensor source used to come back as its own
        # storage, so writing to the "copy" changed the original.
        src = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = lucid.tensor(src)
        out.add_(1.0)
        assert_close(src, np.array([1.0, 2.0, 3.0]))
        assert out.device.type == src.device.type

    def test_a_tensor_source_honours_dtype_and_device(self, device: str) -> None:
        # Both overrides were silently dropped for a Tensor source.
        src = lucid.tensor([1.0, 2.0], device=device)
        assert lucid.tensor(src, dtype=lucid.float16).dtype == lucid.float16
        other = "cpu" if device == "metal" else "metal"
        moved = lucid.tensor(src, device=other)
        assert moved.device.type == other
        assert_close(moved, np.array([1.0, 2.0]))

    def test_a_tensor_source_gives_a_new_leaf(self, device: str) -> None:
        # Detached from the source's graph, as in the reference framework.
        x = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
        out = lucid.tensor(x * 2.0, requires_grad=True)
        assert out.is_leaf and out.grad_fn is None
        (out * 3.0).sum().backward()
        assert out.grad is not None
        assert_close(out.grad, np.array([3.0, 3.0]))
        assert x.grad is None
        assert not lucid.tensor(x).requires_grad


class TestAsTensor:
    def test_passthrough(self, device: str) -> None:
        src = lucid.tensor([1.0, 2.0], device=device)
        out = lucid.as_tensor(src)
        # Same object (no copy when no conversion needed).
        assert_close(out, src)

    def test_returns_the_same_object_when_nothing_converts(self, device: str) -> None:
        # A Tensor that already has the requested dtype and device comes
        # back as-is — the object, not a new wrapper around its storage —
        # whether the request is implicit (``None``) or spelled out.
        src = lucid.tensor([1.0, 2.0], device=device)
        assert lucid.as_tensor(src) is src
        assert lucid.as_tensor(src, dtype=src.dtype, device=src.device) is src
        assert lucid.as_tensor(src, dtype=lucid.float32, device=device) is src

    def test_passthrough_keeps_the_autograd_graph(self, device: str) -> None:
        src = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
        out = lucid.as_tensor(src)
        (out * 3.0).sum().backward()
        assert src.grad is not None
        assert_close(src.grad, np.array([3.0, 3.0]))

    def test_converts_when_dtype_or_device_differ(self, device: str) -> None:
        # The conversion went through ``tensor``, which dropped both
        # requests and handed back the source's own storage.
        src = lucid.tensor([1.0, 2.0], device=device)
        half = lucid.as_tensor(src, dtype=lucid.float16)
        assert half.dtype == lucid.float16
        assert_close(half, np.array([1.0, 2.0]))
        other = "cpu" if device == "metal" else "metal"
        assert lucid.as_tensor(src, device=other).device.type == other

    def test_a_conversion_stays_in_the_graph(self, device: str) -> None:
        # A cast, as in the reference framework — not the detached copy
        # that ``tensor`` makes.
        src = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
        other = "cpu" if device == "metal" else "metal"
        (lucid.as_tensor(src, device=other) * 3.0).sum().backward()
        assert src.grad is not None
        assert_close(src.grad, np.array([3.0, 3.0]))


class TestFromNumpy:
    def test_basic(self) -> None:
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        t = lucid.from_numpy(arr)
        assert_close(t, arr)

    def test_input_and_result_mutations_are_independent(self) -> None:
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        t = lucid.from_numpy(arr)
        arr[0, 0] = 99.0
        assert t[0, 0].item() == 0.0
        t[0, 1] = -7.0
        assert arr[0, 1] == 1.0
