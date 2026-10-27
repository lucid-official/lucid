"""Engine op-math for a half tensor and a wider 0-d operand.

``mul``, ``div`` and ``floordiv`` take a float16 / bfloat16 tensor ``T``
next to a 0-d float32 operand ``S`` (float64 too on the CPU stream), in
either order. ``S`` is read at its own width, the op runs in float32 and the
result is rounded once into ``T``'s dtype. Every other mixed pair is still
refused, and integer floor division keeps its operands' dtype.

The oracle is the float32 op of the widened operands, rounded to the half
dtype by a separate cast. Equality is exact: there is exactly one rounding
on each side.

Floor division is the exact float32 floor (LCD-159): f16 ``1.0 // 0.1``
is 9, as on the reference's CPU backend, although ``1.0 / 0.1`` rounds to
exactly 10 in float32.

Order is where the reference framework disagrees with itself, and with
this contract, so it is not pinned against it: its CPU backend applies
op-math only when the scalar is the right operand. ``S * T`` and ``S / T``
round ``S`` to the half dtype first, so CPU ``1e5 * f16(0.5)`` is inf. Its
Metal backend, like Lucid, is symmetric.
"""

from typing import Any

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap

_HALF = [lucid.float16, lucid.bfloat16]
_OPS = ["mul", "div", "floordiv"]

# Normals, tiny and large magnitudes, both zeros and an exact reciprocal of a
# scalar below.
_VALUES = [
    1.0,
    -1.0,
    3.09375,
    0.3,
    -2.5,
    7.0,
    1e-3,
    -6e-5,
    300.0,
    -1024.0,
    0.0,
    -0.0,
    0.0999755859375,
    12.5,
    -0.7,
    5.5,
]
_SCALARS = [0.1, -3.7, 1e5, 1e-5, 2.0]


def _call(op: str, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
    return _wrap(getattr(_C_engine, op)(a._impl, b._impl))


def _oracle(op: str, t: lucid.Tensor, s: lucid.Tensor, scalar_first: bool) -> lucid.Tensor:
    wide_t = t.detach().to(lucid.float32)
    wide_s = s.detach().to(lucid.float32)
    x, y = (wide_s, wide_t) if scalar_first else (wide_t, wide_s)
    if op == "mul":
        r = x * y
    elif op == "div":
        r = x / y
    else:
        r = x // y
    return r.to(t.dtype)


def _scalar(value: float, device: str, dtype: lucid.dtype = lucid.float32) -> lucid.Tensor:
    return lucid.tensor(value, dtype=dtype, device=device)


def _half(values: Any, dtype: lucid.dtype, device: str, grad: bool = False) -> lucid.Tensor:
    # ``requires_grad_`` after construction: a bfloat16 ``tensor(...,
    # requires_grad=True)`` comes back as a non-leaf whose ``.grad`` stays
    # None, which is a factory defect and not this op's.
    return lucid.tensor(values, dtype=dtype, device=device).requires_grad_(grad)


class TestReferenceValues:
    """The four values that rounding the scalar first got wrong."""

    def test_bfloat16_times_a_tenth(self, device: str) -> None:
        t = _half([3.09375], lucid.bfloat16, device)
        s = _scalar(0.1, device)
        assert _call("mul", t, s).tolist() == [0.30859375]
        assert _call("mul", s, t).tolist() == [0.30859375]

    def test_ones_over_1e5_does_not_underflow(self, device: str) -> None:
        out = _call(
            "div",
            lucid.ones(3, dtype=lucid.float16, device=device),
            _scalar(1e5, device),
        )
        assert out.tolist() == [1.0013580322265625e-05] * 3

    def test_1e5_does_not_overflow(self, device: str) -> None:
        t = lucid.full((2,), 1e-3, dtype=lucid.float16, device=device)
        assert _call("mul", t, _scalar(1e5, device)).tolist() == [100.0625, 100.0625]

    def test_floor_division_is_the_exact_float32_floor(self, device: str) -> None:
        # float32(1.0) / float32(0.1) rounds to exactly 10.0, but 0.1 in
        # float32 is a little over a tenth, so the floor is 9, as the
        # reference says.
        t = _half([1.0, -1.0, 7.0], lucid.float16, device)
        assert _call("floordiv", t, _scalar(0.1, device)).tolist() == [
            9.0,
            -10.0,
            69.0,
        ]
        # f16(1e-3) // 1e-5 is floor(100.04) = 100; with 1e-5 rounded to
        # float16 first it would be floor(99.9) = 99.
        t = _half([1e-3], lucid.float16, device)
        assert _call("floordiv", t, _scalar(1e-5, device)).tolist() == [100.0]
        # 1e5 // f16(2) is 50000, rounded once to 49984; 1e5 rounded to
        # float16 first is inf.
        t = _half([2.0], lucid.float16, device)
        assert _call("floordiv", _scalar(1e5, device), t).tolist() == [49984.0]


class TestMatchesFloat32Oracle:
    @pytest.mark.parametrize("dtype", _HALF, ids=["f16", "bf16"])
    @pytest.mark.parametrize("op", _OPS)
    @pytest.mark.parametrize("scalar_first", [False, True], ids=["T_op_S", "S_op_T"])
    @pytest.mark.parametrize("s", _SCALARS)
    def test_values(
        self, device: str, dtype: lucid.dtype, op: str, scalar_first: bool, s: float
    ) -> None:
        t = _half(_VALUES, dtype, device)
        sc = _scalar(s, device)
        out = _call(op, sc, t) if scalar_first else _call(op, t, sc)
        assert out.dtype == dtype
        assert out.shape == t.shape
        assert out.tolist() == _oracle(op, t, sc, scalar_first).tolist()

    @pytest.mark.parametrize("dtype", _HALF, ids=["f16", "bf16"])
    @pytest.mark.parametrize("op", _OPS)
    def test_strided_view(self, device: str, dtype: lucid.dtype, op: str) -> None:
        base = _half([_VALUES[:4], _VALUES[4:8], _VALUES[8:12]], dtype, device)
        t = base.transpose(0, 1)
        sc = _scalar(0.1, device)
        out = _call(op, t, sc)
        assert out.shape == (4, 3)
        assert out.tolist() == _oracle(op, t.contiguous(), sc, False).tolist()

    @pytest.mark.parametrize("op", _OPS)
    def test_zero_dim_tensor(self, device: str, op: str) -> None:
        t = _half(3.0, lucid.float16, device)
        sc = _scalar(0.1, device)
        out = _call(op, t, sc)
        assert out.shape == ()
        assert out.item() == _oracle(op, t, sc, False).item()

    @pytest.mark.parametrize("op", _OPS)
    def test_empty_tensor(self, device: str, op: str) -> None:
        t = lucid.zeros(0, 3, dtype=lucid.bfloat16, device=device)
        out = _call(op, _scalar(0.1, device), t)
        assert out.shape == (0, 3)
        assert out.dtype == lucid.bfloat16

    @pytest.mark.parametrize("op", _OPS)
    @pytest.mark.parametrize("scalar_first", [False, True], ids=["T_op_S", "S_op_T"])
    def test_float64_scalar_on_cpu(self, op: str, scalar_first: bool) -> None:
        t = _half(_VALUES, lucid.float16, "cpu")
        sc = _scalar(0.1, "cpu", lucid.float64)
        out = _call(op, sc, t) if scalar_first else _call(op, t, sc)
        assert out.dtype == lucid.float16
        assert out.tolist() == _oracle(op, t, sc, scalar_first).tolist()


class TestGradients:
    """One node per op; the gradient is computed widened and rounded once."""

    @pytest.mark.parametrize("dtype", _HALF, ids=["f16", "bf16"])
    @pytest.mark.parametrize("op", _OPS)
    @pytest.mark.parametrize("scalar_first", [False, True], ids=["T_op_S", "S_op_T"])
    def test_first_order(
        self, device: str, dtype: lucid.dtype, op: str, scalar_first: bool
    ) -> None:
        values = [1.5, -2.0, 0.3, 4.0, -0.7, 9.0]
        t = _half(values, dtype, device, grad=True)
        sc = _scalar(0.1, device)
        g = _half([0.5, -1.25, 3.0, 0.1, 2.0, -0.3], dtype, device)
        out = _call(op, sc, t) if scalar_first else _call(op, t, sc)
        assert out.requires_grad
        out.backward(g)
        assert t.grad is not None
        assert t.grad.dtype == dtype

        wide_g = g.to(lucid.float32)
        wide_t = t.detach().to(lucid.float32)
        if op == "mul":
            expected = wide_g * sc
        elif op == "div" and not scalar_first:
            expected = wide_g / sc
        elif op == "div":
            expected = -(sc * wide_g) / (wide_t * wide_t)
        else:
            expected = lucid.zeros_like(wide_g)
        assert t.grad.tolist() == expected.to(dtype).tolist()

    def test_create_graph_reaches_the_second_derivative(self, device: str) -> None:
        # d/dT (S / T) = -S / T^2 and d2/dT2 = 2 S / T^3.
        t = _half([0.5, 2.0, -4.0], lucid.float16, device, grad=True)
        sc = _scalar(0.25, device)
        out = _call("div", sc, t)
        (first,) = lucid.autograd.grad(out, t, [lucid.ones_like(out)], create_graph=True)
        assert first.requires_grad
        assert first.tolist() == [-1.0, -0.0625, -0.015625]
        (second,) = lucid.autograd.grad(first, t, [lucid.ones_like(first)])
        assert second.tolist() == [4.0, 0.0625, -0.0078125]

    @pytest.mark.parametrize("op", ["mul", "div"])
    def test_create_graph_first_order_matches(self, device: str, op: str) -> None:
        t = _half([1.5, -2.0, 0.3], lucid.bfloat16, device, grad=True)
        sc = _scalar(0.1, device)
        out = _call(op, t, sc)
        (first,) = lucid.autograd.grad(out, t, [lucid.ones_like(out)], create_graph=True)
        wide = lucid.ones(3, device=device)
        expected = (wide * sc if op == "mul" else wide / sc).to(lucid.bfloat16)
        assert first.tolist() == expected.tolist()

    def test_scalar_mutated_after_forward_is_caught(self, device: str) -> None:
        t = _half([1.0, 2.0], lucid.float16, device, grad=True)
        sc = _scalar(0.1, device)
        out = _call("mul", t, sc)
        # The CPU stream refuses the write into a saved storage outright; the
        # Metal stream lets it happen and the version check refuses backward.
        with pytest.raises(RuntimeError):
            sc.add_(1.0)
            out.backward(lucid.ones_like(out))

    @pytest.mark.parametrize("dtype", _HALF, ids=["f16", "bf16"])
    @pytest.mark.parametrize("op", _OPS)
    @pytest.mark.parametrize("scalar_first", [False, True], ids=["T_op_S", "S_op_T"])
    def test_scalar_requiring_grad_receives_widened_gradient(
        self, device: str, dtype: lucid.dtype, op: str, scalar_first: bool
    ) -> None:
        t = _half([1.5, -2.25, 3.09375], dtype, device, grad=True)
        sc = lucid.tensor(0.1, dtype=lucid.float32, device=device, requires_grad=True)
        out = _call(op, sc, t) if scalar_first else _call(op, t, sc)
        seed = _half([0.5, -1.25, 2.0], dtype, device)
        out.backward(seed)

        wide_g = seed.to(lucid.float32)
        wide_t = t.detach().to(lucid.float32)
        sc_det = sc.detach()
        if op == "mul":
            expected_s = (wide_g * wide_t).sum()
        elif op == "div" and scalar_first:
            expected_s = (wide_g / wide_t).sum()
        elif op == "div":
            expected_s = (-(wide_g * wide_t) / (sc_det * sc_det)).sum()
        else:
            expected_s = lucid.tensor(0.0, dtype=lucid.float32, device=device)

        assert sc.grad is not None
        assert sc.grad.dtype == lucid.float32
        assert sc.grad.shape == ()
        assert sc.grad.item() == pytest.approx(expected_s.item(), rel=1e-6, abs=1e-6)
        assert t.grad is not None
        assert t.grad.dtype == dtype

    @pytest.mark.parametrize("op", ["mul", "div"])
    @pytest.mark.parametrize("scalar_first", [False, True])
    def test_create_graph_differentiates_with_respect_to_scalar(
        self, device: str, op: str, scalar_first: bool
    ) -> None:
        t = _half([2.0, 4.0], lucid.float16, device, grad=True)
        sc = lucid.tensor(0.5, dtype=lucid.float32, device=device, requires_grad=True)
        out = _call(op, sc, t) if scalar_first else _call(op, t, sc)
        dt, dsc = lucid.autograd.grad(out, [t, sc], [lucid.ones_like(out)], create_graph=True)
        assert dsc.dtype == lucid.float32
        assert dsc.shape == ()
        expected_dsc = 6.0 if op == "mul" else (0.75 if scalar_first else -24.0)
        assert dsc.item() == pytest.approx(expected_dsc, rel=1e-6)
        (ddsc_dt,) = lucid.autograd.grad(dsc, [t])
        assert ddsc_dt.dtype == lucid.float16

    def test_tensor_mutated_after_forward_is_caught_when_scalar_requires_grad(
        self, device: str
    ) -> None:
        t = _half([1.0, 2.0], lucid.float16, device)
        sc = lucid.tensor(0.1, dtype=lucid.float32, device=device, requires_grad=True)
        out = _call("mul", t, sc)
        with pytest.raises(RuntimeError):
            t.add_(1.0)
            out.backward(lucid.ones_like(out))


def _mixed_pairs(device: str) -> list[tuple[lucid.Tensor, lucid.Tensor]]:
    f16 = lucid.ones(3, dtype=lucid.float16, device=device)
    bf16 = lucid.ones(3, dtype=lucid.bfloat16, device=device)
    pairs = [
        (f16, lucid.ones(3, device=device)),  # wider operand is not 0-d
        (f16, lucid.ones(1, device=device)),  # one element is not 0-d
        (lucid.ones(3, device=device), _scalar(2.0, device, lucid.float16)),
        (f16, _scalar(2.0, device, lucid.bfloat16)),
        (bf16, _scalar(2.0, device, lucid.float16)),
        (f16, lucid.tensor(2, dtype=lucid.int32, device=device)),
        (lucid.ones(3, dtype=lucid.int32, device=device), _scalar(2.0, device)),
    ]
    if device == "cpu":
        pairs.append((lucid.ones(3, device=device), _scalar(2.0, device, lucid.float64)))
    return pairs


class TestOtherMixedPairsStillRefused:
    @pytest.mark.parametrize("op", [*_OPS, "add", "sub"])
    def test_dtype_mismatch(self, device: str, op: str) -> None:
        for a, b in _mixed_pairs(device):
            for x, y in ((a, b), (b, a)):
                with pytest.raises(_C_engine.DtypeMismatch):
                    _call(op, x, y)

    @pytest.mark.parametrize("op", ["add", "sub"])
    def test_other_ops_do_not_take_the_pair(self, device: str, op: str) -> None:
        with pytest.raises(_C_engine.DtypeMismatch):
            _call(
                op,
                lucid.ones(3, dtype=lucid.float16, device=device),
                _scalar(0.1, device),
            )


class TestTrace:
    @pytest.mark.parametrize("op", _OPS)
    @pytest.mark.parametrize("scalar_first", [False, True], ids=["T_op_S", "S_op_T"])
    def test_one_node_with_both_operands(self, device: str, op: str, scalar_first: bool) -> None:
        t = _half([1.0, 2.0], lucid.float16, device)
        sc = _scalar(0.1, device)
        with lucid.compile._tracing() as tracer:
            _ = _call(op, sc, t) if scalar_first else _call(op, t, sc)
        ops = tracer.graph.ops
        assert [node.name for node in ops] == [op]
        assert len(ops[0].inputs) == 2


class TestAutocast:
    def test_cpu_float16_autocast_keeps_op_math(self) -> None:
        # ``mul`` has no row in the policy table (LCD-500), so a CPU float16
        # autocast leaves the float16 operand alone and the op computes as
        # it does outside the scope.
        t = _half([3.09375, 1e-3], lucid.float16, "cpu")
        sc = _scalar(0.1, "cpu")
        with lucid.amp.autocast(device_type="cpu", dtype=lucid.float16):
            out = _call("mul", t, sc)
        assert out.dtype == lucid.float16
        assert out.tolist() == _call("mul", t, sc).tolist()

    def test_metal_autocast_in_the_half_dtype_keeps_op_math(self) -> None:
        from lucid.test._fixtures.devices import metal_available

        if not metal_available():
            pytest.skip("Metal unavailable")
        t = _half([3.09375], lucid.bfloat16, "metal")
        with lucid.amp.autocast(device_type="metal", dtype=lucid.bfloat16):
            out = _call("mul", t, _scalar(0.1, "metal"))
        assert out.dtype == lucid.bfloat16
        assert out.tolist() == [0.30859375]


_INT = [lucid.int8, lucid.int16, lucid.int32, lucid.int64]


class TestIntegerFloorDivision:
    @pytest.mark.parametrize("dtype", _INT, ids=["i8", "i16", "i32", "i64"])
    def test_keeps_the_input_dtype(self, device: str, dtype: lucid.dtype) -> None:
        a_vals = [7, -7, 7, -7, 0, 9, -1, 100]
        b_vals = [2, 2, -2, -2, 3, 3, 5, -7]
        a = lucid.tensor(a_vals, dtype=dtype, device=device)
        b = lucid.tensor(b_vals, dtype=dtype, device=device)
        out = _call("floordiv", a, b)
        assert out.dtype == dtype
        assert out.tolist() == [x // y for x, y in zip(a_vals, b_vals, strict=True)]

    def test_broadcast_keeps_the_input_dtype(self, device: str) -> None:
        a = lucid.tensor([[7, -7], [9, -9]], dtype=lucid.int16, device=device)
        b = lucid.tensor([2, -4], dtype=lucid.int16, device=device)
        out = _call("floordiv", a, b)
        assert out.dtype == lucid.int16
        assert out.tolist() == [[3, 1], [4, 2]]

    def test_int8_min_over_minus_one_wraps(self, device: str) -> None:
        a = lucid.tensor([-128], dtype=lucid.int8, device=device)
        b = lucid.tensor([-1], dtype=lucid.int8, device=device)
        assert _call("floordiv", a, b).tolist() == [-128]

    @pytest.mark.parametrize("dtype", [lucid.bool_, lucid.complex64], ids=["bool", "c64"])
    def test_non_real_dtypes_are_refused(self, device: str, dtype: lucid.dtype) -> None:
        a = lucid.ones(2, dtype=dtype, device=device)
        with pytest.raises(NotImplementedError):
            _call("floordiv", a, a)


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _HALF, ids=["f16", "bf16"])
@pytest.mark.parametrize("op", ["mul", "div"])
@pytest.mark.parametrize("s", _SCALARS)
def test_matches_the_reference(
    device: str, dtype: lucid.dtype, op: str, s: float, ref: Any
) -> None:
    """``T op S`` against the reference's CPU backend, which applies op-math
    when the scalar is the right operand.  Both Lucid devices must agree with
    it."""
    ref_dtype = ref.float16 if dtype is lucid.float16 else ref.bfloat16
    rt = ref.tensor(_VALUES, dtype=ref_dtype)
    rs = ref.tensor(s, dtype=ref.float32)
    expected = (rt * rs if op == "mul" else rt / rs).float().tolist()
    out = _call(op, _half(_VALUES, dtype, device), _scalar(s, device))
    assert out.to(lucid.float32).tolist() == expected
