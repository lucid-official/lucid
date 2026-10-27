"""Python scalars as the second operand of a binary op.

Three spellings reach the same engine op — the operator, the method, and the
free function — and only the operator used to coerce a scalar.  The other two
handed the raw number to the engine, where the dtype-promotion step read
``.dtype`` off an ``int`` and raised ``AttributeError``.  Every op below is
one the reference framework accepts a scalar for, so the method and free
forms have to as well.

The promotion these tests pin is the *weak scalar* rule: a scalar contributes
its kind (bool < int < float) but never its width.  Taking the tensor's dtype
unconditionally, as the coercion first did, silently truncated ``1.5`` to an
integer beside an integer tensor and returned wrong numbers with no error.
"""

import pytest

import lucid

# Ops whose method and free-function forms accept a scalar, with a base
# tensor whose dtype the op is defined for.
_FLOAT_OPS = [
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "maximum",
    "minimum",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "fmod",
    "remainder",
]
_INT_OPS = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
]
# Arithmetic ops that also have an in-place form.
_INPLACE_OPS = ["add_", "sub_", "mul_", "div_", "pow_", "maximum_", "minimum_"]

# Operator spellings, to pin that the method agrees with them.
_OPERATORS = {
    "add": lambda t, s: t + s,
    "sub": lambda t, s: t - s,
    "mul": lambda t, s: t * s,
    "div": lambda t, s: t / s,
    "pow": lambda t, s: t**s,
    "eq": lambda t, s: t == s,
    "ne": lambda t, s: t != s,
    "lt": lambda t, s: t < s,
    "le": lambda t, s: t <= s,
    "gt": lambda t, s: t > s,
    "ge": lambda t, s: t >= s,
}


def _base(device: str, integral: bool) -> lucid.Tensor:
    if integral:
        return lucid.tensor([2, 3, 6], dtype=lucid.int32, device=device)
    return lucid.tensor([2.0, 3.0, 6.0], dtype=lucid.float32, device=device)


class TestScalarOperandAccepted:
    @pytest.mark.parametrize("name", _FLOAT_OPS)
    def test_method_and_free_function_take_a_scalar(self, device: str, name: str) -> None:
        t = _base(device, integral=False)
        method = getattr(t, name)(2.0)
        free = getattr(lucid, name)(t, 2.0)
        assert method.tolist() == free.tolist()

    @pytest.mark.parametrize("name", _INT_OPS)
    def test_bitwise_ops_take_an_int_scalar(self, device: str, name: str) -> None:
        t = _base(device, integral=True)
        method = getattr(t, name)(1)
        free = getattr(lucid, name)(t, 1)
        assert method.tolist() == free.tolist()

    @pytest.mark.parametrize("name", _INPLACE_OPS)
    def test_inplace_forms_take_a_scalar(self, device: str, name: str) -> None:
        t = _base(device, integral=False)
        getattr(t, name)(2.0)
        expected = getattr(_base(device, integral=False), name.rstrip("_"))(2.0)
        assert t.tolist() == expected.tolist()

    @pytest.mark.parametrize("name", sorted(_OPERATORS))
    def test_method_agrees_with_the_operator(self, device: str, name: str) -> None:
        # The operator form always coerced; the method is what regressed, so
        # the two have to land on the same values *and* the same dtype.
        t = _base(device, integral=False)
        by_method = getattr(t, name)(2.0)
        by_operator = _OPERATORS[name](t, 2.0)
        assert by_method.tolist() == by_operator.tolist()
        assert by_method.dtype == by_operator.dtype

    def test_scalar_operand_still_carries_gradient(self, device: str) -> None:
        # The scalar becomes a constant, so the tensor keeps the whole
        # gradient: d(x**3)/dx = 3x**2.
        x = lucid.tensor([2.0, 3.0], device=device, requires_grad=True)
        x.pow(3).sum().backward()
        assert x.grad is not None
        assert x.grad.tolist() == pytest.approx([12.0, 27.0])

    def test_a_non_number_is_still_rejected(self, device: str) -> None:
        t = _base(device, integral=False)
        with pytest.raises(TypeError):
            t.add("not a number")


class TestScalarPromotion:
    """A scalar widens the kind but never the precision."""

    def test_float_scalar_promotes_an_integer_tensor(self, device: str) -> None:
        t = lucid.tensor([2, 3], dtype=lucid.int32, device=device)
        out = t + 1.5
        assert out.dtype == lucid.float32
        assert out.tolist() == pytest.approx([3.5, 4.5])

    def test_int_scalar_leaves_an_integer_tensor_alone(self, device: str) -> None:
        t = lucid.tensor([2, 3], dtype=lucid.int32, device=device)
        out = t + 2
        assert out.dtype == lucid.int32
        assert out.tolist() == [4, 5]

    def test_bool_scalar_never_widens(self, device: str) -> None:
        t = lucid.tensor([2, 3], dtype=lucid.int32, device=device)
        out = t + True
        assert out.dtype == lucid.int32
        assert out.tolist() == [3, 4]

    @pytest.mark.parametrize("scalar", [True, 2, 1.5])
    def test_a_scalar_never_widens_a_float_tensor(self, device: str, scalar: object) -> None:
        # float32 stays float32 even beside a Python float, which is a
        # double: the scalar contributes no width of its own.
        t = lucid.tensor([2.0, 3.0], dtype=lucid.float32, device=device)
        assert (t + scalar).dtype == lucid.float32

    def test_the_method_form_promotes_the_same_way(self, device: str) -> None:
        t = lucid.tensor([2, 3], dtype=lucid.int32, device=device)
        assert t.add(1.5).dtype == (t + 1.5).dtype
        assert t.add(1.5).tolist() == (t + 1.5).tolist()

    def test_promotion_follows_the_default_float_dtype(self, device: str) -> None:
        # A float scalar beside an integer tensor lands on the session
        # default, not a hardcoded float32 -- except on the GPU stream, which
        # cannot hold float64 at all, so promoting to it would build an
        # operand the device has no representation for.
        t = lucid.tensor([2, 3], dtype=lucid.int32, device=device)
        expected = lucid.float64 if device == "cpu" else lucid.float32
        original = lucid.get_default_dtype()
        try:
            lucid.set_default_dtype(lucid.float64)
            assert (t + 1.5).dtype == expected
            assert t.add(1.5).dtype == expected
        finally:
            lucid.set_default_dtype(original)

    @pytest.mark.parametrize(
        ("bounds", "expected"),
        [
            ({"min": 0.5}, [0.5, 2.0, 3.0]),
            ({"max": 2.5}, [0.0, 2.0, 2.5]),
            ({"min": 0.5, "max": 2.5}, [0.5, 2.0, 2.5]),
        ],
    )
    def test_a_float_bound_clamps_an_integer_tensor_in_float(
        self, device: str, bounds: dict[str, float], expected: list[float]
    ) -> None:
        # The engine kept the tensor's dtype: one bound raised
        # DtypeMismatch, and two truncated 0.5 and 2.5 to whole numbers.
        t = lucid.tensor([0, 2, 3], dtype=lucid.int32, device=device)
        for out in (
            lucid.clamp(t, **bounds),
            t.clamp(**bounds),
            lucid.clip(t, **bounds),
        ):
            assert out.dtype == lucid.float32
            assert out.tolist() == expected

    def test_a_clamp_bound_never_widens_the_width(self, device: str) -> None:
        half = lucid.tensor([0.0, 2.0], dtype=lucid.float16, device=device)
        assert lucid.clamp(half, min=0.5).dtype == lucid.float16
        assert lucid.clamp(half, 0.5, 1.5).dtype == lucid.float16
        ints = lucid.tensor([0, 5], dtype=lucid.int32, device=device)
        assert lucid.clamp(ints, 1, 3).dtype == lucid.int32
        assert lucid.clamp(ints, 1, 3).tolist() == [1, 3]
        assert lucid.clamp(ints, min=1).tolist() == [1, 5]
