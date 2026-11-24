"""A 0-d operand beside tensors with a dimension, for ``inner``, ``lerp``, ``addr`` and the stacks.

LCD-1068: ``inner`` and ``lerp`` handed the engine's single-dtype kernel a 0-d operand of
another dtype and refused it, where the reference ranks the 0-d operand below the others
(``a * b``'s rule) and answers.  ``addr`` ranked its 0-d accumulator below the vectors,
where the reference expands it first.  LCD-1069: ``column_stack`` indexed a 0-d input as if
it had an axis.

The whole 0-d x dimensioned dtype lattice, on both devices and against the reference, is in
``dtype/test_promotion_tiers.py``.  This file pins the headline cells without the
reference, what is still refused, and the ``column_stack`` / ``atleast_*`` / ``*stack``
family for a 0-d input.
"""

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import device_supports


def _inner_spellings() -> dict[str, Callable[[Any, Any], Any]]:
    return {
        "linalg.inner": lambda a, b: lucid.linalg.inner(a, b),
        "Tensor.inner": lambda a, b: a.inner(b),
    }


@pytest.mark.parametrize("spelling", list(_inner_spellings()))
@pytest.mark.parametrize("zero_first", [True, False])
def test_inner_of_a_zero_dim_operand_is_the_product_at_the_ranked_dtype(
    spelling: str, zero_first: bool, device: str
) -> None:
    zero = lucid.tensor(3, dtype=lucid.int16, device=device)
    vector = lucid.tensor([2], dtype=lucid.int8, device=device)
    pair = (zero, vector) if zero_first else (vector, zero)
    out = _inner_spellings()[spelling](*pair)
    assert out.dtype is lucid.int8
    assert out.tolist() == [6]


def test_inner_of_a_zero_dim_float32_beside_a_half_vector_stays_half(device: str) -> None:
    out = lucid.linalg.inner(
        lucid.tensor(0.5, device=device),
        lucid.tensor([2.0, 4.0], dtype=lucid.float16, device=device),
    )
    assert out.dtype is lucid.float16
    assert out.tolist() == [1.0, 2.0]


def test_inner_of_a_zero_dim_operand_back_propagates() -> None:
    zero = lucid.tensor(3.0, requires_grad=True)
    vector = lucid.tensor([1.0, 2.0])
    lucid.linalg.inner(zero, vector).sum().backward()
    assert zero.grad is not None
    assert zero.grad.item() == 3.0


def test_inner_of_two_dimensioned_tensors_is_unchanged() -> None:
    a, b = lucid.tensor([1.0, 2.0]), lucid.tensor([3.0, 4.0])
    assert lucid.linalg.inner(a, b).item() == 11.0
    out = lucid.linalg.inner(lucid.ones((2, 3)), lucid.ones((4, 3)))
    assert out.shape == (2, 4)


def test_inner_still_refuses_two_dtypes_when_both_have_a_dimension() -> None:
    with pytest.raises(_C_engine.DtypeMismatch):
        lucid.linalg.inner(lucid.ones(2, dtype=lucid.int16), lucid.ones(2, dtype=lucid.int8))


@pytest.mark.parametrize("spelling", list(_inner_spellings()))
def test_inner_refuses_a_python_number(spelling: str) -> None:
    with pytest.raises(TypeError):
        _inner_spellings()[spelling](lucid.ones(2), 2.0)


@pytest.mark.parametrize("weight_dtype", [lucid.float32, lucid.float64, lucid.bfloat16])
def test_lerp_with_a_zero_dim_weight_keeps_the_halves_dtype(
    weight_dtype: lucid.dtype, device: str
) -> None:
    if not device_supports(device, weight_dtype):
        pytest.skip(f"{weight_dtype} is not held on {device}")
    half = lucid.float16
    start = lucid.ones(2, dtype=half, device=device)
    end = lucid.zeros(2, dtype=half, device=device)
    out = lucid.lerp(start, end, lucid.tensor(0.5, dtype=weight_dtype, device=device))
    assert out.dtype is half
    assert out.tolist() == [0.5, 0.5]


def test_lerp_with_a_zero_dim_complex_weight_answers_complex(device: str) -> None:
    start = lucid.tensor([2.0, 1.0], device=device)
    out = lucid.lerp(start, start, lucid.tensor(0.5 + 0j, device=device))
    assert out.dtype is lucid.complex64
    assert out.tolist() == [2 + 0j, 1 + 0j]


def test_lerp_with_a_zero_dim_weight_back_propagates() -> None:
    start = lucid.ones(2, dtype=lucid.float16, requires_grad=True)
    weight = lucid.tensor(0.5, requires_grad=True)
    lucid.lerp(start, lucid.zeros(2, dtype=lucid.float16), weight).sum().backward()
    assert start.grad is not None and weight.grad is not None
    assert start.grad.tolist() == [0.5, 0.5]
    assert weight.grad.item() == -2.0


def test_lerp_in_place_with_a_zero_dim_weight_of_another_dtype() -> None:
    start = lucid.ones(2, dtype=lucid.float16)
    start.lerp_(lucid.zeros(2, dtype=lucid.float16), lucid.tensor(0.5))
    assert start.dtype is lucid.float16
    assert start.tolist() == [0.5, 0.5]


def test_lerp_still_refuses_what_the_reference_refuses() -> None:
    half = lucid.ones(2, dtype=lucid.float16)
    zeros = lucid.zeros(2, dtype=lucid.float16)
    # A weight with a dimension must share the dtype, a 0-d one need not.
    with pytest.raises(_C_engine.DtypeMismatch):
        half.lerp(zeros, lucid.full((2,), 0.5))
    with pytest.raises(_C_engine.DtypeMismatch):
        half.lerp(zeros.float(), lucid.tensor(0.5))
    # Integer operands stay refused (``Tensor.lerp``'s contract), a float weight included.
    ints = lucid.tensor([0, 10], dtype=lucid.int64)
    for weight in (0.5, lucid.tensor(1, dtype=lucid.int64), lucid.tensor(0.5)):
        with pytest.raises(NotImplementedError):
            ints.lerp(ints, weight)


@pytest.mark.parametrize(
    ("accumulator", "vectors", "expected"),
    [
        (lucid.int16, lucid.int8, lucid.int16),
        (lucid.float32, lucid.float16, lucid.float32),
        (lucid.float64, lucid.float32, lucid.float64),
    ],
)
@pytest.mark.parametrize("accumulator_shape", [(), (2, 2)], ids=["0d", "matrix"])
def test_addr_forms_its_product_at_the_dtype_all_three_promote_to(
    accumulator: lucid.dtype,
    vectors: lucid.dtype,
    expected: lucid.dtype,
    accumulator_shape: tuple[int, ...],
) -> None:
    """``63 * 63`` wraps in ``int8``: the product is formed after the cast, not before it.

    A 0-d accumulator also ranks as dimensioned, so it is not cast down to the vectors.
    """
    out = lucid.addr(
        lucid.zeros(accumulator_shape, dtype=accumulator),
        lucid.tensor([63, 63], dtype=vectors),
        lucid.tensor([63, 63], dtype=vectors),
    )
    assert out.dtype is expected
    assert out.tolist() == [[3969, 3969]] * 2


@pytest.mark.parametrize("vectors", [lucid.int8, lucid.float32])
def test_addr_with_a_complex_accumulator_and_real_vectors(vectors: lucid.dtype) -> None:
    """The product of real vectors is real, and the sum lifts it to complex."""
    out = lucid.addr(
        lucid.tensor(1 + 0j),
        lucid.tensor([63, 63], dtype=vectors),
        lucid.tensor([63, 63], dtype=vectors),
    )
    assert out.dtype is lucid.complex64
    assert out.tolist() == [[3970 + 0j, 3970 + 0j]] * 2


@pytest.mark.parity
@pytest.mark.parametrize("accumulator", ["complex64", "complex128"])
@pytest.mark.parametrize("vectors", ["float32", "float64", "int64"])
@pytest.mark.parametrize(
    "scalars",
    [
        pytest.param({"alpha": 1j}, id="alpha=1j"),
        pytest.param({"alpha": 1 + 2j, "beta": 1 - 1j}, id="alpha=1+2j,beta=1-1j"),
        pytest.param({"beta": 2j}, id="beta=2j"),
        pytest.param({"beta": 0, "alpha": 1j}, id="beta=0,alpha=1j"),
        pytest.param({}, id="default"),
    ],
)
def test_addr_with_a_complex_accumulator_keeps_a_complex_scalars_imaginary_part(
    accumulator: str, vectors: str, scalars: dict[str, complex], ref: ModuleType
) -> None:
    """Real vectors beside a complex accumulator: ``alpha`` scales the product after it is lifted.

    The product of real vectors is real, and a complex
    ``alpha`` applied to it as a real tensor dropped its imaginary part:
    ``alpha=1j`` gave ``1+0j`` where the reference gives ``[[1+3j, 1+4j], [1+6j, 1+8j]]``.
    """
    results = []
    for lib in (lucid, ref):
        results.append(
            lib.addr(
                lib.full((2, 2), 1, dtype=getattr(lib, accumulator)),
                lib.tensor([1, 2], dtype=getattr(lib, vectors)),
                lib.tensor([3, 4], dtype=getattr(lib, vectors)),
                **scalars,
            )
        )
    got, want = results
    assert str(got.dtype).rsplit(".", 1)[-1] == str(want.dtype).rsplit(".", 1)[-1]
    assert got.tolist() == want.tolist()


def test_addr_complex_alpha_on_a_complex_accumulator_headline() -> None:
    out = lucid.addr(
        lucid.full((2, 2), 1, dtype=lucid.complex64),
        lucid.tensor([1.0, 2.0]),
        lucid.tensor([3.0, 4.0]),
        alpha=1j,
    )
    assert out.tolist() == [[1 + 3j, 1 + 4j], [1 + 6j, 1 + 8j]]


def _stack_cases(dtype: lucid.dtype) -> list[tuple[str, Sequence[lucid.Tensor]]]:
    zero = lucid.tensor(1, dtype=dtype)
    return [
        ("0d", [zero]),
        ("0d,0d", [zero, zero + 1]),
        ("0d,1d", [zero, lucid.tensor([2], dtype=dtype)]),
        ("1d,0d", [lucid.tensor([2], dtype=dtype), zero]),
        ("2d,0d", [lucid.tensor([[2, 3]], dtype=dtype), zero]),
        ("0d,2d", [zero, lucid.tensor([[2], [3]], dtype=dtype)]),
    ]


def test_column_stack_lifts_a_zero_dim_input_to_a_column() -> None:
    out = lucid.column_stack([lucid.tensor(1.0), lucid.tensor([2.0])])
    assert out.shape == (1, 2)
    assert out.tolist() == [[1.0, 2.0]]
    both = lucid.column_stack([lucid.tensor(1.0), lucid.tensor(2.0)])
    assert both.shape == (1, 2)
    wider = lucid.column_stack([lucid.tensor([[1.0, 2.0]]), lucid.tensor(3.0)])
    assert wider.tolist() == [[1.0, 2.0, 3.0]]


def test_column_stack_still_refuses_rows_that_do_not_line_up() -> None:
    with pytest.raises(_C_engine.ShapeMismatch):
        lucid.column_stack([lucid.tensor(1.0), lucid.tensor([2.0, 3.0])])


_STACKS: dict[str, Callable[[Any, Sequence[Any]], Any]] = {
    "column_stack": lambda lib, ts: lib.column_stack(ts),
    "hstack": lambda lib, ts: lib.hstack(ts),
    "vstack": lambda lib, ts: lib.vstack(ts),
    "row_stack": lambda lib, ts: lib.row_stack(ts),
    "dstack": lambda lib, ts: lib.dstack(ts),
    "atleast_1d": lambda lib, ts: lib.atleast_1d(*ts),
    "atleast_2d": lambda lib, ts: lib.atleast_2d(*ts),
    "atleast_3d": lambda lib, ts: lib.atleast_3d(*ts),
}


def _as_list(out: Any) -> list[Any]:
    """A stack's single tensor, or ``atleast_*``'s one tensor or sequence, as a list."""
    return list(out) if isinstance(out, (tuple, list)) else [out]


@pytest.mark.parity
@pytest.mark.parametrize("op", list(_STACKS))
@pytest.mark.parametrize("dtype_name", ["float32", "int16"])
def test_stacks_of_a_zero_dim_input_match_the_reference(
    op: str, dtype_name: str, ref: ModuleType
) -> None:
    """Where the reference answers, so does Lucid, in the same shape, dtype and values."""
    lucid_dtype = getattr(lucid, dtype_name)
    ref_dtype = getattr(ref, dtype_name)
    problems: list[str] = []
    compared = 0
    for label, lucid_tensors in _stack_cases(lucid_dtype):
        ref_tensors = [ref.tensor(t.tolist(), dtype=ref_dtype) for t in lucid_tensors]
        try:
            want = _as_list(_STACKS[op](ref, ref_tensors))
        except RuntimeError:
            with pytest.raises(RuntimeError):
                _STACKS[op](lucid, lucid_tensors)
            continue
        got = _as_list(_STACKS[op](lucid, lucid_tensors))
        compared += 1
        for g, w in zip(got, want, strict=True):
            if tuple(g.shape) != tuple(w.shape) or g.tolist() != w.tolist():
                problems.append(
                    f"{label}: {g.tolist()} {g.shape}, reference {w.tolist()} {w.shape}"
                )
    assert compared > 0
    assert not problems, "\n".join(problems)
