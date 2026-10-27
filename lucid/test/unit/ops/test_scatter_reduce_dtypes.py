"""CPU ``scatter_reduce`` amax / amin / prod run in every real dtype.

LCD-284 / LCD-162.  The CPU kernel dispatched on float32 and float64
only, so integer, bool and 16-bit float inputs raised
``NotImplementedError`` where Metal and the reference reduce them.  Each
dtype now runs at its own width (half through float32); complex has no
order, so amax / amin refuse it as the reference does, and prod multiplies
complex values.
"""

import pytest

import lucid

_REAL = [
    lucid.bool,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
]

#: ``base = [1, 1]``, ``src = [1, 0, 3, 4]`` sent to ``[0, 1, 0, 1]``.
_EXPECT = {"amax": [3, 4], "amin": [1, 0], "prod": [3, 0]}
_EXPECT_BOOL = {"amax": [True, True], "amin": [True, False], "prod": [True, False]}


def _operands(dtype: lucid.dtype) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    base = lucid.ones(2).to(dtype)
    index = lucid.tensor([0, 1, 0, 1])
    src = lucid.tensor([1, 0, 3, 4]).to(dtype)
    return base, index, src


@pytest.mark.parametrize("reduce", ["amax", "amin", "prod"])
@pytest.mark.parametrize("dtype", _REAL, ids=str)
def test_cpu_scatter_reduce_real_dtypes(dtype: lucid.dtype, reduce: str) -> None:
    base, index, src = _operands(dtype)
    out = lucid.scatter_reduce(base, 0, index, src, reduce)
    assert out.dtype == dtype
    want = _EXPECT_BOOL[reduce] if dtype == lucid.bool else _EXPECT[reduce]
    assert out.tolist() == want


@pytest.mark.parametrize("reduce", ["amax", "amin", "prod"])
@pytest.mark.parametrize("dtype", [lucid.int32, lucid.int64, lucid.bfloat16], ids=str)
def test_cpu_scatter_reduce_2d_inner_axis(dtype: lucid.dtype, reduce: str) -> None:
    """Rows reduce independently along ``dim=1``, negative index included."""
    base = lucid.tensor([[2, 2, 2], [5, 5, 5]]).to(dtype)
    index = lucid.tensor([[0, -1, 0, 2], [1, 1, 0, -3]])
    src = lucid.tensor([[3, 7, 1, 4], [6, 2, 9, 8]]).to(dtype)
    out = lucid.scatter_reduce(base, 1, index, src, reduce)
    rows = []
    for b_row, i_row, s_row in zip(base.tolist(), index.tolist(), src.tolist(), strict=True):
        row = list(b_row)
        for i, s in zip(i_row, s_row, strict=True):
            j = i % 3
            if reduce == "amax":
                row[j] = max(row[j], s)
            elif reduce == "amin":
                row[j] = min(row[j], s)
            else:
                row[j] = row[j] * s
        rows.append(row)
    assert out.tolist() == rows


@pytest.mark.parametrize("dtype", [lucid.complex64, lucid.complex128], ids=str)
def test_cpu_scatter_prod_complex(dtype: lucid.dtype) -> None:
    base = lucid.tensor([2 - 1j, 1 + 0j], dtype=dtype)
    index = lucid.tensor([0, 1, 0, 1])
    src = lucid.tensor([1 + 1j, 0j, 3 + 0j, 4 - 2j], dtype=dtype)
    out = lucid.scatter_reduce(base, 0, index, src, "prod")
    assert out.dtype == dtype
    assert out.tolist() == pytest.approx([(2 - 1j) * (1 + 1j) * 3, 0j])


@pytest.mark.parametrize("reduce", ["amax", "amin"])
@pytest.mark.parametrize("dtype", [lucid.complex64, lucid.complex128], ids=str)
def test_cpu_scatter_amax_amin_refuse_complex(dtype: lucid.dtype, reduce: str) -> None:
    base, index, src = _operands(dtype)
    with pytest.raises(NotImplementedError, match="no order on complex"):
        lucid.scatter_reduce(base, 0, index, src, reduce)


@pytest.mark.parametrize("dtype", [lucid.complex64, lucid.complex128], ids=str)
def test_cpu_scatter_sum_complex(dtype: lucid.dtype) -> None:
    base, index, src = _operands(dtype)
    out = lucid.scatter_reduce(base, 0, index, src, "sum")
    assert out.tolist() == [5 + 0j, 5 + 0j]


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=str)
def test_cpu_scatter_amax_half_gradient(dtype: lucid.dtype) -> None:
    """The half path widens and narrows; the winner mask still finds the max."""
    base = lucid.zeros(2).to(dtype).requires_grad_()
    src = lucid.tensor([1.0, 0.5, 3.0, 4.0]).to(dtype).requires_grad_()
    index = lucid.tensor([0, 1, 0, 1])
    lucid.scatter_reduce(base, 0, index, src, "amax").sum().backward()
    assert src.grad is not None and base.grad is not None
    assert src.grad.float().tolist() == [0.0, 0.0, 1.0, 1.0]
    assert base.grad.float().tolist() == [0.0, 0.0]


@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("reduce", ["amax", "amin"])
@pytest.mark.parametrize(
    "dtype", [lucid.float16, lucid.bfloat16, lucid.float32, lucid.float64], ids=str
)
def test_scatter_reduce_amax_amin_nan_propagates(
    device: str, dtype: lucid.dtype, reduce: str, include_self: bool
) -> None:
    if device == "metal" and dtype == lucid.float64:
        pytest.skip("float64 is CPU-only")
    nan = float("nan")
    base = lucid.tensor([1.0, 1.0], dtype=dtype, device=device)
    index = lucid.tensor([0, 0, 1, 1], device=device)
    src = lucid.tensor([2.0, nan, nan, 2.0], dtype=dtype, device=device)
    out = lucid.scatter_reduce(base, 0, index, src, reduce, include_self=include_self)
    assert lucid.isnan(out).tolist() == [True, True]

    # When include_self=True, NaN already in base also stays NaN when updated.
    base_nan = lucid.tensor([nan, 1.0], dtype=dtype, device=device)
    src_finite = lucid.tensor([2.0, 3.0], dtype=dtype, device=device)
    idx_2 = lucid.tensor([0, 1], device=device)
    out_base = lucid.scatter_reduce(
        base_nan, 0, idx_2, src_finite, reduce, include_self=include_self
    )
    assert lucid.isnan(out_base).tolist() == [include_self, False]
