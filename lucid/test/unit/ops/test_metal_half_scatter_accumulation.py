"""Metal half-precision scatter-add and ``embedding_bag`` accumulate in float32.

MLX's scatter-add kernels accumulate in their operand dtype.  In ``bfloat16``
(7-bit mantissa), ``256.0 + 1.0 == 256.0``, so 1000 duplicate updates of ``1.0``
into one bucket stalled at ``256.0`` in ``index_add``, ``scatter_add``,
``scatter_reduce``, and ``F.embedding`` backward, and ``F.embedding_bag`` with
``mode="mean"`` answered ``256 / N``.  In ``float16`` (10-bit mantissa, max
``65504``), sequential updates of ``1.0`` stalled at ``2048.0`` and a bag count
past ``65504`` overflowed the denominator to ``inf`` (answering ``0.0``), while
a bag mean of 1000 rows of ``100.0`` overflowed the numerator to ``inf`` before
dividing.

``GpuBackend`` widens ``float16`` and ``bfloat16`` scatter-add and
``embedding_bag`` reductions to ``float32`` through ``HalfAccumulation.h`` and
narrows once at the end, leaving ``float32`` paths untouched.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

_DEVICES = [
    "cpu",
    pytest.param(
        "metal",
        marks=pytest.mark.skipif(not metal_available(), reason="needs Metal"),
    ),
]
_HALF_DTYPES = [lucid.float16, lucid.bfloat16]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_thousand_duplicate_index_add_and_scatter_add(device: str, dtype: lucid.dtype) -> None:
    n = 1000
    idx = lucid.zeros(n, dtype=lucid.int64, device=device)
    src = lucid.ones(n, dtype=dtype, device=device)
    base = lucid.zeros(4, dtype=dtype, device=device)

    out_index_add = base.index_add(0, idx, src)
    assert out_index_add.dtype == dtype
    assert out_index_add.to("cpu").to(lucid.float32).tolist() == [1000.0, 0.0, 0.0, 0.0]

    out_scatter_add = base.scatter_add(0, idx, src)
    assert out_scatter_add.dtype == dtype
    assert out_scatter_add.to("cpu").to(lucid.float32).tolist() == [
        1000.0,
        0.0,
        0.0,
        0.0,
    ]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_thousand_duplicate_scatter_reduce_sum_and_mean(device: str, dtype: lucid.dtype) -> None:
    n = 1000
    idx = lucid.zeros(n, dtype=lucid.int64, device=device)
    src = lucid.ones(n, dtype=dtype, device=device)
    base_zero = lucid.zeros(4, dtype=dtype, device=device)
    base_one = lucid.ones(4, dtype=dtype, device=device)

    out_sum = lucid.scatter_reduce(base_zero, 0, idx, src, reduce="sum")
    assert out_sum.dtype == dtype
    assert out_sum.to("cpu").to(lucid.float32).tolist() == [1000.0, 0.0, 0.0, 0.0]

    out_sum_no_self = lucid.scatter_reduce(base_one, 0, idx, src, reduce="sum", include_self=False)
    assert out_sum_no_self.dtype == dtype
    assert out_sum_no_self.to("cpu").to(lucid.float32).tolist() == [
        1000.0,
        1.0,
        1.0,
        1.0,
    ]

    out_mean_no_self = lucid.scatter_reduce(
        base_zero, 0, idx, src, reduce="mean", include_self=False
    )
    assert out_mean_no_self.dtype == dtype
    assert out_mean_no_self.to("cpu").to(lucid.float32).tolist() == [1.0, 0.0, 0.0, 0.0]

    out_mean_with_self = lucid.scatter_reduce(
        base_one, 0, idx, src, reduce="mean", include_self=True
    )
    assert out_mean_with_self.dtype == dtype
    assert out_mean_with_self.to("cpu").to(lucid.float32).tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_thousand_duplicate_embedding_backward(device: str, dtype: lucid.dtype) -> None:
    idx = lucid.zeros(1000, dtype=lucid.int64, device=device)
    weight = lucid.ones(4, 2, dtype=dtype, device=device, requires_grad=True)
    out = F.embedding(idx, weight)
    out.sum().backward()

    assert weight.grad is not None
    assert weight.grad.dtype == dtype
    assert weight.grad.to("cpu").to(lucid.float32).tolist() == [
        [1000.0, 1000.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _HALF_DTYPES)
@pytest.mark.parametrize(
    ("mode", "expected_fwd", "expected_grad"),
    [
        ("sum", 1000.0, 1000.0),
        ("mean", 1.0, 1.0),
        ("max", 1.0, 1.0),
    ],
)
def test_thousand_duplicate_embedding_bag_forward_and_backward(
    device: str,
    dtype: lucid.dtype,
    mode: str,
    expected_fwd: float,
    expected_grad: float,
) -> None:
    idx = lucid.zeros(1000, dtype=lucid.int64, device=device)
    offsets = lucid.tensor([0], dtype=lucid.int64, device=device)
    weight = lucid.ones(4, 2, dtype=dtype, device=device, requires_grad=True)

    out = F.embedding_bag(idx, weight, offsets, mode=mode)
    assert out.dtype == dtype
    assert out.to("cpu").to(lucid.float32).tolist() == [[expected_fwd, expected_fwd]]

    out.sum().backward()
    assert weight.grad is not None
    assert weight.grad.dtype == dtype
    assert weight.grad.to("cpu").to(lucid.float32).tolist() == [
        [expected_grad, expected_grad],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


@pytest.mark.parametrize("device", _DEVICES)
def test_float16_past_2048_and_overflow_in_embedding_bag_mean(device: str) -> None:
    # In float16, 2048.0 + 1.0 == 2048.0, so 3000 ones stalled at 2048.0.
    idx_3k = lucid.zeros(3000, dtype=lucid.int64, device=device)
    src_3k = lucid.ones(3000, dtype=lucid.float16, device=device)
    base = lucid.zeros(2, dtype=lucid.float16, device=device)
    assert base.scatter_add(0, idx_3k, src_3k).to("cpu").to(lucid.float32).tolist() == [
        3000.0,
        0.0,
    ]

    # 1000 duplicates of 100.0 sum to 100,000 > 65504 (float16 max), which
    # overflowed to inf before dividing by 1000 in mode="mean".
    idx_1k = lucid.zeros(1000, dtype=lucid.int64, device=device)
    off = lucid.tensor([0], dtype=lucid.int64, device=device)
    w_100 = lucid.full((2, 2), 100.0, dtype=lucid.float16, device=device)
    mean_100 = F.embedding_bag(idx_1k, w_100, off, mode="mean")
    assert mean_100.dtype == lucid.float16
    assert mean_100.to("cpu").to(lucid.float32).tolist() == [[100.0, 100.0]]

    # 70,000 duplicates of 1.0 overflowed the float16 bag count (70,000 > 65504)
    # to inf and stalled the numerator at 2048.0, returning 2048.0 / inf == 0.0.
    idx_70k = lucid.zeros(70000, dtype=lucid.int64, device=device)
    w_one = lucid.ones(2, 2, dtype=lucid.float16, device=device)
    mean_70k = F.embedding_bag(idx_70k, w_one, off, mode="mean")
    assert mean_70k.dtype == lucid.float16
    assert mean_70k.to("cpu").to(lucid.float32).tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _HALF_DTYPES)
def test_gather_and_index_select_backward_accumulate_in_f32(
    device: str, dtype: lucid.dtype
) -> None:
    idx = lucid.zeros(1000, dtype=lucid.int64, device=device)
    x = lucid.ones(4, dtype=dtype, device=device, requires_grad=True)
    lucid.gather(x, 0, idx).sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == dtype
    assert x.grad.to("cpu").to(lucid.float32).tolist() == [1000.0, 0.0, 0.0, 0.0]

    y = lucid.ones(4, 2, dtype=dtype, device=device, requires_grad=True)
    y.index_select(0, idx).sum().backward()
    assert y.grad is not None
    assert y.grad.dtype == dtype
    assert y.grad.to("cpu").to(lucid.float32).tolist() == [
        [1000.0, 1000.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


@pytest.mark.skipif(not metal_available(), reason="needs Metal")
def test_scatter_prod_and_float32_bit_identity() -> None:
    # bfloat16 scatter_prod widens to float32 on both CPU and Metal.
    idx_prod = lucid.zeros(100, dtype=lucid.int64)
    src_prod = lucid.full((100,), 1.0078125, dtype=lucid.bfloat16)
    base_prod = lucid.ones(2, dtype=lucid.bfloat16)
    cpu_prod = lucid.scatter_reduce(base_prod, 0, idx_prod, src_prod, "prod")
    gpu_prod = lucid.scatter_reduce(
        base_prod.to("metal"),
        0,
        idx_prod.to("metal"),
        src_prod.to("metal"),
        "prod",
    )
    assert gpu_prod.dtype == lucid.bfloat16
    assert np.array_equal(
        cpu_prod.to(lucid.float32).numpy(),
        gpu_prod.to("cpu").to(lucid.float32).numpy(),
    )

    # float32 paths remain bit-identical across CPU and Metal.
    idx_f32 = lucid.zeros(1000, dtype=lucid.int64)
    src_f32 = lucid.ones(1000, dtype=lucid.float32)
    base_f32 = lucid.zeros(4, dtype=lucid.float32)
    cpu_sa = base_f32.scatter_add(0, idx_f32, src_f32)
    gpu_sa = base_f32.to("metal").scatter_add(0, idx_f32.to("metal"), src_f32.to("metal")).to("cpu")
    assert np.array_equal(cpu_sa.numpy(), gpu_sa.numpy())

    w_cpu = lucid.ones(4, 2, dtype=lucid.float32, requires_grad=True)
    w_gpu = lucid.ones(4, 2, dtype=lucid.float32, device="metal", requires_grad=True)
    off_cpu = lucid.tensor([0], dtype=lucid.int64)
    off_gpu = off_cpu.to("metal")
    eb_cpu = F.embedding_bag(idx_f32, w_cpu, off_cpu, mode="mean")
    eb_gpu = F.embedding_bag(idx_f32.to("metal"), w_gpu, off_gpu, mode="mean")
    assert np.array_equal(eb_cpu.numpy(), eb_gpu.to("cpu").numpy())
    eb_cpu.sum().backward()
    eb_gpu.sum().backward()
    assert w_cpu.grad is not None and w_gpu.grad is not None
    assert np.array_equal(w_cpu.grad.numpy(), w_gpu.grad.to("cpu").numpy())


@pytest.mark.parametrize("device", _DEVICES)
def test_bfloat16_scatter_reduce_amax_ties_and_grid_sample_backward(
    device: str,
) -> None:
    # 1000 tied entries for amax split a gradient of 1000.0 into 1.0 each;
    # previously the tie-count scatter_add stalled at 256.0 in bfloat16.
    idx = lucid.zeros(1000, dtype=lucid.int64, device=device)
    base = lucid.zeros(2, dtype=lucid.bfloat16, device=device)
    src = lucid.ones(1000, dtype=lucid.bfloat16, device=device, requires_grad=True)
    out = lucid.scatter_reduce(base, 0, idx, src, "amax", include_self=False)
    (out * 1000.0).sum().backward()
    assert src.grad is not None
    assert src.grad.dtype == lucid.bfloat16
    assert src.grad[0].to("cpu").to(lucid.float32).item() == pytest.approx(1.0, rel=1e-2)

    # 1000 output grid points sampling the top-left corner (-1, -1) accumulate
    # 1000.0 into input[0, 0, 0, 0] in bfloat16.
    x = lucid.ones(1, 1, 2, 2, dtype=lucid.bfloat16, device=device, requires_grad=True)
    grid = lucid.full((1, 25, 40, 2), -1.0, dtype=lucid.bfloat16, device=device)
    F.grid_sample(x, grid, mode="bilinear", align_corners=True).sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == lucid.bfloat16
    assert x.grad[0, 0, 0, 0].to("cpu").to(lucid.float32).item() == 1000.0
