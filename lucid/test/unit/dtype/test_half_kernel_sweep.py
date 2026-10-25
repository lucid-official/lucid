"""Index-moving ops in float16 / bfloat16 match float32 rounded to half.

CHA-225.  The CPU has no 16-bit float kernels, so every one of these ops
widens half to float32, computes, and rounds once on the way out.
``embedding_backward`` did not.  Its dispatch named float32 and sent
every other dtype to a bare ``else`` that summed in ``double`` lanes, so
a half gradient came back as ``[[192, 192, 192, 1.02], [1, 1, 1, 1.02]]``
where ``[[3, 3, 3, 3], [0, 0, 0, 0]]`` was due, and the last row wrote
past its allocation.  Separately, ``sort`` and ``topk`` tested
"differentiable" as ``F32 || F64`` and dropped the grad_fn of every half
input on both devices.

Each case here runs one op forward and backward in half and compares the
result with the same op run in float32 on the half-rounded inputs, then
rounded to half.  On the CPU that comparison is exact: widening, the
float32 kernel and one rounding is the contract.  Metal computes in half
itself (accumulation order and precision are CHA-238), so there it is
held to a few half ulps.  A row nobody indexed must hold exactly zero on
both devices, which is what the overflow broke.

``tools/check_dtype_dispatch.py`` guards the source pattern; this sweep
guards the behaviour.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn.functional as F

_HALVES = [lucid.float16, lucid.bfloat16]
#: Machine epsilon of each half format (spacing of values in [1, 2)).
_EPS = {lucid.float16: 2.0**-10, lucid.bfloat16: 2.0**-7}


def _flat(values: object) -> list[float]:
    if isinstance(values, list):
        return [v for item in values for v in _flat(item)]
    return [float(values)]  # type: ignore[arg-type]


def _assert_half_matches(
    got: lucid.Tensor, ref32: lucid.Tensor, dtype: lucid.dtype, device: str
) -> None:
    """``got`` (half, on ``device``) equals ``ref32`` (float32) rounded to half."""
    assert got.dtype == dtype
    assert tuple(got.shape) == tuple(ref32.shape)
    g = _flat(got.to("cpu").float().tolist())
    r = _flat(ref32.to("cpu").to(dtype).float().tolist())
    if device == "cpu":
        assert g == r
        return
    scale = max([abs(v) for v in r] + [1.0])
    tol = 4 * _EPS[dtype] * scale
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(g, r)) if abs(a - b) > tol]
    assert not bad, f"{len(bad)} element(s) off by more than {tol}: {bad[:4]}"


def _distinct(shape: tuple[int, ...], dtype: lucid.dtype) -> lucid.Tensor:
    """Distinct values, exact in both half formats, in a shuffled order.

    Distinct so a sort has one answer: a tie would let two equally right
    permutations send the gradient to different places.
    """
    n = 1
    for d in shape:
        n *= d
    order = lucid.randperm(n).float()
    return ((order - n / 2) * 0.125).reshape(*shape).to(dtype)


def _sweep(
    op: Callable[[list[lucid.Tensor]], lucid.Tensor],
    inputs: list[lucid.Tensor],
    dtype: lucid.dtype,
    device: str,
) -> None:
    """Run ``op`` forward and backward in half and in float32; compare.

    ``inputs`` are half tensors on the CPU; the float ones get gradients.
    """
    halves = [t.to(device).detach() for t in inputs]
    wides = [t.float().to(device).detach() for t in inputs]
    for h, w in zip(halves, wides):
        if h.is_floating_point():
            h.requires_grad_(True)
            w.requires_grad_(True)
    out_h = op(halves)
    out_w = op(wides)
    _assert_half_matches(out_h, out_w.detach(), dtype, device)
    assert out_h.grad_fn is not None, "the half result lost its gradient"

    cot = lucid.randn(*out_h.shape).to(dtype)
    out_h.backward(cot.to(device))
    out_w.backward(cot.float().to(device))
    for h, w in zip(halves, wides):
        if h.is_floating_point():
            assert h.grad is not None
            _assert_half_matches(h.grad, w.grad, dtype, device)


def _ids(*values: int) -> lucid.Tensor:
    return lucid.tensor(list(values), dtype=lucid.int64)


# ── embedding ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", _HALVES)
def test_embedding_backward_repro_is_exact(dtype: lucid.dtype, device: str) -> None:
    w = lucid.ones(2, 4, dtype=dtype, device=device, requires_grad=True)
    F.embedding(lucid.zeros(3, dtype=lucid.int64, device=device), w).sum().backward()
    assert w.grad.dtype == dtype
    assert w.grad.to("cpu").float().tolist() == [[3.0] * 4, [0.0] * 4]


@pytest.mark.parametrize("dtype", _HALVES)
@pytest.mark.parametrize("padding_idx", [None, 1])
def test_embedding(dtype: lucid.dtype, device: str, padding_idx: int | None) -> None:
    # Row 3 is never looked up; row 4, the last, takes the most hits, so an
    # over-wide write from it runs off the end of the gradient buffer.
    idx = lucid.tensor([[4, 0, 1], [4, 2, 4]], dtype=lucid.int64).to(device)
    weight = lucid.randn(5, 6).to(dtype)

    def op(t: list[lucid.Tensor]) -> lucid.Tensor:
        return F.embedding(idx, t[0], padding_idx=padding_idx)

    _sweep(op, [weight], dtype, device)

    w = weight.to(device).detach().requires_grad_(True)
    F.embedding(idx, w, padding_idx=padding_idx).sum().backward()
    rows = w.grad.to("cpu").float().tolist()
    assert rows[3] == [0.0] * 6
    if padding_idx is not None:
        assert rows[padding_idx] == [0.0] * 6


@pytest.mark.parametrize("dtype", _HALVES)
@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_embedding_bag(dtype: lucid.dtype, device: str, mode: str) -> None:
    idx = _ids(0, 2, 2, 4, 1, 4, 0).to(device)
    offsets = _ids(0, 3, 5).to(device)
    weight = _distinct((5, 4), dtype)

    def op(t: list[lucid.Tensor]) -> lucid.Tensor:
        return F.embedding_bag(idx, t[0], offsets, mode=mode)

    _sweep(op, [weight], dtype, device)


# ── sort / topk ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", _HALVES)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("descending", [False, True])
def test_sort(dtype: lucid.dtype, device: str, dim: int, descending: bool) -> None:
    def op(t: list[lucid.Tensor]) -> lucid.Tensor:
        return lucid.sort(t[0], dim, descending).values

    _sweep(op, [_distinct((3, 8), dtype)], dtype, device)


@pytest.mark.parametrize("dtype", _HALVES)
@pytest.mark.parametrize("largest", [True, False])
def test_topk(dtype: lucid.dtype, device: str, largest: bool) -> None:
    def op(t: list[lucid.Tensor]) -> lucid.Tensor:
        return lucid.topk(t[0], 3, -1, largest)[0]

    _sweep(op, [_distinct((3, 8), dtype)], dtype, device)


# ── gather / index_select / scatter_add ──────────────────────────────────


@pytest.mark.parametrize("dtype", _HALVES)
def test_gather(dtype: lucid.dtype, device: str) -> None:
    # Repeated indices so the backward has to sum, not just place.
    index = lucid.tensor([[0, 3, 3], [1, 1, 2], [3, 0, 0]], dtype=lucid.int64)
    index = index.to(device)

    def op(t: list[lucid.Tensor]) -> lucid.Tensor:
        return lucid.gather(t[0], 1, index)

    _sweep(op, [lucid.randn(3, 4).to(dtype)], dtype, device)


@pytest.mark.parametrize("dtype", _HALVES)
@pytest.mark.parametrize("dim", [0, 1])
def test_index_select(dtype: lucid.dtype, device: str, dim: int) -> None:
    index = _ids(2, 0, 2, 2).to(device)

    def op(t: list[lucid.Tensor]) -> lucid.Tensor:
        return lucid.index_select(t[0], dim, index)

    _sweep(op, [lucid.randn(3, 4).to(dtype)], dtype, device)


@pytest.mark.parametrize("dtype", _HALVES)
def test_scatter_add(dtype: lucid.dtype, device: str) -> None:
    index = lucid.tensor([[0, 2, 2, 1], [2, 2, 0, 1]], dtype=lucid.int64)
    index = index.to(device)

    def op(t: list[lucid.Tensor]) -> lucid.Tensor:
        return lucid.scatter_add(t[0], 0, index, t[1])

    base, src = lucid.randn(3, 4).to(dtype), lucid.randn(2, 4).to(dtype)
    _sweep(op, [base, src], dtype, device)


# ── complex sort / topk ──────────────────────────────────────────────────


def _complex_input(device: str) -> lucid.Tensor:
    re = lucid.tensor([3.0, 1.0, 2.0, 0.5])
    im = lucid.tensor([1.0, -1.0, 0.5, 2.0])
    return lucid.complex(re, im).to(device).detach().requires_grad_(True)


def test_complex_sort_and_topk_keep_a_grad_fn(device_gpu_only: str) -> None:
    # The CPU refuses complex sort in the forward; Metal answers it, and
    # what it answers carries a gradient like any other float result.
    x = _complex_input(device_gpu_only)
    assert lucid.sort(x).values.grad_fn is not None
    assert lucid.topk(x, 2)[0].grad_fn is not None


@pytest.mark.parametrize("op", ["sort", "topk"])
def test_complex_sort_and_topk_backward(device_gpu_only: str, op: str) -> None:
    x = _complex_input(device_gpu_only)
    # Where each output element came from: MLX orders complex values by
    # their real part here (sort ascending, topk largest first).
    if op == "sort":
        out, order = lucid.sort(x).values, [3, 1, 2, 0]
    else:
        out, order = lucid.topk(x, 2)[0], [0, 2]
    re = [1.0 + k for k in range(len(order))]
    im = [5.0 + k for k in range(len(order))]
    out.backward(lucid.complex(lucid.tensor(re), lucid.tensor(im)).to(device_gpu_only))
    want = [0j] * 4
    for k, src in enumerate(order):
        want[src] = complex(re[k], im[k])
    assert [complex(z) for z in x.grad.to("cpu").tolist()] == want
