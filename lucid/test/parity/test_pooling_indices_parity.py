"""Parity: max pooling's ``return_indices`` vs the reference.

The indices come from a composite beside the engine's values — ``-inf``
padding, sliding windows, an argmax — where the reference computes both in
one kernel.  Values and indices have to agree exactly: one index off is a
gradient that ``max_unpool`` routes to the wrong pixel.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

_CASES: list[tuple[str, tuple[int, ...], dict[str, Any]]] = [
    ("max_pool1d", (2, 3, 17), {"kernel_size": 3, "stride": 2, "padding": 1}),
    ("max_pool1d", (2, 3, 9), {"kernel_size": 2, "stride": 2, "ceil_mode": True}),
    ("max_pool2d", (2, 3, 8, 10), {"kernel_size": 2}),
    ("max_pool2d", (2, 3, 11, 13), {"kernel_size": 3, "stride": 2, "padding": 1}),
    ("max_pool2d", (1, 2, 10, 10), {"kernel_size": 3, "stride": 2, "ceil_mode": True}),
    ("max_pool3d", (1, 2, 6, 6, 6), {"kernel_size": 2}),
    ("max_pool3d", (1, 2, 7, 9, 5), {"kernel_size": 3, "stride": 2, "padding": 1}),
    ("adaptive_max_pool1d", (2, 3, 21), {"output_size": 4}),
    ("adaptive_max_pool2d", (1, 4, 11, 13), {"output_size": (3, 3)}),
    ("adaptive_max_pool2d", (1, 4, 8, 8), {"output_size": (4, 4)}),
    ("adaptive_max_pool3d", (1, 2, 5, 7, 9), {"output_size": (2, 3, 4)}),
]


@pytest.mark.parity
@pytest.mark.parametrize(
    ("op", "shape", "kwargs"),
    _CASES,
    ids=[f"{op}-{i}" for i, (op, _shape, _kwargs) in enumerate(_CASES)],
)
def test_values_and_indices_match_the_reference(
    ref: Any, op: str, shape: tuple[int, ...], kwargs: dict[str, Any]
) -> None:
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    got, got_idx = getattr(F, op)(lucid.from_numpy(x.copy()), return_indices=True, **kwargs)
    want, want_idx = getattr(ref.nn.functional, op)(
        ref.from_numpy(x.copy()), return_indices=True, **kwargs
    )
    np.testing.assert_array_equal(got.numpy(), want.numpy())
    np.testing.assert_array_equal(got_idx.numpy(), want_idx.numpy())


@pytest.mark.parity
def test_the_indices_round_trip_through_max_unpool2d(ref: Any) -> None:
    x = np.random.default_rng(1).standard_normal((2, 3, 8, 10)).astype(np.float32)
    pooled, idx = F.max_pool2d(lucid.from_numpy(x.copy()), 2, return_indices=True)
    rpooled, ridx = ref.nn.functional.max_pool2d(ref.from_numpy(x.copy()), 2, return_indices=True)
    got = F.max_unpool2d(pooled, idx, 2, output_size=(8, 10)).numpy()
    want = ref.nn.functional.max_unpool2d(rpooled, ridx, 2, output_size=(8, 10))
    np.testing.assert_array_equal(got, want.numpy())


@pytest.mark.parity
def test_overlapping_windows_unpool_like_the_reference_forward_and_back(
    ref: Any,
) -> None:
    # Kernel 3, stride 2: neighbouring windows share an element and can
    # both report it, so an index repeats.  The forward writes it once, as
    # the reference does.  The gradient deliberately differs there: Lucid
    # gives each copy 1 / count, the derivative of that forward, where the
    # reference gives each copy the whole gradient and so counts a shared
    # maximum twice once pooling adds the copies back up.
    x = np.random.default_rng(3).standard_normal((1, 2, 9)).astype(np.float32)
    pooled, idx = F.max_pool1d(lucid.from_numpy(x.copy()), 3, 2, 1, return_indices=True)
    rpooled, ridx = ref.nn.functional.max_pool1d(
        ref.from_numpy(x.copy()), 3, 2, 1, return_indices=True
    )
    p = lucid.tensor(pooled.numpy(), requires_grad=True)
    rp = ref.tensor(rpooled.detach().numpy(), requires_grad=True)
    out = F.max_unpool1d(p, idx, 3, 2, 1)
    rout = ref.nn.functional.max_unpool1d(rp, ridx, 3, 2, 1)
    g = np.random.default_rng(4).standard_normal(tuple(out.shape)).astype(np.float32)
    (out * lucid.from_numpy(g)).sum().backward()
    (rout * ref.from_numpy(g)).sum().backward()
    np.testing.assert_array_equal(out.detach().numpy(), rout.detach().numpy())
    assert p.grad is not None
    flat = idx.numpy().reshape(idx.shape[0] * idx.shape[1], -1)
    copies = np.stack([np.bincount(row, minlength=9)[row] for row in flat])
    expected = rp.grad.numpy() / copies.reshape(tuple(idx.shape))
    assert (copies > 1).any(), "the case has to contain a shared maximum"
    np.testing.assert_allclose(p.grad.numpy(), expected, rtol=0, atol=1e-6)


@pytest.mark.parity
@pytest.mark.parametrize("n", [1, 2, 3])
def test_unpooling_infers_the_output_size_the_reference_does(ref: Any, n: int) -> None:
    shape = (1, 2) + (9,) * n
    x = np.random.default_rng(2).standard_normal(shape).astype(np.float32)
    kw: dict[str, Any] = {"kernel_size": 3, "stride": 2, "padding": 1}
    out, idx = getattr(F, f"max_pool{n}d")(lucid.from_numpy(x.copy()), return_indices=True, **kw)
    rout, ridx = getattr(ref.nn.functional, f"max_pool{n}d")(
        ref.from_numpy(x.copy()), return_indices=True, **kw
    )
    got = getattr(F, f"max_unpool{n}d")(out, idx, **kw)
    want = getattr(ref.nn.functional, f"max_unpool{n}d")(rout, ridx, **kw)
    # In 2-D and 3-D an element can be the maximum of three or more
    # overlapping windows; its copies are summed and divided back, which
    # is exact for two and within an ulp beyond that.
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-6, atol=0)
