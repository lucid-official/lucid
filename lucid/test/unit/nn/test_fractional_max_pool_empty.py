"""``fractional_max_pool{2,3}d`` of an input with no plane to pool.

``F.fractional_max_pool2d(lucid.zeros(0, 3, 5, 5), 2, output_size=(2, 2))``
raised ``stack: empty input``: the body pools plane by plane and stacked an
empty list.  The reference answers an empty batch with an empty tensor of the
pooled shape that requires grad when the input does.  Empty channels get the
same answer, as Lucid's other pools give them.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

_SIZE = 5
_OUT = 2

# (label, leading axes): empty batch, empty channels, unbatched empty channels.
_LEADS = [("empty-batch", (0, 3)), ("empty-channels", (2, 0)), ("unbatched", (0,))]


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("lead", [lead for _, lead in _LEADS], ids=[k for k, _ in _LEADS])
@pytest.mark.parametrize("return_indices", [False, True], ids=["values", "indices"])
def test_no_plane_pools_to_an_empty_result(
    device: str, n: int, lead: tuple[int, ...], return_indices: bool
) -> None:
    x = lucid.zeros(*lead, *(_SIZE,) * n, device=device, requires_grad=True)
    fn = getattr(F, f"fractional_max_pool{n}d")
    got = fn(x, 2, output_size=_OUT, return_indices=return_indices)
    out, index = got if return_indices else (got, None)
    want = (*lead, *(_OUT,) * n)
    assert tuple(out.shape) == want
    assert out.dtype == x.dtype and out.device == x.device
    assert out.requires_grad
    if index is not None:
        assert tuple(index.shape) == want
        assert index.dtype == lucid.int64 and index.device == x.device
    out.sum().backward()
    assert x.grad is not None and tuple(x.grad.shape) == tuple(x.shape)


@pytest.mark.parametrize("n", [2, 3])
def test_the_module_pools_an_empty_batch(device: str, n: int) -> None:
    pool = getattr(nn, f"FractionalMaxPool{n}d")(2, output_size=_OUT)
    x = lucid.zeros(0, 3, *(_SIZE,) * n, device=device)
    assert tuple(pool(x).shape) == (0, 3, *(_OUT,) * n)
