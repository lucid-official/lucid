"""A GPU scatter with nothing to place, or nowhere to place it (LCD-382).

MLX's multi-axis scatter (one index array per axis) checks each update
slice against the base's extents before it counts the updates.  So a base
with an empty axis raised "[scatter] Updates with shape (0,1,1) are too
large for array with shape (0,0)" on Metal while the CPU answered: the
backward of ``diagonal`` on a (0, 3) input, of ``interpolate`` and
``grid_sample`` over an empty batch, and ``scatter_reduce`` into an empty
base.  Every multi-axis scatter on the GPU stream now goes through the
guarded helpers in ``lucid/_C/backend/gpu/AxisIndex.h``; this file holds
each caller to the reference on empty shapes, and holds the backend to
calling MLX's multi-axis scatter only through those helpers.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

_EMPTY_MATRICES = [(0, 0), (0, 3), (3, 0), (2, 0, 0), (0, 2, 2), (2, 0, 3)]


def _grad_matches(ref, device, shape, call):
    x = lucid.zeros(shape, device=device, requires_grad=True)
    xr = ref.zeros(shape, requires_grad=True)
    call(lucid, x).sum().backward()
    call(ref, xr).sum().backward()
    assert tuple(x.grad.shape) == tuple(xr.grad.shape) == shape
    np.testing.assert_array_equal(x.grad.numpy(), xr.grad.numpy())


@pytest.mark.parametrize("offset", [0, 1, -1, 5])
@pytest.mark.parametrize("shape", _EMPTY_MATRICES, ids=str)
def test_diagonal_backward(ref, device, shape, offset):
    _grad_matches(ref, device, shape, lambda lib, t: lib.diagonal(t, offset=offset))


@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3)], ids=str)
def test_diag_embed_backward(ref, device, shape):
    _grad_matches(ref, device, shape, lambda lib, t: lib.diag_embed(t))


@pytest.mark.parametrize(
    ("mode", "shape", "size"),
    [
        ("nearest", (0, 2, 4, 4), (8, 8)),
        ("nearest", (0, 2, 4, 4, 4), (8, 8, 8)),
        ("bilinear", (0, 2, 4, 4), (8, 8)),
        ("trilinear", (0, 2, 4, 4, 4), (8, 8, 8)),
    ],
    ids=str,
)
def test_interpolate_backward_of_an_empty_batch(ref, device, mode, shape, size):
    _grad_matches(
        ref,
        device,
        shape,
        lambda lib, t: (F if lib is lucid else lib.nn.functional).interpolate(
            t, size=size, mode=mode
        ),
    )


def test_grid_sample_backward_of_an_empty_batch(ref, device):
    x = lucid.zeros((0, 2, 4, 4), device=device, requires_grad=True)
    xr = ref.zeros((0, 2, 4, 4), requires_grad=True)
    grid = lucid.zeros((0, 3, 3, 2), device=device)
    F.grid_sample(x, grid, align_corners=False).sum().backward()
    ref.nn.functional.grid_sample(xr, ref.zeros((0, 3, 3, 2)), align_corners=False).sum().backward()
    assert tuple(x.grad.shape) == tuple(xr.grad.shape) == (0, 2, 4, 4)


@pytest.mark.parametrize("reduce", ["amax", "amin", "prod", "sum"])
def test_scatter_reduce_into_an_empty_base(ref, device, reduce):
    base = lucid.zeros((0, 3), device=device)
    idx = lucid.zeros((0, 3), dtype=lucid.int64, device=device)
    got = lucid.scatter_reduce(base, 1, idx, lucid.zeros((0, 3), device=device), reduce)
    want = ref.zeros((0, 3)).scatter_reduce(
        1, ref.zeros((0, 3), dtype=ref.int64), ref.zeros((0, 3)), reduce
    )
    assert tuple(got.shape) == tuple(want.shape) == (0, 3)


_GPU_BACKEND = Path(lucid.__file__).parent / "_C" / "backend" / "gpu"
_RAW_MULTIAXIS_SCATTER = re.compile(r"(::mlx::core::|\bmx::)scatter(_add|_max|_min|_prod)?\(")
# The one place a raw call belongs: the branch of a guarded helper that runs
# once ``scatter_places_nothing`` has said there is something to place.
_HELPER_BRANCH = re.compile(r"\s*: ::mlx::core::scatter")


def test_the_gpu_backend_scatters_through_the_guarded_helpers():
    """MLX's multi-axis scatter is called only inside the guarded helpers."""
    offenders = [
        f"{path.name}:{lineno}"
        for path in sorted(_GPU_BACKEND.glob("*.h")) + sorted(_GPU_BACKEND.glob("*.mm"))
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if _RAW_MULTIAXIS_SCATTER.search(line)
        and not (path.name == "AxisIndex.h" and _HELPER_BRANCH.match(line))
    ]
    assert offenders == []
