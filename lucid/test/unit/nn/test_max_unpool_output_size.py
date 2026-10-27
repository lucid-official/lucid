"""``max_unpool`` allocated from an unvalidated ``output_size``.

``output_size`` is annotated ``tuple[int, ...]``, and at runtime anything
indexable satisfies that.  A tensor passed by mistake was read
element-wise, ``int(-1.0)`` became a spatial extent of ``-1``, and the
engine allocated from it — SIGSEGV, with no Python-level error to catch.

Found by the audit's module axis: once every ``nn.Module`` class could be
constructed, the forward probe tried a three-argument call, and
``MaxUnpool1d``'s third positional is ``output_size``.  The whole sweep
died with exit 138 rather than reporting anything.

A wrong argument has to raise.  It must not take the process down.
"""

import subprocess
import sys

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


def _pair(shape=(2, 3, 4)):
    x = lucid.tensor(np.zeros(shape, dtype=np.float32))
    indices = lucid.tensor(np.zeros(shape, dtype=np.int32))
    return x, indices


# ── the guard ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [(-1,), (0,), (2, -3)])
def test_a_non_positive_extent_is_refused(bad) -> None:
    x, indices = _pair()
    with pytest.raises(ValueError, match="positive"):
        F.max_unpool1d(x, indices, kernel_size=2, output_size=bad)


def test_a_tensor_passed_as_output_size_raises_rather_than_crashing() -> None:
    """The exact shape of the original failure: the module form, whose
    third positional is ``output_size``."""
    x, indices = _pair()
    module = nn.MaxUnpool1d(kernel_size=2)
    bogus = lucid.tensor(np.array([1.0, -1.0], dtype=np.float64))
    with pytest.raises(Exception, match="positive"):
        module(x, indices, bogus)


def test_the_process_survives_it() -> None:
    """A guard that raised in-process would still be worth nothing if the
    engine had already been reached, so this asserts on the exit code of
    a fresh interpreter."""
    code = (
        "import numpy as np, lucid, lucid.nn as nn\n"
        "x = lucid.tensor(np.zeros((2,3,4), dtype=np.float32))\n"
        "i = lucid.tensor(np.zeros((2,3,4), dtype=np.int32))\n"
        "m = nn.MaxUnpool1d(kernel_size=2)\n"
        "try:\n"
        "    m(x, i, lucid.tensor(np.array([1.0, -1.0])))\n"
        "except Exception as e:\n"
        "    print(type(e).__name__)\n"
    )
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True
    )
    assert done.returncode == 0, f"exited {done.returncode}: {done.stderr[-300:]}"
    assert "Error" in done.stdout, done.stdout


# ── and the ordinary path is untouched ────────────────────────────────────────


def test_a_valid_unpool_still_works() -> None:
    x, indices = _pair()
    out = F.max_unpool1d(x, indices, kernel_size=2, output_size=(8,))
    assert tuple(out.shape) == (2, 3, 8)


def test_values_land_where_the_indices_say() -> None:
    x = lucid.tensor(np.array([[[1.0, 2.0]]], dtype=np.float32))
    indices = lucid.tensor(np.array([[[0, 3]]], dtype=np.int32))
    out = np.asarray(F.max_unpool1d(x, indices, kernel_size=2, output_size=(4,)).numpy())
    assert np.allclose(out, [[[1.0, 0.0, 0.0, 2.0]]])


@pytest.mark.parametrize(
    "fn,shape,size",
    [
        (F.max_unpool1d, (1, 1, 2), (4,)),
        (F.max_unpool2d, (1, 1, 2, 2), (4, 4)),
        (F.max_unpool3d, (1, 1, 2, 2, 2), (4, 4, 4)),
    ],
)
def test_every_rank_shares_the_guard(fn, shape, size) -> None:
    x = lucid.tensor(np.zeros(shape, dtype=np.float32))
    indices = lucid.tensor(np.zeros(shape, dtype=np.int32))
    assert fn(x, indices, kernel_size=2, output_size=size) is not None
    with pytest.raises(ValueError, match="positive"):
        fn(x, indices, kernel_size=2, output_size=tuple(-1 for _ in size))


@pytest.mark.parametrize("explicit", [False, True], ids=["default-size", "given-size"])
@pytest.mark.parametrize(
    "fn,n,shape",
    [
        (F.max_unpool1d, 1, ()),
        (F.max_unpool2d, 2, (2,)),
        (F.max_unpool2d, 2, (2, 2)),
        (F.max_unpool3d, 3, (2, 2)),
        (F.max_unpool3d, 3, (1, 1, 1, 1, 2, 2)),
    ],
)
def test_an_input_of_the_wrong_rank_is_refused(fn, n, shape, explicit) -> None:
    """Only ``(N, C, *spatial)`` and ``(C, *spatial)`` inputs are taken.

    With the default size, too few axes reached a ``zip`` against the
    kernel, which named neither the function nor the shape once it turned
    strict; with a given size, a 1-D input came back as a 2-D image.
    """
    x = lucid.zeros(*shape)
    indices = lucid.zeros(*shape, dtype=lucid.int64)
    size = {"output_size": (4,) * n} if explicit else {}
    with pytest.raises(ValueError, match=rf"max_unpool{n}d: expected a {n + 1}-D"):
        fn(x, indices, kernel_size=2, **size)
