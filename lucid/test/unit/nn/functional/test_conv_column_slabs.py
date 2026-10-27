"""A CPU convolution lowers its columns a slab at a time, with 64-bit offsets.

A (1, 96, 6, 480, 832) conv3d asked im2col for 96 * 27 x 4 * 480 * 832 =
4.1 G column elements.  The row offset ``row * M`` was an int, so past row
1344 it wrapped negative and the write landed 8 GB before the buffer: the
process died with SIGSEGV (reported while porting Self-Forcing's VAE, where
it also killed ``coreml.export``, which traces on the CPU).  Offsets are
64-bit now, and the column matrix is built in slabs along the first output
axis so it never holds more than a fixed budget — 16.5 GB for that forward
became 0.5 GB.

A slab is the same im2col with the first output extent cut and the padding
shifted, so slabbing must not change a single value.  The budget is lowered
through ``LUCID_CONV_COLUMN_BUDGET`` in a child process to force many slabs
on small inputs, and the answers compared with the unslabbed run.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np

_SCRIPT = textwrap.dedent("""
    import sys
    import numpy as np
    import lucid
    import lucid.nn.functional as F

    rng = np.random.default_rng(0)
    cases = [
        (F.conv1d, (2, 4, 21), (6, 2, 3),
         dict(stride=2, padding=1, dilation=2, groups=2)),
        (F.conv2d, (2, 3, 13, 11), (4, 3, 3, 2),
         dict(stride=(2, 1), padding=(1, 0), dilation=(1, 2))),
        (F.conv3d, (1, 4, 7, 6, 5), (6, 2, 3, 2, 3),
         dict(stride=(1, 2, 1), padding=(1, 0, 2), dilation=(2, 1, 1), groups=2)),
    ]
    out = {}
    for i, (op, xs, ws, kw) in enumerate(cases):
        x = lucid.tensor(rng.standard_normal(xs), requires_grad=True)
        w = lucid.tensor(rng.standard_normal(ws), requires_grad=True)
        b = lucid.tensor(rng.standard_normal(ws[0]), requires_grad=True)
        y = op(x, w, b, **kw)
        (y * y).sum().backward()
        out[f"y{i}"] = y.numpy()
        out[f"gx{i}"] = x.grad.numpy()
        out[f"gw{i}"] = w.grad.numpy()
        out[f"gb{i}"] = b.grad.numpy()
    np.savez(sys.argv[1], **out)
    """)


def _run(path: str, budget: str | None) -> dict[str, np.ndarray]:
    env = dict(os.environ)
    env.pop("LUCID_CONV_COLUMN_BUDGET", None)
    if budget is not None:
        env["LUCID_CONV_COLUMN_BUDGET"] = budget
    subprocess.run([sys.executable, "-c", _SCRIPT, path], env=env, check=True, timeout=300)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def test_slabbed_columns_give_the_unslabbed_answer(tmp_path) -> None:  # type: ignore[no-untyped-def]
    whole = _run(str(tmp_path / "whole.npz"), None)
    slabbed = _run(str(tmp_path / "slabbed.npz"), "64")
    assert whole.keys() == slabbed.keys()
    for key in whole:
        np.testing.assert_allclose(slabbed[key], whole[key], rtol=1e-12, atol=1e-12)
