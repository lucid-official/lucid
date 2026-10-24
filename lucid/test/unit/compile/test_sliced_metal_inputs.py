"""A compiled module reads a sliced input from where the slice begins.

A leading-axis slice of a Metal tensor, ``x[3:]``, is row-contiguous but
begins part-way into its buffer.  The compiled executable handed MPSGraph
the buffer alone, with no offset, so it read from the buffer's start: the
module computed on ``x[:n]`` and returned it as the answer for ``x[3:]``.
``.contiguous()`` did not help, because the slice already is contiguous.
Minibatching a Metal tensor — ``model(X[i:i + bs])`` — trained on the first
batch every step, with no error.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import metal_available

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="metal unavailable"),
    pytest.mark.usefixtures("inference"),
]


@pytest.mark.parametrize("start", [1, 3, 5])
def test_a_slice_is_read_from_its_start(start: int) -> None:
    lucid.manual_seed(0)
    model = nn.Linear(4, 3).to("metal").eval()
    compiled = lucid.compile(model)
    x = lucid.randn(8, 4).to("metal")
    with lucid.no_grad():
        want = model(x[start:]).numpy()
        got = compiled(x[start:]).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_a_minibatch_loop_sees_every_batch() -> None:
    lucid.manual_seed(1)
    model = nn.Linear(4, 2).to("metal").eval()
    compiled = lucid.compile(model, dynamic=False)
    data = lucid.randn(12, 4).to("metal")
    with lucid.no_grad():
        for i in range(0, 12, 4):
            batch = data[i : i + 4]
            np.testing.assert_allclose(
                compiled(batch).numpy(), model(batch).numpy(), rtol=1e-5, atol=1e-6
            )
