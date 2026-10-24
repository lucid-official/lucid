"""Fixtures shared by the compile tests."""

from collections.abc import Iterator

import pytest

import lucid


@pytest.fixture
def inference() -> Iterator[None]:
    """Run the test under ``no_grad``, where a compiled forward is used.

    Under grad mode a call that reads a tensor requiring grad — any model
    with trainable parameters — runs eagerly so that its result is
    differentiable (LCD-582).  A test of the compiled forward itself opts
    in with ``pytestmark = pytest.mark.usefixtures("inference")``.
    """
    with lucid.no_grad():
        yield
