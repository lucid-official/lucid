"""Parity-tier conftest.

Auto-skip every test under ``lucid/test/parity/`` when the reference
framework isn't installed.  Doing this at collection time means the
parity tier never even attempts to import the reference module from
inside individual test modules.

Also hosts the MNIST fixture.  It lives here rather than in either MNIST
module so the download happens once per session no matter how many of them
run, and so the two share the exact same arrays.
"""

import numpy as np
import pytest

from lucid.test._fixtures.ref_framework import ref_module, require_ref_vision


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if ref_module() is not None:
        return
    skip_marker = pytest.mark.skip(
        reason="reference framework not installed — parity tier auto-skipped"
    )
    for item in items:
        if "lucid/test/parity/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def mnist(tmp_path_factory: pytest.TempPathFactory) -> tuple[np.ndarray, ...]:
    """Real MNIST as 32x32 float32 arrays, downloaded once per session."""
    from lucid.test.parity._mnist_harness import load_mnist

    root = tmp_path_factory.mktemp("mnist")
    return load_mnist(require_ref_vision(), root)
