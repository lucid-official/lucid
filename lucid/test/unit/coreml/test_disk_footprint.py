"""A test in this directory leaves nothing on the disk behind it.

Core ML specialises every package it loads into a bundle holding a full
copy of the weights, and never reads that bundle again once the handle
is closed.  Left to accumulate, a run of this directory took a disk with
38 GB free past macOS's near-low-disk mark, and the purge that followed
landed under a prediction Core ML was preparing — the process aborted
inside Core ML's GPU encoder (CHA-22; the conftest has the account).

So the conftest gives Core ML a home inside the session's temporary
directory, and Lucid's compile cache a directory beside it, and removes,
after each passing test, the packages it wrote, the compiled models kept
for them and the bundles they produced.  These tests hold it to that: the
first writes and loads a package, the second looks for what the first
left.
"""

import os

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.test.unit.coreml._helpers import bundles_under

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)

#: What the writing test left, for the one after it to look for.
_WRITTEN: dict[str, list[str]] = {}


class _Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Conv2d(3, 8, 3, padding=1)
        self.head = nn.Linear(8, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.body(x).relu().mean(dim=(2, 3)))


def test_core_ml_is_given_a_home_inside_the_session(core_ml_home, tmp_path_factory):
    if core_ml_home is None:
        pytest.skip("CFFIXED_USER_HOME was chosen by whoever started the session")
    assert os.environ["CFFIXED_USER_HOME"] == str(core_ml_home)
    assert core_ml_home.is_relative_to(tmp_path_factory.getbasetemp())


def test_and_the_compile_cache_a_directory_inside_it(compile_cache, tmp_path_factory):
    if compile_cache is None:
        pytest.skip("LUCID_COREML_CACHE_DIR was chosen by whoever started the session")
    assert os.environ["LUCID_COREML_CACHE_DIR"] == str(compile_cache)
    assert compile_cache.is_relative_to(tmp_path_factory.getbasetemp())


def test_a_test_writes_a_package_and_core_ml_specialises_it(tmp_path, core_ml_home, compile_cache):
    lucid.manual_seed(0)
    model = _Small().eval()
    x = lucid.randn(1, 3, 16, 16)
    path = tmp_path / "small.mlpackage"
    before = bundles_under(core_ml_home) if core_ml_home is not None else set()

    exported = cml.export(model, x, str(path))
    try:
        assert tuple(exported.predict(x).shape) == (1, 4)
        kept = exported._lease.path
    finally:
        exported.close()

    assert path.exists()
    _WRITTEN["paths"] = [str(path)]
    if compile_cache is not None:
        # The compiled model outlives the handle by design — the next load
        # of this package opens it — but not the test.
        assert os.path.isdir(kept)
        _WRITTEN["paths"].append(kept)
    if core_ml_home is None:
        return
    made = sorted(bundles_under(core_ml_home) - before)
    if not made:
        # Core ML reads the home once per process. A session that loaded a
        # package from another directory first has already fixed it.
        pytest.skip("Core ML fixed its home before this directory's first load")
    _WRITTEN["paths"] += made


def test_and_the_next_test_finds_none_of_it():
    if "paths" not in _WRITTEN:
        pytest.skip("follows the test above, which did not get as far as writing")
    left = [path for path in _WRITTEN["paths"] if os.path.exists(path)]
    assert not left, f"outlived the test that wrote them: {left}"
