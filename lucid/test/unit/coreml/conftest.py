"""What the Core ML tests may not do to the machine running them.

Inside a virtual machine
------------------------
A hosted CI runner is a macOS virtual machine, and the Core ML runtime
inside it is not the one on a device.  There is no Neural Engine, and the
GPU goes unused: the probe in ``.github/workflows/coreml-vm-probe.yml``
got the same answer, to the last digit, from CPU_ONLY, CPU_AND_GPU and
ALL — every package on BNNS's CPU path.  That path lands one comparison
a thousand times further from eager than on hardware and traps at
prediction on two unrelated packages: SIGTRAP, nothing in the log, and
the rest of the suite gone with the process.  Other compute units do not
route around it.

So under a hypervisor, loading a package — the step
every prediction, verification and compute plan goes through — skips the
test instead.  Tracing and writing the package still run there; executing
it is left to hardware.

Session-scoped, so a module- or class-scoped fixture that loads a
package is refused too rather than slipping in before a per-test patch.

On the disk
-----------
Every package Core ML loads is compiled and then specialised into a
bundle under ``~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache`` that
holds a full copy of the weights — 1.1 GB for YOLO v1.  The bundle is
keyed by the compiled model's path, and the runtime used to compile to a
fresh path on every load, so no bundle was ever read twice; they stay
until macOS purges them.  By 41% of one run of this directory there were
7 GB of them, beside 14 GB of packages kept in ``tmp_path`` for the rest
of the session, and the disk that had 38 GB free was down to 118 MB.
(``lucid.coreml`` now keeps one compiled model per package content, so a
package opened again reuses its bundle — but every *distinct* package a
test writes still costs a compiled model and a bundle, and this
directory writes hundreds.)

Long before full, the disk crosses macOS's near-low-disk mark, and
``cache_delete`` purges those bundles — 3.4 GB in under a second.  Twice
that landed while Core ML was preparing the first YOLO prediction, and
Core ML aborted the process from inside its own GPU encoder
(``MPSGraphTensorData.mm:266``, ``shape.count != strides.count``) 95 ms
and 54 ms after the purge finished.  Nothing of Lucid's is on that path;
what Lucid controls is how much disk the suite eats.

So the session gives Core ML a home of its own inside the session's
temporary directory — ``CFFIXED_USER_HOME``, which Core ML reads once, at
the first load in the process — and Lucid's compile cache a directory of
its own beside it (``LUCID_COREML_CACHE_DIR``), and every test removes the
packages it wrote, the compiled models kept for them and the bundles they
produced.  A failing test keeps all three, to be looked at.
"""

import os
import shutil
from collections.abc import Generator, Iterator
from pathlib import Path

import pytest

from lucid._C import engine as _C_engine
from lucid.coreml import _cache
from lucid.test.unit.coreml._helpers import bundles_under, under_hypervisor

#: Set on an item whose setup, call or teardown failed.
_FAILED = pytest.StashKey[bool]()


def _refuse(*args: object, **kwargs: object) -> object:
    pytest.skip("the Core ML runtime is not exercised inside a VM; runs on hardware")


@pytest.fixture(autouse=True, scope="session")
def _no_core_ml_runtime_in_a_vm() -> Iterator[None]:
    if not under_hypervisor() or not hasattr(_C_engine, "coreml"):
        yield
        return
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_C_engine.coreml, "load_model", _refuse)
        patch.setattr(_C_engine.coreml, "compile_model", _refuse)
        yield


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if report.failed:
        item.stash[_FAILED] = True
    return report


@pytest.fixture(autouse=True, scope="session")
def core_ml_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path | None]:
    """The home Core ML caches under for this session, or ``None``.

    ``None`` when the caller already chose one with ``CFFIXED_USER_HOME``;
    that choice is theirs to clean up.
    """
    if os.environ.get("CFFIXED_USER_HOME"):
        yield None
        return
    home = tmp_path_factory.mktemp("coreml-home")
    os.environ["CFFIXED_USER_HOME"] = str(home)
    try:
        yield home
    finally:
        os.environ.pop("CFFIXED_USER_HOME", None)
        shutil.rmtree(home, ignore_errors=True)


@pytest.fixture(autouse=True, scope="session")
def compile_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path | None]:
    """Where this session's compiled models are kept, or ``None``.

    ``None`` when the caller already chose with ``LUCID_COREML_CACHE_DIR``;
    that choice is theirs to clean up.
    """
    if os.environ.get("LUCID_COREML_CACHE_DIR"):
        yield None
        return
    root = tmp_path_factory.mktemp("coreml-cache")
    os.environ["LUCID_COREML_CACHE_DIR"] = str(root)
    try:
        yield root
    finally:
        os.environ.pop("LUCID_COREML_CACHE_DIR", None)
        shutil.rmtree(root, ignore_errors=True)


def _cache_files(root: Path | None) -> set[Path]:
    """What the compile cache holds: the versioned store under ``root``."""
    if root is None:
        return set()
    store = Path(_cache.store_dir(str(root)))
    return set(store.iterdir()) if store.is_dir() else set()


@pytest.fixture(autouse=True)
def _nothing_written_outlives_its_test(
    request: pytest.FixtureRequest, core_ml_home: Path | None, compile_cache: Path | None
) -> Iterator[None]:
    room: Path | None = None
    if "tmp_path" in request.fixturenames:
        room = request.getfixturevalue("tmp_path")
    # A module-scoped handle is set up before this snapshot, so its bundle
    # and its compiled model are in it and survive the tests that share it.
    before = bundles_under(core_ml_home) if core_ml_home is not None else set()
    kept = _cache_files(compile_cache)
    yield
    if request.node.stash.get(_FAILED, False):
        return
    if room is not None:
        shutil.rmtree(room, ignore_errors=True)
    if core_ml_home is not None:
        for bundle in bundles_under(core_ml_home) - before:
            shutil.rmtree(bundle, ignore_errors=True)
    for left in _cache_files(compile_cache) - kept:
        if left.is_dir():
            shutil.rmtree(left, ignore_errors=True)
        else:
            left.unlink(missing_ok=True)
