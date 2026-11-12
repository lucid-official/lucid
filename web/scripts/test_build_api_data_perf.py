#!/usr/bin/env python3
"""Guards for what ``build-api-data.py`` costs and how it ends (LCD-969).

Two defects made the nightly ``Docs api-data refresh`` die on the hosted
runner ("lost communication") and the Docs workflow fail on every push:

* **A whole-tree re-parse per documented module.**  ``GriffeLoader.load(path)``
  parses the *top-level package* from disk on every call and returns only the
  object asked for; the builder called it once per slug, so a build parsed
  ``lucid/`` 160+ times (~4 s each) and left a cyclic tree behind each time —
  46 modules took 227 s and 6 GB, the full build 26 min on the runner.  The
  cost was quadratic (modules x tree size), so it grew with every package
  added.  Fixed in :func:`_load_object`; pinned here by *counting walks*, not
  by timing them.
* **A process that did not end.**  Shutdown runs a full collection over the
  leftover Griffe graph: 83 s for those 46 modules, 25 min on the runner, after
  the last output.  ``__main__`` now ends the process itself once every output
  is written (:func:`_end_process`).

Standalone like the other ``web/scripts/test_*.py`` — ``python3
web/scripts/test_build_api_data_perf.py``, no pytest needed, also
pytest-collectable.  The builds need Griffe (``pip install griffe``); without
it the build tests are skipped, the pure ones still run.  Each build is a few
slugs: the first parse of ``lucid/`` is ~4 s, so the file takes ~15 s.
"""

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "build-api-data.py"
_CACHED = _HERE / "build-api-data-cached.py"

# One slug of each kind the builder serves: a plain package, the Tensor class
# (its own loader call), and a synthesised page (the lazy-name rescue loads a
# dozen home modules).  Together they reach every place that asked the loader
# for an object.
_ONE_OF_EACH = ["lucid.einops", "lucid.tensor", "lucid.ops"]
_SMALL = ["lucid.einops", "lucid.peft.tuners.beft"]

# Once the "Done." line is out the process has nothing left to do.  Measured:
# ~0.01 s with ``_end_process``, ~1.3 s without it on this very build (the
# interpreter's final collection over the loaded tree), minutes on the full one.
_EXIT_BOUND_S = 1.0


def _griffe_available() -> bool:
    return importlib.util.find_spec("griffe") is not None


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("_build_api_data_perf", _SCRIPT)
    assert spec and spec.loader, f"cannot load {_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # main() is behind `if __name__ == "__main__":`
    return mod


def _load_cached() -> Any:
    spec = importlib.util.spec_from_file_location("_build_api_data_cached_perf", _CACHED)
    assert spec and spec.loader, f"cannot load {_CACHED}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeLoader:
    """What ``_load_object`` touches of a ``GriffeLoader``: ``load`` and a
    ``modules_collection`` that holds the packages loaded so far."""

    class _Collection(dict):
        def get_member(self, path: str) -> Any:
            node: Any = self[path.split(".", 1)[0]]
            for part in path.split(".")[1:]:
                node = node[part]
            return node

    def __init__(self) -> None:
        self.modules_collection = self._Collection()
        self.loads: list[str] = []

    def load(self, top: str) -> None:
        self.loads.append(top)
        self.modules_collection[top] = {"nn": {"functional": "F"}, "optim": "O"}


def test_load_object_walks_a_package_once() -> None:
    m = _load_builder()
    loader = _FakeLoader()
    assert m._load_object(loader, "lucid.nn.functional") == "F"
    assert m._load_object(loader, "lucid.optim") == "O"
    assert m._load_object(loader, "lucid.nn.functional") == "F"
    assert loader.loads == ["lucid"], (
        "every request after the first must read the package already in the "
        f"collection, not ask the loader to parse it again; loads: {loader.loads}"
    )


def test_load_object_reports_an_unknown_path() -> None:
    m = _load_builder()
    loader = _FakeLoader()
    try:
        m._load_object(loader, "lucid.no_such_module")
    except KeyError:
        return
    raise AssertionError("an unknown path must raise KeyError, as GriffeLoader.load did")


def test_peak_rss_is_reported_in_gigabytes() -> None:
    peak = _load_builder()._peak_rss_gb()
    assert 0.001 < peak < 1024, (
        f"a Python process is not {peak} GB: the platform's ru_maxrss unit is off"
    )


def test_a_build_parses_the_package_once() -> None:
    """The complexity pin: the number of whole-package walks is 1, however many
    slugs are built and whichever way each reaches the loader."""
    if not _griffe_available():
        print("  skip  test_a_build_parses_the_package_once (griffe not installed)")
        return
    import griffe

    m = _load_builder()
    walks: list[str] = []
    real_load = griffe.GriffeLoader.load

    def counting_load(self: Any, objspec: Any, *args: Any, **kwargs: Any) -> Any:
        walks.append(str(objspec))
        return real_load(self, objspec, *args, **kwargs)

    saved_out, saved_argv = m.OUT_DIR, sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        m.OUT_DIR = Path(tmp)
        sys.argv = ["build-api-data.py", "--slugs", *_ONE_OF_EACH]
        griffe.GriffeLoader.load = counting_load
        try:
            m.main()
        finally:
            griffe.GriffeLoader.load = real_load
            m.OUT_DIR, sys.argv = saved_out, saved_argv
        written = sorted(p.stem for p in Path(tmp).glob("*.json"))
    assert written == sorted(_ONE_OF_EACH), f"the build must still write every page: {written}"
    assert len(walks) == 1, (
        f"{len(walks)} whole-package parses for {len(_ONE_OF_EACH)} slugs ({walks}); "
        "something asks GriffeLoader.load() per object again — go through _load_object"
    )


def _run_builder(*args: str, out: Path, hash_seed: str | None = None) -> tuple[int, str, float]:
    """Run the script as ``__main__``; return (status, output, seconds from the
    last output to the process being gone)."""
    env = {**os.environ, "LUCID_API_DATA_OUT": str(out), "PYTHONUNBUFFERED": "1"}
    if hash_seed is not None:
        env["PYTHONHASHSEED"] = hash_seed
    proc = subprocess.Popen(
        [sys.executable, str(_SCRIPT), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    last_output = time.monotonic()
    for line in proc.stdout:  # ends when the process closes the pipe, i.e. exits
        lines.append(line)
        last_output = time.monotonic()
    try:
        status = proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError(
            "the builder did not exit within 120 s of closing its output"
        ) from None
    return status, "".join(lines), time.monotonic() - last_output


def test_the_process_ends_promptly_after_its_last_output() -> None:
    if not _griffe_available():
        print("  skip  test_the_process_ends_promptly_after_its_last_output (griffe not installed)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        status, output, gap = _run_builder("--slugs", *_SMALL, out=Path(tmp))
        written = sorted(p.stem for p in Path(tmp).glob("*.json"))
    assert status == 0, f"build failed ({status}):\n{output[-2000:]}"
    assert "Done." in output, f"no completion line:\n{output[-2000:]}"
    assert written == sorted(_SMALL), f"pages written: {written}"
    print(f"  last output -> process gone: {gap:.3f} s (bound {_EXIT_BOUND_S} s)")
    assert gap < _EXIT_BOUND_S, (
        f"the process lingered {gap:.2f} s after its last output (bound {_EXIT_BOUND_S} s) — "
        "interpreter shutdown is collecting the Griffe graph again; __main__ must end through _end_process"
    )


def test_a_failed_build_still_exits_non_zero() -> None:
    """The fast exit must not turn the failure paths into success."""
    if not _griffe_available():
        print("  skip  test_a_failed_build_still_exits_non_zero (griffe not installed)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        status, output, _gap = _run_builder("--module", "lucid.no_such_module", out=Path(tmp))
    assert status == 1, f"an unknown slug must exit 1, got {status}:\n{output[-1000:]}"
    assert "unknown slug" in output, output[-1000:]


def test_the_output_does_not_depend_on_the_hash_seed() -> None:
    """``lucid.peft``'s documented constants were appended in set-iteration
    order, so two builds of the same source differed in the committed page and
    every nightly refresh churned it.  Same source, different seed: same bytes."""
    if not _griffe_available():
        print("  skip  test_the_output_does_not_depend_on_the_hash_seed (griffe not installed)")
        return
    pages: list[bytes] = []
    for seed in ("1", "2"):
        with tempfile.TemporaryDirectory() as tmp:
            status, output, _gap = _run_builder(
                "--slugs", "lucid.peft", out=Path(tmp), hash_seed=seed
            )
            assert status == 0, f"build failed ({status}):\n{output[-2000:]}"
            pages.append((Path(tmp) / "lucid.peft.json").read_bytes())
    assert pages[0] == pages[1], "lucid.peft.json differs between PYTHONHASHSEED=1 and 2"


def test_a_signal_death_is_named_and_exits_128_plus_the_signal() -> None:
    """A build the kernel kills (the out-of-memory killer is SIGKILL) used to
    leave the wrapper exiting 247 with no word in the log."""
    cached = _load_cached()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert cached._exit_status(0) == 0
        assert cached._exit_status(1) == 1
        assert cached._exit_status(-9) == 137
    assert "SIGKILL" in out.getvalue() and "out-of-memory" in out.getvalue(), out.getvalue()


def test_the_wrapper_reports_a_killed_build_end_to_end() -> None:
    cached = _load_cached()
    names = (
        "BUILD_SCRIPT",
        "OUT_DIR",
        "CACHE_DIR",
        "CACHE_FILE",
        "MODULE_CACHE",
        "_documented_slugs",
    )
    saved = {name: getattr(cached, name) for name in names}
    saved_argv, saved_force = sys.argv, os.environ.get("FORCE_API_BUILD")
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "killer.py").write_text("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
        cached.BUILD_SCRIPT = root / "killer.py"
        cached.OUT_DIR = root / "api-data"
        cached.CACHE_DIR = cached.OUT_DIR / ".cache"
        cached.CACHE_FILE = cached.OUT_DIR / ".cache-key"
        cached.MODULE_CACHE = cached.CACHE_DIR / "modules.json"
        cached._documented_slugs = lambda: []
        sys.argv, os.environ["FORCE_API_BUILD"] = ["build-api-data-cached.py"], "1"
        try:
            with contextlib.redirect_stdout(out):
                status = cached.main()
        finally:
            for name, value in saved.items():
                setattr(cached, name, value)
            sys.argv = saved_argv
            if saved_force is None:
                os.environ.pop("FORCE_API_BUILD", None)
            else:
                os.environ["FORCE_API_BUILD"] = saved_force
    assert status == 137, f"a SIGKILLed build must exit 137, got {status}:\n{out.getvalue()}"
    assert "SIGKILL" in out.getvalue(), out.getvalue()


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    failed = 0
    for t in _TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {exc}", file=sys.stderr)
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} api-data build-cost tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
