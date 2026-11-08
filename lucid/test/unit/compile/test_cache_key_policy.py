"""The executable cache keys on the settings that decide what a compile emits (LCD-922).

``LUCID_COMPILE_RANK_CANON_MIN_LEAD`` (and the manual-VJP switches) are read at
every compile, but the session cache, the disk cache and Python's per-module
cache used to key on the trace alone: flipping the variable after a graph had
compiled handed back the executable built under the old value, so the pass
looked as if it had not run, or as if it had run while turned off.  Every
rank-canon test cleared the session cache to get around it.

The signal here is timing-free: the number of cached executables, which grows
exactly when a flipped setting forces a new compile and stays put when the
setting returns to a value already compiled.
"""

import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile._core.signature import signature_of
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

_LEAD = "LUCID_COMPILE_RANK_CANON_MIN_LEAD"
_VJP = "LUCID_MANUAL_VJP"
_VJP_REQUIRE = "LUCID_MANUAL_VJP_REQUIRE"

needs_metal = pytest.mark.skipif(not metal_available(), reason="compile needs Metal")

# A rank-5 elementwise program at lead 96 (>= the default gate of 64).
_SHAPE = (96, 8, 2, 4, 4)


@pytest.fixture(autouse=True)
def _env() -> Iterator[None]:
    """Each test starts from the default policy and an empty session cache."""
    with mock.patch.dict(os.environ):
        for name in (_LEAD, _VJP, _VJP_REQUIRE):
            os.environ.pop(name, None)
        _C_engine.compile.session_cache_clear()
        yield
        _C_engine.compile.session_cache_clear()


def _fn(t: lucid.Tensor) -> lucid.Tensor:
    return t * 2.0 + 1.0


def _input() -> lucid.Tensor:
    return lucid.arange(96 * 8 * 2 * 4 * 4, dtype=lucid.float32).reshape(*_SHAPE).to(COMPILE_DEVICE)


def _run(x: lucid.Tensor) -> lucid.Tensor:
    """A fresh ``lucid.compile`` of ``_fn`` (so only the shared C++ cache can serve it)."""
    with lucid.no_grad():
        return lucid.compile(_fn)(x)


# ── the policy ────────────────────────────────────────────────────────────────────────


def test_the_policy_reads_each_setting_at_every_call() -> None:
    assert _C_engine.compile.compile_policy() == (64, 1, 0)
    with mock.patch.dict(os.environ, {_LEAD: "0"}):
        assert _C_engine.compile.compile_policy() == (0, 1, 0)
    with mock.patch.dict(os.environ, {_LEAD: "-1"}):
        assert _C_engine.compile.compile_policy() == (-1, 1, 0)
    with mock.patch.dict(os.environ, {_VJP: "0"}):
        assert _C_engine.compile.compile_policy() == (64, 0, 0)
    with mock.patch.dict(os.environ, {_VJP_REQUIRE: "1"}):
        assert _C_engine.compile.compile_policy() == (64, 1, 1)
    assert _C_engine.compile.compile_policy() == (64, 1, 0)


# ── Python's per-module cache ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,value", [(_LEAD, "0"), (_LEAD, "-1"), (_VJP, "0"), (_VJP_REQUIRE, "1")]
)
def test_the_call_signature_follows_the_policy(name: str, value: str) -> None:
    model = nn.Linear(4, 2)
    x = lucid.ones(3, 4)
    base = signature_of(model, (x,), {}, dynamic=False)
    assert base == signature_of(model, (x,), {}, dynamic=False)
    with mock.patch.dict(os.environ, {name: value}):
        assert signature_of(model, (x,), {}, dynamic=False) != base
    assert signature_of(model, (x,), {}, dynamic=False) == base


@needs_metal
@pytest.mark.usefixtures("inference")
def test_one_compiled_function_recompiles_when_the_gate_flips_and_not_when_it_returns() -> None:
    compiled = lucid.compile(_fn)
    x = _input()
    with mock.patch.dict(os.environ, {_LEAD: "0"}):
        want = compiled(x)
    assert len(compiled._cache) == 1
    with mock.patch.dict(os.environ, {_LEAD: "-1"}):
        off = compiled(x)
    assert len(compiled._cache) == 2
    with mock.patch.dict(os.environ, {_LEAD: "0"}):
        again = compiled(x)
    assert len(compiled._cache) == 2
    assert off.tolist() == want.tolist() == again.tolist()


# ── the session cache ─────────────────────────────────────────────────────────────────


@needs_metal
@pytest.mark.usefixtures("inference")
def test_flipping_the_gate_between_two_compiles_builds_two_executables() -> None:
    x = _input()
    with mock.patch.dict(os.environ, {_LEAD: "0"}):
        always = _run(x)
    assert _C_engine.compile.session_cache_size() == 1
    with mock.patch.dict(os.environ, {_LEAD: "-1"}):
        off = _run(x)
    assert _C_engine.compile.session_cache_size() == 2
    with mock.patch.dict(os.environ, {_LEAD: "0"}):
        _run(x)
    with mock.patch.dict(os.environ, {_LEAD: "-1"}):
        _run(x)
    assert _C_engine.compile.session_cache_size() == 2
    assert always.tolist() == off.tolist()


@needs_metal
@pytest.mark.usefixtures("inference")
def test_the_gate_value_not_only_its_sign_is_part_of_the_key() -> None:
    """Lead 96 sits below 128 and above 64: the default and 128 gate it differently."""
    x = _input()
    for value in ("64", "128", "64"):
        with mock.patch.dict(os.environ, {_LEAD: value}):
            _run(x)
    assert _C_engine.compile.session_cache_size() == 2


@needs_metal
@pytest.mark.usefixtures("inference")
def test_a_manual_vjp_switch_flips_the_key() -> None:
    x = _input()
    _run(x)
    with mock.patch.dict(os.environ, {_VJP: "0"}):
        _run(x)
    assert _C_engine.compile.session_cache_size() == 2


# ── the disk cache ────────────────────────────────────────────────────────────────────

_DISK_PROBE = textwrap.dedent(
    """
    import os
    import sys

    import lucid
    from lucid._C import engine as _C_engine


    def main(cache_dir: str) -> None:
        x = lucid.arange(96 * 8 * 2 * 4 * 4, dtype=lucid.float32).reshape(96, 8, 2, 4, 4)
        x = x.to("metal")
        counts = []
        for gate in ("0", "-1", "0"):
            os.environ["LUCID_COMPILE_RANK_CANON_MIN_LEAD"] = gate
            _C_engine.compile.session_cache_clear()  # make the disk file the only way back
            with lucid.no_grad():
                lucid.compile(lambda t: t * 2.0 + 1.0)(x)
            # One executable is a ``.meta`` and a ``.mpsgraphpackage`` of one stem.
            counts.append(len({name.split(".")[0] for name in os.listdir(cache_dir)}))
        print("COUNTS", *counts)


    if __name__ == "__main__":
        main(sys.argv[1])
    """
)


@needs_metal
def test_the_disk_cache_files_one_executable_per_gate(tmp_path: Path) -> None:
    """``LUCID_COMPILE_DISK_CACHE`` is latched at first use, so the probe is a subprocess.

    pytest's temp directory is also long enough (118 characters) to have cut every file
    name to one shared prefix when the path was formatted into a 128-byte buffer."""
    script = tmp_path / "probe.py"
    script.write_text(_DISK_PROBE)
    cache_dir = tmp_path / "disk"
    cache_dir.mkdir()
    env = {
        **os.environ,
        "LUCID_COMPILE_DISK_CACHE": "1",
        "LUCID_COMPILE_DISK_CACHE_DIR": str(cache_dir),
    }
    env.pop(_LEAD, None)
    done = subprocess.run(
        [sys.executable, "-I", str(script), str(cache_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    line = next(ln for ln in done.stdout.splitlines() if ln.startswith("COUNTS"))
    first, flipped, back = (int(n) for n in line.split()[1:])
    assert first >= 1
    assert flipped > first, "the flipped gate reused the file built under the other value"
    assert back == flipped, "the first gate's file was not found again"
