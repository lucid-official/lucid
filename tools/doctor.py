"""Diagnose a source checkout even when its native engine cannot import.

Run ``.venv/bin/python -m tools.doctor`` from the checkout. ``--json`` emits
the same checks for CI. This command inspects only; it never installs or builds.
The runtime probe lives in a child process so a native crash is reported too.

The engine runs its own pinned MLX (``[tool.lucid.mlx]``, staged per machine
by ``tools/fetch_mlx.py``), never the ``mlx`` Python package: that package is
reported for zero-copy interop only, and its absence is not an error.
"""

import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from typing import TypedDict

from tools import check_engine_link, fetch_mlx


class Check(TypedDict):
    name: str
    status: str
    detail: str


ROOT = Path(__file__).resolve().parents[1]
REBUILD = "uv pip install --python .venv/bin/python -e . --no-build-isolation"
RESTAGE = "python3 tools/fetch_mlx.py"


def _version(text: str) -> tuple[int, ...]:
    """The leading numeric fields of a version string."""
    fields = re.match(r"[0-9]+(?:\.[0-9]+)*", text)
    return tuple(int(f) for f in fields.group(0).split(".")) if fields else ()


def mlx_interop_floor(root: Path = ROOT) -> str:
    """The oldest ``mlx`` package the ``[mlx]`` extra admits, e.g. ``"0.32"``."""
    with (root / "pyproject.toml").open("rb") as fh:
        extra = tomllib.load(fh)["project"]["optional-dependencies"]["mlx"]
    match = re.search(r">=\s*([0-9][0-9.]*)", extra[0])
    if match is None:
        raise ValueError(f"the [mlx] extra {extra[0]!r} names no >= floor")
    return match.group(1)


def source_abi(root: Path) -> tuple[int, int]:
    """Read both contracts without importing the package they protect."""
    tree = ast.parse((root / "lucid/version.py").read_text())
    expected = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_EXPECTED_ABI"
        and node.value is not None
    )
    header = (root / "lucid/_C/version.h").read_text()
    match = re.search(r"^#define\s+LUCID_ABI_VERSION\s+(\d+)\s*$", header, re.MULTILINE)
    if match is None or not isinstance(expected, int):
        raise ValueError("cannot read the Python and native ABI declarations")
    return expected, int(match[1])


def _check(name: str, ok: bool, detail: str) -> Check:
    return {"name": name, "status": "ok" if ok else "error", "detail": detail}


def diagnose(timeout: float = 30.0) -> list[Check]:
    checks: list[Check] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append(_check(name, ok, detail))

    record(
        "python",
        sys.version_info[:2] == (3, 14),
        f"{platform.python_version()} at {sys.executable}; requires Python 3.14",
    )
    try:
        pin: fetch_mlx.Pin | None = fetch_mlx.read_pin(ROOT / "pyproject.toml")
    except (OSError, tomllib.TOMLDecodeError, fetch_mlx.FetchError) as exc:
        pin = None
        record("mlx_pin", False, str(exc))
    system, machine = platform.system(), platform.machine()
    release = platform.mac_ver()[0]
    record(
        "platform",
        system == "Darwin"
        and machine == "arm64"
        and bool(release)
        and (pin is None or _version(release) >= _version(pin.macos)),
        f"{system} {release} {machine}; requires macOS {pin.macos if pin else '?'}+ arm64 "
        "(the minos of the MLX build the engine ships)",
    )
    venv = ROOT / ".venv"
    record(
        "environment",
        not venv.is_dir() or Path(sys.prefix).resolve() == venv.resolve(),
        f"prefix={sys.prefix}; use {venv / 'bin/python'} for this checkout",
    )
    for name in ("numpy", "safetensors", "setuptools", "wheel", "pybind11"):
        try:
            dist = importlib.metadata.distribution(name)
            record(name, True, f"{dist.version} at {dist.locate_file('')}")
        except importlib.metadata.PackageNotFoundError:
            record(name, False, f"not installed in {sys.executable}")

    search_path = os.pathsep.join((str(Path(sys.executable).parent), os.environ.get("PATH", "")))
    for name in ("cmake", "ninja", "xcrun", "uv"):
        executable = shutil.which(name, path=search_path)
        record(name, executable is not None, executable or "not found on PATH")

    try:
        expected, native = source_abi(ROOT)
        record(
            "source_abi",
            expected == native,
            f"Python expects {expected}; C++ header declares {native}",
        )
    except (OSError, SyntaxError, ValueError, StopIteration) as exc:
        record("source_abi", False, str(exc))

    if pin is not None:
        checks.extend(_vendored_mlx_checks(pin))
    checks.append(_interop_mlx_check())
    checks.extend(_runtime_checks(pin, timeout))
    return checks


def _runtime_checks(pin: fetch_mlx.Pin | None, timeout: float) -> list[Check]:
    """Import the engine in a child process: a native crash cannot take the doctor down."""
    probe = (
        "import json, lucid; from lucid._C import engine as _C_engine; "
        "x = lucid.ones(2, device='cpu'); "
        "assert x.sum().item() == 2; "
        "print(json.dumps({'version': lucid.__version__, "
        "'abi': _C_engine.ABI_VERSION, 'package': lucid.__file__, "
        "'engine': _C_engine.__file__, 'cpu_smoke': True, "
        "'mlx_runtime': _C_engine.mlx_runtime_version()}))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [_check("runtime", False, f"{exc}\nRebuild: {REBUILD}")]
    if result.returncode != 0:
        detail = f"exit={result.returncode}\n{result.stderr.strip()}\nRebuild: {REBUILD}"
        return [_check("runtime", False, detail)]
    checks = [_check("runtime", True, result.stdout.strip())]
    if pin is not None:
        running = json.loads(result.stdout)["mlx_runtime"]
        ok = running == pin.version
        detail = f"the engine runs MLX {running}; pinned {pin.version}"
        checks.append(_check("mlx_runtime", ok, detail if ok else f"{detail}\nRebuild: {REBUILD}"))
    return checks


def _vendored_mlx_checks(pin: fetch_mlx.Pin) -> list[Check]:
    """The pinned MLX's cache entry, and whether the built engine links it soundly."""
    entry = fetch_mlx.pinned_entry(pin)
    staged = (entry / fetch_mlx.STAMP).is_file()
    detail = f"MLX {pin.version} at {entry}"
    checks = [
        _check("mlx_cache", staged, detail if staged else f"{detail}: not staged; run {RESTAGE}")
    ]
    engine = check_engine_link.default_engine()
    if engine is None:
        return checks  # the runtime check reports the missing engine
    try:
        problems = check_engine_link.check(engine, pin.macos)
    # FetchError: a tool missing, failing or timing out; OSError: an unreadable
    # engine or library.  Either is a finding to report, not a doctor crash.
    except (fetch_mlx.FetchError, OSError) as exc:
        problems = [str(exc)]
    if problems:
        detail = "\n".join(problems) + f"\nRestage: {RESTAGE}; or rebuild: {REBUILD}"
    else:
        detail = f"{engine.name} links liblucid_mlx: every MLX symbol exported, minos {pin.macos}"
    checks.append(_check("engine_link", not problems, detail))
    return checks


def _interop_mlx_check() -> Check:
    """The ``mlx`` package: optional, used only for zero-copy DLPack interop."""
    name = "mlx (interop)"
    try:
        floor = mlx_interop_floor()
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
        return _check(name, False, f"cannot read the [mlx] extra: {exc!r}")
    try:
        version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return _check(
            name, True, f"not installed; only DLPack exchange with mlx needs it (mlx>={floor})"
        )
    if _version(version) < _version(floor):
        return {
            "name": name,
            "status": "warn",
            "detail": f"{version} runs beside Lucid, but Metal DLPack exchange needs mlx>={floor}",
        }
    return _check(name, True, f"{version}; the engine never loads it (interop only)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable checks")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="runtime probe timeout in seconds"
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    checks = diagnose(args.timeout)
    ok = all(check["status"] != "error" for check in checks)
    if args.json:
        print(json.dumps({"ok": ok, "checks": checks, "rebuild": REBUILD}, indent=2))
    else:
        for check in checks:
            print(f"[{check['status']}] {check['name']}: {check['detail']}")
        if not ok:
            print(f"\nAfter resolving the checks above, rebuild from {ROOT}:\n  {REBUILD}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
