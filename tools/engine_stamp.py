"""The engine build stamp: which source tree this checkout's engine binary was built from.

A binary carries no record of its sources, and a file's mtime is not one: ``unison`` keeps
times across machines and ``cp -c`` of an ``.so`` sets "now", so an engine that nobody built
from this tree passes an mtime check whenever the ABI number matches (LCD-881).  The build
therefore leaves ``build/.engine_stamp``: one digest of what builds the engine, written by
every path that builds it in place -- ``setup.py`` (``uv pip install -e .``,
``agent_ws.py build``) and a plain ``cmake --build`` alike, through the one CMake target
``lucid_engine_stamp`` that calls this module.  ``EngineTree.digest`` is the only place that
says what "the sources" are; readers compare the recorded digest with the tree's current one.

Stdlib only and import-by-path safe: ``setup.py`` and ``tools/agent_ws.py`` load it by path
(``tools/`` is not a package in an sdist, and ``agent_ws.py`` runs as a script), so it
imports nothing from ``tools``.

    python3 tools/engine_stamp.py write [--engine-dir DIR]   stamp this tree (a build calls it)
    python3 tools/engine_stamp.py check                      0 when the stamp matches the tree
"""

import argparse
import enum
import hashlib
import json
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

ENGINE: Final = Path("lucid/_C")
# What builds the engine besides lucid/_C; pyproject.toml's [tool.lucid.mlx] pins the MLX.
INPUTS: Final = ("setup.py", "tools/fetch_mlx.py")
# The files under lucid/_C that make up the binary; its .py stubs and the .so are not sources.
SOURCES: Final = ("*.cpp", "*.h", "*.hpp", "*.mm", "*.def", "*.metal", "*.cmake", "CMakeLists.txt")
STAMP: Final = Path("build/.engine_stamp")


def mlx_pin(pyproject_text: str) -> dict[str, object]:
    """``[tool.lucid.mlx]`` of a pyproject.toml's text; raises ``tomllib.TOMLDecodeError``."""
    table = tomllib.loads(pyproject_text).get("tool", {}).get("lucid", {}).get("mlx", {})
    return table if isinstance(table, dict) else {}


class Freshness(enum.Enum):
    MATCH = "the stamp equals the tree's digest"
    MISMATCH = "the stamp names other sources than this tree holds"
    MISSING = "the stamp is not there"


@dataclass(frozen=True)
class EngineTree:
    """One checkout's engine inputs and the stamp that says what its binary was built from."""

    root: Path

    @property
    def engine_dir(self) -> Path:
        return self.root / ENGINE

    @property
    def stamp_path(self) -> Path:
        return self.root / STAMP

    def sources(self) -> list[Path]:
        found = {p for pattern in SOURCES for p in self.engine_dir.rglob(pattern) if p.is_file()}
        return sorted(found | {p for name in INPUTS if (p := self.root / name).is_file()})

    def _pin(self) -> str:
        pyproject = self.root / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8") if pyproject.is_file() else ""
        return json.dumps(mlx_pin(text), sort_keys=True)

    def digest(self) -> str:
        """SHA-256 over the sources' paths and bytes and the MLX pin; ``""`` with no C++."""
        if not any(self.engine_dir.rglob("*.cpp")):
            return ""
        digest = hashlib.sha256()
        for path in self.sources():
            digest.update(f"{path.relative_to(self.root).as_posix()}\0".encode())
            digest.update(path.read_bytes() + b"\0")
        digest.update(self._pin().encode())
        return digest.hexdigest()

    def recorded(self) -> str:
        """What the stamp says, ``""`` when there is none."""
        try:
            return self.stamp_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""

    def freshness(self) -> Freshness:
        recorded, current = self.recorded(), self.digest()
        if not recorded or not current:
            return Freshness.MISSING
        return Freshness.MATCH if recorded == current else Freshness.MISMATCH

    def write(self, engine_dir: Path | None = None) -> bool:
        """Record the tree's digest, for a binary just built into ``engine_dir`` (default here).

        A binary built anywhere else (a wheel's staging directory) vouches for nothing in
        this tree, so it is not stamped.  False when nothing was written.
        """
        built_into = (engine_dir or self.engine_dir).resolve()
        digest = self.digest()
        if built_into != self.engine_dir.resolve() or not digest:
            return False
        self._record(digest)
        return True

    def adopt(self, donor: Self) -> bool:
        """Stamp this tree with ``donor``'s digest, for a binary copied from ``donor``'s build.

        The stamp then says the binary matches ``donor``'s sources, so this tree reads as
        fresh only while its own sources are byte-identical to them.
        """
        digest = donor.digest()
        if digest:
            self._record(digest)
        return bool(digest)

    def _record(self, digest: str) -> None:
        self.stamp_path.parent.mkdir(parents=True, exist_ok=True)
        self.stamp_path.write_text(f"{digest}\n", encoding="utf-8")


# The tree a script belongs to is where it sits, so a worktree's build stamps the worktree.
HERE: Final = EngineTree(Path(__file__).resolve().parents[1])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write", help="stamp this tree after a build into --engine-dir")
    write.add_argument("--engine-dir", type=Path, default=None)
    commands.add_parser("check", help="exit 0 when the stamp matches this tree")
    args = parser.parse_args(argv)
    if args.command == "write":
        stamped = HERE.write(args.engine_dir)
        if stamped:
            print(f"engine stamp: {HERE.digest()[:12]} -> {HERE.stamp_path}")
        else:
            print(f"engine stamp: skipped, that binary is not in {HERE.engine_dir}")
        return 0
    state = HERE.freshness()
    print(f"engine stamp: {state.name.lower()} ({state.value})")
    return 0 if state is Freshness.MATCH else 1


if __name__ == "__main__":
    sys.exit(main())
