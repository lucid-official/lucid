import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import types
from pathlib import Path

import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext

try:
    import pybind11
except ImportError as e:
    raise RuntimeError(
        "pybind11 is required to build C++ extensions. Install it first: pip install pybind11"
    ) from e


def _enforce_apple_silicon_only() -> None:
    system = platform.system()
    machine = platform.machine()
    if system != "Darwin" or machine != "arm64":
        raise RuntimeError(
            "Lucid C++ engine supports macOS arm64 Apple Silicon only "
            f"(got system={system!r}, machine={machine!r})"
        )
    os.environ["ARCHFLAGS"] = "-arch arm64"


def _cmake_executable() -> str:
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    try:
        import cmake as cmake_pkg
    except ImportError as e:
        raise RuntimeError(
            "CMake is required to build lucid._C.engine. Install it with "
            "`python -m pip install cmake` or use PEP 517 build isolation."
        ) from e
    return str(Path(cmake_pkg.CMAKE_BIN_DIR) / "cmake")


def _maybe_enable_ccache(cmake_args: list[str]) -> None:
    if os.environ.get("LUCID_USE_CCACHE", "1") == "0":
        return
    ccache = shutil.which("ccache")
    if ccache:
        cmake_args.append(f"-DCMAKE_CXX_COMPILER_LAUNCHER={ccache}")


def _build_jobs() -> str:
    jobs_env = os.environ.get("LUCID_BUILD_JOBS")
    if jobs_env:
        return jobs_env
    return str(os.cpu_count() or 1)


def _macos_sdkroot() -> str:
    return subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()


def _load_fetch_mlx() -> types.ModuleType:
    """Import tools/fetch_mlx.py by path: tools/ is not a package in an sdist."""
    path = Path(__file__).resolve().parent / "tools" / "fetch_mlx.py"
    spec = importlib.util.spec_from_file_location("_lucid_fetch_mlx", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _deployment_target(floor: str) -> str:
    """The requested target, raised to the pinned MLX binary's minos if lower."""
    requested = os.environ.get("MACOSX_DEPLOYMENT_TARGET", floor)
    if _version_tuple(requested) >= _version_tuple(floor):
        return requested
    print(
        f"lucid: MACOSX_DEPLOYMENT_TARGET={requested} is below {floor}, the "
        f"minimum of the pinned MLX build ([tool.lucid.mlx] macos); using {floor}",
        file=sys.stderr,
    )
    return floor


# Copied next to engine*.so in a wheel build; the engine's first rpath entry
# is @loader_path, and MLX finds mlx.metallib beside liblucid_mlx.dylib.
_MLX_RUNTIME_FILES = ("liblucid_mlx.dylib", "liblucid_jaccl.dylib", "mlx.metallib")


class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str = "lucid/_C") -> None:
        super().__init__(name, sources=[])
        self.source_dir = str(Path(source_dir).resolve())


# NOTE on wheel platform tag:
#
# Python 3.14 from python.org is itself a `universal2` build, so
# `sysconfig.get_platform()` returns `macosx-10.15-universal2` and
# setuptools would tag our wheel `macosx_26_0_universal2`. That tag
# advertises x86_64 compatibility, but the Lucid engine is arm64-only
# (`_enforce_apple_silicon_only()` aborts on anything else), so the
# universal2 tag is actively wrong.
#
# The canonical fix is to set `_PYTHON_HOST_PLATFORM=macosx-26.0-arm64`
# in the build environment before invoking `python -m build`. This is
# what cibuildwheel and modern release tooling do; the CI workflows
# (`publish.yml`, `release-testpypi.yml`) and `scripts/ci_publish.sh`
# set it explicitly. Adding a `bdist_wheel` cmdclass override is *not*
# enough — PEP 517 build backends route around setup.py's cmdclass
# for `bdist_wheel`, so the override silently fails.
#
# The tag says 26_0 although the engine needs macOS 26.2 ([tool.lucid.mlx]
# macos, the minos of the MLX build the wheel ships): pip treats every macOS
# 11+ release as X.0, so no wheel tag can say 26.2.  `lucid._C` refuses to
# import on 26.0-26.1 with a message naming the floor instead.


class CMakeBuildExt(build_ext):
    _lucid_in_tree = False

    def run(self) -> None:
        # build_ext.run() clears ``inplace`` while extensions build and copies
        # them into the tree afterwards, so record the caller's intent here.
        # Editable / in-place builds rely on the absolute rpath into the staged
        # MLX (shared per machine) instead of copying ~210 MB into the tree.
        self._lucid_in_tree = bool(self.inplace or self.editable_mode)
        super().run()

    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        _enforce_apple_silicon_only()

        cmake = _cmake_executable()
        fetch_mlx = _load_fetch_mlx()
        mlx_pin = fetch_mlx.read_pin()
        mlx_dir = fetch_mlx.stage(mlx_pin)
        mlx_include_dir, mlx_library_dir = mlx_dir / "include", mlx_dir / "lib"
        ext_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        ext_dir = ext_path.parent
        build_temp = Path(self.build_temp).resolve() / ext.name.replace(".", "_")
        build_temp.mkdir(parents=True, exist_ok=True)
        ext_dir.mkdir(parents=True, exist_ok=True)

        build_mode = os.environ.get("LUCID_BUILD_MODE", "release").lower()
        sanitize = os.environ.get("LUCID_SANITIZE", "").strip()
        # The floor is the pinned MLX binary's: one wheel for that macOS and later.
        deployment_target = _deployment_target(mlx_pin.macos)
        cmake_args = [
            "-S",
            ext.source_dir,
            "-B",
            str(build_temp),
            "-G",
            "Ninja",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_OSX_ARCHITECTURES=arm64",
            f"-DCMAKE_OSX_SYSROOT={_macos_sdkroot()}",
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}",
            f"-DLUCID_BUILD_MODE={build_mode}",
            f"-DLUCID_SANITIZE={sanitize}",
            f"-DLUCID_COVERAGE={os.environ.get('LUCID_COVERAGE', 'OFF')}",
            f"-DLUCID_PYTHON_INCLUDE_DIR={sysconfig.get_paths()['include']}",
            f"-DLUCID_PYBIND11_INCLUDE_DIR={pybind11.get_include()}",
            f"-DLUCID_MLX_INCLUDE_DIR={mlx_include_dir}",
            f"-DLUCID_MLX_LIBRARY_DIR={mlx_library_dir}",
            f"-DLUCID_MLX_PIN={mlx_pin.version}",
            f"-DLUCID_MLX_BUNDLED={'OFF' if self._lucid_in_tree else 'ON'}",
            f"-DLUCID_PYTHON_EXTENSION_SUFFIX={sysconfig.get_config_var('EXT_SUFFIX')}",
            f"-DLUCID_EXTENSION_OUTPUT_DIR={ext_dir}",
        ]
        _maybe_enable_ccache(cmake_args)

        subprocess.check_call([cmake, *cmake_args])
        subprocess.check_call([cmake, "--build", str(build_temp), "--parallel", _build_jobs()])
        if not self._lucid_in_tree:
            for name in _MLX_RUNTIME_FILES:
                shutil.copyfile(mlx_library_dir / name, ext_dir / name)


_enforce_apple_silicon_only()

# All distribution metadata (name, version, deps, classifiers, packages,
# extras_require) lives in pyproject.toml — this file is the *build*
# entrypoint only (CMake-driven C++ extension + Apple-Silicon guard).
# The version is sourced statically from pyproject.toml's [project.version]
# (PEP 621); the legacy {{VERSION_PLACEHOLDER}} sed scheme was removed
# alongside the publish.yml rewrite.
setuptools.setup(
    ext_modules=[CMakeExtension("lucid._C.engine")], cmdclass={"build_ext": CMakeBuildExt}
)
