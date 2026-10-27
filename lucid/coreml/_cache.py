"""Compile a package once, and open that compiled model every time after.

Core ML runs a compiled model (``.mlmodelc``), and the first time one is
loaded it is specialised into a bundle under the user's caches —
``~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache`` — which for a GPU
segment holds a full copy of the weights: 1.1 GB for YOLO v1.  That bundle
is keyed by the compiled model's path *and* its file identity.  Measured on
macOS 27: opening the same ``.mlmodelc`` again, from the same process or
another, adds nothing; a copy of it adds a new bundle, even one put back at
the very same path.

Compiling on every load therefore wrote a bundle on every load that nothing
would read again.  A development machine's cache reached 121 GB, and the
test suite pushed the disk past macOS's near-low mark, where the purge that
followed aborted a process from inside Core ML (CHA-22, see
``lucid/test/unit/coreml/conftest.py``).

So the compiled model is kept — one per package *content* — and every load
of that content opens the same directory.

Key
    SHA-256 over every file in the package (relative path, size, bytes),
    after the Core ML build and the OS build — a system update compiles
    afresh instead of reusing an older compiler's output.  Content, not
    path: the same package opened from anywhere, or copied elsewhere, is one
    entry, and a package rewritten in place with new weights is a new one.
    So is every *export*, even of an unchanged model: the package's
    ``Manifest.json`` carries identifiers drawn fresh for each one.
Location
    ``LUCID_COREML_CACHE_DIR`` when set; else ``$LUCID_HOME/coreml``, beside
    the weights cache that variable already moves; else
    ``~/.cache/lucid/coreml``.  Entries live in a ``v<format>`` directory
    under it, and only names this module writes there — a 64-digit hex key
    with ``.mlmodelc``, ``.json`` or ``.lock``, or a dot-prefixed scratch
    name with a 32-digit hex suffix — are ever read, swept or removed.  The
    variable can name a directory that holds anything else without that
    being at risk.
Bound
    ``LUCID_COREML_CACHE_LIMIT`` bytes (``K``/``M``/``G``/``T`` suffixes
    accepted), 4 GiB by default — three models the size of YOLO v1.
    Inserting past it removes the least recently used entries, except the
    one just made and any entry a live handle holds, in this process or
    another, so a single model larger than the limit is still compiled once
    and reused while it is open.  ``0`` turns the cache off: each handle
    compiles privately and removes its copy when it closes — which, as
    before, leaves Core ML one bundle per load.
    :func:`lucid.coreml.empty_cache` removes every entry no handle holds.
Concurrency
    A compile is moved into a staging directory inside the cache and
    renamed onto its entry.  ``rename`` is atomic and refuses a non-empty
    target, so of two processes that compile the same package at once one
    wins and the other discards its copy and opens the winner's; two
    threads of one process do not both compile.  Every handle holds a
    shared ``flock`` on its entry's lock file for as long as its model
    lives — the engine keeps a duplicate of the descriptor and closes it
    with the model, so a prediction still running when the handle is closed
    keeps it too.  Eviction asks for an exclusive lock without waiting and
    passes over what it cannot have.  The kernel drops a dead process's
    locks, so a crash leaves nothing held.  A volume that cannot lock files
    — some network shares — cannot keep a cache safely, and is treated like
    one that cannot be written.
Disk
    Core ML does not fail cleanly when the disk fills: it ends the process,
    ``LLVM ERROR: IO failure on output stream: No space left on device``.
    Before compiling or loading, each volume the step writes to is checked
    for what it will write — the compiled model where Core ML compiles it
    and where it is kept, a bundle the size of the package where Core ML
    keeps those — and a shortfall raises :class:`OSError` (``ENOSPC``).

A path that is already a ``.mlmodelc`` is opened where it is: there is
nothing to compile, and opening it in place is what lets its bundle be
reused.

Not bounded here: the bundle Core ML made from an entry that is later
evicted stays in Core ML's cache until macOS purges it.  Its directory names
cannot be traced back to the entry, and Core ML's cache is not Lucid's to
prune.  Reopening a package — the case that grew without bound — writes
none.
"""

import errno
import fcntl
import functools
import hashlib
import json
import os
import plistlib
import re
import shutil
import tempfile
import threading
import time
import uuid
import warnings

from lucid._C import engine as _C_engine

__all__ = ["Lease", "empty_cache", "open_compiled"]

#: Bumped when the layout of an entry or its key changes.  Entries live in
#: ``v<_FORMAT>`` under the root, so another layout is never read as this
#: one; it is left where it is.
_FORMAT = 1

#: The bound when ``LUCID_COREML_CACHE_LIMIT`` is unset.
_DEFAULT_LIMIT = 4 << 30

#: Kept free on every volume beyond what a step is estimated to write.  The
#: estimate is the package's size; what Core ML writes for it is near that
#: rather than exactly it.
_HEADROOM = 256 << 20

#: A staging or trash directory, or a lock with no entry, older than this
#: belongs to a process that died part way; a younger one may be in use.
_ABANDONED_AFTER = 3600.0

_SUFFIX = ".mlmodelc"

#: The only names this module writes in its store, and so the only ones it
#: reads, sweeps or removes.
_KEYED = re.compile(r"(?P<key>[0-9a-f]{64})\.(?P<kind>mlmodelc|json|lock)")
_SCRATCH = re.compile(r"\.(?:staging|trash|note)-[0-9a-f]{32}")

#: Errors a volume answers ``flock`` with when it does not do locks.
_NO_LOCKS = frozenset({errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOLCK})

#: ``fcntl.flock``, through a name tests can replace.
_flock = fcntl.flock

#: Guards the collections below.
_lock = threading.Lock()
#: One eviction at a time in this process.
_evicting = threading.Lock()
#: One lock per key, so two threads loading the same new package compile
#: it once rather than racing to the rename.
_compiling: dict[str, threading.Lock] = {}
#: ``realpath -> (stat signature, key, bytes)``, so a package opened again
#: unchanged in the same process is not read again.
_known: dict[str, tuple[tuple[tuple[str, int, int, int, int], ...], str, int]] = {}
#: Stores this process found it can write and lock in.
_lockable: set[str] = set()
#: Stores this process already warned it could not use.
_unusable: set[str] = set()


class Lease:
    """What a handle opens, held for as long as the handle needs it.

    ``path`` is the cached ``.mlmodelc`` to open, and ``lock_fd`` the shared
    lock that keeps eviction away from it; the engine duplicates that into
    the model it loads, so releasing the lease ends this side's hold and the
    model's own ends when the model goes.  Otherwise ``path`` is a package
    for the engine to compile privately — it removes its copy when the last
    use of the handle ends — or a ``.mlmodelc`` the caller named, opened in
    place, and ``lock_fd`` is ``-1``.  Safe to release twice.
    """

    __slots__ = ("_fd", "path")

    def __init__(self, path: str, fd: int | None) -> None:
        self.path = path
        self._fd = fd

    @property
    def held(self) -> bool:
        """Whether this lease still keeps a cached compiled model in place."""
        return self._fd is not None

    @property
    def lock_fd(self) -> int:
        """The descriptor holding the entry's shared lock, or ``-1``."""
        return -1 if self._fd is None else self._fd

    def release(self) -> None:
        """Let the compiled model go."""
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)


def cache_dir() -> str:
    """Where compiled models are kept, whether or not it exists yet.

    Returns
    -------
    str
        ``LUCID_COREML_CACHE_DIR`` when set; else ``$LUCID_HOME/coreml``,
        beside the weights cache that variable already moves; else
        ``~/.cache/lucid/coreml``. Read on every call, so a change to the
        environment takes effect at the next load. The entries themselves
        are in :func:`store_dir` beneath it.
    """
    chosen = os.environ.get("LUCID_COREML_CACHE_DIR")
    if chosen:
        return os.path.abspath(os.path.expanduser(chosen))
    home = os.environ.get("LUCID_HOME")
    if home is not None:
        return os.path.abspath(os.path.join(os.path.expanduser(home), "coreml"))
    return os.path.join(os.path.expanduser("~"), ".cache", "lucid", "coreml")


def store_dir(root: str | None = None) -> str:
    """The directory this layout keeps its entries in, under ``root``.

    Parameters
    ----------
    root : str, optional
        The cache root; :func:`cache_dir` when omitted.

    Returns
    -------
    str
        ``<root>/v<format>``. Everything this module writes is in here, and
        it touches nothing in here it did not name.
    """
    return os.path.join(cache_dir() if root is None else root, f"v{_FORMAT}")


def cache_limit() -> int:
    """The bound the cache keeps to, in bytes.

    Returns
    -------
    int
        ``LUCID_COREML_CACHE_LIMIT`` read as a size — bytes, or a number
        with ``K``/``M``/``G``/``T`` (binary; ``B`` or ``iB`` optional) —
        or 4 GiB when it is unset. ``0`` means the cache is off.

    Raises
    ------
    ValueError
        The variable is set to something that is not a size.
    """
    raw = os.environ.get("LUCID_COREML_CACHE_LIMIT", "").strip()
    if not raw:
        return _DEFAULT_LIMIT
    text = raw.upper().removesuffix("B").removesuffix("I")
    scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}.get(text[-1:], 1)
    try:
        value = float(text[:-1] if scale > 1 else text)
    except ValueError:
        value = -1.0
    if not 0 <= value < float("inf"):
        raise ValueError(
            f"lucid.coreml: LUCID_COREML_CACHE_LIMIT={raw!r} is not a size — give "
            f"bytes, optionally with K, M, G or T (4G), or 0 to turn the cache off"
        )
    return int(value * scale)


def _core_ml_home() -> str:
    """The home Core ML keeps its bundles under for this process."""
    return os.environ.get("CFFIXED_USER_HOME") or os.path.expanduser("~")


@functools.cache
def _fingerprint() -> str:
    """What produced a compiled model, beyond the package it came from."""

    def read(path: str, key: str) -> str:
        try:
            with open(path, "rb") as handle:
                return str(plistlib.load(handle).get(key, "unknown"))
        except (OSError, ValueError):
            return "unknown"

    coreml = read(
        "/System/Library/Frameworks/CoreML.framework/Resources/version.plist",
        "CFBundleVersion",
    )
    system = read("/System/Library/CoreServices/SystemVersion.plist", "ProductBuildVersion")
    return f"lucid-coreml-cache/{_FORMAT};coreml={coreml};os={system}"


def _is_compiled(path: str) -> bool:
    return path.rstrip("/").endswith(_SUFFIX)


def _files(package: str) -> list[tuple[str, str]]:
    """Every file in the package as ``(relative path, absolute path)``."""
    if os.path.isfile(package):
        return [(os.path.basename(package), package)]
    found: list[tuple[str, str]] = []
    for here, dirs, names in os.walk(package):
        dirs.sort()
        for name in sorted(names):
            full = os.path.join(here, name)
            found.append((os.path.relpath(full, package), full))
    return found


def _package_bytes(package: str) -> int:
    return sum(os.stat(full).st_size for _relative, full in _files(package))


def _identify(package: str) -> tuple[str, int]:
    """The package's cache key and its size in bytes.

    Read in full unless every file has the size, mtime, ctime and inode it
    had when this process last read it. ctime is there because the other
    three can all be put back after a rewrite — ``cp -p``, ``touch -r``,
    ``rsync --inplace -t`` — and the kernel sets ctime on every change.
    """
    files = _files(package)
    stamp = []
    for relative, full in files:
        info = os.stat(full)
        stamp.append((relative, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino))
    signature = tuple(stamp)
    real = os.path.realpath(package)
    with _lock:
        known = _known.get(real)
    if known is not None and known[0] == signature:
        return known[1], known[2]

    digest = hashlib.sha256(_fingerprint().encode())
    size = 0
    chunk = bytearray(8 << 20)
    view = memoryview(chunk)
    for (relative, full), (_r, length, _m, _c, _i) in zip(files, signature, strict=True):
        size += length
        digest.update(f"\0{relative}\0{length}\0".encode())
        with open(full, "rb") as handle:
            while read := handle.readinto(chunk):
                digest.update(view[:read])
    key = digest.hexdigest()
    with _lock:
        _known[real] = (signature, key, size)
    return key, size


def _free_bytes(path: str) -> int:
    """Free space on the volume holding ``path``; replaced in tests."""
    return shutil.disk_usage(path).free


def _existing(path: str) -> str:
    """``path``, or the nearest ancestor of it that exists."""
    path = os.path.abspath(path)
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    return path


def _human(count: int) -> str:
    if count < 1 << 20:
        return f"{count / (1 << 10):.1f} KB"
    if count < 1 << 30:
        return f"{count / (1 << 20):.1f} MB"
    return f"{count / (1 << 30):.2f} GB"


def _require_room(package: str, size: int, *, compile_into: str | None) -> None:
    """Raise ``ENOSPC`` unless every volume the load writes to has room.

    ``compile_into`` is where the compiled model will be kept, or ``None``
    when nothing is to be compiled and only Core ML's bundle may still be
    written.
    """
    writes: list[tuple[str, str]] = []
    if compile_into is not None:
        scratch = tempfile.gettempdir()
        writes.append((scratch, "the compiled model"))
        if os.stat(_existing(compile_into)).st_dev != os.stat(_existing(scratch)).st_dev:
            writes.append((compile_into, "the compiled model's kept copy"))
    writes.append(
        (
            os.path.join(_core_ml_home(), "Library", "Caches"),
            "the bundle Core ML specialises it into",
        )
    )

    volumes: dict[int, tuple[str, int, list[str]]] = {}
    for where, what in writes:
        anchor = _existing(where)
        device = os.stat(anchor).st_dev
        first, total, whats = volumes.get(device, (anchor, 0, []))
        volumes[device] = (first, total + size, [*whats, what])

    short = []
    for anchor, needed, whats in volumes.values():
        free = _free_bytes(anchor)
        if free < needed + _HEADROOM:
            short.append(
                f"{_human(needed + _HEADROOM)} free on the volume holding {anchor} "
                f"({' and '.join(whats)}, plus {_human(_HEADROOM)} to spare) and "
                f"{_human(free)} is"
            )
    if short:
        raise OSError(
            errno.ENOSPC,
            f"lucid.coreml: loading {package} ({_human(size)}) needs "
            + "; it also needs ".join(short)
            + ". Core ML does not fail cleanly when the disk fills — it ends "
            "the process — so the load stops here instead. Free some space, "
            "or point LUCID_COREML_CACHE_DIR at a volume that has it.",
        )


def _hold(lock_path: str, *, exclusive: bool) -> int | None:
    """An fd holding ``lock_path`` shared, or exclusive if free right now.

    ``None`` when an exclusive lock was asked for and someone holds it.  An
    eviction can unlink the lock file while this waits on it, and a lock on
    an unlinked file guards nothing, so the inode is checked and the lock
    taken again on the file that is there now.

    Raises
    ------
    OSError
        From ``flock`` itself — on a volume that does not lock files, an
        errno in ``_NO_LOCKS``.
    """
    operation = (fcntl.LOCK_EX | fcntl.LOCK_NB) if exclusive else fcntl.LOCK_SH
    while True:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _flock(fd, operation)
        except BlockingIOError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise
        try:
            current = os.stat(lock_path).st_ino
        except FileNotFoundError:
            current = -1
        if current == os.fstat(fd).st_ino:
            return fd
        os.close(fd)


def _is_whole(entry: str) -> bool:
    return os.path.isfile(os.path.join(entry, "coremldata.bin"))


def _tree_bytes(path: str) -> int:
    total = 0
    for here, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.lstat(os.path.join(here, name)).st_size
            except OSError:
                # Concurrent eviction may unlink an entry while walking its tree.
                pass
    return total


def _unlink(*paths: str) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            # Best-effort cleanup: path may already have been unlinked.
            pass


def _scratch(store: str, kind: str) -> str:
    return os.path.join(store, f".{kind}-{uuid.uuid4().hex}")


def _write_note(store: str, key: str, package: str) -> None:
    """Record the entry's size and source; its mtime is when it was last used."""
    size = _tree_bytes(os.path.join(store, f"{key}{_SUFFIX}"))
    scratch = _scratch(store, "note")
    with open(scratch, "w") as handle:
        json.dump({"bytes": size, "package": package, "created": time.time()}, handle)
    os.replace(scratch, os.path.join(store, f"{key}.json"))


def _sweep(store: str, name: str, now: float) -> None:
    """Remove what a process that died part way left behind.

    ``name`` is one this module writes: a scratch directory or file, or the
    lock or note of a key with no entry.  Anything younger than
    ``_ABANDONED_AFTER`` may still be in use, and is left.
    """
    full = os.path.join(store, name)
    try:
        if now - os.lstat(full).st_mtime <= _ABANDONED_AFTER:
            return
    except OSError:
        return
    if _SCRATCH.fullmatch(name):
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full, ignore_errors=True)
        else:
            _unlink(full)
        return
    keyed = _KEYED.fullmatch(name)
    if keyed is None or keyed["kind"] == "mlmodelc":
        return
    key = keyed["key"]
    try:
        fd = _hold(os.path.join(store, f"{key}.lock"), exclusive=True)
    except OSError:
        return
    if fd is None:
        return
    try:
        if not os.path.exists(os.path.join(store, f"{key}{_SUFFIX}")):
            _unlink(os.path.join(store, f"{key}.json"), os.path.join(store, f"{key}.lock"))
    finally:
        os.close(fd)


def _entries(store: str) -> list[tuple[float, int, str]]:
    """Every entry as ``(last used, bytes, key)``, sweeping up as it goes."""
    found: list[tuple[float, int, str]] = []
    try:
        names = os.listdir(store)
    except FileNotFoundError:
        return found
    present = set(names)
    now = time.time()
    for name in names:
        if _SCRATCH.fullmatch(name):
            _sweep(store, name, now)
            continue
        keyed = _KEYED.fullmatch(name)
        if keyed is None:
            continue  # not this module's: never read, never removed
        key = keyed["key"]
        if keyed["kind"] != "mlmodelc":
            if f"{key}{_SUFFIX}" not in present:
                _sweep(store, name, now)
            continue
        full = os.path.join(store, name)
        if not os.path.isdir(full) or os.path.islink(full):
            continue
        note = os.path.join(store, f"{key}.json")
        try:
            used = os.stat(note).st_mtime
            with open(note) as handle:
                size = int(json.load(handle)["bytes"])
        except (OSError, ValueError, KeyError, TypeError):
            try:
                used = os.stat(full).st_mtime
            except OSError:
                continue
            size = _tree_bytes(full)
        found.append((used, size, key))
    return found


def _remove(store: str, key: str) -> bool:
    """Remove one entry unless a handle holds it; whether it went."""
    lock_path = os.path.join(store, f"{key}.lock")
    try:
        fd = _hold(lock_path, exclusive=True)
    except OSError:
        return False
    if fd is None:
        return False
    trash = _scratch(store, "trash")
    try:
        try:
            os.rename(os.path.join(store, f"{key}{_SUFFIX}"), trash)
        except FileNotFoundError:
            trash = ""
        _unlink(os.path.join(store, f"{key}.json"), lock_path)
    finally:
        os.close(fd)
    if trash:
        shutil.rmtree(trash, ignore_errors=True)
    return True


def _evict(store: str, limit: int, keep: str | None) -> int:
    """Remove least recently used entries until the cache fits ``limit``."""
    with _evicting:
        entries = sorted(_entries(store))
        total = sum(size for _used, size, _key in entries)
        freed = 0
        for _used, size, key in entries:
            if total <= limit:
                break
            if key == keep or not _remove(store, key):
                continue
            total -= size
            freed += size
        return freed


def _unkept(package: str, size: int, *, compile: bool) -> Lease:
    """A lease on nothing in the cache: ``package`` as the engine opens it.

    A package is compiled privately by the engine; a ``.mlmodelc`` is opened
    where it is.
    """
    if size:
        _require_room(package, size, compile_into=tempfile.gettempdir() if compile else None)
    return Lease(package, None)


def _cannot_keep(store: str, why: str, stacklevel: int) -> bool:
    with _lock:
        first = store not in _unusable
        _unusable.add(store)
    if first:
        warnings.warn(
            f"lucid.coreml: cannot keep compiled models in {store} ({why}); "
            f"compiling on every load instead, which leaves Core ML a new "
            f"cache bundle each time. Set LUCID_COREML_CACHE_DIR to a "
            f"writable directory on a local volume.",
            RuntimeWarning,
            stacklevel=stacklevel + 1,
        )
    return False


def _usable(store: str) -> bool:
    """Whether ``store`` can be written and locked in; warns once if not.

    The lock probe is made once per store per process; the directory is
    made on every call, since anyone may have removed it since.
    """
    try:
        os.makedirs(store, exist_ok=True)
        with _lock:
            if store in _lockable:
                return True
        if not os.access(store, os.W_OK | os.X_OK):
            raise PermissionError(errno.EACCES, "not writable")
        probe = _scratch(store, "note")
        fd = os.open(probe, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        finally:
            os.close(fd)
            _unlink(probe)
    except OSError as exc:
        why = "its volume does not support file locks" if exc.errno in _NO_LOCKS else str(exc)
        return _cannot_keep(store, why, stacklevel=5)
    with _lock:
        _lockable.add(store)
    return True


def _insert(store: str, key: str, package: str, size: int) -> None:
    """Compile ``package`` onto its entry, unless another process just did."""
    _require_room(package, size, compile_into=store)
    entry = os.path.join(store, f"{key}{_SUFFIX}")
    compiled = _C_engine.coreml.compile_model(package)
    staging = _scratch(store, "staging")
    try:
        shutil.move(compiled, staging)
        if os.path.isdir(entry) and not _is_whole(entry):
            # Never left by this module — a rename is whole or absent — so
            # something else damaged it; clear it rather than open it.
            shutil.rmtree(entry, ignore_errors=True)
        try:
            os.rename(staging, entry)
        except OSError as exc:
            # Another process's copy got there first; open that one.
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(compiled, ignore_errors=True)
    _write_note(store, key, package)


def open_compiled(package: str) -> Lease:
    """The compiled model for ``package``, compiling it only if no one has.

    Parameters
    ----------
    package : str
        The ``.mlpackage`` to open — or a ``.mlmodelc``, which is opened in
        place without the cache.

    Returns
    -------
    Lease
        Holds the cached ``.mlmodelc`` until released — or, with the cache
        off or unusable, names the package for the engine to compile.

    Raises
    ------
    OSError
        ``ENOSPC`` when a volume the load writes to lacks the room.
    RuntimeError
        Core ML could not compile the package.
    """
    if not os.path.exists(package):
        # Core ML's own error names the missing path; nothing to cache.
        return Lease(package, None)
    if _is_compiled(package):
        return _unkept(package, _package_bytes(package), compile=False)
    limit = cache_limit()
    store = store_dir()
    if limit == 0 or not _usable(store):
        return _unkept(package, _package_bytes(package), compile=True)

    key, size = _identify(package)
    entry = os.path.join(store, f"{key}{_SUFFIX}")
    try:
        fd = _hold(os.path.join(store, f"{key}.lock"), exclusive=False)
    except OSError as exc:
        if exc.errno not in _NO_LOCKS:
            raise
        _cannot_keep(store, "its volume does not support file locks", stacklevel=4)
        return _unkept(package, size, compile=True)
    if fd is None:  # a shared lock waits rather than refusing
        raise RuntimeError("lucid.coreml: could not lock the compile cache")
    inserted = False
    try:
        with _lock:
            serial = _compiling.setdefault(key, threading.Lock())
        with serial:
            if _is_whole(entry):
                _require_room(package, size, compile_into=None)
                try:
                    os.utime(os.path.join(store, f"{key}.json"))
                except FileNotFoundError:
                    _write_note(store, key, package)
            else:
                _insert(store, key, package, size)
                inserted = True
    except BaseException:
        os.close(fd)
        raise
    lease = Lease(entry, fd)
    if inserted:
        try:
            _evict(store, limit, keep=key)
        except OSError:
            pass  # best effort: the cache stays over its bound, the load stands
        except BaseException:
            lease.release()
            raise
    return lease


def empty_cache() -> int:
    """Remove every cached compiled model that no handle holds.

    An entry a live handle holds — in this process or another — keeps its
    shared lock, and is passed over. Nothing in the cache directory that
    this module did not write is touched.

    Returns
    -------
    int
        Bytes freed; ``0`` when the cache is empty or does not exist.
    """
    store = store_dir()
    if not os.path.isdir(store):
        return 0
    return _evict(store, 0, keep=None)
