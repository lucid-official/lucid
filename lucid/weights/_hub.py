"""Checkpoint download with SHA-256 verification + local cache.

A thin, provider-agnostic fetch layer: callers supply ``url + sha256``
(from a :class:`~lucid.weights._base.WeightEntry`) and get back a local
path to the verified file.  Lucid hosts weights on the Hugging Face
Hub, but nothing here knows that — any HTTPS URL works.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def _cache_dir() -> Path:
    r"""Resolve the cache root, honouring ``LUCID_HOME`` if set.

    Returns
    -------
    pathlib.Path
        ``$LUCID_HOME/weights`` when the environment variable is set,
        ``~/.cache/lucid/weights`` otherwise.  Not created here —
        :func:`download` makes it lazily.
    """
    base = os.environ.get("LUCID_HOME")
    root = Path(base) if base is not None else Path.home() / ".cache" / "lucid"
    return root / "weights"


def _verify_sha256(path: Path, expected: str) -> None:
    r"""Raise :class:`RuntimeError` unless ``path`` hashes to ``expected``.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.
    expected : str
        Hex-encoded expected SHA-256 digest.

    Raises
    ------
    RuntimeError
        On hash mismatch — the message reports both digests so callers
        can diagnose corruption or a supply-chain issue.

    Notes
    -----
    Streams the file in 1 MiB chunks so multi-GB checkpoints never load
    fully into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise RuntimeError(f"SHA256 mismatch for {path.name}: expected {expected}, got {got}")


def _filename_from_url(url: str) -> str:
    """Derive the on-disk filename from a URL, defaulting sensibly."""
    name = Path(urlparse(url).path).name
    return name or "model.safetensors"


def download(url: str, sha256: str, *, name: str) -> Path:
    r"""Download ``url`` to the local cache, verify SHA-256, return the path.

    Idempotent — when the file already exists in cache and verifies, it
    is returned without re-fetching.  A cached file that *fails*
    verification is treated as corrupt and re-downloaded.

    Parameters
    ----------
    url : str
        Remote location of the checkpoint blob.
    sha256 : str
        Hex-encoded expected SHA-256 digest.  Validated after download
        and on every cache hit.
    name : str, keyword-only
        Short identifier (typically ``"<model>/<tag>"``) used to compose
        the cache subdirectory so different checkpoints never collide.

    Returns
    -------
    pathlib.Path
        Local path of the verified checkpoint.

    Raises
    ------
    RuntimeError
        If the freshly downloaded file fails SHA-256 verification.
    OSError / urllib.error.URLError
        From ``urllib.request.urlopen`` when the network is unreachable
        or the URL is invalid.
    TimeoutError
        If connection or an individual blocking read makes no progress for
        60 seconds. This is not a total duration limit for large downloads.

    Notes
    -----
    Cache layout::

        {cache_root}/{name}/{sha256[:8]}/{url-basename}

    Download is atomic: bytes go to a temp file in the cache directory,
    the SHA is verified, then the temp file is renamed into place.  On
    failure the temp file is removed; the cache directory is kept for
    the next attempt.
    """
    safe_name = name.replace("/", os.sep)
    cache_root = _cache_dir() / safe_name / sha256[:8]
    cache_root.mkdir(parents=True, exist_ok=True)
    final = cache_root / _filename_from_url(url)

    if final.exists():
        try:
            _verify_sha256(final, sha256)
            return final
        except RuntimeError:
            final.unlink()

    fd, tmp_name = tempfile.mkstemp(dir=cache_root, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        _verify_sha256(tmp_path, sha256)
        tmp_path.replace(final)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return final
