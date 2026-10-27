"""A stalled checkpoint fetch must release its owned temporary file."""

import hashlib
import io

import pytest

from lucid.weights import _hub


def test_download_timeout_is_per_read_and_successful_cache_is_reused(monkeypatch, tmp_path) -> None:
    content = b"verified checkpoint"
    calls = []
    monkeypatch.setenv("LUCID_HOME", str(tmp_path))

    def fetch(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(content)

    monkeypatch.setattr(_hub.urllib.request, "urlopen", fetch)
    url = "https://example.invalid/model.safetensors"
    digest = hashlib.sha256(content).hexdigest()
    path = _hub.download(url, digest, name="test-model")
    assert path.read_bytes() == content
    assert calls == [(url, 60)]
    assert _hub.download(url, digest, name="test-model") == path
    assert len(calls) == 1


def test_timed_out_read_removes_only_the_partial_download(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LUCID_HOME", str(tmp_path))
    unrelated = tmp_path / "keep.safetensors"
    unrelated.write_bytes(b"existing data")

    class Stalled(io.BytesIO):
        def read(self, size=-1):
            raise TimeoutError("stalled read")

    monkeypatch.setattr(_hub.urllib.request, "urlopen", lambda url, *, timeout: Stalled())
    with pytest.raises(TimeoutError, match="stalled read"):
        _hub.download("https://example.invalid/model.safetensors", "0" * 64, name="test-model")
    assert unrelated.read_bytes() == b"existing data"
    assert not list((tmp_path / "weights").rglob("*.tmp"))
    assert not list((tmp_path / "weights").rglob("*.safetensors"))


@pytest.mark.parametrize("received", [b"checkpoint", b"CHECKPOINT CONTENT"])
def test_failed_integrity_check_never_publishes_received_bytes(
    monkeypatch, tmp_path, received: bytes
) -> None:
    monkeypatch.setenv("LUCID_HOME", str(tmp_path))
    expected = hashlib.sha256(b"checkpoint content").hexdigest()
    unrelated = tmp_path / "weights" / "other-model" / "model.safetensors"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"keep this existing checkpoint")
    monkeypatch.setattr(
        _hub.urllib.request, "urlopen", lambda url, *, timeout: io.BytesIO(received)
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _hub.download("https://example.invalid/model.safetensors", expected, name="test-model")
    assert list((tmp_path / "weights" / "test-model").rglob("*.safetensors")) == []
    assert list((tmp_path / "weights").rglob("*.tmp")) == []
    assert unrelated.read_bytes() == b"keep this existing checkpoint"
