"""Unit tests for ``tools/check_weight_fit.py`` (LCD-894).

The check's verdict must not say more than it read.  ``--family drifting`` met a 401 on
every entry, printed ``0 checkpoints read … OK — every checkpoint fits`` and exited 2.
Here the registry, the factory build and the header reads are replaced, so a case is a
fake entry and a fake network and nothing is downloaded or built.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

_TOOLS_DIR = str(Path(__file__).resolve().parents[3] / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import check_weight_fit as cwf
from lucid.utils.transforms import ImageClassification
from lucid.weights import WeightEntry, WeightsEnum

_HEADER = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
_OTHER_HEADER = {"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]}}
_REAL_DIGEST = "a" * 64


def _entry(tag: str, sha256: str = _REAL_DIGEST) -> WeightEntry:
    return WeightEntry(
        url=f"https://example.invalid/fake/{tag}/model.safetensors",
        sha256=sha256,
        num_classes=10,
        transforms=ImageClassification(crop_size=224, resize_size=256),
    )


class _Fake(WeightsEnum):
    ONE = _entry("ONE")
    TWO = _entry("TWO")


class _Unpublished(WeightsEnum):
    ONE = _entry("ONE")
    DRAFT = _entry("DRAFT", sha256=cwf._PLACEHOLDER_DIGEST)


Network = Callable[[list[str]], dict[str, dict[str, object] | str]]


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    enum: type[WeightsEnum],
    network: Network,
) -> tuple[int, str, str, list[str]]:
    """``main`` over ``enum`` with ``network`` answering; the urls it asked for."""
    asked: list[str] = []

    def read(urls: list[str]) -> dict[str, dict[str, object] | str]:
        asked.extend(urls)
        return network(urls)

    def build(name: str, tag: str) -> dict[str, object]:
        return {"entry": enum[tag].entry, "wanted": {"w": (2,)}}

    monkeypatch.setattr(cwf, "_WEIGHTS_BY_MODEL", {"fake": enum})
    monkeypatch.setattr(cwf, "_build", build)
    monkeypatch.setattr(cwf, "_read_headers", read)
    code = cwf.main(["--model", "fake"])
    said = capsys.readouterr()
    return code, said.out, said.err, asked


def _refused(urls: list[str]) -> dict[str, dict[str, object] | str]:
    return {url: "HTTPError: HTTP Error 401: Unauthorized" for url in urls}


def _read(urls: list[str]) -> dict[str, dict[str, object] | str]:
    return {url: dict(_HEADER) for url in urls}


def test_nothing_read_says_so_and_never_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err, _ = _run(monkeypatch, capsys, _Fake, _refused)
    assert code == 2
    assert "0 checkpoints read, 2 could not be read" in out
    assert "OK" not in out + err
    assert "NOTHING CHECKED" in err
    assert err.count("HTTPError") == 2  # each entry named, with why


def test_some_read_says_how_many_are_left_and_never_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def half(urls: list[str]) -> dict[str, dict[str, object] | str]:
        return {url: (_HEADER if "ONE" in url else "URLError: timed out") for url in urls}

    code, out, err, _ = _run(monkeypatch, capsys, _Fake, half)
    assert code == 2
    assert "1 checkpoints read, 1 could not be read" in out
    assert "OK" not in out + err
    assert "INCOMPLETE" in err


def test_every_checkpoint_read_and_fitting_is_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err, _ = _run(monkeypatch, capsys, _Fake, _read)
    assert code == 0, out + err
    assert "2 checkpoints read, 0 could not be read" in out
    assert "OK — every checkpoint fits" in out


def test_a_checkpoint_that_does_not_fit_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def odd(urls: list[str]) -> dict[str, dict[str, object] | str]:
        return {url: dict(_OTHER_HEADER if "TWO" in url else _HEADER) for url in urls}

    code, out, err, _ = _run(monkeypatch, capsys, _Fake, odd)
    assert code == 1
    assert "shape differs" in err and "OK" not in out + err


def test_a_placeholder_digest_is_an_error_by_itself(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err, asked = _run(monkeypatch, capsys, _Unpublished, _read)
    assert code == 1, out + err
    assert "fake [DRAFT]: sha256 is the placeholder 00000000" in err
    assert "fake [ONE]" not in err  # a real digest beside it is no problem
    assert "OK" not in out + err
    # Nothing is published there to read, so nothing asks for it.
    assert asked == [_Unpublished.ONE.entry.url]


def test_a_placeholder_digest_fails_even_when_the_network_is_down(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 2 (unknown) would let a gate that tolerates it pass a digest that is known wrong."""
    code, _, err, _ = _run(monkeypatch, capsys, _Unpublished, _refused)
    assert code == 1 and "placeholder" in err


def test_placeholders_lists_only_the_stand_ins() -> None:
    found = cwf._placeholders([("fake", _Fake), ("unpublished", _Unpublished)])
    assert [(name, tag) for name, tag, _ in found] == [("unpublished", "DRAFT")]
    assert cwf._placeholders([("fake", _Fake)]) == []


@pytest.mark.parametrize(
    ("checked", "unreachable", "bad", "code", "word"),
    [
        (3, 0, 0, 0, "OK"),
        (0, 3, 0, 2, "NOTHING CHECKED"),
        (2, 1, 0, 2, "INCOMPLETE"),
        (2, 0, 1, 1, "problem(s)"),
        (0, 0, 0, 2, "NOTHING CHECKED"),  # nothing selected can read as nothing wrong
        (0, 3, 1, 1, "problem(s)"),
    ],
    ids=["all-read", "none-read", "some-read", "a-misfit", "no-entries", "misfit-and-none-read"],
)
def test_conclude_claims_only_what_was_read(
    capsys: pytest.CaptureFixture[str],
    checked: int,
    unreachable: int,
    bad: int,
    code: int,
    word: str,
) -> None:
    got = cwf._conclude(
        checked=checked,
        declined=0,
        unreachable=["  x"] * unreachable,
        bad=["  y"] * bad,
    )
    said = capsys.readouterr()
    assert got == code
    assert f"{checked} checkpoints read, {unreachable} could not be read" in said.out
    assert word in said.out + said.err
    assert ("OK" in said.out) is (word == "OK")
