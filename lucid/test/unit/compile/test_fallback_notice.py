"""A compiled call that runs eagerly says so — once per kind of reason.

``lucid.compile`` falls back to eager whenever it cannot build an
executable, and the answer is right either way. That made the fallback
easy to miss entirely: a model that fell back ran eagerly on every call
and said nothing unless ``LUCID_COMPILE_VERBOSE=1`` was set. Each fallback
now goes through one notice that warns the first time a kind of reason is
met in a process and stays quiet after, since a model that falls back does
so on every call.
"""

import warnings
from collections.abc import Iterator

import pytest

import lucid
import lucid.nn as nn
from lucid.compile._core import fallback
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


@pytest.fixture(autouse=True)
def _fresh_notices(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test starts as a process that has announced nothing yet."""
    monkeypatch.setattr(fallback, "_NOTICED", set())
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)
    yield


def _fallback_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message) for w in caught if issubclass(w.category, fallback.CompileFallbackWarning)
    ]


class TestTheNotice:
    def test_a_category_warns_once(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback._fallback_notice("lowering", "op 'inv' has no emitter")
            fallback._fallback_notice("lowering", "op 'inv' has no emitter")
            fallback._fallback_notice("lowering", "op 'qr' has no emitter")
        said = _fallback_warnings(caught)
        assert len(said) == 1
        assert "op 'inv' has no emitter" in said[0]
        assert "running eagerly" in said[0]

    def test_each_category_warns_on_its_own(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback._fallback_notice("lowering", "op 'inv' has no emitter")
            fallback._fallback_notice("first-run", "the forward wrote its argument")
        assert len(_fallback_warnings(caught)) == 2

    def test_verbose_prints_every_one_and_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("LUCID_COMPILE_VERBOSE", "1")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback._fallback_notice("lowering", "first")
            fallback._fallback_notice("lowering", "second")
        assert not _fallback_warnings(caught)
        err = capsys.readouterr().err
        assert "[compile] eager fallback: first" in err
        assert "[compile] eager fallback: second" in err

    def test_the_warning_points_at_the_caller(self) -> None:
        """Not at a line inside ``lucid.compile``, which the reader cannot act on."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback._fallback_notice("lowering", "op 'inv' has no emitter")
        (only,) = [w for w in caught if issubclass(w.category, fallback.CompileFallbackWarning)]
        assert only.filename == __file__


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
def test_a_write_to_an_argument_is_announced() -> None:
    class _Bumps(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            with lucid.no_grad():
                x.add_(1.0)
            return self.lin(x)

    compiled = lucid.compile(_Bumps().to(COMPILE_DEVICE))
    x = lucid.zeros(2, 4, device=COMPILE_DEVICE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled(x)
        compiled(x)
    said = _fallback_warnings(caught)
    assert len(said) == 1
    assert "own arguments" in said[0]
