"""A dropped network is not a parity finding, and white noise is not a fair DETR input.

LCD-953.  The weekly pretrained-parity run was red three weeks running for
reasons that were not conversion defects: a DNS blip on ``vgg_16_cls`` and
``vgg_19_bn_cls`` read as an ``error``, and DETR-R50's boxes sat 1.2e-4 apart
(tolerance 1e-4) on seed 0's noise because its decoder's attention scores,
near 90, turn float32 rounding into 1e-4.  Nothing here touches the network:
failures are injected and the backoff's sleep is a recorder.
"""

import ast
import email.message
import io
import json
import socket
import sys
import urllib.error
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import check_pretrained_parity as parity
from tools._network_retry import Backoff, is_network_failure


def _dns() -> urllib.error.URLError:
    """What 2026-11-03's job saw: ``URLError`` over ``gaierror(8)``."""
    return urllib.error.URLError(socket.gaierror(8, "nodename nor servname provided, or not known"))


def _http(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.invalid/x", status, "reason", email.message.Message(), io.BytesIO()
    )


def _raised_from(outer: Exception, cause: BaseException) -> Exception:
    outer.__cause__ = cause
    return outer


NETWORK = [
    _dns(),
    socket.gaierror(8, "nodename nor servname provided, or not known"),
    TimeoutError("timed out"),
    ConnectionResetError(54, "Connection reset by peer"),
    ConnectionRefusedError(61, "Connection refused"),
    OSError(8, "nodename nor servname provided, or not known"),
    _http(503),
    _http(429),
    # How the hub clients report it: their own class on the transport's.
    _raised_from(OSError("We couldn't connect to 'https://huggingface.co'"), _dns()),
    _raised_from(RuntimeError("download failed"), _raised_from(OSError("hub"), TimeoutError())),
]

NOT_NETWORK = [
    ValueError("shape (1, 3) against (1, 4)"),
    KeyError("pred_boxes"),
    RuntimeError("SHA256 mismatch for vgg16.pth"),
    # macOS numbers EAI_AGAIN 2, as it does ENOENT: a missing hubconf.py must
    # not be mistaken for a resolver failure.
    FileNotFoundError(2, "No such file or directory"),
    PermissionError(13, "Permission denied"),
    _http(404),
    _http(403),
    # A bare OSError with only a message carries no evidence either way.
    OSError("We couldn't connect to 'https://huggingface.co'"),
]


@pytest.mark.parametrize("exc", NETWORK, ids=lambda e: type(e).__name__)
def test_what_the_network_raises_is_recognised(exc: BaseException) -> None:
    assert is_network_failure(exc)


@pytest.mark.parametrize("exc", NOT_NETWORK, ids=lambda e: f"{type(e).__name__}-{e}"[:40])
def test_what_it_did_not_raise_is_not(exc: BaseException) -> None:
    assert not is_network_failure(exc)


def test_a_failure_raised_from_none_does_not_inherit_the_context() -> None:
    # ``raise X from None`` says the earlier error is not the reason.
    exc = RuntimeError("the checkpoint is corrupt")
    exc.__context__ = _dns()
    exc.__suppress_context__ = True
    assert not is_network_failure(exc)


def test_backoff_spends_its_delays_in_order_then_gives_up() -> None:
    slept: list[float] = []
    backoff = Backoff(delays=(5.0, 15.0, 45.0), sleep=slept.append)
    assert [backoff.wait(retry) for retry in range(4)] == [True, True, True, False]
    assert slept == [5.0, 15.0, 45.0]


def test_backoff_allowance_is_shared_by_every_job_in_the_run() -> None:
    slept: list[float] = []
    backoff = Backoff(delays=(5.0, 5.0), allowance=7.0, sleep=slept.append)
    assert backoff.wait(0) and not backoff.wait(1)  # 2 s left: a second 5 s will not fit
    assert not backoff.wait(0)  # ... for the next checkpoint either
    assert slept == [5.0]


class _Script:
    """A stand-in ``_compare_once`` that answers from a list, then repeats the last."""

    def __init__(self, *answers: dict[str, object]) -> None:
        self._answers = list(answers)
        self.calls = 0

    def __call__(self, *args: object) -> dict[str, object]:
        self.calls += 1
        return self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]


GONE = parity._failure(_dns())
AGREES = {"max_diff": 1e-6, "scale": 1.0, "top1_agrees": True}


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(parity, "_NETWORK", Backoff(sleep=delays.append))
    return delays


def _run(monkeypatch: pytest.MonkeyPatch, script: _Script) -> dict[str, object]:
    monkeypatch.setattr(parity, "_compare_once", script)
    return parity._compare("m", "timm/m", (1, 3, 8, 8), None)


def test_a_failure_the_network_clears_is_tried_again(
    monkeypatch: pytest.MonkeyPatch, slept: list[float]
) -> None:
    script = _Script(GONE, GONE, AGREES)
    assert _run(monkeypatch, script) == AGREES
    assert (script.calls, slept) == (3, [5.0, 15.0])


def test_a_persistent_failure_is_unreachable_network_after_every_retry(
    monkeypatch: pytest.MonkeyPatch, slept: list[float]
) -> None:
    script = _Script(GONE)
    result = _run(monkeypatch, script)
    assert (script.calls, slept) == (4, [5.0, 15.0, 45.0])
    assert result["network"] is True
    assert str(result["unreachable"]).startswith("unreachable (network): URLError")
    assert "error" not in result


@pytest.mark.parametrize(
    "answer",
    [
        {"max_diff": 1.0, "scale": 1.0, "top1_agrees": False},
        {"error": "ValueError: shape (1, 3) against (1, 4)"},
        {"unreachable": "the timm reference is not installed"},
    ],
)
def test_nothing_but_the_network_is_retried(
    monkeypatch: pytest.MonkeyPatch, slept: list[float], answer: dict[str, object]
) -> None:
    script = _Script(answer)
    assert _run(monkeypatch, script) == answer
    assert (script.calls, slept) == (1, [])


def _unreachable_host(*args: object, **kwargs: object) -> None:
    raise _dns()


@pytest.mark.parametrize("side", ["lucid", "reference"])
def test_both_downloads_of_a_comparison_are_retried(
    monkeypatch: pytest.MonkeyPatch, slept: list[float], side: str
) -> None:
    model = SimpleNamespace(eval=lambda: object())
    loads = {"lucid": 0, "reference": 0}

    def create_model(*args: object, **kwargs: object) -> SimpleNamespace:
        loads["lucid"] += 1
        if side == "lucid":
            raise _dns()
        return model

    def build_reference(*args: object) -> None:
        loads["reference"] += 1
        raise _dns()

    monkeypatch.setattr(parity, "create_model", create_model)
    monkeypatch.setattr(parity, "_build_reference", build_reference)
    monkeypatch.setattr(parity, "zoo_module", lambda: object())
    monkeypatch.setattr(parity, "ref_module", lambda: object())
    result = parity._compare("m", "timm/m.tag", (1, 3, 8, 8), None)
    assert result["network"] is True
    assert slept == [5.0, 15.0, 45.0]
    # The side that fails is attempted four times; the one behind it only
    # when the first loaded.
    assert loads["lucid"] == 4
    assert loads["reference"] == (4 if side == "reference" else 0)


def test_the_adapters_report_a_dropped_download_as_unreachable_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parity, "ref_module", lambda: object())
    monkeypatch.setattr(parity, "ref_vision_module", lambda: object())
    monkeypatch.setattr(parity, "create_model", _unreachable_host)
    gone = parity._compare_detr("detr_resnet50", "detr_resnet50", (1, 3, 8, 8))
    assert gone["network"] is True and "error" not in gone

    def broken(*args: object, **kwargs: object) -> None:
        raise ValueError("unexpected checkpoint layout")

    monkeypatch.setattr(parity, "create_model", broken)
    wrong = parity._compare_detr("detr_resnet50", "detr_resnet50", (1, 3, 8, 8))
    assert wrong == {"error": "ValueError: unexpected checkpoint layout"}


def test_every_comparison_reports_its_exceptions_through_the_one_classifier() -> None:
    """A handler that built its own ``error`` would bring the DNS-blip red back."""
    tree = ast.parse(Path(parity.__file__).read_text())
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "Exception"
    ]
    assert len(handlers) >= 10  # the comparisons; a rewrite that loses them is not a pass
    for handler in handlers:
        (statement,) = handler.body
        assert isinstance(statement, ast.Return), ast.dump(handler)
        call = statement.value
        assert (
            isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_failure"
        ), f"line {handler.lineno}: report the exception with _failure(exc)"


# --- the verdict: exit codes -------------------------------------------------


def _main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    answers: dict[str, dict[str, object]],
    *flags: str,
) -> tuple[int, Path]:
    member = SimpleNamespace(value=SimpleNamespace(meta={"source": "timm/x", "num_params": 1}))
    monkeypatch.setattr(parity, "_WEIGHTS_BY_MODEL", {name: object() for name in answers})
    monkeypatch.setattr(parity, "_factory_default", lambda name: member)
    monkeypatch.setattr(parity, "_compare", lambda name, *args: answers[name])
    monkeypatch.setattr(parity, "_reference_io", lambda: nullcontext())
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["parity", "--json", str(report), *flags])
    return parity.main(), report


def test_a_network_unreachable_checkpoint_is_listed_and_does_not_fail_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    code, report = _main(monkeypatch, tmp_path, {"a_ok": AGREES, "b_gone": GONE})
    assert code == 0
    captured = capsys.readouterr()
    assert "1 unreachable (network) after retries: b_gone" in captured.err
    assert "::warning" not in captured.out
    evidence = {row["model"]: row for row in json.loads(report.read_text())["evidence"]}
    assert evidence["b_gone"]["network"] is True and "error" not in evidence["b_gone"]
    assert json.loads(report.read_text())["complete"] is False  # not claimed as verified


def test_under_github_actions_the_unreachable_list_is_also_a_warning_annotation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    code, _ = _main(monkeypatch, tmp_path, {"a_ok": AGREES, "b_gone": GONE})
    assert code == 0
    assert "::warning title=Pretrained parity::1 checkpoint(s) unreachable (network)" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "answers,code",
    [
        ({"a": AGREES, "b": GONE}, 0),
        ({"a": AGREES, "b": {"max_diff": 0.5, "scale": 1.0, "top1_agrees": True}}, 1),
        ({"a": GONE, "b": {"max_diff": 0.5, "scale": 1.0, "top1_agrees": True}}, 1),
        ({"a": AGREES, "b": {"error": "ValueError: wrong shape"}}, 1),
        ({"a": AGREES, "b": {"max_diff": 1e-9, "scale": 1.0, "top1_agrees": False}}, 1),
        ({"a": GONE, "b": GONE}, 2),
    ],
    ids=[
        "network-only-unreachable-passes",
        "numeric-mismatch-fails",
        "mismatch-beside-unreachable-still-fails",
        "non-network-error-fails",
        "top1-disagreement-fails",
        "nothing-compared-is-2",
    ],
)
def test_exit_code_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    answers: dict[str, dict[str, object]],
    code: int,
) -> None:
    assert _main(monkeypatch, tmp_path, answers)[0] == code


# --- DETR's input -------------------------------------------------------------


def test_the_smooth_image_is_deterministic_float32_and_low_frequency() -> None:
    image = parity._smooth_image((1, 3, 224, 224))
    assert image.dtype == np.float32 and image.shape == (1, 3, 224, 224)
    assert np.array_equal(image, parity._smooth_image((1, 3, 224, 224)))
    assert not np.array_equal(image, parity._smooth_image((1, 3, 224, 224), seed=1))
    np.testing.assert_allclose(image.reshape(3, -1).std(axis=1), 1.0, rtol=1e-5)
    # Neighbouring pixels agree: white noise would differ by about 1.4 here.
    assert np.abs(np.diff(image, axis=-1)).mean() < 0.1


class _Tensor:
    """What ``.numpy()`` is called on, on either side of the comparison."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def numpy(self) -> np.ndarray:
        return self._values


def test_detr_is_given_the_same_smooth_image_on_both_sides(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, np.ndarray] = {}
    boxes, logits = np.full((1, 5, 4), 0.5, np.float32), np.full((1, 5, 3), 2.0, np.float32)

    def theirs(values: _Tensor) -> dict[str, _Tensor]:
        seen["reference"] = values.numpy()
        return {"pred_logits": _Tensor(logits), "pred_boxes": _Tensor(boxes)}

    def ours(values: object) -> SimpleNamespace:
        seen["lucid"] = getattr(values, "numpy")()
        return SimpleNamespace(logits=_Tensor(logits), pred_boxes=_Tensor(boxes))

    reference = SimpleNamespace(
        no_grad=nullcontext,
        from_numpy=_Tensor,
        hub=SimpleNamespace(load=lambda *a, **k: SimpleNamespace(eval=lambda: theirs)),
    )
    monkeypatch.setattr(parity, "ref_module", lambda: reference)
    monkeypatch.setattr(parity, "ref_vision_module", lambda: object())
    monkeypatch.setattr(parity, "create_model", lambda *a, **k: SimpleNamespace(eval=lambda: ours))
    result = parity._compare_detr("detr_resnet50", "detr_resnet50", (1, 3, 32, 32))
    assert result["max_diff"] == 0.0
    expected = parity._smooth_image((1, 3, 32, 32))
    assert np.array_equal(seen["reference"], expected)
    assert np.array_equal(seen["lucid"], expected)
