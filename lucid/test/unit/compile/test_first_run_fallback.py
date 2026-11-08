"""A first run MPSGraph fails is answered eagerly by every compile entry (LCD-936).

MPSGraph specializes an executable inside its first run, and has gaps there that
compiling does not reveal.  Two entries (``compile``, ``fused_step``) used to
answer such a failure eagerly while three (``make_step``, its segmented form and
``compiled_decode_step``) raised: one ``RuntimeError`` ended a training loop
that the same model under ``lucid.compile`` survived.  ``is_first_run_failure``
decides now, and each entry follows it the same way: the run wrote nothing, so
the call is answered eagerly, its signature remembered, and one
``CompileFallbackWarning`` says so.

The natural trigger is gone (``392240a28`` fixed the int64 ``square`` gap), so
each test injects the failure at ``run_executable`` / ``run_executable_inplace``
and compares the answer, and the BatchNorm statistics where the model has them,
with a twin model run eagerly.
"""

import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.compile import fused_step, make_step
from lucid.compile._core.attention_probe import ensure_attention_workaround_probed
from lucid.compile._core import fallback
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.compile.test_compiled_decode import _D, _L, _MaskedDecoder, _B
from lucid.utils.cache import StaticCache

_TOL = 1e-5


def _inject(monkeypatch: pytest.MonkeyPatch, runner: str, fail_calls: set[int]) -> list[int]:
    """Make the ``fail_calls``-th (0-based) call of ``compile.<runner>`` raise as MPSGraph does.

    Returns the list that grows by one per call, failed or not: its length is how
    many times the engine was asked to run an executable.
    """
    real: Callable[..., object] = getattr(_C_engine.compile, runner)
    calls: list[int] = []

    def wrapper(*args: object) -> object:
        n = len(calls)
        calls.append(n)
        if n in fail_calls:
            raise RuntimeError(f"{runner}: MPSGraph failed: injected")
        return real(*args)

    monkeypatch.setattr(_C_engine.compile, runner, wrapper)
    return calls


def _fallback_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message) for w in caught if issubclass(w.category, fallback.CompileFallbackWarning)
    ]


def _assert_announced_once(caught: list[warnings.WarningMessage]) -> None:
    said = _fallback_warnings(caught)
    assert len(said) == 1 and "MPSGraph failed" in said[0]
    assert fallback._NOTICED == {"first-run"}


def _bn_trunk() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 2)).to(
        COMPILE_DEVICE
    )


def _twin(model: nn.Sequential) -> nn.Sequential:
    other = _bn_trunk()
    other.load_state_dict(model.state_dict())
    return other


def _same(a: lucid.Tensor, b: lucid.Tensor) -> None:
    assert a.shape == b.shape
    assert float((a - b).abs().max().item()) <= _TOL


def _same_bn(a: nn.Sequential, b: nn.Sequential) -> None:
    bn_a, bn_b = a[1], b[1]
    assert isinstance(bn_a, nn.BatchNorm1d) and isinstance(bn_b, nn.BatchNorm1d)
    _same(bn_a.running_mean, bn_b.running_mean)
    _same(bn_a.running_var, bn_b.running_var)
    assert int(bn_a.num_batches_tracked.item()) == int(bn_b.num_batches_tracked.item())


def test_the_check_reads_the_engine_message() -> None:
    assert fallback.is_first_run_failure(RuntimeError("run_executable: MPSGraph failed: x"))
    assert not fallback.is_first_run_failure(RuntimeError("run_executable: out of memory"))


@pytest.mark.usefixtures("inference")
def test_compile_answers_eagerly(monkeypatch: pytest.MonkeyPatch) -> None:
    lucid.manual_seed(1)
    model = _bn_trunk()
    eager = _twin(model)
    compiled = lucid.compile(model)
    calls = _inject(monkeypatch, "run_executable", {0})
    xs = [lucid.randn(5, 4, device=COMPILE_DEVICE) for _ in range(2)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for x in xs:
            _same(compiled(x), eager(x))
            _same_bn(model, eager)
    _assert_announced_once(caught)
    assert len(calls) == 1  # the second call never reached the engine: no retrace


def test_make_step_answers_eagerly(monkeypatch: pytest.MonkeyPatch) -> None:
    lucid.manual_seed(2)
    model = _bn_trunk()
    eager = _twin(model)
    step = make_step(model, F.mse_loss)
    calls = _inject(monkeypatch, "run_executable", {0})
    batches = [
        (lucid.randn(5, 4, device=COMPILE_DEVICE), lucid.randn(5, 2, device=COMPILE_DEVICE))
        for _ in range(2)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for x, t in batches:
            for p in (*model.parameters(), *eager.parameters()):
                p.grad = None
            got = step(x, t)
            got.backward()
            want = F.mse_loss(eager(x), t)
            want.backward()
            _same(got, want)
            for p_c, p_e in zip(model.parameters(), eager.parameters(), strict=True):
                assert p_c.grad is not None and p_e.grad is not None
                _same(p_c.grad, p_e.grad)
            _same_bn(model, eager)
    _assert_announced_once(caught)
    assert len(calls) == 1
    assert len(step.cache) == 0 and len(step.eager_only) == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize("fails_at", [0, 1], ids=["first-segment", "second-segment"])
def test_segmented_step_answers_eagerly(monkeypatch: pytest.MonkeyPatch, fails_at: int) -> None:
    """A later segment failing must not leave the earlier segment's statistics applied twice."""
    lucid.manual_seed(3)
    model = _bn_trunk()  # segment 0 = Linear + BatchNorm, segment 1 = ReLU + Linear
    eager = _twin(model)
    step = make_step(model, F.mse_loss, segments=2)
    calls = _inject(monkeypatch, "run_executable", {fails_at})
    batches = [
        (lucid.randn(5, 4, device=COMPILE_DEVICE), lucid.randn(5, 2, device=COMPILE_DEVICE))
        for _ in range(2)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for x, t in batches:
            for p in (*model.parameters(), *eager.parameters()):
                p.grad = None
            got = step(x, t)
            got.backward()
            want = F.mse_loss(eager(x), t)
            want.backward()
            _same(got, want)
            for p_c, p_e in zip(model.parameters(), eager.parameters(), strict=True):
                assert p_c.grad is not None and p_e.grad is not None
                _same(p_c.grad, p_e.grad)
            _same_bn(model, eager)
    _assert_announced_once(caught)
    assert len(calls) == fails_at + 1  # the failing run was the last the engine saw
    assert len(step.eager_only) == 1  # type: ignore[attr-defined]


def test_decode_step_answers_eagerly(monkeypatch: pytest.MonkeyPatch) -> None:
    lucid.manual_seed(4)
    model = _MaskedDecoder().to(COMPILE_DEVICE).eval()

    def decode(compiled: bool) -> list[lucid.Tensor]:
        cache = StaticCache(_L)

        def fwd(h: lucid.Tensor, cp: lucid.Tensor) -> lucid.Tensor:
            return model(h, cache, cp)

        step = compiled_decode_step(fwd, cache) if compiled else fwd
        outs = []
        for pos in range(4):
            h = lucid.ones(_B, 1, _D, device=COMPILE_DEVICE) * ((pos + 1) * 0.1)
            cp = lucid.tensor([pos], device=COMPILE_DEVICE).long()
            outs.append(step(h, cp))
        return outs

    # The decode graph holds attention, which runs the process-wide capability probe
    # on first sight; an injected failure inside it would pin the workaround on.
    ensure_attention_workaround_probed()
    with lucid.no_grad():
        want = decode(compiled=False)
        calls = _inject(monkeypatch, "run_executable", {0})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = decode(compiled=True)
    for g, w in zip(got, want, strict=True):
        _same(g, w)
    _assert_announced_once(caught)
    assert len(calls) == 1


def test_fused_step_answers_eagerly(monkeypatch: pytest.MonkeyPatch) -> None:
    lucid.manual_seed(5)
    model = _bn_trunk()
    eager = _twin(model)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    eager_opt = optim.SGD(eager.parameters(), lr=0.1, momentum=0.5)
    step = fused_step(model, F.mse_loss, opt)
    calls = _inject(monkeypatch, "run_executable_inplace", {0})
    batches = [
        (lucid.randn(5, 4, device=COMPILE_DEVICE), lucid.randn(5, 2, device=COMPILE_DEVICE))
        for _ in range(2)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for x, t in batches:
            got = step(x, t)
            eager_opt.zero_grad()
            want = F.mse_loss(eager(x), t)
            want.backward()
            eager_opt.step()
            _same(got, want)
            _same_bn(model, eager)
    _assert_announced_once(caught)
    for p_c, p_e in zip(model.parameters(), eager.parameters(), strict=True):
        _same(p_c, p_e)
    # The fused run failed once; what ran after it was the eager optimizer's own.
    assert calls[0] == 0 and len(calls) >= 1
