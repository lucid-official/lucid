"""A fallback to eager should be findable, not just correct.

``compile`` builds an MPSGraph when it can and runs the model eagerly
when it cannot. The result is right either way, which is what made the
silence tolerable — and also what made it invisible: a caller who asked
for a compiled model got an eager one with nothing to read anywhere, not
even under ``LUCID_COMPILE_VERBOSE=1``, which narrates everything else
about the build.

The engine already says why. ``compile_trace`` writes a reason and the
binding raises it; the Python layer caught that exception to fall back
and dropped the message with it.

Printed under ``LUCID_COMPILE_VERBOSE=1`` was still silent for anyone who
did not know to set it — a model the emitters could not take ran eagerly
with nothing said.
So a fallback now also warns, once per kind of reason in a process: the
news is that it happened, and a warning on every call would be noise.
"""

import subprocess
import sys
import textwrap

import pytest

import lucid
from lucid._C import engine as _C_engine


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to("metal")
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


#: The inverse emitter takes a matrix up to 24 rows; 40 (also written into ``PROGRAM``) is
#: one it registers for and declines.
_TOO_BIG = 40

PROGRAM = textwrap.dedent("""
    import lucid, lucid.nn as nn, lucid.nn.functional as F
    from lucid._C import engine as _C_engine

    class Inverse(nn.Module):
        def forward(self, x):
            return lucid.linalg.inv(x)

    model = Inverse().eval().to("metal")
    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model)
    compiled(lucid.randn(40, 40).to("metal"))
    compiled(lucid.randn(40, 40).to("metal"))
    # Another model with the same gap: the same kind of reason, already told.
    lucid.compile.compile(Inverse().eval().to("metal"))(lucid.randn(40, 40).to("metal"))
    print("CACHE", _C_engine.compile.session_cache_size())
    """)


def _run(verbose: bool) -> subprocess.CompletedProcess:
    import os

    environment = dict(os.environ)
    if verbose:
        environment["LUCID_COMPILE_VERBOSE"] = "1"
    else:
        environment.pop("LUCID_COMPILE_VERBOSE", None)
    return subprocess.run(
        [sys.executable, "-c", PROGRAM],
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
    )


class TestFallbackIsReported:
    def test_verbose_names_the_operation_and_the_reason(self) -> None:
        done = _run(verbose=True)
        assert done.returncode == 0, done.stderr
        # It really did fall back — otherwise the assertion below would
        # pass for a run that compiled and said nothing.
        assert "CACHE 0" in done.stdout
        assert "eager fallback" in done.stderr
        assert "inv" in done.stderr

    def test_by_default_it_warns_once(self) -> None:
        """A fallback is correct, so it is one warning, not one per call."""
        done = _run(verbose=False)
        assert done.returncode == 0, done.stderr
        assert "CACHE 0" in done.stdout
        # Not the verbose narration...
        assert "[compile] eager fallback" not in done.stderr
        # ...but said, once, naming the reason — for three fallbacks.
        assert done.stderr.count("CompileFallbackWarning") == 1
        assert "inv" in done.stderr
        assert "running eagerly" in done.stderr


class TestRegistrationIsNotCompilation:
    def test_a_registered_emitter_may_still_decline(self) -> None:
        """``emitter_registered`` answers a narrower question than it looks.

        It is a registry lookup. An emitter may be registered and still
        refuse the variant it is handed, which is how six operations
        report as supported and run eagerly.
        """
        assert _C_engine.compile.emitter_registered("inv")

        import lucid.nn as nn

        class Inverse(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return lucid.linalg.inv(x)

        model = Inverse().eval().to("metal")
        _C_engine.compile.session_cache_clear()
        compiled = lucid.compile.compile(model)
        compiled(lucid.randn(_TOO_BIG, _TOO_BIG).to("metal"))
        assert _C_engine.compile.session_cache_size() == 0
