"""Lucid runs without numpy installed — smoke test the main API surface
under a meta_path blocker that simulates a numpy-free environment.

The test runs in a fresh subprocess so the meta_path blocker doesn't
contaminate the rest of pytest (numpy is deliberately part of the test
infra).  We only assert that the listed API calls do not pull numpy
into ``sys.modules`` — actual numerical results are exercised by the
rest of the suite under normal conditions.
"""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.smoke
def test_lucid_works_without_numpy() -> None:
    script = textwrap.dedent("""
        import sys

        # Eject any cached numpy and refuse new imports.
        for mod in list(sys.modules):
            if mod == "numpy" or mod.startswith("numpy."):
                del sys.modules[mod]

        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "numpy" or name.startswith("numpy."):
                    raise ImportError("numpy blocked for test")
                return None

        sys.meta_path.insert(0, _Blocker())

        import lucid
        import lucid.nn as nn
        import lucid.nn.functional as F
        import lucid.optim as optim

        # Core factory + repr (engine to_string).
        t = lucid.zeros(3, 4)
        assert t.shape == (3, 4)
        assert "tensor(" in repr(t)

        # Scalar extraction (engine to_bytes + struct unpack).
        assert lucid.zeros(1).item() == 0.0
        assert (lucid.ones(1) + 0.5).item() == 1.5

        # Forward + backward + optimizer step.
        model = nn.Linear(4, 2)
        x = lucid.zeros(8, 4) + 0.5
        y = lucid.zeros(8, 2) + 1.0
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            assert p.grad.shape == p.shape
        opt = optim.SGD(model.parameters(), lr=0.01)
        opt.step()

        # Save / load round-trip (engine to_bytes / from_bytes).
        import io
        buf = io.BytesIO()
        lucid.save(model.state_dict(), buf)
        buf.seek(0)
        sd = lucid.load(buf, weights_only=True)
        assert "weight" in sd and "bias" in sd

        # Final assertion: numpy never made it into sys.modules.
        assert "numpy" not in sys.modules, (
            "numpy was imported during a numpy-blocked Lucid session"
        )
        print("OK")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, (
        f"numpy-blocked smoke test failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout
