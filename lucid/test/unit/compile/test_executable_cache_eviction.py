"""A compiled module keeps its executables after the session cache evicts them.

The process-wide executable cache holds at most ``LUCID_COMPILE_MAX_CACHE``
entries (32 by default) and used to delete an executable on eviction, while
every CompiledModule kept a borrowed pointer to it.  A model compiled with
more distinct signatures than that, across the process, then called freed
memory.  Executables are shared now: eviction drops the cache's reference
and the module's keeps the executable alive.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from lucid.test._fixtures.devices import metal_available

_SCRIPT = textwrap.dedent("""
    import numpy as np
    import lucid
    import lucid.nn as nn

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    model = model.to("metal").eval()
    compiled = lucid.compile(model)
    xs = [lucid.randn(b, 8, device="metal") for b in (1, 2, 3, 4, 5)]
    # Under grad mode a trainable model's call runs eagerly (LCD-582).
    with lucid.no_grad():
        for x in xs:
            compiled(x)
        assert compiled.cache_info()["entries"] == 5
        for x in xs:
            np.testing.assert_allclose(
                compiled(x).numpy(), model(x).numpy(), atol=1e-5
            )
    print("ok")
    """)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_an_evicted_executable_still_runs_for_the_module_that_holds_it() -> None:
    env = dict(os.environ, LUCID_COMPILE_MAX_CACHE="2")
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("ok")
