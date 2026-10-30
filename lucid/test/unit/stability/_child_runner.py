"""Run cases in child interpreters, so a crash fails one case, not the run.

A case that ends the process — a segfault, an ``abort``, the kernel's
SIGKILL — would take the test runner with it, and no in-process assertion
could report it.  So the cases run in a child: one per device runs every
case in order, printing ``@@BEGIN <name>`` before each and ``@@END <name>
<json verdict>`` after.  A child that dies is blamed on the case it had
begun and is restarted after it, so one crash does not hide the next.

The child script reads ``[[name, code], ...]`` as JSON on stdin and takes
the device as ``argv[1]``; what a verdict holds is the caller's choice.
"""

import json
import os
import signal
import subprocess
import sys
from collections.abc import Sequence

CHILD_TIMEOUT_S = 600


def describe_exit(code: int) -> str:
    """How a child ended, for the crash verdict."""
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    if code == 255:
        return "exit 255 (Accelerate's BLAS error handler exits with 255)"
    return f"exit {code}"


def run_child_cases(
    child: str, cases: Sequence[tuple[str, str]], device: str, crash_extra: Sequence[object] = ()
) -> dict[str, list[object]]:
    """Every case's verdict on ``device``, ``["crash", detail, *crash_extra]``
    for one that killed its child."""
    results: dict[str, list[object]] = {}
    pending = list(cases)
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    while pending:
        done = subprocess.run(
            [sys.executable, "-c", child, device],
            input=json.dumps(pending),
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
            env=env,
        )
        current = None
        for line in done.stdout.splitlines():
            if line.startswith("@@BEGIN "):
                current = line[len("@@BEGIN ") :]
            elif line.startswith("@@END "):
                name, _, verdict = line[len("@@END ") :].partition(" ")
                results[name] = json.loads(verdict)
                current = None
        if done.returncode == 0:
            break
        crash = ["crash", f"{describe_exit(done.returncode)}: {done.stderr.strip()[-600:]}"]
        crash.extend(crash_extra)
        if current is None:
            # Died outside any case — the prelude itself.  Blame everything left.
            for name, _ in pending:
                results.setdefault(name, list(crash))
            break
        results[current] = crash
        pending = [(n, c) for n, c in pending if n not in results]
    return results
