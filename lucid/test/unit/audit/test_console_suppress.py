"""``Suppress`` has to silence the descriptors, not just the Python names.

Rebinding ``sys.stdout`` and ``sys.stderr`` silences Python and nothing
else.  The libraries underneath write to file descriptors 1 and 2
directly and never look at either name, so three of them wrote straight
through it and into the middle of a live-progress frame:

    ** On entry to DGESDD, parameter number  5 had an illegal value
    ** On entry to DGEBAL, parameter number  3 had an illegal value
    mx.metal.device_info is deprecated and will be removed ...

The display could not redraw over what it had not written, so each left a
stale half-drawn copy of itself in the scrollback.  Three writes, three
stale frames — the same three at every terminal height, which is what
showed the display was not simply too tall for the screen.
"""

import os
import subprocess
import sys

from lucid.test.audit._console import Suppress


def _write_to_fd(fd: int, text: str) -> None:
    """Write past Python's file objects, the way a C library does."""
    os.write(fd, text.encode())


def test_python_level_writes_are_silenced(capfd) -> None:
    with Suppress():
        print("python stdout")
        print("python stderr", file=sys.stderr)
    out, err = capfd.readouterr()
    assert "python stdout" not in out
    assert "python stderr" not in err


def test_descriptor_level_writes_are_silenced(capfd) -> None:
    """The part that swapping ``sys.stdout`` could never do."""
    with Suppress():
        _write_to_fd(1, "lapack complains here\n")
        _write_to_fd(2, "and here\n")
    out, err = capfd.readouterr()
    assert "lapack complains" not in out
    assert "and here" not in err


def test_both_streams_come_back(capfd) -> None:
    with Suppress():
        _write_to_fd(1, "swallowed\n")
    print("visible stdout")
    print("visible stderr", file=sys.stderr)
    _write_to_fd(1, "visible fd\n")
    out, err = capfd.readouterr()
    assert "visible stdout" in out
    assert "visible stderr" in err
    assert "visible fd" in out
    assert "swallowed" not in out


def test_disabled_suppress_lets_everything_through(capfd) -> None:
    with Suppress(enabled=False):
        print("still here")
        _write_to_fd(1, "fd too\n")
    out, _ = capfd.readouterr()
    assert "still here" in out
    assert "fd too" in out


def test_nesting_restores_in_order(capfd) -> None:
    with Suppress():
        _write_to_fd(1, "outer\n")
        with Suppress():
            _write_to_fd(1, "inner\n")
        _write_to_fd(1, "outer again\n")
    _write_to_fd(1, "after\n")
    out, _ = capfd.readouterr()
    assert out.strip() == "after"


def test_descriptors_are_not_leaked() -> None:
    """Entered once per probed cell — a leak would exhaust the table.

    Runs in a subprocess so the count is not perturbed by whatever the
    test session already holds open.  The first ``Suppress`` is outside
    the measurement because it opens the shared ``/dev/null`` that every
    later one reuses.
    """
    code = (
        "import os\n"
        "from lucid.test.audit._console import Suppress\n"
        "def n():\n"
        "    return len(os.listdir('/dev/fd'))\n"
        "with Suppress():\n"
        "    pass\n"
        "before = n()\n"
        "for _ in range(200):\n"
        "    with Suppress():\n"
        "        os.write(1, b'x')\n"
        "os.write(2, str(n() - before).encode())\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stderr.strip() == "0", done.stderr
