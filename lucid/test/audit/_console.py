"""Terminal rendering for the audit CLI — standard library only.

No third-party console package.  An audit tool has to run in the one
situation where the environment is least trustworthy, so it carries no
dependency it does not need; everything here is ANSI escapes and
``shutil.get_terminal_size``.

Degrades on purpose.  When stdout is not a TTY, or ``NO_COLOR`` is set,
or ``TERM=dumb``, colour and the in-place live region are dropped and the
same information is emitted as plain sequential lines — so piping to a
file or running under CI produces a readable log rather than a screenful
of escape codes.
"""

import os
import shutil
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from types import TracebackType
    from typing import TextIO

# ── palette ──────────────────────────────────────────────────────────────────
# 256-colour indices, chosen to stay legible on both light and dark
# backgrounds (nothing below index 240, no pure blue on black).
_C = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "red": "\x1b[38;5;203m",
    "green": "\x1b[38;5;114m",
    "yellow": "\x1b[38;5;221m",
    "blue": "\x1b[38;5;110m",
    "magenta": "\x1b[38;5;176m",
    "cyan": "\x1b[38;5;115m",
    "grey": "\x1b[38;5;245m",
    "white": "\x1b[38;5;253m",
}

_BLOCKS = " ▏▎▍▌▋▊▉█"
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def supports_colour(force: bool | None = None) -> bool:
    """Whether to emit escape sequences at all."""
    if force is not None:
        return force
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") in ("dumb", ""):
        return False
    return sys.stdout.isatty()


class Console:
    """Everything the audit prints goes through one of these.

    Parameters
    ----------
    colour : bool, optional
        Force colour on or off.  ``None`` auto-detects.
    width : int, optional
        Force a terminal width.  ``None`` measures it.
    quiet : bool, default=False
        Suppress everything except the final summary and findings.
    log : Path, optional
        Mirror every line to this file as it is printed, without colour.
        The transcript ignores ``quiet``: a run reduced to its summary on
        screen still records in full, which is the point of having one.
        Opening failures are reported and then ignored — a missing log is
        not a reason to lose the run that would have been written to it.
    """

    def __init__(
        self,
        colour: bool | None = None,
        width: int | None = None,
        quiet: bool = False,
        log: Path | None = None,
    ) -> None:
        self.colour = supports_colour(colour)
        self.live = self.colour and sys.stdout.isatty()
        self._width = width
        self.quiet = quiet
        self._live_lines = 0
        self.log_path = log
        self._log: TextIO | None = None
        if log is not None:
            try:
                log.parent.mkdir(parents=True, exist_ok=True)
                # Truncating, not appending: the question a transcript
                # answers is "what did *this* run do", and a file that grows
                # across runs answers it only after the reader finds the
                # right boundary.
                self._log = open(log, "w", encoding="utf-8", buffering=1)
            except OSError as exc:
                self.log_path = None
                sys.stdout.write(f"  could not open {log} for the transcript: {exc}\n")

    def _record(self, text: str) -> None:
        """Mirror one line into the transcript, stripped of styling."""
        if self._log is None:
            return
        try:
            self._log.write(_strip(text) + "\n")
            # Flushed per line rather than per run.  A transcript is worth
            # having while the run is still going — the stages this gate is
            # slowest in are exactly the ones that get killed before the
            # end, and a buffered file loses precisely those.
            self._log.flush()
        except OSError:
            self._log = None
            self.log_path = None

    def note(self, text: str) -> None:
        """Put a line in the transcript that is not printed to the screen.

        For provenance the reader needs and the operator already knows —
        which command produced this file, and when.
        """
        self._record(text)

    def close(self) -> None:
        """Release the transcript.  Safe to call more than once."""
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    @property
    def width(self) -> int:
        """int: Usable terminal width, clamped to something readable."""
        if self._width is not None:
            return self._width
        return max(60, min(shutil.get_terminal_size((100, 24)).columns, 160))

    # ── primitives ───────────────────────────────────────────────────────────

    def paint(self, text: str, *styles: str) -> str:
        """Wrap ``text`` in the named styles, or return it unchanged."""
        if not self.colour or not styles:
            return text
        return "".join(_C[s] for s in styles if s in _C) + text + _C["reset"]

    def write(self, text: str = "") -> None:
        # Recorded before the quiet check, so the transcript is the whole
        # run whatever the screen was asked to show.
        self._record(text)
        if self.quiet:
            return
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def always(self, text: str = "") -> None:
        """Write even in quiet mode — summaries and findings."""
        self._record(text)
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    # ── structure ────────────────────────────────────────────────────────────

    def banner(self, title: str, subtitle: str = "") -> None:
        """The block at the top of a run.

        The structural methods below do not gate on ``quiet`` themselves —
        they hand every line to ``write``, which suppresses the screen and
        still records.  Returning early here instead would cost the
        transcript its shape on exactly the runs that are hardest to read.
        """
        w = self.width
        self.write()
        self.write(self.paint("╭" + "─" * (w - 2) + "╮", "grey"))
        pad = w - 4 - len(title)
        self.write(
            self.paint("│ ", "grey")
            + self.paint(title, "bold", "cyan")
            + " " * max(pad, 0)
            + self.paint(" │", "grey")
        )
        if subtitle:
            pad = w - 4 - len(subtitle)
            self.write(
                self.paint("│ ", "grey")
                + self.paint(subtitle, "grey")
                + " " * max(pad, 0)
                + self.paint(" │", "grey")
            )
        self.write(self.paint("╰" + "─" * (w - 2) + "╯", "grey"))

    def rule(self, title: str = "", style: str = "grey") -> None:
        w = self.width
        if not title:
            self.write(self.paint("─" * w, style))
            return
        left = "── " + title + " "
        self.write(self.paint(left + "─" * max(w - len(left), 0), style))

    def kv(self, key: str, value: str, key_width: int = 22) -> None:
        self.write("  " + self.paint(key.ljust(key_width), "grey") + self.paint(value, "white"))

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        aligns: Sequence[str] | None = None,
        always: bool = False,
    ) -> None:
        """A box-drawn table sized to its contents.

        ``aligns`` is one of ``"l"`` / ``"r"`` / ``"c"`` per column.
        """
        emit = self.always if always else self.write
        if not rows:
            return
        cols = len(headers)
        aligns = list(aligns or ["l"] * cols)
        widths = [len(_strip(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row[:cols]):
                widths[i] = max(widths[i], len(_strip(str(cell))))

        def fit(cell: str, i: int) -> str:
            raw = _strip(cell)
            pad = widths[i] - len(raw)
            if aligns[i] == "r":
                return " " * pad + cell
            if aligns[i] == "c":
                half = pad // 2
                return " " * half + cell + " " * (pad - half)
            return cell + " " * pad

        bar = "─"
        emit(self.paint("  ┌" + "┬".join(bar * (w + 2) for w in widths) + "┐", "grey"))
        emit(
            self.paint("  │ ", "grey")
            + self.paint(" │ ", "grey").join(
                self.paint(fit(h, i), "bold") for i, h in enumerate(headers)
            )
            + self.paint(" │", "grey")
        )
        emit(self.paint("  ├" + "┼".join(bar * (w + 2) for w in widths) + "┤", "grey"))
        for row in rows:
            cells = [fit(str(c), i) for i, c in enumerate(row[:cols])]
            emit(
                self.paint("  │ ", "grey")
                + self.paint(" │ ", "grey").join(cells)
                + self.paint(" │", "grey")
            )
        emit(self.paint("  └" + "┴".join(bar * (w + 2) for w in widths) + "┘", "grey"))

    # ── progress ─────────────────────────────────────────────────────────────

    def bar(self, done: int, total: int, width: int = 28) -> str:
        """A sub-cell-resolution progress bar."""
        if total <= 0:
            return " " * width
        frac = max(0.0, min(1.0, done / total))
        exact = frac * width
        full = int(exact)
        rest = exact - full
        cell = _BLOCKS[int(rest * (len(_BLOCKS) - 1))] if full < width else ""
        body = "█" * full + cell
        return body + " " * max(width - len(_strip(body)), 0)

    def spinner(self, tick: int) -> str:
        return _SPINNER[tick % len(_SPINNER)]

    def live_block(self, lines: Sequence[str]) -> None:
        """Rewrite an in-place region.

        Falls back to nothing when the terminal cannot move the cursor —
        the caller is expected to also emit real lines for the log.
        """
        if not self.live or self.quiet:
            return
        out = []
        if self._live_lines:
            out.append(f"\x1b[{self._live_lines}A")
        for line in lines:
            out.append("\x1b[2K" + line[: self.width] + "\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._live_lines = len(lines)

    def live_done(self) -> None:
        """Leave the live region in place and stop tracking it."""
        self._live_lines = 0


def _strip(text: str) -> str:
    """Length of ``text`` as displayed, ignoring escape sequences."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "\x1b":
            j = text.find("m", i)
            if j == -1:
                break
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


class Timer:
    """Elapsed / rate / ETA for the progress line."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def eta(self, done: int, total: int) -> str:
        if done <= 0 or total <= 0:
            return "--:--"
        remaining = self.elapsed / done * (total - done)
        return fmt_duration(remaining)

    def __str__(self) -> str:
        return fmt_duration(self.elapsed)


def fmt_duration(seconds: float) -> str:
    """``m:ss`` below an hour, ``h:mm:ss`` above."""
    seconds = max(0.0, seconds)
    if seconds < 3600:
        return f"{int(seconds // 60):d}:{int(seconds % 60):02d}"
    return f"{int(seconds // 3600):d}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


def iter_ticks() -> Iterator[int]:
    """An endless counter for spinner frames."""
    tick = 0
    while True:
        yield tick
        tick += 1


#: A write-only ``/dev/null``, opened once.  ``Suppress`` is entered per
#: probed cell — over ten thousand times in a full sweep — and opening
#: and closing the same file that many times is work for nothing.
_DEVNULL_FD: int | None = None


def _devnull_fd() -> int:
    """int: A file descriptor for ``/dev/null``, shared by every Suppress."""
    global _DEVNULL_FD
    if _DEVNULL_FD is None:
        _DEVNULL_FD = os.open(os.devnull, os.O_WRONLY)
    return _DEVNULL_FD


class Suppress:
    """Silence stdout/stderr while probing.

    Ops under audit print warnings, and a survey that emits one line per
    op would bury its own progress display.

    Notes
    -----
    The redirection is at the **file-descriptor** level, not merely
    ``sys.stdout`` and ``sys.stderr``.  Rebinding those two names only
    silences Python; the libraries underneath write to descriptors 1 and
    2 themselves and never consult either.  Three of them did:

    * Accelerate's LAPACK prints its own argument complaints from the
      Fortran runtime — ``** On entry to DGESDD, parameter number 5 had
      an illegal value``;
    * so does its Hessenberg path, under ``func.linearize``;
    * MLX warns once that ``mx.metal.device_info`` is deprecated.

    Each landed in the middle of a live-progress frame, which the display
    then could not redraw over, leaving a stale half-drawn copy of itself
    in the scrollback.  Three writes, three stale frames — the same three
    at every terminal size, which is what showed it was not the display
    being too tall for the screen.

    Both levels are still swapped: a descriptor pointed at ``/dev/null``
    catches the C libraries, and rebinding the Python names keeps
    anything buffered in the real ``sys.stdout`` from being flushed into
    the terminal after the descriptor is restored.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._devnull: object | None = None
        self._saved: tuple[object, object] | None = None
        self._saved_fds: tuple[int, int] | None = None

    def __enter__(self) -> Suppress:
        if not self.enabled:
            return self
        sys.stdout.flush()
        sys.stderr.flush()
        null = _devnull_fd()
        self._saved_fds = (os.dup(1), os.dup(2))
        os.dup2(null, 1)
        os.dup2(null, 2)
        self._devnull = open(os.devnull, "w")
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = self._devnull  # type: ignore[assignment]
        sys.stderr = self._devnull  # type: ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self.enabled:
            return
        if self._saved is not None:
            sys.stdout, sys.stderr = self._saved  # type: ignore[assignment]
            self._saved = None
        if self._devnull is not None:
            self._devnull.close()  # type: ignore[attr-defined]
            self._devnull = None
        if self._saved_fds is not None:
            out_fd, err_fd = self._saved_fds
            os.dup2(out_fd, 1)
            os.dup2(err_fd, 2)
            os.close(out_fd)
            os.close(err_fd)
            self._saved_fds = None


__all__ = ["Console", "Suppress", "Timer", "fmt_duration", "iter_ticks", "supports_colour"]
