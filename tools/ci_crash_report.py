#!/usr/bin/env python3
"""Parse macOS ``.ips`` crash reports written during a CI job (LCD-1022).

Prints the faulting thread's top frames (always including ``imageOffset``),
prints the main thread's top frames when a non-main thread faulted (so
exit-time teardown races such as ``exit`` / ``__cxa_finalize`` show up in
the CI log), and copies every ``.ips`` written during the job into a staging
directory for ``actions/upload-artifact``.
"""

import argparse
import json
import shutil
import stat
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeGuard

DEFAULT_SEARCH_DIRS: tuple[Path, ...] = (
    Path.home() / "Library/Logs/DiagnosticReports",
    Path("/Library/Logs/DiagnosticReports"),
)
DEFAULT_FAULTING_FRAMES = 30
DEFAULT_MAIN_FRAMES = 15


def _is_int(value: object) -> TypeGuard[int]:
    """True for plain integers, excluding ``bool`` (a Python subclass of ``int``)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _format_frame(frame: dict[str, Any], images: Sequence[dict[str, Any]]) -> str:
    """Format one stack frame with image name, symbol (if any), and hex ``imageOffset``."""
    index = frame.get("imageIndex")
    image = (
        str(images[index].get("name") or "?")
        if _is_int(index) and 0 <= index < len(images)
        else "?"
    )
    raw_offset = frame.get("imageOffset")
    offset = f"0x{raw_offset:x}" if _is_int(raw_offset) else "?"
    symbol = frame.get("symbol")
    sym_loc = frame.get("symbolLocation")
    if isinstance(symbol, str) and symbol:
        if _is_int(sym_loc):
            return f"  {image:36s} {symbol} + {sym_loc} (imageOffset={offset})"
        return f"  {image:36s} {symbol} (imageOffset={offset})"
    return f"  {image:36s} imageOffset={offset}"


def _thread_label(thread: dict[str, Any]) -> str:
    parts: list[str] = []
    name = thread.get("name")
    queue = thread.get("queue")
    if isinstance(name, str) and name:
        parts.append(f"name={name}")
    if isinstance(queue, str) and queue:
        parts.append(f"queue={queue}")
    return ", ".join(parts)


def _find_main_thread(threads: Sequence[dict[str, Any]]) -> int:
    for idx, thread in enumerate(threads):
        if thread.get("queue") == "com.apple.main-thread":
            return idx
    return 0


def format_report(
    text: str,
    *,
    faulting_frames: int = DEFAULT_FAULTING_FRAMES,
    main_frames: int = DEFAULT_MAIN_FRAMES,
) -> str:
    """Format a macOS v2 JSON ``.ips`` report for CI logs."""
    header_raw, sep, body_raw = text.partition("\n")
    if not sep or not body_raw.strip():
        return _fallback_head(text)
    try:
        header = json.loads(header_raw)
        report = json.loads(body_raw)
    except json.JSONDecodeError:
        return _fallback_head(text)
    if not isinstance(header, dict) or not isinstance(report, dict):
        return _fallback_head(text)

    lines: list[str] = [f"process: {header.get('app_name')} | {report.get('exception')}"]
    termination = report.get("termination")
    asi = report.get("asi")
    if termination is not None or asi is not None:
        lines.append(f"termination: {termination} | asi: {asi}")

    images_raw = report.get("usedImages", [])
    images: list[dict[str, Any]] = (
        [img for img in images_raw if isinstance(img, dict)] if isinstance(images_raw, list) else []
    )

    threads_raw = report.get("threads")
    if not isinstance(threads_raw, list) or not threads_raw:
        # A JetsamEvent or similar: a report, but not of a crash.
        lines.append("not a crash report — no threads to unwind")
        return "\n".join(lines)
    threads: list[dict[str, Any]] = [t for t in threads_raw if isinstance(t, dict)]
    if not threads:
        lines.append("not a crash report — no threads to unwind")
        return "\n".join(lines)

    raw_fault = report.get("faultingThread", 0)
    fault_idx = raw_fault if _is_int(raw_fault) and 0 <= raw_fault < len(threads) else 0
    main_idx = _find_main_thread(threads)

    fault_thread = threads[fault_idx]
    fault_meta = _thread_label(fault_thread)
    is_main = fault_idx == main_idx
    suffix_parts = ([fault_meta] if fault_meta else []) + (["main thread"] if is_main else [])
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    lines.append(f"faulting thread {fault_idx}{suffix}:")
    for frame in _thread_frames(fault_thread)[:faulting_frames]:
        lines.append(_format_frame(frame, images))

    if not is_main:
        main_thread = threads[main_idx]
        main_meta = _thread_label(main_thread)
        main_suffix = f" ({main_meta})" if main_meta else ""
        lines.append(f"main thread {main_idx}{main_suffix}:")
        for frame in _thread_frames(main_thread)[:main_frames]:
            lines.append(_format_frame(frame, images))

    return "\n".join(lines)


def _thread_frames(thread: dict[str, Any]) -> list[dict[str, Any]]:
    raw = thread.get("frames", [])
    return [f for f in raw if isinstance(f, dict)] if isinstance(raw, list) else []


def _fallback_head(text: str, limit: int = 120) -> str:
    return "\n".join(text.splitlines()[:limit])


def find_reports(search_dirs: Sequence[Path], *, since_ns: int | None = None) -> list[Path]:
    """Return regular ``.ips`` files under ``search_dirs`` newer than ``since_ns``, oldest first."""
    seen: set[Path] = set()
    found: list[tuple[int, str, Path]] = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for candidate in directory.glob("*.ips"):
            try:
                resolved = candidate.resolve()
                st = candidate.stat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if since_ns is not None and st.st_mtime_ns < since_ns:
                continue
            found.append((st.st_mtime_ns, str(candidate), candidate))
    found.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in found]


def collect_reports(
    search_dirs: Sequence[Path],
    *,
    since: Path | None = None,
    wait_seconds: float = 30.0,
    poll_interval: float = 5.0,
) -> list[Path]:
    """Poll ``search_dirs`` until at least one matching ``.ips`` arrives or timeout elapses.

    If ``since`` is supplied but does not exist (for example, the job failed before
    creating its start marker), returns an empty list rather than sweeping unrelated
    pre-existing reports on the runner.
    """
    since_ns: int | None = None
    if since is not None:
        try:
            since_ns = since.stat().st_mtime_ns
        except OSError:
            print(f"crash-report marker missing ({since}); skipping pre-existing reports")
            return []

    wait_s = max(wait_seconds, 0.0)
    deadline = time.monotonic() + wait_s
    first_poll = True
    while True:
        reports = find_reports(search_dirs, since_ns=since_ns)
        if reports:
            # If a report was already present on the first poll (e.g. an earlier crash in
            # the same job), give ReportCrash one settle window to flush a final crash too.
            if first_poll and wait_s > 0.0 and poll_interval > 0.0:
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(min(poll_interval, remaining))
                    reports = find_reports(search_dirs, since_ns=since_ns)
            return reports
        first_poll = False
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or poll_interval <= 0.0:
            return []
        time.sleep(min(poll_interval, remaining))


def stage_reports(reports: Sequence[Path], dest_dir: Path) -> list[Path]:
    """Copy readable ``reports`` into ``dest_dir``, disambiguating colliding basenames."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    used: set[str] = set()
    for src in reports:
        name = src.name
        if name in used:
            stem = src.stem
            suffix = src.suffix
            counter = 1
            while f"{stem}_{counter}{suffix}" in used:
                counter += 1
            name = f"{stem}_{counter}{suffix}"
        dst = dest_dir / name
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            print(f"could not stage {src}: {exc}")
            continue
        used.add(name)
        staged.append(dst)
    return staged


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print and stage macOS .ips crash reports.")
    parser.add_argument(
        "--since",
        type=Path,
        default=None,
        help="Only include .ips files modified at or after this marker file.",
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="Copy every matched .ips report into this directory for artifact upload.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for ReportCrash to flush a .ips file.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between DiagnosticReports polls while waiting.",
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        action="append",
        default=None,
        help="Override directories to search (repeatable; defaults to DiagnosticReports).",
    )
    args = parser.parse_args(argv)

    search_dirs: Sequence[Path] = args.search_dir or DEFAULT_SEARCH_DIRS
    reports = collect_reports(
        search_dirs,
        since=args.since,
        wait_seconds=args.wait_seconds,
        poll_interval=args.poll_interval,
    )
    if not reports:
        print("no crash report was written")
        return 0

    for report_path in reports:
        print(f"== {report_path}")
        try:
            text = report_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"could not read {report_path}: {exc}")
            continue
        print(format_report(text))

    if args.copy_to is not None:
        staged = stage_reports(reports, args.copy_to)
        print(f"staged {len(staged)} crash report(s) into {args.copy_to}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
