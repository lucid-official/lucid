"""
lucid.autograd.profiler — namespace alias to ``lucid.profiler``.

The reference framework places its profile context manager under
``autograd.profiler.profile``; Lucid's primary entry point lives
at ``lucid.profiler.profile``.  This module re-exports it so existing
code that uses the ``autograd.profiler.profile`` path keeps working
without forcing a redundant copy of the context manager.
"""

from lucid.profiler import MemoryStats, OpEvent, ProfileSummary, Profiler, profile, record_function

__all__ = ["profile", "record_function", "Profiler", "ProfileSummary", "OpEvent", "MemoryStats"]
