"""Shared compile machinery for :mod:`lucid.compile` (internal).

* :mod:`~lucid.compile._core.signature` — input-signature / cache-key derivation.
* :mod:`~lucid.compile._core.module_state` — which module attributes the key
  is taken over.
* :mod:`~lucid.compile._core.fallback` — eager-escape helpers for ops/shapes
  the graph path cannot capture.
* :mod:`~lucid.compile._core.bn_runstats` — BatchNorm running-stats plumbing
  shared across the entry points.
"""
