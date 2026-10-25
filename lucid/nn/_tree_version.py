"""Compatibility alias for :mod:`lucid._tree_version`."""

import sys

import lucid._tree_version as _tree_version_mod
from lucid._tree_version import _TrackedDict, _tree_changed, _tree_version

__all__ = ["_TrackedDict", "_tree_changed", "_tree_version"]

sys.modules[__name__] = _tree_version_mod
