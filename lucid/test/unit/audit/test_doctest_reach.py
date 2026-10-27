"""The doctest stage reaches methods built in a local scope (LCD-316).

``doctest.DocTestFinder`` keeps a class's function only when its globals are
the class's module, so every ``Tensor`` dunder (made inside
``_inject_dunders``) was invisible and ``doctest.json`` read 0 failing.
"""

import importlib

import lucid
from lucid.test.audit import _doctests


def test_installed_methods_include_the_tensor_dunders_and_to() -> None:
    tensor_mod = importlib.import_module("lucid._tensor.tensor")
    found = dict(_doctests._installed_methods([tensor_mod]))
    for name in ("__getitem__", "__setitem__", "__add__", "to", "cpu"):
        assert f"lucid._tensor.tensor.Tensor.{name}" in found, name
    assert found["lucid._tensor.tensor.Tensor.__getitem__"] is vars(lucid.Tensor)["__getitem__"]


def test_installed_examples_count_under_their_defining_module() -> None:
    tensor_mod = importlib.import_module("lucid._tensor.tensor")
    per_module: dict[str, int] = {}
    failed, attempted = _doctests._run_installed([tensor_mod], per_module)
    assert attempted > 0
    assert failed == sum(per_module.values())
    assert set(per_module) <= {"lucid._tensor._dunders", "lucid._tensor._to"}
