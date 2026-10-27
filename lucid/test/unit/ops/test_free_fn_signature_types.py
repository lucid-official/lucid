"""A registry free function's runtime signature speaks ``Tensor`` (LCD-547).

``inspect.signature(lucid.contiguous)`` read ``(input: TensorImpl) ->
TensorImpl`` — the adapter's engine types — for 70 functions, where the
stub and every caller see ``Tensor``: ``help()``, an editor's tooltip and
the docs build all showed a type no user holds.
"""

import inspect

import pytest

import lucid
import lucid._ops as ops
from lucid._C import engine as _C_engine


def _mentions_impl(annotation: object) -> bool:
    return annotation is _C_engine.TensorImpl or "TensorImpl" in repr(annotation)


@pytest.mark.parametrize("name", sorted(ops._FREE_FN_NAMES))
def test_no_annotation_names_the_engine_type(name: str) -> None:
    sig = inspect.signature(getattr(lucid._ops, name))
    leaked = [p.name for p in sig.parameters.values() if _mentions_impl(p.annotation)]
    if _mentions_impl(sig.return_annotation):
        leaked.append("return")
    assert leaked == []


def test_the_retyped_signatures_keep_their_shape() -> None:
    sig = inspect.signature(lucid.nonzero)
    assert list(sig.parameters) == ["input", "as_tuple"]
    assert sig.return_annotation == lucid.Tensor | tuple[lucid.Tensor, ...]
    assert inspect.signature(lucid.contiguous).parameters["input"].annotation is lucid.Tensor
    assert inspect.signature(lucid.split).return_annotation == list[lucid.Tensor]
