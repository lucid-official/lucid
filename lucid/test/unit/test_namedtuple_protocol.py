"""Every ``NamedTuple`` defined under ``lucid`` keeps the tuple protocol intact.

``collections.namedtuple._make`` (hence ``_replace``), ``copy`` and pickling
all trust ``len(record)`` to equal the number of fields.  A subclass that
overrides ``__len__`` / ``__iter__`` / ``__getitem__`` breaks them on the
Python versions whose stdlib checks through the builtin (3.14.0 and 3.14.1).
"""

import importlib
import pkgutil
import sys

import pytest

import lucid


def _namedtuple_classes() -> list[type[tuple[object, ...]]]:
    for info in pkgutil.walk_packages(lucid.__path__, "lucid."):
        if ".test" in info.name or "._C" in info.name or info.name.endswith("__main__"):
            continue
        importlib.import_module(info.name)
    found: dict[str, type[tuple[object, ...]]] = {}
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("lucid") or ".test" in module_name:
            continue
        for value in list(vars(module).values()):
            if (
                isinstance(value, type)
                and issubclass(value, tuple)
                and hasattr(value, "_fields")
                and value.__module__.startswith("lucid")
            ):
                found[f"{value.__module__}.{value.__qualname__}"] = value
    return [found[name] for name in sorted(found)]


_CLASSES = _namedtuple_classes()


def test_walk_finds_the_records() -> None:
    assert len(_CLASSES) > 20


@pytest.mark.parametrize("cls", _CLASSES, ids=lambda c: f"{c.__module__}.{c.__qualname__}")
@pytest.mark.parametrize("method", ["__len__", "__iter__", "__getitem__"])
def test_namedtuple_does_not_override_tuple_protocol(
    cls: type[tuple[object, ...]], method: str
) -> None:
    assert getattr(cls, method) is getattr(tuple, method), (
        f"{cls.__qualname__} overrides {method}; a record's tuple protocol "
        "must match its fields (use a named property instead)"
    )
