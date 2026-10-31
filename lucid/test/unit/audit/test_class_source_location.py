"""Every public class is found where its source says it is.

A class whose ``__module__`` names a module other than the one that defines it
sends ``inspect`` to the wrong file: since 3.13 the class's ``__firstlineno__``
is read against that other file, so ``inspect.getsource`` returns the wrong
lines or fails to tokenize them (``ReduceLROnPlateau`` did, until it stopped
naming ``lucid.optim.lr_scheduler`` as its module).
"""

import inspect
import sys

import pytest

from lucid.test.audit._surface import enumerate_surface


def _public_classes() -> list[tuple[str, type]]:
    seen: dict[int, tuple[str, type]] = {}
    for symbol in enumerate_surface():
        obj = symbol.obj
        if isinstance(obj, type) and "__firstlineno__" in vars(obj):
            seen.setdefault(id(obj), (symbol.qualname, obj))
    return sorted(seen.values())


@pytest.mark.parametrize(
    ("qualname", "cls"), _public_classes(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_a_public_class_is_defined_in_the_module_it_names(qualname: str, cls: type) -> None:
    module = sys.modules[cls.__module__]
    lines = inspect.getsource(module).splitlines()
    first = lines[vars(cls)["__firstlineno__"] - 1].lstrip()
    assert first.startswith(("class ", "@")), (
        f"{qualname}: line {vars(cls)['__firstlineno__']} of {cls.__module__} is {first!r}, "
        "not its class statement; its __module__ names a module that does not define it"
    )
