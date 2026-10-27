"""Every public callable can say what it takes.

``inspect.signature`` is not a typing nicety. It is what ``help()``
prints, what an editor shows in a tooltip, and what the documentation
build reads to produce a page's parameter table. A callable it cannot
read has no signature anywhere those look.

Under PEP 649 a function's annotations are evaluated, on demand, in that
function's own module globals. A name imported only inside an
``if TYPE_CHECKING:`` block is therefore absent at the moment somebody
asks — and ``Tensor`` is the return type of most of this package. A
third of ``lucid.__all__`` was raising ``NameError`` when asked, silently
enough that nothing noticed until an export needed one factory's
signature.

That rule once covered ``lucid.__all__`` alone, by binding ``Tensor`` at the
foot of each file; 297 callables across ``nn``, ``compile``,
``quantization``, ``models`` and the rest still raised.  Now
:mod:`lucid._annotations` binds every name a module imports under
``TYPE_CHECKING`` once the module has finished executing, from the table
at that file's foot which ``tools/gen_annotation_names.py`` writes — and the
whole surface is held here, not only the top level.
"""

import annotationlib
import inspect
import pkgutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import lucid

#: One that stays unreadable, for a reason that is not an oversight.
#:
#: ``from_numpy`` annotates a numpy type, and numpy is a dependency of
#: the bridge rather than of the package: ``converters`` imports it
#: inside the one function that needs it so a Lucid install without
#: numpy still works. Binding it at module scope to make this signature
#: readable would turn an optional import into a required one, which is
#: the worse trade.
#:
#: A test below asserts the reason still holds, so the entry cannot
#: outlive it unnoticed.
NO_PYTHON_SIGNATURE = {"from_numpy"}


def _public_callables() -> list[str]:
    """Names in ``lucid.__all__`` that are callable.

    ``dir(lucid)`` is not the list: the package resolves most of its
    surface through ``__getattr__``, so the names that matter are absent
    from it until something asks for them — which is exactly how this
    class of defect stayed invisible.
    """
    found = []
    for name in lucid.__all__:
        try:
            value = getattr(lucid, name)
        except Exception:  # noqa: BLE001 — an unreachable name is another test's
            continue
        if callable(value):
            found.append(name)
    return found


@pytest.mark.parametrize("name", [n for n in _public_callables() if n not in NO_PYTHON_SIGNATURE])
def test_a_public_callable_has_a_readable_signature(name: str) -> None:
    inspect.signature(getattr(lucid, name))


def test_the_exemptions_are_still_public_callables() -> None:
    """An entry for a name that left the surface is a stale entry."""
    assert NO_PYTHON_SIGNATURE <= set(_public_callables())


def test_generator_reads_as_a_python_class() -> None:
    """``Generator`` was the bound C++ class, which ``inspect`` cannot
    describe; the Python subclass that carries the reference's methods
    (LCD-374) reads like any other callable, so it left the exemptions."""
    assert list(inspect.signature(lucid.Generator).parameters) == ["seed", "device"]


def test_from_numpy_is_unreadable_only_through_its_numpy_annotation() -> None:
    """The one thing missing is the lazily imported ``np``.

    Evaluating the annotations fails on that name and nothing else: asked
    for forward references instead of values, the signature reads.
    """
    with pytest.raises(NameError, match="'np' is not defined"):
        inspect.signature(lucid.from_numpy)
    signature = inspect.signature(
        lucid.from_numpy, annotation_format=annotationlib.Format.FORWARDREF
    )
    assert list(signature.parameters) == ["ndarray"]


def test_the_tensor_constructor_is_readable() -> None:
    """The class this package is named for, in particular.

    ``Tensor.__init__`` annotates ``dtype`` and ``device`` — and the
    class body defines properties of those names. PEP 649 evaluates the
    annotations with the class namespace in scope, so the bare names
    resolved to the properties and the signature raised ``TypeError``.
    The parameters are spelled through aliases now.
    """
    signature = inspect.signature(lucid.Tensor)
    assert "dtype" in signature.parameters
    assert "device" in signature.parameters


def test_a_module_that_annotates_tensor_binds_it() -> None:
    """The rule stated as a check, so a new module inherits it.

    A module can annotate ``Tensor`` without importing it at runtime and
    look perfectly correct — until something reads a signature. This
    walks the modules the public surface actually comes from.
    """
    import sys

    missing = []
    for name in _public_callables():
        if name in NO_PYTHON_SIGNATURE:
            continue
        value = getattr(lucid, name)
        module = sys.modules.get(getattr(value, "__module__", ""))
        if module is None or not module.__name__.startswith("lucid."):
            continue
        annotates = "Tensor" in str(getattr(value, "__annotations__", {}))
        if annotates and not hasattr(module, "Tensor"):
            missing.append(f"{module.__name__}.{name}")
    assert not missing, (
        "these modules annotate Tensor but do not bind it at runtime, so "
        f"inspect.signature cannot read them: {sorted(set(missing))}"
    )


_REPO = Path(__file__).resolve().parents[3]


def test_every_public_callable_in_every_subpackage_has_a_readable_signature() -> None:
    """The whole surface the audit walks, not only ``lucid.__all__``."""
    from lucid.test.audit import _surface

    unreadable = []
    for symbol in _surface.enumerate_surface():
        value = _surface.resolve(symbol)
        if value is None or not callable(value):
            continue
        if symbol.qualname.rsplit(".", 1)[-1] in NO_PYTHON_SIGNATURE:
            continue
        try:
            inspect.signature(value)
        except NameError as exc:
            unreadable.append(f"{symbol.qualname}: {exc}")
        except (ValueError, TypeError):
            pass  # a builtin or C type: not something an annotation fixes
    assert not unreadable, "\n".join(unreadable)


def test_the_annotation_table_is_fresh() -> None:
    """``NAMES`` in ``lucid/_annotations.py`` matches the ``TYPE_CHECKING`` blocks."""
    result = subprocess.run(
        [sys.executable, "-m", "tools.gen_annotation_names", "--check"],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_annotation_table_check_reads_only_the_generated_region() -> None:
    """A stale table is caught; an edit to the hand-written resolver is not.

    The table shares ``lucid/_annotations.py`` with the resolver, so the
    check must compare the region between the markers and nothing else —
    and must refuse a file whose markers are missing or doubled rather than
    rewrite the wrong lines.
    """
    from tools import gen_annotation_names as gen

    text = gen.OUT.read_text(encoding="utf-8")
    fresh = gen.render(gen.collect())
    assert gen.region(text) == fresh

    resolver_edited = text.replace('"""Bind the names', '"""Binds the names', 1)
    assert resolver_edited != text
    assert gen.region(resolver_edited) == fresh

    first_row = next(line for line in fresh.splitlines() if line.startswith("    "))
    stale = gen.splice(text, fresh.replace(first_row + "\n", "", 1))
    assert gen.region(stale) != fresh
    assert gen.splice(stale, fresh) == text

    for broken in (
        text.replace(gen.END, ""),
        text.replace(gen.BEGIN, ""),
        text + gen.BEGIN + "\n",
        text.replace(gen.BEGIN, "\0").replace(gen.END, gen.BEGIN).replace("\0", gen.END),
    ):
        with pytest.raises(gen.MarkerError):
            gen.region(broken)


def test_binding_names_imports_nothing_the_package_defers() -> None:
    """Readable signatures, at no cost to ``import lucid`` or to numpy.

    ``lucid.nn`` stays unloaded by ``import lucid`` alone, and reading every
    ``lucid.nn`` signature never imports numpy — a dependency of the bridge,
    not of the package.
    """
    code = (
        "import sys, inspect, lucid\n"
        "assert 'lucid.nn' not in sys.modules, 'import lucid now loads lucid.nn'\n"
        "import lucid.nn as nn\n"
        "for name in dir(nn):\n"
        "    obj = getattr(nn, name)\n"
        "    if callable(obj):\n"
        "        try: inspect.signature(obj)\n"
        "        except (ValueError, TypeError): pass\n"
        "assert 'numpy' not in sys.modules, 'reading a signature imported numpy'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO)
    assert result.returncode == 0, result.stderr[-2000:]


def test_importing_any_subpackage_first_leaves_every_package_whole() -> None:
    """Binding names never leaves a half-imported package behind.

    The resolver imports a module's annotation sources for it — and once
    did so while an enclosing module was still executing: ``import
    lucid.nn`` loaded the rollout driver, whose ``RSSM`` source needed
    ``lucid.nn.GELU`` before ``lucid.nn`` had it.  The failure was
    swallowed, as an annotation must never break an import, but it took
    ``lucid.models`` out of ``sys.modules`` and left thirteen of its
    submodules in, and the next ``from lucid.models.generative.vjepa2_ac
    import ...`` failed as a circular import.  Only a fresh interpreter
    whose first import is the subpackage shows it — the nightly model-zoo
    job, one process per file, found it in two files.  One interpreter
    per subpackage, run side by side.
    """
    subpackages = sorted(
        module.name
        for module in pkgutil.iter_modules(lucid.__path__, "lucid.")
        if module.ispkg and not module.name.startswith(("lucid._", "lucid.test"))
    )

    def first(name: str) -> str | None:
        code = (
            "import importlib, sys\n"
            f"importlib.import_module({name!r})\n"
            "orphans = sorted(\n"
            "    n for n in sys.modules\n"
            "    if n.startswith('lucid.') and n.rpartition('.')[0] not in sys.modules\n"
            ")\n"
            "assert not orphans, f'packages missing under {orphans}'\n"
            "import lucid.models.generative.vjepa2_ac\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO
        )
        if result.returncode == 0:
            return None
        return f"import {name} first:\n{result.stderr[-1500:]}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        broken = [why for why in pool.map(first, subpackages) if why]
    assert len(subpackages) > 15, subpackages
    assert not broken, "\n\n".join(broken)


def test_every_tensor_method_and_property_resolves_annotations_without_method_shadowing() -> None:
    """PEP 649 annotations on ``Tensor`` resolve ``int``, ``bool``, and ``float``
    to the builtins rather than ``Tensor.int``, ``Tensor.bool``, or ``Tensor.float``
    (LCD-442)."""
    import types
    import typing

    tensor_fns = {
        getattr(lucid.Tensor, n)
        for n in dir(lucid.Tensor)
        if isinstance(getattr(lucid.Tensor, n), types.FunctionType)
    }

    def contains_tensor_fn(ann: object) -> bool:
        if isinstance(ann, types.FunctionType) and ann in tensor_fns:
            return True
        if isinstance(ann, list):
            return any(contains_tensor_fn(a) for a in ann)
        return any(contains_tensor_fn(arg) for arg in typing.get_args(ann))

    failed: list[str] = []
    shadowed: list[str] = []
    for name in dir(lucid.Tensor):
        if name == "__class__":
            continue
        attr = getattr(lucid.Tensor, name)
        targets = [(name, attr)] if callable(attr) else []
        if isinstance(attr, property):
            for pfn_name in ("fget", "fset"):
                pfn = getattr(attr, pfn_name, None)
                if pfn is not None:
                    targets.append((f"{name}.{pfn_name}", pfn))
        for label, fn in targets:
            try:
                inspect.signature(fn)
                hints = typing.get_type_hints(fn)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{label}: {type(exc).__name__}: {exc}")
                continue
            for param_name, hint in hints.items():
                if contains_tensor_fn(hint):
                    shadowed.append(f"{label}.{param_name}: {hint!r}")
    assert not failed, "\n".join(failed)
    assert not shadowed, "\n".join(shadowed)
    assert typing.get_type_hints(lucid.Tensor.data_ptr) == {"return": int}
    assert typing.get_type_hints(lucid.Tensor.stride)["return"] == tuple[int, ...] | int
