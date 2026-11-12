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
whole surface is held here, not only the top level: every public class's
``__init__`` and public methods and properties, the model zoo included.

When that table is stale, or a name cannot be bound, the signature raises a
bare ``NameError`` that names neither the callable nor the parameter.  The
guard for the whole surface reports, per signature, its qualified name, the
parameter, the unresolved name and the fix.  It is a unit test rather than a
commit-level rule: the static half (a ``TYPE_CHECKING`` import the table does
not list) is ``.githooks/pre-commit`` 1a already, and what is left — a name
the resolver cannot bind — can only be seen by evaluating the signature.
"""

import annotationlib
import ast
import functools
import inspect
import pkgutil
import subprocess
import sys
import types
import typing
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import lucid
from lucid._annotations import NAMES

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


Fn = Callable[..., object]


def _accessors(raw: object) -> list[tuple[str, Fn]]:
    """The functions behind one class attribute, each with the suffix that names it."""
    if isinstance(raw, (staticmethod, classmethod)):
        return [("", raw.__func__)]
    if isinstance(raw, property):
        return [(f".{kind}", fn) for kind in ("fget", "fset") if (fn := getattr(raw, kind))]
    if isinstance(raw, functools.cached_property):
        return [("", raw.func)]
    return [("", raw)] if isinstance(raw, types.FunctionType) else []


def _defined_here(fn: Fn) -> bool:
    """Whether ``fn`` is Lucid's own, not a method a base class took from the standard library."""
    module = getattr(fn, "__module__", None) or ""
    return module == "lucid" or module.startswith("lucid.")


def _targets(label: str, obj: object) -> list[tuple[str, Fn]]:
    """What reading ``obj``'s public signatures evaluates: itself and, for a class, its methods.

    A class reads through its ``__init__``.  Its public methods and properties
    are separate callables with annotations of their own, which nothing but a
    caller of that method evaluates, until a documentation build does.
    """
    if not inspect.isclass(obj):
        found = [(label + suffix, fn) for suffix, fn in _accessors(obj)]
        return found or ([(label, obj)] if callable(obj) else [])
    found = [(label, obj)]
    for name in dir(obj):
        if name.startswith("_"):
            continue
        for suffix, fn in _accessors(inspect.getattr_static(obj, name, None)):
            if _defined_here(fn):
                found.append((f"{label}.{name}{suffix}", fn))
    return found


def _public_signatures() -> Iterator[tuple[str, Fn]]:
    """``(label, callable)`` for every public signature of the released surface, once each.

    The symbols ``lucid.test.audit`` enumerates, which leave the model zoo to a
    stage of its own, then ``lucid.models.__all__``; for each class, its
    ``__init__``, methods and properties.  A function reached under two names
    is read once, under the first.
    """
    import lucid.models as models
    from lucid.test.audit import _surface

    roots = [(symbol.qualname, symbol.obj) for symbol in _surface.enumerate_surface()]
    roots += [(f"lucid.models.{name}", getattr(models, name, None)) for name in models.__all__]
    seen: set[int] = set()
    for label, obj in roots:
        for target_label, target in _targets(label, obj):
            if id(target) not in seen:
                seen.add(id(target))
                yield target_label, target


def _function_of(target: Fn) -> Fn:
    """The function whose annotations ``inspect.signature(target)`` reads."""
    return getattr(target, "__init__", target) if inspect.isclass(target) else target


def _unresolved(function: Fn) -> dict[str, list[str]]:
    """Per parameter (``return`` too), the names its annotation uses that do not exist.

    Read as forward references, which is what ``inspect.signature`` declines to
    do: a name the module lacks comes back as a ``ForwardRef``, and the names
    that did resolve come back as placeholders, so what is left in the text is
    what is missing.
    """
    try:
        annotations = annotationlib.get_annotations(
            function, format=annotationlib.Format.FORWARDREF
        )
    except TypeError:
        return {}

    def refs(value: object) -> Iterator[annotationlib.ForwardRef]:
        if isinstance(value, annotationlib.ForwardRef):
            yield value
        for part in value if isinstance(value, list) else typing.get_args(value):
            yield from refs(part)

    found: dict[str, list[str]] = {}
    for parameter, annotation in annotations.items():
        for ref in refs(annotation):
            for node in ast.walk(ast.parse(ref.__forward_arg__, mode="eval")):
                if isinstance(node, ast.Name) and not node.id.startswith("__annotationlib_name_"):
                    found.setdefault(parameter, []).append(node.id)
    return found


@functools.cache
def _imported_for_type_checkers() -> dict[str, tuple[str, ...]]:
    """Module -> the names it imports only under ``TYPE_CHECKING``, read from the source."""
    from tools import gen_annotation_names

    return {
        module: tuple(bound for bound, _, _ in entries)
        for module, entries in gen_annotation_names.collect().items()
    }


def _remedy(name: str, module: str, *, listed: bool, imported: bool) -> str:
    """What to do about ``name``, which ``module`` annotates with and does not define."""
    regenerate = "python -m tools.gen_annotation_names"
    if imported and not listed:
        return (
            f"{module} imports {name} under TYPE_CHECKING and the table in "
            f"lucid/_annotations.py does not list it yet: run `{regenerate}`"
        )
    if listed:
        return (
            f"the table lists {name} for {module} but its source did not bind it: "
            f"import {name} at runtime in {module}, outside TYPE_CHECKING, "
            f"then run `{regenerate}`"
        )
    return (
        f"{module} does not import {name}: import {name} at runtime, "
        f"then run `{regenerate}` (a TYPE_CHECKING import needs the table row it writes)"
    )


def _excused(label: str, names: set[str]) -> bool:
    """Whether ``np`` is all that is missing, on a signature this file declares unreadable for it."""
    return label.rsplit(".", 1)[-1] in NO_PYTHON_SIGNATURE and names <= {"np"}


def _describe(label: str, function: Fn, missing: dict[str, list[str]]) -> list[str]:
    module = getattr(function, "__module__", None) or "?"
    listed = {bound for bound, _, _ in NAMES.get(module, ())}
    imported = _imported_for_type_checkers().get(module, ())
    lines = []
    for parameter, names in missing.items():
        where = {"": "an annotation", "return": "the return annotation"}.get(
            parameter, f"parameter {parameter!r}"
        )
        for name in dict.fromkeys(names):
            fix = _remedy(name, module, listed=name in listed, imported=name in imported)
            lines.append(
                f"{label}: {where} names {name!r}, which {module} does not define at run time"
                f"\n    fix: {fix}"
            )
    return lines


def _unreadable(targets: Iterable[tuple[str, Fn]]) -> list[str]:
    """A report for each target whose annotations raise ``NameError``: where, which, and the fix."""
    reports: list[str] = []
    for label, target in targets:
        try:
            inspect.signature(target)
        except NameError as exc:
            function = _function_of(target)
            missing = _unresolved(function) or {"": [exc.name or "?"]}
            if not _excused(label, {name for names in missing.values() for name in names}):
                reports += _describe(label, function, missing)
        except (ValueError, TypeError):
            pass  # a builtin or C type: not something an annotation fixes
    return reports


def test_every_public_signature_of_the_released_surface_is_readable() -> None:
    """The whole surface the audit walks and the model zoo, methods included.

    ``inspect.signature`` raises a bare ``NameError`` that names neither the
    callable nor the parameter; the report here does, and says what to run.
    """
    reports = _unreadable(_public_signatures())
    assert not reports, (
        f"{len(reports)} public annotation(s) name something that does not exist at run time, "
        "so inspect.signature raises NameError and help(), editors and the docs build "
        "cannot read the callable:\n" + "\n".join(reports)
    )


def _function(source: str) -> Fn:
    """The function ``f`` that ``source`` defines, in a made-up module of the package."""
    namespace: dict[str, object] = {"__name__": "lucid.scratch", "Callable": Callable}
    exec(source, namespace)
    function = namespace["f"]
    assert callable(function)
    return function


def test_the_report_names_the_callable_the_parameter_the_name_and_the_fix() -> None:
    function = _function(
        "def f(self, size: int, config: Gone | None = None, *, hook: Callable[[Hook], Out]"
        " = None, table: dict[str, Row | int] = {}) -> Result[int]: ..."
    )
    reports = _unreadable([("lucid.scratch.Layer.route", function)])
    assert len(reports) == 5, reports
    expected = [
        ("parameter 'config'", "'Gone'"),
        ("parameter 'hook'", "'Hook'"),
        ("parameter 'hook'", "'Out'"),
        ("parameter 'table'", "'Row'"),
        ("the return annotation", "'Result'"),
    ]
    for report, (where, name) in zip(reports, expected, strict=True):
        assert report.startswith("lucid.scratch.Layer.route: ")
        assert where in report and name in report, report
        assert "lucid.scratch does not define" in report
        assert "`python -m tools.gen_annotation_names`" in report
    assert not any("'size'" in report for report in reports)


@pytest.mark.parametrize(
    ("listed", "imported", "says"),
    [
        (False, True, "does not list it yet: run `python -m tools.gen_annotation_names`"),
        (True, False, "did not bind it: import Gone at runtime in lucid.scratch"),
        (False, False, "lucid.scratch does not import Gone: import Gone at runtime"),
    ],
    ids=["stale table", "unbindable source", "never imported"],
)
def test_the_fix_follows_how_the_module_comes_to_lack_the_name(
    monkeypatch: pytest.MonkeyPatch, listed: bool, imported: bool, says: str
) -> None:
    monkeypatch.setattr(
        f"{__name__}._imported_for_type_checkers",
        lambda: {"lucid.scratch": ("Gone",)} if imported else {},
    )
    if listed:
        monkeypatch.setitem(NAMES, "lucid.scratch", (("Gone", "lucid.nn", "Module"),))
    [report] = _unreadable([("lucid.scratch.f", _function("def f(x: Gone) -> None: ..."))])
    assert says in report, report
    assert "python -m tools.gen_annotation_names" in report


def test_a_class_is_read_through_its_constructor_methods_and_properties() -> None:
    namespace: dict[str, object] = {"__name__": "lucid.scratch"}
    exec(
        "class Layer:\n"
        "    def __init__(self, base: InInit) -> None: ...\n"
        "    def route(self, config: InMethod | None = None) -> None: ...\n"
        "    @property\n"
        "    def size(self) -> InProperty: ...\n"
        "    def _hidden(self, x: InPrivate) -> None: ...\n",
        namespace,
    )
    targets = _targets("lucid.scratch.Layer", namespace["Layer"])
    assert [label for label, _ in targets] == [
        "lucid.scratch.Layer",
        "lucid.scratch.Layer.route",
        "lucid.scratch.Layer.size.fget",
    ]
    reports = _unreadable(targets)
    assert [report.split(":")[0] for report in reports] == [label for label, _ in targets]
    assert "InInit" in reports[0] and "InMethod" in reports[1] and "InProperty" in reports[2]
    assert not any("InPrivate" in report for report in reports)


def test_np_is_excused_only_where_this_file_declares_it() -> None:
    numpy_only = _function("def f(a: np.ndarray) -> None: ...")
    assert _unreadable([("lucid.from_numpy", numpy_only)]) == []
    assert len(_unreadable([("lucid.SupportsNumpyConversion.numpy", numpy_only)])) == 1
    assert len(_unreadable([("lucid.somewhere_else", numpy_only)])) == 1
    also_gone = _function("def f(a: np.ndarray, b: Gone) -> None: ...")
    assert any("'Gone'" in report for report in _unreadable([("lucid.from_numpy", also_gone)]))


def test_the_numpy_protocol_reads_without_numpy() -> None:
    """``SupportsNumpyConversion.numpy`` names its return through a lazy alias.

    ``-> np.ndarray`` was evaluated by ``inspect.signature`` itself, so the
    protocol the numpy bridge accepts could not be read in an interpreter
    that had not imported numpy, and this file excused it by name.  The alias
    (``lucid._types._NumpyArray``, a PEP 695 ``type`` whose value is evaluated
    only when asked) reads at no cost to numpy, which stays the bridge's
    dependency and not the package's.
    """
    method = lucid.SupportsNumpyConversion.numpy
    assert list(inspect.signature(method).parameters) == ["self"]
    assert _unresolved(method) == {}
    code = (
        "import sys, inspect, lucid\n"
        "signature = inspect.signature(lucid.SupportsNumpyConversion.numpy)\n"
        "assert repr(signature) == '<Signature (self) -> _NumpyArray>', repr(signature)\n"
        "assert 'numpy' not in sys.modules, 'reading the protocol imported numpy'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO)
    assert result.returncode == 0, result.stderr[-2000:]


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
