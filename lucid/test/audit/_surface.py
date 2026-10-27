"""Enumerating what there is to audit.

Deliberately excludes ``lucid.models``: the zoo is 593 symbols of
composition over the primitives audited here, it has its own contract
test and its own device sweep, and folding it in would triple the run
time while diluting the coverage number this tool exists to report
honestly.

Some callables cannot be surveyed because *calling them changes the
process*.  A first version of this sweep walked names alphabetically,
reached ``set_grad_enabled``, called it with a tensor, and every op after
it failed — 278 of them.  ``STATEFUL`` is that lesson.
"""

import importlib
import inspect
import re
import types
from typing import TYPE_CHECKING, Any

import lucid
import lucid._tensor.tensor as _tensor_module
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor._free_fn_methods import FREE_FN_METHODS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

#: Subsystem name -> (module path, kind).  ``kind`` selects how symbols
#: are probed: ``"op"`` are functions over tensors, ``"module"`` are
#: ``nn.Module`` subclasses, ``"optim"`` are optimizers, ``"other"`` is
#: enumerated for the coverage denominator but has no numeric axis.
SUBSYSTEMS: dict[str, tuple[str, str]] = {
    "lucid": ("lucid", "op"),
    "tensor": ("lucid._tensor.tensor", "op"),
    "nn.functional": ("lucid.nn.functional", "op"),
    "nn": ("lucid.nn", "module"),
    "nn.init": ("lucid.nn.init", "op"),
    "nn.utils": ("lucid.nn.utils", "op"),
    "nn.quantized": ("lucid.nn.quantized", "module"),
    "nn.qat": ("lucid.nn.qat", "module"),
    "nn.intrinsic": ("lucid.nn.intrinsic", "module"),
    "linalg": ("lucid.linalg", "op"),
    "fft": ("lucid.fft", "op"),
    "special": ("lucid.special", "op"),
    "einops": ("lucid.einops", "op"),
    "signal.windows": ("lucid.signal.windows", "op"),
    "autograd": ("lucid.autograd", "op"),
    "func": ("lucid.func", "op"),
    "distributions": ("lucid.distributions", "distribution"),
    "optim": ("lucid.optim", "optim"),
    "optim.lr_scheduler": ("lucid.optim.lr_scheduler", "scheduler"),
    "diffeq": ("lucid.diffeq", "diffeq"),
    "serialization": ("lucid.serialization", "serialize"),
    "amp": ("lucid.amp", "util"),
    "quantization": ("lucid.quantization", "quant"),
    "compile": ("lucid.compile", "compiled"),
    "coreml": ("lucid.coreml", "compiled"),
    "profiler": ("lucid.profiler", "util"),
    "metal": ("lucid.metal", "util"),
    "backends": ("lucid.backends", "util"),
    "weights": ("lucid.weights", "util"),
    "utils": ("lucid.utils", "util"),
    "utils.data": ("lucid.utils.data", "data"),
    "utils.transforms": ("lucid.utils.transforms", "transform"),
    "utils.transforms.functional": ("lucid.utils.transforms.functional", "op"),
    "utils.tokenizer": ("lucid.utils.tokenizer", "tokenizer"),
    "utils.cache": ("lucid.utils.cache", "class"),
    "utils.checkpoint": ("lucid.utils.checkpoint", "op"),
}

#: Excluded from the whole audit, by design.  Stated here rather than
#: silently omitted so the coverage figure means what it says.
EXCLUDED: dict[str, str] = {
    "lucid.models": "the model zoo — 593 symbols with their own contract and device suites",
    "lucid.benchmarks": "benchmark drivers, not API — they time the ops audited here",
    "<re-exports>": (
        "a sub-module reached as an attribute of its parent (lucid.nn.modules "
        "under lucid.nn, lucid.signal.windows under lucid.signal).  Each is "
        "enumerated once under its own key; counting the attribute too would "
        "double-count it"
    ),
    "<foreign names>": (
        "objects a module re-exports from outside Lucid — typing.Callable, "
        "collections.OrderedDict, Protocol.  Visible in dir() only where the "
        "module declares no __all__, and not this framework's API to verify"
    ),
}

#: Calling these mutates process-wide state, opens files, spawns work or
#: blocks.  They are counted in the denominator and never invoked.
#:
#: Split into exact names and prefixes, because the single list this
#: replaces was matched with ``tail.startswith(p)`` and a two-letter
#: entry then claimed whatever began with it:
#:
#:     "to"    ->  topk, tolist, to_dlpack, to_engine_dtype
#:     "item"  ->  itemsize
#:     "get_"  ->  get_total_norm, get_default_qconfig, get_worker_info
#:     "apply_"->  apply_rotary_emb
#:
#: ``topk`` is a pure function of its input and ``itemsize`` is an
#: integer read off a dtype; neither writes anything.  Marked stateful
#: they were excluded from every numeric axis and left with the smoke
#: floor, which is how a differentiable selection op came to have no
#: gradient check — 25 symbols, none of them stateful, all of them
#: silently outside the audit.
#:
#: The second confusion the list made was **reading** state against
#: **writing** it.  ``get_rng_state`` and ``are_deterministic_algorithms
#: _enabled`` answer a question and change nothing; calling them is safe
#: and is the only way to check what the setters did.  Only the writers
#: need the guard, so only the writers are named here.
STATEFUL_EXACT: frozenset[str] = frozenset(
    {
        # the RNG and the global defaults
        "manual_seed",
        "seed",
        "use_deterministic_algorithms",
        # grad-mode context managers, which take effect on construction
        "no_grad",
        "enable_grad",
        "inference_mode",
        # the filesystem
        "save",
        "load",
        "download",
        "load_weight_entry",
        # tracing and timing, which install hooks of their own
        "compile",
        "coreml",
        "profile",
        "record_function",
        # the device
        "synchronize",
        "empty_cache",
        "reset_peak_memory_stats",
        "to",
        "to_shared",
        "cpu",
        "metal",
        "pin_memory",
        "share_memory_",
        "record_stream",
        # the tensor's own autograd state
        "backward",
        "requires_grad_",
        "detach_",
        "apply_",
    }
)

#: Prefixes, each of which ends at a word boundary so a longer name
#: cannot be claimed by accident.
STATEFUL_PREFIX: tuple[str, ...] = (
    "set_",
    "init_",
    "register_",
    "retain_",
    "save_",
    "load_",
    "compile_",
    "compiled_",
)

#: Kept as the union, in that order, for anything reading the old name.
STATEFUL: tuple[str, ...] = tuple(sorted(STATEFUL_EXACT)) + STATEFUL_PREFIX


def mutates_process(tail: str) -> bool:
    """Whether calling ``tail`` writes something the next check reads."""
    return tail in STATEFUL_EXACT or tail.startswith(STATEFUL_PREFIX)


#: Draw fresh random numbers, so two calls never agree.  Probed for shape
#: and dtype but never compared value-for-value, and the determinism axis
#: asks these and only these to reproduce under a fixed seed.
#:
#: Matched as a regex against the symbol's last component rather than as a
#: substring, which is what the list used to be.  Substrings caught
#: whatever happened to contain them:
#:
#:     "normal"  in "normalize"            F.normalize
#:     "sample"  in "grid_sample"          F.grid_sample, nn.Upsample
#:     "poisson" in "poisson_nll_loss"     F.poisson_nll_loss
#:
#: Nine deterministic symbols were flagged that way, and the determinism
#: axis then reported each as vacuous — "a different seed gave identical
#: numbers, this op may not draw" — which was true and was not a finding
#: about the op.  A resampler is not a sampler.
#:
#: ``empty`` is deliberately absent.  It returns uninitialised memory
#: rather than a draw, so there is no seed for it to respect and nothing
#: for this axis to assert; its shape and dtype are checked everywhere
#: else.
STOCHASTIC_PATTERN = (
    r"^rand"  # rand, randn, randint, randperm, random_split, ...
    r"|(^|_)normal_?$"  # normal, kaiming_normal, trunc_normal_
    r"|(^|_)uniform_?$"  # uniform, xavier_uniform
    r"|dropout"  # dropout, dropout2d, feature_alpha_dropout
    r"|bernoulli|multinomial|gumbel|rrelu"
    r"|^poisson$"
    r"|^r?sample$"  # a distribution's sample / rsample
)


class Symbol:
    """One auditable name.

    Attributes
    ----------
    qualname : str
        How a user would write it — ``"F.conv2d"``, ``"lucid.exp"``.
    subsystem : str
        Key into :data:`SUBSYSTEMS`.
    kind : str
        ``"op"`` / ``"module"`` / ``"optim"`` / ``"other"``.
    obj : object
        The callable or class itself.
    """

    __slots__ = ("qualname", "subsystem", "kind", "obj", "flags")

    def __init__(self, qualname: str, subsystem: str, kind: str, obj: Any) -> None:
        self.qualname = qualname
        self.subsystem = subsystem
        self.kind = kind
        self.obj = obj
        self.flags: set[str] = set()
        tail = qualname.rsplit(".", 1)[-1]
        if mutates_process(tail):
            self.flags.add("stateful")
        if tail.endswith("_") and not tail.startswith("_"):
            self.flags.add("inplace")
        if re.search(STOCHASTIC_PATTERN, tail):
            self.flags.add("stochastic")

    @property
    def inert(self) -> bool:
        """bool: Safe to call during a survey."""
        return "stateful" not in self.flags

    @property
    def short(self) -> str:
        return self.qualname.rsplit(".", 1)[-1]

    def __repr__(self) -> str:
        return f"Symbol({self.qualname!r}, {self.kind})"


_PREFIX = {
    "lucid": "lucid.",
    "tensor": "Tensor.",
    "nn.functional": "F.",
    "nn": "nn.",
}


def _names(module: Any) -> list[str]:
    declared = getattr(module, "__all__", None)
    if declared:
        return list(declared)
    return [n for n in dir(module) if not n.startswith("_")]


#: Names that are unmistakably imported machinery rather than API, kept
#: as a list because a bool or a sentinel carries no ``__module__`` to
#: attribute it by.
_FOREIGN_NAMES = frozenset({"TYPE_CHECKING", "annotations"})


def _is_lucid_owned(name: str, obj: Any) -> bool:
    """Whether ``obj`` is this framework's to verify.

    A module without ``__all__`` shows everything it imported, so ``dir()``
    on it hands back ``typing.Callable``, ``collections.OrderedDict`` and
    ``Protocol`` alongside the real API.  Auditing those would inflate the
    denominator with names Lucid does not own and cannot fix.

    Only callables and classes are attributed by ``__module__`` — a plain
    value (a dtype singleton, a constant) does not carry one, so those go
    through the name list instead.
    """
    if name in _FOREIGN_NAMES:
        return False
    if not callable(obj):
        return True
    origin = getattr(obj, "__module__", None)
    if not isinstance(origin, str):
        return True
    return origin == "lucid" or origin.startswith("lucid.")


#: Package prefixes never walked.  Each has its own suite; see EXCLUDED.
_SKIP_PACKAGES = ("lucid.models", "lucid.test", "lucid._C", "lucid.benchmarks")


def _walkable_modules() -> list[str]:
    """Every importable public module under ``lucid``, shallowest first.

    Recursive rather than a hand-maintained list.  The list *was* hand
    maintained: it named nineteen modules, and an independent walk found
    the API was reachable through a hundred and thirty.  ``lucid.signal``
    was in that list and contributed nothing, because everything it
    exports is a sub-module and the enumeration skipped those — so the
    subsystem showed up in ``--list-subsystems`` with no symbols under it
    and nobody noticed.  Discovering modules instead of declaring them is
    what stops that recurring the next time a namespace is added.

    Shallowest first so ``lucid.nn.init.kaiming_uniform`` is the name
    reported for that function rather than a deeper re-export of it.
    """
    import pkgutil

    found = ["lucid"]
    for info in pkgutil.walk_packages(lucid.__path__, "lucid."):
        name = info.name
        if name.startswith(_SKIP_PACKAGES):
            continue
        if any(part.startswith("_") for part in name.split(".")[1:]):
            continue
        found.append(name)
    return sorted(set(found), key=lambda m: (m.count("."), m))


def _import(path: str) -> types.ModuleType | None:
    """The module at ``path``, or ``None`` when importing it raises."""
    try:
        return importlib.import_module(path)
    except Exception:  # noqa: BLE001 - an unimportable module is its own finding
        return None


def _declaring_module(obj: Any, found_in: str) -> str:
    """Where ``obj`` was defined, falling back to where it was found.

    Attribution has to follow the definition, not the re-export.  ``save``
    and ``load`` are written in ``lucid.serialization`` and re-exported
    from ``lucid``; enumerating shallowest-first finds them under
    ``lucid`` and, keyed on that, they came out as plain ops and the
    serialization axis dropped to zero applicable symbols.  The qualname
    still reports the shallow spelling — that is what a user writes — but
    the *kind* follows the definition.
    """
    origin = getattr(obj, "__module__", None)
    if isinstance(origin, str) and (origin == "lucid" or origin.startswith("lucid.")):
        return origin
    return found_in


def _longest_prefix(path: str) -> tuple[str, str] | None:
    """The most specific ``SUBSYSTEMS`` entry covering ``path``.

    ``lucid.utils.transforms.functional`` is a module of functions living
    under a package of classes, so the most specific declaration has to
    win or its 33 functions get run through the transform lifecycle.
    """
    best: tuple[str, str] | None = None
    best_len = -1
    for key, (mod_path, kind) in SUBSYSTEMS.items():
        if (path == mod_path or path.startswith(mod_path + ".")) and len(mod_path) > best_len:
            best, best_len = (key, kind), len(mod_path)
    return best


def _kind_for(path: str) -> str:
    match = _longest_prefix(path)
    return match[1] if match else "op"


def _subsystem_for(path: str) -> str:
    """Which subsystem row a module's symbols are reported under."""
    match = _longest_prefix(path)
    return match[0] if match else "lucid"


def enumerate_surface(subsystems: list[str] | None = None) -> list[Symbol]:
    """Every public symbol in scope, in a stable order.

    Parameters
    ----------
    subsystems : list of str, optional
        Restrict to these keys of :data:`SUBSYSTEMS`.  ``None`` takes all.

    Returns
    -------
    list of Symbol
    """
    wanted = set(subsystems) if subsystems else None
    out: list[Symbol] = []
    # One object re-exported under two names is one thing to verify, not
    # two.  ``lucid.nn`` re-exports the quantized layers, ``nn.intrinsic``
    # and ``nn.qat`` share fused classes — without this the denominator
    # counts the same class up to three times and the coverage figure
    # flatters itself.  Modules are walked shallowest-first, so the
    # spelling that survives is the one a user would write.
    claimed: set[int] = set()

    if wanted is None or "tensor" in wanted:
        for name in sorted(dir(_tensor_module.Tensor)):
            if name.startswith("_"):
                continue
            attr = inspect.getattr_static(_tensor_module.Tensor, name, None)
            if attr is None:
                continue
            out.append(Symbol(f"Tensor.{name}", "tensor", "method", attr))

    for path in _walkable_modules():
        module = _import(path)
        if module is None:
            continue
        for name in sorted(_names(module)):
            obj = getattr(module, name, None)
            if obj is None:
                continue
            if isinstance(obj, types.ModuleType):
                continue  # a namespace; it is walked in its own right
            if not _is_lucid_owned(name, obj):
                continue
            if id(obj) in claimed:
                continue
            home = _declaring_module(obj, path)
            key = _subsystem_for(home)
            if wanted is not None and key not in wanted:
                continue
            claimed.add(id(obj))
            prefix = _PREFIX.get(_subsystem_for(path))
            qualname = prefix + name if prefix else f"{path}.{name}"
            out.append(Symbol(qualname, key, _classify(obj, _kind_for(home)), obj))
    return out


def _classify(obj: Any, subsystem_kind: str) -> str:
    """What kind of thing this is, for axis dispatch.

    An ``nn.Module`` subclass is a module wherever it lives — the
    quantized layers are in ``lucid.quantization`` and still want the
    module lifecycle run over them.

    ``"declaration"`` is the kind for names that exist for the type
    checker or for subclasses to fill in, and have no behaviour of their
    own to audit: :class:`typing.Protocol` definitions, metaclasses, and
    abstract bases.  They were being counted in the denominator and then
    skipped one axis at a time, which makes a coverage figure that can
    never reach its own ceiling and buries the symbols that *should* be
    reachable under ones that never can be.  Naming them is not an
    exemption — every concrete subclass is still audited in its own
    right, which is where the behaviour actually is.
    """
    if isinstance(obj, type):
        if _is_declaration(obj):
            return "declaration"
        if _is_nn_module(obj):
            return "module"
        # ``PaddingMode`` lives among the conv layers and ``PackedSequence``
        # among the rnn utilities; neither has a forward, so the module
        # lifecycle would report a failure that is really a miscategory.
        if subsystem_kind in ("op", "module"):
            return "class"
        return subsystem_kind
    if callable(obj):
        # ``nn`` is a subsystem of classes, but ``register_module_forward_hook``
        # is a function and wants the function axes, not the module lifecycle.
        return "op" if subsystem_kind in ("op", "module") else subsystem_kind
    return "value"


def _is_declaration(obj: type) -> bool:
    """A name with nothing of its own to run.

    Three kinds, and each fails a survey for a different reason: a
    Protocol has no implementation, a metaclass builds classes rather
    than values, and an abstract base refuses instantiation by design.
    """
    if getattr(obj, "_is_protocol", False):
        return True
    try:
        if issubclass(obj, type):  # a metaclass
            return True
    except TypeError:
        return False
    return bool(getattr(obj, "__abstractmethods__", frozenset()))


def _is_nn_module(obj: type) -> bool:
    try:
        return issubclass(obj, nn.Module)
    except TypeError:
        return False


def independent_walk() -> dict[int, str]:
    """Every public Lucid-owned object, found without using the surface.

    The check on :func:`enumerate_surface`, and deliberately not sharing
    its machinery beyond the two ownership predicates: it walks every
    module, takes every public name, and asks nothing about subsystems,
    kinds or prefixes.  If the two disagree, the surface is missing
    something.

    Comparing a count against itself is how the enumeration sat at 73.8%
    of the framework while reporting coverage as if the other quarter did
    not exist.

    Returns
    -------
    dict
        ``id(object) -> the first dotted path it was found at``.
    """
    found: dict[int, str] = {}
    for path in _walkable_modules():
        module = _import(path)
        if module is None:
            continue
        for name in sorted(_names(module)):
            obj = getattr(module, name, None)
            if obj is None or isinstance(obj, types.ModuleType):
                continue
            if not _is_lucid_owned(name, obj):
                continue
            found.setdefault(id(obj), f"{path}.{name}")
    return found


def resolve(symbol: Symbol) -> Any:
    """The callable to invoke for ``symbol``, or ``None``.

    ``Tensor`` methods are the reason this is not just ``symbol.obj``.
    They were enumerated with :func:`inspect.getattr_static`, so a plain
    method comes back as the underlying function — callable as
    ``fn(tensor, ...)`` — while a property comes back as the descriptor
    and has to be unwrapped.  253 symbols had no axis at all until this
    distinguished the two.
    """
    if symbol.kind != "method":
        return symbol.obj
    obj = symbol.obj
    if isinstance(obj, property):
        getter = obj.fget
        if getter is None:
            return None
        return lambda tensor, *args, **kwargs: getter(tensor)
    if isinstance(obj, (staticmethod, classmethod)):
        return obj.__func__
    if callable(obj):
        return obj
    return None


#: Names whose ``F.`` spelling is a different operation from the free and
#: method ones, in the reference as well: ``Tensor.unfold(dim, size, step)``
#: slides a window along one dimension, ``F.unfold(input, kernel_size)`` is
#: im2col.  Comparing them asked whether two different ops agree, and the
#: entry axis failed both the moment ``Tensor.unfold``'s signature became
#: readable.
_FUNCTIONAL_IS_ANOTHER_OP: frozenset[str] = frozenset({"unfold"})


def counterparts(symbol: Symbol) -> Iterator[tuple[str, Any]]:
    """Every entry point the same operation is reachable through.

    One op with three spellings is one op with three chances to drift —
    which is how scalar coercion came to exist on the operator path only.
    """
    name = symbol.short
    split = name in _FUNCTIONAL_IS_ANOTHER_OP
    is_functional = symbol.qualname.startswith("F.")
    free = getattr(lucid, name, None)
    if not (split and is_functional):
        if callable(free) and not isinstance(free, type):
            yield "free", free
        method = inspect.getattr_static(_tensor_module.Tensor, name, None)
        if callable(method):
            yield "method", method
    if split and not is_functional:
        return
    functional = getattr(F, name, None)
    if callable(functional) and not isinstance(functional, type) and functional is not free:
        yield "functional", functional


def spellings(short: str) -> frozenset[str]:
    """The qualnames :func:`counterparts` compares for an op named ``short``."""
    return frozenset({f"lucid.{short}", f"Tensor.{short}", f"F.{short}"})


def recorded_as(qualname: str) -> str:
    """The qualname the audit files what it knows about ``qualname`` under.

    The tables keyed by qualname — where the reference is finite
    (``stability_contract.json``), which dtypes it accepts
    (``dtype_contract.json``), why an op has no derivative
    (``NOT_DIFFERENTIABLE``) — are measured or written per spelling.  A
    :data:`~lucid._tensor._free_fn_methods.FREE_FN_METHODS` method *is*
    ``lucid.<name>`` with ``self`` first, so it has no knowledge of its
    own to record: what is true of the free function is true of it.  When
    LCD-547 added 32 of them, each was a stranger to every table, and
    eight cells went red on CI for facts already on file under the free
    function — ``x.acosh()`` NaN below 1, ``x.heaviside(v)`` a step.
    """
    head, _, tail = qualname.rpartition(".")
    if head == "Tensor" and tail in FREE_FN_METHODS:
        return f"lucid.{tail}"
    return qualname


#: The module a qualname's first component stands for, where that
#: component is a :data:`_PREFIX` rather than the head of a module path.
_ROOTS = {"lucid": "lucid", "F": "lucid.nn.functional", "nn": "lucid.nn"}


def _names_nothing(qualname: str, modules: dict[str, types.ModuleType]) -> bool:
    """Whether ``qualname`` resolves to nothing, given every walkable module.

    A sub-module wins over an attribute of the same name: ``lucid.compile``
    is a function and a package at once, and ``CompiledModule`` lives in the
    package.  A spelling this cannot read is never called absent.
    """
    head, _, rest = qualname.partition(".")
    if head == "Tensor":
        return inspect.getattr_static(_tensor_module.Tensor, rest, None) is None
    root = _ROOTS.get(head, "")
    obj: object = modules.get(root)
    if obj is None or not rest:
        return False
    path = root
    for part in rest.split("."):
        path = f"{path}.{part}"
        obj = modules.get(path) or getattr(obj, part, None)
        if obj is None:
            return True
    return False


def removed(qualnames: Iterable[str]) -> set[str]:
    """The names in ``qualnames`` that no longer name anything public.

    Missing from :func:`enumerate_surface` is not enough on its own: that
    walk skips a module that fails to import without a word, so nothing
    counts as removed while any walkable module fails to import.  Nor is
    a name removed that still resolves — one the walk now reaches under a
    shallower spelling, say.
    """
    wanted = set(qualnames)
    if not wanted:
        return set()
    modules: dict[str, types.ModuleType] = {}
    for path in _walkable_modules():
        module = _import(path)
        if module is None:
            return set()
        modules[path] = module
    present = {symbol.qualname for symbol in enumerate_surface()}
    return {name for name in wanted - present if _names_nothing(name, modules)}


__all__ = [
    "EXCLUDED",
    "independent_walk",
    "STATEFUL",
    "STATEFUL_EXACT",
    "STATEFUL_PREFIX",
    "STOCHASTIC_PATTERN",
    "mutates_process",
    "SUBSYSTEMS",
    "Symbol",
    "counterparts",
    "enumerate_surface",
    "recorded_as",
    "removed",
    "resolve",
    "spellings",
]
