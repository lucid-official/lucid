"""
lucid.compile._core.hook_contract — which forward hooks a compiled call may trace.

A trace runs the model's forward through ``Module.__call__``, so a module's
forward pre-hooks and forward hooks run *while tracing* and the tensor ops they
perform are recorded into the graph.  What a replay cannot do is run Python: a
hook that only rewrites tensors is the same graph on every call, but a hook
that appends to a list, bumps a counter or prints would run once, at trace time,
and never again.  :class:`HookContract` decides which of the two a hook is, and
what makes a traced hook's graph stale:

* **Declared.**  A hook is traced only if it is *declared* a pure tensor
  rewrite: marked with :func:`traceable_hook`, or a callable whose class sets
  ``TRACEABLE_MARKER``.  Nothing is inferred.  A scan of a hook's own code
  cannot see a side effect that lives in a helper it calls (a logger, a stats
  accumulator, a counter), and a replay would skip it silently; so every
  unmarked forward hook and pre-hook runs eagerly, with a one-time notice.
  lucid marks only its own hooks (the prompt-learning sites).  Backward hooks,
  global hooks and parameter gradient hooks are never traced.
* **Guarded.**  A marked hook is still scanned: its own code (and the code
  nested in it) must write nothing outside its frame — no attribute, item,
  global or ``nonlocal`` store, no call of a mutator or an I/O function by
  name — and must have a path that *returns* a value.  A marked hook that fails
  the scan is not traced (eager, with the notice naming why); the scan is a
  check on the declaration, not a substitute for it.
* **Keyed.**  The key (:attr:`CacheKey.module_state`) names each traced hook of
  each module — its registration id, its callable, whether it takes kwargs, and
  the order — so adding, removing, reordering or replacing a hook moves it.  It
  also reads, on every call, the state the hook branches on that no argument
  hands it: its closure cells, the globals it names, and the attributes of the
  object it is a method of or an instance of.  The modules a hook reaches
  (``tuner._apply`` bound to a tuner, a closure over one) are part of the tree,
  so what a hook reads from them — ``disable_adapters``, ``active_adapter`` —
  is keyed with the rest of the tree's state.
"""

import dis
import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from types import CodeType, FunctionType, MethodType
from typing import ClassVar, Final, final

from lucid.compile._core.callable_slots import callable_slots, cell_value
from lucid.compile._core.module_state import StateContract
from lucid.nn.module import Module

__all__ = ["HookContract", "HookSite", "TRACEABLE_MARKER", "traceable_hook"]

#: The attribute that declares a hook a pure tensor rewrite.  A class sets it
#: as a class attribute (lucid's peft sites do, without importing this module);
#: a function gets it from :func:`traceable_hook`.
TRACEABLE_MARKER: Final = "_lucid_traceable_hook"


def traceable_hook[F: Callable[..., object]](hook: F) -> F:
    """Declare ``hook`` a pure tensor rewrite, which a compiled call may trace.

    Private to ``lucid.compile``: a hook that reads state, branches on a
    switch or builds tensors is fine — the key names what it reads — but one
    that acts by side effect (logging, counting, recording) is not, since a
    replay never runs it.

    Parameters
    ----------
    hook : callable
        A function, or a class whose instances are hooks.

    Returns
    -------
    callable
        ``hook`` itself, marked.
    """
    setattr(hook, TRACEABLE_MARKER, True)
    return hook


# What walking a code object found, by the object's address: an entry leaves
# with its code: (why it is not traceable or None, whether it returns a value).
_SCANS: dict[int, tuple[weakref.ref[CodeType], tuple[str | None, bool]]] = {}


@final
@dataclass(frozen=True, slots=True)
class HookSite:
    """A module of the tree that holds forward hooks the key must name.

    Attributes
    ----------
    pos : int
        The module's position in the tree walk.
    module : weakref.ref of Module
        The module, held weakly: the walk is cached against a model that must
        not be kept alive by its hooks.
    internal : frozenset of int
        Ids of the module's pre-hooks that ``nn.utils`` reparametrisations
        installed (weight_norm, spectral_norm, prune); those are traced in the
        graph and keyed by their own state, not here.
    """

    pos: int
    module: weakref.ref[Module]
    internal: frozenset[int]


@final
class HookContract:
    """Which forward hooks a compiled call traces, and how each enters the key.

    A namespace of classmethods over a module's hook registries: :meth:`blocker`
    and :meth:`block_of` judge a hook, :meth:`site_of` and :meth:`read` key the
    ones that are traced, and :meth:`reached` names the modules they hold.
    """

    #: Bytecode that writes outside the hook's frame.
    WRITES: ClassVar[frozenset[str]] = frozenset(
        {
            "STORE_ATTR",
            "STORE_SUBSCR",
            "STORE_SLICE",
            "STORE_GLOBAL",
            "DELETE_ATTR",
            "DELETE_SUBSCR",
            "DELETE_GLOBAL",
        }
    )
    #: Stores into a closure cell: a write only when the cell is not the hook's own.
    CELL_WRITES: ClassVar[frozenset[str]] = frozenset({"STORE_DEREF", "DELETE_DEREF"})
    #: Methods and functions called by name that act through a side effect.
    EFFECTS: ClassVar[frozenset[str]] = frozenset(
        {
            # mutators of a container the hook can reach
            "append",
            "appendleft",
            "extend",
            "extendleft",
            "insert",
            "add",
            "update",
            "pop",
            "popitem",
            "popleft",
            "remove",
            "discard",
            "clear",
            "setdefault",
            "sort",
            "reverse",
            # I/O and logging
            "print",
            "open",
            "input",
            "write",
            "writelines",
            "flush",
            "info",
            "debug",
            "warning",
            "error",
            "exception",
            "critical",
            # attribute and import machinery
            "setattr",
            "delattr",
            "exec",
            "eval",
            "breakpoint",
            # a hook that installs hooks changes the module it runs in
            "register_buffer",
            "register_parameter",
            "register_module",
            "add_module",
        }
    )
    #: ``co_flags`` of a generator, a coroutine and an async generator.
    NOT_PLAIN: ClassVar[int] = 0x20 | 0x80 | 0x100 | 0x200
    #: Instructions whose name operand is a looked-up attribute or global.
    NAME_LOADS: ClassVar[frozenset[str]] = frozenset({"LOAD_ATTR", "LOAD_GLOBAL", "LOAD_NAME"})
    #: Past this many functions a hook closes over, the rest are not followed.
    MAX_FUNCTIONS: ClassVar[int] = 8
    #: The module's hook registries, with the ``with_kwargs`` set of each.
    REGISTRIES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("_forward_pre_hooks", "_forward_pre_hooks_with_kwargs"),
        ("_forward_hooks", "_forward_hooks_with_kwargs"),
    )

    # ── what a hook is made of ───────────────────────────────────────

    @classmethod
    def parts(cls, hook: object) -> tuple[tuple[FunctionType, ...], tuple[object, ...]] | None:
        """The functions that decide what ``hook`` does and the objects whose
        state they read, or ``None`` when ``hook`` is not Python we can read
        (a builtin, a C callable).

        A bound method contributes its function and its ``__self__``; a
        ``functools.partial`` its function and bound arguments; a callable
        object its class's ``__call__`` and itself.  Functions the hook closes
        over are followed too, so a decorator's wrapper is judged with the
        function it wraps.
        """
        funcs: list[FunctionType] = []
        owners: list[object] = []
        if not cls._unwrap(hook, funcs, owners):
            return None
        i = 0
        while i < len(funcs) and len(funcs) < cls.MAX_FUNCTIONS:
            for cell in funcs[i].__closure__ or ():
                inner = cell_value(cell)
                if isinstance(inner, FunctionType) and inner not in funcs:
                    funcs.append(inner)
            i += 1
        return tuple(funcs), tuple(owners)

    @classmethod
    def _unwrap(cls, hook: object, funcs: list[FunctionType], owners: list[object]) -> bool:
        if isinstance(hook, FunctionType):
            funcs.append(hook)
            return True
        if isinstance(hook, MethodType):
            if not isinstance(hook.__func__, FunctionType):
                return False
            funcs.append(hook.__func__)
            owners.append(hook.__self__)
            return True
        if isinstance(hook, partial):
            owners.extend((*hook.args, *hook.keywords.values()))
            return cls._unwrap(hook.func, funcs, owners)
        call = type(hook).__call__  # the class's own: a hook is invoked through its type
        if isinstance(call, FunctionType):
            funcs.append(call)
            owners.append(hook)
            return True
        return False

    # ── is it traceable ──────────────────────────────────────────────

    @classmethod
    def scan(cls, code: CodeType) -> tuple[str | None, bool]:
        """Why ``code`` cannot be traced (or ``None``), and whether it returns a value.

        Cached against the code object: the walk that asks runs once per tree
        version, but a model with a few hundred hooked modules asks for each.
        """
        key = id(code)
        hit = _SCANS.get(key)
        if hit is not None:
            return hit[1]
        found = cls._scan(code, frozenset(), root=True)
        gone = partial(_SCANS.pop, key, None)
        _SCANS[key] = (weakref.ref(code, lambda _: gone()), found)
        return found

    @classmethod
    def _scan(
        cls, code: CodeType, local_cells: frozenset[str], *, root: bool
    ) -> tuple[str | None, bool]:
        if code.co_flags & cls.NOT_PLAIN:
            return "is a generator or coroutine", False
        instructions = list(dis.get_instructions(code))
        for ins in instructions:
            if (reason := cls._violation(code, ins, local_cells)) is not None:
                return reason, False
        inner_cells = frozenset(local_cells | set(code.co_cellvars))
        for const in code.co_consts:
            if isinstance(const, CodeType):
                reason, _ = cls._scan(const, inner_cells, root=False)
                if reason is not None:
                    return reason, False
        return None, root and cls._returns_value(instructions)

    @classmethod
    def _violation(
        cls, code: CodeType, ins: dis.Instruction, local_cells: frozenset[str]
    ) -> str | None:
        """Why one instruction makes ``code`` untraceable, or ``None``."""
        op, name = ins.opname, str(ins.argval)
        if op in cls.WRITES:
            return f"writes outside its frame ({op.lower()} {name})"
        if op in cls.CELL_WRITES and name in code.co_freevars and name not in local_cells:
            return f"rebinds the enclosing variable `{name}`"
        if op in cls.NAME_LOADS and name in cls.EFFECTS:
            return f"calls `{name}`, which acts by side effect"
        return None

    @staticmethod
    def _returns_value(instructions: list[dis.Instruction]) -> bool:
        """Whether some path returns something other than a literal ``None``."""
        for i, ins in enumerate(instructions):
            if ins.opname == "RETURN_CONST":
                if ins.argval is not None:
                    return True
            elif ins.opname == "RETURN_VALUE":
                prev = instructions[i - 1] if i else None
                if prev is None or prev.opname != "LOAD_CONST" or prev.argval is not None:
                    return True
        return False

    @classmethod
    def declared(cls, hook: object) -> bool:
        """Whether ``hook`` carries ``TRACEABLE_MARKER`` (a ``partial`` by its function)."""
        target = hook.func if isinstance(hook, partial) else hook
        return getattr(target, TRACEABLE_MARKER, False) is True

    @classmethod
    def blocker(cls, hook: object) -> str | None:
        """Why ``hook`` cannot be traced, or ``None`` if it can."""
        if not cls.declared(hook):
            return "is not declared a pure tensor rewrite, so it may act by side effect"
        parts = cls.parts(hook)
        if parts is None:
            return "is not a Python callable"
        for fn in parts[0]:
            reason, returns_value = cls.scan(fn.__code__)
            if reason is not None:
                return reason
            if not returns_value:
                return "returns nothing, so it acts only by side effect"
        return None

    # ── the tree ─────────────────────────────────────────────────────

    @classmethod
    def hooks_of(cls, sub: Module, internal: frozenset[int]) -> Iterator[tuple[str, int, object]]:
        """Each forward hook of ``sub`` that is not an ``nn.utils`` reparametrisation's.

        Parameters
        ----------
        sub : Module
            The module whose registries are read.
        internal : frozenset of int
            Ids of the pre-hooks the reparametrisations installed.

        Yields
        ------
        tuple of (str, int, object)
            ``(registry name, hook id, hook)``, pre-hooks first, each in call order.
        """
        for registry, _ in cls.REGISTRIES:
            for key, hook in getattr(sub, registry, {}).items():
                if not (registry == "_forward_pre_hooks" and key in internal):
                    yield registry, key, hook

    @classmethod
    def block_of(cls, sub: Module, internal: frozenset[int]) -> str | None:
        """Why a forward hook of ``sub`` cannot be traced, or ``None``."""
        for registry, key, hook in cls.hooks_of(sub, internal):
            reason = cls.blocker(hook)
            if reason is not None:
                label = registry.removeprefix("_").removesuffix("s").replace("_", " ")
                name = getattr(hook, "__qualname__", type(hook).__qualname__)
                return f"the {label} {name!r} (id {key}) on {type(sub).__name__} {reason}"
        return None

    @classmethod
    def site_of(cls, pos: int, sub: Module, internal: frozenset[int]) -> HookSite | None:
        """``sub``'s :class:`HookSite`, or ``None`` when it holds no hook the key names."""
        if next(cls.hooks_of(sub, internal), None) is None:
            return None
        return HookSite(pos, weakref.ref(sub), internal)

    @classmethod
    def reached(cls, sub: Module) -> list[Module]:
        """The modules ``sub``'s forward hooks hold — bound to, closed over, or
        kept on the hook object — which the tree takes in.

        A hook reads its tuner's switches; a tuner outside the tree would have
        its state, parameters and replacement unwatched.
        """
        found: list[Module] = []
        for registry, _ in cls.REGISTRIES:
            for hook in getattr(sub, registry, {}).values():
                found.extend(v for v in cls.values_of(hook) if isinstance(v, Module))
        return found

    # ── what a traced hook reads ─────────────────────────────────────

    @classmethod
    def values_of(cls, hook: object) -> Iterator[object]:
        """Everything ``hook`` reads that no argument hands it, once each:
        closure cells, the objects it belongs to, and their attributes."""
        parts = cls.parts(hook)
        if parts is None:
            return
        funcs, owners = parts
        for fn in funcs:
            for cell in fn.__closure__ or ():
                yield cell_value(cell)
        for owner in owners:
            yield owner
            if isinstance(owner, Module):
                continue  # a module's own state is keyed with the tree's
            attrs = getattr(owner, "__dict__", None)
            if isinstance(attrs, dict):
                yield from attrs.values()
            for klass in type(owner).__mro__:
                names = klass.__dict__.get("__slots__", ())
                for slot in (names,) if isinstance(names, str) else names:
                    if slot not in ("__dict__", "__weakref__"):
                        yield getattr(owner, slot, None)

    @classmethod
    def _value_sig(cls, value: object) -> object:
        """A value a hook reads, keyed: a tensor and a module by identity (a
        module's own state is the tree's), state by value, anything else by
        identity."""
        from lucid._tensor.tensor import Tensor

        if isinstance(value, Tensor):
            return ("tensor", id(value._impl))
        if isinstance(value, Module):
            return ("module", id(value))
        sig = StateContract.value_sig(value)
        return sig if sig is not None else ("object", id(value))

    @classmethod
    def state_of(cls, hook: object) -> tuple[object, ...]:
        """What ``hook`` reads now: :meth:`values_of` and the globals its code names.

        A global counts only if it is state — a tensor or an atom or short
        container of atoms — so the functions and classes a hook calls cost
        nothing.
        """
        from lucid._tensor.tensor import Tensor

        sigs = [cls._value_sig(v) for v in cls.values_of(hook)]
        parts = cls.parts(hook)
        for fn in parts[0] if parts is not None else ():
            for (kind, name), value in callable_slots(fn):
                if kind == "global" and (
                    isinstance(value, Tensor) or StateContract.value_sig(value) is not None
                ):
                    sigs.append((name, cls._value_sig(value)))
        return tuple(sigs)

    @classmethod
    def read(cls, sites: Iterable[HookSite]) -> tuple[object, ...]:
        """The key's view of the traced hooks, read on every call.

        Per module: each hook's registration id, callable, ``with_kwargs``
        flag and state, in call order — so a hook added, removed, replaced or
        reordered, and a cell, global or attribute it reads that changed, each
        give another key.
        """
        out: list[object] = []
        for site in sites:
            sub = site.module()
            if sub is None:
                out.append((site.pos, None))
                continue
            kwargs = {registry: getattr(sub, flags) for registry, flags in cls.REGISTRIES}
            out.append(
                (
                    site.pos,
                    tuple(
                        (registry, key, id(hook), key in kwargs[registry], cls.state_of(hook))
                        for registry, key, hook in cls.hooks_of(sub, site.internal)
                    ),
                )
            )
        return tuple(out)
