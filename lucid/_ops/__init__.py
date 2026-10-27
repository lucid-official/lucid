"""
Free-function ops exposed as ``lucid.xxx``.

All functions are generated from the registry.  Each generated wrapper carries
an explicit ``__signature__`` derived from the underlying adapter (when one
exists) or parsed from the pybind11 ``__doc__`` of the engine function.  The
result is that ``inspect.signature(lucid.cumsum)`` returns
``(input: Tensor, dim: int = -1)`` rather than the generic
``(*args, **kwargs)`` — so IDE autocomplete, ``help(lucid.foo)``, IPython
introspection, and ``inspect.Signature.bind()`` all see the real parameter
names.

The wrapper *body* still uses ``*args / **kwargs`` so that the unwrap/wrap
boundary remains a single forwarding point, but the visible surface is fully
typed.

For static type-checking, the same per-op signatures are emitted into
``lucid/__init__.pyi`` by ``tools/gen_pyi.py``; both paths are driven by
``_signature_for_entry`` below to keep them in sync.
"""

import builtins
import functools
import inspect
import operator
import re
import sys
import types
import typing
from typing import TYPE_CHECKING

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._registry import OpEntry, _REGISTRY, _bind_leading


# 3.2.2: index ``_REGISTRY`` by ``free_fn_name`` so ``_make_free_fn`` and
# ``_populate_free_fns`` skip the O(N) linear scan over ~500–1000 entries.
# Built once at module load, before ``_populate_free_fns`` runs.  An entry
# without a ``free_fn_name`` is indexed by its ``name``; collisions are
# resolved by first-wins (matching the ``set``-based dedup in
# ``_populate_free_fns``).
def _by_free_fn_name() -> dict[str, OpEntry]:
    by_name: dict[str, OpEntry] = {}
    for entry in _REGISTRY:
        by_name.setdefault(entry.free_fn_name or entry.name, entry)
    return by_name


_REGISTRY_BY_FREE_FN = _by_free_fn_name()

# PEP 749 / Python 3.14: ``Format.FORWARDREF`` keeps unresolved names as
# ``ForwardRef`` objects instead of raising ``NameError``.  Used when
# introspecting adapter signatures whose annotations reference symbols
# that live behind ``TYPE_CHECKING`` (e.g. ``Tensor`` in ``_adapters``).
# Falls back to plain ``inspect.signature`` on 3.13 where ``annotationlib``
# (PEP 649 stdlib module) does not yet exist; the try/except in the
# adapter path picks up any resulting ``NameError``.
if sys.version_info >= (3, 14):
    import annotationlib

    _FORWARDREF: annotationlib.Format | None = annotationlib.Format.FORWARDREF
else:
    _FORWARDREF = None

# Builtins we need are saved up front because we register ops named ``min`` /
# ``max`` / ``sum`` / ``any`` / ``all`` into module ``globals()`` before this
# file finishes executing — bare references to those names below would resolve
# to the wrappers (with TensorImpl-only signatures) instead of the builtins.
_b_min = builtins.min
_b_max = builtins.max

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Signature extraction ─────────────────────────────────────────────────────


# Matches the first line of a pybind11 docstring:
#   e.g. name(param: Type [= default], ...) -> ReturnType
# We don't need to parse types — only param names + defaults — because the
# wrapper forwards everything verbatim.
_PYBIND_SIG_RE = re.compile(r"^[a-zA-Z_][\w]*\((?P<params>.*)\)\s*->", re.DOTALL)


def _parse_pybind_signature(fn: object) -> inspect.Signature | None:
    """Extract an ``inspect.Signature`` from a pybind11 builtin's docstring.

    Returns ``None`` if the docstring is missing or unparseable.  pybind11
    doesn't expose ``__signature__`` on its builtins, but it does emit a
    well-formed signature on the first line of ``__doc__``.
    """
    doc = getattr(fn, "__doc__", None)
    if not doc:
        return None
    first_line = doc.splitlines()[0]
    m = _PYBIND_SIG_RE.match(first_line)
    if not m:
        return None
    raw = m.group("params").strip()
    if not raw:
        return inspect.Signature(parameters=[])

    # Split top-level commas only — types may contain nested brackets.
    params: list[inspect.Parameter] = []
    depth = 0
    cur = ""
    for ch in raw + ",":
        if ch == "," and depth == 0:
            piece = cur.strip()
            cur = ""
            if not piece:
                continue
            # piece looks like "name: Type" or "name: Type = default"
            if "=" in piece:
                head, default_str = piece.split("=", 1)
                default_str = default_str.strip()
                # Try to evaluate simple defaults (numbers, [], (), etc.).
                try:
                    default = eval(default_str, {"__builtins__": {}}, {})
                except (SyntaxError, NameError):
                    # An enum repr (``<Dtype.F32: 6>``) or a qualified name
                    # has no literal form; the wrapper forwards it anyway.
                    default = inspect.Parameter.empty
            else:
                head = piece
                default = inspect.Parameter.empty
            pname = head.split(":", 1)[0].strip()
            if not pname.isidentifier():
                return None
            params.append(
                inspect.Parameter(pname, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
            )
        else:
            if ch in "[(":
                depth += 1
            elif ch in "])":
                depth -= 1
            cur += ch
    return inspect.Signature(parameters=params)


# Internal identifiers (in adapters) and single-letter pybind names that we
# rewrite to user-facing names.  Domain-specific names (``index``, ``mask``,
# ``cond``, etc.) are kept as-is.
_GENERIC_PARAM_REWRITE = {
    "a": "input",
    "x": "input",
    "x_impl": "input",
    "a_impl": "input",
    "self": "input",
    "b": "other",
    "y": "other",
    "y_impl": "other",
    "b_impl": "other",
}


#: Tensor operands the reference names other than ``input`` / ``other``.
#: The wrapper binds its leading operands itself (:func:`_bind_leading`), so
#: these names reach no callee: an engine builtin keeps its ``(a, b)``.
_REFERENCE_OPERANDS: dict[str, tuple[str, ...]] = {
    "pow": ("input", "exponent"),
    "pow_": ("input", "exponent"),
    "mm": ("input", "mat2"),
    "bmm": ("input", "mat2"),
    "dot": ("input", "tensor"),
    "outer": ("input", "vec2"),
    "tensordot": ("a", "b"),
    "split": ("tensor",),
}


def _rename_leading_tensor_params(
    sig: inspect.Signature, n_tensor_args: int, op: str
) -> inspect.Signature:
    """Rewrite the first ``n_tensor_args`` parameters from their internal /
    single-letter names (``a``, ``b``, ``x_impl``) to their user-facing
    counterparts — the reference's (``_REFERENCE_OPERANDS``) for ``op``,
    else ``input`` / ``other``.  Domain-specific names like ``indices`` or
    ``mask`` are left alone — they already convey role."""
    if n_tensor_args <= 0:
        return sig
    params = list(sig.parameters.values())
    upto = _b_min(n_tensor_args, len(params))
    named = _REFERENCE_OPERANDS.get(op, ())
    for i in range(upto):
        new_name = named[i] if i < len(named) else None
        new_name = new_name or _GENERIC_PARAM_REWRITE.get(params[i].name)
        if new_name is not None:
            params[i] = params[i].replace(name=new_name)
    return sig.replace(parameters=params)


def _signature_for_entry(entry: OpEntry) -> inspect.Signature:
    """Best-effort signature for a registry entry.

    Strategy:
      1. ``n_tensor_args == -1`` (list-style ops): synthesise a signature
         starting with ``tensors: list[Tensor]`` and parse the rest from the
         engine_fn (whether adapter or pybind builtin).
      2. Otherwise: introspect the engine_fn directly — adapters have real
         Python signatures, pybind builtins are parsed from ``__doc__``.

    Falls back to ``(*args, **kwargs)`` when extraction fails so the wrapper
    still works (``help()`` will be uninformative for that op).
    """
    fn = entry.engine_fn
    # An adapter that stands in for one pybind11 builtin and keeps its
    # signature (``_keep_engine_signature`` — the joins, ``outer``) carries
    # ``__wrapped__``: its signature is the builtin's docstring, renamed.
    # Every other adapter — the arithmetic ones included, which set their own
    # ``(input, other)`` ``__signature__`` — is introspected directly, with
    # ``annotation_format=Format.FORWARDREF`` (PEP 749) so a name held under
    # ``TYPE_CHECKING`` (``Tensor`` in ``_adapters``) comes back as a
    # ``ForwardRef`` rather than raising ``NameError``.
    pybind_fn = getattr(fn, "__wrapped__", None)
    if pybind_fn is not None:
        # Adapter wrapping a pybind11 builtin: parse the original docstring
        # directly so the signature uses pybind11 param names, then rename.
        # A list-taking one (the joins) goes on to the ``tensors`` rewrite
        # below, as the bare builtin it wraps would.
        sig = _parse_pybind_signature(pybind_fn) or inspect.Signature(
            parameters=[
                inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
                inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
            ]
        )
        if entry.n_tensor_args != -1:
            return _rename_leading_tensor_params(sig, entry.n_tensor_args, entry.name)
    else:
        try:
            if _FORWARDREF is not None:
                sig = inspect.signature(fn, annotation_format=_FORWARDREF)
            else:
                sig = inspect.signature(fn)
        except (TypeError, ValueError, NameError):  # fmt: skip
            sig = _parse_pybind_signature(fn) or inspect.Signature(
                parameters=[
                    inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
                    inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
                ]
            )

    if entry.n_tensor_args == -1:
        # Replace the first parameter (which is the impl-list at the engine
        # level) with the user-facing ``tensors: list[Tensor]``.
        params = list(sig.parameters.values())
        if params:
            params[0] = inspect.Parameter("tensors", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        else:
            params = [inspect.Parameter("tensors", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        return sig.replace(parameters=params)

    return _rename_leading_tensor_params(sig, entry.n_tensor_args, entry.name)


# ── Keyword binding of the leading tensors ───────────────────────────────────

_LEADING_NAMES: dict[str, tuple[str, ...]] = {}


def _leading_tensor_names(entry: OpEntry) -> tuple[str, ...]:
    """The names ``entry``'s signature advertises for its tensor operands.

    The signature renames the callee's leading parameters (``a`` / ``b`` /
    ``a_impl``) to ``input`` / ``other``, which neither a pybind11 callee
    nor most adapters accept by keyword — so ``lucid.exp(input=x)`` and
    ``x.matmul(other=y)`` raised ``TypeError`` though both spellings are the
    ones advertised.  :func:`_bind_leading` puts them back in position.
    """
    names = _LEADING_NAMES.get(entry.name)
    if names is None:
        leading = list(_signature_for_entry(entry).parameters.values())
        taken: list[str] = []
        for param in leading[: _b_max(entry.n_tensor_args, 0)]:
            if param.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
                break
            taken.append(param.name)
        names = _LEADING_NAMES[entry.name] = tuple(taken)
    return names


# ── Wrapper factories ────────────────────────────────────────────────────────


def _make_free_fn(name: str) -> object:
    """Build a free function that unwraps Tensors → TensorImpls, calls the
    engine op, and rewraps the result.  The returned wrapper carries an
    explicit ``__signature__`` derived from the registry entry."""
    # 3.2.2: dict-indexed lookup replaces the O(N) ``for entry in _REGISTRY``
    # scan that paid ~2–3 µs per ``_make_free_fn`` call.  The cumulative
    # cost during module load is ~1 ms, but the same path is also hit by
    # late-bound name resolution (rare, but real) — keep the lookup O(1).
    e = _REGISTRY_BY_FREE_FN.get(name)
    if e is None:
        raise AttributeError(f"No op found for free function: {name}")
    fn_name = e.free_fn_name or e.name

    if e.n_tensor_args == -1:

        def _fn_list(tensors: list[Tensor], *args: object, **kwargs: object) -> object:
            impls = [_unwrap(t) for t in tensors]
            result = e.engine_fn(impls, *args, **kwargs)
            if e.returns_tensor:
                if isinstance(result, (list, tuple)):
                    return type(result)(_wrap(r) for r in result)
                return _wrap(result)
            return result

        _fn_list.__name__ = fn_name
        _fn_list.__qualname__ = fn_name
        _fn_list.__signature__ = _signature_for_entry(e)  # type: ignore[attr-defined]
        return _fn_list

    leading = _leading_tensor_names(e)

    def _fn(*args: object, **kwargs: object) -> object:
        if kwargs and leading:
            args, kwargs = _bind_leading(fn_name, leading, args, kwargs)
        proc: list[object] = []
        for i, a in enumerate(args):
            impl = _promotion.try_unwrap(a) if i < e.n_tensor_args else None
            proc.append(a if impl is None else impl)
        result = e.engine_fn(*proc, **kwargs)
        if e.returns_tensor:
            if isinstance(result, (list, tuple)):
                return type(result)(_wrap(r) for r in result)
            if proc and result is proc[0]:
                return args[0]
            return _wrap(result)
        return result

    _fn.__name__ = fn_name
    _fn.__qualname__ = fn_name
    _fn.__signature__ = _signature_for_entry(e)  # type: ignore[attr-defined]
    return _fn


# ── Module population ────────────────────────────────────────────────────────


_FREE_FN_NAMES: set[str] = set()


def _populate_free_fns() -> None:
    for entry in _REGISTRY:
        fn_name = entry.free_fn_name
        if fn_name is None:
            continue
        if fn_name in _FREE_FN_NAMES:
            continue
        _FREE_FN_NAMES.add(fn_name)
        globals()[fn_name] = _make_free_fn(fn_name)


_populate_free_fns()

# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks
# for them, so a name that exists under ``TYPE_CHECKING`` alone makes
# ``inspect.signature`` raise NameError — which is what help(), an IDE
# and the docs build all call.
from lucid._tensor.tensor import Tensor


def _tensor_typed(annotation: object) -> object:
    """``annotation`` with every engine ``TensorImpl`` in it read as ``Tensor``."""
    if annotation is _C_engine.TensorImpl:
        return Tensor
    args = typing.get_args(annotation)
    typed = tuple(_tensor_typed(arg) for arg in args)
    if typed == args:
        return annotation
    origin = typing.get_origin(annotation)
    if origin is types.UnionType:
        return functools.reduce(operator.or_, typed)
    return origin[typed]


def _publish_signatures() -> None:
    """Retype each free function's signature as its caller sees it.

    The signatures are read from the adapters, which take and give the
    engine's ``TensorImpl``; the wrapper unwraps and wraps around them, so
    ``lucid.contiguous`` takes and returns a ``Tensor``, as the stub says.
    ``Tensor`` is importable only here, after the wrappers are made.
    """
    for name in _FREE_FN_NAMES:
        fn = globals()[name]
        sig = inspect.signature(fn)
        fn.__dict__["__signature__"] = sig.replace(
            parameters=[
                p.replace(annotation=_tensor_typed(p.annotation)) for p in sig.parameters.values()
            ],
            return_annotation=_tensor_typed(sig.return_annotation),
        )


_publish_signatures()
