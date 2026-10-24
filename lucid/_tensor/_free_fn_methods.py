"""``Tensor`` methods that are ``lucid`` free functions with ``self`` first.

``x.expm1_()`` existed and ``x.expm1()`` did not: the in-place methods
(:mod:`lucid._tensor._methods`) were written one family at a time, and the
out-of-place method each one's free function implies was left out — 36 of
them, where the reference has every one.  Each name in
:data:`FREE_FN_METHODS` is ``lucid.<name>`` as a method: the same callee, so
the method and the free function cannot disagree, and the same parameters
after the first — names, kinds, defaults — so ``inspect.signature`` says
what the method takes and ``tools/gen_pyi.py`` writes its stub from it.

The free functions resolve only once ``import lucid`` has defined them,
after ``Tensor`` and its other methods exist, so ``lucid/__init__.py``
attaches these last (:func:`inject_free_fn_methods`) — which is also why
``Tensor`` can be imported here at run time.
"""

import annotationlib
import inspect
from collections.abc import Callable

from lucid._tensor.tensor import Tensor

#: ``x.<name>(...)`` is ``lucid.<name>(x, ...)``.  The out-of-place twins of
#: the in-place methods whose free function existed, and ``frexp``, which
#: has no in-place form in the reference either.
FREE_FN_METHODS: tuple[str, ...] = (
    "absolute",
    "acosh",
    "addbmm",
    "addmv",
    "addr",
    "arccosh",
    "arcsinh",
    "arctan2",
    "arctanh",
    "asinh",
    "atanh",
    "baddbmm",
    "conj_physical",
    "copysign",
    "digamma",
    "erfc",
    "expm1",
    "float_power",
    "frexp",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "i0",
    "lcm",
    "ldexp",
    "less",
    "less_equal",
    "lgamma",
    "logit",
    "multiply",
    "negative",
    "not_equal",
    "put",
    "sinc",
    "subtract",
    "xlogy",
)


def _as_method(name: str, fn: Callable[..., object]) -> Callable[..., object]:
    """``fn`` as a method: ``self`` in its first parameter's place."""

    def method(self: Tensor, *args: object, **kwargs: object) -> object:
        return fn(self, *args, **kwargs)

    own = inspect.signature(fn, annotation_format=annotationlib.Format.FORWARDREF)
    _, *rest = own.parameters.values()
    self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    method.__dict__["__signature__"] = own.replace(parameters=[self_param, *rest])
    method.__name__ = name
    method.__qualname__ = f"Tensor.{name}"
    method.__doc__ = (
        f"Method form of :func:`lucid.{name}`: ``x.{name}(...)`` is "
        f"``lucid.{name}(x, ...)``."
    )
    return method


def inject_free_fn_methods(tensor_cls: type) -> None:
    """Attach every :data:`FREE_FN_METHODS` name to ``tensor_cls``.

    Each is bound to the function ``lucid.<name>`` resolves to now, so it
    must run once that resolves: :mod:`lucid` calls it as it finishes.
    """
    import lucid

    for name in FREE_FN_METHODS:
        setattr(tensor_cls, name, _as_method(name, getattr(lucid, name)))
