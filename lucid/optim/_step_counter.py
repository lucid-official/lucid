"""
Which calls count as an optimizer step, for the LR schedulers on it.

A scheduler stepped before its optimizer skips the schedule's first value,
and it can only say so if the optimizer's steps are counted.  The first
scheduler built on an optimizer puts a counter in place of
``optimizer.step``.  A step that does not go through ``optimizer.step`` —
a compiled one — reports itself with :func:`_record_step`.
"""

from collections.abc import Callable
import functools
import types
from typing import TYPE_CHECKING, override
import weakref

if TYPE_CHECKING:
    from lucid.optim.optimizer import Optimizer

#: Marks a step that reports itself: the counter, and a compiled step that
#: takes ``optimizer.step`` over.  A step without the mark was put in place
#: after the scheduler, which then cannot see the optimizer step.  The
#: reference framework's name, so code that looks for it finds it.
_STEP_MARK = "_wrapped_by_lr_sched"

#: The optimizer attribute that says it stepped since a scheduler first
#: watched it (the reference's name).  Absent until then, so the steps
#: taken before the first scheduler was built do not count.
_STEPPED = "_opt_called"


def _counts_as_step[F: Callable[..., object]](fn: F) -> F:
    """Mark ``fn`` as a step that reports itself through :func:`_record_step`."""
    setattr(fn, _STEP_MARK, True)
    return fn


def _record_step(optimizer: object) -> None:
    """Note that ``optimizer`` stepped, if a scheduler is watching it."""
    state = vars(optimizer)
    if _STEPPED in state:
        state[_STEPPED] = True


@_counts_as_step
class _CountedStep:
    """``optimizer.step`` once a scheduler watches it: note the call, then step.

    It carries the step's name, docstring and ``__wrapped__``, as the
    reference's counter does.  A class rather than a closure, because it
    sits in the optimizer's ``__dict__``, which is how the optimizer
    pickles.  It holds the optimizer weakly: a strong reference from inside
    the optimizer is a cycle, and that kept the parameters, gradients and
    state alive until the cyclic collector ran.
    """

    def __init__(
        self, optimizer: Optimizer, inner: Callable[..., object] | None
    ) -> None:
        self._optimizer = weakref.ref(optimizer)
        # A step already on the instance is the one that runs.  A method of
        # this optimizer is kept unbound, as the class's step (``None``,
        # looked up at each call) is: bound, it would hold the optimizer.
        self._method: Callable[..., object] | None = None
        self._inner: Callable[..., object] | None = None
        if isinstance(inner, types.MethodType) and inner.__self__ is optimizer:
            self._method = inner.__func__
        else:
            self._inner = inner
        functools.update_wrapper(
            self, self._method or self._inner or type(optimizer).step
        )

    def on(self, optimizer: Optimizer) -> _CountedStep:
        """The same step, counted for ``optimizer`` — a copy's own counter."""
        if self._method is not None:
            return _CountedStep(optimizer, types.MethodType(self._method, optimizer))
        return _CountedStep(optimizer, self._inner)

    @override
    def __getstate__(self) -> dict[str, object]:
        """Pickle the step alone; the optimizer's ``__setstate__`` rebinds it."""
        return {"_method": self._method, "_inner": self._inner}

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Record the step, then take it, forwarding every argument."""
        optimizer = self._optimizer()
        if optimizer is None:
            raise ReferenceError(
                "optimizer.step was called after its optimizer was freed"
            )
        _record_step(optimizer)
        if self._inner is not None:
            return self._inner(*args, **kwargs)
        step: Callable[..., object] = self._method or type(optimizer).step
        return step(optimizer, *args, **kwargs)


def _watch_steps(optimizer: Optimizer) -> None:
    """From now on, have ``optimizer`` note its steps for the schedulers on it."""
    own = vars(optimizer)
    if not hasattr(optimizer.step, _STEP_MARK):
        own["step"] = _CountedStep(optimizer, own.get("step"))
    own.setdefault(_STEPPED, False)
