"""The Python half of a hooked module call's backward (``Module._prepare_backward_hooks``).

Split out of ``module.py`` -- which sits at the file-size cap -- rather
than left to grow it; nothing here is reachable except through that one
call site.
"""

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, cast, final

from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._tensor.tensor import Tensor
from lucid.nn.hooks import _GLOBAL_BACKWARD_HOOKS, _GLOBAL_BACKWARD_PRE_HOOKS

if TYPE_CHECKING:
    from lucid.nn.module import Module


@final
class _ModuleBackwardState:
    """The Python half of one hooked module call's backward.

    The engine's two barriers (``ModuleHookNode``) hold the call's
    gradients and call back into :meth:`run_pre_hooks` and
    :meth:`run_full_hooks`.  Each calls one kind of hook in turn, global
    ones first, and yields every result that is not ``None``.  The engine
    checks that result against the gradients it replaces — the count, then
    each gradient's dtype, device and shape — and puts it in place before
    the generator goes on, so the next hook is handed only a tuple the
    engine accepted, and what flows on is the last hook's.  Nothing is
    filtered here: an entry that is not a tensor reaches the engine, which
    refuses it.

    The engine's state holds this object through its two bound methods, so
    it must hold nothing that leads back to that state: the garbage
    collector cannot see the engine's edges, and such a cycle would keep
    the module and every call's state alive forever (LCD-167).  It holds
    the module, as the reference's graph does, so a module that is only
    reachable from a pending graph still runs its hooks.
    """

    def __init__(self, module: Module) -> None:
        self.module = module

    def run_pre_hooks(
        self, grad_output_impls: tuple[_C_engine.TensorImpl | None, ...]
    ) -> Iterator[object]:
        """Run the backward pre-hooks on the output gradients."""
        grad_outputs = _wrap_grads(grad_output_impls)
        hooks = (
            *_GLOBAL_BACKWARD_PRE_HOOKS.values(),
            *self.module._backward_pre_hooks.values(),
        )
        for hook in hooks:
            result = hook(self.module, grad_outputs)
            if result is not None:
                yield result
                grad_outputs = tuple(cast(Iterable[Tensor | None], result))

    def run_full_hooks(
        self,
        grad_input_impls: tuple[_C_engine.TensorImpl | None, ...],
        grad_output_impls: tuple[_C_engine.TensorImpl | None, ...],
    ) -> Iterator[object]:
        """Run the full backward hooks on the input and output gradients."""
        grad_inputs = _wrap_grads(grad_input_impls)
        grad_outputs = _wrap_grads(grad_output_impls)
        hooks = (
            *_GLOBAL_BACKWARD_HOOKS.values(),
            *self.module._backward_hooks.values(),
        )
        for hook in hooks:
            result = hook(self.module, grad_inputs, grad_outputs)
            if result is not None:
                yield result
                grad_inputs = tuple(cast(Iterable[Tensor | None], result))


def _wrap_grads(
    impls: tuple[_C_engine.TensorImpl | None, ...],
) -> tuple[Tensor | None, ...]:
    """The engine's gradients as the tensors a backward hook is handed."""
    return tuple(None if g is None else _wrap(g) for g in impls)
