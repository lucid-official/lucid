"""
Gradient checkpointing (``lucid.autograd.checkpoint``).

Gradient checkpointing trades compute for memory: instead of storing
all intermediate activations during the forward pass, only the inputs
to a *segment* are kept.  During the backward pass the segment is
re-executed under ``enable_grad`` to rebuild the local computation graph,
then backpropagated through it.

This is especially useful for large models (transformers, deep ResNets)
where GPU memory is the bottleneck.

Usage
-----
.. code-block:: python

    from lucid.autograd import checkpoint

    def segment(x):
        return model_block(x)

    y = checkpoint(segment, x)    # memory-efficient
    loss = criterion(y, target)
    loss.backward()

Notes
-----
* ``function`` may return one tensor or a tuple of them; each output's
  gradient seeds the recomputed graph.
* RNG state is saved before the forward and restored for the
  recomputation (``preserve_rng_state=True``, the default), so a dropout
  inside the segment masks the same units both times.  Without it the
  recomputed segment drew a fresh mask and the gradient belonged to a
  different network than the one that produced the loss.
* ``use_reentrant=True`` (the default, for compatibility) derives the
  output's ``requires_grad`` from the positional inputs alone, so a
  segment fed only constants produces no gradient for the parameters it
  closes over — silently, while the same block without ``checkpoint``
  trains.  ``use_reentrant=False`` keeps the node in the graph and is the
  safer choice; it is the default in the reference framework for the
  same reason.
"""

from typing import Callable, cast, final, override

from lucid._factories.creation import zeros
from lucid._factories.random import get_rng_state, set_rng_state
from lucid._tensor.tensor import Tensor
from lucid.autograd._grad_mode import no_grad, enable_grad
from lucid.autograd.function import Function, FunctionCtx


def _roots_and_seeds(
    output: object, grad_outputs: tuple[Tensor | None, ...]
) -> tuple[list[Tensor], list[Tensor]]:
    """Pair each recomputed output that needs a gradient with its seed.

    ``Function.apply`` hands ``backward`` one gradient per *tensor* output,
    so a ``None`` or a count beside the tensors gets none.  A seed arrives
    shaped ``(1,)`` for a 0-d output — harmless for an ordinary backward,
    which broadcasts, but the engine checks the shape exactly, so a segment
    returning a scalar could not be checkpointed at all.
    """
    outputs = output if isinstance(output, tuple) else (output,)
    tensor_outputs = [o for o in outputs if isinstance(o, Tensor)]
    roots: list[Tensor] = []
    seeds: list[Tensor] = []
    for out, seed in zip(tensor_outputs, grad_outputs, strict=True):
        if seed is None or not out.requires_grad:
            continue
        if tuple(seed.shape) != tuple(out.shape):
            seed = seed.reshape(out.shape)
        roots.append(out)
        seeds.append(seed)
    return roots, seeds


def checkpoint(
    function: Callable[..., Tensor | tuple[Tensor, ...]],
    *args: Tensor,
    preserve_rng_state: bool = True,
    use_reentrant: bool = True,
    **kwargs: object,
) -> Tensor | tuple[Tensor, ...]:
    """Run *function* under gradient checkpointing.

    Executes ``function(*args, **kwargs)`` during the forward pass
    **without** tracking intermediate activations (``no_grad`` context).
    During the backward pass the function is re-executed under
    ``enable_grad`` to reconstruct the local autograd graph, and the
    gradients are computed through that graph.

    Parameters
    ----------
    function : callable
        The differentiable segment to checkpoint.  Must accept tensors as
        positional arguments and return a :class:`~lucid.Tensor` or a
        tuple of them.
    *args : Tensor
        Positional tensor inputs to *function*.
    preserve_rng_state : bool
        Restore the random state the forward saw before recomputing, so
        stochastic layers draw the same values twice.  Default ``True``.
    use_reentrant : bool
        ``True`` (default, kept for backward compatibility) ties the
        recomputation to the positional inputs: if none of them requires
        grad, backward never runs and nothing inside the segment receives
        a gradient — including parameters it closed over.  ``False``
        anchors the node in the graph so the segment is always recomputed
        and those parameters accumulate normally.  Prefer ``False``
        unless something depends on the old behaviour.
    **kwargs
        Extra keyword arguments forwarded to *function* on both the
        forward and recomputation passes.

    Returns
    -------
    Tensor or tuple of Tensor
        Output of ``function(*args, **kwargs)``.

    Examples
    --------
    >>> import lucid
    >>> W = lucid.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    >>> b = lucid.tensor([0.0, 0.0], requires_grad=True)
    >>> x = lucid.tensor([[1.0, -2.0]], requires_grad=True)
    >>> def block(x):
    ...     return lucid.nn.functional.relu(x @ W + b)
    >>> y = lucid.autograd.checkpoint(block, x, use_reentrant=False)
    >>> y.sum().backward()
    >>> x.grad
    tensor([[1., 0.]])
    """
    fn = function
    kw = kwargs
    n_args = len(args)
    # Non-reentrant mode appends one extra input (see below), so backward
    # has to know how many gradients the caller is actually owed.
    n_inputs = n_args if use_reentrant else n_args + 1

    @final
    class _CheckpointFn(Function):
        _once_differentiable = True  # second derivatives through it are unverified
        _exact_derivative = True  # backward differentiates a re-run of ``fn``

        @override
        @staticmethod
        def forward(ctx: FunctionCtx, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
            real = inputs[:n_args]
            ctx.save_for_backward(*real)
            # The segment runs twice, and the second run has to draw what
            # the first drew — or a dropout inside it masks different units
            # and the gradient belongs to a different network.
            ctx.rng_state = get_rng_state() if preserve_rng_state else None
            # Run without tracking so intermediate activations are NOT stored.
            with no_grad():
                output = fn(*real, **kw)
            return output

        @override
        @staticmethod
        def backward(
            ctx: FunctionCtx, *grad_outputs: Tensor
        ) -> tuple[Tensor | None, ...]:
            inputs = ctx.saved_tensors

            # Detach inputs so the re-run doesn't accumulate into their .grad
            # before we collect the fresh gradients below.
            detached = tuple(t.detach().requires_grad_(t.requires_grad) for t in inputs)

            # Re-run the forward segment to rebuild the local graph.  Anything
            # the segment closed over — module parameters, most often — is
            # part of that graph and accumulates through the backward below.
            saved_state = cast(Tensor | None, ctx.rng_state)
            resume = get_rng_state() if saved_state is not None else None
            if saved_state is not None:
                set_rng_state(saved_state)
            try:
                with enable_grad():
                    output = fn(*detached, **kw)
            finally:
                if resume is not None:
                    set_rng_state(resume)

            # Backward through the re-computed graph, every output seeded
            # with its own gradient.
            roots, seeds = _roots_and_seeds(output, grad_outputs)
            if roots:
                from lucid.autograd._backward import backward

                backward(roots, seeds)

            grads: tuple[Tensor | None, ...] = tuple(
                t.grad if t.requires_grad else None for t in detached
            )
            # Trailing ``None`` for the anchor, when one was added.
            return grads if n_inputs == n_args else grads + (None,)

    if use_reentrant:
        return cast(Tensor | tuple[Tensor, ...], _CheckpointFn.apply(*args))

    # ── non-reentrant ────────────────────────────────────────────────────
    # The reentrant form derives the output's ``requires_grad`` from its
    # positional inputs alone.  A segment whose inputs are all constants but
    # which closes over trainable parameters — a first layer fed raw data, a
    # block after a frozen stem — therefore produces an output that requires
    # no grad, so backward never runs and *nothing* is recomputed: the
    # parameters silently receive no gradient at all, while the identical
    # block without ``checkpoint`` trains normally.
    #
    # Passing a trainable scalar the segment never reads is enough to keep
    # the node in the graph.  Its own gradient is discarded; what matters is
    # that backward fires, the segment is recomputed under ``enable_grad``,
    # and the closed-over parameters accumulate.
    anchor = zeros(1, requires_grad=True)
    return cast(Tensor | tuple[Tensor, ...], _CheckpointFn.apply(*args, anchor))


__all__ = ["checkpoint"]
