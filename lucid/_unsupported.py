"""Refusals for arguments that are accepted for signature parity but not implemented.

An argument that a function accepts, documents, and then ignores is the worst
kind of gap: the caller gets a plausible-looking tensor computed from
different settings than the ones they asked for, and nothing anywhere says so.
``conv_transpose2d(..., groups=4)`` silently returned a single output channel
instead of four; ``linalg.svd(..., full_matrices=True)`` silently returned the
reduced factorisation.  Both are wrong *shapes*, which means downstream code
either crashes far from the cause or — worse — broadcasts and keeps going.

These helpers turn that silence into an error at the call site.  They fire
only when the caller asks for the unimplemented behaviour; passing the default
costs nothing and changes nothing.
"""

from typing import Any

__all__ = ["unsupported_arg", "unsupported_if"]


def unsupported_arg(func: str, arg: str, value: Any, *, detail: str = "") -> None:
    """Raise for an argument whose requested value is not implemented.

    Parameters
    ----------
    func : str
        Qualified function name, used in the message.
    arg : str
        The argument the caller set.
    value : Any
        What they set it to, echoed back so the message is actionable.
    detail : str, optional
        What the implementation does instead, when that helps.
    """
    tail = f"  {detail}" if detail else ""
    raise NotImplementedError(f"{func}: {arg}={value!r} is not implemented.{tail}")


def unsupported_if(condition: bool, func: str, arg: str, value: Any, *, detail: str = "") -> None:
    """:func:`unsupported_arg`, but only when ``condition`` holds.

    Lets a call site state the refusal inline instead of wrapping it in an
    ``if``, which keeps the guard next to the argument it guards.

    Parameters
    ----------
    condition : bool
        Whether the caller asked for the unimplemented behaviour.  ``False``
        returns immediately, so passing the supported default costs nothing.
    func : str
        Qualified function name, used in the message.
    arg : str
        The argument the caller set.
    value : Any
        What they set it to, echoed back so the message is actionable.
    detail : str, optional, keyword-only
        What the implementation does instead, when that helps.

    Returns
    -------
    None
        Returns normally when ``condition`` is false.

    Raises
    ------
    NotImplementedError
        When ``condition`` is true.
    """
    if condition:
        unsupported_arg(func, arg, value, detail=detail)
