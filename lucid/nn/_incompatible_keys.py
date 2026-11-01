"""The result type of ``Module.load_state_dict``.

Its own module so that ``Module`` can name it in a signature: ``_state_dict``
imports ``Module``, so defining it there would be a cycle.  The type stays
private, as in the reference.
"""

from typing import NamedTuple, final


@final
class _IncompatibleKeys(NamedTuple):
    """Result tuple returned by ``Module.load_state_dict``.

    Lightweight named tuple summarising the diff between the
    checkpoint's ``state_dict`` keys and the module's actual parameter
    / buffer names.  Each field is a list of strings; the loader is
    advisory — keys reported here are merely *not loaded*, not raised.

    Attributes
    ----------
    missing_keys : list[str]
        Module parameter / buffer names not present in the checkpoint.
        These end up using the module's existing (init-time) values.
    unexpected_keys : list[str]
        Checkpoint entries with no corresponding parameter / buffer in
        the module — typically a sign of a stale checkpoint or a
        rename.
    """

    missing_keys: list[str]
    unexpected_keys: list[str]
