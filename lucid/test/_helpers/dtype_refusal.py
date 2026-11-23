"""The dtype refusals the engine declares, read back for the tests that must agree with them.

An op's ``OpSchema`` names the dtypes it turns away and the class it raises for them
(``kernel``: ``NotImplementedError``, the reference has no kernel; ``guard``: ``DtypeMismatch``,
a check on the input dtype), and ``SchemaGuard`` enforces it for every caller (LCD-1020,
LCD-1026).  ``test_engine_bool_refusal_schema.py`` pins the declared set, op by op.  A test of
anything else that meets a refused dtype -- a replay matrix, a shadow-allocation parity sweep, an
autocast eligibility table -- asks :func:`engine_refusal` instead of listing the refused
(op, dtype) pairs again, so the policy has one literal and every other file follows it.

The one exception to "the registry says" is :data:`UNREGISTERED`: the ops that refuse without
publishing a schema (``var`` builds its own forward, ``frac`` is composed).  They are declared
here, once, and the registry pin imports them.
"""

import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Final

import pytest

import lucid
from lucid._C import engine as _C_engine

_D: Final = _C_engine.Dtype

#: The class each kind raises.  Both are ``RuntimeError``s, and the reference's callers catch one
#: without the other, so a test holds the exact class.
KIND_CLASS: Final[dict[str, type[RuntimeError]]] = {
    "kernel": _C_engine.NotImplementedError,
    "guard": _C_engine.DtypeMismatch,
}

_BOOL_AND_INTS: Final = frozenset({_D.Bool, _D.I8, _D.I16, _D.I32, _D.I64})

#: ``op -> (kind, cast hint, refused dtypes)`` for the ops that refuse and that no
#: ``LUCID_REGISTER_OP`` publishes, so the registry does not list them: ``var`` builds its own
#: forward and is reached through ``std`` and ``var`` only, ``frac`` is composed and asks
#: ``check_schema_dtype`` of a schema of its own that is never registered.
UNREGISTERED: Final[dict[str, tuple[str, str, frozenset[_C_engine.Dtype]]]] = {
    "var": ("guard", "a floating dtype, e.g. x.float()", _BOOL_AND_INTS),
    "frac": ("kernel", "a floating dtype, e.g. x.float()", _BOOL_AND_INTS),
}


@dataclass(frozen=True)
class Refusal:
    """What the engine says for one (op, dtype): the class and the message it raises."""

    op: str
    kind: str
    dtype: str
    hint: str

    @property
    def error(self) -> type[RuntimeError]:
        return KIND_CLASS[self.kind]

    @property
    def message(self) -> str:
        return f"{self.op} does not accept {self.dtype} — cast to {self.hint}"

    def raises(self) -> AbstractContextManager[pytest.ExceptionInfo[RuntimeError]]:
        """``pytest.raises`` for this refusal: the declared class, the message naming op and dtype."""
        return pytest.raises(self.error, match=re.escape(self.message))


def engine_refusal(op: str, dtype: lucid.dtype) -> Refusal | None:
    """The refusal the engine declares for ``op`` on ``dtype``, or ``None`` if it answers.

    ``op`` is the engine's name for the operation (``hard_sigmoid`` for ``F.hardsigmoid``,
    ``var`` for ``std``), the one its error message words.
    """
    for schema in _C_engine.op_registry_all():
        if schema.name == op:
            rule = schema.refusal_for(dtype._engine)
            break
    else:
        declared = UNREGISTERED.get(op)
        if declared is None or dtype._engine not in declared[2]:
            return None
        rule = (declared[0], declared[1])
    if rule is None:
        return None
    return Refusal(op, rule[0], dtype._name, rule[1])
