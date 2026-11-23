"""Public aliases: one op under a second spelling.

``lucid.multiply`` is ``lucid.mul``, ``lucid.arctan2`` is ``lucid.atan2``: the
reference publishes both names and every caller expects them to take the same
calls.  They did not — each alias was a function of its own, re-annotated by
hand, and the annotation is what the tensor-argument guard reads
(:mod:`lucid._tensor_args`).  ``multiply(input: Tensor, ...)`` therefore
refused ``multiply(2.0, t)`` while ``mul(2.0, t)`` took it.

This table is the one place an alias is declared.  An entry is one of three
kinds, and ``lucid/test/unit/ops/test_op_aliases.py`` holds all three to their
target:

* **derived** — no function of its own.  The registry
  (:func:`lucid._ops._registry._derived_entries`) builds the alias's entry from
  its target's, so it shares the target's adapter, signature, validation and
  engine path, and cannot differ from it.  ``fixed`` binds keywords the
  reference's alias does not take (``true_divide`` is ``div`` with
  ``rounding_mode=None``) and drops them from the signature.
* **written** — a function the package writes for both spellings, whose two
  signatures are equal today.  The test keeps them equal.
* **declared difference** — the reference's own spelling differs from the
  target's.  ``differs`` says why and ``params`` pins the parameters the alias
  has, so it cannot drift further.  An alias that differs from its target
  without being declared here fails the test.

A new alias is added here; the test finds a hand-written forwarder that is not.
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class OpAlias:
    """``name``, a public spelling of the op ``target``.

    Attributes
    ----------
    name : str
        The alias, as ``lucid.<name>`` spells it.
    target : str
        The canonical op, as ``lucid.<target>`` spells it.
    derived : bool
        Built from the target's registry entry (see the module docstring).
    fixed : Mapping[str, object]
        Keywords of the target the alias fixes to a value and does not take.
        Only a derived alias has any.
    differs : str
        Why the alias deliberately has a signature other than its target's;
        empty when it has the target's.
    params : tuple[str, ...]
        The parameter names a declared difference has, in order.
    """

    name: str
    target: str
    derived: bool = False
    fixed: Mapping[str, object] = field(default_factory=dict)
    differs: str = ""
    params: tuple[str, ...] = ()


ALIASES: tuple[OpAlias, ...] = (
    # ── derived: the registry builds them from the target's entry ──────────
    OpAlias("multiply", "mul", derived=True),
    OpAlias("subtract", "sub", derived=True),
    OpAlias("divide", "div", derived=True),
    OpAlias("true_divide", "div", derived=True, fixed={"rounding_mode": None}),
    OpAlias("arctan2", "atan2", derived=True),
    OpAlias("absolute", "abs", derived=True),
    OpAlias("negative", "neg", derived=True),
    OpAlias("fix", "trunc", derived=True),
    OpAlias("row_stack", "vstack", derived=True),
    OpAlias("concat", "cat", derived=True),
    # ── written for both spellings, equal ──────────────────────────────────
    OpAlias("clamp", "clip"),
    OpAlias("asin", "arcsin"),
    OpAlias("acos", "arccos"),
    OpAlias("atan", "arctan"),
    OpAlias("greater", "gt"),
    OpAlias("greater_equal", "ge"),
    OpAlias("less", "lt"),
    OpAlias("less_equal", "le"),
    OpAlias("not_equal", "ne"),
    OpAlias("acosh", "arccosh"),
    OpAlias("asinh", "arcsinh"),
    OpAlias("atanh", "arctanh"),
    OpAlias("moveaxis", "movedim"),
    # ── declared differences: the reference's own spelling ─────────────────
    OpAlias(
        "concatenate",
        "cat",
        differs="the reference spells the axis `axis`; `dim` stays for callers of `cat`",
        params=("tensors", "dim", "axis"),
    ),
    OpAlias(
        "swapaxes",
        "transpose",
        differs="the reference's swapaxes names its axes `axis0` / `axis1` and requires both; "
        "`transpose` with no axes reverses the last two",
        params=("input", "axis0", "axis1"),
    ),
    OpAlias(
        "swapdims",
        "transpose",
        differs="the reference's swapdims requires both dims; `transpose` with none reverses "
        "the last two",
        params=("input", "dim0", "dim1"),
    ),
    OpAlias(
        "argwhere",
        "nonzero",
        differs="argwhere is `nonzero` with `as_tuple=False` and takes no other form",
        params=("input",),
    ),
)

ALIAS_NAMES: frozenset[str] = frozenset(alias.name for alias in ALIASES)


def narrowed(
    adapter: Callable[..., object], name: str, fixed: Mapping[str, object]
) -> Callable[..., object]:
    """``adapter`` with the keywords in ``fixed`` bound and gone from its signature.

    The wrapper of a registry op forwards what it is given, so a keyword the
    alias fixes is refused by the adapter's own parameter list, as an
    unexpected keyword is: ``true_divide(a, b, rounding_mode="floor")`` raises
    ``TypeError``, as the reference's does.

    Parameters
    ----------
    adapter : Callable
        The target's engine function or adapter.
    name : str
        The alias's name.
    fixed : Mapping[str, object]
        The keywords to bind, with their values.

    Returns
    -------
    Callable
        An adapter taking ``adapter``'s parameters but those in ``fixed``.
    """
    signature = inspect.signature(adapter)
    missing = fixed.keys() - signature.parameters.keys()
    if missing:
        raise ValueError(f"{name}() fixes {sorted(missing)}, which its target does not take")

    def _alias(*operands: object) -> object:
        return adapter(*operands, **fixed)

    _alias.__name__ = _alias.__qualname__ = name
    _alias.__dict__["__signature__"] = signature.replace(
        parameters=[p for p in signature.parameters.values() if p.name not in fixed]
    )
    return _alias
