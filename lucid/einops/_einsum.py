"""Front end of :func:`lucid.einops.einsum` — parse, validate, normalise.

The engine's contraction planner assumes each operand carries every label
at most once, every label has one size across all operands and the
equation is written out explicitly without ``...``.  An operand whose
subscripts repeat (``"ii->i"``, ``"iji->j"``) broke that assumption
silently: the planner treated the two axes as independent and summed the
whole matrix, or returned the input unchanged.

This module establishes the planner's preconditions instead of the
planner guessing around them.  Each operand is reduced to its diagonal
over a repeated label (a strided view, so autograd follows it), ``...``
is spelled out as fresh labels, a size-1 axis that broadcasts against a
longer one is squeezed away, and the output is made explicit.  Malformed
equations are refused here, at the public boundary, with a message that
names the offending subscript.
"""

import string

from lucid._C import engine as _C_engine

_ELLIPSIS = "..."

# A parsed term: subscript letters in order, ``None`` where ``...`` stood.
_Term = list[str | None]


def _invalid(message: str) -> _C_engine.InvalidArgument:
    return _C_engine.InvalidArgument(f"einsum(): {message}")


def _split_equation(equation: str) -> tuple[str, str | None]:
    eq = "".join(equation.split())
    arrow = eq.find("->")
    if arrow < 0:
        return eq, None
    return eq[:arrow], eq[arrow + 2 :]


def _parse_term(term: str, where: str) -> _Term:
    tokens: _Term = []
    i = 0
    while i < len(term):
        c = term[i]
        if c == ".":
            if term[i : i + 3] != _ELLIPSIS:
                raise _invalid(f"found '.' in {where} that is not part of '...'")
            if None in tokens:
                raise _invalid(f"{where} has more than one '...'")
            tokens.append(None)
            i += 3
            continue
        if c not in string.ascii_letters:
            raise _invalid(f"invalid subscript {c!r} in {where}; subscripts must be in [a-zA-Z]")
        tokens.append(c)
        i += 1
    return tokens


def _ellipsis_dims(terms: list[_Term], impls: list[_C_engine.TensorImpl]) -> list[int]:
    """Return how many axes ``...`` covers in each operand (0 without one)."""
    dims: list[int] = []
    for k, (term, impl) in enumerate(zip(terms, impls, strict=True)):
        n_labels = len(term) - (1 if None in term else 0)
        ndim = len(impl.shape)
        extra = ndim - n_labels
        if extra < 0 or (extra > 0 and None not in term):
            raise _C_engine.ShapeMismatch(
                f"einsum(): operand {k} has {ndim} dimensions but its subscripts "
                f"{''.join(t or _ELLIPSIS for t in term)!r} name {n_labels}"
            )
        dims.append(extra)
    return dims


def _fresh_labels(used: set[str], count: int) -> list[str]:
    free = [c for c in string.ascii_letters if c not in used]
    if count > len(free):
        raise _invalid(
            f"'...' spans {count} dimensions but only {len(free)} subscript "
            "letters are left unused by the equation"
        )
    return free[:count]


def _expand(term: _Term, ell: list[str]) -> list[str]:
    out: list[str] = []
    for t in term:
        out.extend(ell if t is None else [t])
    return out


def _output_labels(rhs: str | None, labels: list[list[str]], ell: list[str]) -> list[str]:
    """Spell the output out — implicit form included — and validate it."""
    if rhs is None:
        counts: dict[str, int] = {}
        for term in labels:
            for c in term:
                counts[c] = counts.get(c, 0) + 1
        ell_set = set(ell)
        return ell + sorted(c for c, n in counts.items() if n == 1 and c not in ell_set)
    out = _expand(_parse_term(rhs, "the output"), ell)
    seen_in = {c for term in labels for c in term}
    seen_out: set[str] = set()
    for c in out:
        if c in seen_out:
            raise _invalid(f"output subscript {c!r} appears more than once")
        if c not in seen_in:
            raise _invalid(f"output subscript {c!r} does not appear in any input")
        seen_out.add(c)
    return out


def _take_diagonals(
    impl: _C_engine.TensorImpl, labels: list[str], k: int
) -> tuple[_C_engine.TensorImpl, list[str]]:
    """Collapse every label repeated within one operand onto its diagonal."""
    while len(set(labels)) != len(labels):
        c = next(c for c in labels if labels.count(c) > 1)
        p = labels.index(c)
        q = labels.index(c, p + 1)
        shape = impl.shape
        if shape[p] != shape[q]:
            raise _C_engine.ShapeMismatch(
                f"einsum(): subscript {c!r} is repeated in operand {k} over "
                f"dimensions of different sizes, {shape[p]} != {shape[q]}"
            )
        # ``diagonal`` drops both axes and appends the diagonal last.
        impl = _C_engine.diagonal(impl, 0, p, q)
        labels = [x for i, x in enumerate(labels) if i not in (p, q)] + [c]
    return impl, labels


def _label_sizes(impls: list[_C_engine.TensorImpl], labels: list[list[str]]) -> dict[str, int]:
    """Broadcast each label's size across operands (equal, or 1)."""
    sizes: dict[str, int] = {}
    for k, (impl, term) in enumerate(zip(impls, labels, strict=True)):
        for c, n in zip(term, impl.shape, strict=True):
            m = sizes.get(c, 1)
            if m != n and 1 not in (m, n):
                raise _C_engine.ShapeMismatch(
                    f"einsum(): subscript {c!r} has size {n} in operand {k}, "
                    f"which does not broadcast with size {m}"
                )
            sizes[c] = n if m == 1 else m
    return sizes


def _drop_broadcast_axes(
    impl: _C_engine.TensorImpl, labels: list[str], sizes: dict[str, int]
) -> tuple[_C_engine.TensorImpl, list[str]]:
    """Squeeze size-1 axes whose label is longer elsewhere.

    The operand is constant along such a label, so it can leave the term:
    a kept label still comes from the operand that carries it in full, a
    summed one is summed over that operand alone.
    """
    shape = impl.shape
    for i in reversed(range(len(labels))):
        if shape[i] == 1 and sizes[labels[i]] != 1:
            impl = _C_engine.squeeze(impl, i)
            labels = labels[:i] + labels[i + 1 :]
    return impl, labels


def normalise(
    equation: str, impls: list[_C_engine.TensorImpl]
) -> tuple[str, list[_C_engine.TensorImpl]]:
    """Return an explicit equation and operands the engine can contract.

    Parameters
    ----------
    equation : str
        The equation as the caller wrote it — implicit or explicit, with
        ``...`` and repeated subscripts allowed.
    impls : list of TensorImpl
        The operands, already promoted to one dtype.

    Returns
    -------
    tuple of (str, list of TensorImpl)
        An equation with ``->`` and no ``...`` in which every subscript
        occurs at most once per operand and has one size across operands,
        and the operands reshaped to match it (diagonals taken, broadcast
        size-1 axes squeezed).

    Raises
    ------
    lucid._C.engine.InvalidArgument
        The equation is malformed.
    lucid._C.engine.ShapeMismatch
        The operands' shapes do not fit the equation.
    """
    lhs, rhs = _split_equation(equation)
    raw_terms = lhs.split(",")
    if len(raw_terms) != len(impls):
        raise _invalid(
            f"the equation names {len(raw_terms)} operand(s) but " f"{len(impls)} were given"
        )
    terms = [_parse_term(t, f"operand {k}") for k, t in enumerate(raw_terms)]
    ell_dims = _ellipsis_dims(terms, impls)
    used = {c for c in lhs + (rhs or "") if c in string.ascii_letters}
    ell = _fresh_labels(used, max(ell_dims, default=0))
    labels = [_expand(term, ell[len(ell) - n :]) for term, n in zip(terms, ell_dims, strict=True)]
    out = _output_labels(rhs, labels, ell)

    diag = [
        _take_diagonals(impl, lab, k)
        for k, (impl, lab) in enumerate(zip(impls, labels, strict=True))
    ]
    sizes = _label_sizes([d[0] for d in diag], [d[1] for d in diag])
    squeezed = [_drop_broadcast_axes(impl, lab, sizes) for impl, lab in diag]
    spec = ",".join("".join(s[1]) for s in squeezed)
    return f"{spec}->{''.join(out)}", [s[0] for s in squeezed]
