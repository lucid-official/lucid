"""Deriving implicit Runge-Kutta tableaux instead of tabulating them.

Most implicit methods worth having are *collocation* methods: fix a set of
nodes inside the step and the entire coefficient table follows from requiring
the interpolating polynomial through those nodes to integrate exactly.  The
nodes themselves are roots of Legendre polynomials.

So none of it needs to be typed in.  That matters more here than elsewhere —
``gl6`` and ``radauIIA5`` have irrational coefficients, and a wall of
twenty-digit constants is exactly the transcription risk that still has
``dopri8`` on hold.  A wrong digit does not raise; it quietly costs an order,
which is the hardest kind of bug to notice.  Deriving them removes the risk
outright and makes the check free: a collocation method's order is fixed by
its node count, so the derivation can be verified against theory.

Everything here runs once at import, in 60-digit decimal, and lands in the
tableaux as ordinary floats.
"""

from decimal import Decimal, getcontext
from typing import Callable, Sequence

__all__: list[str] = []


# Enough headroom that the derived coefficients are exact to well past what a
# float64 can hold, so rounding happens once, at the very end.
_PRECISION = 60
_CONVERGED = Decimal(10) ** -50
# Bisection only has to get close enough for Newton to be safe; Newton then
# doubles its digits per step, so grinding the bracket down further is waste.
_BRACKET = Decimal(10) ** -8

# Roots are isolated by scanning for sign changes, then polished.  The scan
# only has to separate the roots, not locate them, so it stays coarse — this
# resolves node sets far larger than any tableau here needs, and _roots raises
# if it ever comes up short rather than returning a degenerate table.  An odd
# count avoids landing exactly on the symmetric interior root of an odd-degree
# polynomial, where a sign change cannot be seen.
_SCAN_INTERVALS = 401

getcontext().prec = _PRECISION


def _legendre(degree: int, x: Decimal) -> tuple[list[Decimal], list[Decimal]]:
    """Legendre polynomials and their derivatives up to ``degree`` at ``x``.

    Parameters
    ----------
    degree : int
        Highest degree to compute.
    x : Decimal
        Evaluation point.

    Returns
    -------
    tuple of list of Decimal
        Values ``P_0..P_degree`` and derivatives ``P'_0..P'_degree``.

    Notes
    -----
    The derivative uses ``P'_{k+1} = (2k+1) P_k + P'_{k-1}`` rather than the
    more familiar closed form, which divides by ``x**2 - 1`` and so blows up
    at exactly the endpoints Radau and Lobatto nodes sit on.
    """
    values = [Decimal(1)]
    derivatives = [Decimal(0)]
    if degree >= 1:
        values.append(x)
        derivatives.append(Decimal(1))
    while len(values) <= degree:
        k = len(values) - 1
        values.append(((2 * k + 1) * x * values[k] - k * values[k - 1]) / (k + 1))
        derivatives.append((2 * k + 1) * values[k] + derivatives[k - 1])
    return values, derivatives


_Fn = Callable[[Decimal], tuple[Decimal, Decimal]]


def _polish(f: _Fn, low: Decimal, high: Decimal) -> Decimal:
    """Bisect to a tight bracket, then finish with Newton."""
    sign_low = f(low)[0]
    for _ in range(200):
        middle = (low + high) / 2
        if f(middle)[0] * sign_low > 0:
            low = middle
        else:
            high = middle
        if high - low < _BRACKET:
            break

    x = (low + high) / 2
    for _ in range(100):
        value, slope = f(x)
        if slope == 0:
            break
        step = value / slope
        x -= step
        if abs(step) < _CONVERGED:
            break
    return x


def _roots(f: _Fn, low: Decimal, high: Decimal, expected: int) -> list[Decimal]:
    """Every root of ``f`` in ``[low, high]``, found by scanning for sign changes.

    Parameters
    ----------
    f : callable
        Returns ``(value, derivative)`` at a point.
    low, high : Decimal
        Interval to search.
    expected : int
        How many roots there should be; a mismatch raises rather than
        silently returning a short list that would go on to produce a
        degenerate tableau.

    Returns
    -------
    list of Decimal
        The roots, ascending.

    Raises
    ------
    ValueError
        If the number of roots found differs from ``expected``.
    """
    found: list[Decimal] = []
    previous_x, previous_value = low, f(low)[0]
    for index in range(1, _SCAN_INTERVALS + 1):
        x = low + (high - low) * Decimal(index) / _SCAN_INTERVALS
        value = f(x)[0]
        if value == 0:
            found.append(x)
        elif previous_value * value < 0:
            found.append(_polish(f, previous_x, x))
        previous_x, previous_value = x, value

    if len(found) != expected:
        raise ValueError(f"expected {expected} root(s) in [{low}, {high}], found {len(found)}")
    return found


def gauss_nodes(stages: int) -> list[Decimal]:
    """Gauss-Legendre collocation nodes on ``[0, 1]``.

    Parameters
    ----------
    stages : int
        Number of nodes.

    Returns
    -------
    list of Decimal
        The nodes, ascending.  A method built on them has order ``2*stages``,
        the highest any Runge-Kutta method of that stage count can reach.
    """

    def f(x: Decimal) -> tuple[Decimal, Decimal]:
        values, derivatives = _legendre(stages, x)
        return values[stages], derivatives[stages]

    return [(root + 1) / 2 for root in _roots(f, Decimal(-1), Decimal(1), stages)]


def radau_nodes(stages: int) -> list[Decimal]:
    """Radau IIA collocation nodes on ``[0, 1]``.

    Parameters
    ----------
    stages : int
        Number of nodes.

    Returns
    -------
    list of Decimal
        The nodes, ascending, always ending at ``1``.  A method built on them
        has order ``2*stages - 1`` and is L-stable, which is what makes this
        family the usual choice for stiff problems.

    Notes
    -----
    The nodes are the roots of ``P_{s-1} - P_s``.  That difference vanishes at
    ``x = 1`` for every degree, which is precisely why the family includes the
    right endpoint; the remaining roots are found after dividing it out.
    """
    if stages == 1:
        return [Decimal(1)]

    def f(x: Decimal) -> tuple[Decimal, Decimal]:
        values, derivatives = _legendre(stages, x)
        value = values[stages - 1] - values[stages]
        slope = derivatives[stages - 1] - derivatives[stages]
        return value / (x - 1), (slope * (x - 1) - value) / (x - 1) ** 2

    free = _roots(f, Decimal(-1), Decimal("0.999999"), stages - 1)
    return [(root + 1) / 2 for root in free] + [Decimal(1)]


def lobatto_nodes(stages: int) -> list[Decimal]:
    """Lobatto IIIA collocation nodes on ``[0, 1]``.

    Parameters
    ----------
    stages : int
        Number of nodes, at least two.

    Returns
    -------
    list of Decimal
        The nodes, ascending, spanning both endpoints.  The two-node case is
        the trapezoidal rule.

    Raises
    ------
    ValueError
        If ``stages`` is less than two, which no Lobatto family has.
    """
    if stages < 2:
        raise ValueError(f"Lobatto needs at least two nodes, got {stages}")
    if stages == 2:
        return [Decimal(0), Decimal(1)]

    def f(x: Decimal) -> tuple[Decimal, Decimal]:
        values, derivatives = _legendre(stages - 1, x)
        index = stages - 1
        second = (2 * x * derivatives[index] - index * (index + 1) * values[index]) / (1 - x * x)
        return derivatives[index], second

    free = _roots(f, Decimal("-0.999999"), Decimal("0.999999"), stages - 2)
    return [Decimal(0)] + [(root + 1) / 2 for root in free] + [Decimal(1)]


def _lagrange_basis(nodes: Sequence[Decimal], index: int) -> list[Decimal]:
    """Coefficients, ascending in power, of one Lagrange basis polynomial."""
    poly = [Decimal(1)]
    for other, node in enumerate(nodes):
        if other == index:
            continue
        scale = nodes[index] - node
        extended = [Decimal(0)] * (len(poly) + 1)
        for power, coefficient in enumerate(poly):
            extended[power] += -node / scale * coefficient
            extended[power + 1] += coefficient / scale
        poly = extended
    return poly


def _integrate(poly: Sequence[Decimal], upper: Decimal) -> Decimal:
    """Integrate an ascending-power polynomial from zero to ``upper``."""
    total = Decimal(0)
    for power, coefficient in enumerate(poly):
        total += coefficient * upper ** (power + 1) / (power + 1)
    return total


def collocation_tableau(
    nodes: Sequence[Decimal],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    r"""Build the coefficient table a set of collocation nodes determines.

    Parameters
    ----------
    nodes : sequence of Decimal
        Collocation nodes on ``[0, 1]``.

    Returns
    -------
    tuple
        The square stage matrix ``a`` and the solution weights ``b``, as
        plain floats.

    Notes
    -----
    :math:`a_{ij} = \int_0^{c_i} L_j`, :math:`b_j = \int_0^1 L_j`, with
    :math:`L_j` the Lagrange basis on the nodes — that is the entire
    definition.  The stage matrix comes out square and generally dense, which
    is what makes the method implicit.
    """
    basis = [_lagrange_basis(nodes, j) for j in range(len(nodes))]
    a = tuple(tuple(float(_integrate(basis[j], node)) for j in range(len(nodes))) for node in nodes)
    b = tuple(float(_integrate(basis[j], Decimal(1))) for j in range(len(nodes)))
    return a, b


def quadrature_order(b: Sequence[float], c: Sequence[float], limit: int = 12) -> int:
    r"""Highest ``p`` satisfying the quadrature conditions :math:`B(p)`.

    Parameters
    ----------
    b, c : sequence of float
        Solution weights and nodes.
    limit : int, default=12
        Highest order to test.

    Returns
    -------
    int
        The largest ``p`` with :math:`\sum_i b_i c_i^{k-1} = 1/k` for all
        ``k`` up to ``p``.

    Notes
    -----
    For a collocation method this equals the method's order, so it is a
    direct check on the derivation: the answer must come out at ``2s`` for
    Gauss and ``2s - 1`` for Radau, and anything else means a node is wrong.
    """
    order = 0
    for k in range(1, limit + 1):
        total = sum(b[i] * (c[i] ** (k - 1) if k > 1 else 1.0) for i in range(len(c)))
        if abs(total - 1.0 / k) > 1e-12:
            break
        order = k
    return order
