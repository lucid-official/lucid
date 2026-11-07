"""Compile probes behind ``test_compile_walk_budget.py`` (LCD-838).

Run as ``python -m lucid.test.unit.compile._jacobi_probes <group>``.  A group
prints ``ok <name>`` after each probe, so a parent that had to kill it can say
which probe never came back.  A compile fallback fails the probe: the point is
that these graphs compile, and an eager answer would prove nothing about them.
"""

import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid import Tensor
from lucid.compile import make_step
from lucid.compile._core.fallback import CompileFallbackWarning

_LA = lucid.linalg
_DTYPE = lucid.float32
_EPS = 1.1920929e-7
#: Relative to the input's largest entry.  The worst measured is 3e-6 to 8x8.
_TOL = 100 * _EPS
#: Relative to eager's largest entry.  The worst measured is 2.1e-5, to 8x8.
_SVD_GRAD_LIMIT = 1e-4
#: Exit status of a probe that returned a wrong answer, as against one that broke.
WRONG_ANSWER = 3


def _rand(m: int, n: int, seed: int) -> Tensor:
    lucid.manual_seed(838 + seed)
    return lucid.randn(m, n, device="metal", dtype=_DTYPE)


def _tensor(rows: list[list[float]] | list[float]) -> Tensor:
    return lucid.tensor(rows, device="metal", dtype=_DTYPE)


def _eye(n: int) -> Tensor:
    return lucid.eye(n, device="metal", dtype=_DTYPE)


def _symmetric(n: int, seed: int) -> Tensor:
    r = _rand(n, n, seed)
    return (r + r.mT) / 2


def _spd(n: int, seed: int) -> Tensor:
    r = _rand(n, n, seed)
    return r @ r.mT + 4 * _eye(n)


def _with_spectrum(values: list[float], seed: int) -> Tensor:
    n = len(values)
    q = _LA.qr(_rand(n, n, seed))[0]
    m = (q * _tensor(values)[None, :]) @ q.mT
    return (m + m.mT) / 2


def _hard_symmetric(n: int, seed: int = 0) -> dict[str, Tensor]:
    """Spectra Jacobi can get wrong; ``swap`` has a zero diagonal and orthogonal columns."""
    half = n // 2
    v = _rand(n, 1, seed)
    return {
        "random": _symmetric(n, seed),
        "spd": _spd(n, seed),
        "swap": _tensor([[float(i + j == n - 1) for j in range(n)] for i in range(n)]),
        "plus_minus": _with_spectrum([(-1.0) ** i for i in range(n)], seed),
        "repeated": _with_spectrum([1.0] * half + [-2.0] * (n - half), seed),
        "rank_one": v @ v.mT,
        "cluster": _with_spectrum([1.0 + 1e-4 * i for i in range(n)], seed),
        "wide": _with_spectrum([10.0 ** (3.0 - 6.0 * i / max(n - 1, 1)) for i in range(n)], seed),
        "identity": _eye(n),
        "negative_identity": -_eye(n),
        "graded": lucid.diag(_tensor([10.0 ** (-i) for i in range(n)])),
        "zero": lucid.zeros(n, n, device="metal", dtype=_DTYPE),
    }


def _max_abs(t: Tensor) -> float:
    return float(t.abs().amax().item())


def _assert_close(got: Tensor, want: Tensor, scale: float, what: str) -> None:
    err = _max_abs(got.to(_DTYPE) - want.to(_DTYPE))
    assert err <= _TOL * max(scale, 1e-30), f"{what}: off by {err:.3g} at scale {scale:.3g}"


def _assert_orthonormal(q: Tensor, what: str, tol: float = _TOL) -> None:
    drift = _max_abs(q.mT @ q - _eye(q.shape[-1]))
    assert drift <= tol, f"{what} is not orthonormal: drift {drift:.3g}"


def _assert_none_finite(t: Tensor, what: str) -> None:
    finite = int(lucid.isfinite(t).sum().item())
    assert finite == 0, f"{what} holds {finite} finite entries from a poisoned input"


def _check_eigvals(a: Tensor, w: Tensor) -> None:
    _assert_close(w, _LA.eigvalsh(a), _max_abs(a), "eigenvalues")


def _check_eigh(a: Tensor, out: tuple[Tensor, Tensor]) -> None:
    w, v = out
    _check_eigvals(a, w)
    _assert_orthonormal(v, "V")
    _assert_close(a @ v, v * w[None, :], _max_abs(a), "A V - V diag(w)")


def _check_eigh_members(a: Tensor, out: tuple[Tensor, Tensor]) -> None:
    """Each batch member is held to its own scale."""
    w, v = out
    for i in range(a.shape[0]):
        _check_eigh(a[i], (w[i], v[i]))


def _upper_triangle_matrix(a: Tensor) -> Tensor:
    return lucid.triu(a) + lucid.triu(a, 1).mT


def _lower_triangle_matrix(a: Tensor) -> Tensor:
    return lucid.tril(a) + lucid.tril(a, -1).mT


def _check_eigh_upper(a: Tensor, out: tuple[Tensor, Tensor]) -> None:
    _check_eigh(_upper_triangle_matrix(a), out)


def _check_eigh_lower(a: Tensor, out: tuple[Tensor, Tensor]) -> None:
    _check_eigh(_lower_triangle_matrix(a), out)


def _check_svd_factors(a: Tensor, out: tuple[Tensor, Tensor, Tensor]) -> None:
    u, s, vh = out
    k = min(a.shape)
    _assert_close(s, _LA.svdvals(a), _max_abs(a), "singular values")
    _assert_orthonormal(u, "U")
    _assert_orthonormal(vh.mT, "Vh")
    _assert_close((u[:, :k] * s[None, :]) @ vh[:k, :], a, _max_abs(a), "U S Vh - A")


def _check_svd(a: Tensor, out: tuple[Tensor, Tensor, Tensor]) -> None:
    u, _, vh = out
    m, n = a.shape
    assert u.shape == (m, m) and vh.shape == (n, n), f"{u.shape} {vh.shape} for {m}x{n}"
    _check_svd_factors(a, out)


def _check_thin_svd(a: Tensor, out: tuple[Tensor, Tensor, Tensor]) -> None:
    u, _, vh = out
    m, n = a.shape
    k = min(m, n)
    assert u.shape == (m, k) and vh.shape == (k, n), f"{u.shape} {vh.shape} for {m}x{n}"
    _check_svd_factors(a, out)


def _check_thin_svd_members(a: Tensor, out: tuple[Tensor, Tensor, Tensor]) -> None:
    """Each batch member is held to its own scale."""
    u, s, vh = out
    for i in range(a.shape[0]):
        _check_thin_svd(a[i], (u[i], s[i], vh[i]))


def _check_svdvals(a: Tensor, s: Tensor) -> None:
    _assert_close(s, _LA.svdvals(a), _max_abs(a), "singular values")


def _check_u(a: Tensor, u: Tensor) -> None:
    assert u.shape == (a.shape[0], a.shape[0]), f"{u.shape} for {tuple(a.shape)}"
    _assert_orthonormal(u, "U")


def _check_vh(a: Tensor, vh: Tensor) -> None:
    assert vh.shape == (a.shape[1], a.shape[1]), f"{vh.shape} for {tuple(a.shape)}"
    _assert_orthonormal(vh.mT, "Vh")


def _check_rank(a: Tensor, rank: Tensor) -> None:
    want = int(_LA.matrix_rank(a).item())
    assert int(rank.item()) == want, f"rank {int(rank.item())}, eager {want}"


def _check_cond(a: Tensor, c: Tensor) -> None:
    want = _LA.cond(a)
    _assert_close(c, want, _max_abs(want), "cond")


def _check_pinv(a: Tensor, p: Tensor) -> None:
    want = _LA.pinv(a)
    _assert_close(p, want, _max_abs(want), "pinv")
    _assert_close(a @ p @ a, a, _max_abs(a), "A P A - A")
    _assert_close(p @ a @ p, p, _max_abs(p), "P A P - P")


def _check_qr_factors(a: Tensor, out: tuple[Tensor, Tensor]) -> None:
    q, r = out
    _assert_orthonormal(q, "Q")
    assert _max_abs(lucid.tril(r, -1)) == 0.0, "R is not upper triangular"
    _assert_close(q @ r, a, _max_abs(a), "Q R - A")


def _check_qr(a: Tensor, out: tuple[Tensor, Tensor]) -> None:
    """Eager's reflector signs are LAPACK's, so its factors are the compiled ones."""
    _check_qr_factors(a, out)
    q_eager, r_eager = _LA.qr(a)
    _assert_close(out[0], q_eager, 1.0, "Q against eager")
    _assert_close(out[1], r_eager, _max_abs(a), "R against eager")


def _check_qr_columns(a: Tensor, out: tuple[Tensor, Tensor]) -> None:
    """Householder is backward stable column by column, so each is held to its own length."""
    q, r = out
    _assert_orthonormal(q, "Q")
    gap = (q @ r - a).abs()
    for j in range(a.shape[1]):
        own = _max_abs(a[:, j])
        err = _max_abs(gap[:, j])
        assert err <= _TOL * own, f"column {j}: off by {err:.3g} at its scale {own:.3g}"


def _graded_columns(m: int, n: int, grades: list[float], seed: int) -> Tensor:
    return _rand(m, n, seed) * _tensor(grades)[None, :]


def _near_parallel(scale: float) -> Tensor:
    """A lower block of parallel columns, small enough to escape an absolute cut."""
    return scale * _tensor([[1.0, 0.0, 0.0], [0.0, 6e-7, 6e-7], [0.0, 6e-7, 6e-7]])


def _deficient_inputs() -> dict[str, Tensor]:
    return {
        "full": _rand(3, 3, 0),
        "rank_one": _tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]),
        "rank_two": _tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        "near_parallel_10": _near_parallel(10.0),
        "near_parallel_1000": _near_parallel(1000.0),
        "zero": lucid.zeros(3, 3, device="metal", dtype=_DTYPE),
    }


def _check_near_parallel_pinv(a: Tensor, p: Tensor) -> None:
    scale = float(a[0, 0].item())
    want = 1.0 / (2.4e-6 * scale)
    got = float(p[1, 1].item())
    assert abs(got - want) <= 1e-3 * want, f"pinv[1, 1] = {got:.6g}, truth {want:.6g}"


def _low_rank(m: int, n: int, rank: int, seed: int) -> Tensor:
    return _rand(m, rank, seed) @ _rand(rank, n, seed + 100)


def _rank_cases(m: int, n: int) -> dict[str, Tensor]:
    """U's columns for singular values that are zero or nothing but rounding.

    ``last_row`` puts U's first column on an identity column: completing from
    that one instead of the one furthest from the span leaves nothing to normalize.
    ``ones`` cancels exactly, so every later column comes from the spare identity
    columns.  ``near`` is the graded case: at 2x2 one sweep leaves its second
    column up to 0.85 of the way along the first.
    """
    last_row = lucid.zeros(m, n, device="metal", dtype=_DTYPE)
    last_row[m - 1, 0] = 1.0
    rank_one = _low_rank(m, n, 1, 2)
    cases = {
        "first": _rand(m, n, 0),
        "second": _rand(m, n, 1),
        "zero": lucid.zeros(m, n, device="metal", dtype=_DTYPE),
        "ones": lucid.ones(m, n, device="metal", dtype=_DTYPE),
        "rank_one": rank_one,
        "last_row": last_row,
        "tiny_rank_one": rank_one * 1e-30,
        "huge_rank_one": rank_one * 1e30,
    }
    below_full = _low_rank(m, n, max(min(m, n) - 1, 1), 3)
    noise = _rand(m, n, 4)
    for delta in (1e-9, 1e-7, 1e-6, 1e-5):
        cases[f"near {delta:g}"] = below_full + delta * noise
    cases["rank k - 1"] = below_full
    cases["tiny rank k - 1"] = below_full * 1e-30
    cases["huge rank k - 1"] = below_full * 1e30
    return cases


def _thin_svd(a: Tensor) -> tuple[Tensor, ...]:
    return tuple(_LA.svd(a, full_matrices=False))


def _check_keeps_small_column(a: Tensor, p: Tensor) -> None:
    """Under the old 1e-12 cut, U stayed at length 1e-13 and pinv came back 13 orders short."""
    want = _LA.pinv(a, rtol=1e-20)
    _assert_close(p, want, _max_abs(want), "pinv of a small column")


def _check_each_eigenvalue(a: Tensor, w: Tensor) -> None:
    """To eager, each against its own size: a 1 beside a 3e38 has to come back a 1."""
    want = _LA.eigvalsh(a)
    worst = _max_abs((w.to(_DTYPE) - want) / want)
    assert worst <= _TOL, f"eigenvalues {w.tolist()}, eager {want.tolist()}"


def _check_eigh_near_limit(a: Tensor, out: tuple[Tensor, Tensor]) -> None:
    _check_each_eigenvalue(a, out[0])
    _assert_orthonormal(out[1], "V")


def _tight_diamonds(x: Tensor, depth: int) -> Tensor:
    """Both arms one op long, the shape the clone walk doubles on; the value stays ``x``."""
    for _ in range(depth):
        x = (x * 0.75) + (x * 0.25)
    return x


def _slice_concat_chain(x: Tensor, depth: int) -> Tensor:
    """The two-sided Jacobi pattern: a whole tensor rebuilt from its own slices."""
    for _ in range(depth):
        head, tail = x[:2], x[2:]
        x = lucid.cat([head * 0.5 + tail, tail * 0.25 + head], dim=0)
    return x


def _check_tight(x: Tensor, y: Tensor) -> None:
    _assert_close(y, x, _max_abs(x), "64 tight diamonds")


def _check_slice_concat(x: Tensor, y: Tensor) -> None:
    want = _slice_concat_chain(x, 32)
    _assert_close(y, want, _max_abs(want), "32 slice-concat rounds")


@dataclass(frozen=True)
class _Probe:
    """One compiled function, fed every input: one compile per shape, however many matrices."""

    name: str
    call: Callable[[Tensor], Tensor | tuple[Tensor, ...]]
    inputs: dict[str, Tensor]
    check: Callable[..., None]


def _outputs(out: Tensor | tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return (out,) if isinstance(out, Tensor) else out


def _run(probe: _Probe) -> None:
    fn = lucid.compile(probe.call)
    answers = []
    for label, a in probe.inputs.items():
        out = fn(a)
        try:
            probe.check(a, out)
        except AssertionError as failure:
            raise AssertionError(f"{probe.name} on {label}: {failure}") from failure
        answers.append(_outputs(out))
    moved = (
        _max_abs(x.to(_DTYPE) - y.to(_DTYPE)) for x, y in zip(answers[0], answers[1], strict=True)
    )
    assert max(moved) > 0.0, f"{probe.name}: the answer did not follow"
    print("ok", probe.name, flush=True)


def _eigen_group() -> list[_Probe]:
    probes = [
        _Probe(f"eigh:{n}", lambda a: tuple(_LA.eigh(a)), _hard_symmetric(n), _check_eigh)
        for n in (1, 2, 3, 4, 5, 6, 8)
    ]
    probes += [
        _Probe(f"eigvalsh:{n}", lambda a: _LA.eigvalsh(a), _hard_symmetric(n), _check_eigvals)
        for n in (3, 4, 6)
    ]
    # Two kernels in one graph: 5x5 was under the cliff alone and over it doubled.
    probes.append(
        _Probe(
            "eigvalsh_twice:5",
            lambda a: _LA.eigvalsh(a) + _LA.eigvalsh(2 * a) / 2,
            {"first": _symmetric(5, 0), "second": _symmetric(5, 1)},
            lambda a, w: _assert_close(w, 2 * _LA.eigvalsh(a), _max_abs(a), "two kernels"),
        )
    )
    return probes


def _triangle_group() -> list[_Probe]:
    """A non-symmetric input: the emitter must read the named triangle and only that."""
    probes: list[_Probe] = []
    for n in (3, 6):
        inputs = {"first": _rand(n, n, 0), "second": _rand(n, n, 1), "third": _rand(n, n, 2)}
        probes.append(
            _Probe(f"eigh_U:{n}", lambda a: tuple(_LA.eigh(a, "U")), inputs, _check_eigh_upper)
        )
        probes.append(
            _Probe(f"eigh_L:{n}", lambda a: tuple(_LA.eigh(a, "L")), inputs, _check_eigh_lower)
        )
    return probes


def _mixed_members(m: int, n: int, seed: int) -> Tensor:
    return lucid.stack(
        [
            _rand(m, n, seed),
            _low_rank(m, n, 1, seed + 1),
            lucid.zeros(m, n, device="metal", dtype=_DTYPE),
            1e20 * _rand(m, n, seed + 2),
            1e-20 * _low_rank(m, n, 2, seed + 3),
        ]
    )


def _batch_group() -> list[_Probe]:
    def members(seed: int) -> Tensor:
        return lucid.stack(
            [
                _symmetric(4, seed),
                1e-7 * _symmetric(4, seed + 1),
                1e7 * _with_spectrum([1.0, -1.0, 1.0, -1.0], seed),
                1e-20 * _spd(4, seed + 2),
                lucid.zeros(4, 4, device="metal", dtype=_DTYPE),
            ]
        )

    probes = [
        _Probe(
            "eigh_member_scales",
            lambda a: tuple(_LA.eigh(a)),
            {"first": members(0), "second": members(10)},
            _check_eigh_members,
        )
    ]
    for m, n in ((5, 3), (3, 5)):
        inputs = {"first": _mixed_members(m, n, 0), "second": _mixed_members(m, n, 10)}
        probes.append(
            _Probe(f"svd_mixed_members:{m}x{n}", _thin_svd, inputs, _check_thin_svd_members)
        )
    return probes


def _scale_group() -> list[_Probe]:
    """The rotations decide from inner products, the square of the input's scale."""
    inputs = {f"{scale:g}": _spd(4, 0) * scale for scale in (1e-7, 1e-20, 1e-30, 1e18, 1e30)}
    # LCD-848: a representable matrix whose diagonal doubles past float32's limit.
    near_limit = {
        "diagonal": _tensor([[3e38, 0.0], [0.0, 1.0]]),
        "negative": _tensor([[-3e38, 0.0], [0.0, 1.0]]),
        "coupled": _tensor([[3e38, 1e37], [1e37, 1e38]]),
        "off_diagonal": _tensor([[1.0, 3e38], [3e38, 1.0]]),
    }
    # LCD-862: Householder takes a column's norm from squares, the same scale.
    qr_scales = (1.0, 1e-30, 1e-25, 1e-15, 1e15, 1e20)
    qr_graded = {
        "ascending": [1e-25, 1e-10, 1.0, 1e15],
        "descending": [1e15, 1.0, 1e-10, 1e-25],
        "interleaved": [1.0, 1e-30, 1e20, 1e-15],
    }
    qr_probes = [
        _Probe(
            f"qr_scales:{m}x{n}",
            lambda a: tuple(_LA.qr(a)),
            {f"{s:g}": _rand(m, n, 0) * s for s in qr_scales},
            _check_qr,
        )
        for m, n in ((4, 4), (5, 3), (3, 5))
    ]
    qr_probes.append(
        _Probe(
            "qr_graded",
            lambda a: tuple(_LA.qr(a)),
            {k: _graded_columns(4, 4, v, 1) for k, v in qr_graded.items()},
            _check_qr_columns,
        )
    )
    return [
        *qr_probes,
        _Probe("svdvals_scales", lambda a: _LA.svdvals(a), inputs, _check_svdvals),
        _Probe("eigh_scales", lambda a: tuple(_LA.eigh(a)), inputs, _check_eigh),
        _Probe(
            "eigvalsh_near_limit", lambda a: _LA.eigvalsh(a), near_limit, _check_each_eigenvalue
        ),
        _Probe("eigh_near_limit", lambda a: tuple(_LA.eigh(a)), near_limit, _check_eigh_near_limit),
        _Probe(
            "eigvalsh_U_near_limit",
            lambda a: _LA.eigvalsh(a, "U"),
            near_limit,
            _check_each_eigenvalue,
        ),
    ]


def _svd_group() -> list[_Probe]:
    probes: list[_Probe] = []
    for m, n in ((4, 4), (6, 4), (4, 6), (5, 3)):
        inputs = {"first": _rand(m, n, 0), "second": _rand(m, n, 1)}
        probes.append(_Probe(f"svd:{m}x{n}", lambda a: tuple(_LA.svd(a)), inputs, _check_svd))
    # Full-matrices completion is an eigh of the longer side, over the old cliff here.
    for m, n in ((6, 4), (4, 6)):
        inputs = {"first": _rand(m, n, 0), "second": _rand(m, n, 1)}
        probes.append(_Probe(f"svd.U:{m}x{n}", lambda a: _LA.svd(a)[0], inputs, _check_u))
        probes.append(_Probe(f"svd.Vh:{m}x{n}", lambda a: _LA.svd(a)[2], inputs, _check_vh))
    # The zero second: a rank need not change between two random inputs.
    for m, n in ((4, 4), (6, 6), (5, 3)):
        inputs = {
            "first": _rand(m, n, 0),
            "zero": lucid.zeros(m, n, device="metal", dtype=_DTYPE),
            "second": _rand(m, n, 1),
        }
        probes.append(_Probe(f"svdvals:{m}x{n}", lambda a: _LA.svdvals(a), inputs, _check_svdvals))
        probes.append(_Probe(f"rank:{m}x{n}", lambda a: _LA.matrix_rank(a), inputs, _check_rank))
        probes.append(_Probe(f"pinv:{m}x{n}", lambda a: _LA.pinv(a), inputs, _check_pinv))
    for n in (4, 6):
        inputs = {"first": _rand(n, n, 0), "second": _rand(n, n, 1)}
        probes.append(_Probe(f"cond:{n}x{n}", lambda a: _LA.cond(a), inputs, _check_cond))
    return probes


def _deficient_group() -> list[_Probe]:
    """Rank the threshold must find; the near-parallel block is what an absolute cut misses."""
    deficient = _deficient_inputs()
    parallel = {"10": _near_parallel(10.0), "1000": _near_parallel(1000.0)}
    small_column = {
        f"{small:g}": _tensor([[1.0, 0.0], [0.0, -small]]) for small in (1e-8, 1e-13, 1e-16)
    }
    probes = [
        _Probe("rank_deficient", lambda a: _LA.matrix_rank(a), deficient, _check_rank),
        _Probe("pinv_deficient", lambda a: _LA.pinv(a), deficient, _check_pinv),
        _Probe("svdvals_deficient", lambda a: _LA.svdvals(a), deficient, _check_svdvals),
        _Probe("near_parallel_pinv", lambda a: _LA.pinv(a), parallel, _check_near_parallel_pinv),
        _Probe(
            "pinv_small_column",
            lambda a: _LA.pinv(a, rtol=1e-20),
            small_column,
            _check_keeps_small_column,
        ),
        _Probe(
            "svd_rank_2_of_3",
            lambda a: tuple(_LA.svd(a)),
            {f"seed {seed}": _low_rank(3, 3, 2, seed) for seed in range(8)},
            _check_svd,
        ),
    ]
    for m, n in ((2, 2), (3, 3), (5, 3), (3, 5), (4, 1), (1, 4)):
        probes.append(_Probe(f"svd_thin:{m}x{n}", _thin_svd, _rank_cases(m, n), _check_thin_svd))
    # The Python completion projects through U; a wide matrix gets it on Vh.
    for m, n in ((5, 3), (3, 5)):
        probes.append(
            _Probe(f"svd_full:{m}x{n}", lambda a: tuple(_LA.svd(a)), _rank_cases(m, n), _check_svd)
        )
    return probes


def _chain_group() -> list[_Probe]:
    inputs = {"first": _rand(4, 1, 0).reshape(4), "second": _rand(4, 1, 1).reshape(4)}
    return [
        _Probe("tight_diamonds:64", lambda x: _tight_diamonds(x, 64), inputs, _check_tight),
        _Probe(
            "slice_concat:32", lambda x: _slice_concat_chain(x, 32), inputs, _check_slice_concat
        ),
    ]


class _ChainStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.tensor([1.0, 0.5, -0.5, 2.0], dtype=_DTYPE))

    def forward(self, x: Tensor) -> Tensor:
        return _tight_diamonds(x * self.w, 24)


def _run_training_step() -> None:
    """The fused path and autodiff's own walk over the same shape."""
    x = lucid.ones(4, device="metal", dtype=_DTYPE)
    t = lucid.zeros(4, device="metal", dtype=_DTYPE)
    model = _ChainStep().to("metal")
    step = make_step(model, F.mse_loss)
    step(x, t).backward()
    assert not step.eager_only, "the step fell back to eager"
    compiled_grad = model.w.grad.detach().to("cpu")

    eager = _ChainStep().to("metal")
    F.mse_loss(eager(x), t).backward()
    _assert_close(compiled_grad, eager.w.grad.to("cpu"), _max_abs(eager.w.grad), "gradient")
    print("ok make_step:tight_diamonds:24", flush=True)


class _SvdLoss(nn.Module):
    """Squared entries of U and Vh, so a column's sign cannot tell two runs apart."""

    def __init__(self, a: Tensor) -> None:
        super().__init__()
        m, n = a.shape
        k = min(m, n)
        self.a = nn.Parameter(a.clone())
        self.register_buffer("weight_s", _rand(k, 1, 1).reshape(k))
        self.register_buffer("weight_u", _rand(m, k, 2))
        self.register_buffer("weight_vh", _rand(k, n, 3))

    def forward(self, x: Tensor) -> Tensor:
        u, s, vh = _thin_svd(self.a)
        total = (s * self.weight_s).sum()
        total = total + (u * u * self.weight_u).sum() + (vh * vh * self.weight_vh).sum()
        return total * x


def _svd_loss_gap(out: Tensor, target: Tensor) -> Tensor:
    return (out - target).sum()


def _run_svd_step() -> None:
    """The compiled VJP reads the emitter's U, S and Vh, so a change to U reaches it directly."""
    x = lucid.ones(1, device="metal", dtype=_DTYPE)
    t = lucid.zeros(1, device="metal", dtype=_DTYPE)
    worst = 0.0
    for m, n in ((3, 3), (5, 3), (3, 5)):
        a = _rand(m, n, 0)
        model = _SvdLoss(a).to("metal")
        step = make_step(model, _svd_loss_gap)
        for _ in range(2):
            model.a.grad = None
            step(x, t).backward()
        assert not step.eager_only, "the step fell back to eager"
        eager = _SvdLoss(a).to("metal")
        _svd_loss_gap(eager(x), t).backward()
        want = eager.a.grad.detach().to("cpu")
        worst = max(worst, _max_abs(model.a.grad.detach().to("cpu") - want) / _max_abs(want))
    assert worst <= _SVD_GRAD_LIMIT, f"gradient off by {worst:.3g}, limit {_SVD_GRAD_LIMIT:.3g}"
    print(f"ok make_step:svd, worst {worst:.3g}", flush=True)


def _poison(a: Tensor, value: float) -> Tensor:
    bad = a.clone()
    bad[..., 0, 0] = value
    return bad


def _run_poisoned_inputs() -> None:
    """A compiled call cannot raise as eager does; its answer must not look clean either.

    Traced on a finite input, since the trace runs eagerly.  The poisoned members
    of a batch must not reach the others.
    """
    with lucid.no_grad():
        eigh = lucid.compile(lambda a: tuple(_LA.eigh(a)))
        svd = lucid.compile(lambda a: tuple(_LA.svd(a)))
        for n in (1, 2, 3, 6):
            good = _symmetric(n, 0)
            eigh(good)
            svd(good)
            for value in (float("nan"), float("inf")):
                for label, tensor in zip(("w", "V"), eigh(_poison(good, value)), strict=True):
                    _assert_none_finite(tensor, f"eigh:{n} {label} on {value}")
                for label, tensor in zip(("U", "S", "Vh"), svd(_poison(good, value)), strict=True):
                    _assert_none_finite(tensor, f"svd:{n} {label} on {value}")
        print("ok poisoned matrices", flush=True)

        first, second = _symmetric(3, 1), 1e-6 * _symmetric(3, 2)
        finite = lucid.stack([first, first, second, second])
        batch = lucid.stack(
            [first, _poison(first, float("nan")), second, _poison(second, float("inf"))]
        )
        eigh(finite)
        w, v = eigh(batch)
        svd(finite)
        u, s, vh = svd(batch)
        for i in (0, 2):
            _check_eigh(batch[i], (w[i], v[i]))
            _check_svd(batch[i], (u[i], s[i], vh[i]))
        for i in (1, 3):
            for tensor in (w[i], v[i], u[i], s[i], vh[i]):
                _assert_none_finite(tensor, f"batch member {i}")
        print("ok poisoned batch members", flush=True)


def _run_batch_isolation() -> None:
    """A full-rank member's answer owes nothing to its neighbours, down to the bit."""
    with lucid.no_grad():
        kept = _rand(5, 3, 0)
        full = lucid.stack([kept, _rand(5, 3, 1), _rand(5, 3, 2)])
        mixed = lucid.stack([kept, _low_rank(5, 3, 1, 3), lucid.zeros_like(kept)])
        fn = lucid.compile(_thin_svd)
        fn(full)
        for got, want in zip(fn(mixed), fn(full), strict=True):
            assert _max_abs(got[0] - want[0]) == 0.0, "a neighbour moved member 0"
    print("ok batch isolation", flush=True)


def _run_quotient_accuracy() -> None:
    """Length drift in V reaches the eigenvalues at first order unless the quotient divides it out."""
    limit = 7 * _EPS  # Measured at 8x8: 5.3e-7 with the division, 1.3e-6 without.
    worst = 0.0
    with lucid.no_grad():
        fn = lucid.compile(lambda a: tuple(_LA.eigh(a)))
        for seed in range(10):
            family = _hard_symmetric(8, seed)
            for name in ("random", "spd", "plus_minus", "repeated", "cluster", "wide"):
                a = family[name]
                truth = _LA.eigvalsh(a.to("cpu").to(lucid.float64))
                w = fn(a)[0].to("cpu").to(lucid.float64)
                worst = max(worst, _max_abs(w - truth) / _max_abs(a))
    assert worst <= limit, f"eigenvalue error {worst:.3g} of ||A||, limit {limit:.3g}"
    print(f"ok quotient accuracy, worst {worst:.3g}", flush=True)


def _run_small_beside_large() -> None:
    """LCD-846: truth and eager give +/-1.00499e-8, and a rotation of 2.85 degrees in the block."""
    a = _tensor([[1.0, 0.0, 0.0], [0.0, 1e-8, 1e-9], [0.0, 1e-9, -1e-8]])
    with lucid.no_grad():
        fn = lucid.compile(lambda x: _LA.eigvalsh(x))
        fn(_spd(3, 0))
        w = fn(a)
    got = float(w[1].item())
    assert abs(got - 1.00499e-8) <= 1e-3 * 1.00499e-8, f"small eigenvalue {got:.6g}"
    print("ok small eigenpairs beside a large entry", flush=True)


def _rotation_chain(x: Tensor, rounds: int) -> Tensor:
    """LCD-847: plain user code, a rotation of neighbours rebuilt through slice and cat."""
    for i in range(rounds):
        p, q = i % 7, i % 7 + 1
        xp, xq = x[p : p + 1], x[q : q + 1]
        parts = [0.8 * xp - 0.6 * xq, 0.6 * xp + 0.8 * xq]
        if p > 0:
            parts = [x[:p], *parts]
        if q + 1 < 8:
            parts = [*parts, x[q + 1 :]]
        x = lucid.cat(parts, dim=0)
    return x


def _run_rotation_chain() -> None:
    x = _rand(8, 1, 0).reshape(8)
    rounds = 384
    with lucid.no_grad():
        y = lucid.compile(lambda t: _rotation_chain(t, rounds))(x)
        _assert_close(y, _rotation_chain(x, rounds), _max_abs(x), "rotation chain")
    print("ok rotation chain", flush=True)


_GROUPS: dict[str, Callable[[], list[_Probe]]] = {
    "eigen": _eigen_group,
    "triangle": _triangle_group,
    "batch": _batch_group,
    "svd": _svd_group,
    "deficient": _deficient_group,
    "scale": _scale_group,
    "chain": _chain_group,
}

_SCRIPTS: dict[str, Callable[[], None]] = {
    "step": _run_training_step,
    "svd_step": _run_svd_step,
    "poison": _run_poisoned_inputs,
    "batch_isolation": _run_batch_isolation,
    "quotient": _run_quotient_accuracy,
    "small_beside_large": _run_small_beside_large,
    "rotation_chain": _run_rotation_chain,
}

GROUP_NAMES = (*_GROUPS, "step", "svd_step", "poison", "batch_isolation", "quotient")


def main(group: str) -> None:
    warnings.simplefilter("error", CompileFallbackWarning)
    try:
        if group in _SCRIPTS:
            _SCRIPTS[group]()
            return
        with lucid.no_grad():
            for probe in _GROUPS[group]():
                _run(probe)
    except AssertionError as failure:
        print(failure, file=sys.stderr, flush=True)
        sys.exit(WRONG_ANSWER)


if __name__ == "__main__":
    main(sys.argv[1])
