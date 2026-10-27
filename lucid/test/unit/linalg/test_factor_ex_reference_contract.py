"""``ldl_factor``'s ``hermitian`` and ``cholesky_ex``'s partial factor (LCD-395).

``ldl_factor(A, hermitian=True)`` took ``hermitian`` by position and
refused ``False``; the reference's is ``(A, *, hermitian=False)``, and for
a real ``A`` the flag changes nothing.  ``cholesky_ex`` zero-filled a
matrix that failed, where LAPACK defines its leading factor: the factor of
the ``(k-1)``-square minor that succeeded.  Only that block is held to the
reference — what ``potrf`` leaves outside it depends on its blocking.
"""

import inspect
from collections.abc import Sequence
from types import ModuleType

import pytest

import lucid
import lucid.linalg as LA

_FAILING = [
    [[1.0, 2.0], [2.0, 1.0]],
    [[-1.0, 0.0], [0.0, 1.0]],
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[float("nan"), 0.0], [0.0, 4.0]],
    [[4.0, 2.0, 1.0, 1.0], [2.0, 5.0, 3.0, 1.0], [1.0, 3.0, 1.0, 2.0], [1.0, 1.0, 2.0, 6.0]],
    [[1.0, 2.0, 1.0, 1.0], [2.0, 1.0, 3.0, 1.0], [1.0, 3.0, 5.0, 2.0], [1.0, 1.0, 2.0, 6.0]],
]


def _broken(n: int, order: int) -> list[list[float]]:
    """An SPD ``n``-square matrix whose leading minor ``order`` fails."""
    a = [[float(n if i == j else 1.0 / (1 + abs(i - j))) for j in range(n)] for i in range(n)]
    a[order - 1][order - 1] = -5.0
    return a


_FAILING += [_broken(n, k) for n in (5, 6, 8) for k in (2, 3, 4)]


def _assert_leading_factor(
    ref: ModuleType, got: Sequence[object], want: Sequence[object], order: int
) -> None:
    """The ``(order - 1)``-square block LAPACK defines, and zeros elsewhere."""
    done = order - 1
    got_t, want_t = ref.tensor(got), ref.tensor(want)
    assert ref.allclose(got_t[:done, :done], want_t[:done, :done], atol=1e-5)
    got_t[:done, :done] = 0
    assert (got_t == 0).all()


@pytest.mark.parametrize("upper", [False, True], ids=["lower", "upper"])
@pytest.mark.parametrize("index", range(len(_FAILING)))
def test_cholesky_ex_returns_the_leading_factor(
    ref: ModuleType, device: str, upper: bool, index: int
) -> None:
    a = _FAILING[index]
    got, info = LA.cholesky_ex(lucid.tensor(a, device=device), upper=upper)
    want, rinfo = ref.linalg.cholesky_ex(ref.tensor(a), upper=upper)
    assert int(info.item()) == int(rinfo.item()) != 0
    _assert_leading_factor(ref, got.tolist(), want.tolist(), int(rinfo.item()))


@pytest.mark.parametrize("upper", [False, True], ids=["lower", "upper"])
def test_a_batch_mixes_full_and_partial_factors(ref: ModuleType, device: str, upper: bool) -> None:
    ok = [[4.0, 2.0], [2.0, 3.0]]
    a = [ok, _FAILING[0], [[9.0, 3.0], [3.0, 5.0]], _FAILING[1]]
    got, info = LA.cholesky_ex(lucid.tensor(a, device=device), upper=upper)
    want, rinfo = ref.linalg.cholesky_ex(ref.tensor(a), upper=upper)
    assert info.tolist() == rinfo.tolist() == [0, 2, 0, 1]
    for b, code in enumerate(rinfo.tolist()):
        if code:
            _assert_leading_factor(ref, got[b].tolist(), want[b].tolist(), code)
        else:
            assert ref.allclose(ref.tensor(got[b].tolist()), want[b], atol=1e-5)


def test_ldl_hermitian_is_keyword_only_and_off_by_default(ref: ModuleType) -> None:
    for name in ("ldl_factor", "ldl_factor_ex"):
        param = inspect.signature(getattr(LA, name)).parameters["hermitian"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is False
    with pytest.raises(TypeError):
        ref.linalg.ldl_factor(ref.eye(2), True)


@pytest.mark.parametrize("hermitian", [False, True])
def test_ldl_factors_a_real_matrix_either_way(
    ref: ModuleType, device: str, hermitian: bool
) -> None:
    a = [[4.0, 1.0, -2.0], [1.0, -3.0, 0.5], [-2.0, 0.5, 1.0]]
    want_ld, want_piv, want_info = ref.linalg.ldl_factor_ex(ref.tensor(a), hermitian=hermitian)
    x = lucid.tensor(a, device=device)
    for ld, piv in (
        LA.ldl_factor(x, hermitian=hermitian),
        LA.ldl_factor_ex(x, hermitian=hermitian)[:2],
    ):
        assert ref.allclose(ref.tensor(ld.tolist()), want_ld, atol=1e-5)
        assert piv.tolist() == want_piv.tolist()
    assert int(LA.ldl_factor_ex(x, hermitian=hermitian)[2].item()) == int(want_info)


def test_hermitian_by_position_is_refused() -> None:
    # Positional ``hermitian`` answered with a warning until LCD-410; the
    # reference takes it by keyword only, and so does Lucid now.
    a = lucid.tensor([[4.0, 1.0], [1.0, 3.0]])
    for call in (lambda: LA.ldl_factor(a, True), lambda: LA.ldl_factor_ex(a, True)):
        with pytest.raises(TypeError, match="positional"):
            call()
