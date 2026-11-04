"""Error classes, op naming, and failure explanations (LCD-673, LCD-675)."""

import pytest

import lucid
import lucid.linalg as LA
import lucid.nn.functional as F


def test_public_op_names_replace_internal_prefixes_and_omit_trace_by_default() -> None:
    x = lucid.zeros(2, 3)
    with pytest.raises(ValueError, match=r"^unfold:") as exc_info:
        x.unfold(0, 5, 1)
    assert "[trace=" not in str(exc_info.value)

    with pytest.raises(IndexError, match=r"^transpose:") as exc_info:
        x.transpose(0, 5)
    assert "[trace=" not in str(exc_info.value)

    with pytest.raises(IndexError, match=r"^take:") as exc_info:
        lucid.take(x, lucid.tensor([10]))
    assert "[trace=" not in str(exc_info.value)

    with pytest.raises(IndexError, match=r"^index_select:") as exc_info:
        lucid.index_select(x, 0, lucid.tensor([10]))
    assert "[trace=" not in str(exc_info.value)

    with pytest.raises(ValueError, match=r"^clamp:") as exc_info:
        lucid.clamp(x)
    assert "[trace=" not in str(exc_info.value)

    with pytest.raises((ValueError, IndexError), match=r"^fft") as exc_info:
        lucid.fft.fft(lucid.zeros(0))
    assert "[trace=" not in str(exc_info.value)


def test_index_out_of_bounds_includes_index_dimension_and_extent() -> None:
    x = lucid.zeros(2, 3)
    with pytest.raises(
        IndexError, match=r"^gather:.*index 5 is out of bounds for dimension 1 with size 3"
    ):
        lucid.gather(x, 1, lucid.tensor([[5, 0], [0, 0]]))

    with pytest.raises(IndexError, match=r"index 4 is out of bounds for dimension 0 with size 2"):
        _ = x[lucid.tensor([4]), :]


def test_lapack_failure_reasons_are_human_readable() -> None:
    singular = lucid.zeros(3, 3)
    with pytest.raises(LA.LinAlgError, match=r"D\[1,1\] is zero"):
        LA.ldl_factor(singular)
    bad = lucid.eye(3)
    bad[0, 0] = float("nan")
    with pytest.raises(LA.LinAlgError, match=r"failed to converge"):
        LA.eigh(bad)


def test_value_error_raised_for_invalid_shape_and_value_arguments() -> None:
    x = lucid.zeros(2, 3)
    with pytest.raises(ValueError, match=r"reshape"):
        x.reshape(5)
    with pytest.raises(ValueError, match=r"cat|concatenate"):
        lucid.cat([])
    with pytest.raises(ValueError, match=r"diag"):
        lucid.diag(lucid.zeros(2, 2, 2))
    with pytest.raises(ValueError, match=r"trace"):
        lucid.trace(lucid.zeros(2))
    with pytest.raises(ValueError, match=r"pad"):
        F.pad(x, (1,))
