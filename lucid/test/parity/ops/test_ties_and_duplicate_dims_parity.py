"""Equal values in a selection, and an axis named twice, on both devices.

Metal's ``topk`` read the stable ascending order backwards, so equal
values came out highest index first: ``topk([0, 5, 4, 4], 2)`` named 1
and 3 on Metal, 1 and 2 on the CPU and in the reference.  The CPU's
partial sort was not stable either.  Both now give equal values lowest
index first (LCD-407).  The reference promises no order among ties in a
long top-k, so there the indices are held to that rule and to the
reference's values; on the short inputs below its order is the same.

``flip`` with one axis named twice flipped it twice, which is no flip at
all; the reference refuses it, and so do ``permute`` and ``movedim``.
``roll`` with an axis named twice rolls it twice, in both.
"""

import math
import random
from typing import Any

import numpy as np
import pytest

import lucid

_SHORT = [
    [[0.0, 5.0, 4.0, 4.0]],
    [[1.0, 1.0, 1.0, 1.0, 1.0]],
    [[2.0, 7.0, 7.0, 1.0, 7.0, 2.0]],
]


def _long() -> list[list[list[float]]]:
    rng = random.Random(407)
    rows = [[float(rng.randint(0, 3)) for _ in range(n)] for n in (64, 1000, 5000)]
    rows.append([math.nan if i % 7 == 3 else float(i % 3) for i in range(50)])
    return [[r] for r in rows]


def _stable_largest(row: list[float], k: int) -> list[int]:
    """The ``k`` largest, largest first, ties lowest index first, NaN first."""
    key = [(-math.inf if math.isnan(v) else -v, i) for i, v in enumerate(row)]
    return [i for _, i in sorted(key)[:k]]


@pytest.mark.parity
class TestTopkTies:
    @pytest.mark.parametrize("rows", _SHORT)
    def test_short_rows_match_the_reference(
        self, ref: Any, device: str, rows: list[list[float]]
    ) -> None:
        x = lucid.tensor(rows, device=device)
        for k in range(len(rows[0]) + 1):
            values, indices = lucid.topk(x, k)
            r_values, r_indices = ref.topk(ref.tensor(rows), k)
            assert indices.to("cpu").tolist() == r_indices.tolist(), k
            assert values.to("cpu").tolist() == r_values.tolist(), k

    @pytest.mark.parametrize("rows", _long(), ids=["64", "1000", "5000", "nan"])
    def test_ties_come_lowest_index_first(
        self, ref: Any, device: str, rows: list[list[float]]
    ) -> None:
        x = lucid.tensor(rows, device=device)
        for k in sorted({1, 2, 10, len(rows[0])}):
            values, indices = lucid.topk(x, k)
            assert indices.to("cpu").tolist()[0] == _stable_largest(rows[0], k)
            r_values, _ = ref.topk(ref.tensor(rows), k)
            np.testing.assert_array_equal(np.asarray(values.to("cpu").numpy()), r_values.numpy())


@pytest.mark.parity
class TestOrderedSelections:
    @pytest.mark.parametrize("rows", _SHORT + _long())
    def test_stable_argsort_both_directions(
        self, ref: Any, device: str, rows: list[list[float]]
    ) -> None:
        x = lucid.tensor(rows, device=device)
        for descending in (False, True):
            got = lucid.argsort(x, dim=-1, descending=descending).to("cpu").tolist()
            want = ref.argsort(
                ref.tensor(rows), dim=-1, descending=descending, stable=True
            ).tolist()
            assert got == want, descending

    @pytest.mark.parametrize("rows", _SHORT + _long())
    def test_kthvalue_of_tied_rows(self, ref: Any, device: str, rows: list[list[float]]) -> None:
        x = lucid.tensor(rows, device=device)
        n = len(rows[0])
        for k in sorted({1, n // 2 + 1, n}):
            got = lucid.kthvalue(x, k).values.to("cpu").tolist()
            want = ref.kthvalue(ref.tensor(rows), k).values.tolist()
            assert np.array_equal(np.asarray(got), np.asarray(want), equal_nan=True)


@pytest.mark.parity
class TestDuplicateDims:
    @pytest.mark.parametrize(
        "call",
        [
            lambda m, t: m.flip(t, [0, 0]),
            lambda m, t: m.flip(t, [2, -1]),
            lambda m, t: t.permute(0, 0, 1),
            lambda m, t: t.permute(0, 2, -1),
            lambda m, t: m.movedim(t, (0, 0), (1, 2)),
            lambda m, t: m.movedim(t, (0, 1), (2, -1)),
        ],
        ids=[
            "flip",
            "flip-negative",
            "permute",
            "permute-negative",
            "movedim-source",
            "movedim-destination",
        ],
    )
    def test_an_axis_named_twice_is_refused(self, ref: Any, device: str, call: Any) -> None:
        with pytest.raises(RuntimeError):
            call(ref, ref.zeros(2, 3, 4))
        with pytest.raises(RuntimeError):
            call(lucid, lucid.zeros(2, 3, 4, device=device))

    def test_flip_of_a_scalar_names_its_axis_once(self, ref: Any, device: str) -> None:
        with pytest.raises(RuntimeError):
            ref.flip(ref.tensor(1.0), [0, 0])
        with pytest.raises(RuntimeError):
            lucid.flip(lucid.tensor(1.0, device=device), [0, 0])

    def test_roll_rolls_an_axis_named_twice_twice(self, ref: Any, device: str) -> None:
        data = np.arange(24.0, dtype=np.float32).reshape(2, 3, 4)
        got = lucid.roll(lucid.tensor(data, device=device), (1, 2), (2, -1))
        want = ref.roll(ref.tensor(data), (1, 2), (2, -1))
        assert got.to("cpu").tolist() == want.tolist()
