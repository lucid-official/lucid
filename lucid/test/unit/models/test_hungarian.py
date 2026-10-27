"""Unit tests for the custom Kuhn-Munkres / Hungarian algorithm used by
DETR / MaskFormer / Mask2Former matchers.

Validates ``solve_assignment`` against ``scipy.optimize.linear_sum_assignment``
on a range of randomised rectangular cost matrices.  ``scipy`` is only used inside
this test module — production code never imports it.
"""

import random
import unittest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models.vision.detr._model import _hungarian_match

import pytest

# Skip the whole module if scipy isn't available (it's optional [test] extra)
scipy_opt = pytest.importorskip("scipy.optimize")
from lucid.models._utils._detection import solve_assignment


def _ref_assignment(cost: list[list[float]]) -> dict[int, int]:
    """Reference: scipy returns optimal row->col assignment for rectangular matrices."""
    row_ind, col_ind = scipy_opt.linear_sum_assignment(cost)
    return dict(zip(row_ind.tolist(), col_ind.tolist()))


def _our_assignment(cost: list[list[float]]) -> dict[int, int]:
    rows, cols = solve_assignment(cost)
    return dict(zip(rows, cols))


def _total_cost(cost: list[list[float]], assignment: dict[int, int]) -> float:
    return sum(cost[r][c] for r, c in assignment.items())


class TestHungarianCorrectness(unittest.TestCase):
    """Compare custom Hungarian against scipy on random + handcrafted cases."""

    def test_trivial_3x3(self) -> None:
        cost = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
        ]
        ours = _our_assignment(cost)
        # Optimal: rows 0/1/2 → cols 0/1/2, total = 3.0
        self.assertAlmostEqual(_total_cost(cost, ours), 3.0, places=5)

    def test_rectangular_3x5(self) -> None:
        """3 rows × 5 cols — assign each row to a distinct col."""
        cost = [
            [10.0, 1.0, 9.0, 8.0, 7.0],
            [5.0, 6.0, 2.0, 8.0, 9.0],
            [4.0, 5.0, 6.0, 7.0, 1.0],
        ]
        ours = _our_assignment(cost)
        ref = _ref_assignment(cost)
        self.assertAlmostEqual(
            _total_cost(cost, ours),
            _total_cost(cost, ref),
            places=5,
        )

    def test_obvious_match_5x3(self) -> None:
        """5 queries × 3 GTs — rows 0/1/2 should match cols 0/1/2 (diagonal cheap)."""
        # Note: API is (n_rows ≤ n_cols), so caller should construct cost as (M, N).
        cost = [
            [0.0, 10.0, 10.0, 100.0, 100.0],
            [10.0, 0.0, 10.0, 100.0, 100.0],
            [10.0, 10.0, 0.0, 100.0, 100.0],
        ]
        rows, cols = solve_assignment(cost)
        self.assertEqual(rows, [0, 1, 2])
        self.assertEqual(cols, [0, 1, 2])

    def test_negative_costs(self) -> None:
        """DETR uses negative costs (since cost = -log_prob - GIoU)."""
        cost = [
            [-5.0, -1.0, -3.0],
            [-2.0, -8.0, -4.0],
            [-3.0, -2.0, -7.0],
        ]
        ours = _our_assignment(cost)
        ref = _ref_assignment(cost)
        self.assertAlmostEqual(
            _total_cost(cost, ours),
            _total_cost(cost, ref),
            places=5,
        )

    def test_random_rectangular(self) -> None:
        """100 randomised M×N matrices (M ≤ N) vs scipy."""
        rng = random.Random(42)
        for trial in range(100):
            M = rng.randint(1, 8)
            N = rng.randint(M, 12)
            cost = [[rng.uniform(-5.0, 5.0) for _ in range(N)] for _ in range(M)]
            ours = _our_assignment(cost)
            ref = _ref_assignment(cost)
            ours_cost = _total_cost(cost, ours)
            ref_cost = _total_cost(cost, ref)
            self.assertAlmostEqual(
                ours_cost,
                ref_cost,
                places=4,
                msg=f"trial {trial} M={M} N={N}: ours={ours_cost} ref={ref_cost}",
            )

    def test_detr_scale(self) -> None:
        """DETR-scale random matrix: 5 GTs × 100 queries."""
        rng = random.Random(0)
        M, N = 5, 100
        cost = [[rng.uniform(-2.0, 2.0) for _ in range(N)] for _ in range(M)]
        ours = _our_assignment(cost)
        ref = _ref_assignment(cost)
        self.assertAlmostEqual(
            _total_cost(cost, ours),
            _total_cost(cost, ref),
            places=4,
        )


class TestDETRMatcher(unittest.TestCase):
    """Pin DETR's matcher itself, not just the LAP solver underneath it.

    ``solve_assignment`` is exercised above; what is DETR-specific — the
    1/5/2 cost weighting, the (pred, gt) return order, and which query a
    given ground-truth object claims — lives in ``_hungarian_match``.
    """

    @staticmethod
    def _case() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Two GT objects, three queries, with the answer built in.

        Query 0 is placed exactly on GT 1 and confidently predicts its
        class; query 2 is placed exactly on GT 0.  Query 1 is a decoy far
        from both.  The only sane assignment is 0->1 and 2->0.
        """
        # 3 queries, 2 foreground classes + no-object
        pred_logits = lucid.tensor(
            [
                [0.0, 8.0, 0.0],  # query 0 — confident class 1
                [0.0, 0.0, 8.0],  # query 1 — confident no-object
                [8.0, 0.0, 0.0],  # query 2 — confident class 0
            ]
        )
        pred_boxes = lucid.tensor(
            [
                [0.70, 0.70, 0.20, 0.20],  # on GT 1
                [0.10, 0.90, 0.05, 0.05],  # decoy
                [0.25, 0.25, 0.30, 0.30],  # on GT 0
            ]
        )
        gt_labels = lucid.tensor([0, 1]).long()
        gt_boxes = lucid.tensor([[0.25, 0.25, 0.30, 0.30], [0.70, 0.70, 0.20, 0.20]])
        return pred_logits, pred_boxes, gt_labels, gt_boxes

    def test_returns_pred_then_gt_and_pairs_correctly(self) -> None:
        pred_logits, pred_boxes, gt_labels, gt_boxes = self._case()
        pred_idx, gt_idx = _hungarian_match(pred_logits, pred_boxes, gt_labels, gt_boxes)
        # Order of the returned tuple is (pred, gt) — swapping them would
        # still typecheck and still have length 2, so pin the pairing, which
        # is the only thing that distinguishes them.  Query 2 sits on GT 0
        # and query 0 sits on GT 1.
        self.assertEqual(list(zip(pred_idx, gt_idx)), [(2, 0), (0, 1)])
        # Rows of the cost matrix are ground truths, so pairs arrive in
        # ascending GT order — not ascending query order.
        self.assertEqual(gt_idx, sorted(gt_idx))

    def test_no_ground_truth_matches_nothing(self) -> None:
        pred_logits, pred_boxes, _, _ = self._case()
        pred_idx, gt_idx = _hungarian_match(
            pred_logits,
            pred_boxes,
            lucid.tensor([]).long(),
            lucid.zeros(0, 4),
        )
        self.assertEqual((pred_idx, gt_idx), ([], []))

    def test_cost_weights_are_one_five_two(self) -> None:
        """The box terms outweigh the class term at DETR's default weights.

        Query 1 is given the winning class logits for GT 0 but a box far
        from it; query 2 keeps the right box and the wrong class.  With
        cost_l1=5 and cost_giou=2 against cost_cls=1, geometry decides and
        GT 0 goes to query 2.  Re-weighting so classification dominates
        flips it — which is what makes this a test of the weights and not
        just of the solver.
        """
        pred_logits = lucid.tensor([[0.0, 0.0, 8.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
        pred_boxes = lucid.tensor(
            [
                [0.90, 0.90, 0.05, 0.05],
                [0.10, 0.90, 0.05, 0.05],  # right class, wrong place
                [0.25, 0.25, 0.30, 0.30],  # wrong class, right place
            ]
        )
        gt_labels = lucid.tensor([0]).long()
        gt_boxes = lucid.tensor([[0.25, 0.25, 0.30, 0.30]])

        pred_idx, gt_idx = _hungarian_match(pred_logits, pred_boxes, gt_labels, gt_boxes)
        self.assertEqual((pred_idx, gt_idx), ([2], [0]))

        flipped_pred, flipped_gt = _hungarian_match(
            pred_logits,
            pred_boxes,
            gt_labels,
            gt_boxes,
            cost_cls=100.0,
            cost_l1=1.0,
            cost_giou=0.0,
        )
        self.assertEqual((flipped_pred, flipped_gt), ([1], [0]))


if __name__ == "__main__":
    unittest.main()
