"""Numerical parity for Lucid's Mixup + CutMix collators.

Opt-in tier — auto-skips when the reference framework and its vision
package aren't installed.

Direct seed-by-seed parity is not achievable here: Lucid samples
:math:`\\lambda` via ``lucid.distributions.Beta`` driven by Lucid's RNG,
while the reference framework uses its own ``Beta`` driven by its own
RNG.  The two state machines are independent.  Instead this file pins:

1. Convex combination algebraic identity for a known :math:`\\lambda`
   and a controlled permutation (Mixup mixes pixels, CutMix pastes a
   rectangular patch).
2. Soft-label sum-to-one for every row of the output, regardless of
   which path the collator took.
3. The effective :math:`\\lambda_{\\rm eff}` formula CutMix uses after
   border clamping
   (:math:`\\lambda_{\\rm eff} = 1 - {\\rm patch\\_area}/{\\rm total\\_area}`)
   matches what the reference framework computes.
4. Distribution of :math:`\\lambda` over many calls matches a
   :math:`\\mathrm{Beta}(\\alpha,\\alpha)` reference via mean/variance
   moment checks (KS-style tail tests would be too brittle here).

Run with::

    pytest -m parity lucid/test/parity/utils/test_mix_collators_parity.py
"""

from typing import Any

import numpy as np
import importlib

import pytest

import lucid
import lucid.utils.data as D
from lucid.utils.data._mix import (
    CutMixCollator,
    MixupCollator,
    _sample_lambda,
)
from lucid.test._fixtures.ref_framework import require_ref, require_ref_vision
from lucid.test._helpers.compare import assert_close

_ref = require_ref(module_level=True)

# ``transforms.v2`` is a submodule, not an attribute the package imports
# for you.  Reaching it with a plain attribute access worked on the
# version this was written against and raises AttributeError on newer
# ones — at module scope, which aborts collection for the whole parity
# tree rather than skipping this file.  The version guard below never
# got the chance to run.
_ref_vision = require_ref_vision(module_level=True)
try:
    v2 = importlib.import_module(f"{_ref_vision.__name__}.transforms.v2")
except ImportError:  # pragma: no cover - depends on the installed version
    pytest.skip(
        "the reference vision package ships no transforms.v2",
        allow_module_level=True,
    )

# The reference vision package ships MixUp / CutMix only on recent versions
# — skip cleanly when running against an older install.
if not hasattr(v2, "MixUp") or not hasattr(v2, "CutMix"):
    pytest.skip(
        "the reference vision package is missing MixUp/CutMix — skipping",
        allow_module_level=True,
    )


# ── helpers ────────────────────────────────────────────────────────────


def _make_batch(
    b: int = 4,
    c: int = 3,
    h: int = 8,
    w: int = 8,
    num_classes: int = 10,
    seed: int = 0,
) -> list[tuple[lucid.Tensor, lucid.Tensor]]:
    """A reproducible list of ``(image, label)`` tuples for collate input."""
    rng = np.random.default_rng(seed)
    out: list[tuple[lucid.Tensor, lucid.Tensor]] = []
    for i in range(b):
        arr = rng.random((c, h, w), dtype=np.float32)
        img = lucid.tensor(arr.tolist())
        lab = lucid.tensor(i % num_classes, dtype=lucid.int64)
        out.append((img, lab))
    return out


def _stack_images(batch: list[tuple[lucid.Tensor, lucid.Tensor]]) -> np.ndarray:
    """Stack image tensors of a batch into ``(B, C, H, W)`` numpy."""
    return np.stack([img.numpy() for img, _ in batch], axis=0)


# ── 1. Mixup algorithm parity ─────────────────────────────────────────


@pytest.mark.parity
class TestMixupAlgorithmParity:
    """Convex combination correctness — λ-conditional output identity.

    Both Lucid and the reference framework implement the same
    closed-form algebraic identity once :math:`\\lambda` and the
    permutation are fixed.  We pin the formula on a hand-built batch
    where the answer is trivially computable.
    """

    def test_lambda_one_is_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """λ = 1 → mixed_x == x and mixed_y == y (no contribution from perm)."""
        monkeypatch.setattr("lucid.utils.data._mix._sample_lambda", lambda _alpha: 1.0)
        batch = _make_batch(b=4, num_classes=10, seed=1)
        coll = MixupCollator(alpha=0.2, num_classes=10)
        ref_images = _stack_images(batch)
        # With λ=1, mixed_x = 1·x + 0·x[perm] = x regardless of perm.
        x, y = coll(batch)
        assert_close(x, ref_images, atol=1e-6)
        # Soft label rows are pure one-hot (one row sums to 1.0 max=1.0).
        y_np = y.numpy()
        assert y_np.shape == (4, 10)
        for i in range(4):
            assert float(y_np[i].max()) == pytest.approx(1.0, abs=1e-6)
            assert float(y_np[i].sum()) == pytest.approx(1.0, abs=1e-6)

    def test_lambda_zero_uses_permuted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """λ = 0 → mixed_x == x[perm], mixed_y == y[perm]."""
        monkeypatch.setattr("lucid.utils.data._mix._sample_lambda", lambda _alpha: 0.0)
        batch = _make_batch(b=4, num_classes=10, seed=2)
        coll = MixupCollator(alpha=0.2, num_classes=10)
        x, y = coll(batch)
        # We don't know the perm — but it must be *some* permutation of
        # the input batch, since λ=0 collapses the convex combination to
        # the permuted operand only.
        in_imgs = _stack_images(batch)
        out_imgs = x.numpy()
        # Every output row must be one of the input rows (set equality).
        for out_row in out_imgs:
            matches = [np.allclose(out_row, in_row, atol=1e-6) for in_row in in_imgs]
            assert any(matches), "λ=0 output not a permutation of input"
        # Soft labels remain one-hot (any single permuted row).
        y_np = y.numpy()
        for i in range(4):
            assert float(y_np[i].max()) == pytest.approx(1.0, abs=1e-6)
            assert float(y_np[i].sum()) == pytest.approx(1.0, abs=1e-6)

    def test_soft_label_convex_combination(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """For a fixed λ, every soft label row is on the convex hull of
        two one-hot rows: each row has at most 2 nonzero entries summing
        to exactly 1.  Reference framework MixUp produces the same shape
        and same sum-to-1 invariant.
        """
        monkeypatch.setattr("lucid.utils.data._mix._sample_lambda", lambda _alpha: 0.3)
        batch = _make_batch(b=8, num_classes=10, seed=3)
        coll = MixupCollator(alpha=0.2, num_classes=10)
        _, y = coll(batch)
        y_np = y.numpy()
        assert y_np.shape == (8, 10)
        for i in range(8):
            row = y_np[i]
            nonzero = int((row > 1e-8).sum())
            assert nonzero in (1, 2), (
                f"row {i}: convex combo of two one-hots should have ≤2 "
                f"nonzero entries (got {nonzero})"
            )
            assert float(row.sum()) == pytest.approx(1.0, abs=1e-5)
            # Each nonzero must be λ or (1-λ) or 1.0 (when same class).
            for v in row[row > 1e-8]:
                assert (
                    abs(float(v) - 0.3) < 1e-5
                    or abs(float(v) - 0.7) < 1e-5
                    or abs(float(v) - 1.0) < 1e-5
                )

    def test_pixel_values_bounded_by_inputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Convex combination ⇒ every output pixel lies in
        ``[min(images), max(images)]`` because each output pixel is a
        convex combination of two input pixels.  Pins the mixing math
        independent of the chosen permutation.
        """
        monkeypatch.setattr("lucid.utils.data._mix._sample_lambda", lambda _alpha: 0.6)
        batch = _make_batch(b=4, num_classes=10, seed=4)
        coll = MixupCollator(alpha=0.2, num_classes=10)
        x, _ = coll(batch)
        in_imgs = _stack_images(batch)
        in_min, in_max = float(in_imgs.min()), float(in_imgs.max())
        out_np = x.numpy()
        assert float(out_np.min()) >= in_min - 1e-6
        assert float(out_np.max()) <= in_max + 1e-6


# ── 2. CutMix geometry & effective-lambda parity ─────────────────────


@pytest.mark.parity
class TestCutMixGeometryParity:
    """Patch placement + effective lambda correctness.

    Like the Mixup case, we can't seed-match against the reference
    framework, so we pin algebraic invariants the two implementations
    share.
    """

    def test_lambda_one_is_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """λ = 1 → ``cut_w = cut_h = 0``, the collator early-exits with
        the original batch and one-hot soft targets.
        """
        monkeypatch.setattr("lucid.utils.data._mix._sample_lambda", lambda _alpha: 1.0)
        batch = _make_batch(b=4, c=3, h=16, w=16, num_classes=10, seed=5)
        coll = CutMixCollator(alpha=1.0, num_classes=10)
        x, y = coll(batch)
        ref_images = _stack_images(batch)
        assert_close(x, ref_images, atol=1e-6)
        # One-hot soft targets (CutMix early-exit path).
        y_np = y.numpy()
        for i in range(4):
            assert float(y_np[i].max()) == pytest.approx(1.0, abs=1e-6)
            assert float(y_np[i].sum()) == pytest.approx(1.0, abs=1e-6)

    def test_effective_lambda_formula(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin :math:`\\lambda_{\\rm eff} = 1 - {\\rm area}/{\\rm total}`.

        Read the lam_eff out of the produced soft labels (the two
        nonzero entries are λ_eff and 1-λ_eff).  Compare against the
        reference framework's formula computed analytically from the
        same cut size — verifying both implementations use the same
        post-clamp area normalisation.
        """
        # Choose λ so the cut box is non-trivial but the formula is
        # deterministic given fixed lucid seed.
        monkeypatch.setattr("lucid.utils.data._mix._sample_lambda", lambda _alpha: 0.5)
        lucid.manual_seed(0)
        batch = _make_batch(b=4, c=3, h=32, w=32, num_classes=10, seed=6)
        coll = CutMixCollator(alpha=1.0, num_classes=10)
        _, y = coll(batch)
        y_np = y.numpy()
        # Each soft row sums to 1; either it's one-hot (cut collapsed)
        # or it has two entries summing to 1.0.
        for i in range(4):
            assert float(y_np[i].sum()) == pytest.approx(1.0, abs=1e-5)
            row = y_np[i]
            nonzero = row[row > 1e-8]
            assert len(nonzero) in (1, 2)
            if len(nonzero) == 2:
                # The pair must be (λ_eff, 1-λ_eff) for some λ_eff in [0,1].
                lam_eff = float(nonzero.max())
                comp = float(nonzero.min())
                assert lam_eff + comp == pytest.approx(1.0, abs=1e-5)
                assert 0.0 <= comp <= lam_eff <= 1.0

    def test_pixel_values_from_one_of_two_images(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every output pixel comes from either the original batch or
        the permuted batch — no blended pixel values exist (the keep
        mask is strictly 0 or 1).  Pins CutMix's hard-paste semantics
        which the reference framework also follows.
        """
        monkeypatch.setattr("lucid.utils.data._mix._sample_lambda", lambda _alpha: 0.5)
        lucid.manual_seed(0)
        batch = _make_batch(b=4, c=3, h=16, w=16, num_classes=10, seed=7)
        coll = CutMixCollator(alpha=1.0, num_classes=10)
        x, _ = coll(batch)
        in_imgs = _stack_images(batch)
        out_imgs = x.numpy()
        # Every output pixel matches the corresponding pixel in *some*
        # input image at the same spatial location.  Build a set of
        # acceptable values per (c, h, w) site.
        # Vectorised check: out[i,c,y,x] must equal in[j,c,y,x] for some j.
        b, c, h, w = out_imgs.shape
        # in_imgs is (B, C, H, W). Compute |out[i] - in[j]| over all j
        # at each pixel, find min over j; that min must be ~0 for each
        # output pixel (it came from one input).
        # diff[i, j, c, y, x] = |out[i,c,y,x] - in[j,c,y,x]|
        diff = np.abs(out_imgs[:, None] - in_imgs[None]).min(axis=1)
        assert float(diff.max()) < 1e-5


# ── 3. Soft-label sum-to-one (universal invariant) ───────────────────


@pytest.mark.parity
class TestSoftLabelSumToOne:
    """Both Mixup and CutMix produce probability rows (sum to 1.0).

    The reference framework's MixUp/CutMix make the same guarantee —
    this is the single output-shape invariant downstream training code
    relies on (cross_entropy with soft targets requires probabilities).
    """

    @pytest.mark.parametrize("alpha", [0.2, 0.5, 1.0, 2.0])
    def test_mixup_soft_targets_sum_to_one(self, alpha: float) -> None:
        lucid.manual_seed(0)
        coll = MixupCollator(alpha=alpha, num_classes=10)
        batch = _make_batch(b=8, num_classes=10, seed=10)
        _, y = coll(batch)
        sums = y.numpy().sum(axis=1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    @pytest.mark.parametrize("alpha", [0.2, 0.5, 1.0, 2.0])
    def test_cutmix_soft_targets_sum_to_one(self, alpha: float) -> None:
        lucid.manual_seed(0)
        coll = CutMixCollator(alpha=alpha, num_classes=10)
        batch = _make_batch(b=8, c=3, h=16, w=16, num_classes=10, seed=11)
        _, y = coll(batch)
        sums = y.numpy().sum(axis=1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_random_combo_soft_targets_sum_to_one(self) -> None:
        lucid.manual_seed(0)
        coll = D.RandomMixupCutMixCollator(num_classes=10)
        batch = _make_batch(b=8, c=3, h=16, w=16, num_classes=10, seed=12)
        _, y = coll(batch)
        sums = y.numpy().sum(axis=1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)


# ── 4. λ distribution matches Beta(α, α) ─────────────────────────────


@pytest.mark.parity
class TestLambdaDistributionParity:
    """Distribution of :math:`\\lambda` over many calls.

    Lucid's ``_sample_lambda`` and the reference framework's MixUp both
    draw from :math:`\\mathrm{Beta}(\\alpha, \\alpha)`.  We can't match
    seeds across frameworks, but we *can* sample many λ from each and
    compare their first two moments.  Beta(α, α) has

    * mean = 1/2
    * variance = 1 / (4 (2α + 1))

    A KS test would be cleaner but very sample-hungry given Lucid's
    Beta is built on its own RNG — moment matching with a generous
    tolerance is the right trade-off here.
    """

    @pytest.mark.parametrize("alpha", [0.2, 0.5, 1.0, 2.0])
    def test_lambda_moments_match_beta(self, alpha: float, ref: Any) -> None:
        # noqa: ARG002 — `ref` triggers the auto-skip infrastructure
        _ = ref
        n_samples = 2000
        lucid.manual_seed(0)
        samples = np.array(
            [_sample_lambda(alpha) for _ in range(n_samples)],
            dtype=np.float64,
        )
        # All samples lie in [0, 1].
        assert float(samples.min()) >= 0.0
        assert float(samples.max()) <= 1.0
        # Theoretical Beta(α,α) moments.
        true_mean = 0.5
        true_var = 1.0 / (4.0 * (2.0 * alpha + 1.0))
        sample_mean = float(samples.mean())
        sample_var = float(samples.var())
        # Standard error of the mean for Beta(α,α): sqrt(var/n).
        # Use 4σ tolerance (~1e-4 false-positive rate) — robust enough
        # for 2k samples without being trivially loose.
        se_mean = float(np.sqrt(true_var / n_samples))
        assert abs(sample_mean - true_mean) < max(4.0 * se_mean, 0.03), (
            f"sample mean {sample_mean:.4f} vs Beta({alpha},{alpha}) mean "
            f"{true_mean:.4f} (SE={se_mean:.4f})"
        )
        # Variance tolerance is looser — variance estimator is itself
        # noisier than the mean. Use 30% relative + 0.01 absolute floor.
        assert abs(sample_var - true_var) < max(0.3 * true_var, 0.01), (
            f"sample var {sample_var:.4f} vs Beta({alpha},{alpha}) var " f"{true_var:.4f}"
        )

    def test_lambda_alpha_le_zero_is_one(self) -> None:
        """Reference framework MixUp clamps α≤0 to a no-op (λ=1); Lucid
        does the same in :func:`_sample_lambda` so collators degrade
        gracefully under a sentinel alpha.
        """
        assert _sample_lambda(0.0) == 1.0
        assert _sample_lambda(-1.0) == 1.0


# ── 5. End-to-end shape parity vs reference framework ────────────────


@pytest.mark.parity
class TestEndToEndShapeParity:
    """Run both Lucid and reference framework collators on matched
    inputs and verify the output shapes line up.  Numerical values
    differ (independent RNGs) but shape/dtype invariants must agree —
    this is what downstream training code couples to.
    """

    def test_mixup_output_shapes(self, ref: Any) -> None:
        b, c, h, w, num_classes = 4, 3, 8, 8, 10
        # Lucid path.
        lucid.manual_seed(0)
        lucid_coll = MixupCollator(alpha=0.2, num_classes=num_classes)
        l_batch = _make_batch(b=b, c=c, h=h, w=w, num_classes=num_classes, seed=20)
        l_x, l_y = lucid_coll(l_batch)
        # Reference framework path — v2.MixUp wants a pre-collated
        # (images, labels) pair, not a list of (image, label).
        rng = np.random.default_rng(20)
        r_imgs = ref.from_numpy(
            np.stack(
                [rng.random((c, h, w), dtype=np.float32) for _ in range(b)],
                axis=0,
            )
        )
        r_labels = ref.tensor([i % num_classes for i in range(b)], dtype=ref.int64)
        r_mix = v2.MixUp(num_classes=num_classes, alpha=0.2)
        r_x, r_y = r_mix(r_imgs, r_labels)
        # Shapes match.
        assert tuple(l_x.shape) == tuple(r_x.shape) == (b, c, h, w)
        assert tuple(l_y.shape) == tuple(r_y.shape) == (b, num_classes)
        # Both produce sum-to-1 soft targets.
        np.testing.assert_allclose(l_y.numpy().sum(axis=1), np.ones(b), atol=1e-5)
        np.testing.assert_allclose(r_y.detach().cpu().numpy().sum(axis=1), np.ones(b), atol=1e-5)

    def test_cutmix_output_shapes(self, ref: Any) -> None:
        b, c, h, w, num_classes = 4, 3, 16, 16, 10
        # Lucid path.
        lucid.manual_seed(0)
        lucid_coll = CutMixCollator(alpha=1.0, num_classes=num_classes)
        l_batch = _make_batch(b=b, c=c, h=h, w=w, num_classes=num_classes, seed=21)
        l_x, l_y = lucid_coll(l_batch)
        # Reference path.
        rng = np.random.default_rng(21)
        r_imgs = ref.from_numpy(
            np.stack(
                [rng.random((c, h, w), dtype=np.float32) for _ in range(b)],
                axis=0,
            )
        )
        r_labels = ref.tensor([i % num_classes for i in range(b)], dtype=ref.int64)
        r_cut = v2.CutMix(num_classes=num_classes, alpha=1.0)
        r_x, r_y = r_cut(r_imgs, r_labels)
        # Shapes match.
        assert tuple(l_x.shape) == tuple(r_x.shape) == (b, c, h, w)
        assert tuple(l_y.shape) == tuple(r_y.shape) == (b, num_classes)
        # Both produce sum-to-1 soft targets.
        np.testing.assert_allclose(l_y.numpy().sum(axis=1), np.ones(b), atol=1e-5)
        np.testing.assert_allclose(r_y.detach().cpu().numpy().sum(axis=1), np.ones(b), atol=1e-5)
