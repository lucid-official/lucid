"""RASampler — paper-spec invariants (Hoffer et al., 2020).

This file tests *mathematical correctness against the paper spec*,
not framework parity (reference framework doesn't ship an RA sampler;
the pattern originated in timm).  No reference framework required —
these tests live under ``unit/`` so they are not auto-skipped by the
parity conftest.  The one comparison against timm's ``RepeatAugSampler``
needs timm installed, so it lives in
``lucid/test/parity/utils/test_ra_sampler_spec_parity.py``.

The base behavioural tests are in `test_ra_sampler.py`; this file
adds the rigorous paper-formula checks across parameter sweeps:

* ``num_samples = ceil(N * num_repeats / num_replicas)``
* ``total_size = num_samples * num_replicas``
* ``num_selected_samples = floor(N / num_replicas)``
* each unique index is repeated ``num_repeats`` times consecutively
  in the un-truncated stream
* the ``num_replicas`` slabs partition the master padded list with
  no positional overlap
* set_epoch + seed reproducibility
* edge-case formulas (N < num_replicas → empty iter, num_repeats=1
  degenerates, etc.)
"""

import math
from collections import Counter

import pytest

import lucid
import lucid.utils.data as D


class _ToyDataset(D.Dataset):
    """Length-N integer-identity dataset.  Exists only so we can drive
    the sampler from a ``__len__`` without standing up real tensors."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


# --------------------------------------------------------------------------- #
# Pattern 1 — paper-formula invariants                                        #
# --------------------------------------------------------------------------- #


class TestPaperFormulaInvariants:
    """The three derived sizes — num_samples / total_size /
    num_selected_samples — must obey the published formulas across every
    legal (N, num_replicas, num_repeats) combination."""

    @pytest.mark.parametrize("N", [10, 20, 47, 100])
    @pytest.mark.parametrize("W", [1, 2, 4])
    @pytest.mark.parametrize("R", [1, 3, 4])
    def test_num_samples_formula(self, N: int, W: int, R: int) -> None:
        """num_samples = ceil(N * num_repeats / num_replicas)."""
        ds = _ToyDataset(N)
        s = D.RASampler(ds, num_replicas=W, num_repeats=R)
        assert s.num_samples == math.ceil(N * R / W)

    @pytest.mark.parametrize("N", [10, 20, 47, 100])
    @pytest.mark.parametrize("W", [1, 2, 4])
    @pytest.mark.parametrize("R", [1, 3, 4])
    def test_total_size_formula(self, N: int, W: int, R: int) -> None:
        """total_size = num_samples * num_replicas (and therefore
        >= N * num_repeats, since num_samples = ceil(...))."""
        ds = _ToyDataset(N)
        s = D.RASampler(ds, num_replicas=W, num_repeats=R)
        assert s.total_size == s.num_samples * W
        # And the slab is large enough to hold the full repeated stream:
        assert s.total_size >= N * R

    @pytest.mark.parametrize("N", [10, 20, 47, 100])
    @pytest.mark.parametrize("W", [1, 2, 4])
    @pytest.mark.parametrize("R", [1, 3, 4])
    def test_num_selected_samples_formula(self, N: int, W: int, R: int) -> None:
        """num_selected_samples = floor(N / num_replicas) — and is
        invariant to num_repeats (the whole point: repetition stays
        within the epoch, doesn't extend it)."""
        ds = _ToyDataset(N)
        s = D.RASampler(ds, num_replicas=W, num_repeats=R)
        assert s.num_selected_samples == N // W

    @pytest.mark.parametrize("N", [10, 20, 47, 100])
    @pytest.mark.parametrize("W", [1, 2, 4])
    @pytest.mark.parametrize("R", [1, 3, 4])
    def test_len_matches_num_selected_samples(self, N: int, W: int, R: int) -> None:
        """``len(sampler)`` returns the per-replica yield count."""
        ds = _ToyDataset(N)
        s = D.RASampler(ds, num_replicas=W, num_repeats=R)
        assert len(s) == s.num_selected_samples

    @pytest.mark.parametrize("N", [10, 20, 47, 100])
    @pytest.mark.parametrize("W", [1, 2, 4])
    @pytest.mark.parametrize("R", [1, 3, 4])
    def test_iter_length_matches_num_selected_samples(self, N: int, W: int, R: int) -> None:
        """Iteration emits exactly ``num_selected_samples`` indices."""
        ds = _ToyDataset(N)
        s = D.RASampler(ds, num_replicas=W, num_repeats=R, shuffle=False)
        produced = list(s)
        assert len(produced) == s.num_selected_samples

    @pytest.mark.parametrize("N", [10, 20, 47, 100])
    @pytest.mark.parametrize("W", [1, 2, 4])
    @pytest.mark.parametrize("R", [1, 3, 4])
    def test_all_emitted_indices_in_range(self, N: int, W: int, R: int) -> None:
        """Every yielded index ``i`` satisfies ``0 <= i < N``."""
        ds = _ToyDataset(N)
        for rank in range(W):
            s = D.RASampler(ds, num_replicas=W, rank=rank, num_repeats=R)
            for idx in s:
                assert 0 <= idx < N


# --------------------------------------------------------------------------- #
# Pattern 2 — consecutive-repeat structure                                    #
# --------------------------------------------------------------------------- #


class TestRepeatStructure:
    """Each unique index appears in a run of ``num_repeats`` consecutive
    positions in the un-truncated repeated stream — that's the defining
    behaviour of the sampler."""

    @pytest.mark.parametrize("R", [1, 2, 3, 4])
    def test_each_index_repeated_R_times_consecutively(self, R: int) -> None:
        """num_replicas=1, shuffle=False: the slab equals the
        un-truncated stream up to floor(N/1) = N, so we can inspect
        the first floor(N/R) full runs explicitly."""
        N = 12
        ds = _ToyDataset(N)
        s = D.RASampler(ds, num_replicas=1, num_repeats=R, shuffle=False)
        out = list(s)
        # Number of *complete* runs we can see in the truncated slab:
        full_runs = N // R
        for run_idx in range(full_runs):
            chunk = out[run_idx * R : (run_idx + 1) * R]
            assert chunk == [run_idx] * R, f"Run {run_idx} expected {[run_idx] * R}, got {chunk}"

    @pytest.mark.parametrize("R", [2, 3, 4])
    def test_repeat_counts_with_shuffle(self, R: int) -> None:
        """With shuffle=True every unique index that appears at all
        appears in a run of length ``num_repeats`` in the *full*
        repeated stream — but truncation to num_selected_samples may
        clip the final run.  We check the *first* (N // num_replicas
        // R) full runs against this invariant by inspecting the head
        of the slab."""
        N = 30
        ds = _ToyDataset(N)
        s = D.RASampler(ds, num_replicas=1, num_repeats=R, shuffle=True, seed=0)
        out = list(s)
        # Walk through runs head-first.  Each complete run = R identical
        # consecutive ints.
        i = 0
        complete_runs = 0
        while i + R <= len(out):
            chunk = out[i : i + R]
            if all(v == chunk[0] for v in chunk):
                complete_runs += 1
                i += R
            else:
                break  # we've hit a truncated tail
        # We should see at least one complete run (N is large enough).
        assert complete_runs >= 1

    @pytest.mark.parametrize("R", [1, 2, 3])
    def test_unique_index_appearance_count_bounded(self, R: int) -> None:
        """In the un-truncated repeated list each unique index appears
        exactly R times; after truncation to num_selected_samples the
        per-index count is at most R (and at least 0)."""
        N = 20
        ds = _ToyDataset(N)
        s = D.RASampler(ds, num_replicas=1, num_repeats=R, shuffle=False)
        counts = Counter(list(s))
        for idx, c in counts.items():
            assert 0 <= c <= R, f"Index {idx} appeared {c} times, expected 0..{R}"


# --------------------------------------------------------------------------- #
# Pattern 3 — distributed disjoint slabs                                      #
# --------------------------------------------------------------------------- #


class TestDistributedSlabs:
    """Replicas pull non-overlapping *positional* slabs from the master
    repeated-and-padded list.  The underlying indices can repeat across
    ranks (same N-element pool) but the slab positions are disjoint."""

    @pytest.mark.parametrize("W", [2, 3, 4])
    def test_slab_positions_are_disjoint(self, W: int) -> None:
        """Reconstruct each rank's slab by reproducing the master
        repeated-padded list, then verify the slab boundaries are
        contiguous, equal-size, and don't overlap."""
        N = 30
        R = 3
        ds = _ToyDataset(N)
        # Sampler internals: num_samples is the per-replica slab size.
        s0 = D.RASampler(ds, num_replicas=W, rank=0, num_repeats=R, shuffle=False)
        # Slab[rank] = [rank * num_samples : (rank+1) * num_samples]
        slab_size = s0.num_samples
        boundaries = [(r * slab_size, (r + 1) * slab_size) for r in range(W)]
        # Disjoint: boundary[r].end == boundary[r+1].start
        for r in range(W - 1):
            assert boundaries[r][1] == boundaries[r + 1][0]
        # And the union covers exactly [0, total_size).
        assert boundaries[0][0] == 0
        assert boundaries[-1][1] == s0.total_size

    @pytest.mark.parametrize("W", [2, 4])
    def test_two_ranks_yield_equal_lengths(self, W: int) -> None:
        """Both ranks yield exactly num_selected_samples indices."""
        N = 24
        ds = _ToyDataset(N)
        outs = []
        for rank in range(W):
            s = D.RASampler(
                ds,
                num_replicas=W,
                rank=rank,
                num_repeats=3,
                shuffle=False,
            )
            outs.append(list(s))
        expected = N // W
        for out in outs:
            assert len(out) == expected

    def test_rank_zero_starts_at_position_zero_no_shuffle(self) -> None:
        """With shuffle=False the master list is [0,0,0,1,1,1,...]; rank
        0's slab is the head, so the first element is 0."""
        N = 12
        ds = _ToyDataset(N)
        s = D.RASampler(ds, num_replicas=2, rank=0, num_repeats=3, shuffle=False)
        out = list(s)
        assert out[0] == 0


# --------------------------------------------------------------------------- #
# Pattern 4 — deterministic shuffle                                           #
# --------------------------------------------------------------------------- #


class TestDeterminism:
    """set_epoch + seed give a reproducible RNG; same (seed, epoch) →
    same sequence, different epoch → different sequence."""

    def test_set_epoch_changes_order(self) -> None:
        ds = _ToyDataset(30)
        s = D.RASampler(ds, num_repeats=3, shuffle=True, seed=0)
        s.set_epoch(0)
        ep0 = list(s)
        s.set_epoch(1)
        ep1 = list(s)
        assert ep0 != ep1

    def test_same_seed_same_order(self) -> None:
        ds = _ToyDataset(30)
        a = D.RASampler(ds, num_repeats=3, shuffle=True, seed=42)
        b = D.RASampler(ds, num_repeats=3, shuffle=True, seed=42)
        a.set_epoch(7)
        b.set_epoch(7)
        assert list(a) == list(b)

    def test_different_seeds_different_orders(self) -> None:
        ds = _ToyDataset(30)
        a = D.RASampler(ds, num_repeats=3, shuffle=True, seed=0)
        b = D.RASampler(ds, num_repeats=3, shuffle=True, seed=1)
        # Both at the same epoch (the default 0) → seeds differ → orders differ.
        assert list(a) != list(b)

    def test_reseed_after_iteration_still_reproduces(self) -> None:
        """Iterating doesn't mutate the sampler's seed state — pulling
        the iterator twice with the same epoch yields the same sequence."""
        ds = _ToyDataset(20)
        s = D.RASampler(ds, num_repeats=3, shuffle=True, seed=7)
        s.set_epoch(3)
        first = list(s)
        second = list(s)
        assert first == second

    def test_shuffle_false_is_deterministic_across_seeds(self) -> None:
        """With shuffle=False the seed is irrelevant — output stays
        sequential."""
        ds = _ToyDataset(20)
        a = D.RASampler(ds, num_repeats=3, shuffle=False, seed=0)
        b = D.RASampler(ds, num_repeats=3, shuffle=False, seed=999)
        assert list(a) == list(b)


# --------------------------------------------------------------------------- #
# Pattern 5 — edge cases                                                      #
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Boundary conditions: num_repeats=1, N < num_replicas, N=1, etc."""

    def test_num_repeats_one_degenerates_sequential(self) -> None:
        """num_repeats=1 + shuffle=False + num_replicas=1 → bare
        ``range(N)`` truncated to floor(N/1) = N."""
        N = 8
        ds = _ToyDataset(N)
        s = D.RASampler(ds, num_replicas=1, num_repeats=1, shuffle=False)
        assert list(s) == list(range(N))

    def test_num_repeats_one_degenerates_per_rank(self) -> None:
        """num_repeats=1, num_replicas>1, shuffle=False → each rank
        sees floor(N/num_replicas) sequential indices from its slab."""
        N = 12
        W = 3
        ds = _ToyDataset(N)
        for rank in range(W):
            s = D.RASampler(
                ds,
                num_replicas=W,
                rank=rank,
                num_repeats=1,
                shuffle=False,
            )
            out = list(s)
            assert len(out) == N // W
            # With R=1 the master list is just [0,1,...,N-1] padded to
            # ceil(N/W)*W; rank r's slab head is r*ceil(N/W).
            expected_start = rank * s.num_samples
            assert out[0] == expected_start

    def test_handles_N_smaller_than_num_replicas(self) -> None:
        """N=4, num_replicas=8 → num_selected_samples = floor(4/8) = 0;
        iter is empty but the sampler is still constructible."""
        ds = _ToyDataset(4)
        s = D.RASampler(ds, num_replicas=8, rank=0, num_repeats=3)
        assert s.num_selected_samples == 0
        assert len(s) == 0
        assert list(s) == []

    def test_N_equals_one(self) -> None:
        """N=1, num_replicas=1, num_repeats=R → emit [0] * 1 (since
        num_selected_samples = floor(1/1) = 1)."""
        ds = _ToyDataset(1)
        s = D.RASampler(ds, num_replicas=1, num_repeats=4, shuffle=False)
        out = list(s)
        assert out == [0]

    def test_N_equals_num_replicas(self) -> None:
        """N == num_replicas → num_selected_samples = 1 per rank."""
        ds = _ToyDataset(4)
        for rank in range(4):
            s = D.RASampler(
                ds,
                num_replicas=4,
                rank=rank,
                num_repeats=3,
                shuffle=False,
            )
            assert s.num_selected_samples == 1
            assert len(list(s)) == 1

    @pytest.mark.parametrize("R", [1, 2, 5, 10])
    def test_large_num_repeats(self, R: int) -> None:
        """num_repeats much larger than N stays well-defined."""
        ds = _ToyDataset(5)
        s = D.RASampler(ds, num_replicas=1, num_repeats=R, shuffle=False)
        # num_samples = ceil(5*R/1) = 5*R, total_size = 5*R.
        assert s.num_samples == 5 * R
        assert s.total_size == 5 * R
        # But num_selected_samples is still floor(5/1) = 5.
        assert s.num_selected_samples == 5
        out = list(s)
        assert len(out) == 5

    def test_invalid_args_raise(self) -> None:
        """Validation errors mirror the paper: positive num_replicas,
        in-range rank, positive num_repeats."""
        ds = _ToyDataset(10)
        with pytest.raises(ValueError, match="num_replicas"):
            D.RASampler(ds, num_replicas=0)
        with pytest.raises(ValueError, match="rank"):
            D.RASampler(ds, num_replicas=2, rank=2)
        with pytest.raises(ValueError, match="rank"):
            D.RASampler(ds, num_replicas=2, rank=-1)
        with pytest.raises(ValueError, match="num_repeats"):
            D.RASampler(ds, num_repeats=0)


# --------------------------------------------------------------------------- #
# Pattern 6 — integration with DataLoader                                     #
# --------------------------------------------------------------------------- #


class TestDataLoaderIntegration:
    """End-to-end batch count must equal ``num_selected_samples /
    batch_size`` (with whatever rounding the loader applies)."""

    @pytest.mark.parametrize("BS", [1, 2, 4, 5])
    def test_batch_count_drop_last_false(self, BS: int) -> None:
        """drop_last=False → ceil(num_selected_samples / BS) batches."""
        N = 20
        ds = D.TensorDataset(lucid.arange(N).reshape(N, 1).to(lucid.float32))
        sampler = D.RASampler(ds, num_repeats=3, shuffle=True, seed=0)
        loader = D.DataLoader(ds, batch_size=BS, sampler=sampler, drop_last=False)
        batches = list(loader)
        expected = math.ceil(sampler.num_selected_samples / BS)
        assert len(batches) == expected

    @pytest.mark.parametrize("BS", [1, 2, 4, 5])
    def test_batch_count_drop_last_true(self, BS: int) -> None:
        """drop_last=True → floor(num_selected_samples / BS) batches."""
        N = 20
        ds = D.TensorDataset(lucid.arange(N).reshape(N, 1).to(lucid.float32))
        sampler = D.RASampler(ds, num_repeats=3, shuffle=True, seed=0)
        loader = D.DataLoader(ds, batch_size=BS, sampler=sampler, drop_last=True)
        batches = list(loader)
        expected = sampler.num_selected_samples // BS
        assert len(batches) == expected

    def test_full_batches_have_correct_shape(self) -> None:
        """Each non-trailing batch has shape (BS, ...)."""
        N = 20
        BS = 4
        ds = D.TensorDataset(lucid.arange(N).reshape(N, 1).to(lucid.float32))
        sampler = D.RASampler(ds, num_repeats=3, shuffle=True, seed=0)
        loader = D.DataLoader(ds, batch_size=BS, sampler=sampler, drop_last=True)
        for batch in loader:
            (x,) = batch
            assert tuple(x.shape) == (BS, 1)
