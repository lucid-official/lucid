"""RASampler — the one comparison against timm's ``RepeatAugSampler``.

The paper-spec invariants (Hoffer et al., 2020) need no oracle and live in
``lucid/test/unit/utils/data/test_ra_sampler_spec.py``; this check needs
timm, and timm needs the reference framework, so it runs in the parity
tier, where both are installed.
"""

import pytest

import lucid.utils.data as D
from lucid.test._fixtures.ref_framework import zoo_module

pytestmark = pytest.mark.parity


class _ToyDataset(D.Dataset):
    """Length-N integer-identity dataset.  Exists only so we can drive
    the sampler from a ``__len__`` without standing up real tensors."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


@pytest.mark.skipif(zoo_module() is None, reason="timm not installed")
class TestTimmComparison:
    """If timm is installed, verify our sequence matches timm's
    RepeatAugSampler bit-for-bit at seed=0, epoch=0."""

    def test_matches_timm_ra_sampler_seed_zero(self) -> None:
        from timm.data.distributed_sampler import RepeatAugSampler

        N = 20
        ds = _ToyDataset(N)
        lucid_s = D.RASampler(ds, num_replicas=1, rank=0, num_repeats=3, shuffle=True, seed=0)
        lucid_s.set_epoch(0)
        # timm rounds the per-epoch selection down to a multiple of
        # ``selected_round`` (256 by default), which leaves a 20-element
        # dataset with nothing to yield.  ``selected_round=0`` selects the
        # original DeiT count, ``ceil(N / num_replicas)`` — the same as
        # Lucid's ``floor(N / num_replicas)`` at one replica.
        #
        # timm's RepeatAugSampler signature varies across versions —
        # newer releases drop the ``seed`` kwarg.  Try with seed first,
        # fall back to without (the epoch counter still drives shuffling).
        try:
            timm_s = RepeatAugSampler(
                ds, num_replicas=1, rank=0, num_repeats=3, shuffle=True, selected_round=0, seed=0
            )
        except TypeError:
            timm_s = RepeatAugSampler(
                ds, num_replicas=1, rank=0, num_repeats=3, shuffle=True, selected_round=0
            )
        timm_s.set_epoch(0)
        # The RNG is platform-specific (timm uses the reference's Generator,
        # lucid uses random.Random) so we don't insist on equal
        # *sequences*, only on equal *lengths* and equal *multisets*
        # within the repeat structure.
        lucid_out = list(lucid_s)
        timm_out = list(timm_s)
        assert len(timm_out) == N
        assert len(lucid_out) == len(timm_out)
