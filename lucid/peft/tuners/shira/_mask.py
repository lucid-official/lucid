r"""Which entries of a weight a SHiRA adapter trains: the mask functions.

A mask function takes the module being adapted and the rank ``r`` and returns a
``(d_out, d_in)`` tensor of zeros and ones with exactly ``r (d_out + d_in)`` ones — as
many trained entries as a rank-``r`` LoRA pair has parameters.  The default picks them
at random: the first entries of a random permutation of the weight's positions, which for
a ``random_seed`` are the ones the reference framework's CPU generator picks (the same
permutation FourierFT's locations come from), so an adapter file moves between the
two libraries.
"""

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft.tuners.fourierft._locations import _spectrum_positions

#: The reference framework's default generator seed, which an unseeded ``random_mask`` uses.
_DEFAULT_SEED = 67280421310721


def random_mask(base_layer: nn.Module, r: int, random_seed: int | None = None) -> Tensor:
    r"""A random sparse mask for ``base_layer``'s weight with :math:`r\,(d_{out} + d_{in})` ones.

    Parameters
    ----------
    base_layer : nn.Module
        The module whose weight the mask is for.
    r : int
        The rank the number of ones is sized after.
    random_seed : int, optional
        Seed of the permutation; ``None`` is the reference's default seed.

    Returns
    -------
    Tensor
        ``(d_out, d_in)`` float tensor of 0 and 1, on the weight's device.

    Raises
    ------
    ValueError
        ``r (d_out + d_in)`` exceeds the number of weight entries.
    """
    weight = base_layer.weight
    assert isinstance(weight, Tensor)
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    count = r * (out_features + in_features)
    if count > out_features * in_features:
        raise ValueError(
            f"The set rank {r} results in more shira params than the total number of params in "
            f"the base layer {out_features * in_features} and this is not allowed."
        )
    seed = _DEFAULT_SEED if random_seed is None else random_seed
    positions = _spectrum_positions(out_features * in_features, count, seed)
    flat = lucid.zeros(out_features * in_features, device=weight.device)
    flat = flat.index_put(
        (lucid.tensor(list(positions), dtype=lucid.int64, device=weight.device),),
        lucid.ones(count, device=weight.device),
    )
    return flat.reshape(out_features, in_features)
