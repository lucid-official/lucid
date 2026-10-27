"""
lucid.nn.utils: utility functions for neural networks.
"""

from lucid.nn.utils.clip_grad import (
    clip_grad_norm_,
    clip_grad_norm,
    clip_grad_value_,
    get_total_norm,
)
from lucid.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from lucid.nn.utils.weight_norm import remove_weight_norm, weight_norm
from lucid.nn.utils.spectral_norm import remove_spectral_norm, spectral_norm
from lucid.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pack_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from lucid.nn.utils._copy import copy_parameters_and_buffers
from lucid.nn.utils import parametrize as parametrize
from lucid.nn.utils import parametrizations as parametrizations
from lucid.nn.utils import prune as prune
from lucid.nn.utils import fusion as fusion
from lucid.nn.utils.fusion import (
    fuse_conv_bn_eval,
    fuse_conv_bn_weights,
    fuse_linear_bn_eval,
    fuse_linear_bn_weights,
)
from lucid.nn.utils.skip_init import skip_init

__all__ = [
    "clip_grad_norm_",
    "clip_grad_norm",
    "clip_grad_value_",
    "get_total_norm",
    "parameters_to_vector",
    "vector_to_parameters",
    "weight_norm",
    "remove_weight_norm",
    "spectral_norm",
    "remove_spectral_norm",
    "PackedSequence",
    "pack_padded_sequence",
    "pack_sequence",
    "pad_packed_sequence",
    "pad_sequence",
    "copy_parameters_and_buffers",
    "fuse_conv_bn_eval",
    "fuse_conv_bn_weights",
    "fuse_linear_bn_eval",
    "fuse_linear_bn_weights",
    "parametrize",
    "parametrizations",
    "prune",
    "fusion",
    "skip_init",
]
