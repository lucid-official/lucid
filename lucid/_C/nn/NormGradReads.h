// lucid/_C/nn/NormGradReads.h
//
// Which saved inputs the gradients of an affine normalisation read.
//
// LayerNorm, GroupNorm and BatchNorm all take ``(x, gamma, beta)`` and share
// the one answer, so a write into an input that only an unneeded gradient
// reads (``x`` when only ``dbeta`` is wanted) is not refused — see
// :type:`GradReads` and :meth:`AutogradNode::validate_versions`.

#pragma once

#include "../autograd/AutogradNode.h"

namespace lucid {

// dx reads x and gamma; dgamma reads x; dbeta reads only grad_out.
inline constexpr GradReads<3> kAffineNormGradReads = {
    {{true, true, false}, {true, false, false}, {false, false, false}}};

}  // namespace lucid
