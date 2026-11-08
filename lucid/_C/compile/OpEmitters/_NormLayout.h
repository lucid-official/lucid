// lucid/_C/compile/OpEmitters/_NormLayout.h
//
// The shapes a layer_norm / rms_norm emitter and its VJP work in (LCD-927).
//
// Both normalise over the trailing K axes of ``x``.  Spelled out on the input's own
// shape, the mean, variance and every elementwise op of the forward and the backward
// read a tensor of rank 5 or more, which MPSGraph runs as a slice loop over the dims
// above the last four: ``layer_norm(x, (16,))`` on ``(4096, 4, 8, 16, 16)`` took 93.5 ms
// against 12 ms for the same elements at rank 2, and its backward 169 ms against 17.
// Which elements share a row is all the statistics need, so the canonical form merges
// the leading axes and the normalised axes: ``(prod(lead), prod(norm))`` with the
// reduction over axis 1, gamma / beta broadcast as ``(1, prod(norm))``, and ``dgamma`` /
// ``dbeta`` summed over axis 0.  Forward and VJP take the form from here, so they agree.
//
// The policy is the pass's (``rank_canon_wants``): a small lead keeps the native form —
// nothing MPSGraph loops over is worth a reshape — and ``LUCID_COMPILE_RANK_CANON_MIN_LEAD``
// turns it off.  A symbolic dim keeps the native form too: its batch survives a reshape
// only as a ``-1``, and a product with it is unknowable here.
//
// Pure C++ — include from ``.mm`` files only.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "../RankCanon.h"

namespace lucid::compile {

struct TrailingNormLayout {
    // ``(prod(lead), prod(norm))``: where the statistics and the normalisation happen.
    std::vector<std::int64_t> work;
    // gamma / beta's broadcast shape on ``work``: ``(1, prod(norm))``.
    std::vector<std::int64_t> affine;
};

// The canonical layout for a norm over the trailing ``k`` axes of ``x_shape``; nullopt
// when the native form stays (rank below 5, a small lead, a symbolic dim, ``k`` out of range).
[[maybe_unused]] inline std::optional<TrailingNormLayout>
trailing_norm_layout(const std::vector<std::int64_t>& x_shape, std::size_t k) {
    if (k == 0 || k > x_shape.size() || !rank_canon_wants(x_shape))
        return std::nullopt;
    std::int64_t lead = 1;
    std::int64_t norm = 1;
    const std::size_t split = x_shape.size() - k;
    for (std::size_t i = 0; i < x_shape.size(); ++i) {
        if (x_shape[i] <= 0)
            return std::nullopt;
        (i < split ? lead : norm) *= x_shape[i];
    }
    return TrailingNormLayout{{lead, norm}, {1, norm}};
}

}  // namespace lucid::compile
