// lucid/_C/compile/OpEmitters/_Pool3dLayout.h
//
// The shapes a 3-D pool emitter and its VJP work in (LCD-927).
//
// MPSGraph ships 2-D and 4-D pooling.  The 4-D operation pools the last four axes of its
// source with a window per axis, so a 3-D volume ``(N, C, D, H, W)`` reached it by a
// length-1 axis in front of the depth — ``(N, C, 1, D, H, W)``, rank 6.  A rank-6 source
// runs as a slice loop over ``N * C`` (``avg_pool3d`` k=(1,2,2) on ``(64, 16, 8, 32, 32)``
// took 8.2 ms against 1.5 ms at rank 4), and a max pool's backward, which slices that
// source once per window offset, 168 ms against 9.  The unit axis is not needed: the
// pooled axes are the last three, and the 4-D operation's leading window of 1 can sit on
// ``N * C`` merged into one axis — ``(N*C, D, H, W)``, rank 4, the same descriptor.  The
// max pool's gradient, which wants a batch and a channel axis in front of the pooled
// ones, reads ``(1, N*C, D, H, W)`` instead: rank 5 with a lead of 1, which costs nothing.
//
// The policy is the pass's: a small ``N * C`` keeps the rank-6 form (``rank_canon_wants``
// of that shape), and a symbolic dim keeps it too (the batch survives a reshape only as a
// ``-1``).
//
// Pure C++ — include from ``.mm`` files only.

#pragma once

#include <cstdint>
#include <vector>

#include "../RankCanon.h"

namespace lucid::compile {

// Whether a pool of a ``(N, C, D, H, W)`` volume runs on the merged ``N * C`` axis.
[[maybe_unused]] inline bool pool3d_canonical(const std::vector<std::int64_t>& x_shape) {
    if (x_shape.size() != 5)
        return false;
    for (std::int64_t d : x_shape)
        if (d <= 0)
            return false;
    return rank_canon_wants({x_shape[0], x_shape[1], 1, x_shape[2], x_shape[3], x_shape[4]});
}

// ``(N, C, a, b, c)`` as the 4-D operation sees it: ``(N*C, a, b, c)`` when canonical, else
// ``(N, C, 1, a, b, c)``.
[[maybe_unused]] inline std::vector<std::int64_t> pool3d_lift(const std::vector<std::int64_t>& s,
                                                              bool canonical) {
    if (canonical)
        return {s[0] * s[1], s[2], s[3], s[4]};
    return {s[0], s[1], 1, s[2], s[3], s[4]};
}

// ``(N, C, a, b, c)`` for the max pool's gradient kernels: ``(1, N*C, a, b, c)``.
[[maybe_unused]] inline std::vector<std::int64_t>
pool3d_fold_batch(const std::vector<std::int64_t>& s) {
    return {1, s[0] * s[1], s[2], s[3], s[4]};
}

}  // namespace lucid::compile
