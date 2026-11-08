// lucid/_C/compile/OpEmitters/_GroupNormLayout.h
//
// The shapes a group_norm emitter and its VJP work in.
//
// The textbook form reshapes ``(N, C, *spatial)`` to ``(N, G, C/G, *spatial)``
// and reduces the trailing axes — rank 5 for a 4-D input.  MPSGraph runs any
// elementwise op or reduction on a rank >= 5 tensor through a slice loop, so
// ``N = 256`` took 9.8 ms against 2.7 ms for the same elements at rank 2
// (LCD-920d).  A group's statistics need only "which elements share a row", so
// the canonical form is ``(N*G, C/G * prod(spatial))`` with the reduction over
// axis 1, and the per-channel affine runs on ``(N, C, prod(spatial))`` (rank 3
// whatever the input rank).  A symbolic dim keeps the legacy grouped form: its
// batch survives the reshape only as a ``-1``, and a product with it is
// unknowable here.
//
// Pure C++ — include from ``.mm`` files only.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "_ReduceCanon.h"

namespace lucid::compile {

struct GroupNormLayout {
    // Where mean / variance / normalisation happen and the axes reduced there.
    std::vector<std::int64_t> work;
    std::vector<std::int64_t> work_axes;
    // Where the per-channel affine and the dgamma / dbeta sums happen, the axes
    // those sums reduce, and gamma / beta's broadcast shape there.
    std::vector<std::int64_t> chan;
    std::vector<std::int64_t> chan_sum_axes;
    std::vector<std::int64_t> affine;
    bool canonical = false;
};

// The layout for ``x_shape`` split into ``num_groups`` groups; nullopt when the
// input is not a (N, C, ...) tensor with C divisible by ``num_groups``.
[[maybe_unused]] inline std::optional<GroupNormLayout>
group_norm_layout(const std::vector<std::int64_t>& x_shape, std::int64_t num_groups) {
    const std::size_t rank = x_shape.size();
    if (rank < 2 || num_groups <= 0)
        return std::nullopt;
    const std::int64_t C = x_shape[1];
    if (C % num_groups != 0)
        return std::nullopt;
    const std::int64_t per_group = C / num_groups;
    GroupNormLayout out;
    out.affine.assign(rank, 1);
    out.affine[1] = C;

    bool concrete = true;
    std::int64_t spatial = 1;
    for (std::size_t i = 0; i < rank; ++i) {
        concrete = concrete && x_shape[i] > 0;
        if (i >= 2 && x_shape[i] > 0)
            spatial *= x_shape[i];
    }
    // The grouped form has rank + 1 axes; the same threshold as the reductions.
    if (concrete && rank + 1 >= kReduceCanonMinRank) {
        out.canonical = true;
        out.work = {x_shape[0] * num_groups, per_group * spatial};
        out.work_axes = {1};
        out.chan = {x_shape[0], C, spatial};
        out.chan_sum_axes = {0, 2};
        out.affine = {1, C, 1};
        return out;
    }

    out.work = {x_shape[0], num_groups, per_group};
    out.work.insert(out.work.end(), x_shape.begin() + 2, x_shape.end());
    for (std::size_t i = 2; i < out.work.size(); ++i)
        out.work_axes.push_back(static_cast<std::int64_t>(i));
    out.chan = x_shape;
    for (std::size_t i = 0; i < rank; ++i) {
        if (i != 1)
            out.chan_sum_axes.push_back(static_cast<std::int64_t>(i));
    }
    return out;
}

}  // namespace lucid::compile
