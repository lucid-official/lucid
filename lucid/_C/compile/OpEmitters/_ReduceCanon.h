// lucid/_C/compile/OpEmitters/_ReduceCanon.h
//
// Rank canonicalisation for the reduction emitters and their VJPs.
//
// MPSGraph runs a tensor of rank 5 or more through a far slower kernel than
// the same elements at rank 4: an elementwise op or a reduction over
// ``(6144, 16, 2, 16, 2)`` took ~55 ms against ~2 ms for ``(6144, 16, 2, 32)``
// (LCD-918).  A reduction never needs that rank — what matters is which
// elements land in the same output cell, not the grouping of the axes that
// produced them.  Adjacent kept axes can be fused (they stay in one row-major
// block), adjacent reduced axes can be fused, and a size-1 axis sits between
// them for free, so any reduction collapses to alternating kept / reduced
// blocks: rank <= 4 for the ``(.., r, .., r)`` patterns pooling-style reshapes
// produce, and rank 1 for a full reduction.  Values are the same elements
// reduced in the same sets; only the order a float sum accumulates in is up to
// MPSGraph, as it already was.
//
// Pure C++ plus a few ObjC++ helpers — include from ``.mm`` files only.

#pragma once

#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "_AttrHelpers.h"

namespace lucid::compile {

// Ranks below this keep their emitted shape: the measured slowness starts at
// rank 5 (scout, LCD-918), and rewriting rank <= 4 graphs would only change
// graphs that already run well.
inline constexpr std::size_t kReduceCanonMinRank = 5;

// A reduction rewritten at lower rank.  ``merged_in`` is the input reshaped so
// each maximal run of same-kind axes is one axis; ``merged_axes`` are the
// reduced ones; ``merged_keep`` is ``merged_in`` with those set to 1 (the
// result's keepdim form).  A symbolic (-1) dim fuses into -1.
struct ReduceLayout {
    std::vector<std::int64_t> merged_in;
    std::vector<std::int64_t> merged_keep;
    std::vector<std::int64_t> merged_axes;
    // The input's leading dim is the symbolic batch and survives the reduction,
    // so the result's leading dim is symbolic too.
    bool symbolic_lead_kept = false;
};

// Which axes ``dims`` names (negative ones counted from the end); nullopt for an
// axis out of range.
[[maybe_unused]] inline std::optional<std::vector<bool>>
reduce_canon_mask(std::size_t rank, const std::vector<std::int64_t>& dims) {
    std::vector<bool> reduced(rank, false);
    const auto r = static_cast<std::int64_t>(rank);
    for (std::int64_t d : dims) {
        if (d < 0)
            d += r;
        if (d < 0 || d >= r)
            return std::nullopt;
        reduced[static_cast<std::size_t>(d)] = true;
    }
    return reduced;
}

// Fuse ``in_shape`` into alternating kept / reduced blocks; returns each
// block's kind and fills ``merged`` with its size.  A size-1 axis is neutral —
// it joins whichever block surrounds it — and a tensor of nothing else is one
// reduced block of 1.
[[maybe_unused]] inline std::vector<bool> reduce_canon_blocks(const std::vector<std::int64_t>& in_shape,
                                                             const std::vector<bool>& reduced,
                                                             std::vector<std::int64_t>& merged) {
    std::vector<bool> block_reduced;
    for (std::size_t i = 0; i < in_shape.size(); ++i) {
        const std::int64_t s = in_shape[i];
        if (s == 1)
            continue;
        if (!block_reduced.empty() && block_reduced.back() == reduced[i]) {
            std::int64_t& tail = merged.back();
            tail = (tail < 0 || s < 0) ? -1 : tail * s;
        } else {
            merged.push_back(s);
            block_reduced.push_back(reduced[i]);
        }
    }
    if (merged.empty()) {
        merged.push_back(1);
        block_reduced.push_back(true);
    }
    return block_reduced;
}

// The canonical layout of reducing ``dims`` of an ``in_shape`` tensor, or
// nullopt when the legacy form should stand: input rank under the threshold, no
// axes named, an axis out of range, an empty tensor, or nothing to fuse.
[[maybe_unused]] inline std::optional<ReduceLayout>
canonical_reduce_layout(const std::vector<std::int64_t>& in_shape,
                        const std::vector<std::int64_t>& dims) {
    if (in_shape.size() < kReduceCanonMinRank || dims.empty() ||
        std::find(in_shape.begin(), in_shape.end(), 0) != in_shape.end())
        return std::nullopt;
    const auto reduced = reduce_canon_mask(in_shape.size(), dims);
    if (!reduced)
        return std::nullopt;
    ReduceLayout out;
    const std::vector<bool> block_reduced = reduce_canon_blocks(in_shape, *reduced, out.merged_in);
    if (out.merged_in.size() >= in_shape.size())
        return std::nullopt;
    out.symbolic_lead_kept = in_shape[0] < 0 && !(*reduced)[0];
    out.merged_keep = out.merged_in;
    for (std::size_t i = 0; i < block_reduced.size(); ++i) {
        if (!block_reduced[i])
            continue;
        out.merged_keep[i] = 1;
        out.merged_axes.push_back(static_cast<std::int64_t>(i));
    }
    return out;
}

// The shapes a reduction VJP works in: the input's own, or — for a rank-5+
// input whose shape is fully concrete — its canonical lower rank.  ``enter``
// moves the input there, ``keep`` moves the reduction's result or gradient to
// its keepdim form there (so it broadcasts against the input), ``leave`` moves
// a gradient back to the input's own shape.  A symbolic dim keeps the legacy
// form: the broadcasts below name the input shape outright.
class ReduceWork {
public:
    ReduceWork(const std::vector<std::int64_t>& in_shape, const std::vector<bool>& reduced)
        : in_shape_(in_shape), kept_(in_shape), work_(in_shape) {
        std::vector<std::int64_t> dims;
        for (std::size_t d = 0; d < in_shape.size(); ++d) {
            if (reduced[d])
                dims.push_back(static_cast<std::int64_t>(d));
        }
        const bool symbolic =
            std::any_of(in_shape.begin(), in_shape.end(), [](std::int64_t s) { return s < 0; });
        if (auto layout = symbolic ? std::nullopt : canonical_reduce_layout(in_shape, dims)) {
            canonical_ = true;
            work_ = std::move(layout->merged_in);
            kept_ = std::move(layout->merged_keep);
            axes_ = std::move(layout->merged_axes);
            return;
        }
        for (std::int64_t d : dims)
            kept_[static_cast<std::size_t>(d)] = 1;
        axes_ = std::move(dims);
    }

    bool canonical() const { return canonical_; }
    const std::vector<std::int64_t>& work_shape() const { return work_; }
    const std::vector<std::int64_t>& kept_shape() const { return kept_; }
    const std::vector<std::int64_t>& axes() const { return axes_; }
    NSArray<NSNumber*>* axes_ns() const { return shape_to_ns(axes_); }

    MPSGraphTensor* enter(MPSGraph* g, MPSGraphTensor* x) const {
        return canonical_ ? [g reshapeTensor:x withShape:shape_to_ns(work_) name:nil] : x;
    }
    MPSGraphTensor* keep(MPSGraph* g, MPSGraphTensor* t) const {
        return [g reshapeTensor:t withShape:shape_to_ns(kept_) name:nil];
    }
    MPSGraphTensor* leave(MPSGraph* g, MPSGraphTensor* dx) const {
        return canonical_ ? [g reshapeTensor:dx withShape:shape_to_ns(in_shape_) name:nil] : dx;
    }

private:
    std::vector<std::int64_t> in_shape_;
    std::vector<std::int64_t> kept_;
    std::vector<std::int64_t> work_;
    std::vector<std::int64_t> axes_;
    bool canonical_ = false;
};

}  // namespace lucid::compile
