// lucid/_C/compile/OpEmitters/_AxisSplitLayout.h
//
// The shapes a repeat / tile emitter and its VJP work in (LCD-995).
//
// ``repeat`` is spelled as "insert a unit axis after ``axis``, broadcast it to ``reps``,
// collapse the two", its VJP as the mirror ("fold the axis into ``(d, r)``, sum ``r``"), and
// a tile's VJP interleaves ``(r, d)`` for every axis at once.  On a rank-4 operand the first
// two read and write a rank-5 tensor and the last a rank-2n one, and MPSGraph runs an op
// on a rank >= 5 tensor as a slice loop over the dims above the last four (~4.5 us each):
// ``repeat`` of ``(8192, 8, 16, 16)`` took 51 ms against 3.2 ms for the same elements at
// rank 3, and the backward of a tile of ``(65536, 16, 16)`` 460 ms against ~10.  The pass
// (``RankCanon``) cannot see it: it keys on a node's own rank, and the VJP is built after it.
//
// The op names one axis (``tile``: one at a time), so the axes before it merge and the axes
// after it merge: the intermediate is ``(pre, d, r, post)`` — rank 4 whatever the operand's.
// The policy is the pass's (``rank_canon_wants``, of the intermediate the native spelling
// would build): a small lead keeps the native form, ``LUCID_COMPILE_RANK_CANON_MIN_LEAD=-1``
// turns the layout off, and a symbolic dim keeps the native form too (its batch survives a
// reshape only as a ``-1``, and a product with it is unknowable here).  The forward emitter and
// the VJP take their shapes from here, so they agree.
//
// Pure C++ — include from ``.mm`` files only.

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <vector>

#include "../RankCanon.h"

namespace lucid::compile {

// The axes before and after the one an op names, each merged into one size.
struct AxisBands {
    std::int64_t pre = 1;
    std::int64_t post = 1;

    // ``pre``, ``mid`` and ``post`` as a shape; a band of size 1 carries nothing and is left out.
    std::vector<std::int64_t> shape(std::initializer_list<std::int64_t> mid) const {
        std::vector<std::int64_t> out;
        if (pre != 1)
            out.push_back(pre);
        out.insert(out.end(), mid.begin(), mid.end());
        if (post != 1)
            out.push_back(post);
        return out;
    }

    // Where ``mid[k]`` sits in ``shape(mid)``.
    std::size_t index_of(std::size_t k) const { return (pre != 1 ? 1U : 0U) + k; }
};

// Whether the intermediate ``native`` — what the op would build with no merging — is one to
// keep out of MPSGraph's slice loop: concrete, rank 5+ and over the pass's lead gate.
[[maybe_unused]] inline bool wants_axis_bands(const std::vector<std::int64_t>& native) {
    for (std::int64_t d : native) {
        if (d <= 0)
            return false;
    }
    return rank_canon_wants(native);
}

// ``shape``'s axes before ``axis`` merged and its axes after it merged.
[[maybe_unused]] inline AxisBands axis_bands(const std::vector<std::int64_t>& shape,
                                             std::size_t axis) {
    AxisBands bands;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i < axis)
            bands.pre *= shape[i];
        else if (i > axis)
            bands.post *= shape[i];
    }
    return bands;
}

// ``repeat(x, reps, axis)`` as shapes: ``x`` is read as ``unit`` (a unit axis after the
// repeated one) and broadcast to ``spread``, which a reshape turns into the result (the
// repeated axis and ``reps`` collapse in row-major order).  The VJP reads the result back
// as ``spread`` and sums ``spread_axis``.
struct RepeatLayout {
    std::vector<std::int64_t> unit;
    std::vector<std::int64_t> spread;
    std::size_t spread_axis = 0;
};

[[maybe_unused]] inline RepeatLayout
repeat_layout(const std::vector<std::int64_t>& x, std::size_t axis, std::int64_t reps) {
    RepeatLayout out;
    out.unit = x;
    out.unit.insert(out.unit.begin() + static_cast<std::ptrdiff_t>(axis) + 1, 1);
    out.spread = out.unit;
    out.spread[axis + 1] = reps;
    out.spread_axis = axis + 1;
    if (wants_axis_bands(out.spread)) {
        const AxisBands bands = axis_bands(x, axis);
        out.unit = bands.shape({x[axis], 1});
        out.spread = bands.shape({x[axis], reps});
        out.spread_axis = bands.index_of(1);
    }
    return out;
}

// One step of a tile's backward: the gradient read as ``split`` (``(pre, r, d, post)``),
// summed over ``sum_axis`` (``r``) and read as ``folded`` (``(pre, d, post)``).
struct TileFold {
    std::vector<std::int64_t> split;
    std::size_t sum_axis = 0;
    std::vector<std::int64_t> folded;
};

// The steps of the backward of ``tile(x, reps)``, one per tiled axis, each at rank <= 4;
// nullopt keeps the native single reduction over the interleaved ``(r0, d0, r1, d1, ...)``,
// which has rank 2n.
[[maybe_unused]] inline std::optional<std::vector<TileFold>>
tile_folds(const std::vector<std::int64_t>& x, const std::vector<std::int64_t>& reps) {
    if (x.size() != reps.size())
        return std::nullopt;
    std::vector<std::int64_t> interleaved;
    std::vector<std::int64_t> tiled = x;
    for (std::size_t a = 0; a < x.size(); ++a) {
        interleaved.push_back(reps[a]);
        interleaved.push_back(x[a]);
        tiled[a] = x[a] * reps[a];
    }
    if (!wants_axis_bands(interleaved))
        return std::nullopt;

    std::vector<TileFold> folds;
    for (std::size_t a = 0; a < x.size(); ++a) {
        if (reps[a] == 1)
            continue;
        const AxisBands bands = axis_bands(tiled, a);
        folds.push_back({bands.shape({reps[a], x[a]}), bands.index_of(0), bands.shape({x[a]})});
        tiled[a] = x[a];
    }
    return folds;
}

}  // namespace lucid::compile
