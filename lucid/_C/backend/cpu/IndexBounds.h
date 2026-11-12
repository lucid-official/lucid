// lucid/_C/backend/cpu/IndexBounds.h
//
// Index values read by a CPU kernel: at their own width, and checked against
// what they index before anything is read or written through them.
//
// The CPU kernels answer an out-of-range index with IndexError, as the
// reference does.  Several did not: embedding copied a row from wherever the
// index pointed, embedding_bag read every index through ``int`` (2^32 + 1
// became row 1) and walked offsets past the end of the indices, and one_hot
// left a bad class's row zero.  Metal isolates the same indices in the graph
// instead (policy B, ``lucid/_C/backend/gpu/AxisIndex.h``, LCD-228), and
// refuses them in the same words when ``LUCID_STRICT_INDEX`` is set.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/IndexRefusal.h"
#include "../../core/Storage.h"

namespace lucid::backend::cpu {

// An index buffer read at its own width.  Read at another width an index is
// a different number (an int64 read as int32 lanes, an int32 as the halves of
// an int64), and one far enough off addresses memory outside the table.
struct IndexReader {
    const std::byte* data;
    Dtype dtype;

    std::int64_t operator()(std::size_t i) const {
        switch (dtype) {
        case Dtype::I64:
            return reinterpret_cast<const std::int64_t*>(data)[i];
        case Dtype::I32:
            return reinterpret_cast<const std::int32_t*>(data)[i];
        case Dtype::I16:
            return reinterpret_cast<const std::int16_t*>(data)[i];
        case Dtype::I8:
            return reinterpret_cast<const std::int8_t*>(data)[i];
        case Dtype::F32:
            return static_cast<std::int64_t>(reinterpret_cast<const float*>(data)[i]);
        case Dtype::F64:
            return static_cast<std::int64_t>(reinterpret_cast<const double*>(data)[i]);
        default:  // Bool; index_reader / real_index_reader admit nothing else
            return reinterpret_cast<const std::uint8_t*>(data)[i];
        }
    }
};

// A reader for ``s``, which must hold an integer (or bool) index.  Any other
// dtype is refused rather than guessed at.
inline IndexReader index_reader(const char* op, const CpuStorage& s) {
    switch (s.dtype) {
    case Dtype::I8:
    case Dtype::I16:
    case Dtype::I32:
    case Dtype::I64:
    case Dtype::Bool:
        return IndexReader{s.ptr.get(), s.dtype};
    default:
        ErrorBuilder(op).dtype_mismatch(Dtype::I64, s.dtype, "indices must be an integer tensor");
    }
}

// A reader for ``s`` that also truncates floating-point indices toward zero
// (used by ``one_hot`` and class-index loss targets).
inline IndexReader real_index_reader(const char* op, const CpuStorage& s) {
    switch (s.dtype) {
    case Dtype::I8:
    case Dtype::I16:
    case Dtype::I32:
    case Dtype::I64:
    case Dtype::Bool:
    case Dtype::F32:
    case Dtype::F64:
        return IndexReader{s.ptr.get(), s.dtype};
    default:
        ErrorBuilder(op).dtype_mismatch(Dtype::I64, s.dtype, "indices must be a real tensor");
    }
}

// ``id`` as one of ``rows`` table rows or classes, or IndexError.  Nothing
// lies below zero: a row or a class is never counted from the end.
inline std::int64_t checked_row(const char* op, std::int64_t id, std::int64_t rows) {
    if (id < 0 || id >= rows)
        refuse_row_index(op, std::to_string(id), rows);
    return id;
}

// Wrap a signed index against ``extent`` and refuse anything still outside
// ``[0, extent)``.
inline std::int64_t
checked_axis(const char* op, std::int64_t id, std::int64_t extent, const char* msg, int axis = 0) {
    const std::int64_t raw = id;
    if (id < 0)
        id += extent;
    if (id < 0 || id >= extent)
        refuse_axis_index(op, msg, std::to_string(raw), axis, extent);
    return id;
}

// The bags of an EmbeddingBag over ``n_idx`` indices: bag ``b`` is
// ``[start[b], end[b])``.  Under include_last_offset the final offset is a
// sentinel that ends the last bag, so there is one bag fewer than offsets;
// otherwise the last bag runs to the end of the indices.  The forward and
// the backward both read them here, so they cannot disagree about which
// indices a bag owns.  An offset outside [0, n_idx] would walk the index
// buffer past its end, so it is refused.
struct EmbeddingBags {
    std::vector<std::size_t> start;
    std::vector<std::size_t> end;
};

inline EmbeddingBags embedding_bags(const char* op,
                                    const CpuStorage& offsets,
                                    std::size_t n_idx,
                                    bool include_last_offset) {
    const auto read = index_reader(op, offsets);
    const std::size_t n_offsets = offsets.nbytes / dtype_size(offsets.dtype);
    const std::size_t n_bags = include_last_offset && n_offsets > 0 ? n_offsets - 1 : n_offsets;
    const auto bound = [&](std::size_t i) {
        const std::int64_t at = read(i);
        if (at < 0 || at > static_cast<std::int64_t>(n_idx))
            ErrorBuilder(op).index_error("offset " + std::to_string(at) + " is out of range [0, " +
                                         std::to_string(n_idx) + "] for the indices");
        return static_cast<std::size_t>(at);
    };
    EmbeddingBags bags;
    bags.start.reserve(n_bags);
    bags.end.reserve(n_bags);
    for (std::size_t b = 0; b < n_bags; ++b) {
        bags.start.push_back(bound(b));
        bags.end.push_back(b + 1 < n_offsets ? bound(b + 1) : n_idx);
    }
    return bags;
}

}  // namespace lucid::backend::cpu
