// lucid/_C/core/IndexRefusal.h
//
// The words of an out-of-range index refusal, said once.
//
// A CPU kernel raises IndexError for an index outside its axis.  On Metal
// that same refusal is opt-in (``LUCID_STRICT_INDEX``, see
// ``backend/gpu/AxisIndex.h``) and must read the same on both devices, so
// both backends build the message here instead of each keeping a copy.
#pragma once

#include <cstdint>
#include <string>

#include "ErrorBuilder.h"

namespace lucid {

// ``<what>: index <raw> is out of bounds for dimension <axis> with size
// <extent>``, as a gather or scatter along ``axis`` names it.  ``raw`` is the
// index as the caller wrote it, before a negative one was wrapped.
[[noreturn]] inline void refuse_axis_index(
    const char* op, const char* what, const std::string& raw, int axis, std::int64_t extent) {
    ErrorBuilder(op).index_error(std::string(what) + ": index " + raw +
                                 " is out of bounds for dimension " + std::to_string(axis) +
                                 " with size " + std::to_string(extent));
}

// ``index <id> is out of range [0, <rows>)``, as a table row or a class
// names it.
[[noreturn]] inline void
refuse_row_index(const char* op, const std::string& id, std::int64_t rows) {
    ErrorBuilder(op).index_error("index " + id + " is out of range [0, " + std::to_string(rows) +
                                 ")");
}

}  // namespace lucid
