// lucid/_C/core/PointerOffset.h
//
// ``base + offset`` for a pointer that may be null, which is how an empty tensor's
// buffer reads.
//
// ``allocate_aligned_bytes(0)`` answers a null pointer, so every CPU tensor with no
// elements has a null buffer, and an empty *view* of it still carries the offset its
// slice or ``as_strided`` gave it.  ``nullptr + offset`` is undefined behaviour for any
// non-zero offset (UBSan: "applying non-zero offset to null pointer") even when nothing
// is read through the result.  A reduction over a zero-length axis forms exactly that:
// it offsets its input by the index of an axis that is not empty, then hands the pointer
// to a loop that runs zero times.  The nightly sanitizer job halted on it in the p-norm
// kernel and in ``make_numpy_view`` (LCD-1047).
//
// One rule, kept here once: a null base stays null.  Every other pointer moves exactly
// as ``base + offset`` does, so a non-null empty view keeps the address its bookkeeping
// (``offset_in_allocation``) expects.  Code that adds an offset the tensor's extents do
// not scale (a column, a slice index, a view's storage offset) goes through
// :func:`offset_or_null`; an offset that is a product with the extent (``o * len *
// inner`` with ``len == 0``) is zero already and needs no help.  An optional operand a
// backward kernel was told not to write is a null pointer for the same reason and is
// offset the same way (``backend/GradMask.h``).

#pragma once

#include <cstddef>

namespace lucid {

// ``base + offset``, or ``nullptr`` when ``base`` is.
//
// Parameters
// ----------
// base : T*
//     A buffer's first element, null for an empty tensor or an unwanted gradient.
// offset : integral
//     Elements (bytes, for a ``std::byte*``) from ``base``; negative moves back.
template <typename T, typename Offset>
[[nodiscard]] constexpr T* offset_or_null(T* base, Offset offset) noexcept {
    return base != nullptr ? base + static_cast<std::ptrdiff_t>(offset) : nullptr;
}

}  // namespace lucid
