// lucid/_C/core/ResultExtent.h
//
// A GPU op result holds exactly the elements its shape names.
//
// A result tensor carries two facts that come from different places: the shape
// the op computed, and the array the backend returned.  Nothing compared them,
// so an array of another extent became a tensor whose storage disagreed with its
// shape.  A shapeless-compiled ``maximum`` of ``(0,)`` and ``(1,)`` returned one
// element under the label ``(0,)`` (LCD-1062, see ``ShapelessFused.h``): an
// in-place form failed much later in ``copy_from`` with ``nbytes mismatch (GPU)``
// or left its receiver on the wrong buffer, and the out-of-place form carried the
// wrong buffer without a word.
//
// The check is a byte-count comparison on what the storage already records, with
// no evaluation and no allocation.  It is made in :func:`helpers::fresh` — the one
// function an op's forward builds its result tensor through, whether it is a
// kernel family's (``UnaryKernel``, ``BinaryKernel``, ``ReduceKernel``) or an op's
// own (bitwise, compare, linalg, nn, a generator) — so a forward that builds its
// result by hand is held to the same count and no op remembers to ask for it
// (LCD-1072).  It is not made in the :class:`TensorImpl` constructor: a view is
// built by the same constructor over its parent's buffer and holds more or fewer
// bytes than its shape on purpose, so the constructor cannot tell a view from a
// result that went wrong.
#pragma once

#include <cstddef>
#include <variant>

#include "Dtype.h"
#include "Shape.h"
#include "Storage.h"

namespace lucid {

// Raises :class:`LucidError` naming the innermost op on the error-context stack,
// the shape and dtype, and both byte counts.  Defined out of line in
// ``ErrorBuilder.cpp``, where the stack lives, like :func:`throw_negative_dim`;
// not exported: only the engine itself calls it.
[[noreturn]] void refuse_result_extent(std::size_t held, const Shape& shape, Dtype dtype);

// Refuses a GPU ``storage`` whose byte count is not ``shape``'s element count
// times ``dtype``'s size.  CPU and phantom storage are left alone: a CPU buffer
// is sized by the shape at allocation, and a phantom has no buffer to compare.
inline void require_result_extent(const Storage& storage, const Shape& shape, Dtype dtype) {
    const auto* gpu = std::get_if<GpuStorage>(&storage);
    if (gpu == nullptr)
        return;
    if (gpu->nbytes != shape_numel(shape) * dtype_size(dtype)) [[unlikely]]
        refuse_result_extent(gpu->nbytes, shape, dtype);
}

}  // namespace lucid
