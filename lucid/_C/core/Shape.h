// lucid/_C/core/Shape.h
//
// Type aliases and utility functions describing a tensor's geometric layout:
// the dimension vector :type:`Shape` and the byte-stride vector :type:`Stride`.
//
// Lucid follows the row-major (C-contiguous) convention used by NumPy and the
// reference framework — innermost dimension varies fastest in memory, and
// :func:`contiguous_stride` produces the canonical stride for that layout.
// Strides are stored as **byte** offsets (not element counts), so kernels can
// move through memory without a dtype lookup at every step.
//
// See Also
// --------
// :class:`Dtype`      — element size used by :func:`contiguous_stride`.
// :class:`TensorMeta` — owns the per-tensor :type:`Shape` and :type:`Stride`.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace lucid {

// Sequence of dimension sizes describing a tensor's shape.
//
// Dimensions are signed ``int64`` so that an op's *argument* can carry a
// wildcard — reshape's inferred ``-1``, expand's keep-this-size ``-1`` — until
// the op resolves it.  A tensor's shape is always concrete: every extent is
// ``>= 0``, and :func:`shape_numel` / :func:`contiguous_stride` refuse any
// other (the tensor constructor builds its stride through the latter).
// Free-form construction via braced-init is supported and idiomatic:
//
// Examples
// --------
// ::
//
//     Shape s{B, C, H, W};                    // 4-D image batch
//     Shape scalar_shape;                     // empty vector → scalar (numel=1)
//
// See Also
// --------
// :func:`shape_numel`        — total element count.
// :func:`contiguous_stride`  — canonical row-major byte stride.
using Shape = std::vector<std::int64_t>;

// Sequence of per-axis byte strides.
//
// ``stride[i]`` is the **number of bytes** to advance in the underlying
// :class:`Storage` buffer to move one element along axis ``i``.  Storing the
// stride in bytes rather than element counts lets the same vector type cover
// every dtype without per-call multiplications inside hot loops.
//
// A tensor is *contiguous* when its stride vector equals
// :func:`contiguous_stride` for its shape and element size.
using Stride = std::vector<std::int64_t>;

// Refuses a shape with a negative extent.
//
// Raises :class:`InvalidArgument` naming the innermost op on the error-context
// stack and the first negative extent, in the reference framework's words:
// ``zeros: Trying to create tensor with negative dimension -3: [2, -3]``.
// Defined out of line, in ``ErrorBuilder.cpp``: the op name lives on
// :class:`ErrorContext`, whose header includes this one.  Not exported: only
// the engine itself calls it.
[[noreturn]] void throw_negative_dim(const Shape& shape);

// The most elements a tensor may hold: ``(2^63 - 1) / 16``, about 5.8e17.
//
// At the widest element (16 bytes, ``complex128``) that many still fit the
// signed 64-bit byte offsets a :type:`Stride` is kept in, so every
// ``numel * dtype_size`` the engine computes — and it computes one at every
// allocation — is exact.  Nothing near the limit can be allocated: at one
// byte an element it is 512 PiB.
inline constexpr std::size_t kMaxNumel =
    static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) / 16;

// Refuses a shape whose element count — or, for :func:`contiguous_stride`,
// whose byte stride — overflows, the way :func:`throw_negative_dim` refuses a
// negative extent: :class:`InvalidArgument` naming the innermost op, in the
// reference framework's words (``Storage size calculation overflowed with
// sizes=[...]``).  Defined out of line in ``ErrorBuilder.cpp``.
[[noreturn]] void throw_numel_overflow(const Shape& shape);

// Refuses an extent computed as ``a * b`` that does not fit in ``int64``.
// Same class and wording as :func:`throw_numel_overflow`; out of line in
// ``ErrorBuilder.cpp``.
[[noreturn]] void throw_extent_overflow(std::int64_t a, std::int64_t b);

// ``a * b`` for an output extent made from two others — a repeated or tiled
// axis, ``reshape``'s product of the known extents.  Unchecked, a product
// past ``int64`` wrapped into a plausible extent (or a negative one, refused
// as a negative dimension the caller never passed).
inline std::int64_t mul_extent(std::int64_t a, std::int64_t b) {
    std::int64_t r = 0;
    if (__builtin_mul_overflow(a, b, &r)) [[unlikely]]
        throw_extent_overflow(a, b);
    return r;
}

// Bytes for ``numel`` elements of ``elem_size`` bytes, for a buffer sized by
// a count rather than a shape (a bin count, a class count).  The count is
// held to :data:`kMaxNumel` as :func:`shape_numel` holds a shape's, so the
// product is exact; past it, ``2^62`` bins of ``float64`` wrapped to a
// zero-byte buffer that the first count was written through.
inline std::size_t count_nbytes(std::size_t numel, std::size_t elem_size) {
    if (numel > kMaxNumel) [[unlikely]]
        throw_numel_overflow(Shape{static_cast<std::int64_t>(
            std::min<std::size_t>(numel, std::numeric_limits<std::int64_t>::max()))});
    return numel * elem_size;
}

// Returns the total number of elements implied by a shape — the one
// element-count rule; :func:`TensorMeta::numel` and every allocation sized
// by a shape go through it.
//
// The product is computed in :type:`std::size_t` so the result can size an
// allocation directly.
//
// Parameters
// ----------
// shape : Shape
//     A concrete shape.  May be empty (scalar).
//
// Returns
// -------
// std::size_t
//     Product of all dimensions.  ``1`` for an empty shape (scalar
//     convention).
//
// Raises
// ------
// InvalidArgument
//     A dimension is negative (:func:`throw_negative_dim`).  It has no
//     count: as a ``size_t`` it is ``2^64 - 1``, and a kernel that walks
//     that count runs off its buffer.
// InvalidArgument
//     The product exceeds :data:`kMaxNumel` (:func:`throw_numel_overflow`).
//     Unchecked, ``{2^33, 2^33}`` wrapped to a count of 0: the tensor was
//     built with no buffer and the first kernel to read it segfaulted.
//
// Examples
// --------
// ::
//
//     shape_numel({})            // 1   (scalar)
//     shape_numel({3, 4})        // 12
//     shape_numel({-1, 8})       // throws InvalidArgument
//     shape_numel({1LL << 33, 1LL << 33})   // throws InvalidArgument
//     shape_numel({1LL << 62, 1LL << 62, 0})  // 0: an empty extent
inline std::size_t shape_numel(const Shape& shape) {
    std::size_t n = 1;
    bool wrapped = false;
    for (auto d : shape) {
        if (d < 0)
            throw_negative_dim(shape);
        wrapped |= __builtin_mul_overflow(n, static_cast<std::size_t>(d), &n);
    }
    if (wrapped || n > kMaxNumel) [[unlikely]] {
        // Any product past the limit with a zero extent in it is zero.
        if (std::find(shape.begin(), shape.end(), std::int64_t{0}) != shape.end())
            return 0;
        throw_numel_overflow(shape);
    }
    return n;
}

// Refuses ``shape`` unless every extent is ``>= 0`` and the element count is
// at most :data:`kMaxNumel`.
//
// For an op that must turn a shape away before it dispatches: the backend
// sizes its work by the shape it is handed, before any tensor is built.
inline void check_concrete_shape(const Shape& shape) {
    static_cast<void>(shape_numel(shape));
}

// Computes the row-major (C-contiguous) byte-stride vector for a tensor with
// the given shape and element size.
//
// The stride is built back-to-front: the innermost (last) axis has stride
// ``elem_size``, and each outer axis's stride is the product of the inner
// stride and the inner dimension size.  This matches NumPy's default
// ``'C'`` order and is the layout expected by Apple Accelerate BLAS / vDSP
// routines.
//
// Parameters
// ----------
// shape : Shape
//     A concrete shape.  Empty shape yields an empty stride.
// elem_size : std::size_t
//     Element size in bytes — typically obtained from :func:`dtype_size`.
//
// Returns
// -------
// Stride
//     Vector of the same length as ``shape``.  When the result is fed back
//     into :class:`TensorMeta`, :func:`TensorImpl::is_contiguous` returns
//     ``true``.
//
// Raises
// ------
// InvalidArgument
//     A dimension is negative (:func:`throw_negative_dim`).  The tensor
//     constructor derives its stride here, so no tensor is built with one.
// InvalidArgument
//     A stride does not fit in ``int64`` (:func:`throw_numel_overflow`).
//     Below :data:`kMaxNumel` elements none can overflow; an empty shape
//     can, ``{0, 2^62, 4}``, since its zero extent spans no bytes but its
//     outer stride still has to be written down.
//
// Math
// ----
// $$\text{stride}[i] = \text{elem\_size} \cdot \prod_{j > i} \text{shape}[j]$$
//
// Examples
// --------
// ::
//
//     contiguous_stride({2, 3, 4}, 4)    // {48, 16, 4}  (float32)
//     contiguous_stride({}, 4)           // {}           (scalar)
//
// See Also
// --------
// :func:`TensorMeta::is_contiguous_for`
//     Reverse check — verifies a stored stride against the canonical one.
inline Stride contiguous_stride(const Shape& shape, std::size_t elem_size) {
    Stride s(shape.size());
    if (shape.empty())
        return s;
    std::int64_t acc = static_cast<std::int64_t>(elem_size);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i) {
        const std::int64_t d = shape[static_cast<std::size_t>(i)];
        if (d < 0)
            throw_negative_dim(shape);
        s[static_cast<std::size_t>(i)] = acc;
        // The outermost product is no stride, so it is not taken.
        if (i > 0 && __builtin_mul_overflow(acc, std::max<std::int64_t>(d, 1), &acc)) [[unlikely]]
            throw_numel_overflow(shape);
    }
    return s;
}

}  // namespace lucid
