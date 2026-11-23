// lucid/_C/ops/linalg/_Detail.h
//
// Shared internal helpers for the engine-side linalg ops.
//
// Nothing in this header is part of the public API; it is included only by
// ``.cpp`` files under ``lucid/_C/ops/linalg/``.  Centralising these
// utilities avoids copy-paste across the dozen linalg ops (cholesky, lu,
// qr, svd, eig, eigh, solve, …) and keeps each per-op file focused on its
// forward / backward logic.
//
// Design notes
// ------------
// - All helpers are either ``inline`` free functions or type aliases, so the
//   header is pure header-only with zero linkage cost.
// - The helpers address three recurring concerns:
//     (a) GPU/MLX interop — extracting ``mlx::core::array`` from a
//         ``GpuStorage`` and wrapping the result back into a ``Storage``.
//     (b) CPU batch dispatch — computing the number of independent matrices
//         in a batched input before looping over LAPACK calls.
//     (c) Input validation — dtype and shape guards that must run *before*
//         any storage allocation so failures are cheap and obvious.
// - ``kMlxLinalgStream`` is the one architectural constant that appears in
//   every GPU linalg kernel; its rationale is documented at the definition.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/SchemaGuard.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"
#include "../../kernel/BinaryKernel.h"
#include "../utils/Layout.h"
#include "../utils/Promote.h"
#include "../utils/View.h"

namespace lucid::linalg_detail {

// Re-export common helper aliases into this namespace so callers do not have
// to spell out long ``::lucid::helpers::`` / ``::lucid::gpu::`` prefixes:
// - ``fresh()`` wraps a ``Storage`` + ``Shape`` + ``Dtype`` + ``Device`` into
//   a freshly allocated ``TensorImpl``.
// - ``mlx_shape_to_lucid()`` converts an MLX shape vector to a Lucid
//   ``Shape``.
using ::lucid::gpu::mlx_shape_to_lucid;
using ::lucid::helpers::fresh;

// MLX linalg stream — pinned to the CPU device.
//
// MLX routes its linear-algebra kernels (which are LAPACK wrappers under the
// hood) through the CPU execution queue, not the Metal GPU queue.  This means
// the helpers ``mlx::core::linalg::inv``, ``qr``, ``svd``, ``eig``, ``eigh``,
// ``solve``, ``cholesky`` all expect ``Device::cpu`` as their stream argument
// — even when their inputs live in unified GPU-accessible memory.  Passing
// the GPU device would either silently fall back or trip an assertion in
// recent MLX builds.
//
// The resulting ``mlx::core::array`` still resides in GPU-accessible memory
// and is forwarded back into Lucid's GPU storage normally; only the *stream*
// distinction is on the CPU side.
//
// All engine-side GPU linalg ops pass this constant as the ``StreamOrDevice``
// argument to the MLX call.
inline const ::mlx::core::Device kMlxLinalgStream{::mlx::core::Device::cpu};

// Extract the underlying ``mlx::core::array`` from a GPU ``TensorImpl``.
//
// Used by every GPU linalg dispatch to obtain the MLX handle backing the
// input.  The explicit device check is load-bearing: ``storage_gpu()`` on a
// non-GPU ``Storage`` would be undefined behaviour, so this guard converts
// an upstream dispatch bug into a clear engine-level error.
//
// Parameters
// ----------
// t : const TensorImplPtr&
//     A GPU tensor.
//
// Returns
// -------
// mlx::core::array
//     A handle copy referring to the same backing memory as ``t``.
//
// Raises
// ------
// LucidError
//     If ``t->device() != Device::GPU``.
inline ::mlx::core::array as_mlx_array_gpu(const TensorImplPtr& t) {
    if (t->device() != Device::GPU)
        ErrorBuilder("as_mlx_array_gpu").fail("not a GPU tensor");
    const auto& g = storage_gpu(t->storage());
    return *g.arr;
}

// Wrap an ``mlx::core::array`` produced by a linalg kernel back into a
// ``Storage`` so it can be stored inside a ``TensorImpl``.
//
// The MLX array is *moved* (not copied) into the new ``GpuStorage`` so there
// is no data duplication; the underlying buffer is shared.
//
// Parameters
// ----------
// out : mlx::core::array&&
//     Array returned by an MLX kernel.  Consumed.
// dtype : Dtype
//     The Lucid dtype the caller will tag the new ``TensorImpl`` with; must
//     agree with the element type of ``out``.  The helper does not enforce
//     this — responsibility lies with the caller.
//
// Returns
// -------
// Storage
//     A GPU ``Storage`` referring to ``out``.
inline Storage wrap_gpu_result(::mlx::core::array&& out, Dtype dtype) {
    return Storage{gpu::wrap_mlx_array(std::move(out), dtype)};
}

// Re-export ``allocate_cpu`` into this namespace.  It allocates a raw CPU
// buffer of the given shape and dtype and returns a ``CpuStorage`` backed
// by a managed ``unique_ptr``.
using ::lucid::helpers::allocate_cpu;

// Return the number of leading-batch matrices packed into a tensor of shape
// ``shape`` whose trailing ``mat_dims`` axes constitute the matrix part.
//
// Used by batched CPU linalg kernels: they loop over independent matrices,
// calling LAPACK once per slice.  The stride between consecutive matrices
// is ``shape[-2] * shape[-1]`` elements (assuming the standard contiguous
// row-major layout enforced by the dispatcher).
//
// Parameters
// ----------
// shape : const Shape&
//     The full tensor shape, including matrix axes.
// mat_dims : std::size_t
//     Number of trailing matrix dimensions — almost always ``2``.
//
// Returns
// -------
// std::int64_t
//     The product of all leading (non-matrix) dimensions.  Equals ``1`` for
//     a non-batched ``mat_dims``-dimensional input.
//
// Examples
// --------
// ``leading_batch_count({4, 3, 8, 8}, 2)`` returns ``12`` — twelve
// independent $8 \times 8$ matrices packed into a single tensor.
//
// Raises
// ------
// LucidError
//     If ``shape.size() < mat_dims``.
inline std::int64_t leading_batch_count(const Shape& shape, std::size_t mat_dims) {
    if (shape.size() < mat_dims)
        ErrorBuilder("linalg").fail("input rank too small");
    std::int64_t b = 1;
    for (std::size_t i = 0; i + mat_dims < shape.size(); ++i)
        b *= shape[i];
    return b;
}

// Reject non-float dtypes with a clear "not implemented" error.
//
// Apple Accelerate LAPACK (``sgetrf``, ``dgetrf``, ``spotrf``, ``dsyev``,
// ``ssytrf``, ``strtrs``, …) only accepts single- and double-precision real
// inputs.  Integer and half-precision dtypes are rejected here *before* any
// allocation or dispatch, so callers see an actionable error rather than a
// silent miscompute or a crash inside LAPACK.
//
// All linalg ops call this guard before dispatching to the CPU backend; the
// equivalent check on the GPU path is handled inside the MLX wrappers.
//
// Parameters
// ----------
// dt : Dtype
//     The dtype to validate.
// op : const char*
//     Symbolic op name used in the error message (e.g. ``"cholesky"``).
//
// Raises
// ------
// LucidError
//     With a not-implemented status if ``dt`` is neither ``F32`` nor
//     ``F64``.
inline void require_float(Dtype dt, const char* op) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder(op).not_implemented("only F32/F64 supported (got" +
                                         std::string(dtype_name(dt)) + ")");
}

// The dtype gate of the solvers that have no schema (cholesky, eig, ldl_factor,
// pinv, qr, svd).  An integer input is the reference's dtype guard
// (``RuntimeError``), so it is refused as a ``DtypeMismatch`` that names the cast;
// what else the kernels do not take (bool, half, complex) keeps
// :func:`require_float`'s not-implemented error.  One helper, so no solver
// carries the branch itself.
inline void require_solver_dtype(Dtype dt, const char* op) {
    if (is_integral(dt))
        throw_dtype_refusal(op, dt, DtypeRefusal::Guard, CastHint::Floating);
    require_float(dt, op);
}

// Whether the trailing matrix of ``sh`` has no entries.
//
// An empty matrix is a legal object, not a malformed one: a 0x3 matrix is
// the unique linear map from R^3 to R^0, its rank is 0, its reduced SVD
// has no singular values, and the determinant of the 0x0 matrix is the
// empty product, 1.  Every linalg op here has a defined answer for it and
// none of them used to give it.
//
// They dispatched instead, and LAPACK refused the call: its leading
// dimensions must be at least 1 even when the extent they describe is 0,
// so ``dgesdd`` on a 0x3 matrix got ``LDA = 0`` and reported argument 5
// illegal.  That surfaced two ways, both wrong.  The Fortran runtime
// prints the complaint itself, straight to file descriptor 2 —
//
//     ** On entry to DGESDD, parameter number  5 had an illegal value
//
// — which is not routed through anything Lucid can catch, and then the
// negative ``info`` became ``LucidError: LAPACK invalid argument index5``,
// which reads like a numerical failure in the caller's data and is
// actually this library calling LAPACK wrongly.
//
// The leading dimensions are clamped at their source now, so no call is
// malformed.  That is not sufficient on its own: LAPACK returns early and
// successfully for a degenerate matrix *without writing to the output
// buffers*, so the answer would have been whatever the allocation held.
// Every op that can receive one checks here and builds its result
// directly.
inline bool empty_matrix(const Shape& sh) {
    const auto rank = sh.size();
    if (rank < 2)
        return false;
    return sh[rank - 2] == 0 || sh[rank - 1] == 0;
}

// The shape a degenerate result takes, with the trailing matrix replaced.
//
// Batch dimensions are carried through unchanged; only the last two axes
// are rewritten.  ``trailing`` may name one axis (a vector of eigenvalues
// or singular values) or two (a matrix).
inline Shape with_matrix_shape(const Shape& sh, std::initializer_list<std::int64_t> trailing) {
    Shape out(sh.begin(), sh.end() - 2);
    for (auto d : trailing)
        out.push_back(d);
    return out;
}

// How a right-hand side B is read against a square A in ``solve``,
// ``lu_solve`` and ``solve_triangular`` — the one place that decides it.
//
// The backends pair matrices one to one: they take the batch count from A,
// the column count from B's last axis, and walk both buffers in lockstep.
// Anything the caller did not align first was read wrongly — a single A
// against a batch of B solved only the first batch, a B with the wrong row
// count was solved anyway (and indexed past A's batch), and a batch of
// vectors had its batch axis taken for the column count.  So the shape is
// settled here, before any backend sees it:
//
// - B is a vector right-hand side when it is 1-D.  Under
//   ``VectorRhs::OrBatched`` (``solve`` only, as the reference reads it)
//   a B of exactly ``A.shape[:-1]`` is one too — one vector per matrix.
//   ``lu_solve`` and ``solve_triangular`` take ``VectorRhs::OneD``: there
//   the reference has no batch-of-vectors reading, and such a B — e.g.
//   (3, 3) against A (3, 3, 3) — is a matrix broadcast over the batch.
//   A vector is solved as a single column and handed back without it.
// - Otherwise B must be ``(*, n, k)`` with ``n = A.shape[-1]``.
// - The batch axes of A and B broadcast against each other.
//
// Any other B is refused with ``ShapeMismatch`` here, so no LAPACK or MLX
// call ever receives operands whose extents disagree.
enum class VectorRhs { OneD, OrBatched };

struct SolveRhs {
    bool vector = false;
    Shape a_shape;    // batch + (n, n): A as the backend receives it
    Shape b_shape;    // batch + (n, k): B as the backend receives it
    Shape out_shape;  // the solution's shape as returned to the caller
};

inline SolveRhs
solve_rhs_contract(const Shape& a, const Shape& b, VectorRhs vectors, const char* op) {
    const std::size_t ra = a.size();
    const std::int64_t n = a[ra - 1];
    SolveRhs rhs;
    const bool one_per_matrix = vectors == VectorRhs::OrBatched && b.size() + 1 == ra &&
                                std::equal(b.begin(), b.end(), a.begin());
    rhs.vector = b.size() == 1 || one_per_matrix;
    Shape b_mat = b;
    if (rhs.vector)
        b_mat.push_back(1);
    if (b_mat.size() < 2 || b_mat[b_mat.size() - 2] != n)
        throw ShapeMismatch(Shape{n, b.empty() ? 1 : b.back()}, b,
                            std::string(op) + ": B must be (*, n, k) or a vector right-hand side, "
                                              "with n = A.shape[-1]");
    const Shape a_batch(a.begin(), a.end() - 2);
    const Shape b_batch(b_mat.begin(), b_mat.end() - 2);
    auto batch = ::lucid::detail::try_broadcast_shapes(a_batch, b_batch);
    if (batch.is_err())
        throw ShapeMismatch(a_batch, b_batch,
                            std::string(op) + ": batch dimensions of A and B do not broadcast");
    rhs.a_shape = batch.value();
    rhs.a_shape.push_back(n);
    rhs.a_shape.push_back(n);
    rhs.b_shape = batch.value();
    rhs.b_shape.push_back(n);
    rhs.b_shape.push_back(b_mat.back());
    rhs.out_shape = rhs.b_shape;
    if (rhs.vector)
        rhs.out_shape.pop_back();
    return rhs;
}

// ``t`` broadcast to ``shape`` — a no-op when it already has it.  The
// broadcast records its own (reducing) backward, so a solve node built on
// the aligned operands never has to know the caller broadcast at all.
inline TensorImplPtr align_to(const TensorImplPtr& t, const Shape& shape) {
    return t->shape() == shape ? t : expand_op(t, shape);
}

// B as the ``(batch, n, k)`` matrix the backend receives.
inline TensorImplPtr align_rhs(const TensorImplPtr& b, const SolveRhs& rhs) {
    return align_to(rhs.vector ? unsqueeze_op(b, -1) : b, rhs.b_shape);
}

// The aligned solution in the shape the caller's B asked for.
inline TensorImplPtr restore_rhs(const TensorImplPtr& x, const SolveRhs& rhs) {
    return rhs.vector ? squeeze_op(x, -1) : x;
}

}  // namespace lucid::linalg_detail
