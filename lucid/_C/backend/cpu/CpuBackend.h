// lucid/_C/backend/cpu/CpuBackend.h
//
// Concrete :class:`IBackend` implementation for Apple Silicon CPU using
// the Apple Accelerate framework.
//
// All compute runs on ``CpuStorage`` (host-side aligned allocations);
// no raw scalar loops are used where a ``vDSP`` or ``vForce`` vector
// intrinsic exists.  Per DEVELOPMENT.md H3 this is the *only* compute
// path allowed on the CPU stream — no NumPy / SciPy / external library
// fallbacks.
//
// Notes
// -----
// **Elementwise ops** (add, sub, mul, exp, …) delegate to ``cpu::vadd_f32``
// and friends (vDSP / vForce wrappers) through the private
// ``unary_op`` / ``binary_op`` template helpers, which dispatch on the
// runtime :class:`Dtype`.
//
// **Axis reductions** (``reduce_sum``, ``reduce_mean``, ``reduce_max``,
// ``reduce_min``) use the private ``reduce_axes`` helper, which iterates
// axes from highest to lowest (sorted descending) and calls
// ``cpu::sum_axis_f32`` etc. from :file:`Reduce.h`.  For mean reduction
// the result is scaled by $1 / d$ after the sum.
//
// **Convolution** uses an im2col + GEMM strategy: for each batch element
// the input patches are unrolled into a column matrix by
// ``conv_nd_im2col_f32`` / ``_f64``, then ``cblas_sgemm`` / ``dgemm``
// computes the full output in a single call.  The backward pass uses
// col2im for the input gradient and a second GEMM for the weight gradient.
//
// **Normalization** (BatchNorm, LayerNorm, GroupNorm, RMSNorm) delegates
// to the :file:`Norm.h` wrappers which use per-row vDSP operations for
// F32 throughput.
//
// **Linear algebra** (eig, eigh, svd, qr, chol, inv, solve) delegates to
// the :file:`Lapack.h` wrappers which handle the row-major ↔
// column-major conversion required by Accelerate's LAPACK interface.
//
// **Pooling** (MaxPool, AvgPool) delegates to :file:`Pool.h` for 1-D,
// 2-D, and 3-D.
//
// **Private helpers.**  ``fill_ones(ptr, n, dt)`` fills a raw buffer
// with 1.  ``unary_op(a, shape, dt, …)`` / ``binary_op(a, b, shape, dt,
// …)`` allocate the output and dispatch by ``Dtype`` (F32 / F64 / I32 /
// I64).  ``reduce_axes(a, in_shape, opts, dt, op)`` runs a single-axis
// reduction loop.  ``cast_impl(src, dst, n, src_dt, dst_dt)`` performs
// an element-by-element type cast.
//
// **Method family units.**  Non-template ``CpuBackend`` member definitions
// live out-of-line in ``lucid/_C/backend/cpu/methods/<Family>.cpp`` across 14
// symmetric op-family translation units:
//   • ``Factory.cpp``    — allocation, conversion, random, cloning, shared storage.
//   • ``Binary.cpp``     — elementwise binary arithmetic, comparisons, bitwise.
//   • ``Unary.cpp``      — elementwise unary math, special functions, rounding, clip.
//   • ``Activation.cpp`` — neural activations and softmax / log-softmax.
//   • ``Reduction.cpp``  — axis reductions, cumulative scans, sort / topk, histogram.
//   • ``Layout.cpp``     — reshape, permute, slice, pad, roll, flip, tile, repeat.
//   • ``Indexing.cpp``   — gather, scatter, masked ops, where, embedding.
//   • ``Linalg.cpp``     — Cholesky, QR, SVD, eig/eigh, solve, inv, matrix_exp, FFT.
//   • ``Matmul.cpp``     — matmul, bmm, dot, outer, inner, tensordot, linear.
//   • ``Norm.cpp``       — LayerNorm, BatchNorm, GroupNorm, InstanceNorm, RMSNorm.
//   • ``Loss.cpp``       — MSE, L1, Huber, BCE, NLL, CrossEntropy, KL, CTC, distance.
//   • ``Conv.cpp``       — ConvNd, ConvTransposeNd, im2col/col2im, fold/unfold.
//   • ``Spatial.cpp``    — MaxPoolNd, AvgPoolNd, AdaptivePool, interpolate, grid_sample.
//   • ``Sequence.cpp``   — RNN, LSTM, GRU, scaled_dot_product_attention.
//
// **Self-registration.**  ``BackendInit.cpp`` defines ``CpuBackendRegistrar``
// to register a :class:`CpuBackend` singleton with :class:`Dispatcher` at
// static-init time.
//
// See Also
// --------
// :class:`IBackend` — abstract interface fulfilled by this class.
// :class:`Dispatcher` — routes ``Device::CPU`` tensors here.
// :class:`GpuBackend` — MLX/Metal sibling backend.

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/BroadcastPlan.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Half.h"
#include "../../core/Shape.h"
#include "../../core/StridedCopy.h"
#include "../Dispatcher.h"
#include "../IBackend.h"
#include "../LapackInfo.h"
#include "BatchNormEval.h"
#include "Blas.h"
#include "CtcLoss.h"
#include "DivMod.h"
#include "DtypeDoors.h"
#include "ErfPoly.h"
#include "GeluGrad.h"
#include "GlobalResponseNorm.h"
#include "GridSample.h"
#include "Im2Col.h"
#include "IndexBounds.h"
#include "IntPow.h"
#include "Interpolate.h"
#include "Lapack.h"
#include "LinalgFactor.h"
#include "Lstm.h"
#include "NonFinite.h"
#include "Norm.h"
#include "Parallel.h"
#include "Pool.h"
#include "Reduce.h"
#include "ScatterReduce.h"
#include "SdpaMask.h"
#include "Shape.h"
#include "Softmax.h"
#include "SortSelect.h"
#include "Vdsp.h"
#include "Vforce.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace lucid {
namespace backend {

// CPU (Apple Accelerate-backed) concrete :class:`IBackend`.
//
// Every public method satisfies the :class:`IBackend` contract; the
// private section holds low-level type-dispatch helpers (``unary_op``,
// ``binary_op``, ``reduce_axes``) plus per-op compute routines that
// delegate to the Accelerate helpers in :file:`Blas.h`,
// :file:`Lapack.h`, :file:`Norm.h`, :file:`Pool.h`, :file:`Reduce.h`,
// :file:`Vdsp.h`, and :file:`Vforce.h`.
//
// Notes
// -----
// Marked ``final`` — there is exactly one CPU backend at runtime,
// installed by :class:`CpuBackendRegistrar` at static-init time.
//
// See Also
// --------
// :class:`IBackend` — abstract base.
// :class:`GpuBackend` — MLX/Metal counterpart.
// :class:`Dispatcher` — owns the singleton instance.
class CpuBackend final : public IBackend {
public:
    // Registers this backend with the :class:`Dispatcher` for ``Device::CPU``.
    //
    // Invoked exactly once at process start by ``BackendInit.cpp``.
    //
    // See Also
    // --------
    // :func:`Dispatcher::register_backend` — receiving end.
    static void register_self() {
        Dispatcher::register_backend(Device::CPU, std::make_unique<CpuBackend>());
    }

    // Returns ``Device::CPU``.
    //
    // Returns
    // -------
    // Device
    //     The device tag this backend services.
    Device device() const noexcept override { return Device::CPU; }

    // Wraps an incoming ``CpuStorage`` into the variant directly.
    //
    // The CPU backend already operates on ``CpuStorage`` natively, so
    // this is a move with no copy.  The ``shape`` argument is ignored.
    //
    // Parameters
    // ----------
    // cpu : CpuStorage
    //     Source host buffer (consumed).
    //
    // Returns
    // -------
    // Storage
    //     ``cpu`` re-wrapped as a :class:`Storage` variant.
    Storage from_cpu(CpuStorage cpu, const Shape&) override;

    Storage zeros(const Shape& shape, Dtype dt) override {
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        std::memset(ptr.get(), 0, nb);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage ones(const Shape& shape, Dtype dt) override;

    Storage clone(const Storage& src, const Shape& shape, Dtype dt) override;

    Storage contiguous(const Storage& src,
                       const Shape& shape,
                       const Stride& stride,
                       std::size_t storage_offset,
                       bool already_contiguous,
                       Dtype dt) override;

    // An interleaved complex op over two same-shape buffers.  ``binary_op``
    // has no complex branch: its kernels are typed on one real scalar.
    template <typename Scalar>
    static Storage complex_binary(const Storage& a,
                                  const Storage& b,
                                  const Shape& shape,
                                  void (*fn)(const Scalar*, const Scalar*, Scalar*, std::size_t));

    Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage bitwise_binary(
        const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) override;

    Storage compare_binary(
        const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) override;

    Storage maximum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage minimum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage exp(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage log(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage sqrt(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage rsqrt(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage abs(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage neg(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage sign(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage floor(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage ceil(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage round(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage sin(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage cos(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage tanh(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage sigmoid(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage relu(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage log2(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage erf(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage erfinv(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage reciprocal(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage square(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage cube(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage cube_root(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage tan(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage asin(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage acos(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage atan(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage sinh(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage cosh(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage invert(const Storage& a, const Shape& shape, Dtype dt) override;

    // ── Complex viewing (interleaved [re, im, re, im, ...] storage) ────────

    // Project one lane out of an interleaved complex buffer.
    //
    // ``lane`` is 0 for the real half and 1 for the imaginary one.  The
    // lane width follows the storage's own dtype rather than being
    // assumed float32: reading a complex128 buffer as pairs of floats
    // would report the top half of a double's mantissa as a number.
    static Storage project_lane(const Storage& a, const Shape& shape, std::size_t lane);

    Storage complex_real(const Storage& a, const Shape& shape) override {
        return project_lane(a, shape, 0);
    }

    Storage complex_imag(const Storage& a, const Shape& shape) override {
        return project_lane(a, shape, 1);
    }

    Storage complex_combine(const Storage& re, const Storage& im, const Shape& shape) override;

    Storage complex_conj(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage silu(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage gelu(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage
    gelu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) override;

    Storage
    silu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) override;

    Storage gelu_exact(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage gelu_exact_backward(const Storage& a,
                                const Storage& grad,
                                const Shape& shape,
                                Dtype dt) override;

    Storage leaky_relu(const Storage& a, const Shape& shape, Dtype dt, double slope) override;

    Storage softplus(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage elu(const Storage& a, const Shape& shape, Dtype dt, double alpha) override;

    Storage elu_backward(
        const Storage& a, const Storage& grad, const Shape& shape, Dtype dt, double alpha) override;

    Storage selu(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage
    selu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) override;

    Storage mish(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage
    mish_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) override;

    Storage hard_sigmoid(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage hard_sigmoid_backward(const Storage& a,
                                  const Storage& grad,
                                  const Shape& shape,
                                  Dtype dt) override;

    Storage hard_swish(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage hard_swish_backward(const Storage& a,
                                const Storage& grad,
                                const Shape& shape,
                                Dtype dt) override;

    Storage relu6(const Storage& a, const Shape& shape, Dtype dt) override;

    // isinf / isnan / isfinite share one body.  The dtype picks the bit
    // layout and the question is a template argument (see NonFinite.h).
    // Only float32 and float64 used to be tested.  Every other dtype got the
    // integer answer, so a float16 or bfloat16 infinity read as finite.  An
    // integer or bool tensor really does hold no NaN and no infinity, so
    // that answer is kept for those dtypes only.
    template <cpu::NonFiniteProbe P>
    static Storage nonfinite_probe(const Storage& a, const Shape& shape, Dtype dt);

    Storage isinf(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage isnan(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage isfinite(const Storage& a, const Shape& shape, Dtype dt) override;

    // Truth of element ``i`` in its own dtype.  ``any`` / ``all`` handled
    // F32 / F64 / I32 / Bool and sent every other dtype to a byte loop that
    // read one *byte* per element: an int64 1 is the bytes 01 00 00 …, so
    // ``all([[1, 2], [3, 4]])`` saw the zero second byte and answered False,
    // and ``any`` looked only at the first n bytes — the first n/8 int64s.
    // Floating zero includes -0.0, so the half types mask off the sign bit.
    static bool element_nonzero(const std::byte* base, std::size_t i, Dtype dt);

    Storage any(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage all(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage nan_to_num(const Storage& a,
                       const Shape& shape,
                       Dtype dt,
                       double nan_val,
                       double posinf_val,
                       double neginf_val) override;

    Storage
    reduce_sum(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override;

    Storage
    reduce_mean(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override;

    Storage
    variance(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override;

    Storage
    reduce_max(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override;

    Storage
    reduce_min(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override;

    Storage cumsum(const Storage& a, const Shape& shape, int axis, Dtype dt) override;

    Storage cumprod(const Storage& a, const Shape& shape, int axis, Dtype dt) override;

    Storage cummax(const Storage& a, const Shape& shape, int axis, Dtype dt) override;

    Storage cummin(const Storage& a, const Shape& shape, int axis, Dtype dt) override;

    Storage softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) override;

    Storage softmax_backward(
        const Storage& z, const Storage& grad_out, const Shape& shape, int axis, Dtype dt) override;

    Storage log_softmax_backward(
        const Storage& y, const Storage& grad_out, const Shape& shape, int axis, Dtype dt) override;

    Storage log_softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) override;

    Storage reverse_along_axis(const Storage& a, const Shape& shape, int axis, Dtype dt) override;

    Storage trace(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage trace_backward(const Storage& grad_out, const Shape& input_shape, Dtype dt) override;

    std::vector<Storage> meshgrid(const std::vector<Storage>& xs,
                                  const Shape& out_shape,
                                  Dtype dt,
                                  bool indexing_xy) override;

    Storage where_branch(const Storage& grad,
                         const Storage& cond,
                         const Shape& shape,
                         Dtype dt,
                         bool true_branch) override;

    Storage masked_fill(
        const Storage& a, const Storage& mask, const Shape& shape, Dtype dt, double value) override;

    Storage gather(const Storage& a,
                   const Storage& indices,
                   const Shape& input_shape,
                   const Shape& output_shape,
                   int axis,
                   Dtype index_dtype,
                   Dtype dt) override;

    Storage gather_backward(const Storage& grad,
                            const Storage& indices,
                            const Shape& input_shape,
                            const Shape& output_shape,
                            int axis,
                            Dtype index_dtype,
                            Dtype dt) override;

    Storage diagonal(const Storage& a,
                     const Shape& input_shape,
                     int offset,
                     int axis1,
                     int axis2,
                     Dtype dt) override;

    Storage diagonal_backward(const Storage& grad,
                              const Shape& input_shape,
                              const Shape& output_shape,
                              int offset,
                              int axis1,
                              int axis2,
                              Dtype dt) override;

    Storage roll(const Storage& a,
                 const Shape& shape,
                 Dtype dt,
                 const std::vector<std::int64_t>& shifts,
                 const std::vector<int>& axes) override;

    Storage reshape(const Storage& a, const Shape&, const Shape&, Dtype dt) override;

    Storage slice_axis(const Storage& a,
                       const Shape& src_shape,
                       const Shape& slice_shape,
                       int axis,
                       std::int64_t offset,
                       Dtype dt) override;

    Storage insert_axis_slice(const Storage& a,
                              const Shape& src_shape,
                              const Shape& dst_shape,
                              int axis,
                              std::int64_t offset,
                              Dtype dt) override;

    Storage concatenate(const std::vector<Storage>& xs,
                        const std::vector<Shape>& shapes,
                        int axis,
                        Dtype dt) override;

    Storage
    stack(const std::vector<Storage>& xs, const Shape& input_shape, int axis, Dtype dt) override;

    std::vector<Storage> split_equal(
        const Storage& a, const Shape& shape, int axis, std::int64_t num_splits, Dtype dt) override;

    std::vector<Storage> split_at(const Storage& a,
                                  const Shape& shape,
                                  int axis,
                                  const std::vector<std::int64_t>& indices,
                                  Dtype dt) override;

    Storage repeat_backward(const Storage& grad_out,
                            const Shape& input_shape,
                            const Shape& output_shape,
                            int axis,
                            std::int64_t repeats,
                            Dtype dt) override;

    Storage tile_backward(const Storage& grad_out,
                          const Shape& input_shape,
                          const Shape& padded_shape,
                          const Shape& output_shape,
                          const std::vector<std::int64_t>& p4_,
                          Dtype dt) override;

    std::pair<Storage, Storage> sort_select_impl(const Storage& a,
                                                 const Shape& input_shape,
                                                 const Shape& output_shape,
                                                 int axis,
                                                 Dtype dt,
                                                 bool descending,
                                                 bool need_values);

    std::pair<Storage, Storage> sort_select(const Storage& a,
                                            const Shape& input_shape,
                                            const Shape& output_shape,
                                            int axis,
                                            Dtype dt,
                                            bool descending) override;

    Storage argsort(const Storage& a, const Shape& shape, int axis, Dtype dt) override;

    Storage arg_reduce_index(const Storage& a,
                             const Shape& shape,
                             int axis,
                             bool keepdims,
                             Dtype dt,
                             bool is_min) override;

    Storage scatter_add_axis(const Storage& grad,
                             const Storage& indices,
                             const Shape& output_shape,
                             const Shape& grad_shape,
                             int axis,
                             Dtype dt) override;

    // User-facing scatter-add: copies base then adds src at positions given by indices.
    Storage scatter_add(const Storage& base,
                        const Storage& indices,
                        const Storage& src,
                        const Shape& base_shape,
                        const Shape& idx_shape,
                        int dim,
                        Dtype dt) override;

private:
    // Where an axis scatter's writes land in ``base``.
    //
    // The walk is over the *index's* extents, which may be shorter than
    // base's on any axis but ``dim`` (src arrives cut to the index's shape),
    // so each outer (before ``dim``) and inner (after) coordinate of the
    // index is placed by base's strides.  When the shapes agree this is
    // ``(o * base_dim + tgt) * inner + j`` exactly.  Index and src are dense
    // in the index's shape: element ``(o, k, j)`` is at
    // ``(o * idx_dim + k) * inner + j``.
    struct AxisScatterWalk {
        std::vector<std::size_t> outer_off;  // base offset of each outer coordinate
        std::vector<std::size_t> inner_off;  // base offset of each inner coordinate
        std::size_t idx_dim = 0;             // index extent along ``dim``
        std::size_t base_dim = 0;            // base extent along ``dim``
        std::size_t dim_stride = 0;          // base stride of ``dim``, in elements
    };

    static AxisScatterWalk
    axis_scatter_walk(const Shape& base_shape, const Shape& idx_shape, int dim);

    // The overwrite scatter's loop, for ``W``-byte elements and index type
    // ``I``.  It never does arithmetic on a value, so it moves bytes: one
    // loop per element width serves every dtype of that width.  The
    // ``memcpy`` of a constant ``W`` compiles to a single load and store.
    //
    // ``k`` ascends for every (outer, inner) line, so a position named twice
    // keeps the value written last, as the reference's CPU kernel does.
    // ``j`` innermost keeps the src reads and the base writes sequential.
    template <std::size_t W>
    static void scatter_set_lines(std::byte* dst,
                                  const std::byte* src,
                                  cpu::IndexReader read_idx,
                                  const AxisScatterWalk& w);

    // Generic scatter-reduce loop shared by scatter_amax / scatter_amin / scatter_prod.
    // Every dtype at its own width but half, which widens to float32 here;
    // see ScatterReduce.h.
    //
    // The index is read at its own width and checked against the axis.  This
    // loop read every index as int32 lanes (an int64 index of 2^32 + 1 was 1)
    // and wrote wherever the value pointed, so an out-of-range index wrote
    // past the buffer.  It also placed the index's elements by the base's
    // strides, so an index narrower than the base off ``dim`` read past the
    // index and src and took the process down with SIGBUS.
    // ``axis_scatter_walk`` places them now, as it does for scatter_add and
    // scatter_set.
    Storage scatter_reduce_loop(const Storage& base,
                                const Storage& indices,
                                const Storage& src,
                                const Shape& base_shape,
                                const Shape& idx_shape,
                                int dim,
                                Dtype dt,
                                const char* name,
                                cpu::ScatterReduce reduce);

public:
    Storage scatter_amax(const Storage& base,
                         const Storage& indices,
                         const Storage& src,
                         const Shape& base_shape,
                         const Shape& idx_shape,
                         int dim,
                         Dtype dt) override;

    Storage scatter_amin(const Storage& base,
                         const Storage& indices,
                         const Storage& src,
                         const Shape& base_shape,
                         const Shape& idx_shape,
                         int dim,
                         Dtype dt) override;

    Storage scatter_prod(const Storage& base,
                         const Storage& indices,
                         const Storage& src,
                         const Shape& base_shape,
                         const Shape& idx_shape,
                         int dim,
                         Dtype dt) override;

    // Overwrite scatter: a copy of ``base`` with
    // ``out[..., idx[..., k, ...], ...] = src[..., k, ...]`` along ``dim``.
    //
    // Every dtype, by element width.  It shared the reduce loop above, which
    // instantiates f32 and f64 only, so ``index_copy`` of a bool, integer,
    // half or complex tensor refused (CHA-27) — and ``x[key] = v`` had been
    // routed around it through an additive delta that could not express an
    // overwrite at all (CHA-158).  The index may be shorter than ``base`` on
    // every axis but ``dim``, as for ``scatter_add``; a repeated index keeps
    // the value written last.
    Storage scatter_set(const Storage& base,
                        const Storage& indices,
                        const Storage& src,
                        const Shape& base_shape,
                        const Shape& idx_shape,
                        int dim,
                        Dtype dt) override;

    // Sliding-window view: out shape is (*in_shape[:dim], L, *in_shape[dim+1:], size)
    Storage unfold_dim(
        const Storage& a, const Shape& in_shape, int dim, int size, int step, Dtype dt) override;

    Storage matmul(const Storage& a, const Storage& b, const MatmulOpts& opts, Dtype dt) override;

    Storage linear(const Storage& x,
                   const Storage& weight,
                   const Storage& bias,
                   const Shape& x_shape,
                   const Shape& weight_shape,
                   const Shape& out_shape,
                   Dtype dt) override;

    std::vector<Storage> linear_backward(const Storage& grad,
                                         const Storage& x,
                                         const Storage& weight,
                                         const Shape& x_shape,
                                         const Shape& weight_shape,
                                         const Shape& bias_shape,
                                         Dtype dt) override;

    StoragePair rms_norm_forward(const Storage& x,
                                 const Storage& gamma,
                                 std::size_t outer,
                                 std::size_t normalized_size,
                                 double eps,
                                 const Shape& p5_,
                                 Dtype dt) override;

    StoragePair rms_norm_backward(const Storage& x,
                                  const Storage& gamma,
                                  const Storage& saved_rstd,
                                  const Storage& grad,
                                  std::size_t outer,
                                  std::size_t normalized_size,
                                  const Shape& p6_,
                                  const Shape& p7_,
                                  Dtype dt) override;

    std::vector<Storage> layer_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            std::size_t outer,
                                            std::size_t normalized_size,
                                            double eps,
                                            const Shape& p6_,
                                            Dtype dt) override;

    std::vector<Storage> layer_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& grad,
                                             std::size_t outer,
                                             std::size_t normalized_size,
                                             const Shape& p7_,
                                             const Shape& p8_,
                                             const Shape& p9_,
                                             Dtype dt) override;

    std::vector<Storage> batch_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            int batch,
                                            int channels,
                                            int spatial,
                                            int p6_,
                                            double eps,
                                            const Shape& p8_,
                                            Dtype dt) override;

    std::vector<Storage> batch_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& saved_xnorm,
                                             const Storage& grad,
                                             int batch,
                                             int channels,
                                             int spatial,
                                             int p9_,
                                             const Shape& p10_,
                                             Dtype dt,
                                             double eps_unused) override;

    std::vector<Storage> group_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            int batch,
                                            int channels,
                                            int spatial,
                                            int groups,
                                            const std::vector<int>& p7_,
                                            double eps,
                                            const Shape& p9_,
                                            Dtype dt) override;

    std::vector<Storage> group_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& grad,
                                             int batch,
                                             int channels,
                                             int spatial,
                                             int groups,
                                             const std::vector<int>& p9_,
                                             const Shape& p10_,
                                             Dtype dt) override;

    Storage linalg_norm(const Storage& a,
                        const Shape& shape,
                        double ord,
                        const std::vector<int>& axes,
                        bool keepdims,
                        Dtype dt) override;

    Storage linalg_cholesky(const Storage& a, const Shape& shape, bool upper, Dtype dt) override;

    Storage linalg_inv(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage linalg_solve(const Storage& a,
                         const Storage& b,
                         const Shape& a_shape,
                         const Shape& b_shape,
                         Dtype dt) override;

    Storage linalg_matrix_power(const Storage& a, const Shape& shape, int power, Dtype dt) override;

    Storage linalg_pinv(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage linalg_det(const Storage& a, const Shape& shape, Dtype dt) override;

    StoragePair linalg_qr(const Storage& a,
                          const Shape& shape,
                          const Shape& q_shape,
                          const Shape& r_shape,
                          Dtype dt) override;

    // Unpack one matrix's worth of LAPACK ``?geev`` output into complex.
    //
    // ``?geev`` reports a real matrix's spectrum in two real arrays,
    // ``wr`` and ``wi``, and its eigenvectors in a single real ``VR``
    // where a conjugate pair occupies two adjacent columns.  Reading only
    // ``wr`` and copying ``VR`` verbatim is what this backend used to do,
    // and for a matrix with any complex eigenvalue that is not a lossy
    // answer but a wrong one: the pair a \u00b1 bi came back as a, a — two
    // equal reals where there were two conjugate numbers — with nothing
    // to say the imaginary halves had been dropped.
    //
    // A real matrix having complex eigenvalues is the ordinary case, not
    // a corner one.  Any rotation has them; so does every companion
    // matrix of a polynomial with complex roots.  It went unnoticed
    // because a symmetric or triangular test matrix has a real spectrum,
    // and those are what test suites reach for.
    //
    // The packing, from ``?geev``'s documentation: if ``wi[j]`` is zero,
    // eigenvector j is column j of ``VR``.  Otherwise j and j+1 are a
    // conjugate pair with ``wi[j] > 0``, and
    //
    //     v[j]   = VR[:, j] + i * VR[:, j + 1]
    //     v[j+1] = VR[:, j] - i * VR[:, j + 1]
    //
    // ``vr`` arrives row-major (the LAPACK wrapper transposes on the way
    // out), so eigenvector j is still column j, read with stride n.
    template <typename T>
    static void
    unpack_eig(const T* wr, const T* wi, const T* vr, int n, T* values_out, T* vectors_out);

    StoragePair linalg_eig(const Storage& a,
                           const Shape& shape,
                           const Shape& values_shape,
                           const Shape& vectors_shape,
                           Dtype dt) override;

    StoragePair linalg_eigh(const Storage& a,
                            const Shape& shape,
                            const Shape& values_shape,
                            const Shape& vectors_shape,
                            Dtype dt,
                            bool upper = false) override;

    std::vector<Storage> linalg_svd(const Storage& a,
                                    const Shape& shape,
                                    bool compute_uv,
                                    const Shape& u_shape,
                                    const Shape& s_shape,
                                    const Shape& vt_shape,
                                    Dtype dt) override;

    // ``{LU, pivots, info}`` — getrf completes on a singular matrix, so its
    // ``info`` is returned, not raised (``LinalgFactor.h``).
    std::vector<Storage> linalg_lu_factor(const Storage& a, const Shape& shape, Dtype dt) override;

    // Substitution through BLAS ``trsm``, not the LAPACK driver ``trtrs``.
    // ``trtrs`` checks the diagonal first and refuses a singular triangle
    // (``info > 0``) without solving; the reference answers it with the IEEE
    // result of substitution instead — a nonzero over a zero pivot is ±inf,
    // ``0 / 0`` is NaN — and so does this.  The Metal stream delegates here,
    // so this is the one triangular-solve policy for both devices.
    Storage linalg_solve_triangular(const Storage& a,
                                    const Storage& b,
                                    const Shape& a_shape,
                                    const Shape& b_shape,
                                    bool upper,
                                    bool unitriangular,
                                    Dtype dt) override;

    std::vector<Storage> linalg_lstsq(const Storage& a,
                                      const Storage& b,
                                      const Shape& a_shape,
                                      const Shape& b_shape,
                                      Dtype dt) override;

    Storage linalg_lu_solve(const Storage& LU,
                            const Storage& pivots,
                            const Storage& b,
                            const Shape& lu_shape,
                            const Shape& b_shape,
                            Dtype dt) override;

    Storage linalg_householder_product(const Storage& H,
                                       const Storage& tau,
                                       const Shape& h_shape,
                                       Dtype dt) override;

    // ``{LD, pivots, info}`` — sytrf completes when D is singular, so its
    // ``info`` is returned, not raised (``LinalgFactor.h``).
    std::vector<Storage> linalg_ldl_factor(const Storage& a, const Shape& shape, Dtype dt) override;

    // fold (col2im): scatter-add (N, C*kH*kW, L) → (N, C, outH, outW)
    Storage nn_fold(const Storage& x,
                    const Shape& x_shape,
                    const Shape& out_shape,
                    const std::vector<int>& kernel_size,
                    const std::vector<int>& stride,
                    const std::vector<int>& padding,
                    const std::vector<int>& dilation,
                    Dtype dt) override;

    // EmbeddingBag: gather + reduce per bag
    Storage embedding_bag_forward(const Storage& weight,
                                  const Storage& indices,
                                  const Storage& offsets,
                                  const Shape& weight_shape,
                                  const Shape& indices_shape,
                                  int mode,
                                  int padding_idx,
                                  bool include_last_offset,
                                  Dtype dt) override;

    Storage embedding_bag_backward(const Storage& grad_out,
                                   const Storage& weight,
                                   const Storage& indices,
                                   const Storage& offsets,
                                   const Shape& weight_shape,
                                   const Shape& indices_shape,
                                   int mode,
                                   int padding_idx,
                                   bool include_last_offset,
                                   Dtype dt) override;

    // ── astype ────────────────────────────────────────────────────────────────
    //
    // Generic Cartesian-product cast over the supported scalar dtypes.  Earlier
    // versions of this file hand-spelled every supported (src, dst) pair, which
    // left holes (notably ``Bool -> I64``) that surfaced during the 3.0.2 +
    // followup smoke runs — bool reductions need to promote through I64 first
    // and that promotion was silently NotImplementedError.  The dispatch below
    // covers every pair among {F16, F32, F64, I8, I16, I32, I64, Bool}.  C64
    // and BF16 are handled outside the table, through float32, because their
    // host representation isn't a simple ``static_cast``-able scalar (F16 uses the IEEE half decoder from
    // ``item()`` indirectly via the F32 path — handled with a small helper).
    Storage astype(const Storage& a, const Shape& shape, Dtype src_dt, Dtype dst_dt) override;

    // ── flip ──────────────────────────────────────────────────────────────────
    Storage
    flip(const Storage& a, const Shape& shape, const std::vector<int>& dims, Dtype dt) override;

    // ── masked_select ─────────────────────────────────────────────────────────
    Storage masked_select_count(const Storage& mask, const Shape& shape, Dtype /*dt*/) override;

    Storage masked_select(const Storage& a,
                          const Storage& mask,
                          const Shape& a_shape,
                          const Shape& /*mask_shape*/,
                          std::int64_t n_true,
                          Dtype dt) override;

    // ── ctc_loss ──────────────────────────────────────────────────────────────
    Storage ctc_loss_forward(const Storage& log_probs,
                             const Storage& targets,
                             const Storage& input_lengths,
                             const Storage& target_lengths,
                             const Shape& lp_shape,
                             int blank,
                             bool zero_infinity,
                             Dtype dt) override;

    Storage ctc_loss_backward(const Storage& log_probs,
                              const Storage& targets,
                              const Storage& input_lengths,
                              const Storage& target_lengths,
                              const Storage& grad_out,
                              const Shape& lp_shape,
                              int blank,
                              bool zero_infinity,
                              Dtype dt) override;

    Storage
    broadcast(const Storage& a, const Shape& src_shape, const Shape& dst_shape, Dtype dt) override;

    Storage
    repeat(const Storage& a, const Shape& shape, Dtype dt, std::int64_t repeats, int axis) override;

    Storage tile(const Storage& a,
                 const Shape& shape,
                 Dtype dt,
                 const std::vector<std::int64_t>& reps) override;

    Storage
    permute(const Storage& a, const Shape& shape, const std::vector<int>& perm, Dtype dt) override;

    Storage pad(const Storage& a,
                const Shape& shape,
                Dtype dt,
                const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                double constant) override;

    Storage pow_scalar(const Storage& a, const Shape& shape, Dtype dt, double exp) override;

    Storage rpow_scalar(const Storage& a, const Shape& shape, Dtype dt, double base) override;

    Storage
    clip(const Storage& a, const Shape& shape, Dtype dt, double min_v, double max_v) override;

    Storage cast(const Storage& a, const Shape& shape, Dtype src_dt, Dtype dst_dt) override;

    Storage mse_loss(const Storage& input,
                     const Storage& target,
                     const Shape& shape,
                     Dtype dt,
                     int reduction) override;

    std::pair<Storage, Storage> mse_loss_backward(const Storage& input,
                                                  const Storage& target,
                                                  const Storage& grad,
                                                  const Shape& shape,
                                                  Dtype dt,
                                                  int reduction) override;

    Storage huber_loss(const Storage& input,
                       const Storage& target,
                       const Shape& shape,
                       Dtype dt,
                       double delta,
                       int reduction) override;

    std::pair<Storage, Storage> huber_loss_backward(const Storage& input,
                                                    const Storage& target,
                                                    const Storage& grad,
                                                    const Shape& shape,
                                                    Dtype dt,
                                                    double delta,
                                                    int reduction) override;

    Storage bce_loss(const Storage& input,
                     const Storage& target,
                     const Storage& weight,
                     const Shape& shape,
                     Dtype dt,
                     double eps,
                     int reduction) override;

    std::vector<Storage> bce_loss_backward(const Storage& input,
                                           const Storage& target,
                                           const Storage& weight,
                                           const Storage& grad,
                                           const Shape& shape,
                                           Dtype dt,
                                           double eps,
                                           int reduction) override;

    Storage bce_with_logits_loss(const Storage& input,
                                 const Storage& target,
                                 const Storage& weight,
                                 const Storage& pos_weight,
                                 const Shape& shape,
                                 const Shape& weight_shape,
                                 const Shape& pos_weight_shape,
                                 Dtype dt,
                                 int reduction) override;

    std::vector<Storage> bce_with_logits_backward(const Storage& input,
                                                  const Storage& target,
                                                  const Storage& weight,
                                                  const Storage& pos_weight,
                                                  const Storage& grad,
                                                  const Shape& shape,
                                                  Dtype dt,
                                                  int reduction) override;

    ClassLossForwardResult cross_entropy_loss(const Storage& input,
                                              const Storage& target,
                                              const Storage* weight,
                                              const Shape& input_shape,
                                              const Shape&,
                                              Dtype dt,
                                              double eps,
                                              int ignore_index,
                                              int reduction) override;

    Storage cross_entropy_backward(const Storage& saved_softmax,
                                   const Storage& target,
                                   const Storage* weight,
                                   const Storage& valid_count,
                                   const Storage& grad,
                                   const Shape& input_shape,
                                   Dtype dt,
                                   int ignore_index,
                                   int reduction) override;

    ClassLossForwardResult nll_loss(const Storage& input,
                                    const Storage& target,
                                    const Storage* weight,
                                    const Shape& input_shape,
                                    const Shape&,
                                    Dtype dt,
                                    int ignore_index,
                                    int reduction) override;

    Storage nll_loss_backward(const Storage& target,
                              const Storage* weight,
                              const Storage& valid_count,
                              const Storage& grad,
                              const Shape& input_shape,
                              Dtype dt,
                              int ignore_index,
                              int reduction) override;

    CpuStorage to_cpu(const Storage& a, const Shape&) override;

    std::vector<Storage> sdpa_forward(const Storage& q,
                                      const Storage& k,
                                      const Storage& v,
                                      const Storage* attn_mask,
                                      const Shape& q_shape,
                                      const Shape& k_shape,
                                      const Shape& v_shape,
                                      Dtype mask_dtype,
                                      std::size_t mask_numel,
                                      double scale,
                                      bool is_causal,
                                      bool need_weights,
                                      Dtype dt) override;

    std::vector<Storage> sdpa_backward(const Storage& grad_out,
                                       const Storage& q,
                                       const Storage& k,
                                       const Storage& v,
                                       const Storage& saved_weights,
                                       const Storage* attn_mask,
                                       const Shape& q_shape,
                                       const Shape& k_shape,
                                       const Shape& v_shape,
                                       Dtype mask_dtype,
                                       double scale,
                                       bool is_causal,
                                       Dtype dt) override;

    Storage conv_transpose_nd_forward(const Storage& x,
                                      const Storage& W,
                                      const Storage& b,
                                      int B,
                                      int Cin,
                                      int Cout,
                                      const int* S,
                                      const int* K,
                                      const int* O,
                                      const IBackend::ConvTransposeNdOpts& opts,
                                      const Shape& out_shape,
                                      Dtype dt) override;

    std::vector<Storage> conv_transpose_nd_backward(const Storage& grad_out,
                                                    const Storage& x,
                                                    const Storage& W,
                                                    int B,
                                                    int Cin,
                                                    int Cout,
                                                    const int* S,
                                                    const int* K,
                                                    const int* O,
                                                    const IBackend::ConvTransposeNdOpts& opts,
                                                    Dtype dt) override;

    Storage conv_nd_forward(const Storage& x,
                            const Storage& W,
                            const Storage& b,
                            int B,
                            int Cin,
                            int Cout,
                            int Cin_g,
                            int Cout_g,
                            const int* S,
                            const int* K,
                            const int* O,
                            const IBackend::ConvNdOpts& opts,
                            const Shape& out_shape,
                            Dtype dt) override;

    std::vector<Storage> conv_nd_backward(const Storage& grad_out,
                                          const Storage& x,
                                          const Storage& W,
                                          int B,
                                          int Cin,
                                          int Cout,
                                          int Cin_g,
                                          int Cout_g,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const IBackend::ConvNdOpts& opts,
                                          Dtype dt) override;

    Storage unfold_forward(const Storage& x,
                           int B,
                           int C,
                           const std::vector<int>& S,
                           const std::vector<int>& K,
                           const std::vector<int>& O,
                           const std::vector<int>& stride,
                           const std::vector<int>& pad,
                           const std::vector<int>& dilation,
                           const Shape& p9_,
                           Dtype dt) override;

    Storage unfold_backward(const Storage& grad_out,
                            int B,
                            int C,
                            const std::vector<int>& S,
                            const std::vector<int>& K,
                            const std::vector<int>& O,
                            const std::vector<int>& stride,
                            const std::vector<int>& pad,
                            const std::vector<int>& dilation,
                            Dtype dt) override;

    std::pair<Storage, Storage> expand_and_multiply(const Storage& mask,
                                                    const Storage& x,
                                                    const Shape& mask_shape,
                                                    const Shape& x_shape,
                                                    Dtype dt) override;

    Storage drop_block_mask(const Storage& seed,
                            double drop_prob,
                            std::int64_t block_size,
                            const Shape& x_shape,
                            Dtype dt) override;

    std::vector<Storage> max_pool_nd_forward(const Storage& x,
                                             const Shape& x_shape,
                                             const Shape& out_shape,
                                             const PoolOpts& opts,
                                             Dtype dt) override;

    Storage max_pool_nd_backward(const Storage& grad_out,
                                 const Storage& saved_argmax,
                                 const Shape& x_shape,
                                 const Shape& out_shape,
                                 const PoolOpts& opts,
                                 Dtype dt) override;

    Storage avg_pool_nd_forward(const Storage& x,
                                const Shape& x_shape,
                                const Shape& out_shape,
                                const PoolOpts& opts,
                                Dtype dt) override;

    Storage avg_pool_nd_backward(const Storage& grad_out,
                                 const Shape& x_shape,
                                 const Shape& out_shape,
                                 const PoolOpts& opts,
                                 Dtype dt) override;

    Storage embedding_forward(const Storage& weight,
                              const Storage& indices,
                              const Shape& weight_shape,
                              const Shape& indices_shape,
                              const Shape& out_shape,
                              int padding_idx,
                              Dtype dt) override;

    Storage embedding_backward(const Storage& grad_out,
                               const Storage& indices,
                               const Shape& weight_shape,
                               const Shape& indices_shape,
                               int padding_idx,
                               Dtype dt) override;

    Storage
    sinusoidal_pos_embedding(std::int64_t seq_len, std::int64_t embed_dim, Dtype dt) override;

    std::vector<Storage> rope_forward(const Storage& x,
                                      const Storage* position_ids,
                                      const Shape& x_shape,
                                      bool interleaved,
                                      Dtype pos_dtype,
                                      Dtype dt) override;

    std::vector<Storage> batch_norm_eval_forward(const Storage& x,
                                                 const Storage& mean,
                                                 const Storage& var,
                                                 const Storage& gamma,
                                                 const Storage& beta,
                                                 const Shape& x_shape,
                                                 int C,
                                                 int spatial,
                                                 double eps,
                                                 Dtype dt) override;

    std::vector<Storage> batch_norm_eval_backward(const Storage& x,
                                                  const Storage& mean,
                                                  const Storage& gamma,
                                                  const Storage& rstd,
                                                  const Storage& grad_out,
                                                  const Shape& x_shape,
                                                  int C,
                                                  int spatial,
                                                  Dtype dt) override;

    std::vector<Storage> lp_normalize_forward(const Storage& x,
                                              const Shape& x_shape,
                                              double ord,
                                              int axis,
                                              double eps,
                                              Dtype dt) override;

    Storage lp_normalize_backward(const Storage& x,
                                  const Storage& saved_norm,
                                  const Storage& grad_out,
                                  const Shape& x_shape,
                                  double ord,
                                  int axis,
                                  Dtype dt) override;

    std::vector<Storage> global_response_norm_forward(const Storage& x,
                                                      const Storage& gamma,
                                                      const Storage& beta,
                                                      const Shape& x_shape,
                                                      double eps,
                                                      Dtype dt) override;

    std::vector<Storage> global_response_norm_backward(const Storage& x,
                                                       const Storage& gamma,
                                                       const Storage& beta,
                                                       const Storage& saved_Nx,
                                                       const Storage& grad_out,
                                                       const Shape& x_shape,
                                                       double eps,
                                                       Dtype dt) override;

    Storage interpolate_nearest_2d_forward(
        const Storage& input, const Shape& in_shape, int H_out, int W_out, Dtype dt) override;

    Storage interpolate_nearest_3d_forward(const Storage& input,
                                           const Shape& in_shape,
                                           int D_out,
                                           int H_out,
                                           int W_out,
                                           Dtype dt) override;

    Storage interpolate_nearest_2d_backward(
        const Storage& grad_out, const Shape& in_shape, int H_out, int W_out, Dtype dt) override;

    Storage interpolate_nearest_3d_backward(const Storage& grad_out,
                                            const Shape& in_shape,
                                            int D_out,
                                            int H_out,
                                            int W_out,
                                            Dtype dt) override;

    Storage interpolate_bilinear_forward(const Storage& input,
                                         const Shape& in_shape,
                                         int H_out,
                                         int W_out,
                                         bool align_corners,
                                         Dtype dt) override;

    Storage interpolate_bilinear_backward(const Storage& grad_out,
                                          const Shape& in_shape,
                                          int H_out,
                                          int W_out,
                                          bool align_corners,
                                          Dtype dt) override;

    Storage interpolate_trilinear_forward(const Storage& input,
                                          const Shape& in_shape,
                                          int D_out,
                                          int H_out,
                                          int W_out,
                                          bool align_corners,
                                          Dtype dt) override;

    Storage interpolate_trilinear_backward(const Storage& grad_out,
                                           const Shape& in_shape,
                                           int D_out,
                                           int H_out,
                                           int W_out,
                                           bool align_corners,
                                           Dtype dt) override;

    Storage affine_grid_forward(
        const Storage& theta, int N, int H, int W, bool align_corners, Dtype dt) override;

    Storage affine_grid_backward(
        const Storage& grad_grid, int N, int H, int W, bool align_corners, Dtype dt) override;

    Storage grid_sample_forward(const Storage& input,
                                const Storage& grid,
                                const Shape& in_shape,
                                const Shape& grid_shape,
                                int mode,
                                int padding_mode,
                                bool align_corners,
                                Dtype dt) override;

    std::vector<Storage> grid_sample_backward(const Storage& grad_out,
                                              const Storage& input,
                                              const Storage& grid,
                                              const Shape& in_shape,
                                              const Shape& grid_shape,
                                              int mode,
                                              int padding_mode,
                                              bool align_corners,
                                              Dtype dt) override;

    Storage bilinear_layer_forward(const Storage& x1,
                                   const Storage& x2,
                                   const Storage& weight,
                                   const Storage& bias,
                                   bool has_bias,
                                   const Shape& x1_shape,
                                   const Shape& x2_shape,
                                   const Shape& w_shape,
                                   Dtype dt) override;

    std::vector<Storage> bilinear_layer_backward(const Storage& grad_out,
                                                 const Storage& x1,
                                                 const Storage& x2,
                                                 const Storage& weight,
                                                 const Shape& x1_shape,
                                                 const Shape& x2_shape,
                                                 const Shape& w_shape,
                                                 bool has_bias,
                                                 Dtype dt) override;

    Storage one_hot_forward(const Storage& indices,
                            const Shape& indices_shape,
                            int num_classes,
                            Dtype out_dtype) override;

    Storage rotate_forward(const Storage& input,
                           const Shape& shape,
                           double angle_rad_neg,
                           double cx,
                           double cy,
                           Dtype dt) override;

    Storage rope_backward(const Storage& grad_out,
                          const Storage& saved_cos,
                          const Storage& saved_sin,
                          const Shape& x_shape,
                          bool interleaved,
                          Dtype dt) override;

private:
    // Dispatches im2col to the 1-D, 2-D, or 3-D variant based on N.
    // S/K/O are input spatial sizes, kernel sizes, and output sizes indexed 0..N-1.
    // Output positions are lowered to columns a slab at a time along the
    // first output axis, so the column matrix never holds more than
    // kColumnBudget elements.  A slab is the whole problem with that axis cut
    // to [o0, o0 + count): the same im2col with O[0] = count and the padding
    // moved by o0 * stride, since an input position o * s - p + k * d is
    // unchanged when o and p shift together.  Unsliced, a (1, 96, 6, 480, 832)
    // conv3d asked for 4.1 G column elements — 16.5 GB for the forward and
    // twice that for the backward.
    static constexpr std::int64_t kColumnBudget = std::int64_t{1} << 27;

    // ``LUCID_CONV_COLUMN_BUDGET`` lowers the budget so tests can drive the
    // slab path on small inputs; it is not a tuning knob.
    static std::int64_t column_budget();

    static int checked_extent(const int* dims, int n, const char* what);

    static std::int64_t column_inner(const int* O, int N);

    static int column_slab(const int* O, int N, int K_flat);

    static std::size_t column_slab_elements(const int* O, int N, int K_flat, int slab);

    template <typename Fn>
    static void
    for_each_column_slab(const int* O, const IBackend::ConvNdOpts& opts, int N, int slab, Fn&& fn);

    static void conv_nd_im2col_f32(const float* x,
                                   float* cols,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N);
    static void conv_nd_im2col_f64(const double* x,
                                   double* cols,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N);
    static void conv_nd_col2im_f32(const float* cols,
                                   float* dx,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N);
    static void conv_nd_col2im_f64(const double* cols,
                                   double* dx,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N);
    static void unfold_im2col_f32(const float* x,
                                  float* y,
                                  int C,
                                  const int* S,
                                  const int* K,
                                  const int* O,
                                  const int* stride,
                                  const int* pad,
                                  const int* dilation,
                                  int N);
    static void unfold_im2col_f64(const double* x,
                                  double* y,
                                  int C,
                                  const int* S,
                                  const int* K,
                                  const int* O,
                                  const int* stride,
                                  const int* pad,
                                  const int* dilation,
                                  int N);
    static void unfold_col2im_f32(const float* cols,
                                  float* dx,
                                  int C,
                                  const int* S,
                                  const int* K,
                                  const int* O,
                                  const int* stride,
                                  const int* pad,
                                  const int* dilation,
                                  int N);
    static void unfold_col2im_f64(const double* cols,
                                  double* dx,
                                  int C,
                                  const int* S,
                                  const int* K,
                                  const int* O,
                                  const int* stride,
                                  const int* pad,
                                  const int* dilation,
                                  int N);

    void compute_mse_loss_values(
        const Storage& input, const Storage& target, std::byte* out, std::size_t n, Dtype dt);

    template <typename T>
    T mse_loss_sum(const Storage& input, const Storage& target, std::size_t n);

    void compute_huber_loss_values(const Storage& input,
                                   const Storage& target,
                                   std::byte* out,
                                   std::size_t n,
                                   Dtype dt,
                                   double delta);

    template <typename T>
    T huber_loss_sum(const Storage& input, const Storage& target, std::size_t n, T delta);

    void compute_bce_loss_values(const Storage& input,
                                 const Storage& target,
                                 const Storage& weight,
                                 std::byte* out,
                                 std::size_t n,
                                 Dtype dt,
                                 double eps);

    template <typename T>
    T bce_loss_sum(
        const Storage& input, const Storage& target, const Storage& weight, std::size_t n, T eps);

    void compute_bce_logits_values(const Storage& input,
                                   const Storage& target,
                                   const Storage& weight,
                                   const Storage& pos_weight,
                                   std::byte* out,
                                   std::size_t n,
                                   Dtype dt);

    template <typename T>
    T bce_logits_sum(const Storage& input,
                     const Storage& target,
                     const Storage& weight,
                     const Storage& pos_weight,
                     std::size_t n);

    static int class_loss_spatial(const Shape& input_shape);

    static CpuStorage make_cpu_scalar(double value, Dtype dt);

    // The class of target ``i``, or -1 where the target is ``ignore_index``.
    // A class outside [0, channels) is refused with IndexError, as the
    // reference refuses it.  The losses read the probability and the class
    // weight at whatever offset the target named, so an out-of-range label
    // read memory outside both.  The comparison with ``ignore_index`` is
    // made at full width: cut to ``int`` first, a target of 2^32 - 100 was
    // taken for the default -100 and skipped.  Metal answers NaN for such a
    // target instead (policy B, GpuBackend's ``gpu_axis_index``).
    static std::int64_t class_target(
        const char* op, const CpuStorage& target, std::size_t i, int ignore_index, int channels);

    template <typename T>
    static T sum_values(const T* values, std::size_t n);

    static Storage reduce_class_losses(const CpuStorage& losses,
                                       std::size_t samples,
                                       std::size_t valid_count,
                                       Dtype dt,
                                       int reduction);

    static std::pair<std::size_t, std::size_t> flatten_linear_x(const Shape& x_shape);

    template <typename T>
    static void add_linear_bias_typed(T* y, const T* b, std::size_t M, std::size_t N);

    static void
    add_linear_bias(CpuStorage& y, const CpuStorage& bias, std::size_t M, std::size_t N, Dtype dt);

    template <typename T>
    static void sum_linear_rows_typed(const T* grad, T* db, std::size_t M, std::size_t N);

    static void
    sum_linear_rows(const CpuStorage& grad, CpuStorage& db, std::size_t M, std::size_t N, Dtype dt);

    static Shape
    reduced_norm_shape(const Shape& shape, const std::vector<int>& axes, bool keepdims);

    template <typename T, typename Fn>
    static void norm_elementwise_loop(const T* in,
                                      const Shape& shape,
                                      const std::vector<bool>& reduce_mask,
                                      const Shape& out_shape,
                                      Fn fn);

    template <typename T>
    static void norm_typed(const T* in,
                           T* out,
                           const Shape& shape,
                           const std::vector<int>& axes,
                           double ord,
                           const Shape& out_shape);

    static std::int64_t leading_matrix_batch_count(const Shape& shape, std::size_t mat_dims);

    // Whether every one of the ``n`` values is finite.  LAPACK's drivers
    // reject a NaN as an illegal argument, so the linalg kernels ask first.
    template <typename T>
    static bool all_finite(const T* p, std::size_t n);

    template <typename T>
    static void fill_nan(T* p, std::size_t n);

    // geev has no answer for a non-finite matrix: its balancing step turns
    // the NaN into an illegal argument for the Hessenberg solver (and the
    // Fortran runtime prints that to file descriptor 2).  The reference
    // raises for it too, so this raises before LAPACK is reached, with the
    // message the Metal stream uses.
    template <typename T>
    static void require_finite_eig(const T* p, std::size_t n);

    template <typename T>
    static void set_matrix_identity(T* out, int n);

    template <typename T>
    static void pinv_one(const T* a, int m, int n, T* aplus);

    static float ipiv_sign(const int* ipiv, int n);

    static void batch_norm_forward_f32_fast(const float* x,
                                            const float* gamma,
                                            const float* beta,
                                            float* y,
                                            float* mean_per_c,
                                            float* rstd_per_c,
                                            int batch,
                                            int channels,
                                            int spatial,
                                            double eps);

    template <typename T>
    static void batch_norm_forward_typed(const T* x,
                                         const T* gamma,
                                         const T* beta,
                                         T* y,
                                         T* mean_per_c,
                                         T* rstd_per_c,
                                         int batch,
                                         int channels,
                                         int spatial,
                                         double eps);

    template <typename T>
    static void batch_norm_backward_typed(const T* x,
                                          const T* gamma,
                                          const T* mean,
                                          const T* rstd,
                                          const T* g,
                                          T* dx,
                                          T* dgamma,
                                          T* dbeta,
                                          int batch,
                                          int channels,
                                          int spatial);

    // Reductions over a whole normalisation group run to a million elements
    // at production resolutions, and a serial float sum loses accuracy in
    // proportion to how many terms it has taken -- roughly 2e-04 over 1M
    // against 8e-07 over 4k.  MLX does not, so the same model disagrees with
    // itself across devices and the gap reads as float32 accumulation rather
    // than as a kernel.  LayerNorm escapes it here only because it reaches
    // vDSP instead of a hand-written loop.  Widening the accumulator costs
    // one register and removes the size dependence.
    template <typename T>
    using NormAcc = std::conditional_t<std::is_same_v<T, float>, double, T>;

    template <typename T>
    static void group_norm_forward_typed(const T* x,
                                         const T* gamma,
                                         const T* beta,
                                         T* y,
                                         T* mean_bg,
                                         T* rstd_bg,
                                         int batch,
                                         int channels,
                                         int spatial,
                                         int groups,
                                         double eps);

    template <typename T>
    static void group_norm_backward_typed(const T* x,
                                          const T* gamma,
                                          const T* mean_bg,
                                          const T* rstd_bg,
                                          const T* g,
                                          T* dx,
                                          T* dgamma,
                                          T* dbeta,
                                          int batch,
                                          int channels,
                                          int spatial,
                                          int groups);

    // Fills ptr with the scalar value 1 for dtype dt.  Used by ones().
    void fill_ones(std::byte* ptr, std::size_t n, Dtype dt);

    // Generic unary dispatch: allocates output, calls fn32/fn64/fni32 based on dt.
    // I64 falls back to a scalar loop via static_cast<double>.
    template <class F32Fn, class F64Fn, class I32Fn>
    Storage
    unary_op(const Storage& a, const Shape& shape, Dtype dt, F32Fn fn32, F64Fn fn64, I32Fn fni32);

    // Generic binary dispatch: allocates output, calls fn32/fn64/fni32/fni64 based on dt.
    template <class F32Fn, class F64Fn, class I32Fn, class I64Fn>
    Storage binary_op(const Storage& a,
                      const Storage& b,
                      const Shape& shape,
                      Dtype dt,
                      F32Fn fn32,
                      F64Fn fn64,
                      I32Fn fni32,
                      I64Fn fni64);

    // Tags used by reduce_axes to select the accumulation operation.
    enum class ReduceOp { Sum, Mean, Max, Min };

    // Reduces a over the axes in opts by iterating axes from largest to smallest
    // (descending sort ensures that axis indices remain valid after each step).
    // Each single-axis pass calls the appropriate cpu::sum_axis / max_axis /
    // min_axis primitive.  Mean divides by the reduce dimension after summing.
    Storage reduce_axes(
        const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt, ReduceOp op);

    void cast_impl(const std::byte* src, std::byte* dst, std::size_t n, Dtype src_dt, Dtype dst_dt);

    static CpuStorage alloc_cpu(std::size_t numel, Dtype dt) {
        const std::size_t nb = count_nbytes(numel, dtype_size(dt));
        return CpuStorage{allocate_aligned_bytes(nb, Device::CPU), nb, dt};
    }

    // ``im2col`` / ``col2im`` / GEMM shims for the transposed convolution.
    //
    // The transposed conv is the data gradient of a forward convolution, so
    // the roles are crossed relative to the forward path: the *output*
    // extent ``O`` plays the forward conv's input length and the *input*
    // extent ``S`` plays its output length.  That is why ``O`` is passed
    // where the primitives name their argument ``L`` / ``H`` / ``D``.
    //
    // The ``dilation`` argument used to be hardcoded to ``1`` here, which
    // is what confined the whole op to ``dilation == 1``; the underlying
    // primitives have always accepted it.
    static void ctnd_im2col(const float* src,
                            float* cols,
                            int C,
                            const int* O,
                            const int* K,
                            const int* S,
                            const int* stride,
                            const int* pad,
                            const int* dil,
                            int N);

    static void ctnd_im2col(const double* src,
                            double* cols,
                            int C,
                            const int* O,
                            const int* K,
                            const int* S,
                            const int* stride,
                            const int* pad,
                            const int* dil,
                            int N);

    static void ctnd_col2im(const float* cols,
                            float* dst,
                            int C,
                            const int* O,
                            const int* K,
                            const int* S,
                            const int* stride,
                            const int* pad,
                            const int* dil,
                            int N);

    static void ctnd_col2im(const double* cols,
                            double* dst,
                            int C,
                            const int* O,
                            const int* K,
                            const int* S,
                            const int* stride,
                            const int* pad,
                            const int* dil,
                            int N);

    // Precision-dispatching GEMM so the transposed-conv bodies can be one
    // generic lambda rather than a copy-pasted F32 / F64 pair.
    static void ctnd_gemm(bool ta,
                          bool tb,
                          int M,
                          int Nn,
                          int Kk,
                          float alpha,
                          const float* A,
                          int lda,
                          const float* Bm,
                          int ldb,
                          float beta,
                          float* C,
                          int ldc);

    static void ctnd_gemm(bool ta,
                          bool tb,
                          int M,
                          int Nn,
                          int Kk,
                          double alpha,
                          const double* A,
                          int lda,
                          const double* Bm,
                          int ldb,
                          double beta,
                          double* C,
                          int ldc);

    Storage ge_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage lt_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage add_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) override;

    Storage mul_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) override;

    Storage in_range_mask(const Storage& a,
                          const Shape& shape,
                          Dtype dt,
                          double lo,
                          double hi,
                          bool inclusive) override;

    Storage leaky_mask(const Storage& a, const Shape& shape, Dtype dt, double slope) override;

    Storage positive_mask(const Storage& a, const Shape& shape, Dtype dt) override;

    // 3.4+ Phase A.1: fused ReLU backward.  Single-pass loop replacing
    // positive_mask + multiply, eliminating one intermediate buffer.  See
    // GpuBackend::relu_backward for the GPU rationale.
    Storage
    relu_backward(const Storage& g, const Storage& x, const Shape& shape, Dtype dt) override;

    Storage reduce_grad_to_shape(const Storage& grad,
                                 const Shape& grad_shape,
                                 const Shape& target_shape,
                                 Dtype dt) override;

    Storage broadcast_back_for_reduce(const Storage& grad,
                                      const Shape& grad_shape,
                                      const Shape& input_shape,
                                      const std::vector<int>& axes,
                                      bool keepdims,
                                      Dtype dt) override;

    Storage full(const Shape& shape, Dtype dt, double fill_value) override;

    Storage eye(std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt) override;

    Storage diag(const Storage& v,
                 const Shape& v_shape,
                 std::int64_t k,
                 Dtype dt,
                 Shape& out_shape) override;

    Storage tri(const Storage& input, const Shape& shape, Dtype dt, int k, bool upper) override;

    Storage floordiv(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage inner(const Storage& a,
                  const Storage& b,
                  const Shape& a_shape,
                  const Shape& b_shape,
                  const Shape& out_shape,
                  Dtype dt) override;

    CpuStorage permute_cpu(const CpuStorage& src,
                           const Shape& src_shape,
                           const std::vector<int>& perm,
                           Dtype dt) override;

    Storage tensordot(const Storage&,
                      const Storage&,
                      const Shape&,
                      const Shape&,
                      const Shape&,
                      const std::vector<int>&,
                      const std::vector<int>&,
                      Dtype) override;

    Storage where_op(const Storage& cond,
                     const Storage& x,
                     const Storage& y,
                     const Shape& shape,
                     Dtype dt) override;

    Storage reduce_broadcast(const Storage& grad,
                             const Shape& input_shape,
                             const Shape& output_shape,
                             Dtype dt) override;

    Storage histogram_forward(const Storage& input,
                              const Shape& input_shape,
                              Dtype input_dtype,
                              double lo,
                              double hi,
                              std::int64_t bins,
                              bool density) override;

    CpuStorage nonzero_forward(const Storage& input,
                               const Shape& input_shape,
                               Dtype input_dtype,
                               std::size_t& numel_out) override;

    std::vector<Storage> lstm_forward(const Storage& input,
                                      const Storage& h0,
                                      const Storage& c0,
                                      const std::vector<Storage>& weights,
                                      const LstmOpts& opts,
                                      const Shape& out_shape,
                                      Dtype dt) override;

    // Float32 and float64 run natively; the 16-bit formats widen to float32
    // at the door and round back on the way out.  The saved gate and cell
    // trajectories stay float32 — only lstm_backward reads them.
    std::vector<Storage> lstm_forward_train(const Storage& input,
                                            const Storage& h0,
                                            const Storage& c0,
                                            const std::vector<Storage>& weights,
                                            const LstmOpts& opts,
                                            Dtype dt) override;

    std::vector<Storage> lstm_backward(const Storage& grad_output,
                                       const Storage& grad_hn,
                                       const Storage& grad_cn,
                                       const Storage& input,
                                       const Storage& h0,
                                       const std::vector<Storage>& weights,
                                       const Storage& gates_all,
                                       const Storage& cells_all,
                                       const LstmOpts& opts,
                                       Dtype dt) override;

    static std::vector<Storage> lstm_widen(const std::vector<Storage>& weights);

    Storage fused_linear_relu_forward(const Storage& x,
                                      const Storage& w,
                                      const Storage& b,
                                      const Shape& out_shape,
                                      Dtype dt) override;

    Storage fused_linear_gelu_forward(const Storage& x,
                                      const Storage& w,
                                      const Storage& b,
                                      const Shape& out_shape,
                                      Dtype dt) override;
};

}  // namespace backend
}  // namespace lucid
