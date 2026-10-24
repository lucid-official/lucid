// lucid/_C/backend/gpu/GpuBackend.h
//
// Concrete IBackend implementation for Apple Silicon GPU using MLX.  Almost
// every operation is a thin delegation to the corresponding mlx::core:: API,
// with shape/dtype conversions handled by the MlxBridge helpers.
//
// Key invariants:
//   - All Storage values passed to GpuBackend methods must hold GpuStorage.
//     The Dispatcher guarantees this by routing Device::GPU tensors here.
//   - mlx uses lazy evaluation: operations build a graph but do not execute
//     immediately.  Execution is deferred until a .eval() call or until data
//     is needed by the CPU (e.g. in to_cpu or MetalKernelRunner).
//   - float64 is NOT supported on the GPU (Metal limitation).  Callers must
//     cast to float32 first or keep the tensor on CPU.
//
// Private helpers:
//   mlx_unary(a, fn)  — extracts GpuStorage, applies fn(*arr), wraps result.
//   mlx_binary(a, b, fn) — extracts both GpuStorage arrays, applies fn, wraps.
//   mlx_reduce(a, opts, fn) — handles multi-axis reductions via MLX.
//   k_linalg_stream   — CPU-device stream used for linalg ops because MLX's
//                        linalg routines (eigh, svd, qr, etc.) run on CPU even
//                        when called from the GPU backend.
//
// Layout conversions:
//   MLX convolutions use NHWC (channels-last) order.  The convolution forward/
//   backward methods permute NCHW inputs to NHWC before calling mlx::core::conv*
//   and permute results back.  Helper functions gpu_nchw_to_nhwc_perm,
//   gpu_nhwc_to_nchw_perm, and gpu_w_to_transpose_perm build the required
//   permutation vectors for 1-D, 2-D, and 3-D cases;
//   gpu_w_to_mlx_conv_transpose additionally re-splits the grouped channel
//   axis, which Lucid and MLX place on opposite sides of the weight.
//
// Self-registration:
//   ``BackendInit.cpp`` defines ``GpuBackendRegistrar`` to register a
//   ``GpuBackend`` singleton via ``Dispatcher`` at static-init time.
//
// Method family units
// -------------------
// Non-template ``GpuBackend`` member definitions live out-of-line in
// ``lucid/_C/backend/gpu/methods/<Family>.cpp`` across 14 symmetric op-family
// translation units (``Factory``, ``Binary``, ``Unary``, ``Activation``,
// ``Reduction``, ``Layout``, ``Indexing``, ``Linalg``, ``Matmul``, ``Norm``,
// ``Loss``, ``Conv``, ``Spatial``, ``Sequence``).
//
// API surface
// -----------
// ~225 public method overrides grouped semantically into:
//   • Construction      — ``from_cpu``, ``zeros``, ``ones``, ``clone``, ``contiguous``.
//   • Arithmetic        — ``add``, ``sub``, ``mul``, ``div``, ``pow``, …
//   • Elementwise unary — ``exp``, ``log``, ``sqrt``, ``abs``, ``neg``, trig,
//                          hyperbolic, ``erf``, ``erfinv``, …
//   • Activations       — ``relu``, ``gelu`` (+ exact + backward), ``silu``,
//                          ``elu``, ``selu``, ``mish``, ``softplus``,
//                          ``hard_sigmoid``, ``hard_swish``, ``relu6``, …
//   • Reductions        — ``reduce_sum``, ``reduce_mean``, ``variance``,
//                          ``reduce_max``, ``reduce_min``, ``cumsum``/``cumprod``/
//                          ``cummax``/``cummin``.
//   • Softmax           — ``softmax``, ``log_softmax`` (+ backwards).
//   • Indexing          — ``gather``, ``scatter_add``, ``masked_select``,
//                          ``masked_fill``, ``where_branch``, …
//   • Shape ops         — ``reshape``, ``concatenate``, ``stack``, ``roll``,
//                          ``slice_axis``, ``insert_axis_slice``, ``pad``, …
//   • Sorting           — ``sort``, ``argsort``, ``arg_reduce_index``.
//   • Linear algebra    — ``matmul``, ``linear``, ``linalg_cholesky``,
//                          ``linalg_inv``, ``linalg_solve``, ``linalg_svd``,
//                          ``linalg_norm``, ``linalg_det``, ``linalg_pinv``,
//                          ``linalg_solve_triangular``, ``linalg_lu_solve``,
//                          ``linalg_householder_product``.
//   • Normalisation     — LayerNorm + BatchNorm (forward / backward), MLX.
//   • Convolutions      — Conv1d/2d/3d forward + backward (NHWC permutation).
//   • Loss              — ``mse_loss``, ``huber_loss``, ``bce_loss``,
//                          ``bce_with_logits_loss``, ``cross_entropy_backward``,
//                          ``ctc_loss_forward``.
//   • Misc              — ``embedding_bag_forward``, ``nn_fold``, ``astype``,
//                          ``cast``, ``pad``, ``pow_scalar``, ``clip``, etc.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <mlx/array.h>
#include <mlx/compile.h>
#include <mlx/fast.h>
#include <mlx/linalg.h>
#include <mlx/ops.h>
#include <mlx/random.h>
#include <mlx/transforms.h>  // mlx::core::vjp (SDPA backward)

#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/MemoryStats.h"
#include "../../core/Shape.h"
#include "../Dispatcher.h"
#include "../IBackend.h"
#include "AxisIndex.h"
#include "ComplexParts.h"
#include "Conversion.h"
#include "DivMod.h"
#include "HalfAccumulation.h"
#include "Interpolate.h"
#include "LinalgStatus.h"
#include "LogSoftmax.h"
#include "Lstm.h"
#include "MetalAllocator.h"
#include "MetalKernelRunner.h"
#include "MlxBridge.h"
#include "Pow.h"
#include "SdpaHeadDim.h"
#include "SortOrder.h"

namespace lucid {
namespace backend {

// MLX refuses max, min, argmax and argmin of an empty array — "Cannot max
// reduce zero size array" — even along axes that are not empty, where the
// answer is as empty as the input: the max over channels of a batch of zero
// images.  The sum has the same result shape and holds no values, so it
// stands in.  (A zero-length *reduced* axis has no answer at all; the op
// refuses that before it reaches a backend.)
inline ::mlx::core::array empty_extreme(const ::mlx::core::array& x,
                                        const std::vector<int>& axes,
                                        bool keepdims,
                                        ::mlx::core::Dtype dtype) {
    return ::mlx::core::astype(::mlx::core::sum(x, axes, keepdims), dtype);
}

// ``eye`` that takes an empty side.  MLX refuses ``eye(0, n)`` — "N and M
// must be positive integers" — where the identity of an empty matrix is just
// an empty matrix, and ``matrix_exp`` of a 0 x 0 input reached it.
inline ::mlx::core::array eye_or_empty(int n, int m, int k, ::mlx::core::Dtype dtype) {
    if (n == 0 || m == 0)
        return ::mlx::core::zeros({n, m}, dtype);
    return ::mlx::core::eye(n, m, k, dtype);
}

// Concrete GPU backend routing every Lucid op through Apple's MLX library.
//
// Inherits :class:`IBackend` and overrides each per-op virtual to call
// the matching ``mlx::core::*`` primitive (broadcast, matmul, reduce,
// activations, …).  Results are MLX lazy arrays — ``mx::eval()`` is
// invoked when the engine needs the concrete buffer (e.g. for
// :class:`MetalKernelRunner` or :func:`download_gpu_to_cpu`).  Stream
// tagging uses :class:`Dispatcher`'s default GPU stream unless an op
// requests another (linalg ops dispatch to a CPU MLX stream per the
// H3 carve-out, since MLX's eigh / svd / qr run on CPU).
//
// Attributes
// ----------
// device() : Device
//     Always returns ``Device::GPU``.
//
// Notes
// -----
// - CPU and GPU backends are static singletons registered with the
//   :class:`Dispatcher` at process start (see :struct:`GpuBackendRegistrar`).
// - All Storage values passed in MUST hold ``GpuStorage`` — the
//   Dispatcher guarantees this by routing Device::GPU tensors here.
// - MLX uses lazy evaluation; operations build a graph but defer
//   execution until ``eval()`` or a CPU read forces it.
// - ``Dtype::F64`` is unsupported on the GPU (Metal limitation); callers
//   must cast to F32 or keep the tensor on the CPU.
// - Convolutions internally permute NCHW ↔ NHWC because MLX expects
//   channels-last; see ``gpu_nchw_to_nhwc_perm`` family of helpers.
//
// Examples
// --------
// Construction is automatic via static registration::
//
//     // BackendInit.cpp triggers GpuBackendRegistrar at process start.
//     auto t = lucid::tensor({1,2,3}, lucid::Device::GPU);
//     // → routed to GpuBackend::from_cpu → mlx::core::copy.
//
// See Also
// --------
// :class:`CpuBackend` — Accelerate-backed sibling.
// :class:`MlxBridge` — Storage ↔ mlx::core::array conversion.
// :struct:`GpuBackendRegistrar` — static-init registration.
class GpuBackend final : public IBackend {
public:
    // Construct + register a fresh :class:`GpuBackend` singleton with the Dispatcher.
    //
    // Idempotent only insofar as the Dispatcher replaces any prior
    // registration for ``Device::GPU``.  Normally invoked exactly once
    // by :struct:`GpuBackendRegistrar` at process start.
    static void register_self() {
        Dispatcher::register_backend(Device::GPU, std::make_unique<GpuBackend>());
    }

    // Return the device tag this backend services.
    //
    // Returns
    // -------
    // Device
    //     Always ``Device::GPU``.
    Device device() const noexcept override { return Device::GPU; }

    // ── Construction / cross-device ────────────────────────────────────────
    // Upload a CPU storage to GPU-private memory.
    //
    // Delegates to :func:`gpu::upload_cpu_to_gpu`, which uses
    // ``mlx::core::copy`` to obtain a fresh GPU-private allocation
    // (the raw CPU pointer cannot be aliased on Metal).
    //
    // Parameters
    // ----------
    // cpu : CpuStorage
    //     The source CPU buffer (taken by value so the caller can move).
    // shape : const Shape&
    //     Logical shape of the tensor.
    //
    // Returns
    // -------
    // Storage
    //     A ``GpuStorage`` participating in MemoryTracker accounting.
    Storage from_cpu(CpuStorage cpu, const Shape& shape) override;

    // Allocate a zero-initialised tensor on the GPU.  Delegates to ``mlx::core::zeros``.
    Storage zeros(const Shape& shape, Dtype dt) override;

    // Allocate a ones-initialised tensor on the GPU.  Delegates to ``mlx::core::ones``.
    Storage ones(const Shape& shape, Dtype dt) override;

    // On-device {0,1} Bernoulli keep-mask at rate ``keep_prob`` (see
    // :func:`IBackend::bernoulli_mask`).  Builds an MLX PRNG key from
    // ``key_seed`` (pulled from the framework Generator, so masks are
    // reproducible from the global seed) and fills the whole mask in one kernel
    // via ``mlx::core::random::bernoulli`` — replacing the per-element CPU
    // Philox loop + host->device upload that made Dropout the dominant cost in
    // training-mode forward passes (~19 ms -> <1 ms for a (32,128,768) mask).
    Storage
    bernoulli_mask(double keep_prob, const Shape& shape, Dtype dt, std::uint64_t key_seed) override;

    // On-device random fills via mlx::core::random (see :func:`IBackend`).
    // Each builds a PRNG key from ``key_seed`` and fills in one kernel.
    Storage random_uniform(
        const Shape& shape, double lo, double hi, Dtype dt, std::uint64_t key_seed) override;

    Storage random_normal(
        const Shape& shape, double mean, double std, Dtype dt, std::uint64_t key_seed) override;

    Storage
    random_bernoulli(const Shape& shape, double p, Dtype dt, std::uint64_t key_seed) override;

    Storage random_randint(const Shape& shape,
                           std::int64_t low,
                           std::int64_t high,
                           Dtype dt,
                           std::uint64_t key_seed) override;

    // Deep-copy a tensor into a fresh contiguous MLX array.
    Storage clone(const Storage& src, const Shape&, Dtype dt) override;

    // Materialise a non-contiguous view into a contiguous tensor.
    //
    // Identical to :meth:`clone` on this backend — MLX's
    // ``contiguous`` already handles stride packing internally.
    Storage contiguous(const Storage& src,
                       const Shape& shape,
                       const Stride&,
                       std::size_t,
                       bool,
                       Dtype dt) override;

    // ── Arithmetic ─────────────────────────────────────────────────────────
    // Elementwise add with broadcast.  Delegates to ``mlx::core::add``.
    Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    Storage bitwise_binary(
        const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) override;

    Storage
    compare_binary(const Storage& a, const Storage& b, const Shape& shape, Dtype, int op) override;

    // Elementwise max(a, b) with broadcast.
    Storage maximum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    // Elementwise min(a, b) with broadcast.
    Storage minimum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override;

    // ── Elementwise unary ─────────────────────────────────────────────────
    // Elementwise exp(x).  Each unary op below delegates to the matching
    // ``mlx::core::*`` primitive through the :func:`mlx_unary` helper.
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

    // ── Complex viewing (mlx native) ──────────────────────────────────────

    Storage complex_real(const Storage& a, const Shape&) override;

    Storage complex_imag(const Storage& a, const Shape&) override;

    Storage complex_combine(const Storage& re, const Storage& im, const Shape&) override;

    Storage complex_conj(const Storage& a, const Shape&, Dtype dt) override;

    // ── Activations ────────────────────────────────────────────────────────
    // SiLU / Swish forward.
    //
    // Math
    // ----
    // $y = x \cdot \sigma(x)$ where $\sigma$ is the logistic sigmoid.
    //
    // See Also
    // --------
    // :meth:`silu_backward`.
    Storage silu(const Storage& a, const Shape&, Dtype dt) override;

    // SiLU / Swish backward.  Single fused MLX expression.
    //
    // Math
    // ----
    // $\partial y/\partial x = \sigma(x) \, (1 + x \, (1 - \sigma(x)))$.
    Storage silu_backward(const Storage& a, const Storage& grad, const Shape&, Dtype dt) override;

    // GELU forward (tanh approximation).
    //
    // Math
    // ----
    // $y = 0.5 \, x \, (1 + \tanh[c_1 \, (x + c_2 x^3)])$ where
    // $c_1 = \sqrt{2/\pi}$, $c_2 = 0.044715$.
    //
    // Notes
    // -----
    // 7-op MLX composition (tanh approximation).
    //
    // See Also
    // --------
    // :meth:`gelu_backward`, :meth:`gelu_exact`.
    Storage gelu(const Storage& a, const Shape&, Dtype dt) override;

    // GELU exact (Gaussian-CDF) forward.
    //
    // Math
    // ----
    // $y = 0.5 \, x \, (1 + \operatorname{erf}(x / \sqrt{2}))$.  Composed via
    // ``mlx::core::erf``.
    Storage gelu_exact(const Storage& a, const Shape&, Dtype dt) override;

    Storage gelu_exact_backward(const Storage& a,
                                const Storage& grad,
                                const Shape& shape,
                                Dtype dt) override;

    Storage gelu_backward(const Storage& a, const Storage& grad, const Shape&, Dtype dt) override;

    Storage leaky_relu(const Storage& a, const Shape& shape, Dtype dt, double slope) override;

    Storage softplus(const Storage& a, const Shape&, Dtype dt) override;

    Storage elu(const Storage& a, const Shape& shape, Dtype dt, double alpha) override;

    Storage elu_backward(
        const Storage& a, const Storage& grad, const Shape& shape, Dtype dt, double alpha) override;

    Storage selu(const Storage& a, const Shape&, Dtype dt) override;

    Storage selu_backward(const Storage& a, const Storage& grad, const Shape&, Dtype dt) override;

    Storage mish(const Storage& a, const Shape&, Dtype dt) override;

    Storage mish_backward(const Storage& a, const Storage& grad, const Shape&, Dtype dt) override;

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

    // ── Reductions / boolean predicates ───────────────────────────────────
    // Logical-any over all elements; returns a 0-D bool tensor.
    Storage any(const Storage& a, const Shape&, Dtype dt) override;

    Storage all(const Storage& a, const Shape&, Dtype dt) override;

    // MLX's ``isinf`` and ``isfinite`` judge a complex number as a whole —
    // ``isinf`` is true only for (inf + 0j), so (1 + inf j) and (-inf + 1j)
    // read as finite — where the reference takes a complex number as
    // infinite when either part is.  A complex input is split into its parts.
    // (MLX's ``isnan`` already looks at both parts.)
    static bool is_complex_array(const ::mlx::core::array& x) {
        return x.dtype() == ::mlx::core::complex64;
    }

    Storage isinf(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage isnan(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage isfinite(const Storage& a, const Shape& shape, Dtype dt) override;

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

    // Population variance.  A half input is accumulated in float32 (see
    // ``var_widened``): MLX's ``var`` takes a ``mean`` of its own, so a
    // float16 input overflowed the same way ``reduce_mean`` did.
    Storage variance(const Storage& a, const Shape&, const ReduceOpts& opts, Dtype dt) override;

    Storage
    reduce_max(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override;

    Storage
    reduce_min(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override;

    // PERF: cumsum / cumprod / cummax / cummin all produce contiguous output
    // from MLX; the defensive contiguous() wraps below were forcing
    // re-materialization and breaking the lazy-graph fusion chain.
    //
    // MLX's scan carries its running value in the output dtype, so a half
    // input is scanned in float32 (see ``widen_half``): a float16 cumsum of
    // 20000 x 0.1 ended at 1999 and a cumprod of 1000 x 1.01 at 15528,
    // where the reference gives 2000 and 16624.
    Storage cumsum(const Storage& a, const Shape&, int axis, Dtype dt) override;

    Storage cumprod(const Storage& a, const Shape&, int axis, Dtype dt) override;

    // Once a NaN appears in the scan it is the running extreme, and every
    // later element inherits it — the same rule ``max`` follows.  MLX's
    // cumulative scans skip it instead, so the CPU (fixed to propagate) and
    // Metal disagreed from the NaN onwards.
    //
    // MLX has no flag for it, so it is composed: a running max over the NaN
    // mask says "a NaN has been seen at or before here", and those
    // positions are overwritten.
    static ::mlx::core::array
    poison_after_nan(const ::mlx::core::array& input, const ::mlx::core::array& scanned, int axis);

    Storage cummax(const Storage& a, const Shape&, int axis, Dtype dt) override;

    Storage cummin(const Storage& a, const Shape&, int axis, Dtype dt) override;

    // PERF: softmax + log_softmax produce contiguous output; their backwards
    // are pure elementwise composition.  Dropping the contiguous() wraps lets
    // downstream loss ops (cross_entropy_loss is the typical consumer) fuse
    // into a single MLX kernel.
    // ── Softmax family ────────────────────────────────────────────────────
    // Softmax along ``axis``.  Delegates to ``mlx::core::softmax``.
    //
    // Math
    // ----
    // $y_i = e^{x_i} / \sum_j e^{x_j}$ along the chosen axis.
    Storage softmax(const Storage& a, const Shape&, int axis, Dtype dt) override;

    // Softmax backward, given the saved forward output ``z``.
    // Composes ``z * (grad - sum(z * grad, axis))`` via MLX.
    //
    // Math
    // ----
    // $\partial L / \partial x_i = z_i \, (g_i - \sum_j z_j g_j)$.
    Storage softmax_backward(const Storage& z,
                             const Storage& grad_out,
                             const Shape& shape_param,
                             int axis,
                             Dtype dt) override;

    Storage log_softmax_backward(
        const Storage& y, const Storage& grad_out, const Shape&, int axis, Dtype dt) override;

    // ``x - logsumexp(x)``, NaN across a row whose maximum is infinite —
    // see LogSoftmax.h.
    Storage log_softmax(const Storage& a, const Shape&, int axis, Dtype dt) override;

    Storage reverse_along_axis(const Storage& a, const Shape& shape, int axis, Dtype dt) override;

    Storage trace(const Storage& a, const Shape&, Dtype dt) override;

    Storage trace_backward(const Storage& grad_out, const Shape& input_shape, Dtype dt) override;

    std::vector<Storage>
    meshgrid(const std::vector<Storage>& xs, const Shape&, Dtype dt, bool indexing_xy) override;

    Storage where_branch(const Storage& grad,
                         const Storage& cond,
                         const Shape& shape,
                         Dtype dt,
                         bool true_branch) override;

    Storage masked_fill(
        const Storage& a, const Storage& mask, const Shape&, Dtype dt, double value) override;

    // ── Indexing / scatter ────────────────────────────────────────────────
    // Gather values from ``a`` along ``dim`` using ``indices``.
    //
    // Math
    // ----
    // $\mathrm{out}[i_1, \ldots, i_d, \ldots, i_n] = a[i_1, \ldots, \mathrm{indices}[\ldots], \ldots, i_n]$.
    Storage gather(const Storage& a,
                   const Storage& indices,
                   const Shape&,
                   const Shape&,
                   int axis,
                   Dtype,
                   Dtype dt) override;

    Storage gather_backward(const Storage& grad,
                            const Storage& indices,
                            const Shape& input_shape,
                            const Shape&,
                            int axis,
                            Dtype,
                            Dtype dt) override;

    Storage
    diagonal(const Storage& a, const Shape&, int offset, int axis1, int axis2, Dtype dt) override;

    Storage diagonal_backward(const Storage& grad,
                              const Shape& input_shape,
                              const Shape& output_shape,
                              int offset,
                              int axis1,
                              int axis2,
                              Dtype dt) override;

    Storage roll(const Storage& a,
                 const Shape&,
                 Dtype dt,
                 const std::vector<std::int64_t>& shifts,
                 const std::vector<int>& axes) override;

    Storage reshape(const Storage& a, const Shape&, const Shape& dst_shape, Dtype dt) override;

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
                        const std::vector<Shape>&,
                        int axis,
                        Dtype dt) override;

    Storage stack(const std::vector<Storage>& xs, const Shape&, int axis, Dtype dt) override;

    std::vector<Storage> split_equal(
        const Storage& a, const Shape&, int axis, std::int64_t num_splits, Dtype dt) override;

    std::vector<Storage> split_at(const Storage& a,
                                  const Shape&,
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
                          const Shape&,
                          const std::vector<std::int64_t>& reps,
                          Dtype dt) override;

    std::pair<Storage, Storage> sort_select(const Storage& a,
                                            const Shape&,
                                            const Shape& output_shape,
                                            int axis,
                                            Dtype dt,
                                            bool descending) override;

    Storage argsort(const Storage& a, const Shape&, int axis, Dtype) override;

    Storage arg_reduce_index(
        const Storage& a, const Shape&, int axis, bool keepdims, Dtype dt, bool is_min) override;

    Storage scatter_add_axis(const Storage& grad,
                             const Storage& indices,
                             const Shape& output_shape,
                             const Shape&,
                             int axis,
                             Dtype dt) override;

    // Scatter-add: ``base[indices] += src`` (out-of-place), along ``dim``:
    //   out[..., idx[..., j, ...], ...] += src[..., j, ...]
    //
    // ``mlx::core::scatter_add_axis`` is that primitive exactly.  MLX's
    // multi-axis ``scatter_add`` (np.add.at flavour: one index array per
    // axis) is a different operation and tripped its own "Number of index
    // arrays does not match number of axes" guard.  An out-of-range index
    // drops its update (policy B, ``gpu_axis_index``).
    Storage scatter_add(const Storage& base,
                        const Storage& indices,
                        const Storage& src,
                        const Shape& base_shape,
                        const Shape& idx_shape,
                        int dim,
                        Dtype dt) override;

    // Axis-scatter overwrite — out[..., idx[..., j, ...], ...] = src[..., j, ...]
    // along dim.  ``mlx::core::put_along_axis`` is the exact set-mode analogue
    // of ``scatter_add_axis`` (single kernel); ``x[key] = v``, ``scatter`` and
    // ``index_copy`` route through it.  A position named twice takes one of
    // the values aimed at it, which one unspecified — the GPU threads race.
    //
    // An out-of-range index drops its write (policy B, ``gpu_axis_index``).
    // It is aimed at a one-wide sink slab past the end of ``dim`` that is cut
    // off again (``gpu_sink_index``), because no in-range position can take
    // a write and stay as it was.
    //
    // MLX scatters no 8-byte element on Metal: int64 and complex64 both
    // report "[ScatterAxis::eval_gpu] Does not support".  Writing each as
    // two uint32 halves would be exact for distinct indices, but two writers
    // racing for one position can land one's low half beside the other's
    // high half — a value nobody wrote.  So those two go through the CPU,
    // sink slab and all.
    Storage scatter_set(const Storage& base,
                        const Storage& indices,
                        const Storage& src,
                        const Shape& base_shape,
                        const Shape& idx_shape,
                        int dim,
                        Dtype dt) override;

    // MLX's Metal scatter has no int64 kernel (nor complex64 — see scatter_set).
    //
    // ``[ScatterAxis::eval_gpu] Does not support int64`` — the *values*,
    // not the indices, which it takes at either width.  There is no exact
    // way around it on-device: float32 loses everything above 2^24, and
    // splitting the word into two int32 halves works for a set but not
    // for an add, where the carry between halves is exactly what is lost.
    //
    // So int64 scatter runs on the CPU and comes back.  It is a real cost
    // and it is stated here rather than hidden: the alternative was
    // ``lucid.scatter`` and ``index_copy`` raising on Metal for a dtype
    // the CPU handles, which is the device-dependent behaviour this
    // backend pair exists to avoid.  Same shape as the LAPACK pivot
    // round-trips further up this file.
    template <typename Fn>
    Storage scatter_via_cpu(const Storage& base,
                            const Storage& indices,
                            const Storage& src,
                            const Shape& base_shape,
                            const Shape& idx_shape,
                            Fn&& run);

private:
    // MLX's ``scatter_{max,min,prod}`` have no axis-only variant the way
    // ``scatter_add`` has ``scatter_add_axis``.  So the axis scatter
    // ``a[..., idx[i,j,...], ...] op= src[i,j,...]`` along ``dim`` uses the
    // multi-axis form with K = ndim: one coordinate array per axis.  For
    // every axis but ``dim`` the coordinate is ``arange(idx_shape[a])``
    // reshaped to (1..N..1).  For ``dim`` it is the index, made safe by
    // ``gpu_axis_index``.  Updates then have the index's shape, which meets
    // MLX's "updates.ndim == K + (ndim - K)" invariant.
    //
    // An out-of-range index drops its update (policy B): the update becomes
    // the identity of ``op``.
    template <typename ScatterFn>
    Storage axis_scatter_via_multiaxis(const Storage& base,
                                       const Storage& indices,
                                       const Storage& src,
                                       const Shape& base_shape,
                                       const Shape& idx_shape,
                                       int dim,
                                       Dtype dt,
                                       ScatterReduce op,
                                       ScatterFn scatter_fn);

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

    // Sliding-window unfold along ``dim`` producing rank+1 windows of ``size``.
    //
    // Math
    // ----
    // Output dim ``L = (dim_size - size) / step + 1``.  Indices built
    // as ``starts[l] = l * step``, ``offsets[s] = s``, then gathered.
    Storage unfold_dim(
        const Storage& a, const Shape& in_shape, int dim, int size, int step, Dtype dt) override;

    // ── Linear algebra ────────────────────────────────────────────────────
    // General matrix multiply with optional broadcasted batch dims.
    //
    // Parameters
    // ----------
    // a, b : const Storage&
    //     GPU tensors of compatible shapes.  Either side may be
    //     transposed via ``opts``.
    // opts : const MatmulOpts&
    //     Transpose / batched flags (see :struct:`MatmulOpts`).
    // dt : Dtype
    //     Output dtype.  Must be a real / complex floating type.
    //
    // Returns
    // -------
    // Storage
    //     Result of ``a @ b`` (possibly batched), wrapped as
    //     ``GpuStorage``.
    //
    // Notes
    // -----
    // Calls ``mlx::core::matmul``.  Uses Metal's BLAS-equivalent
    // tiled kernel internally; for very small matrices the dispatch
    // overhead may dominate.
    Storage matmul(const Storage& a, const Storage& b, const MatmulOpts& opts, Dtype dt) override;

    // Fused affine: ``y = x @ W^T + b`` (the canonical ``nn.Linear`` op).
    //
    // Math
    // ----
    // $y_{i,j} = \sum_k x_{i,k} W_{j,k} + b_j$.
    //
    // Notes
    // -----
    // Uses ``mlx::core::addmm`` when ``b`` is present so the multiply +
    // add fuse inside the MLX kernel; falls back to a separate ``matmul``
    // when ``b`` is null.  Deferred contiguous wrap matches the conv2d
    // path (cf perf-conv2d-deferred-contig).
    Storage linear(const Storage& x,
                   const Storage& weight,
                   const Storage& bias,
                   const Shape&,
                   const Shape&,
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
                                 const Shape& x_shape,
                                 Dtype dt) override;

    StoragePair rms_norm_backward(const Storage& x,
                                  const Storage& gamma,
                                  const Storage& saved_rstd,
                                  const Storage& grad,
                                  std::size_t outer,
                                  std::size_t normalized_size,
                                  const Shape& x_shape,
                                  const Shape& gamma_shape,
                                  Dtype dt) override;

    // ── Normalisation ─────────────────────────────────────────────────────
    // LayerNorm forward via MLX's fused ``fast::layer_norm`` kernel.
    //
    // Math
    // ----
    // $$\mu = \mathbb{E}[x], \quad \mathrm{rstd} = (\operatorname{Var}[x] + \varepsilon)^{-1/2}$$
    // $$y = \gamma \, (x - \mu) \, \mathrm{rstd} + \beta$$
    // reducing along the trailing axis of size ``normalized_size``.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     ``{y, saved_mean, saved_rstd}`` — the activation plus saved
    //     stats needed by :meth:`layer_norm_backward`.
    //
    // Notes
    // -----
    // Replaces the 7-op MLX chain (mean → subtract → square → mean →
    // add+rsqrt → multiply → multiply+add) with one fused kernel.
    // Mac Studio Llama-scale forward: 2.4 ms → ~0.6 ms.
    std::vector<Storage> layer_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            std::size_t outer,
                                            std::size_t normalized_size,
                                            double eps,
                                            const Shape& x_shape,
                                            Dtype dt) override;

    // LayerNorm backward.  Composes the gradient as a single MLX
    // expression using the saved mean / rstd.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     ``{dx, dgamma, dbeta}``.
    std::vector<Storage> layer_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& grad,
                                             std::size_t outer,
                                             std::size_t normalized_size,
                                             const Shape& x_shape,
                                             const Shape& gamma_shape,
                                             const Shape& beta_shape,
                                             Dtype dt) override;

    // BatchNorm forward (training mode).
    //
    // Math
    // ----
    // Per-channel reduction over batch + spatial dims:
    // $\mu_c = \mathbb{E}[x_{n,c,\ldots}]$,
    // $\mathrm{rstd}_c = (\operatorname{Var}[x_{n,c,\ldots}] + \varepsilon)^{-1/2}$,
    // $y = \gamma (x - \mu) \mathrm{rstd} + \beta$.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     ``{y, saved_mean, saved_rstd}``.
    //
    // Notes
    // -----
    // Hand-rolled MLX reductions over channel axes.
    std::vector<Storage> batch_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            int,
                                            int channels,
                                            int,
                                            int ndim,
                                            double eps,
                                            const Shape& x_shape_param,
                                            Dtype dt) override;

    // BatchNorm backward (training mode).
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     ``{dx, dgamma, dbeta}``.  MLX composite over channel axes.
    std::vector<Storage> batch_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& saved_xnorm,
                                             const Storage& grad,
                                             int,
                                             int channels,
                                             int,
                                             int ndim,
                                             const Shape& x_shape_param,
                                             Dtype dt,
                                             double eps) override;

    std::vector<Storage> group_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            int batch,
                                            int channels,
                                            int,
                                            int groups,
                                            const std::vector<int>& spatial_dims,
                                            double eps,
                                            const Shape& x_shape,
                                            Dtype dt) override;

    std::vector<Storage> group_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& grad,
                                             int batch,
                                             int channels,
                                             int,
                                             int groups,
                                             const std::vector<int>& spatial_dims,
                                             const Shape& x_shape,
                                             Dtype dt) override;

    // Matrix / vector norm.  Routed to MLX's CPU linalg stream per H3.
    //
    // Math
    // ----
    // Generic ``ord``-norm along ``axes``.  For ``ord=2`` and a matrix
    // axis pair this is the Frobenius / spectral norm depending on
    // mode; for a vector axis it is the L2 norm.
    //
    // Notes
    // -----
    // MLX implements ``linalg::norm`` on the CPU stream, so the result
    // physically lives in unified memory but the compute happens on
    // the CPU (H3 carve-out).
    Storage linalg_norm(const Storage& a,
                        const Shape&,
                        double ord,
                        const std::vector<int>& axes,
                        bool keepdims,
                        Dtype dt) override;

    // Cholesky factorisation $A = L L^T$ (or $U^T U$ if ``upper``), on
    // MLX's CPU linalg stream.
    //
    // MLX reports no failure.  Its *lower* factor keeps LAPACK's failed
    // pivot — the first diagonal entry that is not positive, or is NaN, at
    // the step ``potrf`` stopped — so ``info`` is read there.  Its upper
    // factor does not: ``[[1, 2], [2, 1]]`` came back ``[[1, 2], [0, 1]]``
    // (LCD-206).  So ``U`` of ``A`` is ``L`` of ``A^T``, transposed: the
    // same triangle read, one check for both.
    Storage linalg_cholesky(const Storage& a, const Shape&, bool upper, Dtype dt) override;

    // Dense matrix inverse $A^{-1}$ via the MLX CPU linalg stream.
    //
    // Built from MLX's LU factors rather than ``linalg::inv``, which throws
    // from the CPU stream's worker thread on a singular matrix and so
    // aborts the process — see ``gpu_lu_inverse``.  A singular input raises
    // the same ``LinAlgError`` the CPU stream does.
    Storage linalg_inv(const Storage& a, const Shape&, Dtype dt) override;

    // Solve the linear system $A x = b$ for ``x``.  CPU linalg stream.
    //
    // ``x = A^{-1} b`` with the inverse from ``gpu_lu_inverse``:
    // ``linalg::solve`` inverts the triangular factors with the same
    // throwing routine, so a singular ``A`` aborted the process here too.
    // MLX's own solve is a product with explicit triangular inverses, so
    // this is the same arithmetic in a different order, not a weaker one.
    Storage linalg_solve(const Storage& a,
                         const Storage& b,
                         const Shape& a_shape,
                         const Shape& b_shape,
                         Dtype dt) override;

    Storage linalg_matrix_power(const Storage& a, const Shape& shape, int power, Dtype dt) override;

    // Moore-Penrose pseudo-inverse: the CPU backend's ``gesdd`` SVD, uploaded.
    // Delegating keeps ``CpuBackend::pinv_one`` the one owner of the SVD driver
    // and non-finite policy (NaN vs single/multiple infinities).
    Storage linalg_pinv(const Storage& a, const Shape& shape, Dtype dt) override;

    Storage linalg_det(const Storage& a, const Shape& shape, Dtype dt) override;

    StoragePair
    linalg_qr(const Storage& a, const Shape&, const Shape&, const Shape&, Dtype dt) override;

    StoragePair
    linalg_eig(const Storage& a, const Shape&, const Shape&, const Shape&, Dtype dt) override;

    StoragePair linalg_eigh(const Storage& a,
                            const Shape&,
                            const Shape&,
                            const Shape&,
                            Dtype dt,
                            bool upper = false) override;

    std::vector<Storage> linalg_svd(const Storage& a,
                                    const Shape&,
                                    bool compute_uv,
                                    const Shape& u_shape,
                                    const Shape&,
                                    const Shape& vt_shape,
                                    Dtype dt) override;

    // lu_factor: MLX's ``lu`` returns separate (P, L, U), not LAPACK's packed
    // factor and pivots, so the CPU backend factorises and the result is
    // uploaded.  Delegating — rather than running ``getrf`` again here —
    // keeps the CPU backend the one owner of the ``info`` policy.
    std::vector<Storage> linalg_lu_factor(const Storage& a, const Shape& shape, Dtype dt) override;

    // solve_triangular: the CPU backend's substitution, uploaded.  MLX's own
    // ``linalg::solve_triangular`` goes through ``tri_inv``, whose LAPACK
    // ``trtri`` throws from the CPU stream's worker thread on a zero diagonal
    // — a process abort, not an exception (see the note on MLX CPU-stream
    // linalg) — where the reference returns inf/NaN.  Delegating keeps the
    // CPU backend the one owner of the singular-triangle policy.
    Storage linalg_solve_triangular(const Storage& a,
                                    const Storage& b,
                                    const Shape& a_shape,
                                    const Shape& b_shape,
                                    bool upper,
                                    bool unitriangular,
                                    Dtype dt) override;

    // lstsq: fall back to CPU LAPACK (MLX has no lstsq)
    std::vector<Storage> linalg_lstsq(const Storage& a,
                                      const Storage& b,
                                      const Shape& a_shape,
                                      const Shape& b_shape,
                                      Dtype dt) override;

    // lu_solve: fall back to CPU LAPACK
    Storage linalg_lu_solve(const Storage& LU,
                            const Storage& pivots,
                            const Storage& b,
                            const Shape& lu_shape,
                            const Shape& b_shape,
                            Dtype dt) override;

    // householder_product: fall back to CPU LAPACK
    Storage linalg_householder_product(const Storage& H,
                                       const Storage& tau,
                                       const Shape& h_shape,
                                       Dtype dt) override;

    // ldl_factor: the CPU backend's ``sytrf``, uploaded (MLX has no LDLᵀ).
    std::vector<Storage> linalg_ldl_factor(const Storage& a, const Shape& shape, Dtype dt) override;

    // fold (col2im) — Metal-native via ``scatter_add_axis``.
    //
    // Algorithm: precompute on CPU a length-(kH*kW*L) table of flat output
    // positions (oh*outW + ow) for every (ki, kj, l_idx) triple plus a
    // matching validity mask.  Upload both as small MLX arrays.  Then a
    // single ``scatter_add_axis`` call accumulates the (N, C, kH*kW, L)
    // input slices into the (N, C, outH*outW) output buffer along axis 2.
    // Invalid (out-of-bounds) source elements are zeroed via the mask, so
    // they scatter +0 to position 0 and have no effect on the result.
    // ``nn.Fold`` — reverse of ``nn.Unfold``: scatters unfolded patches back into a feature map.
    //
    // Notes
    // -----
    // Built atop MLX's ``scatter_add`` because patches may overlap and
    // must accumulate.
    Storage nn_fold(const Storage& x,
                    const Shape& x_shape,
                    const Shape& out_shape,
                    const std::vector<int>& kernel_size,
                    const std::vector<int>& stride,
                    const std::vector<int>& padding,
                    const std::vector<int>& dilation,
                    Dtype dt) override;

    // ---------------------------------------------------------------------
    // LSTM — single layer, single direction; the MLX graphs live in
    // ``gpu/Lstm.h``.  The 16-bit formats run in float32 there and round
    // back here.  Other configurations fall through to ``IBackend::lstm_*``
    // (not_implemented); the LSTM node refuses them before that.
    // ---------------------------------------------------------------------

private:
    bool
    lstm_metal_supported(const LstmOpts& opts, const std::vector<Storage>& weights, Dtype dt) const;

public:
    std::vector<Storage> lstm_forward_train(const Storage& input,
                                            const Storage& h0,
                                            const Storage& c0,
                                            const std::vector<Storage>& weights,
                                            const LstmOpts& opts,
                                            Dtype dt) override;

    std::vector<Storage> lstm_forward(const Storage& input,
                                      const Storage& h0,
                                      const Storage& c0,
                                      const std::vector<Storage>& weights,
                                      const LstmOpts& opts,
                                      const Shape& out_shape,
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

    // embedding_bag: use MLX gather + reduce
    // EmbeddingBag forward — gather + per-bag reduction in one op.
    //
    // Notes
    // -----
    // Reduction mode (sum / mean / max) is encoded in the surrounding
    // op-spec; this routine only handles the gather + reduction kernel
    // dispatch.
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

    // How many offsets a buffer holds — the bag count, or one more under
    // include_last_offset, whose final offset only ends the last bag.
    static std::int64_t offsets_count(const Storage& offsets);

    // ── astype — the conversion rules live in ``gpu/Conversion.h`` ───────────
    // Convert ``a`` to ``dst_dt``.  Identical to :meth:`cast` on this backend.
    Storage astype(const Storage& a, const Shape& shape, Dtype /*src_dt*/, Dtype dst_dt) override;

    // ── flip ──────────────────────────────────────────────────────────────────
    // MLX has no built-in flip; implement via take with reversed indices per dim.
    Storage
    flip(const Storage& a, const Shape& shape, const std::vector<int>& dims, Dtype dt) override;

    // ── masked_select ─────────────────────────────────────────────────────────
    Storage masked_select_count(const Storage& mask, const Shape& shape, Dtype dt) override;
    Storage masked_select(const Storage& a,
                          const Storage& mask,
                          const Shape& a_shape,
                          const Shape& mask_shape,
                          std::int64_t n_true,
                          Dtype dt) override;

    // ── ctc_loss ──────────────────────────────────────────────────────────────
    // ── Loss functions ────────────────────────────────────────────────────
    // Connectionist Temporal Classification (CTC) loss forward.
    //
    // Notes
    // -----
    // Implements Graves 2006 forward-only loss on the GPU with MLX
    // primitives.  Backward lives separately in the autograd layer.
    Storage ctc_loss_forward(const Storage& log_probs,
                             const Storage& targets,
                             const Storage& input_lengths,
                             const Storage& target_lengths,
                             const Shape& lp_shape,
                             int blank,
                             bool zero_infinity,
                             Dtype dt) override;

    // The backward recursion runs where the forward's does, on the CPU, for
    // the same reason: alpha and beta are a sequential walk over time.
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
    repeat(const Storage& a, const Shape&, Dtype dt, std::int64_t repeats, int axis) override;

    Storage
    tile(const Storage& a, const Shape&, Dtype dt, const std::vector<std::int64_t>& reps) override;

    Storage
    permute(const Storage& a, const Shape&, const std::vector<int>& perm, Dtype dt) override;

    Storage pad(const Storage& a,
                const Shape&,
                Dtype dt,
                const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                double constant) override;

    Storage pow_scalar(const Storage& a, const Shape&, Dtype dt, double exp) override;

    Storage rpow_scalar(const Storage& a, const Shape&, Dtype dt, double base) override;

    Storage clip(const Storage& a, const Shape&, Dtype dt, double min_v, double max_v) override;

    Storage cast(const Storage& a, const Shape&, Dtype, Dtype dst_dt) override {
        const auto& gs = std::get<GpuStorage>(a);
        auto result = gpu_convert(*gs.arr, dst_dt);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dst_dt)};
    }

    // Mean-squared-error loss.
    //
    // Math
    // ----
    // $L = \mathrm{reduce}((x - t)^2)$ with the chosen reduction
    // (``mean`` / ``sum`` / ``none``).
    Storage mse_loss(const Storage& input,
                     const Storage& target,
                     const Shape&,
                     Dtype dt,
                     int reduction) override;

    std::pair<Storage, Storage> mse_loss_backward(const Storage& input,
                                                  const Storage& target,
                                                  const Storage& grad,
                                                  const Shape& shape,
                                                  Dtype dt,
                                                  int reduction) override;

    // Huber loss (smooth L1).
    //
    // Math
    // ----
    // Piecewise: $0.5 (x - t)^2$ for $|x - t| < \delta$,
    // else $\delta \, (|x - t| - 0.5 \delta)$.
    Storage huber_loss(const Storage& input,
                       const Storage& target,
                       const Shape&,
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

    // Binary cross-entropy from probability inputs (not logits).
    //
    // Math
    // ----
    // $L = -(t \log p + (1 - t) \log (1 - p))$ then reduced.
    Storage bce_loss(const Storage& input,
                     const Storage& target,
                     const Storage& weight,
                     const Shape&,
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

    // Binary cross-entropy from logits, numerically stable.
    //
    // Math
    // ----
    // $L = \max(x, 0) - x t + \log(1 + e^{-|x|})$ — the standard
    // log-sum-exp-stable form.
    Storage bce_with_logits_loss(const Storage& input,
                                 const Storage& target,
                                 const Storage& weight,
                                 const Storage& pos_weight,
                                 const Shape&,
                                 const Shape&,
                                 const Shape&,
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
                                              const Shape&,
                                              const Shape& target_shape,
                                              Dtype dt,
                                              double eps,
                                              int ignore_index,
                                              int reduction) override;

    // Cross-entropy backward given the saved softmax output.
    //
    // Math
    // ----
    // $\partial L / \partial x_i = p_i - \delta_{i, \mathrm{target}}$
    // (probability minus one-hot target), then scaled by the loss
    // reduction weight.
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
                                    const Shape&,
                                    const Shape& target_shape,
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

    CpuStorage to_cpu(const Storage& a, const Shape& shape) override;

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
                                       const Shape&,
                                       const Shape&,
                                       const Shape&,
                                       Dtype mask_dtype,
                                       double scale,
                                       bool is_causal,
                                       Dtype dt) override;

    // N-D transposed convolution (fractionally-strided / "deconv") forward.
    //
    // Notes
    // -----
    // Same NHWC permutation strategy as :meth:`conv_nd_forward`.  MLX's
    // ``conv_transpose{1,2,3}d`` take ``dilation`` and ``groups``
    // directly, so both are handed straight to it; only the weight has
    // to be rearranged, since Lucid and MLX disagree about where the
    // group-divided channel axis lives.
    Storage conv_transpose_nd_forward(const Storage& x,
                                      const Storage& W,
                                      const Storage& b,
                                      int,
                                      int Cin,
                                      int Cout,
                                      const int*,
                                      const int*,
                                      const int*,
                                      const IBackend::ConvTransposeNdOpts& opts,
                                      const Shape& out_shape,
                                      Dtype dt) override;

    // N-D transposed convolution backward producing ``{dx, dW, db}``.
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

    Storage rope_backward(const Storage& grad_out,
                          const Storage& saved_cos,
                          const Storage& saved_sin,
                          const Shape& x_shape,
                          bool interleaved,
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

    // ── Convolutions ──────────────────────────────────────────────────────
    // N-D convolution forward (N ∈ {1, 2, 3}).
    //
    // Math
    // ----
    // $$y_{n,c_o,p_1,\ldots,p_N} = b_{c_o} + \sum_{c_i} \sum_{\Delta} W_{c_o,c_i,\Delta} \, x_{n,c_i,p_1 s_1 + \Delta_1,\ldots}$$
    // where $s$ is the stride and the sum runs over the kernel window.
    //
    // Notes
    // -----
    // MLX uses NHWC (channels-last) ordering, so this routine permutes
    // input + weight NCHW → NHWC, dispatches ``mlx::core::conv1d`` /
    // ``conv2d`` / ``conv3d`` based on ``ndim``, then permutes the
    // result back to NCHW.  See ``gpu_nchw_to_nhwc_perm`` family.
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
                            const ConvNdOpts& opts,
                            const Shape& out_shape,
                            Dtype dt) override;

    // N-D convolution backward producing ``{dx, dW, db}``.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     ``{dx, dW, db}`` in that order; ``db`` may be a stub when the
    //     forward had no bias.
    //
    // Notes
    // -----
    // Like the forward, this routes through the NHWC permutation
    // helpers.  Heavy lifting is two extra MLX convolutions (one for
    // each weight slot) plus a reduction for ``db``.
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
                                          const ConvNdOpts& opts,
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
                           const Shape& out_shape,
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

private:
    // MLX linalg operations (eig, eigh, svd, qr, cholesky, inv, solve) only
    // have CPU implementations inside MLX itself.  Dispatching them on the
    // default GPU stream would stall; using Device::cpu tells MLX to schedule
    // them on the CPU dispatch queue while still returning mlx::core::array.
    inline static const ::mlx::core::Device k_linalg_stream{::mlx::core::Device::cpu};

    // ── Keeping MLX's LAPACK failures out of its worker thread ────────────
    //
    // MLX evaluates a CPU-stream linalg primitive on that stream's worker
    // thread, and a failing LAPACK call there is a ``throw`` nothing
    // catches: the process ends in ``std::terminate``, whichever Lucid call
    // happened to force the evaluation.  A try/catch around ``eval()`` does
    // not help — the exception never reaches this thread.  The failures
    // seen are:
    //
    //   Inverse  — ``getrf`` / ``trtri`` meet an exactly zero pivot
    //              (``inv``, ``solve`` and ``tri_inv`` on a singular matrix);
    //   SVD      — ``gesvdx`` rejects a NaN as an illegal argument
    //              (``svd``, ``pinv``, and the norms built on them);
    //   Eig      — ``geev`` rejects a NaN as an illegal argument.
    //
    // So the inputs these routines see are made safe before the graph is
    // built: the pivots are checked before anything is inverted, and a
    // non-finite matrix never reaches the SVD or eigen solver.

    // Per-matrix flag, shaped ``(..., 1, 1)``: true where the matrix holds a
    // NaN or an infinity.
    static ::mlx::core::array gpu_nonfinite_matrices(const ::mlx::core::array& a);

    // ``a`` with every flagged matrix replaced by zeros.
    static ::mlx::core::array gpu_zero_where(const ::mlx::core::array& bad,
                                             const ::mlx::core::array& a);

    // ``a`` with every flagged matrix (or vector) replaced by NaN.
    static ::mlx::core::array gpu_nan_where(const ::mlx::core::array& bad,
                                            const ::mlx::core::array& a);

    // Raise, the way the CPU stream does, when ``a`` holds a NaN or an
    // infinity.  Costs one host round trip.
    static void gpu_require_finite(const ::mlx::core::array& a, const char* op);

    // $A^{-1}$ for a batch of square matrices, from MLX's LU factors.
    //
    // ``linalg::lu`` reports a zero pivot by leaving it in ``U`` rather than
    // by throwing, so the pivots are read back and a singular matrix raises
    // here — the CPU stream's ``LinAlgError``, ``info`` being the 1-based
    // index of the first zero pivot, as LAPACK numbers it.  Only then are
    // the factors inverted, and ``tri_inv`` fails on nothing but an exactly
    // zero diagonal: ``U``'s was just checked, ``L``'s is all ones.  (A
    // NaN pivot is not zero; it propagates into a NaN inverse, which is
    // what LAPACK's own ``getri`` returns for it.)
    //
    // MLX's ``lu`` gives ``A = L[P] U`` with ``P`` a row-index vector, so
    // $A^{-1} = U^{-1} L^{-1} P^T$, whose column ``j`` is column ``P[j]``
    // of $U^{-1} L^{-1}$.
    static ::mlx::core::array gpu_lu_inverse(const ::mlx::core::array& a, const char* op);

    // Builds a permutation for NCHW → NHWC transpose: [0, 2,..,N+1, 1].
    // N is the number of spatial dimensions (1, 2, or 3).
    static std::vector<int> gpu_nchw_to_nhwc_perm(int N);

    // Builds a permutation for NHWC → NCHW transpose: [0, N+1, 1,..,N].
    static std::vector<int> gpu_nhwc_to_nchw_perm(int N);

    static std::vector<int> gpu_w_to_transpose_perm(int N);

    // Lucid stores the transposed weight as ``(Cin, Cout / G, K...)``; MLX
    // wants ``(Cout, K..., Cin / G)``.  Both disagree about which channel
    // axis the grouping divides, so the leading axis is split into
    // ``(G, Cin / G)``, the group moved next to ``Cout``, and folded back
    // in.  Measured against a per-group reference: MLX pairs Cout-major
    // blocks with the matching Cin block, which is the same convention
    // Lucid uses, so no group reordering is needed — only the relayout.
    //
    // At ``G == 1`` this is exactly the old ``(1, 2..N+1, 0)`` permutation.
    static ::mlx::core::array
    gpu_w_to_mlx_conv_transpose(const ::mlx::core::array& W, int Cin, int Cout, int G, int N);

    // Slice one group's block out of ``axis``.
    static ::mlx::core::array gpu_slice_axis(const ::mlx::core::array& a, int axis, int lo, int hi);

    // Convolution operands with their channels aligned, so MLX never pads them.
    //
    // MLX pads unaligned channels up to a multiple of 16 inside the Metal
    // convolution — 2-D from 0.32.1 (``pad_in_channels_conv_2D_gpu``), 3-D in
    // every release (``pad_and_slice_conv_3D_gpu``) — and fills the padding
    // from a zero scalar it does not keep alive: the kernel reads it on the GPU
    // after the function that made it has freed it.  When the allocator hands
    // that buffer to another array first — any scalar written while the work is
    // still queued, which ``async_eval`` makes routine — the padded channels
    // take its value instead of zero and the convolution comes out wrong, whole
    // layers off and some NaN (a ResNet on MNIST: 4 of 16 Metal training
    // forwards on MLX 0.32.1).  Padding here, as a graph op, gives the same zero
    // channels from buffers MLX tracks, and MLX then runs the kernel it padded
    // for.  Both operands get the same extra channels, all zero, so every added
    // product is zero and the result is exact.
    //
    // The conditions mirror MLX's own: groups == 1 and no input dilation, then
    // for 2-D a stride-1 kernel of area >= 9 whose input channels are above 4
    // and unaligned while its output channels are aligned; for 3-D any
    // unaligned channels, output channels included (sliced off after).
    struct AlignedConv {
        ::mlx::core::array x;
        ::mlx::core::array w;
        int out_channels;  // O before padding; the caller slices when it grew
        bool sliced;
    };

    static AlignedConv gpu_align_conv_channels(
        ::mlx::core::array x, ::mlx::core::array w, int groups, bool stride_one, bool idil_one);

    // ── Convolutions with nothing to convolve ─────────────────────────────
    //
    // MLX's rank-5 convolution dereferences an empty operand: a conv3d or
    // conv_transpose3d with batch 0, no output channels or a zero spatial
    // extent dies with SIGSEGV when it is evaluated (1-D and 2-D happen to
    // survive).  None of those cases needs a convolution.  Every output
    // element is the bias plus a sum over input channels and kernel taps,
    // so an empty input-channel axis leaves the bias alone, and an empty
    // output has nothing to compute.  Every gradient is a sum over output
    // positions, so with none of them each is zero.  The four convolution
    // entry points answer these cases here, for every rank.

    // True when the input, the weight or the output (``out_numel``, the
    // output of the forward or the incoming gradient of the backward) holds
    // no elements.
    static bool gpu_conv_has_empty_operand(const ::mlx::core::array& x,
                                           const ::mlx::core::array& w,
                                           std::size_t out_numel);

    // The forward of an empty convolution: the bias, broadcast over
    // ``out_shape`` (channels on axis 1).  ``b`` always holds one value per
    // output channel — the op passes zeros when the module has no bias.
    static ::mlx::core::array gpu_conv_bias_only(const ::mlx::core::array& b,
                                                 const Shape& out_shape);

    // ``{dx, dW, db}`` of an empty convolution: zeros shaped like the input
    // and the weight, and ``db`` as already reduced from the gradient.
    static std::vector<Storage> gpu_conv_empty_grads(const ::mlx::core::array& x,
                                                     const ::mlx::core::array& w,
                                                     ::mlx::core::array db,
                                                     Dtype dt);

    // ``mlx::core::conv_general`` over channel-aligned operands (see
    // ``gpu_align_conv_channels``).  Every convolution this backend runs —
    // forward, both gradients, transposed — goes through here or through
    // ``gpu_mlx_conv_transpose``.
    static ::mlx::core::array gpu_conv_general(const ::mlx::core::array& x,
                                               const ::mlx::core::array& w,
                                               const std::vector<int>& stride,
                                               const std::vector<int>& pad_lo,
                                               const std::vector<int>& pad_hi,
                                               const std::vector<int>& kernel_dilation,
                                               const std::vector<int>& input_dilation,
                                               int groups,
                                               bool flip);

    // Forward convolution, with the same rank-5 group fallback as
    // ``gpu_mlx_conv_transpose`` below.  ``x_nhwc`` is channels-last and
    // ``W_nhwc`` is ``(out, K..., in / groups)``.
    static ::mlx::core::array gpu_mlx_conv_grouped(const ::mlx::core::array& x_nhwc,
                                                   const ::mlx::core::array& W_nhwc,
                                                   const std::vector<int>& stride,
                                                   const std::vector<int>& pad,
                                                   const std::vector<int>& dil,
                                                   int groups,
                                                   int N);

    static ::mlx::core::array gpu_mlx_conv_transpose(const ::mlx::core::array& x_nhwc,
                                                     const ::mlx::core::array& W_nhwc,
                                                     const int* stride,
                                                     const int* pad,
                                                     const int* dil,
                                                     const int* opad,
                                                     int groups,
                                                     int N);

    // Divide a summed pool by the per-output-position window span.
    //
    // The divisor factorises across axes, so it is built as a product of N
    // rank-broadcast vectors rather than a dense (B,C,O...) tensor.
    static ::mlx::core::array gpu_apply_pool_divisor(::mlx::core::array y,
                                                     int B,
                                                     int C,
                                                     const int* S,
                                                     const int* O,
                                                     const int* K,
                                                     const int* stride,
                                                     const PoolOpts& opts,
                                                     int N,
                                                     Dtype dt);

    // Per-axis span of the average-pool window at output index ``o``.
    // Mirrors the CPU kernel's ``avg_pool_span``: the window end is clamped to
    // ``S + pad``, so a ceil-mode overhang counts toward neither the sum nor
    // the divisor, and ``count_include_pad`` picks between the padded extent
    // and the real-element count.
    static int gpu_avg_pool_span(int o, int S, int K, int stride, int pad, bool count_pad);

    // Right-hand padding the ceil-mode overhang needs so the window view can
    // address the final window without running off the end of the buffer.
    static int gpu_ceil_overhang(int S, int K, int stride, int pad, int O);

    static ::mlx::core::array gpu_build_window_view(const ::mlx::core::array& padded,
                                                    int B,
                                                    int C,
                                                    const int* Sp,
                                                    const int* O,
                                                    const int* K,
                                                    const int* stride,
                                                    int N);

    // The ``sum`` / ``mean`` of a loss, accumulated in float32 for a half
    // loss (see ``widen_half``) and returned in its dtype.  The mean divides
    // in float32 too: as a float16 scalar the count of 65536 elements was
    // inf, so the mean came back 0 — or NaN once the sum had overflowed.
    ::mlx::core::array apply_loss_reduction(const ::mlx::core::array& values, int reduction);

    // ``grad / numel`` for a mean loss, in float32 for a half ``grad``: the
    // count is not representable in float16 past 65504 and 1/count is
    // subnormal past 16384.  Callers form the input gradient in that dtype
    // and round it to ``dt`` once (``narrow_to``), as the reference does.
    ::mlx::core::array scale_loss_grad(const ::mlx::core::array& grad,
                                       int reduction,
                                       std::size_t numel,
                                       ::mlx::core::Dtype dt);

    ::mlx::core::array class_target_indices(const ::mlx::core::array& target,
                                            const Shape& target_shape);

    ::mlx::core::array class_ignore_mask(const ::mlx::core::array& target_idx, int ignore_index);

    // The target with every ignored entry replaced by class 0, so the
    // gathers below read a real class there; the loss masks it out.  A
    // target that is neither ignored nor a class stays as it is, and the
    // gathers answer NaN for it (policy B, ``gpu_axis_index``): the loss
    // there is NaN, the CPU raises IndexError.
    ::mlx::core::array safe_class_indices(const ::mlx::core::array& target_idx,
                                          const ::mlx::core::array& keep_mask);

    // The class-axis value at each target (``take_along_axis`` on axis 1).
    static ::mlx::core::array class_pick(const ::mlx::core::array& x,
                                         const ::mlx::core::array& safe_t);

    ::mlx::core::array class_weight_gather(const Storage* weight,
                                           const ::mlx::core::array& safe_t,
                                           const ::mlx::core::Shape& shape,
                                           ::mlx::core::Dtype dt);

    // The number of targets not ignored — at least 1 — counted in float32
    // whatever the loss dtype: float16 holds an integer exactly only up to
    // 2048 and overflows at 65520, so a float16 batch of 70000 tokens was
    // averaged over inf.
    ::mlx::core::array class_valid_count(const ::mlx::core::array& keep_mask);

    // The class loss reduced; the sum accumulates in float32 for a half loss
    // and divides by the float32 ``valid_count`` before it is rounded back.
    Storage reduce_class_loss(const ::mlx::core::array& values,
                              const ::mlx::core::array& valid_count,
                              Dtype dt,
                              int reduction);

    // The upstream gradient of a class loss, spread over the target axis.
    // For a mean it is divided by the float32 count in float32 (a half
    // ``grad`` is widened), and the caller rounds the input gradient it
    // builds from it back to the loss dtype.
    ::mlx::core::array class_scaled_grad(const ::mlx::core::array& grad,
                                         const ::mlx::core::array& valid_count,
                                         const ::mlx::core::Shape& target_axis_shape,
                                         int reduction);

    static std::size_t linear_batch(const Shape& x_shape);

    // Applies a unary MLX operation fn to the underlying array and wraps the
    // result in a new GpuStorage.
    //
    // No mlx::core::contiguous() call here: element-wise MLX ops (exp, sin,
    // abs, relu, …) always write to a freshly allocated C-contiguous Metal
    // buffer regardless of the input strides.  Adding contiguous() would insert
    // a redundant lazy graph node that MLX must inspect and discard at eval
    // time, adding ~150 µs of overhead per call on 10M-element float32 arrays.
    //
    // If an op genuinely needs the output to be contiguous (e.g. a view op),
    // call mlx_unary_contiguous() instead or wrap explicitly.
    template <class Fn>
    Storage mlx_unary(const Storage& a, const Shape&, Dtype dt, Fn fn) {
        const auto& gs = std::get<GpuStorage>(a);
        // MLX promotes an integer input to float for a transcendental, so
        // the result comes back as F32 while the caller is still expecting
        // ``dt``.  Labelling it ``dt`` without converting reinterprets the
        // float bits: ``exp(int32([2, 3]))`` answered ``[1089237798,
        // 1101049647]`` — the bit patterns of 7.389 and 20.086 — instead
        // of ``[7, 20]``.  Cast, do not relabel.
        auto out = fn(*gs.arr);
        const auto want = gpu::to_mlx_dtype(dt);
        if (out.dtype() != want)
            out = ::mlx::core::astype(out, want);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }

    // Variant of mlx_unary that forces C-contiguous output.  Use only for ops
    // that may produce strided/non-contiguous results (e.g. view ops).
    template <class Fn>
    Storage mlx_unary_contiguous(const Storage& a, const Shape&, Dtype dt, Fn fn) {
        const auto& gs = std::get<GpuStorage>(a);
        // Same conversion rule as ``mlx_unary`` — see the note there.
        auto out = ::mlx::core::contiguous(fn(*gs.arr));
        const auto want = gpu::to_mlx_dtype(dt);
        if (out.dtype() != want)
            out = ::mlx::core::astype(out, want);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }

    // Fuse a multi-op element-wise UNARY composite into a single Metal kernel
    // via mlx::core::compile (shapeless → traced once on first call, then the
    // fused kernel is reused for every shape and dtype).  MLX eager does NOT
    // fuse element-wise chains (that is what compile is for), so a composite
    // such as gelu (~9 primitives) otherwise pays one kernel launch + a DRAM
    // round-trip PER primitive.  Fusing collapses it to one pass — e.g.
    // gelu_exact 1.71 ms -> 0.48 ms on (32,128,3072) F32 (faster than the
    // reference framework), bit-exact (maxdiff <= 2.4e-7; F16/BF16 exact).
    //
    // Use ONLY for genuine multi-op composites; single-primitive ops
    // (relu/sigmoid/exp) gain nothing.  ``fn`` MUST be capture-less — derive any
    // dtype-specific constant from the input's dtype (``x.dtype()``), never from
    // a captured ``Dtype`` (the compile overload needs a function-pointer-
    // convertible lambda).  The compiled function is a function-local static, so
    // it is built once per distinct ``Fn`` type and reused thereafter.
    template <class Fn>
    Storage mlx_unary_fused(const Storage& a, Dtype dt, Fn) {
        static const std::function<std::vector<::mlx::core::array>(
            const std::vector<::mlx::core::array>&)>
            compiled = ::mlx::core::compile(
                [](const std::vector<::mlx::core::array>& ins) -> std::vector<::mlx::core::array> {
                    return {Fn{}(ins[0])};
                },
                /*shapeless=*/true);
        const auto& gs = std::get<GpuStorage>(a);
        return Storage{gpu::wrap_mlx_array(std::move(compiled({*gs.arr})[0]), dt)};
    }

    // Binary counterpart of :meth:`mlx_unary_fused` — for two-input element-wise
    // composites (e.g. activation backward passes consuming the input and the
    // incoming gradient).  Same caching, same capture-less requirement.
    template <class Fn>
    Storage mlx_binary_fused(const Storage& a, const Storage& b, Dtype dt, Fn) {
        static const std::function<std::vector<::mlx::core::array>(
            const std::vector<::mlx::core::array>&)>
            compiled = ::mlx::core::compile(
                [](const std::vector<::mlx::core::array>& ins) -> std::vector<::mlx::core::array> {
                    return {Fn{}(ins[0], ins[1])};
                },
                /*shapeless=*/true);
        const auto& ga = std::get<GpuStorage>(a);
        const auto& gb = std::get<GpuStorage>(b);
        return Storage{gpu::wrap_mlx_array(std::move(compiled({*ga.arr, *gb.arr})[0]), dt)};
    }

    // Applies a binary MLX operation fn to two GpuStorage arrays.  Shape is
    // ignored because MLX handles broadcasting internally.
    //
    // Same rationale as mlx_unary: binary element-wise ops always produce
    // C-contiguous output in MLX, so contiguous() is not needed here.
    template <class Fn>
    Storage mlx_binary(const Storage& a, const Storage& b, const Shape&, Dtype dt, Fn fn) {
        const auto& ga = std::get<GpuStorage>(a);
        const auto& gb = std::get<GpuStorage>(b);
        return Storage{gpu::wrap_mlx_array(fn(*ga.arr, *gb.arr), dt)};
    }

    // Applies an MLX reduction fn(array, axes, keepdims).  If opts.axes is
    // empty all axes are reduced (full reduction).
    //
    // MLX reductions always produce C-contiguous output; contiguous() omitted.
    template <class Fn>
    Storage
    mlx_reduce(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt, Fn fn) {
        const auto& gs = std::get<GpuStorage>(a);
        std::vector<int> axes = opts.axes;
        if (axes.empty()) {
            for (int i = 0; i < static_cast<int>(in_shape.size()); ++i)
                axes.push_back(i);
        }
        return Storage{gpu::wrap_mlx_array(fn(*gs.arr, axes, opts.keepdims), dt)};
    }

    // Computes the sign of a permutation by counting cycles.
    // Returns +1.0 if the permutation is even, -1.0 if odd.
    // Used by linalg_det to apply the correct sign correction from the LU
    // pivot permutation.
    static float perm_index_sign(const std::uint32_t* p, std::size_t n);

    std::vector<Storage> batch_norm_eval_forward(const Storage& x,
                                                 const Storage& mean,
                                                 const Storage& var,
                                                 const Storage& gamma,
                                                 const Storage& beta,
                                                 const Shape& x_shape,
                                                 int C,
                                                 int,
                                                 double eps,
                                                 Dtype dt) override;

    std::vector<Storage> batch_norm_eval_backward(const Storage& x,
                                                  const Storage& mean,
                                                  const Storage& gamma,
                                                  const Storage& rstd,
                                                  const Storage& grad_out,
                                                  const Shape& x_shape,
                                                  int C,
                                                  int,
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
                                  const Shape&,
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

    // ── interpolate — the graphs live in ``gpu/Interpolate.h`` ─────────────
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

    // Built by comparing each index with every class, so no index value is
    // ever used as an offset.  One outside [0, num_classes), negative ones
    // included, matches no class and leaves its row zero: policy B for
    // Metal (``gpu_axis_index``).  The CPU raises IndexError, as the
    // reference does.
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

    // 3.4+ Phase A.1: fused ReLU backward.  Single MLX expression replacing
    // the prior 3-op composition (greater + astype + multiply) which MLX did
    // not fuse — the cast from Bool to F32 between greater and multiply broke
    // the kernel-fusion chain.  ``where(greater(x, 0), g, 0)`` is a single
    // conditional-select kernel that handles the typecast implicitly.  ResNet-
    // 18 ReLU bwd per-op (microbench, M4 Max): the +0.79 ms gap vs the
    // reference framework should close.
    Storage relu_backward(const Storage& g, const Storage& x, const Shape&, Dtype dt) override;

    Storage reduce_grad_to_shape(const Storage& grad,
                                 const Shape& grad_shape,
                                 const Shape& target_shape,
                                 Dtype dt) override;

    Storage broadcast_back_for_reduce(const Storage& grad,
                                      const Shape&,
                                      const Shape& input_shape,
                                      const std::vector<int>& axes,
                                      bool,
                                      Dtype dt) override;

    Storage full(const Shape& shape, Dtype dt, double fill_value) override;

    Storage eye(std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt) override;

    Storage diag(const Storage& v,
                 const Shape& v_shape,
                 std::int64_t k,
                 Dtype dt,
                 Shape& out_shape) override;

    Storage tri(const Storage& input, const Shape& shape, Dtype dt, int k, bool upper) override;

    // Exact, in ``dt`` (DivMod.h).  floordiv_op no longer calls this.
    Storage floordiv(const Storage& a, const Storage& b, const Shape&, Dtype dt) override;

    Storage inner(const Storage& a,
                  const Storage& b,
                  const Shape&,
                  const Shape&,
                  const Shape&,
                  Dtype dt) override;

    CpuStorage
    permute_cpu(const CpuStorage&, const Shape&, const std::vector<int>&, Dtype) override;

    Storage tensordot(const Storage& a,
                      const Storage& b,
                      const Shape&,
                      const Shape&,
                      const Shape&,
                      const std::vector<int>& axes_a,
                      const std::vector<int>& axes_b,
                      Dtype dt) override;

    Storage where_op(
        const Storage& cond, const Storage& x, const Storage& y, const Shape&, Dtype dt) override;

    Storage reduce_broadcast(const Storage& grad,
                             const Shape& input_shape,
                             const Shape& output_shape,
                             Dtype dt) override;

    Storage histogram_forward(
        const Storage&, const Shape&, Dtype, double, double, std::int64_t, bool) override;

    CpuStorage
    nonzero_forward(const Storage&, const Shape&, Dtype, std::size_t& numel_out) override;

    Storage to_shared_storage(const Storage& src, const Shape& shape) override;

    Storage run_custom_metal_kernel(const std::string& kernel_source,
                                    const std::string& function_name,
                                    const std::vector<Storage>& inputs,
                                    const Shape& output_shape,
                                    Dtype output_dtype,
                                    const std::array<std::size_t, 3>& grid,
                                    const std::array<std::size_t, 3>& threads) override;
};

}  // namespace backend
}  // namespace lucid
