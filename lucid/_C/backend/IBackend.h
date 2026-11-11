// lucid/_C/backend/IBackend.h
//
// Pure-virtual interface that every hardware backend must implement.
//
// All compute operations in the Lucid ML engine — elementwise math,
// reductions, matrix multiply, convolution, normalization, pooling,
// loss functions, SDPA, LSTM, RoPE, interpolation, grid sample, and more
// — are declared here as abstract virtual methods.  Higher-level tensor
// code calls through :class:`Dispatcher` to whichever concrete backend
// is registered for a given :class:`Device`.  Adding a new backend means
// subclassing :class:`IBackend` and registering an instance via
// :func:`Dispatcher::register_backend`.
//
// Notes
// -----
// The current device split (per DEVELOPMENT.md H3) is:
//
// - ``Device::CPU`` → :class:`CpuBackend` (Apple Accelerate only —
//   vDSP / vForce / BLAS / LAPACK).
// - ``Device::GPU`` → :class:`GpuBackend` (MLX / Metal only).
//
// All op methods take :class:`Storage` by const-reference and return a
// new :class:`Storage`; no in-place mutation is exposed at this layer.
// Concrete backends choose the most efficient internal representation
// (``CpuStorage`` for CPU, ``GpuStorage`` for GPU).
//
// See Also
// --------
// :class:`Dispatcher` — routes per-device calls to the right backend.
// :class:`CpuBackend` — concrete CPU implementation.
// :class:`GpuBackend` — concrete GPU implementation.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/ErrorBuilder.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "GradMask.h"

namespace lucid {
namespace backend {

// Parameter bag for a single matrix-multiplication call.
//
// Carries the logical shape and transposition flags for one ``a @ b``
// product.  Dimensions are interpreted in *logical* (not transposed)
// order so the result is always $(M \times K) \cdot (K \times N) = (M
// \times N)$ regardless of the ``transA`` / ``transB`` flags.
//
// Attributes
// ----------
// transA : bool
//     If ``true`` the dispatched kernel treats the left operand as
//     transposed before contraction.
// transB : bool
//     If ``true`` the dispatched kernel treats the right operand as
//     transposed before contraction.
// M : int
//     Number of rows of the (post-transpose) left operand.
// K : int
//     Contracted inner dimension.
// N : int
//     Number of columns of the (post-transpose) right operand.
// batch : std::size_t
//     Number of independent matmuls stacked along the leading axis.
//     ``batch == 1`` is the standard 2-D case; ``batch > 1`` indicates a
//     batched matmul where the same A/B strides repeat ``batch`` times.
//
// See Also
// --------
// :func:`IBackend::matmul` — consumes this struct.
struct MatmulOpts {
    bool transA = false;
    bool transB = false;
    int M = 0, K = 0, N = 0;
    std::size_t batch = 1;
};

// Parameter bag for a convolution op (legacy compatibility path).
//
// Bundles batch / channel counts together with shape, stride, padding,
// dilation, and grouping for an N-D convolution.  The primary
// convolution code paths now use :class:`IBackend::ConvNdOpts` plus
// explicit dimension arguments; this struct is retained for older
// callers that build a free-standing ``ConvOpts`` value.
//
// Attributes
// ----------
// ndim : int
//     Number of spatial dimensions (1, 2, or 3).
// N : int
//     Batch size.
// C_in, C_out : int
//     Input and output channel counts.
// input_shape, kernel_shape : Shape
//     Logical input feature-map and kernel-tensor shapes.
// stride, padding, dilation : std::vector<int>
//     Per-spatial-dimension geometry.
// groups : int
//     Channel grouping factor; ``1`` is standard convolution.
// with_bias : bool
//     Whether a bias term participates in the forward pass.
//
// See Also
// --------
// :class:`IBackend::ConvNdOpts` — modern, fixed-size parameter bag.
struct ConvOpts {
    int ndim = 2;
    int N = 0, C_in = 0, C_out = 0;
    Shape input_shape;
    Shape kernel_shape;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    int groups = 1;
    bool with_bias = false;
};

// Parameter bag for axis-reduction operations.
//
// Used by :func:`IBackend::reduce_sum`, :func:`reduce_mean`,
// :func:`variance`, :func:`reduce_max`, :func:`reduce_min`, and similar
// axis-collapsing ops.
//
// Attributes
// ----------
// axes : std::vector<int>
//     List of dimensions to reduce over.  Entries may be negative
//     (Python-style) and are normalised by the callee.  An empty list is
//     interpreted as *reduce over all axes*.
// keepdims : bool
//     If ``true`` the collapsed dimensions are retained as size-1 in the
//     output shape; otherwise they are removed.
//
// See Also
// --------
// :func:`IBackend::reduce_sum` — primary consumer.
struct ReduceOpts {
    std::vector<int> axes;
    bool keepdims = false;
};

// Aggregated return value from classification-loss forward passes.
//
// Used by :func:`IBackend::cross_entropy_loss` and
// :func:`IBackend::nll_loss` to ship the forward output together with
// the saved tensors the corresponding backward pass needs.
//
// Attributes
// ----------
// output : Storage
//     Scalar (reduction != 0) or per-sample (reduction == 0) loss.
// saved_aux : Storage
//     Intermediate values needed by the backward pass — e.g. the
//     log-softmax activations for cross-entropy.
// valid_count : Storage
//     Scalar I64 count of un-ignored samples; consumed by mean reduction
//     so that ``ignore_index`` entries are excluded from the denominator.
//
// See Also
// --------
// :func:`IBackend::cross_entropy_backward` — consumer.
struct ClassLossForwardResult {
    Storage output;
    Storage saved_aux;
    Storage valid_count;
};

// Convenience wrapper bundling two Storage outputs from one op call.
//
// Used by ops that naturally return a pair of arrays — QR
// decomposition, eigendecomposition, RMSNorm forward (output + saved
// rstd), sort+argsort, LU factor, LDL factor, and similar.
//
// Attributes
// ----------
// first, second : Storage
//     The two output tensors; the semantic role of each is documented
//     on the method that returns this struct.
struct StoragePair {
    Storage first;
    Storage second;
};

// Abstract compute backend interface.
//
// One concrete :class:`IBackend` exists per :class:`Device` slot (CPU,
// GPU); :class:`Dispatcher` holds them and routes every op based on the
// tensor's device tag.  Implementations dispatch the actual math to
// their device-native library — Apple Accelerate (vDSP / vForce / BLAS
// / LAPACK) for CPU, MLX for GPU.
//
// Notes
// -----
// **Ownership.**  Each concrete backend is created once and stored
// inside the :class:`Dispatcher` singleton via ``std::unique_ptr``; the
// lifetime of every :class:`IBackend` therefore equals the lifetime of
// the process.
//
// **Thread safety.**  Methods are *thread-compatible* but not internally
// synchronised: the engine never calls into the same backend from
// multiple threads simultaneously on the same op, though distinct ops
// on distinct tensors may overlap.  Callers that share a backend across
// threads must provide their own serialisation, or rely on the
// underlying framework (e.g. MLX's own thread-safety guarantees).
//
// **Design rationale.**  All methods take :class:`Storage` by
// const-reference and return a new :class:`Storage`; no in-place
// mutation is exposed at this layer.  Concrete implementations choose
// the most efficient internal representation — ``CpuStorage`` (Apple
// aligned host buffer) for CPU, ``GpuStorage`` (``mlx::core::array``)
// for GPU.
//
// See Also
// --------
// :class:`CpuBackend` — Accelerate-backed concrete CPU.
// :class:`GpuBackend` — MLX-backed concrete GPU.
// :class:`Dispatcher` — selects which to use.
class IBackend {
public:
    virtual ~IBackend() = default;

    // Returns the :class:`Device` tag associated with this backend.
    //
    // Returns
    // -------
    // Device
    //     ``Device::CPU`` for :class:`CpuBackend`, ``Device::GPU`` for
    //     :class:`GpuBackend`.
    virtual Device device() const noexcept = 0;

    // Transfers a CPU buffer into the backend's native storage format.
    //
    // Parameters
    // ----------
    // cpu : CpuStorage
    //     Source CPU-side buffer (owning).
    // shape : Shape
    //     Logical shape of the buffer.  Used by GPU backends to set the
    //     ``mlx::core::array`` shape; CPU is shape-agnostic at this layer.
    //
    // Returns
    // -------
    // Storage
    //     Backend-native storage variant.
    //
    // Notes
    // -----
    // For :class:`CpuBackend` this is a no-op move into the
    // ``CpuStorage`` slot of the variant; for :class:`GpuBackend` it
    // copies the data to GPU-private memory via ``mlx::core::copy``.
    virtual Storage from_cpu(CpuStorage cpu, const Shape& shape) = 0;

    // Allocates a zero-filled buffer of shape ``shape`` with element type ``dt``.
    //
    // Parameters
    // ----------
    // shape : Shape
    //     Output shape.
    // dt : Dtype
    //     Element dtype.
    //
    // Returns
    // -------
    // Storage
    //     Newly-allocated zero buffer.
    virtual Storage zeros(const Shape& shape, Dtype dt) = 0;

    // Allocates a one-filled buffer of shape ``shape`` with element type ``dt``.
    //
    // Parameters
    // ----------
    // shape : Shape
    //     Output shape.
    // dt : Dtype
    //     Element dtype.
    //
    // Returns
    // -------
    // Storage
    //     Newly-allocated buffer where every element equals 1.
    virtual Storage ones(const Shape& shape, Dtype dt) = 0;

    // Returns a deep copy of ``src`` reinterpreted with the given shape/dtype.
    //
    // Parameters
    // ----------
    // src : const Storage&
    //     Source buffer.  Must already match ``shape`` and ``dt``.
    // shape : Shape
    //     Logical shape of the result.
    // dt : Dtype
    //     Element dtype of the result.
    //
    // Returns
    // -------
    // Storage
    //     A fresh, independent allocation containing the same bytes.
    virtual Storage clone(const Storage& src, const Shape& shape, Dtype dt) = 0;

    // Returns a contiguous copy of a (possibly strided or offset) view.
    //
    // Walks the stride / offset layout to produce a densely-packed
    // output buffer suitable for kernels that assume row-major
    // contiguous data.
    //
    // Parameters
    // ----------
    // src : const Storage&
    //     Source buffer (may overlap a larger allocation).
    // shape : Shape
    //     Logical shape of the view.
    // stride : const Stride&
    //     Per-dimension byte (or element, depending on backend) stride.
    // storage_offset : std::size_t
    //     Byte offset from the start of ``src`` where the view begins.
    // already_contiguous : bool
    //     Hint that the layout is already row-major contiguous; enables
    //     a fast ``memcpy`` path when combined with ``storage_offset == 0``.
    // dt : Dtype
    //     Element dtype.
    //
    // Returns
    // -------
    // Storage
    //     Densely-packed copy of the view.
    virtual Storage contiguous(const Storage& src,
                               const Shape& shape,
                               const Stride& stride,
                               std::size_t storage_offset,
                               bool already_contiguous,
                               Dtype dt) = 0;

    // Elementwise arithmetic on two broadcast-prepared operands.
    //
    // Each of the five primitives below computes ``out[i] = a[i] op b[i]``
    // for every element, with both ``a`` and ``b`` already broadcast (by
    // the caller) to the common output ``shape``.
    //
    // Parameters
    // ----------
    // a, b : const Storage&
    //     Operands, already broadcast to ``shape``.
    // shape : const Shape&
    //     Output shape (used for ``numel`` computation).
    // dt : Dtype
    //     Element dtype (common to ``a``, ``b``, and the result).
    //
    // Returns
    // -------
    // Storage
    //     Newly-allocated output buffer of the same shape and dtype.
    //
    // Math
    // ----
    // $$\mathrm{out}[i] = \begin{cases}
    //   a[i] + b[i] & \text{add} \\
    //   a[i] - b[i] & \text{sub} \\
    //   a[i] \cdot b[i] & \text{mul} \\
    //   a[i] / b[i] & \text{div} \\
    //   a[i]^{b[i]} & \text{pow}
    // \end{cases}$$
    virtual Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    // Elementwise bitwise operation selected by integer ``op`` code.
    //
    // Parameters
    // ----------
    // a, b : const Storage&
    //     Operands, already broadcast to ``shape``.
    // shape : const Shape&
    //     Output shape.
    // dt : Dtype
    //     Element dtype.  Only integer and Bool dtypes are valid.
    // op : int
    //     Op selector: ``0=AND``, ``1=OR``, ``2=XOR`` (GPU also supports
    //     ``3=left_shift``, ``4=right_shift``).
    //
    // Returns
    // -------
    // Storage
    //     Newly-allocated result of the same shape and dtype.
    virtual Storage
    bitwise_binary(const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) = 0;

    // Elementwise comparison returning a Bool Storage of the same shape.
    //
    // Parameters
    // ----------
    // a, b : const Storage&
    //     Operands, already broadcast to ``shape``.
    // shape : const Shape&
    //     Output shape.
    // dt : Dtype
    //     Dtype of the *operands* (the output is always Bool).
    // op : int
    //     Comparison selector: ``0=EQ``, ``1=NE``, ``2=GT``, ``3=GE``,
    //     ``4=LT``, ``5=LE``.
    //
    // Returns
    // -------
    // Storage
    //     Bool buffer of shape ``shape``.
    virtual Storage
    compare_binary(const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) = 0;

    // Elementwise ``max(a, b)`` and ``min(a, b)``.
    //
    // Parameters
    // ----------
    // a, b : const Storage&
    //     Operands, already broadcast to ``shape``.
    // shape : const Shape&
    //     Output shape.
    // dt : Dtype
    //     Common element dtype.
    //
    // Returns
    // -------
    // Storage
    //     Element-wise max or min, same shape and dtype.
    virtual Storage maximum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage minimum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    // Elementwise unary math ops.
    //
    // The group below — :func:`exp`, :func:`log`, :func:`sqrt`,
    // :func:`rsqrt`, :func:`abs`, :func:`neg`, :func:`sign`,
    // :func:`floor`, :func:`ceil`, :func:`round`, :func:`sin`,
    // :func:`cos`, :func:`tanh`, :func:`sigmoid`, :func:`relu` — all
    // share the same call shape.  Each applies the named scalar
    // function element-by-element to the flattened buffer; ``shape`` is
    // used only for ``numel`` computation.
    //
    // Parameters
    // ----------
    // a : const Storage&
    //     Input buffer.
    // shape : const Shape&
    //     Logical shape (drives the element count).
    // dt : Dtype
    //     Element dtype (matches the result).
    //
    // Returns
    // -------
    // Storage
    //     Newly-allocated output of the same shape and dtype.
    //
    // Notes
    // -----
    // Dispatch: CPU implementations call ``vForce`` (transcendentals) or
    // ``vDSP`` (algebraic ops); GPU implementations call the matching
    // ``mlx::core::*`` primitive.
    virtual Storage exp(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage log(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage sqrt(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage rsqrt(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage abs(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage neg(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage sign(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage floor(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage ceil(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage round(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage sin(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage cos(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage tanh(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage sigmoid(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage relu(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Additional elementwise unary ops.
    //
    // Logarithmic (:func:`log2`), error function (:func:`erf`,
    // :func:`erfinv`), reciprocal / power (:func:`reciprocal`,
    // :func:`square`, :func:`cube`, :func:`cube_root`), trigonometric
    // (:func:`tan`, :func:`asin`, :func:`acos`, :func:`atan`,
    // :func:`sinh`, :func:`cosh`) that are not part of the core unary
    // set above.  Same call shape and dispatch contract as that group.
    virtual Storage log2(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage erf(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage erfinv(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage reciprocal(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage square(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage cube(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage cube_root(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage tan(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage asin(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage acos(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage atan(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage sinh(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage cosh(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Bitwise NOT.
    //
    // Parameters
    // ----------
    // a : const Storage&
    //     Input buffer (integer or Bool dtype only).
    // shape : const Shape&
    //     Logical shape.
    // dt : Dtype
    //     Element dtype.
    //
    // Returns
    // -------
    // Storage
    //     Per-element bitwise complement.
    virtual Storage invert(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // ── Complex viewing ────────────────────────────────────────────────────
    //
    // Each backend implements these on its native primitive: CPU uses
    // Apple Accelerate (vDSP plus interleaved-complex element walks),
    // GPU uses ``mlx::core::real`` / ``imag`` / ``conjugate``.

    // Real part of a complex (C64) input — output dtype is F32.
    virtual Storage complex_real(const Storage& a, const Shape& shape) = 0;

    // Imaginary part of a complex (C64) input — output dtype is F32.
    virtual Storage complex_imag(const Storage& a, const Shape& shape) = 0;

    // Build a C64 array from two F32 arrays of the same shape; the result
    // satisfies ``complex_real(out) == re`` and ``complex_imag(out) == im``.
    virtual Storage complex_combine(const Storage& re, const Storage& im, const Shape& shape) = 0;

    // Element-wise complex conjugate.  No-op for real dtypes (returned
    // unchanged); negates the imaginary part for C64.
    virtual Storage complex_conj(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Activation functions with paired forward / backward kernels.
    //
    // Each forward op below has a matching ``*_backward`` that receives
    // the original pre-activation input ``a`` together with the upstream
    // gradient ``grad`` and returns the input gradient $\partial
    // \mathcal{L}/\partial x$.
    //
    // Parameters
    // ----------
    // a : const Storage&
    //     Pre-activation input.
    // grad : const Storage&
    //     Upstream gradient $\partial \mathcal{L}/\partial y$ (for the
    //     ``*_backward`` variants).
    // shape : const Shape&
    //     Common shape of input, output, and gradient.
    // dt : Dtype
    //     Common element dtype.
    //
    // Returns
    // -------
    // Storage
    //     Forward output, or input gradient for the ``*_backward`` form.
    //
    // Notes
    // -----
    // Backward kernels exist as fused single-op delegates so the GPU
    // backend can route to MPSGraph and the CPU backend to hand-rolled
    // scalar loops; previously the autograd composed these from
    // primitive storage ops.
    virtual Storage silu(const Storage& a, const Shape& shape, Dtype dt) = 0;
    // silu_backward — dL/dx = σ(x) * (1 + x*(1 - σ(x))) * dL/dy.  Lucid's
    // autograd node previously composed this from storage primitives; the
    // single-op delegate lets the GPU backend dispatch to MPSGraph and
    // CPU to a hand-rolled scalar loop.
    virtual Storage
    silu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;
    virtual Storage gelu(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    gelu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;
    // gelu_exact — Gaussian-CDF formulation: 0.5*x*(1 + erf(x/sqrt(2))).
    // Distinct from `gelu` (tanh approximation) so callers can pick either
    // numerical recipe; Python F.gelu(approximate="none") routes here.
    virtual Storage gelu_exact(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    gelu_exact_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;
    virtual Storage leaky_relu(const Storage& a, const Shape& shape, Dtype dt, double slope) = 0;
    virtual Storage softplus(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage elu(const Storage& a, const Shape& shape, Dtype dt, double alpha) = 0;
    virtual Storage elu_backward(
        const Storage& a, const Storage& grad, const Shape& shape, Dtype dt, double alpha) = 0;
    virtual Storage selu(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    selu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;
    virtual Storage mish(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    mish_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;
    virtual Storage hard_sigmoid(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    hard_sigmoid_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;
    virtual Storage hard_swish(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    hard_swish_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;

    // ReLU clamped to [0, 6], used in MobileNet-style networks.
    virtual Storage relu6(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Numerically stable log-softmax along a single axis.
    //
    // Math
    // ----
    // $$y_i = x_i - \log\sum_j \exp(x_j),$$
    //
    // computed via the max-subtraction trick to avoid overflow.
    //
    // Parameters
    // ----------
    // a : const Storage&
    //     Input logits.
    // shape : const Shape&
    //     Shape of ``a``.
    // axis : int
    //     Dimension to normalise over (may be negative).
    // dt : Dtype
    //     Element dtype.
    //
    // Returns
    // -------
    // Storage
    //     Log-probabilities of the same shape and dtype as ``a``.
    virtual Storage log_softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;

    // Backward pass for :func:`log_softmax`.
    //
    // Math
    // ----
    // $$\frac{\partial \mathcal{L}}{\partial x} =
    //   \frac{\partial \mathcal{L}}{\partial y}
    //   - \exp(y) \sum_{\text{axis}}\frac{\partial \mathcal{L}}{\partial y},$$
    //
    // where ``y = log_softmax(x)`` is the saved forward output.
    //
    // Parameters
    // ----------
    // y : const Storage&
    //     Saved forward output.
    // grad_out : const Storage&
    //     Upstream gradient.
    // shape : const Shape&
    //     Common shape of ``y``, ``grad_out``, and the returned input
    //     gradient.
    // axis : int
    //     Axis along which the forward log-softmax was taken.
    // dt : Dtype
    //     Element dtype.
    //
    // Returns
    // -------
    // Storage
    //     Input gradient $\partial \mathcal{L}/\partial x$.
    virtual Storage log_softmax_backward(
        const Storage& y, const Storage& grad_out, const Shape& shape, int axis, Dtype dt) = 0;

    // Boolean reductions — reduce the full tensor to a scalar bool.
    virtual Storage any(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage all(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Floating-point predicate ops (output dtype is always Bool).
    virtual Storage isinf(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage isnan(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage isfinite(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Replace NaN/Inf values with finite substitutes.  Output dtype matches input.
    virtual Storage nan_to_num(const Storage& a,
                               const Shape& shape,
                               Dtype dt,
                               double nan_val,
                               double posinf_val,
                               double neginf_val) = 0;

    // Axis reductions.  opts.axes specifies which dimensions to collapse; the
    // output shape follows keepdims semantics.  variance() uses the biased
    // (population) variance formula to match NumPy/reference defaults.
    virtual Storage
    reduce_sum(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) = 0;
    virtual Storage
    reduce_mean(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) = 0;
    virtual Storage
    variance(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) = 0;
    virtual Storage
    reduce_max(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) = 0;
    virtual Storage
    reduce_min(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) = 0;

    // Scan operations along a single axis.
    virtual Storage cumsum(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage cumprod(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage cummax(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage cummin(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;

    // Stable numerically-safe softmax forward and its vector-Jacobian product.
    // softmax_backward receives the softmax output `z` (not the pre-activation
    // input) because the gradient formula only needs the output: dx = z*(dz - (z·dz)).
    virtual Storage softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage softmax_backward(
        const Storage& z, const Storage& grad_out, const Shape& shape, int axis, Dtype dt) = 0;

    // Reverses elements along `axis` (equivalent to a[::-1] on that dimension).
    virtual Storage
    reverse_along_axis(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;

    // Sum of the main diagonal elements; trace_backward scatters the upstream
    // gradient back to the diagonal positions of the input shape.
    virtual Storage trace(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage trace_backward(const Storage& grad_out, const Shape& input_shape, Dtype dt) = 0;

    // Constructs N-D coordinate grids from 1-D input vectors.
    // indexing_xy=true uses x-column / y-row convention (MATLAB-style).
    virtual std::vector<Storage> meshgrid(const std::vector<Storage>& xs,
                                          const Shape& out_shape,
                                          Dtype dt,
                                          bool indexing_xy) = 0;

    // Selects grad where cond is true (true_branch=true) or false, zeroing
    // the other branch.  Used by the Where autograd backward.
    virtual Storage where_branch(const Storage& grad,
                                 const Storage& cond,
                                 const Shape& shape,
                                 Dtype dt,
                                 bool true_branch) = 0;

    // Replaces elements of `a` where `mask` is nonzero with `value`.
    virtual Storage masked_fill(
        const Storage& a, const Storage& mask, const Shape& shape, Dtype dt, double value) = 0;
    // Gathers slices from `a` along `axis` using integer `indices`.
    // index_dtype is the dtype of the indices tensor (I32 or I64).
    virtual Storage gather(const Storage& a,
                           const Storage& indices,
                           const Shape& input_shape,
                           const Shape& output_shape,
                           int axis,
                           Dtype index_dtype,
                           Dtype dt) = 0;

    // Backward pass for gather: scatters grad values back to the source
    // positions identified by indices.  Result has input_shape.
    virtual Storage gather_backward(const Storage& grad,
                                    const Storage& indices,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    int axis,
                                    Dtype index_dtype,
                                    Dtype dt) = 0;

    // Extracts a generalized diagonal between axis1 and axis2 with an
    // integer offset from the main diagonal (positive = above, negative = below).
    virtual Storage diagonal(
        const Storage& a, const Shape& input_shape, int offset, int axis1, int axis2, Dtype dt) = 0;

    // Backward pass for diagonal: scatters the diagonal gradient back into a
    // zero-filled tensor of input_shape.
    virtual Storage diagonal_backward(const Storage& grad,
                                      const Shape& input_shape,
                                      const Shape& output_shape,
                                      int offset,
                                      int axis1,
                                      int axis2,
                                      Dtype dt) = 0;
    virtual Storage roll(const Storage& a,
                         const Shape& shape,
                         Dtype dt,
                         const std::vector<std::int64_t>& shifts,
                         const std::vector<int>& axes) = 0;
    virtual Storage
    reshape(const Storage& a, const Shape& src_shape, const Shape& dst_shape, Dtype dt) = 0;
    virtual Storage slice_axis(const Storage& a,
                               const Shape& src_shape,
                               const Shape& slice_shape,
                               int axis,
                               std::int64_t offset,
                               Dtype dt) = 0;
    // Inverse of slice_axis: scatters `a` into a zero-filled tensor of
    // dst_shape at the given offset along axis.  Used in slice backward passes.
    virtual Storage insert_axis_slice(const Storage& a,
                                      const Shape& src_shape,
                                      const Shape& dst_shape,
                                      int axis,
                                      std::int64_t offset,
                                      Dtype dt) = 0;

    // Concatenates the tensors in `xs` along `axis`; each shape in `shapes`
    // corresponds element-wise to `xs`.
    virtual Storage concatenate(const std::vector<Storage>& xs,
                                const std::vector<Shape>& shapes,
                                int axis,
                                Dtype dt) = 0;

    // Stacks tensors along a new axis inserted at position `axis`.
    // All inputs must have the same `input_shape`.
    virtual Storage
    stack(const std::vector<Storage>& xs, const Shape& input_shape, int axis, Dtype dt) = 0;

    // Splits `a` into num_splits equal-sized pieces along `axis`.
    virtual std::vector<Storage> split_equal(
        const Storage& a, const Shape& shape, int axis, std::int64_t num_splits, Dtype dt) = 0;

    // Splits `a` at the given index boundaries along `axis` (like np.split).
    virtual std::vector<Storage> split_at(const Storage& a,
                                          const Shape& shape,
                                          int axis,
                                          const std::vector<std::int64_t>& indices,
                                          Dtype dt) = 0;

    // Backward pass for repeat(): reduces repeated slices by summing them back
    // to the original input_shape.
    virtual Storage repeat_backward(const Storage& grad_out,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    int axis,
                                    std::int64_t repeats,
                                    Dtype dt) = 0;

    // Backward pass for tile(): accumulates gradients from all tiled copies
    // back into the original input_shape.
    virtual Storage tile_backward(const Storage& grad_out,
                                  const Shape& input_shape,
                                  const Shape& padded_shape,
                                  const Shape& output_shape,
                                  const std::vector<std::int64_t>& reps,
                                  Dtype dt) = 0;

    // Returns (sorted_values, sorted_indices) along `axis`.
    // descending=true sorts from largest to smallest.
    virtual std::pair<Storage, Storage> sort_select(const Storage& a,
                                                    const Shape& input_shape,
                                                    const Shape& output_shape,
                                                    int axis,
                                                    Dtype dt,
                                                    bool descending) = 0;

    // Returns the permutation indices that would sort `a` along `axis`.
    virtual Storage argsort(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;

    // Returns the index of the max (is_min=false) or min (is_min=true) along `axis`.
    // Output dtype is I64.
    virtual Storage arg_reduce_index(
        const Storage& a, const Shape& shape, int axis, bool keepdims, Dtype dt, bool is_min) = 0;

    // Accumulates `grad` values into output_shape positions specified by `indices`
    // along `axis` (sparse scatter-add / gather backward).
    virtual Storage scatter_add_axis(const Storage& grad,
                                     const Storage& indices,
                                     const Shape& output_shape,
                                     const Shape& grad_shape,
                                     int axis,
                                     Dtype dt) = 0;

    // User-facing scatter-add: out[..., index[i], ...] += src[..., i, ...]
    // along dim. base is unchanged except at index positions where src is added.
    virtual Storage scatter_add(const Storage& base,
                                const Storage& indices,
                                const Storage& src,
                                const Shape& base_shape,
                                const Shape& idx_shape,
                                int dim,
                                Dtype dt) = 0;

    // Scatter-reduce variants: base is pre-initialised with the appropriate
    // neutral element (−∞ for amax, +∞ for amin, 1 for prod) when include_self
    // is false; the callers set this up before dispatching.
    virtual Storage scatter_amax(const Storage& base,
                                 const Storage& indices,
                                 const Storage& src,
                                 const Shape& base_shape,
                                 const Shape& idx_shape,
                                 int dim,
                                 Dtype dt) = 0;
    virtual Storage scatter_amin(const Storage& base,
                                 const Storage& indices,
                                 const Storage& src,
                                 const Shape& base_shape,
                                 const Shape& idx_shape,
                                 int dim,
                                 Dtype dt) = 0;
    virtual Storage scatter_prod(const Storage& base,
                                 const Storage& indices,
                                 const Storage& src,
                                 const Shape& base_shape,
                                 const Shape& idx_shape,
                                 int dim,
                                 Dtype dt) = 0;

    // User-facing scatter-set (overwrite): out[..., index[i], ...] = src[..., i, ...]
    // along dim.  base is copied, then the indexed slices are replaced by src.
    // Single-kernel write that index_copy routes through (MPSGraphScatterModeSet
    // when compiled; mlx::put_along_axis eager on GPU).
    virtual Storage scatter_set(const Storage& base,
                                const Storage& indices,
                                const Storage& src,
                                const Shape& base_shape,
                                const Shape& idx_shape,
                                int dim,
                                Dtype dt) = 0;

    // Sliding-window view along a single dimension.
    // Returns shape (*base.shape[:dim], L, *base.shape[dim+1:], size)
    // where L = (dim_size - size) / step + 1.
    virtual Storage
    unfold_dim(const Storage& a, const Shape& in_shape, int dim, int size, int step, Dtype dt) = 0;

    // General batched matrix multiplication.
    //
    // Math
    // ----
    // $$\mathrm{out} = a \cdot b,$$
    //
    // where ``a`` and ``b`` may have a leading batch dimension and may
    // be transposed before contraction.
    //
    // Parameters
    // ----------
    // a, b : const Storage&
    //     Operand matrices (or batches of matrices).
    // opts : const MatmulOpts&
    //     Shape (``M``, ``K``, ``N``, ``batch``) and transpose flags.
    // dt : Dtype
    //     Element dtype of all participants.
    //
    // Returns
    // -------
    // Storage
    //     Result of shape ``(batch?, M, N)``.
    //
    // Notes
    // -----
    // CPU dispatches to ``cblas_sgemm`` / ``cblas_dgemm`` via
    // Accelerate; GPU dispatches to ``mlx::core::matmul`` (which selects
    // an MPS or Metal-shader kernel internally).
    virtual Storage
    matmul(const Storage& a, const Storage& b, const MatmulOpts& opts, Dtype dt) = 0;

    // Fused linear projection $\mathrm{out} = x \cdot W^\top + b$.
    //
    // Parameters
    // ----------
    // x : const Storage&
    //     Input activations of shape ``(*, K)``.
    // weight : const Storage&
    //     Weight matrix of shape ``(N, K)`` (note: pre-transposed).
    // bias : const Storage&
    //     Bias vector of shape ``(N,)``.
    // x_shape, weight_shape, out_shape : const Shape&
    //     Logical shapes of the three operands and result.
    // dt : Dtype
    //     Element dtype.
    //
    // Returns
    // -------
    // Storage
    //     Output activations of shape ``(*, N)``.
    virtual Storage linear(const Storage& x,
                           const Storage& weight,
                           const Storage& bias,
                           const Shape& x_shape,
                           const Shape& weight_shape,
                           const Shape& out_shape,
                           Dtype dt) = 0;

    // Backward pass for linear; returns [grad_x, grad_weight, grad_bias].
    virtual std::vector<Storage> linear_backward(const Storage& grad,
                                                 const Storage& x,
                                                 const Storage& weight,
                                                 const Shape& x_shape,
                                                 const Shape& weight_shape,
                                                 const Shape& bias_shape,
                                                 Dtype dt,
                                                 GradMask3 grad_mask = {}) = 0;
    // RMSNorm forward pass.  outer = product of batch/sequence dims; normalized_size
    // is the last (feature) dimension.  Returns (output, saved_rstd); saved_rstd
    // holds per-row reciprocal standard deviations needed by the backward pass.
    virtual StoragePair rms_norm_forward(const Storage& x,
                                         const Storage& gamma,
                                         std::size_t outer,
                                         std::size_t normalized_size,
                                         double eps,
                                         const Shape& x_shape,
                                         Dtype dt) = 0;

    // RMSNorm backward; returns (grad_x, grad_gamma).
    virtual StoragePair rms_norm_backward(const Storage& x,
                                          const Storage& gamma,
                                          const Storage& saved_rstd,
                                          const Storage& grad,
                                          std::size_t outer,
                                          std::size_t normalized_size,
                                          const Shape& x_shape,
                                          const Shape& gamma_shape,
                                          Dtype dt,
                                          GradMask2 grad_mask = {}) = 0;

    // LayerNorm forward.  Returns [output, saved_mean, saved_rstd].
    virtual std::vector<Storage> layer_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    std::size_t outer,
                                                    std::size_t normalized_size,
                                                    double eps,
                                                    const Shape& x_shape,
                                                    Dtype dt) = 0;

    // LayerNorm backward; returns [grad_x, grad_gamma, grad_beta].
    virtual std::vector<Storage> layer_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& grad,
                                                     std::size_t outer,
                                                     std::size_t normalized_size,
                                                     const Shape& x_shape,
                                                     const Shape& gamma_shape,
                                                     const Shape& beta_shape,
                                                     Dtype dt,
                                                     GradMask3 grad_mask = {}) = 0;
    // BatchNorm training forward.  ndim distinguishes 1-D, 2-D, and 3-D spatial
    // tensors.  Returns [output, saved_mean, saved_rstd, saved_xnorm].  The
    // 4th element (xnorm = (x - mean) * rstd) is the normalised input that
    // backward needs; the MLX forward path already materialises it as a lazy
    // intermediate, so exposing it costs nothing at forward time and saves
    // a recomputation at backward time.  Backends without a meaningful
    // xnorm intermediate (e.g. the MPSGraph dispatch) may return an empty
    // Storage for slot 3 — the backward will then fall back to recomputing.
    virtual std::vector<Storage> batch_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    int batch,
                                                    int channels,
                                                    int spatial,
                                                    int ndim,
                                                    double eps,
                                                    const Shape& x_shape,
                                                    Dtype dt) = 0;

    // BatchNorm backward; returns [grad_x, grad_gamma, grad_beta].
    // `eps` matches the forward's epsilon — GPU backend needs it to
    // reconstruct variance from saved_rstd (var = 1/rstd^2 - eps) for
    // the MPSGraph normalizationGradient* path.  CPU backend may ignore.
    //
    // 3.4+ Phase A.4: ``saved_xnorm`` optionally carries the forward's
    // normalised input ``(x - mean) * rstd`` as a full-size tensor (same
    // shape as ``x``).  When present (i.e. non-empty Storage) the MLX-path
    // backward consumes it directly and skips two element-wise ops per
    // BN backward (centered = x - mean; xnorm = centered * rstd).  Passing
    // an empty Storage triggers the prior recomputation path — required
    // for the MPSGraph dispatch, which has its own xnorm management.
    virtual std::vector<Storage> batch_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& saved_xnorm,
                                                     const Storage& grad,
                                                     int batch,
                                                     int channels,
                                                     int spatial,
                                                     int ndim,
                                                     const Shape& x_shape,
                                                     Dtype dt,
                                                     double eps,
                                                     GradMask3 grad_mask = {}) = 0;

    // GroupNorm forward; groups partitions channels into equal groups before
    // normalizing each group independently.  Returns [output, saved_mean, saved_rstd].
    virtual std::vector<Storage> group_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    int batch,
                                                    int channels,
                                                    int spatial,
                                                    int groups,
                                                    const std::vector<int>& spatial_dims,
                                                    double eps,
                                                    const Shape& x_shape,
                                                    Dtype dt) = 0;

    // GroupNorm backward; returns [grad_x, grad_gamma, grad_beta].
    virtual std::vector<Storage> group_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& grad,
                                                     int batch,
                                                     int channels,
                                                     int spatial,
                                                     int groups,
                                                     const std::vector<int>& spatial_dims,
                                                     const Shape& x_shape,
                                                     Dtype dt,
                                                     GradMask3 grad_mask = {}) = 0;
    // Linear algebra ops.  All accept batched inputs where the last two dims
    // form the matrix.  CPU implementations delegate to LAPACK (via Accelerate);
    // GPU implementations delegate to MLX linalg.

    // Vector/matrix/Frobenius norm; ord controls the norm type.
    virtual Storage linalg_norm(const Storage& a,
                                const Shape& shape,
                                double ord,
                                const std::vector<int>& axes,
                                bool keepdims,
                                Dtype dt) = 0;

    // Entrywise ``(sum |a|^p)^(1/p)`` over ``axes`` (distinct, in range; empty
    // means every axis) for a finite ``p > 1``, accumulated relative to each
    // slice's largest magnitude so no power leaves the float range.  Only the
    // GPU stream has the kernel; the default throws not_implemented.
    virtual Storage p_norm_forward(const Storage& a,
                                   const Shape& shape,
                                   double p,
                                   const std::vector<int>& axes,
                                   bool keepdims,
                                   Dtype dt) {
        (void)a;
        (void)shape;
        (void)p;
        (void)axes;
        (void)keepdims;
        (void)dt;
        ErrorBuilder("IBackend::p_norm_forward")
            .not_implemented("the scaled p-norm is only implemented on the GPU backend");
        return {};
    }

    // Cholesky decomposition.  upper=true returns the upper triangular factor.
    virtual Storage linalg_cholesky(const Storage& a, const Shape& shape, bool upper, Dtype dt) = 0;

    // Matrix inverse using LU factorisation.
    virtual Storage linalg_inv(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Solves a linear system Ax = b; b may have multiple right-hand sides.
    virtual Storage linalg_solve(const Storage& a,
                                 const Storage& b,
                                 const Shape& a_shape,
                                 const Shape& b_shape,
                                 Dtype dt) = 0;

    // A^power computed via repeated squaring or eigendecomposition.
    virtual Storage
    linalg_matrix_power(const Storage& a, const Shape& shape, int power, Dtype dt) = 0;

    // Moore-Penrose pseudoinverse via SVD.
    virtual Storage linalg_pinv(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Determinant (scalar per batch element).
    virtual Storage linalg_det(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Reduced QR decomposition; returns (Q, R).
    virtual StoragePair linalg_qr(const Storage& a,
                                  const Shape& shape,
                                  const Shape& q_shape,
                                  const Shape& r_shape,
                                  Dtype dt) = 0;

    // General eigendecomposition for non-symmetric matrices; returns (eigenvalues, eigenvectors).
    // Eigenvalues may be complex; CPU backend returns packed real+imag components.
    virtual StoragePair linalg_eig(const Storage& a,
                                   const Shape& shape,
                                   const Shape& values_shape,
                                   const Shape& vectors_shape,
                                   Dtype dt) = 0;

    // Symmetric/Hermitian eigendecomposition (real eigenvalues guaranteed).
    // Returns (eigenvalues, eigenvectors); eigenvectors are column-ordered.
    // Reads the upper triangle when ``upper`` is true, the lower triangle otherwise.
    virtual StoragePair linalg_eigh(const Storage& a,
                                    const Shape& shape,
                                    const Shape& values_shape,
                                    const Shape& vectors_shape,
                                    Dtype dt,
                                    bool upper = false) = 0;

    // Singular value decomposition.  compute_uv=false returns only singular
    // values S.  Returns [U, S, Vt] or [S] depending on compute_uv.
    virtual std::vector<Storage> linalg_svd(const Storage& a,
                                            const Shape& shape,
                                            bool compute_uv,
                                            const Shape& u_shape,
                                            const Shape& s_shape,
                                            const Shape& vt_shape,
                                            Dtype dt) = 0;

    // LU factorisation (packed format).  Returns {LU, pivots, info}: LU the
    // packed matrix (LAPACK dgetrf_ format), pivots I32 1-based of shape
    // batch + min(m, n), info I32 per matrix — getrf's own code, 0 or the
    // first exactly-zero pivot (a singular matrix is returned, not raised).
    virtual std::vector<Storage>
    linalg_lu_factor(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Triangular solve: solve A X = B where A is triangular.
    // upper=true → upper triangular; unit=true → unit diagonal.
    virtual Storage linalg_solve_triangular(const Storage& a,
                                            const Storage& b,
                                            const Shape& a_shape,
                                            const Shape& b_shape,
                                            bool upper,
                                            bool unitriangular,
                                            Dtype dt) = 0;

    // Least-squares: solve min||AX-B||_2. Returns [solution, residuals, rank, svd].
    // a_shape=(m,n), b_shape=(m,nrhs). Solution shape=(n,nrhs).
    virtual std::vector<Storage> linalg_lstsq(const Storage& a,
                                              const Storage& b,
                                              const Shape& a_shape,
                                              const Shape& b_shape,
                                              Dtype dt) = 0;

    // Solve AX=B given LU+pivots from linalg_lu_factor. Returns X (same shape as B).
    virtual Storage linalg_lu_solve(const Storage& LU,
                                    const Storage& pivots,
                                    const Storage& b,
                                    const Shape& lu_shape,
                                    const Shape& b_shape,
                                    Dtype dt) = 0;

    // Reconstruct Q (m×k) from Householder reflectors H (m×n) and tau (k,).
    virtual Storage linalg_householder_product(const Storage& H,
                                               const Storage& tau,
                                               const Shape& h_shape,
                                               Dtype dt) = 0;

    // LDL^T factorization of symmetric matrix.  Returns {LD_packed,
    // pivots_i32, info_i32}: info is sytrf's own code per matrix (a singular
    // D is returned, not raised).
    virtual std::vector<Storage>
    linalg_ldl_factor(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Broadcasts `a` from src_shape to dst_shape following NumPy rules.
    virtual Storage
    broadcast(const Storage& a, const Shape& src_shape, const Shape& dst_shape, Dtype dt) = 0;

    // Repeats elements `repeats` times along `axis` (equivalent to np.repeat).
    virtual Storage
    repeat(const Storage& a, const Shape& shape, Dtype dt, std::int64_t repeats, int axis) = 0;

    // Tiles the tensor by replicating it reps[i] times along each dimension.
    virtual Storage
    tile(const Storage& a, const Shape& shape, Dtype dt, const std::vector<std::int64_t>& reps) = 0;

    // Permutes dimensions according to `perm` (equivalent to np.transpose).
    virtual Storage
    permute(const Storage& a, const Shape& shape, const std::vector<int>& perm, Dtype dt) = 0;

    // Zero-pads (or reflects/wraps) the tensor with pad_width[d] = (before, after)
    // extra elements for each dimension d.  constant specifies the fill value.
    virtual Storage pad(const Storage& a,
                        const Shape& shape,
                        Dtype dt,
                        const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                        double constant) = 0;

    // Raises each element to a scalar exponent: a**exp.
    virtual Storage pow_scalar(const Storage& a, const Shape& shape, Dtype dt, double exp) = 0;

    // Raises a scalar base to the power of each element: base**a.
    virtual Storage rpow_scalar(const Storage& a, const Shape& shape, Dtype dt, double base) = 0;

    // Clamps each element to [min_v, max_v].
    virtual Storage
    clip(const Storage& a, const Shape& shape, Dtype dt, double min_v, double max_v) = 0;

    // Type-casts the buffer element-by-element from src_dt to dst_dt.
    virtual Storage cast(const Storage& a, const Shape& shape, Dtype src_dt, Dtype dst_dt) = 0;

    // Loss functions.  reduction: 0=elementwise (no reduction), 1=mean, 2=sum.
    // Backward passes return one gradient tensor per differentiable input.

    // Mean-squared error: 0.5*(input - target)^2 summed/averaged.
    // Backward returns (grad_input, grad_target).
    virtual Storage mse_loss(const Storage& input,
                             const Storage& target,
                             const Shape& shape,
                             Dtype dt,
                             int reduction) = 0;
    virtual std::pair<Storage, Storage> mse_loss_backward(const Storage& input,
                                                          const Storage& target,
                                                          const Storage& grad,
                                                          const Shape& shape,
                                                          Dtype dt,
                                                          int reduction) = 0;

    // Huber loss (smooth L1): quadratic for |r|<=delta, linear otherwise.
    // delta controls the transition point between the two regimes.
    // Backward returns (grad_input, grad_target).
    virtual Storage huber_loss(const Storage& input,
                               const Storage& target,
                               const Shape& shape,
                               Dtype dt,
                               double delta,
                               int reduction) = 0;
    virtual std::pair<Storage, Storage> huber_loss_backward(const Storage& input,
                                                            const Storage& target,
                                                            const Storage& grad,
                                                            const Shape& shape,
                                                            Dtype dt,
                                                            double delta,
                                                            int reduction) = 0;
    virtual Storage bce_loss(const Storage& input,
                             const Storage& target,
                             const Storage& weight,
                             const Shape& shape,
                             Dtype dt,
                             double eps,
                             int reduction) = 0;
    virtual std::vector<Storage> bce_loss_backward(const Storage& input,
                                                   const Storage& target,
                                                   const Storage& weight,
                                                   const Storage& grad,
                                                   const Shape& shape,
                                                   Dtype dt,
                                                   double eps,
                                                   int reduction) = 0;
    virtual Storage bce_with_logits_loss(const Storage& input,
                                         const Storage& target,
                                         const Storage& weight,
                                         const Storage& pos_weight,
                                         const Shape& shape,
                                         const Shape& weight_shape,
                                         const Shape& pos_weight_shape,
                                         Dtype dt,
                                         int reduction) = 0;
    virtual std::vector<Storage> bce_with_logits_backward(const Storage& input,
                                                          const Storage& target,
                                                          const Storage& weight,
                                                          const Storage& pos_weight,
                                                          const Storage& grad,
                                                          const Shape& shape,
                                                          Dtype dt,
                                                          int reduction) = 0;

    // Cross-entropy loss with optional per-class weights and ignore_index.
    // saved_aux inside the result holds the log-softmax activations needed by
    // the backward pass.  weight is optional (may be nullptr for unweighted CE).
    virtual ClassLossForwardResult cross_entropy_loss(const Storage& input,
                                                      const Storage& target,
                                                      const Storage* weight,
                                                      const Shape& input_shape,
                                                      const Shape& target_shape,
                                                      Dtype dt,
                                                      double eps,
                                                      int ignore_index,
                                                      int reduction) = 0;

    // Cross-entropy backward; saved_softmax is the softmax output stored during
    // the forward.  valid_count accounts for ignored-index samples in mean reduction.
    virtual Storage cross_entropy_backward(const Storage& saved_softmax,
                                           const Storage& target,
                                           const Storage* weight,
                                           const Storage& valid_count,
                                           const Storage& grad,
                                           const Shape& input_shape,
                                           Dtype dt,
                                           int ignore_index,
                                           int reduction) = 0;

    // Negative log-likelihood loss; operates on log-probabilities directly
    // (unlike cross_entropy which applies log-softmax internally).
    virtual ClassLossForwardResult nll_loss(const Storage& input,
                                            const Storage& target,
                                            const Storage* weight,
                                            const Shape& input_shape,
                                            const Shape& target_shape,
                                            Dtype dt,
                                            int ignore_index,
                                            int reduction) = 0;

    // NLL backward; returns the input gradient.
    virtual Storage nll_loss_backward(const Storage& target,
                                      const Storage* weight,
                                      const Storage& valid_count,
                                      const Storage& grad,
                                      const Shape& input_shape,
                                      Dtype dt,
                                      int ignore_index,
                                      int reduction) = 0;

    // Materialises the tensor's data in CPU-accessible memory.
    //
    // Parameters
    // ----------
    // a : const Storage&
    //     Backend-native storage to materialise.
    // shape : const Shape&
    //     Logical shape of ``a``.
    //
    // Returns
    // -------
    // CpuStorage
    //     Host-side aligned buffer containing the evaluated data.
    //
    // Notes
    // -----
    // For :class:`CpuBackend` this is effectively a copy of the existing
    // ``CpuStorage``; for :class:`GpuBackend` it calls
    // ``mlx::core::array::eval()`` first (triggering execution of the
    // pending lazy graph) and then copies the result into a fresh
    // ``CpuStorage`` on the host.
    virtual CpuStorage to_cpu(const Storage& a, const Shape& shape) = 0;

    // Scaled dot-product attention forward.  Returns [output, saved_weights]
    // where saved_weights holds the post-softmax attention scores for backward.
    // is_causal=true applies a lower-triangular mask to prevent attending to
    // future tokens without materialising the full mask tensor.
    // ``need_weights`` selects the return contract: ``false`` uses the
    // memory-efficient fused kernel and returns a placeholder weights slot;
    // ``true`` materializes the dense softmax weight matrix (O(T²)) for callers
    // that inspect it.  The CPU reference always materializes W regardless.
    virtual std::vector<Storage> sdpa_forward(const Storage& q,
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
                                              Dtype dt) = 0;

    // SDPA backward; returns [grad_q, grad_k, grad_v].
    //
    // ``attn_mask`` / ``mask_dtype`` / ``is_causal`` mirror ``sdpa_forward`` so a
    // backend that recomputes the forward (e.g. the MLX VJP path) can reproduce
    // the exact masked attention.  ``attn_mask`` is ``nullptr`` when absent; a
    // ``Bool`` ``mask_dtype`` marks a keep-mask, any other dtype an additive mask.
    virtual std::vector<Storage> sdpa_backward(const Storage& grad_out,
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
                                               Dtype dt) = 0;

    // Compact parameter bag for N-D transposed convolution.
    //
    // Mirrors :class:`ConvNdOpts` and adds ``opad``.  A transposed
    // convolution *is* the data gradient of a forward convolution, so
    // ``stride`` / ``pad`` / ``dilation`` / ``groups`` describe that
    // forward convolution — the one whose input this op reconstructs —
    // and ``opad`` picks one output extent out of the ``stride``
    // candidates an integer-dividing stride leaves ambiguous.
    //
    // Attributes
    // ----------
    // N : int
    //     Number of spatial dimensions (1, 2, or 3).
    // groups : int
    //     Channel grouping factor; ``1`` is the dense case and ``Cin``
    //     is depthwise.  Unlike ``ConvNdOpts``, the divided axis of the
    //     weight is the *second* one: ``(Cin, Cout / groups, K...)``.
    // stride, pad, dilation, opad : int[3]
    //     Per-spatial-dimension geometry (only the first ``N`` entries
    //     are meaningful).
    struct ConvTransposeNdOpts {
        int N;       // number of spatial dimensions
        int groups;  // channel grouping factor for grouped transposed convolution
        int stride[3];
        int pad[3];
        int dilation[3];
        int opad[3];
    };

    virtual Storage conv_transpose_nd_forward(const Storage& x,
                                              const Storage& W,
                                              const Storage& b,
                                              int B,
                                              int Cin,
                                              int Cout,
                                              const int* S,
                                              const int* K,
                                              const int* O,
                                              const ConvTransposeNdOpts& opts,
                                              const Shape& out_shape,
                                              Dtype dt) = 0;

    virtual std::vector<Storage> conv_transpose_nd_backward(const Storage& grad_out,
                                                            const Storage& x,
                                                            const Storage& W,
                                                            int B,
                                                            int Cin,
                                                            int Cout,
                                                            const int* S,
                                                            const int* K,
                                                            const int* O,
                                                            const ConvTransposeNdOpts& opts,
                                                            Dtype dt,
                                                            GradMask3 grad_mask = {}) = 0;

    // Compact parameter bag for N-D standard convolution.
    //
    // Used by :func:`IBackend::conv_nd_forward` and
    // :func:`conv_nd_backward` for ``N`` in ``{1, 2, 3}``.  The
    // fixed-size arrays are indexed ``0..N-1``; any unused trailing
    // entries are ignored by the implementation.
    //
    // Attributes
    // ----------
    // N : int
    //     Number of spatial dimensions (1, 2, or 3).
    // groups : int
    //     Channel grouping factor; ``1`` means standard convolution and
    //     ``Cin`` means depthwise.
    // stride, pad, dilation : int[3]
    //     Per-spatial-dimension geometry (only the first ``N`` entries
    //     are meaningful).
    struct ConvNdOpts {
        int N;       // number of spatial dimensions
        int groups;  // channel grouping factor for grouped convolution
        int stride[3];
        int pad[3];
        int dilation[3];
    };

    virtual Storage conv_nd_forward(const Storage& x,
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
                                    Dtype dt) = 0;

    virtual std::vector<Storage> conv_nd_backward(const Storage& grad_out,
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
                                                  Dtype dt,
                                                  GradMask3 grad_mask = {}) = 0;

    virtual Storage unfold_forward(const Storage& x,
                                   int B,
                                   int C,
                                   const std::vector<int>& S,
                                   const std::vector<int>& K,
                                   const std::vector<int>& O,
                                   const std::vector<int>& stride,
                                   const std::vector<int>& pad,
                                   const std::vector<int>& dilation,
                                   const Shape& out_shape,
                                   Dtype dt) = 0;

    virtual Storage unfold_backward(const Storage& grad_out,
                                    int B,
                                    int C,
                                    const std::vector<int>& S,
                                    const std::vector<int>& K,
                                    const std::vector<int>& O,
                                    const std::vector<int>& stride,
                                    const std::vector<int>& pad,
                                    const std::vector<int>& dilation,
                                    Dtype dt) = 0;

    virtual std::pair<Storage, Storage> expand_and_multiply(const Storage& mask,
                                                            const Storage& x,
                                                            const Shape& mask_shape,
                                                            const Shape& x_shape,
                                                            Dtype dt) = 0;

    virtual Storage drop_block_mask(const Storage& seed,
                                    double drop_prob,
                                    std::int64_t block_size,
                                    const Shape& x_shape,
                                    Dtype dt) = 0;

    // Embedding table lookup: out[i] = weight[indices[i]], for every index.
    // padding_idx does not mask the lookup — it only keeps that row's
    // gradient out of embedding_backward.
    virtual Storage embedding_forward(const Storage& weight,
                                      const Storage& indices,
                                      const Shape& weight_shape,
                                      const Shape& indices_shape,
                                      const Shape& out_shape,
                                      int padding_idx,
                                      Dtype dt) = 0;

    // Backward pass for embedding: scatter-adds grad_out rows into a
    // zero-filled weight gradient of weight_shape.
    virtual Storage embedding_backward(const Storage& grad_out,
                                       const Storage& indices,
                                       const Shape& weight_shape,
                                       const Shape& indices_shape,
                                       int padding_idx,
                                       Dtype dt) = 0;

    // Fold (col2im): inverse of unfold/im2col. Accumulates (N, C*kH*kW, L)
    // patches back into (N, C, outH, outW) using scatter-add.
    virtual Storage nn_fold(const Storage& x,
                            const Shape& x_shape,
                            const Shape& out_shape,
                            const std::vector<int>& kernel_size,
                            const std::vector<int>& stride,
                            const std::vector<int>& padding,
                            const std::vector<int>& dilation,
                            Dtype dt) = 0;

    // EmbeddingBag: gather rows from weight at indices, then reduce per bag.
    // mode: 0=sum, 1=mean, 2=max. For 1-D indices, offsets marks bag starts.
    virtual Storage embedding_bag_forward(const Storage& weight,
                                          const Storage& indices,
                                          const Storage& offsets,
                                          const Shape& weight_shape,
                                          const Shape& indices_shape,
                                          int mode,
                                          int padding_idx,
                                          bool include_last_offset,
                                          Dtype dt) = 0;

    // EmbeddingBag backward: accumulate the incoming gradient back onto the
    // rows each bag gathered.  ``sum`` sends the bag's gradient to every row
    // it touched, ``mean`` divides it by the bag size first, and ``max``
    // sends it only to the row that won each column — which is why the
    // weight is a parameter here and not in ``embedding_backward``: the
    // argmax is not recorded by the forward and has to be recomputed.
    virtual Storage embedding_bag_backward(const Storage& grad_out,
                                           const Storage& weight,
                                           const Storage& indices,
                                           const Storage& offsets,
                                           const Shape& weight_shape,
                                           const Shape& indices_shape,
                                           int mode,
                                           int padding_idx,
                                           bool include_last_offset,
                                           Dtype dt) = 0;

    // Flip (reverse) along the given axes.
    virtual Storage
    flip(const Storage& a, const Shape& shape, const std::vector<int>& dims, Dtype dt) = 0;

    // Masked select: extract elements where bool mask == true.
    // Returns a flat 1-D Storage of `n_true` elements.
    virtual Storage masked_select_count(const Storage& mask, const Shape& shape, Dtype dt) = 0;
    virtual Storage masked_select(const Storage& a,
                                  const Storage& mask,
                                  const Shape& a_shape,
                                  const Shape& mask_shape,
                                  std::int64_t n_true,
                                  Dtype dt) = 0;

    // CTC (Connectionist Temporal Classification) loss.
    // log_probs: (T, N, C) log-probabilities; targets: (N, S) or flat (sum_S,)
    // input_lengths: (N,); target_lengths: (N,).
    // Returns per-sample losses of shape (N,) before reduction.
    virtual Storage ctc_loss_forward(const Storage& log_probs,
                                     const Storage& targets,
                                     const Storage& input_lengths,
                                     const Storage& target_lengths,
                                     const Shape& lp_shape,
                                     int blank,
                                     bool zero_infinity,
                                     Dtype dt) = 0;

    // Gradient of each sample's CTC loss with respect to log_probs, (T, N, C).
    // grad_out: (N,) upstream gradient of every per-sample loss.  Follows the
    // reference framework's formula, exp(lp) - exp(log(alpha * beta) + nll - lp):
    // the true gradient with respect to the logits once a log_softmax's
    // backward is applied, which is the only way the loss is meant to be fed.
    // Time steps past a sample's input length get zero; an infeasible
    // alignment gets zero under zero_infinity.
    virtual Storage ctc_loss_backward(const Storage& log_probs,
                                      const Storage& targets,
                                      const Storage& input_lengths,
                                      const Storage& target_lengths,
                                      const Storage& grad_out,
                                      const Shape& lp_shape,
                                      int blank,
                                      bool zero_infinity,
                                      Dtype dt) = 0;

    // Generates a fixed sinusoidal positional encoding of shape
    // (seq_len, embed_dim) following the original Transformer paper formula.
    virtual Storage
    sinusoidal_pos_embedding(std::int64_t seq_len, std::int64_t embed_dim, Dtype dt) = 0;

    // Rotary Position Embedding (RoPE) forward.  Returns [output, cos_cache, sin_cache].
    // interleaved=true uses GPT-NeoX layout; false uses Llama layout.
    // position_ids is optional; if null, positions 0..seq_len-1 are assumed.
    virtual std::vector<Storage> rope_forward(const Storage& x,
                                              const Storage* position_ids,
                                              const Shape& x_shape,
                                              bool interleaved,
                                              Dtype pos_dtype,
                                              Dtype dt) = 0;

    // RoPE backward; applies the inverse rotation using cached cos/sin values.
    virtual Storage rope_backward(const Storage& grad_out,
                                  const Storage& saved_cos,
                                  const Storage& saved_sin,
                                  const Shape& x_shape,
                                  bool interleaved,
                                  Dtype dt) = 0;

    // Compact parameter bag for N-D pooling.
    //
    // Used by :func:`max_pool_nd_forward`, :func:`avg_pool_nd_forward`,
    // and their backward counterparts for ``N`` in ``{1, 2, 3}``.
    //
    // Attributes
    // ----------
    // K : int[3]
    //     Kernel size per spatial dimension.
    // stride : int[3]
    //     Stride per spatial dimension.
    // pad : int[3]
    //     Symmetric (both-sides) padding per spatial dimension.
    // N : int
    //     Number of spatial dimensions actually used.
    struct PoolOpts {
        int K[3];       // kernel size per spatial dimension
        int stride[3];  // stride per spatial dimension
        int pad[3];     // symmetric padding per spatial dimension
        int N;          // number of spatial dimensions
        // Ceiling output sizing.  The caller has already baked this into
        // ``out_shape``; backends need it only to know that the final window
        // may overhang the padded input and must be given room (max) or
        // excluded from the divisor (avg).
        bool ceil_mode = false;
        // Average pooling only: whether zero-padding counts toward the
        // divisor.  The ceil-mode overhang never counts either way, because
        // the window end is clamped to ``S + pad``.
        bool count_include_pad = true;
    };

    // N-D max pooling forward.  Returns [output, argmax]; argmax stores the
    // flat index within the spatial window that produced each maximum, required
    // by max_pool_nd_backward to route gradients.
    virtual std::vector<Storage> max_pool_nd_forward(const Storage& x,
                                                     const Shape& x_shape,
                                                     const Shape& out_shape,
                                                     const PoolOpts& opts,
                                                     Dtype dt) = 0;

    // N-D max pooling backward; scatters grad_out to the argmax positions.
    virtual Storage max_pool_nd_backward(const Storage& grad_out,
                                         const Storage& saved_argmax,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) = 0;

    // N-D average pooling forward (count-include-pad semantics by default).
    virtual Storage avg_pool_nd_forward(const Storage& x,
                                        const Shape& x_shape,
                                        const Shape& out_shape,
                                        const PoolOpts& opts,
                                        Dtype dt) = 0;

    // N-D average pooling backward; distributes grad uniformly across each window.
    virtual Storage avg_pool_nd_backward(const Storage& grad_out,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) = 0;

    // BatchNorm inference-mode forward; uses running statistics (mean, var)
    // rather than computing batch statistics.  Returns [output, rstd].
    virtual std::vector<Storage> batch_norm_eval_forward(const Storage& x,
                                                         const Storage& mean,
                                                         const Storage& var,
                                                         const Storage& gamma,
                                                         const Storage& beta,
                                                         const Shape& x_shape,
                                                         int C,
                                                         int spatial,
                                                         double eps,
                                                         Dtype dt) = 0;

    // BatchNorm inference-mode backward; returns [grad_x, grad_mean, grad_var, grad_gamma, grad_beta].
    virtual std::vector<Storage> batch_norm_eval_backward(const Storage& x,
                                                          const Storage& mean,
                                                          const Storage& gamma,
                                                          const Storage& rstd,
                                                          const Storage& grad_out,
                                                          const Shape& x_shape,
                                                          int C,
                                                          int spatial,
                                                          Dtype dt,
                                                          GradMask5 grad_mask = {}) = 0;

    // Lp-normalizes `x` along `axis` using the given norm order `ord`.
    // Returns [output, saved_norm] so the backward can reuse the computed norm.
    virtual std::vector<Storage> lp_normalize_forward(
        const Storage& x, const Shape& x_shape, double ord, int axis, double eps, Dtype dt) = 0;

    // Lp-normalize backward; saved_norm is the per-row norm from the forward.
    virtual Storage lp_normalize_backward(const Storage& x,
                                          const Storage& saved_norm,
                                          const Storage& grad_out,
                                          const Shape& x_shape,
                                          double ord,
                                          int axis,
                                          Dtype dt) = 0;

    // Global Response Normalization (GRN) from ConvNeXt-v2.
    // Returns [output, saved_Nx] where Nx is the per-channel L2 norm.
    virtual std::vector<Storage> global_response_norm_forward(const Storage& x,
                                                              const Storage& gamma,
                                                              const Storage& beta,
                                                              const Shape& x_shape,
                                                              double eps,
                                                              Dtype dt) = 0;

    // GRN backward; returns [grad_x, grad_gamma, grad_beta].
    virtual std::vector<Storage> global_response_norm_backward(const Storage& x,
                                                               const Storage& gamma,
                                                               const Storage& beta,
                                                               const Storage& saved_Nx,
                                                               const Storage& grad_out,
                                                               const Shape& x_shape,
                                                               double eps,
                                                               Dtype dt,
                                                               GradMask3 grad_mask = {}) = 0;

    // Nearest-neighbor upsampling for 2-D and 3-D feature maps.
    virtual Storage interpolate_nearest_2d_forward(
        const Storage& input, const Shape& in_shape, int H_out, int W_out, Dtype dt) = 0;

    virtual Storage interpolate_nearest_3d_forward(
        const Storage& input, const Shape& in_shape, int D_out, int H_out, int W_out, Dtype dt) = 0;

    // Nearest-neighbor upsampling backward: scatter-adds each output gradient
    // onto its unique source pixel/voxel using the same floor coordinate map
    // as the forward (a many-to-one mapping, so a source accumulates the sum
    // of every output gradient that selected it).
    virtual Storage interpolate_nearest_2d_backward(
        const Storage& grad_out, const Shape& in_shape, int H_out, int W_out, Dtype dt) = 0;

    virtual Storage interpolate_nearest_3d_backward(const Storage& grad_out,
                                                    const Shape& in_shape,
                                                    int D_out,
                                                    int H_out,
                                                    int W_out,
                                                    Dtype dt) = 0;

    // Bilinear interpolation forward/backward.
    // align_corners=true maps corner pixels to corner coordinates; false
    // scales by factor (in_size / out_size).
    virtual Storage interpolate_bilinear_forward(const Storage& input,
                                                 const Shape& in_shape,
                                                 int H_out,
                                                 int W_out,
                                                 bool align_corners,
                                                 Dtype dt) = 0;

    virtual Storage interpolate_bilinear_backward(const Storage& grad_out,
                                                  const Shape& in_shape,
                                                  int H_out,
                                                  int W_out,
                                                  bool align_corners,
                                                  Dtype dt) = 0;

    // Trilinear interpolation for volumetric feature maps.
    virtual Storage interpolate_trilinear_forward(const Storage& input,
                                                  const Shape& in_shape,
                                                  int D_out,
                                                  int H_out,
                                                  int W_out,
                                                  bool align_corners,
                                                  Dtype dt) = 0;

    virtual Storage interpolate_trilinear_backward(const Storage& grad_out,
                                                   const Shape& in_shape,
                                                   int D_out,
                                                   int H_out,
                                                   int W_out,
                                                   bool align_corners,
                                                   Dtype dt) = 0;

    // Affine grid generation for spatial transformer networks.
    // theta is an (N, 2, 3) affine matrix; output has shape (N, H, W, 2).
    virtual Storage affine_grid_forward(
        const Storage& theta, int N, int H, int W, bool align_corners, Dtype dt) = 0;

    // Affine grid backward: returns grad_theta of shape (N, 2, 3).
    virtual Storage affine_grid_backward(
        const Storage& grad_grid, int N, int H, int W, bool align_corners, Dtype dt) = 0;

    // Samples input at grid locations using bilinear (mode=0) or nearest (mode=1)
    // interpolation.  padding_mode: 0=zeros, 1=border, 2=reflection.
    virtual Storage grid_sample_forward(const Storage& input,
                                        const Storage& grid,
                                        const Shape& in_shape,
                                        const Shape& grid_shape,
                                        int mode,
                                        int padding_mode,
                                        bool align_corners,
                                        Dtype dt) = 0;

    // Grid sample backward; returns [grad_input, grad_grid].
    virtual std::vector<Storage> grid_sample_backward(const Storage& grad_out,
                                                      const Storage& input,
                                                      const Storage& grid,
                                                      const Shape& in_shape,
                                                      const Shape& grid_shape,
                                                      int mode,
                                                      int padding_mode,
                                                      bool align_corners,
                                                      Dtype dt) = 0;

    virtual Storage bilinear_layer_forward(const Storage& x1,
                                           const Storage& x2,
                                           const Storage& weight,
                                           const Storage& bias,
                                           bool has_bias,
                                           const Shape& x1_shape,
                                           const Shape& x2_shape,
                                           const Shape& w_shape,
                                           Dtype dt) = 0;

    virtual std::vector<Storage> bilinear_layer_backward(const Storage& grad_out,
                                                         const Storage& x1,
                                                         const Storage& x2,
                                                         const Storage& weight,
                                                         const Shape& x1_shape,
                                                         const Shape& x2_shape,
                                                         const Shape& w_shape,
                                                         bool has_bias,
                                                         Dtype dt,
                                                         GradMask4 grad_mask = {}) = 0;

    // One-hot encoding: produces a (num_classes,)-wide indicator tensor for
    // each index in `indices`.  Output dtype is typically F32 or I32.
    virtual Storage one_hot_forward(const Storage& indices,
                                    const Shape& indices_shape,
                                    int num_classes,
                                    Dtype out_dtype) = 0;

    // Rotates a 2-D image by angle_rad_neg (stored negated for efficiency)
    // around center (cx, cy) using bilinear interpolation.
    virtual Storage rotate_forward(const Storage& input,
                                   const Shape& shape,
                                   double angle_rad_neg,
                                   double cx,
                                   double cy,
                                   Dtype dt) = 0;

    // Element-wise masks that return 0.0/1.0 float results.
    // ge_mask: out[i] = (a[i] >= b[i]) ? 1 : 0
    // lt_mask: out[i] = (a[i] < b[i])  ? 1 : 0
    virtual Storage ge_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage lt_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    // Broadcasts a scalar to the array and adds/multiplies element-wise.
    virtual Storage add_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) = 0;
    virtual Storage mul_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) = 0;

    // Returns a 0/1 mask where a[i] is inside [lo, hi] -- closed when
    // ``inclusive``, open when not.  Both are needed: the reference's
    // ``clamp`` backward passes the gradient at the bound and its
    // ``hardtanh``/``relu6`` backward does not, so the two ops that share
    // this primitive want opposite ends of the same interval.  A NaN is
    // outside both (mask 0), as in the reference's vectorised kernels.
    virtual Storage in_range_mask(
        const Storage& a, const Shape& shape, Dtype dt, double lo, double hi, bool inclusive) = 0;

    // Returns 1 where a[i] > 0, else slope — 0 and NaN included, as the
    // reference's leaky-relu backward has it.
    virtual Storage leaky_mask(const Storage& a, const Shape& shape, Dtype dt, double slope) = 0;

    // Returns 1.0 where a[i] > 0, else 0.0.  Used in relu backward.
    virtual Storage positive_mask(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Fused ReLU backward: returns ``0`` where ``x <= 0``, else ``g`` — the
    // reference's test, so a NaN input passes its gradient.  Single-
    // kernel alternative to ``multiply(g, positive_mask(x))`` — collapses the
    // 3-op chain (greater + astype + multiply) into one MLX expression that
    // MLX can fuse natively (or that GPU dispatches as ``where``).  Falls back
    // to the 2-step composition on backends without a fused path.
    virtual Storage
    relu_backward(const Storage& g, const Storage& x, const Shape& shape, Dtype dt) = 0;

    // Reduces `grad` (of shape grad_shape) to target_shape by summing over the
    // broadcast dimensions.  Used in broadcast backward passes.
    virtual Storage reduce_grad_to_shape(const Storage& grad,
                                         const Shape& grad_shape,
                                         const Shape& target_shape,
                                         Dtype dt) = 0;

    // Broadcasts `grad` back to input_shape after a reduction.  The inverse of
    // reduce_sum when keepdims=false.
    virtual Storage broadcast_back_for_reduce(const Storage& grad,
                                              const Shape& grad_shape,
                                              const Shape& input_shape,
                                              const std::vector<int>& axes,
                                              bool keepdims,
                                              Dtype dt) = 0;

    // Allocates a tensor filled with fill_value.
    virtual Storage full(const Shape& shape, Dtype dt, double fill_value) = 0;

    // Constructs an M×N identity-like matrix with the diagonal at offset k.
    virtual Storage eye(std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt) = 0;

    // Extracts a 1-D diagonal at offset k from a 2-D matrix, or places a 1-D
    // vector on a 2-D diagonal.  out_shape is written with the result shape.
    virtual Storage
    diag(const Storage& v, const Shape& v_shape, std::int64_t k, Dtype dt, Shape& out_shape) = 0;

    // Applies a triangular mask: upper=true zeroes the lower triangle,
    // upper=false zeroes the upper triangle, relative to diagonal offset k.
    virtual Storage tri(const Storage& input, const Shape& shape, Dtype dt, int k, bool upper) = 0;

    // Integer floor division: out[i] = floor(a[i] / b[i]).
    virtual Storage floordiv(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    // Inner product generalised to batches: contracts the last axis of `a`
    // against the last axis of `b`.
    virtual Storage inner(const Storage& a,
                          const Storage& b,
                          const Shape& a_shape,
                          const Shape& b_shape,
                          const Shape& out_shape,
                          Dtype dt) = 0;

    // Permutes a CpuStorage buffer in-place and returns the result as a new
    // CpuStorage.  Used for data layout conversions before GPU upload.
    virtual CpuStorage permute_cpu(const CpuStorage& src,
                                   const Shape& src_shape,
                                   const std::vector<int>& perm,
                                   Dtype dt) = 0;

    // General tensor contraction (generalised matmul) along axes_a and axes_b.
    virtual Storage tensordot(const Storage& a,
                              const Storage& b,
                              const Shape& a_shape,
                              const Shape& b_shape,
                              const Shape& out_shape,
                              const std::vector<int>& axes_a,
                              const std::vector<int>& axes_b,
                              Dtype dt) = 0;

    // Element-wise selection: out[i] = cond[i] ? x[i] : y[i].
    virtual Storage where_op(
        const Storage& cond, const Storage& x, const Storage& y, const Shape& shape, Dtype dt) = 0;

    // Sums grad over the broadcast dimensions so that the result has
    // output_shape (the pre-broadcast shape).
    virtual Storage reduce_broadcast(const Storage& grad,
                                     const Shape& input_shape,
                                     const Shape& output_shape,
                                     Dtype dt) = 0;

    // Computes a histogram over `input` with `bins` equal-width buckets in
    // [lo, hi].  density=true normalises the result to a probability density.
    virtual Storage histogram_forward(const Storage& input,
                                      const Shape& input_shape,
                                      Dtype input_dtype,
                                      double lo,
                                      double hi,
                                      std::int64_t bins,
                                      bool density) = 0;

    // Returns a CpuStorage containing the flat indices of all nonzero elements.
    // numel_out is set to the number of nonzero elements found.
    virtual CpuStorage nonzero_forward(const Storage& input,
                                       const Shape& input_shape,
                                       Dtype input_dtype,
                                       std::size_t& numel_out) = 0;

    // Optional fused linear + activation kernels (gelu is the tanh form).  The
    // defaults throw; a backend overrides them only for the dtypes it really
    // fuses (CpuBackend: F32).  The ``_fused_linear_*`` bindings call these
    // for CPU F32 only and compose linear() with relu()/gelu() elsewhere.
    virtual Storage fused_linear_relu_forward(
        const Storage& x, const Storage& w, const Storage& b, const Shape& out_shape, Dtype dt) {
        (void)x;
        (void)w;
        (void)b;
        (void)out_shape;
        (void)dt;
        ErrorBuilder("IBackend::fused_linear_relu_forward")
            .not_implemented("fused linear+relu not implemented on this backend");
        return {};
    }

    virtual Storage fused_linear_gelu_forward(
        const Storage& x, const Storage& w, const Storage& b, const Shape& out_shape, Dtype dt) {
        (void)x;
        (void)w;
        (void)b;
        (void)out_shape;
        (void)dt;
        ErrorBuilder("IBackend::fused_linear_gelu_forward")
            .not_implemented("fused linear+gelu not implemented on this backend");
        return {};
    }

    // Draw a {0,1} Bernoulli keep-mask at rate ``keep_prob`` directly on-device,
    // deterministically seeded by ``key_seed`` (pulled from the framework
    // Generator so the mask is reproducible from the global seed).  Returned as
    // ``dt``-typed {0,1} values; the inverted-dropout ``1/keep_prob`` scaling is
    // applied by the caller.  Default: not implemented — the CPU path uses the
    // per-element Generator loop in ``bernoulli_mask_storage_shape``.  The GPU
    // backend overrides this to skip that scalar CPU loop + host->device upload,
    // which otherwise dominates Dropout cost (~19 ms for a single (32,128,768)
    // mask vs <1 ms on-device).
    virtual Storage
    bernoulli_mask(double keep_prob, const Shape& shape, Dtype dt, std::uint64_t key_seed) {
        (void)keep_prob;
        (void)shape;
        (void)dt;
        (void)key_seed;
        ErrorBuilder("IBackend::bernoulli_mask")
            .not_implemented("on-device bernoulli_mask not implemented on this backend");
        return {};
    }

    // On-device random fills, seeded by ``key_seed`` (pulled from the framework
    // Generator so results are reproducible from the global seed).  Same
    // motivation as :func:`bernoulli_mask` — avoid the per-element CPU loop +
    // host->device upload in ``random_*_storage`` (Helpers.cpp), which makes
    // weight init and any GPU-tensor RNG slow (e.g. randn(4096,4096) ~228 ms,
    // bert_base() init ~850 ms).  Default: not implemented (CPU keeps its
    // per-element fill); the GPU backend overrides via ``mlx::core::random``.
    virtual Storage
    random_uniform(const Shape& shape, double lo, double hi, Dtype dt, std::uint64_t key_seed) {
        (void)shape;
        (void)lo;
        (void)hi;
        (void)dt;
        (void)key_seed;
        ErrorBuilder("IBackend::random_uniform").not_implemented("not implemented on this backend");
        return {};
    }

    virtual Storage
    random_normal(const Shape& shape, double mean, double std, Dtype dt, std::uint64_t key_seed) {
        (void)shape;
        (void)mean;
        (void)std;
        (void)dt;
        (void)key_seed;
        ErrorBuilder("IBackend::random_normal").not_implemented("not implemented on this backend");
        return {};
    }

    virtual Storage
    random_bernoulli(const Shape& shape, double p, Dtype dt, std::uint64_t key_seed) {
        (void)shape;
        (void)p;
        (void)dt;
        (void)key_seed;
        ErrorBuilder("IBackend::random_bernoulli")
            .not_implemented("not implemented on this backend");
        return {};
    }

    virtual Storage random_randint(
        const Shape& shape, std::int64_t low, std::int64_t high, Dtype dt, std::uint64_t key_seed) {
        (void)shape;
        (void)low;
        (void)high;
        (void)dt;
        (void)key_seed;
        ErrorBuilder("IBackend::random_randint").not_implemented("not implemented on this backend");
        return {};
    }

    // Compiles and launches a user-provided Metal Shading Language (MSL) kernel.
    // Default implementation throws not_implemented; only GpuBackend overrides this.
    // grid and threads follow Metal's threadgroups / threadsPerThreadgroup convention.
    virtual Storage run_custom_metal_kernel(const std::string& kernel_source,
                                            const std::string& function_name,
                                            const std::vector<Storage>& inputs,
                                            const Shape& output_shape,
                                            Dtype output_dtype,
                                            const std::array<std::size_t, 3>& grid,
                                            const std::array<std::size_t, 3>& threads) {
        (void)kernel_source;
        (void)function_name;
        (void)inputs;
        (void)output_shape;
        (void)output_dtype;
        (void)grid;
        (void)threads;
        ErrorBuilder("IBackend::run_custom_metal_kernel")
            .not_implemented("Metal kernel execution is only supported on the GPU backend");
        return {};
    }

    // Cast elements to a different dtype; shape is unchanged.
    // CPU: element-wise static_cast loop.  GPU: mlx::core::astype.
    virtual Storage astype(const Storage& a, const Shape& shape, Dtype src_dt, Dtype dst_dt) = 0;

    // Moves `src` to MTLResourceStorageModeShared memory so that both CPU and
    // GPU can access it without a copy.  Default is a no-op; GpuBackend overrides.
    virtual Storage to_shared_storage(const Storage& src, const Shape&) { return src; }

    // Parameter bag for LSTM operations.
    //
    // Used by :func:`lstm_forward`, :func:`lstm_forward_train`, and
    // :func:`lstm_backward`.  ``weights`` is supplied separately as
    // ``std::vector<Storage>`` in the order
    // ``[W_ih, W_hh, b_ih, b_hh]`` per layer; bidirectional layers
    // double the count (forward block followed by reverse block).
    //
    // When ``proj_size > 0`` (the projected-LSTM / LSTMP variant) an
    // additional weight tensor ``W_hr`` of shape ``(proj_size,
    // hidden_size)`` is appended per layer so the order becomes
    // ``[W_ih, W_hh, b_ih, b_hh, W_hr]``.  The recurrent weight
    // ``W_hh`` then has its second axis sized ``proj_size`` instead of
    // ``hidden_size`` because the projected hidden state feeds the
    // next time step.  The cell state ``c_n`` keeps shape
    // ``hidden_size`` regardless.
    //
    // Attributes
    // ----------
    // input_size, hidden_size : int
    //     Input feature size and recurrent hidden size.
    // num_layers : int
    //     Number of stacked LSTM layers.
    // seq_len, batch_size : int
    //     Time and batch dimensions of the input sequence.
    // batch_first : bool
    //     If ``true`` the input is laid out as ``(batch, seq, input)``;
    //     otherwise ``(seq, batch, input)``.
    // bidirectional : bool
    //     Whether to run a reverse-direction stack alongside the forward.
    // has_bias : bool
    //     Whether the per-gate bias terms are present.
    // proj_size : int
    //     ``0`` ⇒ standard LSTM; ``> 0`` ⇒ projected (LSTMP) variant.
    struct LstmOpts {
        int input_size = 0;
        int hidden_size = 0;
        int num_layers = 1;
        int seq_len = 1;
        int batch_size = 1;
        bool batch_first = false;  // if true, input is (batch, seq, input_size)
        bool bidirectional = false;
        bool has_bias = true;
        int proj_size = 0;  // 0 ⇒ standard LSTM; >0 ⇒ projected LSTM
    };

    // LSTM inference forward.  Returns [output, hn, cn].
    // Default implementation throws not_implemented; concrete backends override.
    virtual std::vector<Storage> lstm_forward(const Storage& input,
                                              const Storage& h0,
                                              const Storage& c0,
                                              const std::vector<Storage>& weights,
                                              const LstmOpts& opts,
                                              const Shape& out_shape,
                                              Dtype dt) {
        (void)input;
        (void)h0;
        (void)c0;
        (void)weights;
        (void)opts;
        (void)out_shape;
        (void)dt;
        ErrorBuilder("IBackend::lstm_forward")
            .not_implemented("LSTM not supported on this backend");
        return {};
    }

    // LSTM training forward; saves gate and cell activations for BPTT.
    // Returns [output, hn, cn, gates_all, cells_all].
    virtual std::vector<Storage> lstm_forward_train(const Storage& input,
                                                    const Storage& h0,
                                                    const Storage& c0,
                                                    const std::vector<Storage>& weights,
                                                    const LstmOpts& opts,
                                                    Dtype dt) {
        (void)input;
        (void)h0;
        (void)c0;
        (void)weights;
        (void)opts;
        (void)dt;
        ErrorBuilder("IBackend::lstm_forward_train")
            .not_implemented("LSTM training not supported on this backend");
        return {};
    }

    // LSTM BPTT backward.  gates_all and cells_all are the saved activations
    // from lstm_forward_train.  Returns [dX, dh0, dc0, dW_ih, dW_hh, db_ih, db_hh].
    virtual std::vector<Storage> lstm_backward(const Storage& grad_output,
                                               const Storage& grad_hn,
                                               const Storage& grad_cn,
                                               const Storage& input,
                                               const Storage& h0,
                                               const std::vector<Storage>& weights,
                                               const Storage& gates_all,
                                               const Storage& cells_all,
                                               const LstmOpts& opts,
                                               Dtype dt) {
        (void)grad_output;
        (void)grad_hn;
        (void)grad_cn;
        (void)input;
        (void)h0;
        (void)weights;
        (void)gates_all;
        (void)cells_all;
        (void)opts;
        (void)dt;
        ErrorBuilder("IBackend::lstm_backward")
            .not_implemented("LSTM backward not supported on this backend");
        return {};
    }
};

}  // namespace backend
}  // namespace lucid
