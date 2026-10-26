// lucid/_C/autograd/Helpers.h
//
// Utility functions shared across the autograd subsystem.  Provides:
//   - Storage factory helpers (zeros, ones).
//   - In-place gradient accumulation.
//   - Element-wise arithmetic on Storage values (used by op backward passes).
//   - Comparison/masking operations needed for ReLU-family gradients.
//   - Random tensor generation (Bernoulli, uniform, normal, randint).
//   - Shape/axis normalisation helpers for reduction backward passes.
//   - Version-counter check for in-place mutation detection.
//
// All functions dispatch to the backend layer (Dispatcher) so that the same
// autograd logic works on both CPU (Accelerate / vDSP) and GPU (MLX).  The
// flat-shape wrappers below package the numel count as a 1-D Shape before
// forwarding to the Dispatcher to avoid needing the full N-D shape for
// element-wise operations.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "../core/Device.h"
#include "../core/Shape.h"
#include "../core/Storage.h"

namespace lucid {

class TensorImpl;

// Allocate a zero-filled :class:`Storage` on the requested device.
//
// Parameters
// ----------
// shape : const Shape&
//     N-D shape of the resulting storage.
// dtype : Dtype
//     Element dtype.
// device : Device
//     Target device (``CPU`` uses Accelerate; ``GPU`` allocates an MLX
//     array).
//
// Returns
// -------
// Storage
//     Newly allocated storage with every element equal to zero.
Storage make_zero_storage(const Shape& shape, Dtype dtype, Device device);

// Allocate a ones-filled :class:`Storage` on the requested device.
//
// Used by :func:`Engine::backward` as the default seed gradient when the
// caller does not supply an explicit ``grad`` argument.
//
// Parameters
// ----------
// shape : const Shape&
//     N-D shape of the resulting storage.
// dtype : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     Storage with every element equal to one.
Storage make_ones_storage(const Shape& shape, Dtype dtype, Device device);

// Reduce ``grad`` back to ``target_shape`` by summing over broadcast axes.
//
// Used in the backward pass of any binary op that broadcast one operand
// against the other.  The gradient initially has the broadcast (output)
// shape; this helper walks the trailing dimensions and sums along any
// axis where ``target_shape`` is size 1 or absent, so the result matches
// the original input shape ready to be accumulated into ``.grad``.
//
// Parameters
// ----------
// grad : const Storage&
//     Incoming gradient at the broadcast shape.
// grad_shape : const Shape&
//     Shape of ``grad``.
// target_shape : const Shape&
//     Original (pre-broadcast) input shape to reduce back to.
// dtype : Dtype
//     Working dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     Storage of shape ``target_shape`` with broadcast axes summed.
//
// Math
// ----
// For each output index :math:`i`,
//
// $$
// \text{out}[i] = \sum_{j \in \text{broadcast}(i)} \text{grad}[j].
// $$
//
// Notes
// -----
// No-op when ``grad_shape == target_shape``.
Storage reduce_grad_to_shape(const Storage& grad,
                             const Shape& grad_shape,
                             const Shape& target_shape,
                             Dtype dtype,
                             Device device);

// Accumulate ``src`` into ``dst`` element-wise (``dst += src``).
//
// Handles all three :class:`Storage` variants:
//
// - ``CpuStorage / CpuStorage`` — hand-written typed loop for
//   ``F32 / F64 / I32 / I64``.
// - ``GpuStorage / GpuStorage`` — MLX add followed by rewrap.
// - ``SharedStorage`` variants — obtain a ``CpuStorage`` view of the
//   shared region and delegate to the CPU path.
//
// Parameters
// ----------
// dst : Storage&
//     Accumulator buffer; modified in place.
// src : const Storage&
//     Buffer to add into ``dst``.
//
// Raises
// ------
// DtypeMismatch
//     ``dst`` and ``src`` have different element dtypes.
// DeviceMismatch
//     ``dst`` and ``src`` belong to different device categories.
//
// Notes
// -----
// The most common caller is :class:`AccumulateGrad`, which uses this to
// fold each newly-arrived gradient into a leaf's running ``.grad``.
void accumulate_into(Storage& dst, const Storage& src);

// Element-wise negation: returns ``-s``.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// numel : std::size_t
//     Number of elements (used to build a 1-D Shape for the dispatcher).
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``-s``.
Storage negate_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise multiplication: returns ``a * b``.
//
// Parameters
// ----------
// a, b : const Storage&
//     Input buffers (must match ``numel`` and ``dt``).
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding the element-wise product.
Storage
multiply_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);

// Element-wise division: returns ``a / b``.
//
// Parameters
// ----------
// a, b : const Storage&
//     Numerator / denominator buffers.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding the element-wise quotient.
Storage
divide_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);

// Element-wise addition: returns ``a + b``.
//
// Parameters
// ----------
// a, b : const Storage&
//     Input buffers.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding the element-wise sum.
Storage
add_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);

// Element-wise subtraction: returns ``a - b``.
//
// Parameters
// ----------
// a, b : const Storage&
//     Input buffers.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding the element-wise difference.
Storage
subtract_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);

// Element-wise square: returns ``s * s``.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding the element-wise square.
Storage square_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Allocate a fresh copy of ``s`` (deep clone).
//
// Parameters
// ----------
// s : const Storage&
//     Source buffer.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     Newly allocated buffer with the same contents as ``s``.
Storage clone_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise natural logarithm: returns :math:`\ln(s)`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer (assumed strictly positive).
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``ln(s)`` element-wise.
Storage log_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise power: returns ``base ** expo``.
//
// Parameters
// ----------
// base, expo : const Storage&
//     Base and exponent buffers.  Both must have ``numel`` elements.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding :math:`\text{base}^{\text{expo}}`.
Storage
pow_storage(const Storage& base, const Storage& expo, std::size_t numel, Dtype dt, Device device);

// Element-wise greater-or-equal mask: ``(a >= b) ? 1 : 0``.
//
// Parameters
// ----------
// a, b : const Storage&
//     Inputs to compare.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Output dtype the mask is cast to.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding the gated mask.
//
// Notes
// -----
// Used by max / clamp-style backward passes to route the gradient only
// through the "winning" element of each comparison.
Storage
ge_mask_storage(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);

// Element-wise less-than mask: ``(a < b) ? 1 : 0``.
//
// Parameters
// ----------
// a, b : const Storage&
//     Inputs to compare.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Output dtype the mask is cast to.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding the gated mask.
Storage
lt_mask_storage(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);

// Broadcast-add a scalar to every element: returns ``s + scalar``.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// scalar : double
//     Value added to every element.  Cast to ``dt`` internally.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``s + scalar`` element-wise.
Storage
add_scalar_storage(const Storage& s, double scalar, std::size_t numel, Dtype dt, Device device);

// Broadcast-multiply every element by a scalar: returns ``s * scalar``.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// scalar : double
//     Multiplier; cast to ``dt`` internally.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``s * scalar`` element-wise.
Storage
mul_scalar_storage(const Storage& s, double scalar, std::size_t numel, Dtype dt, Device device);

// The following unary operations are element-wise and follow the same
// signature (s, numel, dtype, device) -> Storage.  They are generated by
// the LUCID_UNARY_HELPER macro in the .cpp file and declared individually
// here for clarity.

// Element-wise exponential: returns :math:`e^{s}`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``exp(s)``.
Storage exp_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise square root: returns :math:`\sqrt{s}`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer (assumed non-negative).
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``sqrt(s)``.
Storage sqrt_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise absolute value: returns :math:`|s|`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``abs(s)``.
Storage abs_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise sign function: returns -1, 0, or +1.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage with values in {-1, 0, +1} matching the sign of ``s``.
Storage sign_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise reciprocal: returns :math:`1 / s`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer (non-zero).
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``1 / s``.
Storage reciprocal_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Trigonometric functions.

// Element-wise sine: returns :math:`\sin(s)`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer (radians).
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``sin(s)``.
Storage sin_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise cosine: returns :math:`\cos(s)`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer (radians).
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``cos(s)``.
Storage cos_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise tangent: returns :math:`\tan(s)`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer (radians).
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``tan(s)``.
Storage tan_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise arcsine: returns :math:`\arcsin(s)`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer with elements in ``[-1, 1]``.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``asin(s)`` in radians.
Storage asin_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise arccosine: returns :math:`\arccos(s)`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer with elements in ``[-1, 1]``.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``acos(s)`` in radians.
Storage acos_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise arctangent: returns :math:`\arctan(s)`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``atan(s)`` in radians.
Storage atan_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Hyperbolic functions.

// Element-wise hyperbolic sine: returns :math:`\sinh(s)`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``sinh(s)``.
Storage sinh_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise hyperbolic cosine: returns :math:`\cosh(s)`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``cosh(s)``.
Storage cosh_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Element-wise hyperbolic tangent: returns :math:`\tanh(s)`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding ``tanh(s)``.
Storage tanh_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Mask elements that fall inside the closed range ``[lo, hi]``.
//
// Returns ``1`` where ``lo <= s <= hi`` and ``0`` everywhere else, used
// by clamp / hardtanh backward passes to gate the gradient through the
// active region.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer being clamped.
// lo, hi : double
//     Inclusive lower / upper bounds.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Output dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding the {0, 1} gate.
Storage in_range_mask_storage(const Storage& s,
                              double lo,
                              double hi,
                              std::size_t numel,
                              Dtype dt,
                              Device device,
                              bool inclusive);

// Mask elements that are strictly positive: ``(s > 0) ? 1 : 0``.
//
// Used by the ReLU backward pass to gate the gradient through positive
// activations.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer (typically the ReLU's saved input).
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Output dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding the {0, 1} gate.
Storage positive_mask_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// LeakyReLU gradient mask: returns ``1`` where ``s >= 0`` else ``slope``.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer (LeakyReLU's saved input).
// slope : double
//     Negative-side slope; the LeakyReLU's ``negative_slope`` parameter.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Output dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage suitable for element-wise multiplication into
//     ``grad_output``.
Storage
leaky_mask_storage(const Storage& s, double slope, std::size_t numel, Dtype dt, Device device);

// Element-wise sigmoid: returns :math:`1 / (1 + e^{-s})`.
//
// Parameters
// ----------
// s : const Storage&
//     Input buffer.
// numel : std::size_t
//     Number of elements.
// dt : Dtype
//     Element dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage holding the logistic sigmoid of ``s``.
//
// Math
// ----
// $$
// \sigma(s) = \frac{1}{1 + e^{-s}}.
// $$
Storage sigmoid_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// Forward declaration needed by the bernoulli helpers below.
class Generator;

// Generate a Bernoulli dropout mask as a flat 1-D storage.
//
// Each element is independently set to ``1.0`` with probability
// ``keep_prob`` and to ``0.0`` otherwise.  The mask is generated on CPU
// using ``gen`` and then transferred to ``device`` via
// ``Dispatcher::from_cpu`` to keep the random number stream deterministic
// regardless of where the consuming op runs.
//
// Parameters
// ----------
// keep_prob : double
//     Probability of keeping each element (``1 - dropout_rate``).
// numel : std::size_t
//     Number of mask elements.
// dt : Dtype
//     Output dtype; ``F32`` or ``F64`` only.
// device : Device
//     Target device.
// gen : Generator&
//     Random-number generator owning the PRNG state.
//
// Returns
// -------
// Storage
//     New 1-D storage holding the dropout mask.
Storage bernoulli_mask_storage(
    double keep_prob, std::size_t numel, Dtype dt, Device device, Generator& gen);

// Generate a Bernoulli dropout mask matching an explicit N-D shape.
//
// Same semantics as :func:`bernoulli_mask_storage` but the returned
// storage carries ``shape`` directly instead of a flat 1-D shape.
//
// Parameters
// ----------
// keep_prob : double
//     Probability of keeping each element.
// shape : const Shape&
//     Target shape.
// dt : Dtype
//     Output dtype; ``F32`` or ``F64`` only.
// device : Device
//     Target device.
// gen : Generator&
//     Random-number generator.
//
// Returns
// -------
// Storage
//     New storage of shape ``shape`` holding the mask.
Storage bernoulli_mask_storage_shape(
    double keep_prob, const Shape& shape, Dtype dt, Device device, Generator& gen);

// Generate uniform random samples in the half-open interval ``[lo, hi)``.
//
// Samples are produced on the CPU using ``gen`` and transferred to
// ``device`` to make the stream deterministic and reproducible.
//
// Parameters
// ----------
// shape : const Shape&
//     Output shape.
// lo, hi : double
//     Half-open interval bounds.
// dt : Dtype
//     Output dtype; ``F32`` or ``F64`` only.
// device : Device
//     Target device.
// gen : Generator&
//     Random-number generator.
//
// Returns
// -------
// Storage
//     New storage of shape ``shape`` with uniform samples.
Storage random_uniform_storage(
    const Shape& shape, double lo, double hi, Dtype dt, Device device, Generator& gen);

// Generate normally distributed samples using the Box-Muller transform.
//
// Parameters
// ----------
// shape : const Shape&
//     Output shape.
// mean : double
//     Mean of the resulting normal distribution.
// std : double
//     Standard deviation (non-negative).
// dt : Dtype
//     Output dtype; ``F32`` or ``F64`` only.
// device : Device
//     Target device.
// gen : Generator&
//     Random-number generator.
//
// Returns
// -------
// Storage
//     New storage of shape ``shape`` with i.i.d.
//     :math:`\mathcal{N}(\text{mean}, \text{std}^2)` samples.
Storage random_normal_storage(
    const Shape& shape, double mean, double std, Dtype dt, Device device, Generator& gen);

// Generate Bernoulli samples with success probability ``p``.
//
// Parameters
// ----------
// shape : const Shape&
//     Output shape.
// p : double
//     Success probability for value ``1``; failure value is ``0``.
// dt : Dtype
//     Output dtype; ``F32`` or ``F64`` only.
// device : Device
//     Target device.
// gen : Generator&
//     Random-number generator.
//
// Returns
// -------
// Storage
//     New storage of shape ``shape`` with Bernoulli samples.
Storage
random_bernoulli_storage(const Shape& shape, double p, Dtype dt, Device device, Generator& gen);

// Generate uniform random integers in the half-open interval ``[low, high)``.
//
// Uses batched ``uint32`` generation from ``gen`` for throughput, then
// folds the raw bytes into the requested integer range.
//
// Parameters
// ----------
// shape : const Shape&
//     Output shape.
// low : std::int64_t
//     Inclusive lower bound.
// high : std::int64_t
//     Exclusive upper bound (``high > low``).
// dt : Dtype
//     Output dtype; ``I32`` or ``I64`` only.
// device : Device
//     Target device.
// gen : Generator&
//     Random-number generator.
//
// Returns
// -------
// Storage
//     New storage of shape ``shape`` with integer samples.
Storage random_randint_storage(const Shape& shape,
                               std::int64_t low,
                               std::int64_t high,
                               Dtype dt,
                               Device device,
                               Generator& gen);

// Resolve one signed ``dim`` against a tensor of rank ``ndim``.
//
// The rule every dim argument follows.  A 0-d tensor answers to ``dim`` 0
// and -1 as if it had one axis, as the reference does; the result is then 0
// and names no real axis, so an op that indexes ``shape()[dim]`` must take
// the 0-d case apart first (``ops/utils/ScalarAxis.h``).
//
// Raises
// ------
// IndexError
//     ``dim`` falls outside ``[-max(ndim, 1), max(ndim, 1))``.
int wrap_dim(int dim, int ndim, const char* op);

// Canonicalise a list of axis indices for an ``ndim``-rank tensor.
//
// Checks the list through :func:`distinct_dims`, then sorts it ascending.
// When ``axes`` is empty the function treats the call as "reduce over
// everything" and returns ``[0, ndim)``.  A 0-d tensor has no axis to
// reduce, so any accepted axis list comes back empty.
//
// Parameters
// ----------
// axes : const std::vector<int>&
//     Raw axis list; may contain negatives.
// ndim : int
//     Rank of the tensor the axes refer to.
// op : const char*
//     Op name the errors carry.
//
// Returns
// -------
// std::vector<int>
//     Sorted, all-positive axis list.
//
// Raises
// ------
// IndexError
//     Some entry of ``axes`` is refused by :func:`wrap_dim`.
// LucidError
//     Two entries name the same axis.
std::vector<int> normalize_axes(const std::vector<int>& axes, int ndim, const char* op);

// Wrap each entry of ``dims`` through :func:`wrap_dim`, keeping the order,
// and refuse a list that names one axis twice, as the reference does.  An
// op that walks the list (``flip``) would otherwise act on that axis twice
// and undo itself.
//
// Raises
// ------
// IndexError
//     Some entry of ``dims`` is refused by :func:`wrap_dim`.
// LucidError
//     Two entries name the same axis.
std::vector<int> distinct_dims(const std::vector<int>& dims, int ndim, const char* op);

// Compute the output shape of a reduction over selected axes.
//
// Reduced dimensions are removed by default; passing ``keepdims=true``
// leaves them in place as size-1 axes (matching reference-framework
// semantics) so the result broadcasts cleanly against the input.
//
// Parameters
// ----------
// input_shape : const Shape&
//     Shape of the tensor being reduced.
// axes : const std::vector<int>&
//     Pre-normalised axis list (see :func:`normalize_axes`).
// keepdims : bool
//     Whether to retain reduced axes as size 1.
//
// Returns
// -------
// Shape
//     Resulting shape after the reduction.
Shape reduce_output_shape(const Shape& input_shape, const std::vector<int>& axes, bool keepdims);

// Forward declaration so :func:`check_version_match` can reference the
// owning tensor implementation.
class TensorImpl;

// Verify that a saved tensor's version counter has not advanced.
//
// Compares the current ``version()`` of the tensor referenced by ``live``
// against ``saved_version``.  When they differ the saved data has been
// mutated in place since it was stashed for backward, which would silently
// corrupt the gradient — so a :class:`VersionMismatch` is raised naming
// the op and input that owns the broken contract.
//
// Parameters
// ----------
// live : const std::weak_ptr<TensorImpl>&
//     Weak reference to the still-live tensor.  When the referent has been
//     destroyed the check is skipped (no harm if it cannot be mutated
//     after the fact).
// saved_version : std::int64_t
//     Version counter value captured when the tensor was saved.
// op_name : std::string_view
//     Name of the op for diagnostic messages (e.g. ``"matmul"``).
// input_idx : std::size_t
//     Zero-based index of the offending input within ``op_name``.
//
// Raises
// ------
// VersionMismatch
//     The live counter differs from ``saved_version`` and the safety
//     override (:func:`is_mutation_on_saved_allowed`) is false.
void check_version_match(const std::weak_ptr<TensorImpl>& live,
                         std::int64_t saved_version,
                         std::string_view op_name,
                         std::size_t input_idx);

// The same check for a tensor the error names by ``what`` — a node's own
// output it reads back (``"scatter_amax output 0"``) rather than an input.
void check_version_match(const std::weak_ptr<TensorImpl>& live,
                         std::int64_t saved_version,
                         const std::string& what);

void check_version_match(const std::shared_ptr<VersionCounter>& sink,
                         std::int64_t saved_version,
                         const std::string& what);

// Whether a node kept an input's values for backward in the slot ``s``.
//
// The one rule that decides both which inputs a node's version check guards
// (:meth:`AutogradNode::validate_versions`) and which ones graph-mode
// backward can rebuild (:func:`forward_time_input`).  A backward that reads
// an input's values saves them, by value, in that input's slot; an empty
// slot means the formula reads the input's shape at most, so a write into
// the input after forward cannot change the gradient and is not refused —
// the reference's rule for ``add``, ``sum`` or a view.  Checking every input
// instead refused ``y[:, 1] = y[:, 0] + 5`` on the CPU, where the slice is a
// view whose version the write moves.  A zero-element input saves no bytes
// and is not guarded either: no write can change what it holds.
inline bool holds_saved_values(const Storage& s) noexcept {
    return storage_nbytes(s) > 0;
}

struct Edge;

// The tensor a node was handed at forward time, rebuilt — or ``nullptr``
// when ``live`` still is that tensor.
//
// ``live`` stopped being it when an in-place op that records a graph wrote
// into it: the op gives it a grad_fn of its own.  The write moves its
// version too, so this is reached past :func:`check_version_match` only
// when that check is waived (:func:`is_mutation_on_saved_allowed`), or when
// something gave ``live`` a grad_fn without writing into it — a custom
// ``Function`` that returns its input (see
// :meth:`Node::restore_saved_for_graph`).  The rebuilt tensor
// holds the values the node saved by value and sits where ``live`` sat when
// the node was recorded — on ``edge``, the edge the node recorded for it —
// so graph-mode backward computes what eager backward computes, and its own
// derivative flows to what produced those values rather than to the write.
//
// Parameters
// ----------
// live : const std::shared_ptr<TensorImpl>&
//     The tensor the node holds for graph-mode backward.
// values : const Storage&
//     What the node saved by value for the same input.  Empty when the
//     node's formula does not read the input's values.
// shape : const Shape&
//     The input's shape at forward time.
// edge : const Edge&
//     The edge the node recorded for the input.
// op_name : std::string_view
//     Name of the op, for the error message.
// input_idx : std::size_t
//     Zero-based index of the input within ``op_name``.
//
// Returns
// -------
// std::shared_ptr<TensorImpl>
//     The rebuilt tensor; ``nullptr`` when ``live`` is still the tensor the
//     node was handed, or when the node saved no values — its formula then
//     reads ``live`` for its shape alone.
//
// Raises
// ------
// LucidError
//     ``values`` is not the whole input in one dtype, so there is nothing
//     to rebuild it from.
std::shared_ptr<TensorImpl> forward_time_input(const std::shared_ptr<TensorImpl>& live,
                                               const Storage& values,
                                               const Shape& shape,
                                               const Edge& edge,
                                               std::string_view op_name,
                                               std::size_t input_idx);

// Read the process-wide "allow mutation on saved tensors" flag.
//
// Returns
// -------
// bool
//     Current state of the override.  When ``true``,
//     :func:`check_version_match` becomes a no-op.
//
// See Also
// --------
// :func:`set_mutation_on_saved_allowed`
// :func:`lucid.autograd.graph.allow_mutation_on_saved_tensors`
bool is_mutation_on_saved_allowed();

// Toggle the process-wide "allow mutation on saved tensors" flag.
//
// Used by ``lucid.autograd.graph.allow_mutation_on_saved_tensors`` to let
// users opt into the unsafe contract that they will not mutate a saved
// tensor in a way that would corrupt the gradient.
//
// Parameters
// ----------
// v : bool
//     New state of the override.  ``true`` disables version-counter
//     checks; ``false`` re-enables them.
//
// Notes
// -----
// The flag is global to the process — there is no per-thread or
// per-graph scoping.  The Python context manager is responsible for
// restoring the previous value on exit.
void set_mutation_on_saved_allowed(bool v);

class Node;

// Read the process-wide autograd anomaly-detection flag.
//
// Returns
// -------
// bool
//     ``true`` when backward node outputs should be checked for ``NaN``
//     values.
//
// See Also
// --------
// :func:`set_anomaly_enabled`
// :class:`lucid.autograd.detect_anomaly`
bool is_anomaly_enabled();

// Toggle the process-wide autograd anomaly-detection flag.
//
// Parameters
// ----------
// v : bool
//     ``true`` enables ``NaN`` gradient checks on backward node outputs;
//     ``false`` disables them.
void set_anomaly_enabled(bool v);

// Return whether ``s`` contains at least one ``NaN`` element (real or
// complex lane).  Returns ``false`` for integer / boolean storages or
// empty buffers.
bool storage_has_nan(const Storage& s);

// ``node`` as the anomaly report and the profiler name it: its class when
// that reads as a backward (``"PermuteBackward"``), else :meth:`node_name`.
std::string backward_node_name(const Node& node);

// When :func:`is_anomaly_enabled` is ``true``, inspect every storage in
// ``grads`` and raise a :class:`LucidError` naming ``node`` and the
// offending output index on the first ``NaN`` encountered.
void check_anomaly_grads(const Node& node, const std::vector<Storage>& grads);

// Graph-mode counterpart of :func:`check_anomaly_grads` for
// ``backward(create_graph=True)`` and ``grad(..., create_graph=True)``.
void check_anomaly_grads_for_graph(const Node& node,
                                   const std::vector<std::shared_ptr<TensorImpl>>& grads);

// Broadcast ``grad`` back to ``input_shape`` after a forward reduction.
//
// The backward pass for any reduction op (``sum``, ``mean``, ``var``,
// ...) receives a gradient at the reduced shape and must re-expand it
// across the squeezed axes so that the result matches the original input
// shape.  This helper performs that expansion, taking the same
// ``axes`` / ``keepdims`` parameters as the forward call to know which
// dimensions to broadcast back along.
//
// Parameters
// ----------
// grad : const Storage&
//     Incoming gradient at the reduced shape.
// grad_shape : const Shape&
//     Shape of ``grad``; must equal
//     ``reduce_output_shape(input_shape, axes, keepdims)``.
// input_shape : const Shape&
//     Shape to broadcast back to (the original forward input shape).
// axes : const std::vector<int>&
//     Axis list used in the forward reduction.
// keepdims : bool
//     Same value as passed to the forward reduction.
// dt : Dtype
//     Working dtype.
// device : Device
//     Target device.
//
// Returns
// -------
// Storage
//     New storage of shape ``input_shape`` carrying the expanded gradient.
//
// Raises
// ------
// ShapeMismatch
//     ``grad_shape`` does not equal the expected reduced shape.
//
// See Also
// --------
// :func:`reduce_output_shape` — the inverse direction.
Storage broadcast_back_for_reduce(const Storage& grad,
                                  const Shape& grad_shape,
                                  const Shape& input_shape,
                                  const std::vector<int>& axes,
                                  bool keepdims,
                                  Dtype dt,
                                  Device device);

}  // namespace lucid
