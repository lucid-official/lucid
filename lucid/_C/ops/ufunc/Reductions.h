// lucid/_C/ops/ufunc/Reductions.h
//
// Backward nodes and public entry points for the canonical multi-axis
// reductions: ``sum``, ``mean``, ``prod``, ``max``, ``min``, plus the
// ``std`` composite that decomposes into ``sqrt(var(...))``.
//
// All five primary nodes inherit from :class:`ReduceOp` (an alias for
// :class:`ReduceKernel`), which centralises axis normalisation, output
// allocation, dispatch, and the saving of ``reduce_axes_`` /
// ``keepdims_`` / ``full_input_shape_`` on the backward node.  Each
// ``grad_formula`` then performs the broadcast-back of the upstream
// gradient and applies the reduction-specific scaling rule.
//
// Backward strategies in one line
// -------------------------------
// - **sum**  — broadcast-back, no scaling, no saved tensors.
// - **mean** — broadcast-back then divide by $N$ (count of reduced
//   elements).
// - **prod** — saves both input $x$ and output $y$; gradient is
//   $\partial L/\partial x_i = \partial L/\partial y \cdot (y / x_i)$,
//   the "product of all other elements".
// - **max** / **min** — saves only the output; builds an equality mask
//   (argmax/argmin indicator) and routes the gradient through it.  Ties
//   share gradient equally, matching the reference framework.
//
// Empty-input behaviour follows the reference: ``sum -> 0``,
// ``prod -> 1``, ``max`` / ``min`` raise / propagate NaN.

#pragma once

#include <utility>
#include <vector>

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_ReduceOp.h"

namespace lucid {

// Autograd node for the multi-axis reduction sum
// $y = \sum_{i \in \text{axes}} x_i$.
//
// The gradient broadcasts unchanged from the reduced shape back to the
// input shape because $\partial y / \partial x_i = 1$ for every element
// participating in the sum.  ``kSavesInput = false`` and no output is
// saved — the rule is independent of both the forward input and output.
//
// Math
// ----
// $$
//   y = \sum_{i \in \text{axes}} x_i, \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i} =
//   \mathrm{broadcast}\!\left(\frac{\partial \mathcal{L}}{\partial y}
//   \right).
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"sum"`` with ``AmpPolicy::KeepInput`` — integer and
//     boolean inputs are promoted to ``int64`` before reduction so the
//     sum cannot overflow narrow integer dtypes (parity fix found
//     during the M4 Max training sanity check, where ``bool.sum()``
//     previously dropped to ``any()`` semantics on GPU).
// kSavesInput : bool
//     ``false``.  Backward depends only on shape metadata.
//
// Notes
// -----
// Dispatch: Accelerate strided reductions (CPU) / MLX ``sum`` (GPU).
// For empty reductions the output is $0$ (additive identity).
class LUCID_API SumBackward : public ReduceOp<SumBackward> {
public:
    static constexpr bool kSavesInput = false;
    static const OpSchema schema_v1;
    // Forward — reduces ``a`` along ``axes`` via ``IBackend::reduce_sum``.
    static Storage dispatch(backend::IBackend& be,
                            const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes,
                            bool keepdims,
                            Dtype dt) {
        return be.reduce_sum(a, in_shape, {axes, keepdims}, dt);
    }

    // Backward — broadcasts ``grad_out`` back to the input shape (ones gradient).
    Storage grad_formula(const Storage& grad_out);
    // Graph-mode scaling helper — ``sum`` requires no scaling, so the
    // broadcast-expanded gradient is returned unchanged.
    TensorImplPtr scale_graph_grad(const TensorImplPtr& g) { return g; }
};

// Autograd node for the multi-axis reduction mean
// $y = \frac{1}{N} \sum_{i \in \text{axes}} x_i$.
//
// Backward broadcasts the upstream gradient back to the input shape and
// divides by $N$, the product of the reduced dimension sizes.
//
// Math
// ----
// $$
//   y = \frac{1}{N} \sum_{i \in \text{axes}} x_i, \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i} =
//   \frac{1}{N}\,\mathrm{broadcast}\!\left(
//     \frac{\partial \mathcal{L}}{\partial y}\right).
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"mean"`` with ``AmpPolicy::KeepInput``.
// kSavesInput : bool
//     ``false``.  Backward depends only on $N$ and shape metadata.
//
// Notes
// -----
// Dispatch: Accelerate kernel (CPU) / MLX ``mean`` (GPU).  Empty
// reductions return NaN ($0/0$).
class LUCID_API MeanBackward : public ReduceOp<MeanBackward> {
public:
    static constexpr bool kSavesInput = false;
    static const OpSchema schema_v1;
    // Forward — reduces ``a`` along ``axes`` via ``IBackend::reduce_mean``.
    static Storage dispatch(backend::IBackend& be,
                            const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes,
                            bool keepdims,
                            Dtype dt) {
        return be.reduce_mean(a, in_shape, {axes, keepdims}, dt);
    }

    // Backward — broadcasts $\mathrm{grad\_out} / N$ back to input shape.
    Storage grad_formula(const Storage& grad_out);
    // Graph-mode mean backward: divides the broadcast-expanded gradient
    // by the number of reduced elements so second-order graphs flow
    // through the division correctly.
    TensorImplPtr scale_graph_grad(const TensorImplPtr& g);
};

// Autograd node for the multi-axis reduction product
// $y = \prod_{i \in \text{axes}} x_i$.
//
// The gradient is the "product of all other elements" of $x_i$'s slice.
// It is $y / x_i$ only where $x_i \ne 0$, so it is formed from the saved
// input with the slice's zeros split out rather than divided by: a zero
// gets the product of the rest (0 when the slice has another zero).
//
// Because no single ``IBackend`` overload spans every Accelerate /
// MLX combination, ``prod`` provides explicit ``cpu_kernel`` and
// ``gpu_kernel`` overloads instead of a shared ``dispatch``.
//
// Math
// ----
// $$
//   y = \prod_{i \in \text{axes}} x_i, \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i} =
//   \Bigl(\prod_{j \ne i} x_j\Bigr)\,
//   \mathrm{broadcast}\!\left(\frac{\partial \mathcal{L}}{\partial y}
//   \right).
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"prod"`` with ``AmpPolicy::KeepInput``.
// kSavesInput : bool
//     ``true``.
// kSavesOutput : bool
//     ``false``.
//
// Notes
// -----
// CPU path iterates axes in **descending order** so the lower indices
// remain valid as dimensions are successively collapsed.  GPU path
// delegates to ``mlx::core::prod`` with ``keepdims``.  Empty
// reductions return $1$ (multiplicative identity).
class LUCID_API ProdBackward : public ReduceOp<ProdBackward> {
public:
    static constexpr bool kSavesInput = true;
    static constexpr bool kSavesOutput = false;
    static const OpSchema schema_v1;
    // CPU path: iterates axes in descending order (innermost last) to produce
    // a correct sequential multi-axis product using Accelerate primitives.
    static CpuStorage cpu_kernel(const CpuStorage& a,
                                 const Shape& in_shape,
                                 const std::vector<int>& axes,
                                 bool keepdims,
                                 Dtype dt);
    // GPU path: delegates to mlx::core::prod with keepdims.
    static GpuStorage gpu_kernel(const GpuStorage& a,
                                 const Shape& in_shape,
                                 const std::vector<int>& axes,
                                 bool keepdims,
                                 Dtype dt);
    // Backward — $\partial L/\partial x_i = \mathrm{grad\_out} \cdot
    // \prod_{j \ne i} x_j$, formed from the saved input.
    Storage grad_formula(const Storage& grad_out);

    // Graph-mode scaling.  Without this the base class returns the
    // broadcast gradient unchanged — the *sum* rule — so
    // ``create_graph=True`` silently produced the incoming seed at
    // every position while eager backward was correct.
    TensorImplPtr scale_graph_grad(const TensorImplPtr& g);
};

// Autograd node for the multi-axis reduction maximum
// $y = \max_{i \in \text{axes}} x_i$.
//
// The gradient routes only to positions where $x_i$ equals the
// reduced maximum.  The equality mask is built without a dedicated
// equality kernel by composing two ``ge`` masks
// ($a = b \iff a \ge b \wedge b \ge a$).  Ties (multiple equal maxima)
// receive equal gradient — matching the reference framework's
// behaviour.  The forward output is saved so the mask can be formed
// without rerunning the reduction.
//
// Math
// ----
// $$
//   y = \max_{i \in \text{axes}} x_i, \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i} =
//   \frac{\mathbb{1}[x_i = y]}{\#\{j : x_j = y\}}\cdot
//   \mathrm{broadcast}\!\left(\frac{\partial \mathcal{L}}{\partial y}
//   \right).
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"max"`` with ``AmpPolicy::KeepInput`` — integer
//     dtypes pass through unchanged.
// kSavesOutput : bool
//     ``true``.
//
// Notes
// -----
// Empty reductions propagate NaN / raise (backend-dependent); callers
// should guard against zero-sized reduction axes.
class LUCID_API MaxBackward : public ReduceOp<MaxBackward> {
public:
    static constexpr bool kSavesOutput = true;
    // No identity element: an empty reduced axis has no maximum.
    static constexpr bool kNoIdentity = true;
    static const OpSchema schema_v1;
    // Forward — reduces ``a`` along ``axes`` via ``IBackend::reduce_max``.
    static Storage dispatch(backend::IBackend& be,
                            const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes,
                            bool keepdims,
                            Dtype dt) {
        return be.reduce_max(a, in_shape, {axes, keepdims}, dt);
    }

    // Backward — argmax-indicator gradient: routes ``grad_out`` only to
    // positions where $x_i$ equals the reduced max (ties share equally).
    Storage grad_formula(const Storage& grad_out);

    // Graph-mode scaling.  Without this the base class returns the
    // broadcast gradient unchanged — the *sum* rule — so
    // ``create_graph=True`` silently produced the incoming seed at
    // every position while eager backward was correct.
    TensorImplPtr scale_graph_grad(const TensorImplPtr& g);
};

// Autograd node for the multi-axis reduction minimum
// $y = \min_{i \in \text{axes}} x_i$.
//
// Symmetric to :class:`MaxBackward`: the gradient flows to positions
// where $x_i$ equals the reduced minimum.  Ties share gradient equally
// and the same composed ``ge``/``ge`` equality-mask trick is used.
//
// Math
// ----
// $$
//   y = \min_{i \in \text{axes}} x_i, \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i} =
//   \frac{\mathbb{1}[x_i = y]}{\#\{j : x_j = y\}}\cdot
//   \mathrm{broadcast}\!\left(\frac{\partial \mathcal{L}}{\partial y}
//   \right).
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"min"`` with ``AmpPolicy::KeepInput``.
// kSavesOutput : bool
//     ``true``.
class LUCID_API MinBackward : public ReduceOp<MinBackward> {
public:
    static constexpr bool kSavesOutput = true;
    // No identity element: an empty reduced axis has no minimum.
    static constexpr bool kNoIdentity = true;
    static const OpSchema schema_v1;
    // Forward — reduces ``a`` along ``axes`` via ``IBackend::reduce_min``.
    static Storage dispatch(backend::IBackend& be,
                            const Storage& a,
                            const Shape& in_shape,
                            const std::vector<int>& axes,
                            bool keepdims,
                            Dtype dt) {
        return be.reduce_min(a, in_shape, {axes, keepdims}, dt);
    }

    // Backward — argmin-indicator gradient: routes ``grad_out`` only to
    // positions where $x_i$ equals the reduced min (ties share equally).
    Storage grad_formula(const Storage& grad_out);

    // Graph-mode scaling.  Without this the base class returns the
    // broadcast gradient unchanged — the *sum* rule — so
    // ``create_graph=True`` silently produced the incoming seed at
    // every position while eager backward was correct.
    TensorImplPtr scale_graph_grad(const TensorImplPtr& g);
};

// Reduce ``a`` by addition along ``axes``.
//
// Promotes ``bool`` / ``int8`` / ``int16`` / ``int32`` inputs to
// ``int64`` before dispatch so the sum has overflow headroom matching
// the reference framework.  Routes the forward through
// :class:`SumBackward` which also installs the autograd edge when
// ``a->requires_grad()`` is true.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape and any real dtype.
// axes : std::vector<int>
//     Axes to reduce.  May be empty to reduce all axes.  Negative
//     indices wrap around ``ndim``.
// keepdims : bool
//     If ``true``, the reduced dimensions are kept as size-1 entries;
//     otherwise they are collapsed.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor.  Shape is the input shape with the reduced axes
//     either removed (default) or set to $1$ (``keepdims=true``).
//
// See Also
// --------
// :class:`SumBackward` — backward node.
LUCID_API TensorImplPtr sum_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

// ``a`` as the dtype an integer reduction accumulates in.
//
// ``bool``, ``int8``, ``int16`` and ``int32`` widen to ``int64``, which is
// what the reference's ``sum`` / ``prod`` / ``cumsum`` / ``cumprod`` /
// ``trace`` answer for them; every other dtype comes back as it is.  The one
// rule those ops share, so an op that adds up elements cannot pick a
// narrower result than ``sum`` does.
LUCID_API TensorImplPtr promote_int_for_reduce(const TensorImplPtr& a);

// Reduce ``a`` by averaging along ``axes``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// axes : std::vector<int>
//     Axes to reduce.  Empty means "all axes".  Negative indices wrap.
// keepdims : bool
//     If ``true``, retains reduced dims as size-1.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the broadcast-compatible reduced shape.
//
// See Also
// --------
// :class:`MeanBackward` — backward node.
LUCID_API TensorImplPtr mean_op(const TensorImplPtr& a,
                                const std::vector<int>& axes,
                                bool keepdims);

// Reduce ``a`` by multiplication along ``axes``.
//
// Promotes narrow integer dtypes to ``int64`` (same headroom logic as
// :func:`sum_op`).
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// axes : std::vector<int>
//     Axes to reduce.  Negative indices wrap.
// keepdims : bool
//     If ``true``, retains reduced dims as size-1.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor.  Empty reductions yield $1$.
//
// See Also
// --------
// :class:`ProdBackward` — backward node.
LUCID_API TensorImplPtr prod_op(const TensorImplPtr& a,
                                const std::vector<int>& axes,
                                bool keepdims);

// Reduce ``a`` by maximum along ``axes``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// axes : std::vector<int>
//     Axes to reduce.  Negative indices wrap.
// keepdims : bool
//     If ``true``, retains reduced dims as size-1.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the reduced shape.
//
// See Also
// --------
// :class:`MaxBackward` — backward node.
LUCID_API TensorImplPtr max_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

// Reduce ``a`` by minimum along ``axes``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// axes : std::vector<int>
//     Axes to reduce.  Negative indices wrap.
// keepdims : bool
//     If ``true``, retains reduced dims as size-1.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the reduced shape.
//
// See Also
// --------
// :class:`MinBackward` — backward node.
LUCID_API TensorImplPtr min_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

// Compute the (biased) standard deviation along ``axes``.
//
// Defined as $\mathrm{std}(x) = \sqrt{\mathrm{var}(x, \text{axes},
// \text{keepdims})}$, where the underlying variance uses the **biased**
// estimator $\frac{1}{N}\sum (x - \bar{x})^2$.  No new backward node
// is needed: the gradient flows naturally through ``SqrtBackward`` and
// ``VarBackward`` via the chain rule.
//
// Math
// ----
// $$
//   \mathrm{std}(x) = \sqrt{\frac{1}{N}\sum_{i \in \text{axes}}
//   (x_i - \bar{x})^2}.
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of float dtype.
// axes : std::vector<int>
//     Axes to reduce.  Negative indices wrap.
// keepdims : bool
//     If ``true``, retains reduced dims as size-1.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the reduced shape.
//
// See Also
// --------
// :func:`var_op` — variance the standard deviation is built on.
LUCID_API TensorImplPtr std_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims);

}  // namespace lucid
