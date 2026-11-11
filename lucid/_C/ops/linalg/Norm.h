// lucid/_C/ops/linalg/Norm.h
//
// Tensor p-norm reduction with optional axis selection and keepdims.
//
// Currently supports the Frobenius / Euclidean ($p = 2$) and
// Manhattan ($p = 1$) norms with full autograd, and may dispatch
// other ``ord`` values to the backend in the forward pass (with no
// backward).  Reduces the specified axes of a float tensor and either
// removes them (``keepdims=false``) or collapses them to size 1
// (``keepdims=true``).
//
// Forward dispatch goes to ``IBackend::linalg_norm``, which uses
// Apple Accelerate vDSP (CPU) or MLX reduction ops (GPU).
//
// Backward formulas
// -----------------
// For $N = \|A\|_p$ with upstream gradient $\partial L/\partial N$:
// $$
//   p = 2: \quad
//     \frac{\partial L}{\partial A}
//       = \frac{A}{\max(N,\,\varepsilon)} \odot \mathrm{expand}\!\Bigl(\frac{\partial L}{\partial N}\Bigr),
// $$
// $$
//   p = 1: \quad
//     \frac{\partial L}{\partial A}
//       = \mathrm{sign}(A) \odot \mathrm{expand}\!\Bigl(\frac{\partial L}{\partial N}\Bigr).
// $$
// The $\varepsilon = 10^{-12}$ clip in the $p=2$ case avoids
// division by zero when $A \equiv 0$ (subgradient = 0).
// $\mathrm{expand}(\cdot)$ broadcasts the reduced gradient back to
// the full input shape, re-inserting any axes that were collapsed
// without ``keepdims``.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for the tensor p-norm.
//
// Reproduces ``expand_back`` at backward time using the saved
// ``axis_`` / ``keepdims_`` hyperparameters; saves both the input
// $A$ (for the L2 numerator) and the reduced norm $N$ (to avoid
// recomputation).
//
// Math
// ----
// $$
//   p = 2: \quad
//     \frac{\partial L}{\partial A}
//       = \frac{A}{\mathrm{clip}(N, 10^{-12}, \infty)}
//         \odot \mathrm{expand}(\partial L/\partial N).
// $$
// $$
//   p = 1: \quad
//     \frac{\partial L}{\partial A}
//       = \mathrm{sign}(A) \odot \mathrm{expand}(\partial L/\partial N).
// $$
//
// Shape
// -----
// - ``A``: arbitrary float shape.
// - ``N``: ``A``'s shape with ``axis_`` removed (``keepdims=false``)
//   or set to 1 (``keepdims=true``); scalar if ``axis_`` is empty
//   and ``keepdims=false``.
// - Output gradient: same shape as ``A``.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"norm"``, one input, ``AmpPolicy::KeepInput`` so the ``A/N``
//     division does not lose precision via AMP downcasting.
// ord_ : double
//     Norm order used in the forward pass (1.0 or 2.0 supported in
//     backward; other values raise ``not_implemented`` at backward
//     time).
// axis_ : std::vector<int>
//     Axes that were reduced.  Empty means a full reduction over all
//     elements.  Indices may be negative; they are normalised when
//     re-inserting axes during ``expand_back``.
// keepdims_ : bool
//     Whether reduced axes were retained as size-1 stubs in the
//     output shape.
// saved_inputs_[0] : Storage
//     Storage of the forward input $A$ (needed for ord=2 numerator
//     and ord=1 sign).
// saved_output_ : Storage
//     Storage of the reduced norm $N$ (used as the ord=2 denominator).
//
// Raises
// ------
// NotImplementedError
//     If ``ord_`` is neither 1.0 nor 2.0 when ``apply`` is invoked.
//
// References
// ----------
// Petersen & Pedersen, *The Matrix Cookbook* (2012), §2.5.
// Horn & Johnson, *Matrix Analysis* (2nd ed.), §5.6 (matrix norms).
class LUCID_API NormBackward : public FuncOp<NormBackward, 1> {
public:
    static const OpSchema schema_v1;

    // Norm order used in the forward pass (typically 1.0 or 2.0).
    double ord_ = 2.0;

    // Reduced axes; empty means a full reduction over all elements.
    std::vector<int> axis_;

    // Whether keepdims was true in the forward pass; affects how the
    // reduced gradient is broadcast back to the input shape.
    bool keepdims_ = false;

    // Compute the gradient ``{∂L/∂A}`` from the upstream gradient.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient $\partial L/\partial N$ with shape
    //     matching ``out_shape_``.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Single-entry vector ``{∂L/∂A}`` aligned with the one
    //     differentiable input slot.
    //
    // Raises
    // ------
    // NotImplementedError
    //     If ``ord_`` is not 1.0 or 2.0.
    std::vector<Storage> apply(Storage grad_out) override;

    // The same gradient from tracked ops, for ``create_graph=True``.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

private:
    // ``dA`` from the input, the norm and its gradient — eager mode passes
    // detached tensors, graph mode tracked ones.
    TensorImplPtr
    grad_from(const TensorImplPtr& A, const TensorImplPtr& N, const TensorImplPtr& dN) const;
};

// Autograd node of :func:`p_norm_op`.
//
// For $N = \|A\|_p$ with a finite $p > 1$ the gradient is
// $$
//   \frac{\partial L}{\partial A}
//     = \mathrm{sign}(A)\,\Bigl(\frac{|A|}{N}\Bigr)^{p-1}
//       \odot \mathrm{expand}(\partial L/\partial N),
// $$
// zero where $N = 0$ (the subgradient the reference picks).  The ratio
// $|A|/N$ is in $[0, 1]$, so the slope is as range-safe as the forward: a
// slice at 1e-20 gets the same gradient as one at 1.
class LUCID_API PNormBackward : public FuncOp<PNormBackward, 1> {
public:
    static const OpSchema schema_v1;

    double p_ = 2.0;
    std::vector<int> axis_;
    bool keepdims_ = false;

    std::vector<Storage> apply(Storage grad_out) override;
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

private:
    TensorImplPtr
    grad_from(const TensorImplPtr& A, const TensorImplPtr& N, const TensorImplPtr& dN) const;
};

// The entrywise p-norm $(\sum |a_i|^p)^{1/p}$ over ``axis`` for a finite
// $p > 1$, on the GPU stream, accumulated relative to each slice's largest
// magnitude (``backend/gpu/PNorm.h``) so a slice at 1e-20 or 1e25 does not
// flush to 0 or overflow to inf the way ``sum(a * a)`` does.
//
// ``axis`` empty reduces every axis, as in :func:`norm_op`.  A half input
// reduces in float32 and answers in its own dtype.  The CPU stream and an
// active compile trace refuse it: the CPU's accumulation does not flush, and
// the Python reduction that owns the choice (``_p_norm``) never sends either
// here.
LUCID_API TensorImplPtr p_norm_op(const TensorImplPtr& a,
                                  double p,
                                  std::vector<int> axis,
                                  bool keepdims);

// Compute the p-norm of a float tensor over the specified axes.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor (any float dtype, any shape).
// ord : double
//     Norm order.  ``2.0`` selects the Frobenius / Euclidean norm
//     $\|A\|_F = \sqrt{\sum a_{ij}^2}$; ``1.0`` selects the entrywise
//     L1 norm $\|A\|_1 = \sum |a_{ij}|$.  Other values may be
//     forwarded to the backend in the forward pass but have no
//     registered backward.
// axis : std::vector<int>
//     Axes to reduce.  Empty means a global reduction over every
//     element; negative indices are accepted and normalised.
// keepdims : bool
//     If ``true`` the reduced axes are retained as size-1 stubs in
//     the output; otherwise they are removed.
//
// Returns
// -------
// TensorImplPtr
//     Reduced norm tensor.
//
// Math
// ----
// $$
//   \|A\|_p = \Bigl(\sum_i |a_i|^p\Bigr)^{1/p}.
// $$
// For $p = 2$ on a matrix this is the Frobenius norm; for $p = 1$ on
// a matrix this is the entrywise L1 norm (not the operator 1-norm).
//
// Shape
// -----
// - ``a``: arbitrary float shape.
// - Output:
//     * ``axis`` empty, ``keepdims=false`` $\to$ scalar (rank 0).
//     * ``axis`` empty, ``keepdims=true`` $\to$ all-ones shape of
//       the same rank as ``a``.
//     * Otherwise: rank of ``a`` minus the number of reduced axes
//       (``keepdims=false``) or rank preserved with those axes set
//       to 1 (``keepdims=true``).
//
// Notes
// -----
// Operator norms (spectral, nuclear, induced 1-/inf-norm) are not yet
// dispatched here — they would require SVD and a separate backward.
// This op currently covers only the entrywise reductions.
//
// Raises
// ------
// ValueError
//     If ``a`` is non-float.
// NotImplementedError
//     (At backward time.) If ``ord`` is not 1.0 or 2.0 and a
//     gradient is requested.
//
// See Also
// --------
// SVD.h : Underpins operator norms (spectral, nuclear) once exposed.
LUCID_API TensorImplPtr norm_op(const TensorImplPtr& a,
                                double ord,
                                std::vector<int> axis,
                                bool keepdims);

}  // namespace lucid
