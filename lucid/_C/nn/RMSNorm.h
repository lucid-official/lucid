// lucid/_C/nn/RMSNorm.h
//
// Autograd-aware Root Mean Square Layer Normalization (RMSNorm).
//
// Normalises the input by its root-mean-square value over the trailing
// $D$ dimensions (where $D$ is the rank of ``gamma``) and applies a
// per-element scale: $y = (x / \text{rms}(x))\,\gamma$.  Unlike
// LayerNorm there is *no* mean-subtraction step and *no* bias term —
// the layer is rescale-invariant by design and the parameter count is
// half of LayerNorm's.  This is the normalisation layer used by modern
// LLM architectures (LLaMA, T5).
//
// The kernel is registered with :member:`AmpPolicy::ForceFP32`: the
// RMS reduction over a hidden dimension of thousands accumulates
// enough terms that half-precision is unsafe, so the math runs in
// FP32 even under ``AutocastGuard(F16)``.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for RMS Normalization.
//
// Computes the per-slice reciprocal RMS
// $\text{rstd} = 1 / \sqrt{\tfrac{1}{N}\sum_i x_i^2 + \epsilon}$ over
// the trailing $D$ dimensions of ``x``, scales the input by it, and
// applies the per-element gain $\gamma$.  Only ``saved_rstd_`` is
// cached for backward — no mean is needed because RMSNorm does not
// centre the input.
//
// Inherits ``FuncOp<RMSNormBackward, 2>`` with two saved-input slots —
// ``x`` and ``gamma`` (RMSNorm has no bias).  ``outer_ * N_`` always
// equals ``numel(x)``.
//
// Math
// ----
// For each leading-axis slice of ``x`` (flattened length $N$):
//
// $$
//   \text{rms}(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2 + \epsilon},
//   \qquad
//   y_i = \frac{x_i}{\text{rms}(x)} \cdot \gamma_i
// $$
//
// Equivalently, with the saved $\text{rstd} = 1/\text{rms}(x)$:
//
// $$
//   y_i = x_i \cdot \text{rstd} \cdot \gamma_i
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (``name="rms_norm"``, ``version=1``,
//     ``amp_policy=ForceFP32``, ``produces_grad=true``).
// saved_rstd_ : Storage
//     Per-slice reciprocal RMS, shape ``(outer_,)``.
// outer_ : std::size_t
//     Product of leading (non-normalised) dimensions of ``x``.
// N_ : std::size_t
//     Product of normalised (trailing) dimensions of ``x``
//     (== ``numel(gamma)``).
//
// Notes
// -----
// Thread safety: instances are created during forward and consumed
// during backward — no concurrent access expected.
//
// References
// ----------
// Zhang & Sennrich, "Root Mean Square Layer Normalization"
// (NeurIPS 2019, arXiv:1910.07467).
class LUCID_API RMSNormBackward : public FuncOp<RMSNormBackward, 2> {
public:
    static constexpr bool kKeepsOwnValues = true;
    // dx reads x and gamma; dgamma reads x.
    static constexpr GradReads<2> kGradReads = {{{true, true}, {true, false}}};
    // Registered op schema (see class docstring).
    static const OpSchema schema_v1;
    // Per-slice reciprocal RMS, shape ``(outer_,)``.
    Storage saved_rstd_;
    // Product of leading (non-normalised) dims of ``x``.
    std::size_t outer_ = 0;
    // Product of normalised (trailing) dims of ``x`` (== numel(gamma)).
    std::size_t N_ = 0;
    // The forward's epsilon: the graph-mode backward recomputes rstd from x.
    double eps_ = 0.0;

    void release_own_saved() { clear_saved(saved_rstd_); }

    // Run the forward pass.
    //
    // Reduces ``x`` over its last $D$ dimensions (where $D$ is the rank
    // of ``gamma``), divides by the RMS, applies the gain, and (when
    // grad mode is on) registers ``this`` as the ``grad_fn`` of the
    // returned output.  Caches ``saved_rstd_`` for backward.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input tensor of any shape ``(*, normalized_shape)``.
    //     Rank must be ``>= rank(gamma)``.
    // gamma : TensorImplPtr
    //     Per-element scale; its shape must match the trailing dims of
    //     ``x``.
    // eps : double
    //     Small constant added inside the square root for numerical
    //     stability.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Normalised output, same shape as ``x``.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If ``gamma.ndim > x.ndim`` or if the trailing dims of ``x``
    //     do not match ``gamma``.
    static TensorImplPtr forward(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps);

    // Compute gradients for the two saved inputs.
    //
    // Returns ``{dx, d_gamma}`` in the saved-input order.  Closed-form
    // (per leading slice of length $N$, using the cached
    // $\text{rstd} = 1/\text{rms}(x)$):
    //
    // $$
    //   d\gamma_i = \sum_{\text{slice}}
    //              \text{grad\_out}_i\,x_i\,\text{rstd}
    // $$
    //
    // and the input gradient is the standard RMSNorm backward formula
    // (the centred grad is replaced by the *un-centred* grad since
    // RMSNorm has no mean subtraction).
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming gradient, same shape as the forward output.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Two-element vector ``{dx, d_gamma}``.
    std::vector<Storage> apply(Storage grad_out) override;

    // The same gradients, recorded, with rstd recomputed from x so it keeps
    // its dependence on the input — the saved one is data only.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Public free-function entry point for RMS Normalization.
//
// Thin wrapper that delegates to :func:`RMSNormBackward::forward`.
// This is the symbol the pybind11 binding layer forwards to from
// :func:`lucid.nn.functional.rms_norm`.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input tensor of any shape with rank ``>= rank(gamma)``.
// gamma : TensorImplPtr
//     Per-element scale; must match the trailing dims of ``x``.
// eps : double
//     Numerical-stability constant (typically ``1e-8``).
//
// Returns
// -------
// TensorImplPtr
//     Normalised output, same shape as ``x``.
//
// See Also
// --------
// RMSNormBackward : Autograd node implementing the forward + backward.
// layer_norm_op   : Variant that also subtracts the mean and adds a bias.
LUCID_API TensorImplPtr rms_norm_op(const TensorImplPtr& x, const TensorImplPtr& gamma, double eps);

}  // namespace lucid
