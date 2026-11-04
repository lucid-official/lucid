// lucid/_C/nn/LayerNorm.h
//
// Autograd-aware Layer Normalization.
//
// Normalises the input over its trailing $D$ dimensions — where $D$ is
// the rank of ``gamma`` (and ``beta``) — by the per-slice mean and
// variance, then applies the per-element affine
// $y = \gamma \hat{x} + \beta$.  $D$ may be any value from 1 to
// ``rank(x)``: for ``(N, T, C)`` input with ``gamma`` of shape ``(C,)``
// the reduction is per ``(n, t)`` position; for ``gamma`` of shape
// ``(T, C)`` it reduces over the last two axes.
//
// The kernel is registered with :member:`AmpPolicy::ForceFP32`:
// reductions over the trailing axes accumulate enough terms that
// half-precision variance suffers from catastrophic cancellation, so
// the math is always carried out in FP32 even under an outer
// ``AutocastGuard(F16)``.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for Layer Normalization.
//
// Computes
// $\hat{x} = (x - \mu) / \sqrt{\sigma^2 + \epsilon}$ over the last $D$
// dimensions of ``x`` (one ``(\mu, \sigma^2)`` pair per leading-axis
// slice), then applies the element-wise affine
// $y = \gamma\,\hat{x} + \beta$.  The per-slice mean
// :member:`saved_mean_` and reciprocal standard deviation
// :member:`saved_rstd_` are cached for the backward closed-form.
//
// Inherits ``FuncOp<LayerNormBackward, 3>`` with three saved-input
// slots — ``x``, ``gamma``, and ``beta``.  ``outer_ * N_`` always
// equals ``numel(x)``: ``outer_`` is the product of the leading
// (non-normalised) dims and ``N_`` is the product of the trailing
// (normalised) dims.
//
// Math
// ----
// For each leading-axis slice of ``x`` (flattened length $N$):
//
// $$
//   \mu = \frac{1}{N}\sum_i x_i, \qquad
//   \sigma^2 = \frac{1}{N}\sum_i (x_i - \mu)^2
// $$
//
// $$
//   \hat{x}_i = (x_i - \mu) \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}},
//   \qquad y_i = \gamma_i\,\hat{x}_i + \beta_i
// $$
//
// Mean and variance use the biased estimator (``correction=0``), as is
// standard for layer norm.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (``name="layer_norm"``, ``version=1``,
//     ``amp_policy=ForceFP32``, ``produces_grad=true``).
// saved_mean_ : Storage
//     Per-slice mean $\mu$, shape ``(outer_,)``.
// saved_rstd_ : Storage
//     Per-slice reciprocal standard deviation
//     $1 / \sqrt{\sigma^2 + \epsilon}$, shape ``(outer_,)``.
// outer_ : std::size_t
//     Product of all leading (non-normalised) dimensions of ``x``.
// N_ : std::size_t
//     Product of all normalised (trailing) dimensions of ``x``
//     (== ``numel(gamma)``).
//
// Notes
// -----
// Thread safety: instances are created during forward and consumed
// during backward — no concurrent access expected.
//
// References
// ----------
// Ba, Kiros & Hinton, "Layer Normalization" (arXiv:1607.06450, 2016).
class LUCID_API LayerNormBackward : public FuncOp<LayerNormBackward, 3> {
public:
    static constexpr bool kKeepsOwnValues = true;
    // dx reads x and gamma; dgamma reads x; dbeta reads only grad_out.
    static constexpr GradReads<3> kGradReads = {{{true, true, false},
                                                 {true, false, false},
                                                 {false, false, false}}};
    // Registered op schema (see class docstring).
    static const OpSchema schema_v1;
    // Per-slice mean $\mu$, shape ``(outer_,)``.
    Storage saved_mean_;
    // Per-slice reciprocal standard deviation, shape ``(outer_,)``.
    Storage saved_rstd_;
    // Product of leading (non-normalised) dims of ``x``.
    std::size_t outer_ = 0;
    // Product of normalised (trailing) dims of ``x`` (== numel(gamma)).
    std::size_t N_ = 0;
    // Kept so the graph-mode backward can rebuild rstd from x.  The saved
    // rstd is a plain Storage and carries no dependence on the input, so
    // reusing it would cost the second derivative its whole d(rstd)/dx
    // term without anything looking wrong.
    double eps_ = 1e-5;

    void release_own_saved() { clear_saved(saved_mean_, saved_rstd_); }

    // Run the forward pass.
    //
    // Reduces ``x`` over its last $D$ dimensions (where $D$ is the rank
    // of ``gamma``), normalises, applies the affine, and (when grad mode
    // is on) registers ``this`` as the ``grad_fn`` of the returned
    // output.  Caches ``saved_mean_`` and ``saved_rstd_`` for backward.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input tensor of any shape ``(*, normalized_shape)``.
    // gamma : TensorImplPtr
    //     Per-element scale; its shape must match the trailing dims of
    //     ``x``.
    // beta : TensorImplPtr
    //     Per-element shift; must have the same shape as ``gamma``.
    // eps : double
    //     Small constant added to the variance for numerical stability.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Normalised output, same shape as ``x``.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If ``gamma.ndim > x.ndim``, if the trailing dims of ``x`` do
    //     not match ``gamma``, or if ``gamma.shape != beta.shape``.
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);

    // Compute gradients for the three saved inputs.
    //
    // Returns ``{dx, d_gamma, d_beta}`` in the saved-input order.
    // Closed-form (per leading slice of length $N$):
    //
    // $$
    //   d\gamma_i = \sum_{\text{slice}} \hat{x}_i\,\text{grad\_out}_i,
    //   \qquad
    //   d\beta_i  = \sum_{\text{slice}} \text{grad\_out}_i
    // $$
    //
    // and the input gradient follows the standard LN backward expansion
    // (a difference of the centred grad and two reduction terms scaled
    // by $\text{rstd}$).
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming gradient, same shape as the forward output.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Three-element vector ``{dx, d_gamma, d_beta}``.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward — the same three gradients, recomputed from
    // the saved inputs so each stays differentiable.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Public free-function entry point for Layer Normalization.
//
// Thin wrapper that delegates to :func:`LayerNormBackward::forward`.
// This is the symbol the pybind11 binding layer forwards to from
// :func:`lucid.nn.functional.layer_norm`.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input tensor of any shape.
// gamma : TensorImplPtr
//     Per-element scale; must match the trailing dims of ``x``.
// beta : TensorImplPtr
//     Per-element shift; same shape as ``gamma``.
// eps : double
//     Numerical-stability constant.
//
// Returns
// -------
// TensorImplPtr
//     Normalised output, same shape as ``x``.
//
// See Also
// --------
// LayerNormBackward : Autograd node implementing the forward + backward.
// rms_norm_op       : Variant that omits mean subtraction.
LUCID_API TensorImplPtr layer_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      double eps);

}  // namespace lucid
