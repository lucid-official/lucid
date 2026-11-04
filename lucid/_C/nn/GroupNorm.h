// lucid/_C/nn/GroupNorm.h
//
// Autograd-aware Group Normalization for spatial inputs of rank >= 2.
//
// Splits the $C$ channels into ``G`` groups of $C/G$ channels each and
// normalises each ``(batch, group)`` slice over its channels-in-group
// and spatial axes jointly, then applies the per-channel affine
// $y = \gamma\,\hat{x} + \beta$.  Group Norm is independent of the
// batch size, making it a drop-in replacement for Batch Norm at small
// batches (detection, segmentation, generative models).
//
// Limit cases:
//
//   * ``G == 1``  → equivalent to LayerNorm over channels+spatial.
//   * ``G == C``  → equivalent to InstanceNorm (one stat pair per
//                    channel).
//
// The kernel is registered with :member:`AmpPolicy::ForceFP32` for the
// same numerical-stability reason as LayerNorm / BatchNorm.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for Group Normalization.
//
// Computes per-``(batch, group)`` mean and reciprocal standard
// deviation over the channels-in-group ($C/G$) plus all spatial axes,
// then normalises and applies a *per-channel* (not per-group) affine
// $y = \gamma_c\,\hat{x}_{b,c,\mathbf{s}} + \beta_c$.  The saved
// statistics are required by the backward closed-form;
// :member:`spatial_dims_` retains the per-axis spatial sizes so the
// backward can recompute the reduction count.
//
// Inherits ``FuncOp<GroupNormBackward, 3>`` with three saved-input
// slots — ``x``, ``gamma``, and ``beta``.
//
// Math
// ----
// Let $g(c)$ denote the group containing channel $c$ (so
// $g(c) = \lfloor c\,G\,/\,C\rfloor$) and write
// $S = \prod_i \text{spatial\_dims}[i]$.  For each ``(b, g)`` pair:
//
// $$
//   \mu_{b,g} = \frac{1}{(C/G)\,S}
//      \sum_{c\in g}\sum_{\mathbf{s}} x_{b,c,\mathbf{s}},
//   \qquad
//   \sigma_{b,g}^2 = \frac{1}{(C/G)\,S}
//      \sum_{c\in g}\sum_{\mathbf{s}} (x_{b,c,\mathbf{s}} - \mu_{b,g})^2
// $$
//
// $$
//   \hat{x}_{b,c,\mathbf{s}}
//     = (x_{b,c,\mathbf{s}} - \mu_{b,g(c)})
//       \cdot \frac{1}{\sqrt{\sigma_{b,g(c)}^2 + \epsilon}},
//   \qquad
//   y_{b,c,\mathbf{s}} = \gamma_c\,\hat{x}_{b,c,\mathbf{s}} + \beta_c
// $$
//
// Mean and variance use the biased estimator.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (``name="group_norm"``, ``version=1``,
//     ``amp_policy=ForceFP32``, ``produces_grad=true``).
// saved_mean_ : Storage
//     Per-``(batch, group)`` mean $\mu_{b,g}$, shape ``(B*G,)``.
// saved_rstd_ : Storage
//     Per-``(batch, group)`` reciprocal standard deviation
//     $1 / \sqrt{\sigma_{b,g}^2 + \epsilon}$, shape ``(B*G,)``.
// B_, C_, G_ : int
//     Batch, channel, and group counts captured from ``x.shape()``.
// spatial_dims_ : std::vector<int>
//     Per-axis spatial sizes ``[S_0, S_1, ...]`` retained so the
//     backward can recompute the spatial reduction count.
//
// Notes
// -----
// Thread safety: instances are created once during forward and
// consumed once during backward — no concurrent access expected.
//
// References
// ----------
// Wu & He, "Group Normalization" (ECCV 2018, arXiv:1803.08494).
class LUCID_API GroupNormBackward : public FuncOp<GroupNormBackward, 3> {
public:
    static constexpr bool kKeepsOwnValues = true;
    // dx reads x and gamma; dgamma reads x; dbeta reads only grad_out.
    static constexpr GradReads<3> kGradReads = {{{true, true, false},
                                                 {true, false, false},
                                                 {false, false, false}}};
    // Registered op schema (see class docstring).
    static const OpSchema schema_v1;
    // Per-``(batch, group)`` mean $\mu_{b,g}$, shape ``(B*G,)``.
    Storage saved_mean_;
    // Per-``(batch, group)`` reciprocal std, shape ``(B*G,)``.
    Storage saved_rstd_;
    // Batch ($B$), channel ($C$), and group ($G$) counts.
    int B_ = 0, C_ = 0, G_ = 0;
    // Per-axis spatial sizes ``[S_0, S_1, ...]``.
    std::vector<int> spatial_dims_;
    // Needed to rebuild rstd from x in graph mode; the saved rstd is a
    // plain Storage and would drop d(rstd)/dx from the second derivative.
    double eps_ = 1e-5;

    void release_own_saved() { clear_saved(saved_mean_, saved_rstd_); }

    // Run the forward pass.
    //
    // Computes per-``(batch, group)`` statistics, normalises ``x``,
    // applies the per-channel affine, caches ``saved_mean_`` and
    // ``saved_rstd_``, and (when grad mode is on) registers ``this``
    // as the ``grad_fn`` of the returned output.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input tensor of shape ``(B, C, S_0, ...)``.  Rank must be
    //     ``>= 2``.
    // gamma : TensorImplPtr
    //     Per-channel scale of shape ``(C,)``.
    // beta : TensorImplPtr
    //     Per-channel shift of shape ``(C,)``.
    // num_groups : int
    //     Number of groups ``G``; must divide ``C`` evenly.
    // eps : double
    //     Numerical-stability constant added inside the square root.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Normalised output, same shape as ``x``.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If ``C`` is not divisible by ``num_groups`` or if ``gamma``
    //     and ``beta`` do not have shape ``(C,)``.
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 int num_groups,
                                 double eps);

    // Compute gradients for the three saved inputs.
    //
    // Returns ``{dx, d_gamma, d_beta}`` in the saved-input order using
    // the cached statistics.  Closed-form summary (per channel $c$):
    //
    // $$
    //   d\gamma_c = \sum_{b,\mathbf{s}}
    //              \hat{x}_{b,c,\mathbf{s}}\,
    //              \text{grad\_out}_{b,c,\mathbf{s}},
    //   \qquad
    //   d\beta_c = \sum_{b,\mathbf{s}} \text{grad\_out}_{b,c,\mathbf{s}}
    // $$
    //
    // and the input gradient follows the standard GN backward formula
    // restricted to each ``(b, g)`` group.
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

    // Graph-mode backward — the normalisation is rebuilt from the saved
    // inputs so all three gradients stay differentiable.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Public free-function entry point for Group Normalization.
//
// Thin wrapper that delegates to :func:`GroupNormBackward::forward`.
// This is the symbol the pybind11 binding layer forwards to from
// :func:`lucid.nn.functional.group_norm`.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input tensor of shape ``(B, C, S_0, ...)`` with rank ``>= 2``.
// gamma : TensorImplPtr
//     Per-channel scale of shape ``(C,)``.
// beta : TensorImplPtr
//     Per-channel shift of shape ``(C,)``.
// num_groups : int
//     Number of groups; must divide ``C`` evenly.
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
// GroupNormBackward : Autograd node implementing the forward + backward.
// layer_norm_op     : Equivalent when ``num_groups == 1``.
LUCID_API TensorImplPtr group_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      int num_groups,
                                      double eps);

}  // namespace lucid
