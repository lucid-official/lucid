// lucid/_C/nn/NormExt.h
//
// Additional normalization variants not covered by the core
// ``BatchNorm.h`` / ``LayerNorm.h`` / ``GroupNorm.h`` / ``RMSNorm.h``:
//
//   * :class:`BatchNormEvalBackward` — inference-mode Batch
//     Normalization that uses externally supplied
//     ``running_mean`` / ``running_var`` tensors instead of computing
//     batch statistics.  This is the path the Python ``BatchNormNd``
//     module takes after ``model.eval()``.
//   * :class:`LpNormalizeBackward` — Lp-norm normalisation of the
//     input along a single chosen axis ($y = x / \max(\|x\|_p, \epsilon)$).
//   * :class:`GlobalResponseNormBackward` — ConvNeXt-V2 Global Response
//     Normalisation: each channel is rescaled by its L2 spatial norm
//     normalised by the mean L2 norm across channels.
//
// All three nodes are registered with :member:`AmpPolicy::ForceFP32`
// — reductions over batch + spatial axes need FP32 accumulation to
// avoid catastrophic cancellation under autocast.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for inference-mode Batch Normalization.
//
// Uses caller-supplied running mean and variance — *not* computed from
// the current batch — to normalise the input.  This is the path the
// Python ``BatchNormNd`` module dispatches to once
// ``model.eval()`` has been called (and ``track_running_stats=True``).
//
// Computes
// $\hat{x}_{b,c,\mathbf{s}} = (x_{b,c,\mathbf{s}} - \mu_c)\,/\,
//  \sqrt{\sigma_c^2 + \epsilon}$ per channel using the supplied
// ``mean`` and ``var`` tensors, then applies the per-channel affine
// $y = \gamma\,\hat{x} + \beta$.  The reciprocal standard deviation
// :member:`rstd_` is cached for the backward closed-form.
//
// Inherits ``FuncOp<BatchNormEvalBackward, 5>`` with five saved-input
// slots — ``x``, ``mean``, ``var``, ``gamma``, ``beta`` — because
// during training-time fine-tuning users may want gradients with
// respect to all five operands.  In typical inference usage only ``x``
// requires grad, in which case the four remaining gradient slots are
// trivially zero.
//
// Math
// ----
// $$
//   \hat{x}_{b,c,\mathbf{s}} = \frac{x_{b,c,\mathbf{s}} - \mu_c}
//                                  {\sqrt{\sigma_c^2 + \epsilon}},
//   \qquad
//   y = \gamma_c\,\hat{x} + \beta_c
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (``name="batch_norm_eval"``, ``version=1``,
//     ``amp_policy=ForceFP32``, ``produces_grad=true``).
// eps_ : double
//     Forward's ``eps``; retained for the backward.
// rstd_ : Storage
//     Pre-computed reciprocal standard deviation
//     $1 / \sqrt{\sigma_c^2 + \epsilon}$, shape ``(C,)``.
//
// See Also
// --------
// BatchNorm2dBackward : Training-mode counterpart that computes
//     statistics from the current batch.
class LUCID_API BatchNormEvalBackward : public FuncOp<BatchNormEvalBackward, 5> {
public:
    static constexpr bool kKeepsOwnValues = true;
    // Inputs: 0:x, 1:mean, 2:var, 3:gamma, 4:beta.
    // dx and dmean read gamma; dvar reads x, mean, gamma; dgamma reads x, mean;
    // dbeta reads only grad_out.
    static constexpr GradReads<5> kGradReads = {{{false, false, false, true, false},
                                                 {false, false, false, true, false},
                                                 {true, true, false, true, false},
                                                 {true, true, false, false, false},
                                                 {false, false, false, false, false}}};
    // Registered op schema (see class docstring).
    static const OpSchema schema_v1;
    // Forward eps, retained for the backward derivation.
    double eps_ = 1e-5;
    // Pre-computed per-channel reciprocal std, shape ``(C,)``.
    Storage rstd_;

    void release_own_saved() { clear_saved(rstd_); }

    // Run the inference-mode forward pass.
    //
    // Normalises ``x`` channel-wise using the *supplied* ``mean`` and
    // ``var`` (typically the buffers ``running_mean`` / ``running_var``
    // accumulated during training), then applies the affine.  Caches
    // ``rstd_`` for backward and (when grad mode is on) registers
    // ``this`` as the ``grad_fn`` of the returned output.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input tensor of shape ``(B, C, ...)``.
    // mean : TensorImplPtr
    //     Running per-channel mean of shape ``(C,)``.
    // var : TensorImplPtr
    //     Running per-channel variance of shape ``(C,)``.
    // gamma : TensorImplPtr
    //     Per-channel scale of shape ``(C,)``.
    // beta : TensorImplPtr
    //     Per-channel shift of shape ``(C,)``.
    // eps : double
    //     Numerical-stability constant.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Normalised output, same shape as ``x``.
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& mean,
                                 const TensorImplPtr& var,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);

    // Compute gradients for the five saved inputs.
    //
    // Returns ``{dx, d_mean, d_var, d_gamma, d_beta}`` in the
    // saved-input order.  In standard eval usage the gradients with
    // respect to ``mean``, ``var``, ``gamma``, and ``beta`` are zero
    // (running stats and the affine are treated as frozen constants),
    // but the kernel computes them anyway so that fine-tuning use
    // cases work.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming gradient, same shape as the forward output.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Five-element vector ``{dx, d_mean, d_var, d_gamma, d_beta}``.
    std::vector<Storage> apply(Storage grad_out) override;

    // The same gradients from tracked ops, for ``create_graph=True``.
    // ``var`` reaches the node only as ``rstd_`` — it is not saved — so a
    // running variance that itself requires grad is refused here, as the
    // reference refuses differentiating running statistics at all.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Public free-function entry point for inference-mode Batch Normalization.
//
// Thin wrapper that delegates to :func:`BatchNormEvalBackward::forward`.
// This is the symbol the pybind11 binding layer forwards to from
// :func:`lucid.nn.functional.batch_norm` when ``training=False``.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input tensor of shape ``(B, C, ...)``.
// mean : TensorImplPtr
//     Running per-channel mean of shape ``(C,)``.
// var : TensorImplPtr
//     Running per-channel variance of shape ``(C,)``.
// gamma : TensorImplPtr
//     Per-channel scale of shape ``(C,)``.
// beta : TensorImplPtr
//     Per-channel shift of shape ``(C,)``.
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
// BatchNormEvalBackward : Autograd node implementing the forward + backward.
// batch_norm_op         : Training-mode counterpart that uses batch stats.
LUCID_API TensorImplPtr batch_norm_eval_op(const TensorImplPtr& x,
                                           const TensorImplPtr& mean,
                                           const TensorImplPtr& var,
                                           const TensorImplPtr& gamma,
                                           const TensorImplPtr& beta,
                                           double eps);

// Autograd node for Lp normalisation along a single axis.
//
// Computes $y = x / \max(\|x\|_p, \epsilon)$ along ``axis``, dividing
// the input by its Lp-norm per slice while clamping the denominator
// from below by ``eps`` to avoid division by zero.  ``axis`` is
// resolved to a non-negative index inside :func:`forward` so the
// saved value is always normalised.
//
// Inherits ``FuncOp<LpNormalizeBackward, 1>`` with a single
// saved-input slot — ``x``.  The per-slice norm
// :member:`saved_norm_` is cached for the backward rule.
//
// Math
// ----
// Let $S$ denote a slice along the chosen axis (the elements summed
// over):
//
// $$
//   \|x\|_p = \Bigl(\sum_{i\in S} |x_i|^p\Bigr)^{1/p},
//   \qquad
//   y_i = \frac{x_i}{\max(\|x\|_p,\,\epsilon)}
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (``name="lp_normalize"``, ``version=1``,
//     ``amp_policy=ForceFP32``, ``produces_grad=true``).
// ord_ : double
//     Norm order $p$.  Defaults to ``2.0`` (Euclidean L2 norm).
// axis_ : int
//     Resolved non-negative axis along which the norm is computed.
//     Negative axes passed to :func:`forward` are converted before
//     storage.
// eps_ : double
//     Minimum allowed denominator, clamping the norm from below to
//     avoid division by zero.  Default: ``1e-12``.
// saved_norm_ : Storage
//     Per-slice denominator $\max(\|x\|_p, \epsilon)$ retained for backward,
//     negated for a slice whose norm is below $\epsilon$ (the backend's
//     contract; see ``IBackend::lp_normalize_forward``).
class LUCID_API LpNormalizeBackward : public FuncOp<LpNormalizeBackward, 1> {
public:
    static constexpr bool kKeepsOwnValues = true;
    // Registered op schema (see class docstring).
    static const OpSchema schema_v1;
    // Norm order $p$ (e.g. ``2.0`` for L2).
    double ord_ = 2.0;
    // Axis along which to normalise (negative allowed; resolved in forward).
    int axis_ = 1;
    // Minimum denominator to avoid division by zero.
    double eps_ = 1e-12;
    // Per-slice norm values needed for backward.
    Storage saved_norm_;

    void release_own_saved() { clear_saved(saved_norm_); }

    // Run the forward pass.
    //
    // Computes the per-slice Lp norm along ``axis``, divides ``x`` by
    // the clamped norm, caches ``saved_norm_``, and (when grad mode is
    // on) registers ``this`` as the ``grad_fn`` of the returned
    // output.  ``axis`` is normalised to a non-negative index before
    // being stored in :member:`axis_`.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input tensor of any shape.
    // ord : double
    //     Norm order $p$ (e.g. ``2.0`` for the L2 norm).
    // axis : int
    //     Axis along which to compute the norm.  Negative values count
    //     from the back.
    // eps : double
    //     Minimum allowed denominator.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Normalised output, same shape as ``x``.
    static TensorImplPtr forward(const TensorImplPtr& x, double ord, int axis, double eps);

    // Compute the gradient with respect to ``x``.
    //
    // Returns ``{dx}``.  The closed-form involves a projection of
    // ``grad_out`` along the normalised slice direction; see the .cpp
    // for the explicit formula.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming gradient, same shape as the forward output.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Single-element vector ``{dx}``.
    std::vector<Storage> apply(Storage grad_out) override;

    // The same gradient, recorded, with the norm recomputed from x.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Public free-function entry point for Lp normalisation.
//
// Thin wrapper that delegates to :func:`LpNormalizeBackward::forward`.
// This is the symbol the pybind11 binding layer forwards to from
// :func:`lucid.nn.functional.normalize`.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input tensor of any shape.
// ord : double
//     Norm order $p$.
// axis : int
//     Axis along which to compute the norm (negative allowed).
// eps : double
//     Minimum allowed denominator.
//
// Returns
// -------
// TensorImplPtr
//     Normalised output, same shape as ``x``.
//
// See Also
// --------
// LpNormalizeBackward : Autograd node implementing the forward + backward.
LUCID_API TensorImplPtr lp_normalize_op(const TensorImplPtr& x, double ord, int axis, double eps);

// Autograd node for Global Response Normalization (GRN), as used in
// ConvNeXt V2.
//
// For 4-D input of shape ``(N, C, H, W)``, GRN first computes the per
// ``(n, c)`` L2 norm over the spatial axes, then normalises each
// channel's norm by the mean across channels, and finally applies a
// learned per-channel affine that *modulates* the input rather than
// replacing it:
//
// $$
//   G_{n,c} = \|x_{n,c,:,:}\|_2,
//   \qquad
//   N_{n,c} = \frac{G_{n,c}}{\frac{1}{C}\sum_{c'} G_{n,c'} + \epsilon}
// $$
//
// $$
//   y_{n,c,h,w} = \gamma_c \,(x_{n,c,h,w}\,N_{n,c}) + \beta_c
//                 + x_{n,c,h,w}
// $$
//
// (The trailing $+ x$ residual matches the ConvNeXt-V2 paper; the
// affine is applied to the *modulated* signal rather than to
// $\hat{x}$.)  ``saved_Nx_`` stores $N$ broadcast to the per-spatial
// shape needed by the backward closed-form.
//
// Inherits ``FuncOp<GlobalResponseNormBackward, 3>`` with three
// saved-input slots — ``x``, ``gamma``, and ``beta``.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered schema (``name="global_response_norm"``, ``version=1``,
//     ``amp_policy=ForceFP32``, ``produces_grad=true``).
// eps_ : double
//     Numerical-stability constant in the per-channel normalisation
//     denominator.  Default: ``1e-6``.
// saved_Nx_ : Storage
//     Normalised global response $N$, shape ``(N, C, 1, 1)``.
//
// References
// ----------
// Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with
// Masked Autoencoders" (CVPR 2023, arXiv:2301.00808).
class LUCID_API GlobalResponseNormBackward : public FuncOp<GlobalResponseNormBackward, 3> {
public:
    static constexpr bool kKeepsOwnValues = true;
    // dx reads x, gamma, and beta (y = gamma * (x * Nx) + beta * x);
    // dgamma and dbeta read x.
    static constexpr GradReads<3> kGradReads = {
        {{true, true, true}, {true, false, false}, {true, false, false}}};
    // Registered op schema (see class docstring).
    static const OpSchema schema_v1;
    // Numerical-stability constant.
    double eps_ = 1e-6;

    // Normalised global response, shape ``(N, C, 1, 1)``.
    Storage saved_Nx_;

    void release_own_saved() { clear_saved(saved_Nx_); }

    // Run the forward pass.
    //
    // Computes the per ``(n, c)`` L2 spatial norm, normalises it by
    // the per-image mean across channels, modulates ``x`` by the
    // result, applies the per-channel affine + residual, caches
    // ``saved_Nx_``, and (when grad mode is on) registers ``this`` as
    // the ``grad_fn`` of the returned output.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     4-D input tensor of shape ``(N, C, H, W)``.
    // gamma : TensorImplPtr
    //     Per-channel scale of shape ``(C,)``.
    // beta : TensorImplPtr
    //     Per-channel shift of shape ``(C,)``.
    // eps : double
    //     Numerical-stability constant in the cross-channel mean.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Normalised output, same shape as ``x``.
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps);

    // Compute gradients for the three saved inputs.
    //
    // Returns ``{dx, d_gamma, d_beta}`` in the saved-input order
    // using the cached ``saved_Nx_``.
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
};

// Public free-function entry point for Global Response Normalization.
//
// Thin wrapper that delegates to
// :func:`GlobalResponseNormBackward::forward`.  This is the symbol the
// pybind11 binding layer forwards to from the ConvNeXt-V2 building
// block in :mod:`lucid.nn.functional`.
//
// Parameters
// ----------
// x : TensorImplPtr
//     4-D input tensor of shape ``(N, C, H, W)``.
// gamma : TensorImplPtr
//     Per-channel scale of shape ``(C,)``.
// beta : TensorImplPtr
//     Per-channel shift of shape ``(C,)``.
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
// GlobalResponseNormBackward : Autograd node implementing the forward + backward.
LUCID_API TensorImplPtr global_response_norm_op(const TensorImplPtr& x,
                                                const TensorImplPtr& gamma,
                                                const TensorImplPtr& beta,
                                                double eps);

}  // namespace lucid
