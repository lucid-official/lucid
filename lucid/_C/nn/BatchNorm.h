// lucid/_C/nn/BatchNorm.h
//
// Autograd-aware Batch Normalization for 1-D, 2-D, and 3-D spatial inputs.
//
// Batch Normalization standardises each channel of the input by the mean
// and variance estimated across the mini-batch and (when present) all
// spatial axes, then applies a per-channel affine transform.  This
// header declares the training-path autograd node
// ``BatchNormNdBackward<N>`` (parameterised by the number of spatial
// dimensions ``N``) and three free-function entry points
// (:func:`batch_norm1d_op`, :func:`batch_norm_op`,
// :func:`batch_norm3d_op`) that the Python binding layer forwards to.
//
// Inference using cached running statistics is *not* handled here — see
// :class:`BatchNormEvalBackward` in ``NormExt.h``.  Running-statistic
// buffers (``running_mean`` / ``running_var`` /
// ``num_batches_tracked``) are owned by Python and updated outside the
// autograd graph via ``tensor._copy_from()`` after each forward call so
// that the lazy MLX expression chain does not pin previous batches'
// activation graphs.
//
// All variants are registered with :member:`AmpPolicy::ForceFP32`:
// reductions over the batch + spatial axes accumulate enough terms that
// half-precision variance suffers from catastrophic cancellation, so
// the kernels run in FP32 regardless of the outer ``AutocastGuard``.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for training-mode Batch Normalization with N spatial axes.
//
// Computes per-channel statistics over the ``(B, S_0, ..., S_{N-1})``
// axes of the input, normalises the input by them, and applies the
// learnable per-channel affine $y = \gamma \hat{x} + \beta$.  The
// per-channel mean $\mu_c$ and reciprocal standard deviation
// $\text{rstd}_c = 1 / \sqrt{\sigma_c^2 + \epsilon}$ are saved in
// :member:`saved_mean_` / :member:`saved_rstd_` for the backward
// closed-form.
//
// Inherits ``FuncOp<BatchNormNdBackward<N>, 3>`` with three saved-input
// slots — ``x``, ``gamma``, and ``beta``.  The template is explicitly
// instantiated for ``N = 1, 2, 3`` in the .cpp file and exposed via
// the aliases :type:`BatchNorm1dBackward`, :type:`BatchNorm2dBackward`,
// and :type:`BatchNorm3dBackward`.
//
// Math
// ----
// $$
//   \mu_c = \frac{1}{B\,\prod_i S_i}
//           \sum_{b,\,\mathbf{s}} x_{b,c,\mathbf{s}},
//   \qquad
//   \sigma_c^2 = \frac{1}{B\,\prod_i S_i}
//           \sum_{b,\,\mathbf{s}} (x_{b,c,\mathbf{s}} - \mu_c)^2
// $$
//
// $$
//   \hat{x}_{b,c,\mathbf{s}} = (x_{b,c,\mathbf{s}} - \mu_c) \cdot
//       \frac{1}{\sqrt{\sigma_c^2 + \epsilon}},
//   \qquad y = \gamma_c\,\hat{x} + \beta_c
// $$
//
// The Python-side running-stats update applies, in order,
// $$
//   \hat{\mu}_t = (1 - m)\,\hat{\mu}_{t-1} + m\,\mu_c, \qquad
//   \hat{\sigma}^2_t = (1 - m)\,\hat{\sigma}^2_{t-1}
//                     + m \cdot \frac{n}{n-1}\,\sigma_c^2,
// $$
// where $n = B \prod_i S_i$ is the per-channel reduction count
// (the Bessel correction $n/(n-1)$ is applied only to the running
// variance buffer, never to the normalisation itself).
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered op schema.  ``name`` is ``"batch_norm1d"``,
//     ``"batch_norm"``, or ``"batch_norm3d"`` depending on ``N``;
//     ``version=1``; ``amp_policy=ForceFP32``;
//     ``produces_grad=true``.
// saved_mean_ : Storage
//     Per-channel mean $\mu_c$ of shape ``(C,)``, computed by the
//     backend and saved for backward.
// saved_rstd_ : Storage
//     Per-channel reciprocal standard deviation
//     $1 / \sqrt{\sigma_c^2 + \epsilon}$, shape ``(C,)``.
// B_, C_ : int
//     Batch and channel counts captured from ``x.shape()`` for the
//     backward.
// S_ : int[N]
//     Spatial sizes captured from ``x.shape()[2:]`` (length-1 dummy
//     array for the ``N==0`` corner case, which is never instantiated).
// eps_ : double
//     The forward's ``eps``; preserved so that the GPU MPSGraph
//     backward can derive variance from
//     $\sigma_c^2 = 1 / \text{rstd}_c^2 - \epsilon$.
//
// Shape
// -----
// - Input ``x`` : ``(B, C, S_0, ..., S_{N-1})``
// - Output     : same shape as ``x``
//
// Notes
// -----
// Thread safety: an instance is created once during forward and
// consumed once during backward — no concurrent access expected.
//
// References
// ----------
// Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network
// Training by Reducing Internal Covariate Shift" (ICML 2015).
template <int N>

class LUCID_API BatchNormNdBackward : public FuncOp<BatchNormNdBackward<N>, 3> {
public:
    static constexpr bool kKeepsOwnValues = true;
    // dx reads x and gamma; dgamma reads x; dbeta reads only grad_out.
    static constexpr GradReads<3> kGradReads = {
        {{true, true, false}, {true, false, false}, {false, false, false}}};
    // Registered op schema (see class docstring for fields).
    static const OpSchema schema_v1;
    // Per-channel mean $\mu_c$, shape ``(C,)``.
    Storage saved_mean_;
    // Per-channel reciprocal std $1 / \sqrt{\sigma_c^2 + \epsilon}$, shape ``(C,)``.
    Storage saved_rstd_;
    // 3.4+ Phase A.4 — normalised input $\hat{x} = (x - \mu) / \sqrt{\sigma^2 + \varepsilon}$,
    // shape ``(B, C, S_0, ..., S_{N-1})``.  Saving xnorm at forward time lets
    // backward skip the recomputation of ``centered = x - mean`` and
    // ``xnorm = centered * rstd`` — two element-wise ops on a full-size
    // tensor.  Forward already computes xnorm internally as an intermediate
    // (the MLX path materialises it before applying gamma + beta) so saving
    // it has no additional forward compute cost; the lazy chain just retains
    // a reference to the existing intermediate.  Holds memory across the
    // backward call, ~16 MB / BN layer on a 32×64×32×32 F32 ResNet-18 shape.
    Storage saved_xnorm_;
    // Batch ($B$) and channel ($C$) counts.
    int B_ = 0, C_ = 0;
    // Spatial sizes ``S_0, ..., S_{N-1}`` (guard array length 1 when ``N==0``).
    int S_[N > 0 ? N : 1];
    // Forward eps; reused on the GPU backward to derive variance from rstd.
    double eps_ = 1e-5;

    void release_own_saved() { clear_saved(saved_mean_, saved_rstd_, saved_xnorm_); }

    // Run the training-mode forward pass.
    //
    // Computes per-channel batch statistics, normalises ``x``, applies
    // the affine, saves $\mu_c$ and the reciprocal std for backward, and
    // (when grad mode is on) registers ``this`` as the ``grad_fn`` of
    // the returned output.  Dispatches to
    // ``IBackend::batch_norm_forward`` which returns ``[y, mean, rstd]``
    // in a single fused call.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input tensor of shape ``(B, C, S_0, ..., S_{N-1})``.
    // gamma : TensorImplPtr
    //     Per-channel scale of shape ``(C,)``.
    // beta : TensorImplPtr
    //     Per-channel shift of shape ``(C,)``.
    // eps : double
    //     Numerical-stability constant added inside the square root.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Normalised output, same shape as ``x``.
    // Forward.  When `running_mean` / `running_var` are non-null, the
    // EMA update happens INSIDE this call using the same mean+rstd the
    // backend already computed for the saved tensors — no second
    // reduction over `x`.  `momentum` is the EMA coefficient (must be
    // finite; cumulative mode is handled in the Python wrapper).
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& gamma,
                                 const TensorImplPtr& beta,
                                 double eps,
                                 const TensorImplPtr& running_mean = nullptr,
                                 const TensorImplPtr& running_var = nullptr,
                                 double momentum = 0.0);

    // Compute gradients for the three saved inputs.
    //
    // Returns ``{dx, d_gamma, d_beta}`` in the saved-input order using
    // ``IBackend::batch_norm_backward`` and the cached ``saved_mean_`` /
    // ``saved_rstd_``.  Closed-form summary (per channel $c$):
    //
    // $$
    //   d\gamma_c = \sum_{b,\mathbf{s}} \hat{x}_{b,c,\mathbf{s}}\,
    //               \text{grad\_out}_{b,c,\mathbf{s}},
    //   \qquad
    //   d\beta_c  = \sum_{b,\mathbf{s}} \text{grad\_out}_{b,c,\mathbf{s}}
    // $$
    //
    // and the input gradient follows the standard BN backward expansion.
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

    // Graph-mode backward.  Rebuilds the batch statistics from the saved
    // input: the stashed mean and rstd are plain Storages, so reusing them
    // would drop d(rstd)/dx from the second derivative while leaving the
    // first one exact.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Convenience alias: BatchNormNdBackward specialised to 1-D spatial inputs
// (sequences of shape ``(B, C, L)``).
using BatchNorm1dBackward = BatchNormNdBackward<1>;
// Convenience alias: BatchNormNdBackward specialised to 2-D spatial inputs
// (images of shape ``(B, C, H, W)``).
using BatchNorm2dBackward = BatchNormNdBackward<2>;
// Convenience alias: BatchNormNdBackward specialised to 3-D spatial inputs
// (volumes of shape ``(B, C, D, H, W)``).
using BatchNorm3dBackward = BatchNormNdBackward<3>;

// Public entry point for 1-D Batch Normalization.
//
// Applies training-mode batch normalization over the batch and length
// axes of a sequence input.  This is the symbol the pybind11 binding
// layer forwards to from :func:`lucid.nn.functional.batch_norm` when
// the input has rank 3.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input tensor of shape ``(B, C, L)``.
// gamma : TensorImplPtr
//     Per-channel scale of shape ``(C,)``.
// beta : TensorImplPtr
//     Per-channel shift of shape ``(C,)``.
// eps : double, optional
//     Numerical-stability constant.  Default: ``1e-5``.
//
// Returns
// -------
// TensorImplPtr
//     Normalised output, same shape as ``x``.
//
// See Also
// --------
// BatchNorm1dBackward : Autograd node implementing the forward + backward.
// batch_norm_eval_op  : Inference-mode counterpart using running stats.
LUCID_API TensorImplPtr batch_norm1d_op(const TensorImplPtr& x,
                                        const TensorImplPtr& gamma,
                                        const TensorImplPtr& beta,
                                        double eps = 1e-5,
                                        const TensorImplPtr& running_mean = nullptr,
                                        const TensorImplPtr& running_var = nullptr,
                                        double momentum = 0.0);

// Public entry point for 2-D Batch Normalization.
//
// Applies training-mode batch normalization over the batch and spatial
// (height, width) axes.  This is the most common batch-norm variant —
// used by virtually every CNN.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input tensor of shape ``(B, C, H, W)``.
// gamma : TensorImplPtr
//     Per-channel scale of shape ``(C,)``.
// beta : TensorImplPtr
//     Per-channel shift of shape ``(C,)``.
// eps : double, optional
//     Numerical-stability constant.  Default: ``1e-5``.
//
// Returns
// -------
// TensorImplPtr
//     Normalised output, same shape as ``x``.
//
// See Also
// --------
// BatchNorm2dBackward : Autograd node implementing the forward + backward.
LUCID_API TensorImplPtr batch_norm_op(const TensorImplPtr& x,
                                      const TensorImplPtr& gamma,
                                      const TensorImplPtr& beta,
                                      double eps = 1e-5,
                                      const TensorImplPtr& running_mean = nullptr,
                                      const TensorImplPtr& running_var = nullptr,
                                      double momentum = 0.0);

// Public entry point for 3-D Batch Normalization.
//
// Applies training-mode batch normalization over the batch and three
// spatial axes — used by 3-D CNNs for volumetric data (video,
// medical imaging).
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input tensor of shape ``(B, C, D, H, W)``.
// gamma : TensorImplPtr
//     Per-channel scale of shape ``(C,)``.
// beta : TensorImplPtr
//     Per-channel shift of shape ``(C,)``.
// eps : double, optional
//     Numerical-stability constant.  Default: ``1e-5``.
//
// Returns
// -------
// TensorImplPtr
//     Normalised output, same shape as ``x``.
//
// See Also
// --------
// BatchNorm3dBackward : Autograd node implementing the forward + backward.
LUCID_API TensorImplPtr batch_norm3d_op(const TensorImplPtr& x,
                                        const TensorImplPtr& gamma,
                                        const TensorImplPtr& beta,
                                        double eps = 1e-5,
                                        const TensorImplPtr& running_mean = nullptr,
                                        const TensorImplPtr& running_var = nullptr,
                                        double momentum = 0.0);

// Update ``running_mean`` / ``running_var`` / ``num_batches_tracked``
// in-place from ``x``.  Single-call C++ replacement for the per-BN-layer
// Python composition that previously cost ~8.8 ms / forward on ResNet-18
// via ~160 pybind11 crossings.
//
// reduce_axes  — dims of ``x`` to reduce over (e.g. ``[0, 2, 3]`` for
//                BatchNorm2d, ``[2, 3]`` for InstanceNorm2d).
// momentum     — EMA coefficient.  Must be finite; cumulative mode
//                (Python's ``momentum is None``) is rejected because it
//                requires reading ``num_batches_tracked.item()`` which
//                forces a GPU sync — callers should keep that on the
//                Python path.
// unbiased_var — ``true`` (BatchNorm convention) applies the ``n/(n-1)``
//                Bessel correction to the running variance; ``false``
//                for InstanceNorm.
LUCID_API void batch_norm_update_running_stats(std::shared_ptr<TensorImpl> running_mean,
                                               std::shared_ptr<TensorImpl> running_var,
                                               const std::shared_ptr<TensorImpl>& x,
                                               std::vector<int> reduce_axes,
                                               double momentum,
                                               bool unbiased_var = true);

}  // namespace lucid
