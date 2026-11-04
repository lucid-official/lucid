// lucid/_C/nn/Vision.h
//
// Miscellaneous vision utilities that do not belong to a larger module:
//
// * ``BilinearLayerBackward`` — autograd-aware bilinear form
//   $y = x_1^\top W x_2 + b$ with a rank-3 weight tensor.
// * ``one_hot_op`` — non-differentiable encoding of integer labels into
//   one-hot float vectors.
// * ``rotate_op`` — non-differentiable affine rotation of a 4-D image batch
//   around an arbitrary centre.

#pragma once

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for the bilinear (rank-3-weight) layer.
//
// Implements the bilinear form
// $$
//   y_{n, o} = \sum_{p=1}^{D_1} \sum_{q=1}^{D_2}
//     x_{1, n, p} \, W_{o, p, q} \, x_{2, n, q} + b_o
// $$
// over arbitrary leading batch dimensions, equivalent to the contraction
// ``y = einsum("...p,opq,...q->...o", x1, W, x2) + b``.  Both inputs and the
// weight are saved so that the backward can compute
// $$
//   \nabla_{x_1, n, p}     = \sum_{o, q} W_{o, p, q} \, x_{2, n, q} \, g_{n, o}, \quad
//   \nabla_{x_2, n, q}     = \sum_{o, p} W_{o, p, q} \, x_{1, n, p} \, g_{n, o}, \quad
//   \nabla_{W, o, p, q}    = \sum_{n}   x_{1, n, p} \, x_{2, n, q} \, g_{n, o}.
// $$
// The bias gradient is simply ``grad_out`` reduced over all batch axes.
//
// Math
// ----
// $$
//   y = x_1 \,W\, x_2^\top + b, \quad
//   W \in \mathbb{R}^{D_{\text{out}} \times D_1 \times D_2}, \;
//   b \in \mathbb{R}^{D_{\text{out}}}.
// $$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"bilinear_layer"``, ``AmpPolicy::Promote``.
// orig_x1_shape_ : Shape
//     Saved ``x1`` shape for backward reconstruction.
// orig_x2_shape_ : Shape
//     Saved ``x2`` shape for backward reconstruction.
//
// Notes
// -----
// All leading "batch" dimensions of ``x1`` and ``x2`` must match exactly.
// The bias is detected as absent (no bias term) when the saved bias shape
// recorded in ``input_shapes_[3]`` is empty.
class LUCID_API BilinearLayerBackward : public FuncOp<BilinearLayerBackward, 4> {
public:
    static const OpSchema schema_v1;
    // dx1 reads x2 and weight; dx2 reads x1 and weight; dW reads x1 and x2;
    // db reads only grad_out.
    static constexpr GradReads<4> kGradReads = {{{false, true, true, false},
                                                 {true, false, true, false},
                                                 {true, true, false, false},
                                                 {false, false, false, false}}};
    Shape orig_x1_shape_;  // Saved for backward shape reconstruction.
    Shape orig_x2_shape_;

    // Compute the bilinear layer with autograd wiring.
    //
    // Parameters
    // ----------
    // x1 : TensorImplPtr
    //     Tensor whose last dimension equals $D_1$.
    // x2 : TensorImplPtr
    //     Tensor whose last dimension equals $D_2$ and whose leading shape
    //     matches that of ``x1``.
    // weight : TensorImplPtr
    //     Weight tensor of shape ``(D_out, D_1, D_2)``.
    // bias : TensorImplPtr
    //     Optional bias vector of shape ``(D_out,)``.  Pass ``nullptr`` to
    //     disable the additive bias.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor of shape ``(*, D_out)`` where ``*`` denotes the
    //     shared leading shape of ``x1`` / ``x2``.
    static TensorImplPtr forward(const TensorImplPtr& x1,
                                 const TensorImplPtr& x2,
                                 const TensorImplPtr& weight,
                                 const TensorImplPtr& bias);

    // Backward pass: gradients w.r.t. ``x1``, ``x2``, ``weight``, and
    // (optionally) ``bias``.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward: the same three contractions as matmuls, so they
    // can be differentiated again.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Encode integer indices into one-hot float vectors.
//
// Returns a tensor of shape ``(*input.shape(), num_classes)`` where the last
// axis is the one-hot indicator
// $$
//   y_{\ldots, c} = \mathbf{1}\bigl[c = \text{input}_{\ldots}\bigr], \quad
//   c \in \{0, 1, \ldots, \text{num\_classes} - 1\}.
// $$
// The output dtype is ``out_dtype`` (typically a floating-point type so the
// result can feed directly into a loss function).  Indices outside the
// $[0, \text{num\_classes})$ range produce all-zero rows.  No autograd node
// is attached — this op is inference-only.
//
// Parameters
// ----------
// input : TensorImplPtr
//     Integer tensor of arbitrary shape.
// num_classes : int
//     Length of the one-hot dimension.
// out_dtype : Dtype
//     Element type of the output (e.g. ``Dtype::Float32``).
//
// Returns
// -------
// TensorImplPtr
//     One-hot tensor of shape ``(*input.shape(), num_classes)`` and dtype
//     ``out_dtype``.
LUCID_API TensorImplPtr one_hot_op(const TensorImplPtr& input, int num_classes, Dtype out_dtype);

// Rotate a 4-D image batch around an arbitrary centre.
//
// Applies the standard 2-D rotation matrix
// $$
//   \begin{pmatrix} x' \\ y' \end{pmatrix}
//   = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
//     \begin{pmatrix} x - c_x \\ y - c_y \end{pmatrix}
//   + \begin{pmatrix} c_x \\ c_y \end{pmatrix}
// $$
// with $\theta = \text{angle\_deg} \cdot \pi / 180$ to each channel of the
// input independently.  Sampling uses bilinear interpolation with zero
// padding for out-of-bounds destinations.  No autograd node is attached —
// the operation is inference-only.
//
// Parameters
// ----------
// input : TensorImplPtr
//     4-D image batch of shape ``(N, C, H, W)``.
// angle_deg : double
//     Rotation angle in degrees (counter-clockwise).
// cy : double
//     Rotation centre row coordinate (typically ``(H - 1) / 2``).
// cx : double
//     Rotation centre column coordinate (typically ``(W - 1) / 2``).
//
// Returns
// -------
// TensorImplPtr
//     Rotated image batch of shape ``(N, C, H, W)``.
LUCID_API TensorImplPtr rotate_op(const TensorImplPtr& input,
                                  double angle_deg,
                                  double cy,
                                  double cx);

// Public bilinear-layer entry point.
//
// Thin wrapper that delegates to ``BilinearLayerBackward::forward``.
//
// Parameters
// ----------
// x1 : TensorImplPtr
//     First operand of the bilinear form (last dim $= D_1$).
// x2 : TensorImplPtr
//     Second operand of the bilinear form (last dim $= D_2$).
// weight : TensorImplPtr
//     Weight tensor of shape ``(D_out, D_1, D_2)``.
// bias : TensorImplPtr
//     Optional bias vector of shape ``(D_out,)``; pass ``nullptr`` to omit.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of shape ``(*, D_out)``.
LUCID_API TensorImplPtr bilinear_layer_op(const TensorImplPtr& x1,
                                          const TensorImplPtr& x2,
                                          const TensorImplPtr& weight,
                                          const TensorImplPtr& bias);

}  // namespace lucid
