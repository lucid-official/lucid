// lucid/_C/ops/bfunc/Pow.h
//
// Declares PowBackward, the autograd node for element-wise exponentiation, and
// the public free function pow_op.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../autograd/Helpers.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// Autograd node for elementwise tensor-tensor power $y = a^b$ with NumPy
// broadcasting.
//
// Saves both inputs ``a`` and ``b`` (``kSavesInputs = true`` inherited from
// :class:`BinaryKernel`) because both are needed in the backward formulas.
// The gradient w.r.t. ``a`` is finite for any real $a$; the gradient w.r.t.
// ``b`` is undefined when $a \leq 0$ because $\log(a)$ is not real — in that
// regime the returned gradient is non-finite and should be masked by the
// caller.
//
// Math
// ----
// $$
//   y = a^b
// $$
// $$
//   \frac{\partial L}{\partial a} = b \cdot a^{b - 1} \cdot \frac{\partial L}{\partial y}
// $$
// $$
//   \frac{\partial L}{\partial b} = \log(a) \cdot a^b \cdot \frac{\partial L}{\partial y}
// $$
//
// Shape
// -----
// Inputs ``a``, ``b`` follow NumPy broadcasting rules; the output ``y`` takes
// the broadcast shape.  Both backward branches are sum-reduced back to the
// original input shapes by :func:`sum_to_shape` in the apply trampoline.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"pow"``, version ``1``, :enum:`AmpPolicy::ForceFP32`,
//     ``integral_closed``, deterministic.  A floating power runs in FP32
//     under autocast; an integer or bool power stays in its own dtype.
//
// Notes
// -----
// For tensor-scalar power (constant exponent) prefer the scalar variant in
// ``ScalarParam.h`` — it skips saving the exponent tensor and avoids the
// ``log(a) * a^b`` branch entirely.  See :class:`BinaryKernel` for the
// broadcasting / save-tensor / reduce-to-shape machinery.
//
// See Also
// --------
// mul_op : Elementwise multiplication.
// log_op : Natural logarithm (used by the $\partial L/\partial b$ branch).
class LUCID_API PowBackward : public BinaryOp<PowBackward> {
public:
    // Op registration metadata: name "pow", schema version 1, ForceFP32
    // under AMP for floating inputs, integer inputs kept, deterministic.
    static const OpSchema schema_v1;

    // Route the forward computation through the backend's pow primitive.
    static Storage dispatch(
        backend::IBackend& be, const Storage& a, const Storage& b, const Shape& shape, Dtype dt) {
        return be.pow(a, b, shape, dt);
    }

    // Compute the gradients for both inputs given the output gradient.
    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);

    // Graph-mode gradient for create_graph=True.
    // dA = where(b == 0, 0, b * a^(b-1)) * g;
    // dB = where(a == 0 && b >= 0, 0, log(a) * a^b) * g.
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& g,
                                                              const TensorImplPtr& a_ptr,
                                                              const TensorImplPtr& b_ptr);
};

// Compute the elementwise tensor-tensor power $y = a^b$ with broadcasting and
// autograd support.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Base tensor.
// b : TensorImplPtr
//     Exponent tensor.  Broadcast-compatible with ``a``.
//
// Returns
// -------
// TensorImplPtr
//     Tensor of shape ``broadcast(a.shape, b.shape)`` holding $a^b$.
//
// Notes
// -----
// Integer inputs stay in their integer dtype; floating inputs follow
// :class:`PowBackward` schema under AMP.  When grad-tracking is on, registers
// a :class:`PowBackward` node which saves both inputs.
//
// Examples
// --------
// >>> auto y = pow_op(a, b);  // y[i] = a[i] ** b[i] with broadcasting
LUCID_API TensorImplPtr pow_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
