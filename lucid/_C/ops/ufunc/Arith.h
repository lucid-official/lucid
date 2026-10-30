// lucid/_C/ops/ufunc/Arith.h
//
// Autograd backward nodes and public entry points for the six basic arithmetic
// unary operations: neg, abs, sign, reciprocal, square, cube.  Each class
// follows the standard UnaryOp<Derived> CRTP pattern: a static dispatch()
// routes the forward computation through IBackend, and grad_formula()
// implements the analytic gradient rule.

#pragma once

#include <utility>
#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Autograd node for element-wise negation $y = -x$.
//
// The gradient is the negation of the upstream signal; no input or output
// needs to be saved because the rule contains no functional dependence on
// $x$.  ``kSavesInput = false`` opts out of the default save behaviour
// provided by :class:`UnaryKernel`.
//
// Math
// ----
// $$y = -x, \qquad \frac{\partial y}{\partial x} = -1, \qquad
// \frac{\partial \mathcal{L}}{\partial x} = -\frac{\partial \mathcal{L}}
// {\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"neg"`` with ``AmpPolicy::Promote`` — integer inputs
//     are upcast to float before dispatch so the algebraic identity holds
//     for the same dtype on the return path.
// kSavesInput : bool
//     ``false``.  Backward needs no saved value.
//
// Notes
// -----
// Dispatch: Accelerate ``vDSP_vneg`` (CPU) / MLX ``negative`` (GPU).
class LUCID_API NegBackward : public UnaryOp<NegBackward> {
public:
    static constexpr bool kSavesInput = false;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::neg`` to compute $y = -x$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.neg(a, s, dt);
    }
    // Backward — $\partial y/\partial x = -1$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward (default-impl helper): returns ``-g`` so the chain
    // stays differentiable for higher-order gradients.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr&);
};

// Autograd node for the element-wise absolute value $y = |x|$.
//
// Saves the input ``x`` so the backward pass can recover $\mathrm{sign}(x)$
// and scale the upstream gradient by it.  The derivative is undefined at
// $x = 0$; the implementation follows the reference framework's convention
// of returning $0$ there (sub-gradient).
//
// Math
// ----
// $$y = |x|, \qquad
// \frac{\partial y}{\partial x} = \mathrm{sign}(x), \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \mathrm{sign}(x)\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"abs"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``vDSP_vabs`` (CPU) / MLX ``abs`` (GPU).
// Backward for ``abs`` of a *complex* input.
//
// The forward is ``hypot_lanes`` of the real and imaginary lanes, a composite
// chosen to be overflow-safe.  The derivative is stated directly instead of
// taken through it: that chain rule runs through a divide by the larger lane,
// which was NaN at exactly zero (``0 / 0``) when the divisor was floored, so
// ``abs(fft2(x))`` came back all-NaN whenever one coefficient of the
// transform was zero — which for most real inputs, one is.  The stated form
// is exact and simple:
//
//     d|z| / dz = z / |z|,     and 0 at z = 0
//
// which is the subgradient the reference reports there.
LUCID_API TensorImplPtr complex_abs_forward(const TensorImplPtr& a);

class LUCID_API ComplexAbsBackward : public FuncOp<ComplexAbsBackward, 1> {
public:
    static const OpSchema schema_v1;
    // ``z`` itself, for ``z / |z|``, is saved by value in ``saved_inputs_[0]``.
    std::vector<Storage> apply(Storage grad_out) override;
    // The same ``g * z / |z|`` from tracked ops, for ``create_graph=True``.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

class LUCID_API AbsBackward : public UnaryOp<AbsBackward> {
public:
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::abs`` to compute $y = |x|$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.abs(a, s, dt);
    }
    // Backward — $\partial y/\partial x = \mathrm{sign}(x)$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);

    // Graph-mode backward: the same derivative built from composable ops,
    // so the gradient itself stays differentiable.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Autograd node for the element-wise sign function
// $y = \mathrm{sign}(x) \in \{-1, 0, +1\}$.
//
// Because ``sign`` is piecewise constant its derivative vanishes almost
// everywhere; the node therefore carries no gradient information.
// ``kHasGradient = false`` tells :class:`UnaryKernel::forward` to skip the
// autograd wiring entirely, and ``grad_formula`` returns an empty
// ``CpuStorage`` only as a defensive zero-sentinel if it is ever invoked
// manually.
//
// Math
// ----
// $$y = \mathrm{sign}(x), \qquad
// \frac{\partial y}{\partial x} = 0 \text{ almost everywhere}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"sign"`` with ``AmpPolicy::KeepInput`` — integer
//     dtypes pass through unmodified.
// kSavesInput : bool
//     ``false``.
// kHasGradient : bool
//     ``false`` — no autograd edge is created.
//
// Notes
// -----
// Dispatch: Accelerate sign kernel (CPU) / MLX ``sign`` (GPU).
class LUCID_API SignBackward : public UnaryOp<SignBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::sign`` to compute $y = \mathrm{sign}(x) \in \{-1, 0, +1\}$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sign(a, s, dt);
    }
    // Backward — $\partial y/\partial x = 0$ almost everywhere: a zero
    // gradient, kept in the graph as the reference framework keeps it.
    Storage grad_formula(const Storage& g);

    // Graph-mode backward: the same derivative built from composable ops,
    // so the gradient itself stays differentiable.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr&);
};

// Autograd node for the element-wise reciprocal $y = 1/x$.
//
// Saves the input ``x`` so the backward pass can form $x^2$ in the
// denominator without recomputing the forward result.  The forward and
// backward are both undefined at $x = 0$ and the caller is responsible for
// avoiding that domain point.
//
// Math
// ----
// $$y = \frac{1}{x}, \qquad
// \frac{\partial y}{\partial x} = -\frac{1}{x^2}, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// -\frac{1}{x^2}\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"reciprocal"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: a correctly rounded ``1 / x`` on CPU (``vvrec`` for F64, a plain
// loop for F32 because ``vvrecf`` is not exact — backend/cpu/Vforce.cpp) /
// MLX element-wise reciprocal (GPU).
class LUCID_API ReciprocalBackward : public UnaryOp<ReciprocalBackward> {
public:
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::reciprocal`` to compute $y = 1/x$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.reciprocal(a, s, dt);
    }
    // Backward — $\partial y/\partial x = -1/x^2$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward: $\partial x = -g / x^2$ kept inside the autograd graph.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Autograd node for the element-wise square $y = x^2$.
//
// Saves the input ``x`` so the backward pass can scale the upstream
// gradient by $2x$.
//
// Math
// ----
// $$y = x^2, \qquad
// \frac{\partial y}{\partial x} = 2x, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// 2x\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"square"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``vDSP_vsq`` (CPU) / MLX ``square`` (GPU).
class LUCID_API SquareBackward : public UnaryOp<SquareBackward> {
public:
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::square`` to compute $y = x^2$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.square(a, s, dt);
    }
    // Backward — $\partial y/\partial x = 2x$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward: $\partial x = 2x \cdot g$ kept inside the autograd graph.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Autograd node for the element-wise cube $y = x^3$.
//
// Saves the input ``x`` so the backward pass can build $3x^2$ in
// ``grad_formula`` and multiply it against the upstream gradient.
//
// Math
// ----
// $$y = x^3, \qquad
// \frac{\partial y}{\partial x} = 3x^2, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// 3x^2\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"cube"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: dedicated Accelerate kernel (CPU) / MLX element-wise cube (GPU).
class LUCID_API CubeBackward : public UnaryOp<CubeBackward> {
public:
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::cube`` to compute $y = x^3$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cube(a, s, dt);
    }
    // Backward — $\partial y/\partial x = 3x^2$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward: $3x^2 \cdot g$ kept inside the autograd graph.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Public entry points.  Each thin wrapper delegates to the corresponding
// backward node's static forward() method, which handles dispatch and autograd.

// Compute $y = -x$ element-wise.
//
// Allocates a fresh output of the same shape and dtype as ``a`` and routes
// the forward pass through :class:`NegBackward`, which also installs the
// autograd edge.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.
//
// See Also
// --------
// :class:`NegBackward` — backward node.
LUCID_API TensorImplPtr neg_op(const TensorImplPtr& a);

// Compute $y = |x|$ element-wise.
//
// Allocates a fresh output of the same shape and dtype as ``a`` and
// delegates to :class:`AbsBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype, with non-negative values.
//
// See Also
// --------
// :class:`AbsBackward` — backward node.
LUCID_API TensorImplPtr abs_op(const TensorImplPtr& a);

// Compute $y = \mathrm{sign}(x) \in \{-1, 0, +1\}$ element-wise.
//
// Allocates a fresh output of the same shape and dtype as ``a`` and
// delegates to :class:`SignBackward::forward`.  No autograd edge is
// recorded because ``sign`` has a zero gradient almost everywhere.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any real dtype, including integer.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype, with values in
//     $\{-1, 0, +1\}$.
//
// See Also
// --------
// :class:`SignBackward` — backward node.
LUCID_API TensorImplPtr sign_op(const TensorImplPtr& a);

// Compute $y = 1 / x$ element-wise.
//
// Allocates a fresh output of the same shape and dtype as ``a`` and
// delegates to :class:`ReciprocalBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.  Behaviour is undefined at $x = 0$.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.
//
// See Also
// --------
// :class:`ReciprocalBackward` — backward node.
LUCID_API TensorImplPtr reciprocal_op(const TensorImplPtr& a);

// Compute $y = x^2$ element-wise.
//
// Allocates a fresh output of the same shape and dtype as ``a`` and
// delegates to :class:`SquareBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.
//
// See Also
// --------
// :class:`SquareBackward` — backward node.
LUCID_API TensorImplPtr square_op(const TensorImplPtr& a);

// Compute $y = x^3$ element-wise.
//
// Allocates a fresh output of the same shape and dtype as ``a`` and
// delegates to :class:`CubeBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.
//
// See Also
// --------
// :class:`CubeBackward` — backward node.
LUCID_API TensorImplPtr cube_op(const TensorImplPtr& a);

}  // namespace lucid
