// lucid/_C/ops/ufunc/Exponential.h
//
// Autograd backward nodes and entry points for the exponential, logarithmic,
// root, and error-function family: exp, log, log2, sqrt, rsqrt, erf, erfinv.
// On CPU, the backend routes to vForce (vvexpf, vvlogf, …) for SIMD
// throughput.  ``exp`` / ``log`` / ``log2`` request ``AmpPolicy::ForceFP32``
// so half-precision inputs are upcast before computation; the remainder use
// ``AmpPolicy::Promote``.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Autograd node for element-wise exponential $y = e^x$.
//
// Saves the *output* $y$ rather than the input because the backward
// formula is $\partial y / \partial x = e^x = y$ and reusing the saved
// output avoids a second ``vvexpf`` pass.
//
// Math
// ----
// $$y = e^x, \qquad
// \frac{\partial y}{\partial x} = e^x = y, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// y\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"exp"`` with ``AmpPolicy::ForceFP32`` — half-
//     precision inputs are upcast to F32 prior to dispatch to avoid
//     premature overflow / underflow.
// kSavesInput : bool
//     ``false``.
// kSavesOutput : bool
//     ``true`` — saved tensor is $y = e^x$.
//
// Notes
// -----
// Dispatch: Accelerate ``vvexpf``/``vvexp`` (CPU) / MLX ``exp`` (GPU).
// Large-magnitude positive inputs overflow to ``+inf``; large-magnitude
// negative inputs flush to ``0``.
class LUCID_API ExpBackward : public UnaryOp<ExpBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::exp`` to compute $y = e^x$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.exp(a, shape, dt);
    }
    // Backward — $\partial y/\partial x = e^x = y$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward: $\partial x = \mathrm{out} \cdot g$ using the saved
    // output $e^x$, kept inside the autograd graph.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr& out);
};

// Autograd node for the element-wise natural logarithm $y = \ln(x)$.
//
// Saves the input ``x`` so the backward pass can divide the upstream
// gradient by it.  Defined only for strictly positive inputs; non-positive
// values produce NaN / $-\infty$, matching the reference framework.
//
// Math
// ----
// $$y = \ln(x), \qquad
// \frac{\partial y}{\partial x} = \frac{1}{x}, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \frac{1}{x}\,\frac{\partial \mathcal{L}}{\partial y}, \quad x > 0.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"log"`` with ``AmpPolicy::ForceFP32``.
//
// Notes
// -----
// Dispatch: Accelerate ``vvlogf``/``vvlog`` (CPU) / MLX ``log`` (GPU).
class LUCID_API LogBackward : public UnaryOp<LogBackward> {
public:
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::log`` to compute $y = \ln x$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.log(a, shape, dt);
    }
    // Backward — $\partial y/\partial x = 1/x$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward: $\partial x = g / x$ kept inside the autograd graph.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Native real log1p. The graph form of its VJP keeps all higher orders.
class LUCID_API Log1pBackward : public UnaryOp<Log1pBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.log1p(a, shape, dt);
    }
    Storage grad_formula(const Storage& g);
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Autograd node for the element-wise base-2 logarithm $y = \log_2(x)$.
//
// Saves the input ``x``; the backward divides by $x \ln 2$.  The constant
// $\ln 2$ is materialised in ``grad_formula`` as a high-precision literal
// so accuracy is identical to scaling by ``1 / log(2.0)``.
//
// Math
// ----
// $$y = \log_2(x), \qquad
// \frac{\partial y}{\partial x} = \frac{1}{x \ln 2}, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \frac{1}{x \ln 2}\,\frac{\partial \mathcal{L}}{\partial y}, \quad x > 0.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"log2"`` with ``AmpPolicy::ForceFP32``.
//
// Notes
// -----
// Dispatch: Accelerate ``vvlog2f``/``vvlog2`` (CPU) / MLX ``log2`` (GPU).
class LUCID_API Log2Backward : public UnaryOp<Log2Backward> {
public:
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::log2`` to compute $y = \log_2 x$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.log2(a, s, dt);
    }
    // Backward — $\partial y/\partial x = 1/(x \ln 2)$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);

    // Graph-mode backward: the same derivative built from composable ops,
    // so the gradient itself stays differentiable.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Autograd node for the element-wise square root $y = \sqrt{x}$.
//
// Saves the *output* $y$ because the backward formula
// $\partial y / \partial x = 1 / (2y)$ is cheaper and numerically
// identical to recomputing $\sqrt{x}$.  Defined for $x \geq 0$; negative
// inputs produce NaN.
//
// Math
// ----
// $$y = \sqrt{x}, \qquad
// \frac{\partial y}{\partial x} = \frac{1}{2\sqrt{x}} = \frac{1}{2y},
// \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \frac{1}{2y}\,\frac{\partial \mathcal{L}}{\partial y}, \quad x \geq 0.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"sqrt"`` with ``AmpPolicy::Promote``.
// kSavesInput : bool
//     ``false``.
// kSavesOutput : bool
//     ``true`` — saved tensor is $y = \sqrt{x}$.
//
// Notes
// -----
// Dispatch: a correctly rounded root on CPU (``vvsqrt`` for F64, a plain loop
// for F32 because ``vvsqrtf`` is not exact — backend/cpu/Vforce.cpp) / MLX
// ``sqrt`` (GPU).
// Gradient is unbounded as $x \to 0^+$.
class LUCID_API SqrtBackward : public UnaryOp<SqrtBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::sqrt`` to compute $y = \sqrt{x}$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.sqrt(a, shape, dt);
    }
    // Backward — $\partial y/\partial x = 1/(2y)$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward: $\partial x = g / (2y)$ where $y = \sqrt{x}$ is the
    // saved output; kept inside the autograd graph.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr& out);
};

// Autograd node for the element-wise reciprocal square root
// $y = 1/\sqrt{x}$.
//
// Saves the *output* $y$ so the backward pass can form $y^3$ without
// re-evaluating ``rsqrt``.  Defined for $x > 0$.
//
// Math
// ----
// $$y = \frac{1}{\sqrt{x}}, \qquad
// \frac{\partial y}{\partial x} = -\tfrac{1}{2}\,x^{-3/2} =
// -\tfrac{1}{2}\,y^3, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// -\tfrac{1}{2}\,y^3\,\frac{\partial \mathcal{L}}{\partial y},
// \quad x > 0.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"rsqrt"`` with ``AmpPolicy::Promote``.
// kSavesInput : bool
//     ``false``.
// kSavesOutput : bool
//     ``true`` — saved tensor is $y = 1/\sqrt{x}$.
//
// Notes
// -----
// Dispatch: on CPU, the correctly rounded root and then the correctly
// rounded reciprocal of it, which is how the reference computes it. On GPU,
// MLX ``rsqrt``.
class LUCID_API RsqrtBackward : public UnaryOp<RsqrtBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::rsqrt`` to compute $y = 1/\sqrt{x}$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.rsqrt(a, shape, dt);
    }
    // Backward — $\partial y/\partial x = -\tfrac{1}{2}\,y^3$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);

    // Graph-mode backward: the same derivative built from composable ops,
    // so the gradient itself stays differentiable.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr& out);
};

// Autograd node for the element-wise error function
// $y = \mathrm{erf}(x) = (2/\sqrt{\pi})\int_0^x e^{-t^2}\,dt$.
//
// Saves the input ``x`` so the backward pass can build $e^{-x^2}$.
// Defined on all of $\mathbb{R}$ with range $(-1, 1)$.
//
// Math
// ----
// $$y = \mathrm{erf}(x), \qquad
// \frac{\partial y}{\partial x} = \frac{2}{\sqrt{\pi}}\,e^{-x^2},
// \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \frac{2}{\sqrt{\pi}}\,e^{-x^2}\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"erf"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``erf`` kernel (CPU) / MLX ``erf`` (GPU).
class LUCID_API ErfBackward : public UnaryOp<ErfBackward> {
public:
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::erf`` to compute $y = \mathrm{erf}(x)$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.erf(a, shape, dt);
    }
    // Backward — $\partial y/\partial x = (2/\sqrt{\pi})\,e^{-x^2}$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward: $\partial x = (2/\sqrt{\pi})\,e^{-x^2}\,g$ kept inside
    // the autograd graph.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr& x, const TensorImplPtr&);
};

// Compute $y = e^x$ element-wise.  Allocates a fresh output of the same
// shape and dtype as ``a`` and delegates to :class:`ExpBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.  Half-precision inputs are upcast to F32
//     before computation, per ``AmpPolicy::ForceFP32``.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.  Large positive values
//     overflow to ``+inf``; large negative values flush to ``0``.
//
// See Also
// --------
// :class:`ExpBackward` — backward node.
LUCID_API TensorImplPtr exp_op(const TensorImplPtr& a);

// Compute $y = \ln(x)$ element-wise.  Allocates a fresh output of the same
// shape and dtype as ``a`` and delegates to :class:`LogBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.  Must be strictly positive for finite
//     output; non-positive values produce NaN / $-\infty$.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.
//
// See Also
// --------
// :class:`LogBackward` — backward node.
LUCID_API TensorImplPtr log_op(const TensorImplPtr& a);

// Compute $y = \log_2(x)$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`Log2Backward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.  Must be strictly positive.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.
//
// See Also
// --------
// :class:`Log2Backward` — backward node.
LUCID_API TensorImplPtr log2_op(const TensorImplPtr& a);

// Compute $y = \sqrt{x}$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`SqrtBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.  Must be non-negative; negative values
//     produce NaN.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.
//
// See Also
// --------
// :class:`SqrtBackward` — backward node.
LUCID_API TensorImplPtr sqrt_op(const TensorImplPtr& a);

// Compute $y = 1/\sqrt{x}$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`RsqrtBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.  Must be strictly positive.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.
//
// See Also
// --------
// :class:`RsqrtBackward` — backward node.
LUCID_API TensorImplPtr rsqrt_op(const TensorImplPtr& a);

// Compute $y = \mathrm{erf}(x)$ element-wise.  Allocates a fresh output of
// the same shape and dtype as ``a`` and delegates to
// :class:`ErfBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.  Domain is all of $\mathbb{R}$.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype, values in $(-1, 1)$.
//
// See Also
// --------
// :class:`ErfBackward` — backward node.
LUCID_API TensorImplPtr erf_op(const TensorImplPtr& a);

// Autograd node for the element-wise inverse error function
// $y = \mathrm{erfinv}(x)$.
//
// Saves the *output* $y$ so the backward pass can form $e^{y^2}$ without
// needing to invert $\mathrm{erf}$ a second time.  Defined on $(-1, 1)$;
// the gradient is unbounded as $x \to \pm 1$.
//
// Math
// ----
// $$y = \mathrm{erfinv}(x), \qquad
// \frac{\partial y}{\partial x} = \frac{\sqrt{\pi}}{2}\,e^{y^2},
// \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \frac{\sqrt{\pi}}{2}\,e^{y^2}\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"erfinv"`` with ``AmpPolicy::Promote``.
// kSavesInput : bool
//     ``false``.
// kSavesOutput : bool
//     ``true`` — saved tensor is $y = \mathrm{erfinv}(x)$.
//
// Notes
// -----
// Dispatch: Accelerate ``erfinv`` kernel (CPU) / MLX ``erfinv`` (GPU).
class LUCID_API ErfinvBackward : public UnaryOp<ErfinvBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kSavesOutput = true;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::erfinv`` to compute $y = \mathrm{erfinv}(x)$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& shape, Dtype dt) {
        return be.erfinv(a, shape, dt);
    }
    // Backward — $\partial y/\partial x = \tfrac{\sqrt{\pi}}{2}\,e^{y^2}$, scaled by ``grad_out``.
    Storage grad_formula(const Storage& g);
    // Graph-mode backward: $\partial x = \tfrac{\sqrt{\pi}}{2}\,e^{\mathrm{out}^2}\,g$
    // using the saved output, kept inside the autograd graph.
    TensorImplPtr
    grad_formula_impl(const TensorImplPtr& g, const TensorImplPtr&, const TensorImplPtr& out);
};

// Compute $y = \mathrm{erfinv}(x)$ element-wise.  Allocates a fresh output
// of the same shape and dtype as ``a`` and delegates to
// :class:`ErfinvBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape with values in $(-1, 1)$.  Values at
//     $\pm 1$ map to $\pm\infty$; out-of-domain values produce NaN.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.
//
// See Also
// --------
// :class:`ErfinvBackward` — backward node.
LUCID_API TensorImplPtr erfinv_op(const TensorImplPtr& a);

}  // namespace lucid
