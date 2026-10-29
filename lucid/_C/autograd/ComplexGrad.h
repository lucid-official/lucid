// lucid/_C/autograd/ComplexGrad.h
//
// The complex gradient convention, stated once.
//
// For a real loss L of z = x + iy, ``z.grad`` is the reference's
// dL/dx + i dL/dy.  Through a holomorphic w = f(z) that quantity obeys
//
//     grad_z = grad_w * conj(f'(z))
//
// while the engine's backward formulas are the real-dtype ones,
// grad_w * f'(z) — and g @ B^T, not g @ B^H, for a matrix product.  Each such
// formula D is linear in grad_w over the complex numbers, so
//
//     conj(D(conj(g))) = conj(conj(g) * f'(z)) = g * conj(f'(z))
//
// turns it into the complex VJP with no per-op branch.  UnaryKernel and
// BinaryKernel apply that around ``grad_formula`` / ``grad_formula_impl`` on
// every complex dtype; a node with its own ``apply`` (matmul, linear) calls
// the helpers below itself.  Before this, mul and div conjugated their
// operands by hand and every other holomorphic op — exp, log, sqrt, sin, pow,
// matmul, … — returned g * f'(z) for a complex input, wrong with no error
// (LCD-661).

#pragma once

#include <cstdint>

#include "../backend/Dispatcher.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "GraphOps.h"

namespace lucid {

// What a kernel base does around an op's backward formula on a complex dtype.
//
// ``Conjugate`` (the default): the formula is the real-dtype chain rule, and
// the base conjugates the upstream gradient before it and the result after.
// ``AsWritten``: every local derivative the formula multiplies by is real
// (``add``, ``sub``), so it is already the complex VJP and runs unchanged.
enum class ComplexGrad : std::uint8_t { Conjugate, AsWritten };

// ``s`` (at ``shape`` / ``dt``) conjugated when ``dt`` is complex.  A real
// dtype returns ``s`` itself; the backend does not copy it.
inline Storage conj_if_complex(const Storage& s, const Shape& shape, Dtype dt, Device device) {
    if (!is_complex(dt))
        return s;
    return backend::Dispatcher::for_device(device).complex_conj(s, shape, dt);
}

// ``t`` conjugated as a recorded op, so a create_graph gradient built on it
// differentiates again.  A real or null tensor is returned as it is.
inline TensorImplPtr conj_if_complex(const TensorImplPtr& t) {
    return (t && is_complex(t->dtype())) ? conj_op(t) : t;
}

}  // namespace lucid
