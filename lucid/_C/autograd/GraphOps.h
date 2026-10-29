// lucid/_C/autograd/GraphOps.h
//
// The ops the autograd layer calls into the ops layer for, declared in the
// one place the layer check allows it (tools/check_cpp_layers.py, AE-7).
// A create_graph gradient is a tensor whose sums and casts must be graph ops
// so that they differentiate again; the ops layer sits above this one, so
// they are declared here and defined there.

#pragma once

#include "../core/TensorImpl.h"

namespace lucid {

// ``a + b`` as a recorded op (ops/bfunc/Add.cpp).
TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);

// ``conj(a)`` as a recorded op (ops/complex/Conj.cpp), for the complex
// gradient convention in ComplexGrad.h.
TensorImplPtr conj_op(const TensorImplPtr& a);

}  // namespace lucid
