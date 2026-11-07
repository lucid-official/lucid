// lucid/_C/bindings/bind_ufunc.cpp
//
// Registers all unary tensor operations on the top-level engine module.
// Unary ops are grouped into:
//   1. Differentiable arithmetic: neg, abs, sign, reciprocal, square, cube,
//      cube_root.
//   2. Exponential / logarithmic: exp, log, log2, sqrt.
//   3. Trigonometric: sin, cos, tan, arcsin, arccos, arctan.
//   4. Hyperbolic: sinh, cosh, tanh.
//   5. Activation functions: relu, sigmoid, silu, gelu, leaky_relu, softplus,
//      elu, selu, mish, hard_sigmoid, hard_swish, relu6, softmax.
//   6. Scalar-parameter ops: pow_scalar, rpow_scalar, clip.
//   7. Discrete / rounding: round, floor, ceil, invert.
//   8. Reductions (with axes/keepdims): sum, mean, prod, max, min, var.
//   9. Axis manipulation ops (also used by bind_utils): permute, transpose, T,
//      mT, swapaxes.
//  10. Scan ops: cumsum, cumprod.
//  11. In-place variants for groups 1-7 (underscore suffix; no autograd).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../ops/ufunc/Activation.h"
#include "../ops/ufunc/Arith.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/ufunc/CubeRoot.h"
#include "../ops/ufunc/Discrete.h"
#include "../ops/ufunc/Exponential.h"
#include "../ops/ufunc/Hyperbolic.h"
#include "../ops/ufunc/Inplace.h"
#include "../ops/ufunc/Predicate.h"
#include "../ops/ufunc/Reductions.h"
#include "../ops/ufunc/ScalarParam.h"
#include "../ops/ufunc/Scan.h"
#include "../ops/ufunc/Softmax.h"
#include "../ops/ufunc/ToDevice.h"
#include "../ops/ufunc/Trace.h"
#include "../ops/ufunc/Transpose.h"
#include "../ops/ufunc/Trig.h"
#include "../ops/ufunc/Var.h"
#include "BindingGen.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {
// ``max`` / ``min``, with the choice of how a NaN slice's gradient is shaped.
void bind_extremum(py::module_& m) {
    py::enum_<ExtremumNanGrad>(
        m, "ExtremumNanGrad",
        "The gradient of max / min for a slice that holds a NaN: SLICE gives every element "
        "NaN (amax), ELEMENT splits it among the NaN elements (max()), ZERO gives zeros "
        "(vector_norm(inf)).")
        .value("SLICE", ExtremumNanGrad::Slice)
        .value("ELEMENT", ExtremumNanGrad::Element)
        .value("ZERO", ExtremumNanGrad::Zero);
    m.def("max", &max_with_nan_grad_op, py::arg("a"), py::arg("dim") = std::vector<int>{},
          py::arg("keepdims") = false, py::arg("nan_grad") = ExtremumNanGrad::Slice, "Reduce-max.");
    m.def("min", &min_with_nan_grad_op, py::arg("a"), py::arg("dim") = std::vector<int>{},
          py::arg("keepdims") = false, py::arg("nan_grad") = ExtremumNanGrad::Slice, "Reduce-min.");
}
}  // namespace

// Registers all unary, reduction, axis, and scan ops.
void register_ufunc(py::module_& m) {
    // Differentiable arithmetic unary ops.  bind_unary reads the Python name
    // from the BackwardNode's schema_v1 field.
    bind_unary<NegBackward>(m, &neg_op);
    bind_unary<AbsBackward>(m, &abs_op);
    bind_unary<SignBackward>(m, &sign_op);
    bind_unary<ReciprocalBackward>(m, &reciprocal_op);
    bind_unary<SquareBackward>(m, &square_op);
    bind_unary<CubeBackward>(m, &cube_op);
    bind_unary<CubeRootBackward>(m, &cube_root_op);

    // Exponential and logarithmic ops.
    bind_unary<ExpBackward>(m, &exp_op);
    bind_unary<LogBackward>(m, &log_op);
    bind_unary<Log2Backward>(m, &log2_op);
    bind_unary<SqrtBackward>(m, &sqrt_op);

    // Trigonometric ops.
    bind_unary<SinBackward>(m, &sin_op);
    bind_unary<CosBackward>(m, &cos_op);
    bind_unary<TanBackward>(m, &tan_op);
    bind_unary<AsinBackward>(m, &arcsin_op);
    bind_unary<AcosBackward>(m, &arccos_op);
    bind_unary<AtanBackward>(m, &arctan_op);

    // Hyperbolic ops.
    bind_unary<SinhBackward>(m, &sinh_op);
    bind_unary<CoshBackward>(m, &cosh_op);
    bind_unary<TanhBackward>(m, &tanh_op);

    // Activation functions.  leaky_relu and elu use bind_unary_extra because
    // they accept an additional scalar parameter beyond the input tensor.
    bind_unary<ReluBackward>(m, &relu_op);
    bind_unary<SigmoidBackward>(m, &sigmoid_op);
    bind_unary<SiluBackward>(m, &silu_op);
    bind_unary<GeluBackward>(m, &gelu_op);
    bind_unary<GeluExactBackward>(m, &gelu_exact_op);
    bind_unary_extra<LeakyReluBackward>(m, &leaky_relu_op, py::arg("slope") = 0.01);
    bind_unary<SoftplusBackward>(m, &softplus_op);
    bind_unary_extra<EluBackward>(m, &elu_op, py::arg("alpha") = 1.0);
    bind_unary<SeluBackward>(m, &selu_op);
    bind_unary<MishBackward>(m, &mish_op);
    bind_unary<HardSigmoidBackward>(m, &hard_sigmoid_op);
    bind_unary<HardSwishBackward>(m, &hard_swish_op);
    bind_unary<Relu6Backward>(m, &relu6_op);

    // softmax is registered manually because its extra `dim` argument is
    // not captured by the plain bind_unary<> signature.
    m.def("softmax", &softmax_op, py::arg("a"), py::arg("dim") = -1);
    m.def("log_softmax", &log_softmax_op, py::arg("a"), py::arg("dim") = -1);

    // Scalar-parameter ops: base ** exp, base ** a, and clamp.
    m.def("pow_scalar", &pow_scalar_op, py::arg("a"), py::arg("exp"));
    m.def("rpow_scalar", &rpow_scalar_op, py::arg("base"), py::arg("a"));
    m.def("clip", &clip_op, py::arg("a"), py::arg("min"), py::arg("max"));

    // Discrete / rounding ops.
    bind_unary<RoundBackward>(m, &round_op);
    bind_unary<FloorBackward>(m, &floor_op);
    bind_unary<CeilBackward>(m, &ceil_op);
    bind_unary<InvertBackward>(m, &invert_op);
    bind_unary<RsqrtBackward>(m, &rsqrt_op);

    // Reduction ops share the (a, dim=[], keepdims=False) signature; empty
    // `dim` means reduce over all dimensions.
    bind_reduce<SumBackward>(m, &sum_op, "Reduce-sum along given dim (empty = all).");
    bind_reduce<MeanBackward>(m, &mean_op, "Reduce-mean.");
    bind_reduce<ProdBackward>(m, &prod_op, "Reduce-product.");
    bind_extremum(m);

    // Dim manipulation ops live here rather than in bind_utils because their
    // backward nodes are in the ufunc layer.
    m.def("permute", &permute_op, py::arg("a"), py::arg("dims"), "General dim permutation.");
    m.def("transpose", &transpose_op, py::arg("a"), "Reverse all dims (alias for `T`).");
    m.def("T", &T_op, py::arg("a"), "Reverse all dims.");
    m.def("mT", &mT_op, py::arg("a"), "Swap last two dims (matrix transpose).");
    m.def("swapaxes", &swapaxes_op, py::arg("a"), py::arg("dim1"), py::arg("dim2"),
          "Swap two dims.");

    m.def("var", &var_op, py::arg("a"), py::arg("dim") = std::vector<int>{},
          py::arg("keepdims") = false, "Variance reduction (sample variance, ddof=0).");
    m.def("std", &std_op, py::arg("a"), py::arg("dim") = std::vector<int>{},
          py::arg("keepdims") = false, "Standard deviation (sqrt of variance).");
    m.def("trace", &trace_op, py::arg("a"), "Sum of the main diagonal (last 2 dims).");
    m.def("cumsum", &cumsum_op, py::arg("a"), py::arg("dim") = -1);
    m.def("cumprod", &cumprod_op, py::arg("a"), py::arg("dim") = -1);
    m.def("cummax", &cummax_op, py::arg("a"), py::arg("dim") = -1);
    m.def("cummin", &cummin_op, py::arg("a"), py::arg("dim") = -1);
    bind_unary<ErfBackward>(m, &erf_op);
    bind_unary<ErfinvBackward>(m, &erfinv_op);

    // In-place variants.  These mutate `a` directly, bump its version counter,
    // and are not differentiable.  The underscore suffix follows reference convention.
    m.def("neg_", &neg_inplace_op, py::arg("a"));
    m.def("abs_", &abs_inplace_op, py::arg("a"));
    m.def("sign_", &sign_inplace_op, py::arg("a"));
    m.def("reciprocal_", &reciprocal_inplace_op, py::arg("a"));
    m.def("square_", &square_inplace_op, py::arg("a"));
    m.def("cube_", &cube_inplace_op, py::arg("a"));
    m.def("exp_", &exp_inplace_op, py::arg("a"));
    m.def("log_", &log_inplace_op, py::arg("a"));
    m.def("log2_", &log2_inplace_op, py::arg("a"));
    m.def("sqrt_", &sqrt_inplace_op, py::arg("a"));
    m.def("sin_", &sin_inplace_op, py::arg("a"));
    m.def("cos_", &cos_inplace_op, py::arg("a"));
    m.def("tan_", &tan_inplace_op, py::arg("a"));
    m.def("arcsin_", &arcsin_inplace_op, py::arg("a"));
    m.def("arccos_", &arccos_inplace_op, py::arg("a"));
    m.def("arctan_", &arctan_inplace_op, py::arg("a"));
    m.def("sinh_", &sinh_inplace_op, py::arg("a"));
    m.def("cosh_", &cosh_inplace_op, py::arg("a"));
    m.def("tanh_", &tanh_inplace_op, py::arg("a"));
    m.def("sigmoid_", &sigmoid_inplace_op, py::arg("a"));
    m.def("relu_", &relu_inplace_op, py::arg("a"));
    m.def("round_", &round_inplace_op, py::arg("a"));
    m.def("floor_", &floor_inplace_op, py::arg("a"));
    m.def("ceil_", &ceil_inplace_op, py::arg("a"));
    m.def("clip_", &clip_inplace_op, py::arg("a"), py::arg("min"), py::arg("max"));
    m.def("assign_inplace", &assign_inplace_op, py::arg("a"), py::arg("value"), py::arg("name"),
          "Write ``value`` into ``a``'s buffer in place, for a tensor with live views.");

    // Floating-point predicate ops (output is always bool).
    m.def("any", &any_op, py::arg("a"));
    m.def("all", &all_op, py::arg("a"));
    m.def("isinf", &isinf_op, py::arg("a"));
    m.def("isnan", &isnan_op, py::arg("a"));
    m.def("isfinite", &isfinite_op, py::arg("a"));
    // None picks the reference framework's default for the input's dtype:
    // 0 for NaN, the dtype's largest / lowest finite value for the
    // infinities (resolved in nan_to_num_op).
    m.def("nan_to_num", &nan_to_num_op, py::arg("a"), py::arg("nan") = py::none(),
          py::arg("posinf") = py::none(), py::arg("neginf") = py::none());

    // Dtype cast: element-wise conversion to a different dtype.
    // CPU: static_cast loop.  GPU: mlx::core::astype.
    m.def("astype", &astype_op, py::arg("a"), py::arg("dtype"),
          "Cast all elements to dtype. CPU: static_cast loop. GPU: mlx::core::astype.");

    // Device move, differentiable: the gradient goes back to the source
    // device, and a Metal shared buffer is relabelled rather than copied.
    m.def("to_device", &to_device_op, py::arg("a"), py::arg("device"),
          "Move to device; differentiable (the gradient returns to the source device).");
}

}  // namespace lucid::bindings
