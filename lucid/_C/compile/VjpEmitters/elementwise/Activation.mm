// lucid/_C/compile/VjpEmitters/elementwise/Activation.mm
//
// Activation VJPs.  All single-input ops — straight elementwise
// gradient formulas, no broadcast unreduce needed.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// All activation VJPs use the unary skeleton from _VjpHelpers.h:
//   emit_unary_vjp(bctx, node, grad_outs, body) → bool
// where ``body(graph, x, grad_out) → MPSGraphTensor*`` returns dx.

// ────────────────────────────────────────────────────────────────────
// ReLU: dA = grad * (x > 0).
//
// All constants / casts derive from ``go.dataType`` (the chain dtype) —
// ``x`` is pre-cast to that dtype by ``emit_unary_vjp``.
// ────────────────────────────────────────────────────────────────────
class ReluVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "relu"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = go.dataType;
                MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dt];
                MPSGraphTensor* mask =
                    [g greaterThanWithPrimaryTensor:x secondaryTensor:zero name:nil];
                MPSGraphTensor* mask_f = [g castTensor:mask toType:dt name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:mask_f
                                                     name:@"relu_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// Sigmoid: dA = grad * y * (1 - y) where y = sigmoid(x).
// We recompute y from x; same cost as saving it but avoids a cache
// slot on the bctx side.
// ────────────────────────────────────────────────────────────────────
class SigmoidVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sigmoid"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = go.dataType;
                MPSGraphTensor* y = [g sigmoidWithTensor:x name:nil];
                MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* one_m_y =
                    [g subtractionWithPrimaryTensor:one secondaryTensor:y name:nil];
                MPSGraphTensor* y_omy =
                    [g multiplicationWithPrimaryTensor:y secondaryTensor:one_m_y name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:y_omy
                                                     name:@"sigmoid_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// Tanh: dA = grad * (1 - y²) where y = tanh(x).
// ────────────────────────────────────────────────────────────────────
class TanhVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "tanh"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = go.dataType;
                MPSGraphTensor* y = [g tanhWithTensor:x name:nil];
                MPSGraphTensor* y_sq =
                    [g multiplicationWithPrimaryTensor:y secondaryTensor:y name:nil];
                MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* one_m_y_sq =
                    [g subtractionWithPrimaryTensor:one secondaryTensor:y_sq name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:one_m_y_sq
                                                     name:@"tanh_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// SiLU: y = x * sigmoid(x).  dy/dx = sigmoid(x) * (1 + x*(1-sigmoid(x))).
// ────────────────────────────────────────────────────────────────────
class SiluVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "silu"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = go.dataType;
                MPSGraphTensor* sig = [g sigmoidWithTensor:x name:nil];
                MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* one_m_sig =
                    [g subtractionWithPrimaryTensor:one secondaryTensor:sig name:nil];
                MPSGraphTensor* x_omsig =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:one_m_sig name:nil];
                MPSGraphTensor* one_plus =
                    [g additionWithPrimaryTensor:one secondaryTensor:x_omsig name:nil];
                MPSGraphTensor* deriv =
                    [g multiplicationWithPrimaryTensor:sig secondaryTensor:one_plus name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:deriv
                                                     name:@"silu_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// GELU (tanh-approx):
//   y = 0.5 * x * (1 + tanh( k * (x + 0.044715 * x^3) ))   k = sqrt(2/π).
// Derivative (Hendrycks-Gimpel form):
//   let u = k*(x + 0.044715*x^3),  t = tanh(u),  s = sech²(u) = 1 - t².
//   dy/dx = 0.5 * (1 + t) + 0.5 * x * s * k * (1 + 3 * 0.044715 * x²)
//          = 0.5 * (1 + t) + 0.5 * x * (1 - t²) * k * (1 + 0.134145 * x²).
// ────────────────────────────────────────────────────────────────────
class GeluVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "gelu"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = go.dataType;
                MPSGraphTensor* k =
                    [g constantWithScalar:0.7978845608028654 dataType:dt];  // sqrt(2/π)
                MPSGraphTensor* c044 =
                    [g constantWithScalar:0.044715 dataType:dt];
                MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:dt];
                MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* c134 =
                    [g constantWithScalar:0.134145 dataType:dt];  // 3 * 0.044715
                // x²
                MPSGraphTensor* x2 =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
                // x³
                MPSGraphTensor* x3 =
                    [g multiplicationWithPrimaryTensor:x2 secondaryTensor:x name:nil];
                // 0.044715 * x³
                MPSGraphTensor* cx3 =
                    [g multiplicationWithPrimaryTensor:c044 secondaryTensor:x3 name:nil];
                // x + 0.044715 * x³
                MPSGraphTensor* inner_add =
                    [g additionWithPrimaryTensor:x secondaryTensor:cx3 name:nil];
                // u = k * (x + 0.044715 * x³)
                MPSGraphTensor* u =
                    [g multiplicationWithPrimaryTensor:k secondaryTensor:inner_add name:nil];
                MPSGraphTensor* t = [g tanhWithTensor:u name:nil];
                MPSGraphTensor* t2 =
                    [g multiplicationWithPrimaryTensor:t secondaryTensor:t name:nil];
                MPSGraphTensor* sech2 =
                    [g subtractionWithPrimaryTensor:one secondaryTensor:t2 name:nil];
                // first half: 0.5 * (1 + t)
                MPSGraphTensor* one_plus_t =
                    [g additionWithPrimaryTensor:one secondaryTensor:t name:nil];
                MPSGraphTensor* term1 =
                    [g multiplicationWithPrimaryTensor:half secondaryTensor:one_plus_t name:nil];
                // second half: 0.5 * x * (1-t²) * k * (1 + 0.134145 * x²)
                MPSGraphTensor* c134_x2 =
                    [g multiplicationWithPrimaryTensor:c134 secondaryTensor:x2 name:nil];
                MPSGraphTensor* one_plus_c134x2 =
                    [g additionWithPrimaryTensor:one secondaryTensor:c134_x2 name:nil];
                MPSGraphTensor* k_term =
                    [g multiplicationWithPrimaryTensor:k secondaryTensor:one_plus_c134x2 name:nil];
                MPSGraphTensor* x_k_term =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:k_term name:nil];
                MPSGraphTensor* x_k_sech2 =
                    [g multiplicationWithPrimaryTensor:x_k_term secondaryTensor:sech2 name:nil];
                MPSGraphTensor* term2 =
                    [g multiplicationWithPrimaryTensor:half secondaryTensor:x_k_sech2 name:nil];
                // sum + multiply by grad
                MPSGraphTensor* deriv =
                    [g additionWithPrimaryTensor:term1 secondaryTensor:term2 name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:deriv
                                                     name:@"gelu_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// GELU (exact, erf-based):
//   y = 0.5 * x * (1 + erf(x / sqrt(2))).
// Derivative (chain rule):
//   dy/dx = 0.5 * (1 + erf(x/√2))                 ≡ Φ(x)
//         + x * exp(-x²/2) / sqrt(2π)              ≡ x · φ(x)
// where Φ is the standard-normal CDF and φ the PDF.  We do NOT divide
// y by x to recover Φ(x) — the x==0 singularity would blow up; recompute
// erf cheaply instead.  Constants resolve from ``go.dataType`` so the
// chain stays one dtype under autocast (see _VjpHelpers.h:207-230).
// ────────────────────────────────────────────────────────────────────
class GeluExactVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "gelu_exact"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = go.dataType;
                MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:dt];
                MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* neg_half =
                    [g constantWithScalar:-0.5 dataType:dt];
                MPSGraphTensor* inv_sqrt2 =
                    [g constantWithScalar:0.7071067811865475 dataType:dt];  // 1/√2
                MPSGraphTensor* inv_sqrt2pi =
                    [g constantWithScalar:0.3989422804014327 dataType:dt];  // 1/√(2π)
                // term1 = Φ(x) = 0.5 * (1 + erf(x/√2))
                MPSGraphTensor* x_scaled =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:inv_sqrt2 name:nil];
                MPSGraphTensor* erf_t = [g erfWithTensor:x_scaled name:nil];
                MPSGraphTensor* one_plus_erf =
                    [g additionWithPrimaryTensor:one secondaryTensor:erf_t name:nil];
                MPSGraphTensor* term1 =
                    [g multiplicationWithPrimaryTensor:half secondaryTensor:one_plus_erf name:nil];
                // term2 = x * φ(x) = x * exp(-x²/2) / √(2π)
                MPSGraphTensor* x2 =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
                MPSGraphTensor* neg_half_x2 =
                    [g multiplicationWithPrimaryTensor:neg_half secondaryTensor:x2 name:nil];
                MPSGraphTensor* exp_t = [g exponentWithTensor:neg_half_x2 name:nil];
                MPSGraphTensor* pdf =
                    [g multiplicationWithPrimaryTensor:inv_sqrt2pi secondaryTensor:exp_t name:nil];
                MPSGraphTensor* term2 =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:pdf name:nil];
                MPSGraphTensor* deriv =
                    [g additionWithPrimaryTensor:term1 secondaryTensor:term2 name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:deriv
                                                     name:@"gelu_exact_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// astype (dtype cast): dx = cast(grad_out, x.dtype).  The forward
// changes precision; the backward casts the grad back to the input's
// precision so the chain stays self-consistent.  Identity-shape so
// no broadcast unreduce is needed.
// ────────────────────────────────────────────────────────────────────
class AstypeVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "astype"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        // ``cast_forward_to_grad=false`` — this VJP intentionally reads
        // the original ``x`` dtype to cast the grad *back* to it.
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                return cast_mps_tensor(g, go, x.dataType, @"astype_vjp");
            },
            /*cast_forward_to_grad=*/false);
    }
};

// ────────────────────────────────────────────────────────────────────
// relu6 / hard_sigmoid / hard_swish: piecewise-linear, with eager's
// conventions at the kinks —
//   relu6'        = 1 on the open (0, 6), else 0;
//   hard_sigmoid' = 1/6 on the open (-3, 3), else 0;
//   hard_swish'   = 0 below -3, 1 above 3, (2x + 3)/6 between.
// MobileNets train through these.  With the batch norm in training mode
// ruling out MPSGraph's autodiff, a missing VJP here sent their compiled
// training step eager.
// ────────────────────────────────────────────────────────────────────
enum class HardAct { Relu6, HardSigmoid, HardSwish };

template <HardAct KIND>
class HardActivationVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override {
        return KIND == HardAct::Relu6 ? "relu6"
               : KIND == HardAct::HardSigmoid ? "hard_sigmoid"
                                               : "hard_swish";
    }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                const MPSDataType dt = go.dataType;
                x = [g castTensor:x toType:dt name:nil];
                auto c = [&](double v) { return [g constantWithScalar:v dataType:dt]; };
                auto as_f = [&](MPSGraphTensor* b) { return [g castTensor:b toType:dt name:nil]; };
                MPSGraphTensor* slope = nil;
                if (KIND == HardAct::Relu6 || KIND == HardAct::HardSigmoid) {
                    const double lo = KIND == HardAct::Relu6 ? 0.0 : -3.0;
                    const double hi = KIND == HardAct::Relu6 ? 6.0 : 3.0;
                    MPSGraphTensor* inside = [g logicalANDWithPrimaryTensor:
                                                    [g greaterThanWithPrimaryTensor:x
                                                                    secondaryTensor:c(lo)
                                                                               name:nil]
                                                            secondaryTensor:
                                                    [g lessThanWithPrimaryTensor:x
                                                                 secondaryTensor:c(hi)
                                                                            name:nil]
                                                                       name:nil];
                    slope = as_f(inside);
                    if (KIND == HardAct::HardSigmoid)
                        slope = [g multiplicationWithPrimaryTensor:slope
                                                   secondaryTensor:c(1.0 / 6.0)
                                                              name:nil];
                } else {
                    // (2x + 3)/6 inside; 0 below -3; 1 above 3.
                    MPSGraphTensor* ramp = [g multiplicationWithPrimaryTensor:
                                                  [g additionWithPrimaryTensor:
                                                         [g multiplicationWithPrimaryTensor:x
                                                                            secondaryTensor:c(2.0)
                                                                                       name:nil]
                                                               secondaryTensor:c(3.0)
                                                                          name:nil]
                                                              secondaryTensor:c(1.0 / 6.0)
                                                                         name:nil];
                    MPSGraphTensor* not_below =
                        as_f([g greaterThanOrEqualToWithPrimaryTensor:x secondaryTensor:c(-3.0) name:nil]);
                    MPSGraphTensor* above =
                        as_f([g greaterThanWithPrimaryTensor:x secondaryTensor:c(3.0) name:nil]);
                    // not_below * (above + (1 - above) * ramp)
                    MPSGraphTensor* mid = [g multiplicationWithPrimaryTensor:
                                                 [g subtractionWithPrimaryTensor:c(1.0)
                                                                 secondaryTensor:above
                                                                            name:nil]
                                                             secondaryTensor:ramp
                                                                        name:nil];
                    slope = [g multiplicationWithPrimaryTensor:not_below
                                               secondaryTensor:[g additionWithPrimaryTensor:above
                                                                            secondaryTensor:mid
                                                                                       name:nil]
                                                          name:nil];
                }
                return [g multiplicationWithPrimaryTensor:go secondaryTensor:slope name:nil];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// mish(x) = x · tanh(softplus(x)):
//   mish' = tanh(sp) + x · (1 − tanh(sp)²) · sigmoid(x)   — eager's form.
// ────────────────────────────────────────────────────────────────────
class MishVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "mish"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:go.dataType];
                // Softplus as the forward builds it.
                MPSGraphTensor* sp = [g logarithmWithTensor:[g additionWithPrimaryTensor:one
                                                                          secondaryTensor:[g exponentWithTensor:x name:nil]
                                                                                     name:nil]
                                                        name:nil];
                MPSGraphTensor* t = [g tanhWithTensor:sp name:nil];
                MPSGraphTensor* sech2 = [g subtractionWithPrimaryTensor:one
                                                        secondaryTensor:[g squareWithTensor:t name:nil]
                                                                   name:nil];
                MPSGraphTensor* second = [g multiplicationWithPrimaryTensor:[g multiplicationWithPrimaryTensor:x
                                                                                             secondaryTensor:sech2
                                                                                                        name:nil]
                                                            secondaryTensor:[g sigmoidWithTensor:x name:nil]
                                                                       name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:[g additionWithPrimaryTensor:t
                                                                       secondaryTensor:second
                                                                                  name:nil]
                                                     name:@"mish_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// Piecewise activations.  Each forward is a select on ``x >= 0``, and
// MPSGraph's autodiff aborts on a select whose predicate is a comparison
// ("Couldn't get gradient Tensor"), so without these a model using any of
// them could not be trained compiled.  The backward only needs the same
// comparison as a mask, which is harmless here.  The attribute defaults
// match the forward emitters.
// ────────────────────────────────────────────────────────────────────
inline double vjp_double_attr(const OpNode& node, const char* key, double def) {
    auto it = node.attrs.find(key);
    if (it == node.attrs.end())
        return def;
    const auto* p = std::get_if<double>(&it->second);
    return p ? *p : def;
}

// select(x >= 0, pos, neg) as the local derivative, times the incoming grad.
inline MPSGraphTensor* piecewise_grad(MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go,
                                      MPSGraphTensor* pos, MPSGraphTensor* neg) {
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:go.dataType];
    MPSGraphTensor* mask = [g greaterThanOrEqualToWithPrimaryTensor:x secondaryTensor:zero name:nil];
    MPSGraphTensor* d = [g selectWithPredicateTensor:mask
                                 truePredicateTensor:pos
                                falsePredicateTensor:neg
                                                name:nil];
    return [g multiplicationWithPrimaryTensor:go secondaryTensor:d name:nil];
}

class LeakyReluVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "leaky_relu"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        const double slope = vjp_double_attr(node, "slope", 0.01);
        return emit_unary_vjp(bctx, node, grad_outs,
            [slope](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = go.dataType;
                return piecewise_grad(g, x, go, [g constantWithScalar:1.0 dataType:dt],
                                      [g constantWithScalar:slope dataType:dt]);
            });
    }
};

// elu: d/dx = 1 for x >= 0, alpha * exp(x) below.  selu is lambda times the
// same with its fixed alpha.
template <bool SELU>
class EluFamilyVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return SELU ? "selu" : "elu"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        const double alpha = SELU ? 1.6732632423543772 : vjp_double_attr(node, "alpha", 1.0);
        const double scale = SELU ? 1.0507009873554805 : 1.0;
        return emit_unary_vjp(bctx, node, grad_outs,
            [alpha, scale](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = go.dataType;
                MPSGraphTensor* neg = [g multiplicationWithPrimaryTensor:
                                              [g constantWithScalar:alpha * scale dataType:dt]
                                                         secondaryTensor:[g exponentWithTensor:x
                                                                                          name:nil]
                                                                    name:nil];
                return piecewise_grad(g, x, go, [g constantWithScalar:scale dataType:dt], neg);
            });
    }
};

// softplus = log(1 + e^x): d/dx = sigmoid(x).
class SoftplusVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "softplus"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:[g sigmoidWithTensor:x name:nil]
                                                     name:@"softplus_vjp"];
            });
    }
};

struct ActivationVjpRegistrar {
    ActivationVjpRegistrar() {
        register_vjp_emitter(std::make_unique<ReluVjp>());
        register_vjp_emitter(std::make_unique<LeakyReluVjp>());
        register_vjp_emitter(std::make_unique<EluFamilyVjp<false>>());
        register_vjp_emitter(std::make_unique<EluFamilyVjp<true>>());
        register_vjp_emitter(std::make_unique<SoftplusVjp>());
        register_vjp_emitter(std::make_unique<MishVjp>());
        register_vjp_emitter(std::make_unique<HardActivationVjp<HardAct::Relu6>>());
        register_vjp_emitter(std::make_unique<HardActivationVjp<HardAct::HardSigmoid>>());
        register_vjp_emitter(std::make_unique<HardActivationVjp<HardAct::HardSwish>>());
        register_vjp_emitter(std::make_unique<SigmoidVjp>());
        register_vjp_emitter(std::make_unique<TanhVjp>());
        register_vjp_emitter(std::make_unique<SiluVjp>());
        register_vjp_emitter(std::make_unique<GeluVjp>());
        register_vjp_emitter(std::make_unique<GeluExactVjp>());
        register_vjp_emitter(std::make_unique<AstypeVjp>());
    }
};

[[maybe_unused]] static const ActivationVjpRegistrar g_activation_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
