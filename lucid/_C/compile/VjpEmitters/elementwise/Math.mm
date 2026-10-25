// lucid/_C/compile/VjpEmitters/elementwise/Math.mm
//
// VJPs for the unary math ops (log / exp / sqrt / abs / square /
// reciprocal / sin / cos).  All single-input, all closed-form
// gradients with no broadcast unreduce.
//
// Forward emitters live in :file:`OpEmitters/elementwise/Math.mm`,
// :file:`OpEmitters/elementwise/Trig.mm`, etc.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// d(log(x))/dx = 1/x
class LogVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "log"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                return [g divisionWithPrimaryTensor:go
                                     secondaryTensor:x
                                                name:@"log_vjp"];
            });
    }
};

// d(exp(x))/dx = exp(x).  Recomputed (fast) rather than saved.
class ExpVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "exp"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* y = [g exponentWithTensor:x name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:y
                                                     name:@"exp_vjp"];
            });
    }
};

// d(sqrt(x))/dx = 1 / (2 sqrt(x))
class SqrtVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sqrt"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* y = [g squareRootWithTensor:x name:nil];
                MPSGraphTensor* two = [g constantWithScalar:2.0
                                                    dataType:x.dataType];
                MPSGraphTensor* two_y =
                    [g multiplicationWithPrimaryTensor:two
                                       secondaryTensor:y name:nil];
                return [g divisionWithPrimaryTensor:go
                                     secondaryTensor:two_y
                                                name:@"sqrt_vjp"];
            });
    }
};

// d(rsqrt(x))/dx = -1 / (2 * x^(3/2)) = -0.5 * rsqrt(x)^3
class RsqrtVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "rsqrt"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* r = [g reciprocalSquareRootWithTensor:x name:nil];
                MPSGraphTensor* r3a =
                    [g multiplicationWithPrimaryTensor:r secondaryTensor:r name:nil];
                MPSGraphTensor* r3 =
                    [g multiplicationWithPrimaryTensor:r3a secondaryTensor:r name:nil];
                MPSGraphTensor* nhalf = [g constantWithScalar:-0.5 dataType:x.dataType];
                MPSGraphTensor* d =
                    [g multiplicationWithPrimaryTensor:nhalf secondaryTensor:r3 name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:d
                                                     name:@"rsqrt_vjp"];
            });
    }
};

// d(square(x))/dx = 2x
class SquareVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "square"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* two = [g constantWithScalar:2.0 dataType:x.dataType];
                MPSGraphTensor* two_x =
                    [g multiplicationWithPrimaryTensor:two secondaryTensor:x name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:two_x
                                                     name:@"square_vjp"];
            });
    }
};

// d(abs(x))/dx = sign(x)
class AbsVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "abs"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* s = [g signWithTensor:x name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:s
                                                     name:@"abs_vjp"];
            });
    }
};

// d(reciprocal(x))/dx = -1/x²
class ReciprocalVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "reciprocal"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* x_sq =
                    [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
                MPSGraphTensor* neg_go = [g negativeWithTensor:go name:nil];
                return [g divisionWithPrimaryTensor:neg_go
                                     secondaryTensor:x_sq
                                                name:@"reciprocal_vjp"];
            });
    }
};

// d(sin(x))/dx = cos(x); d(cos(x))/dx = -sin(x)
class SinVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sin"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* c = [g cosWithTensor:x name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:c
                                                     name:@"sin_vjp"];
            });
    }
};

class CosVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "cos"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* s = [g sinWithTensor:x name:nil];
                MPSGraphTensor* neg_s = [g negativeWithTensor:s name:nil];
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:neg_s
                                                     name:@"cos_vjp"];
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// Trig / special.  Without a manual VJP these fell to MPSGraph autodiff,
// which returned a wrong gradient for arcsin and aborted on erfinv.
// ────────────────────────────────────────────────────────────────────
template <class F>
class UnaryDerivVjp final : public VjpEmitter {
public:
    UnaryDerivVjp(const char* name, F deriv) : name_(name), deriv_(deriv) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [this](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                return [g multiplicationWithPrimaryTensor:go
                                          secondaryTensor:deriv_(g, x, go.dataType)
                                                     name:nil];
            });
    }

private:
    const char* name_;
    F deriv_;
};

template <class F>
std::unique_ptr<VjpEmitter> unary_deriv(const char* name, F f) {
    return std::make_unique<UnaryDerivVjp<F>>(name, f);
}

inline MPSGraphTensor* cst(MPSGraph* g, double v, MPSDataType dt) {
    return [g constantWithScalar:v dataType:dt];
}

// 1 / sqrt(1 - x²)
inline MPSGraphTensor* inv_sqrt_one_minus_sq(MPSGraph* g, MPSGraphTensor* x, MPSDataType dt) {
    MPSGraphTensor* r = [g subtractionWithPrimaryTensor:cst(g, 1.0, dt)
                                        secondaryTensor:[g squareWithTensor:x name:nil]
                                                   name:nil];
    return [g reciprocalWithTensor:[g squareRootWithTensor:r name:nil] name:nil];
}

// erfinv'(x) = sqrt(pi)/2 · exp(erfinv(x)²).  It needs the forward value
// y = erfinv(x), not x; recomputing it would repeat the forward polynomial,
// so it is read off the context instead.
class ErfinvVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "erfinv"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 1 || node.outputs.empty() || grad_outs.empty() ||
            grad_outs[0] == nullptr || node.inputs[0] < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        MPSGraphTensor* y = as_tensor(bctx.forward(node.outputs[0].id));
        if (g == nil || go == nil || y == nil)
            return false;
        y = cast_if_needed(g, y, go.dataType);
        MPSGraphTensor* d = [g multiplicationWithPrimaryTensor:cst(g, 0.886226925452758, go.dataType)
                                               secondaryTensor:[g exponentWithTensor:
                                                                      [g squareWithTensor:y name:nil]
                                                                                name:nil]
                                                          name:nil];
        bctx.accumulate_grad(node.inputs[0],
                             from_tensor([g multiplicationWithPrimaryTensor:go
                                                            secondaryTensor:d
                                                                       name:@"erfinv_vjp"]));
        return true;
    }
};

// floor / ceil / round / trunc / sign: piecewise constant, so the gradient
// is zero wherever it exists — and eager returns exactly that, zeros shaped
// like the input.  Not ``go * 0``: an infinite ``go`` (erfinv at ±1 just
// downstream) would make it NaN where eager has 0.  These used to be grad
// sinks in the walk, which left a parameter reached only through them with
// no gradient at all while eager handed it zeros.
class PiecewiseConstantVjp final : public VjpEmitter {
public:
    explicit PiecewiseConstantVjp(const char* name) : name_(name) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                return [g broadcastTensor:[g constantWithScalar:0.0 dataType:go.dataType]
                            toShapeTensor:[g shapeOfTensor:x name:nil]
                                     name:@"piecewise_constant_vjp"];
            });
    }

private:
    const char* name_;
};

// nan_to_num: dx = grad where x is finite, 0 where it was replaced — the
// replacements are constants.  Mirrors ``NanToNumBackward``.
class NanToNumVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "nan_to_num"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return emit_unary_vjp(bctx, node, grad_outs,
            [](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                return [g selectWithPredicateTensor:[g isFiniteWithTensor:x name:nil]
                                truePredicateTensor:go
                               falsePredicateTensor:cst(g, 0.0, go.dataType)
                                               name:@"nan_to_num_vjp"];
            });
    }
};

// clip / clamp: the gradient passes where the input was inside the bounds
// (inclusive) and stops where it was clipped.  Used by both BCE losses.
class ClipVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "clip"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        auto bound = [&](const char* key) -> const double* {
            auto it = node.attrs.find(key);
            return it == node.attrs.end() ? nullptr : std::get_if<double>(&it->second);
        };
        const double* lo = bound("min");
        const double* hi = bound("max");
        return emit_unary_vjp(bctx, node, grad_outs,
            [lo, hi](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSGraphTensor* keep = nil;
                if (lo != nullptr)
                    keep = [g greaterThanOrEqualToWithPrimaryTensor:x
                                                    secondaryTensor:cst(g, *lo, go.dataType)
                                                               name:nil];
                if (hi != nullptr) {
                    MPSGraphTensor* under = [g lessThanOrEqualToWithPrimaryTensor:x
                                                                  secondaryTensor:cst(g, *hi, go.dataType)
                                                                             name:nil];
                    keep = keep == nil ? under
                                       : [g logicalANDWithPrimaryTensor:keep secondaryTensor:under name:nil];
                }
                if (keep == nil)
                    return go;
                return [g selectWithPredicateTensor:keep
                                truePredicateTensor:go
                               falsePredicateTensor:cst(g, 0.0, go.dataType)
                                               name:@"clip_vjp"];
            });
    }
};

// pow_scalar: x^p → p·x^(p-1).  rpow_scalar: b^x → b^x·ln b.  Neither had a
// VJP; ``x ** 2`` inside local response normalisation kept ZFNet from
// training compiled.
template <bool REVERSE>
class PowScalarVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return REVERSE ? "rpow_scalar" : "pow_scalar"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        auto it = node.attrs.find(REVERSE ? "base" : "exp");
        if (it == node.attrs.end())
            return false;
        const auto* c = std::get_if<double>(&it->second);
        if (c == nullptr)
            return false;
        const double k = *c;
        return emit_unary_vjp(bctx, node, grad_outs,
            [k](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) {
                MPSDataType dt = go.dataType;
                MPSGraphTensor* zero = cst(g, 0.0, dt);
                if (!REVERSE && k == 0.0) {
                    return [g broadcastTensor:zero
                                toShapeTensor:[g shapeOfTensor:go name:nil]
                                         name:@"pow_scalar_zero_vjp"];
                }
                MPSGraphTensor* d =
                    REVERSE
                        ? [g multiplicationWithPrimaryTensor:[g powerWithPrimaryTensor:cst(g, k, dt)
                                                                         secondaryTensor:x
                                                                                    name:nil]
                                              secondaryTensor:cst(g, std::log(k), dt)
                                                         name:nil]
                        : [g multiplicationWithPrimaryTensor:cst(g, k, dt)
                                              secondaryTensor:[g powerWithPrimaryTensor:x
                                                                         secondaryTensor:cst(g, k - 1.0, dt)
                                                                                    name:nil]
                                                         name:nil];
                if (REVERSE && k == 0.0) {
                    MPSGraphTensor* ge_zero =
                        [g greaterThanOrEqualToWithPrimaryTensor:x
                                                 secondaryTensor:cst(g, 0.0, x.dataType)
                                                            name:nil];
                    d = [g selectWithPredicateTensor:ge_zero
                                 truePredicateTensor:zero
                                falsePredicateTensor:d
                                                name:nil];
                }
                return [g multiplicationWithPrimaryTensor:go secondaryTensor:d name:nil];
            });
    }
};

struct MathVjpRegistrar {
    MathVjpRegistrar() {
        register_vjp_emitter(std::make_unique<LogVjp>());
        register_vjp_emitter(std::make_unique<ErfinvVjp>());
        register_vjp_emitter(std::make_unique<ClipVjp>());
        register_vjp_emitter(std::make_unique<NanToNumVjp>());
        for (const char* name : {"floor", "ceil", "round", "trunc", "sign"})
            register_vjp_emitter(std::make_unique<PiecewiseConstantVjp>(name));
        register_vjp_emitter(std::make_unique<PowScalarVjp<false>>());
        register_vjp_emitter(std::make_unique<PowScalarVjp<true>>());
        register_vjp_emitter(unary_deriv("arcsin", [](MPSGraph* g, MPSGraphTensor* x, MPSDataType dt) {
            return inv_sqrt_one_minus_sq(g, x, dt);
        }));
        register_vjp_emitter(unary_deriv("arccos", [](MPSGraph* g, MPSGraphTensor* x, MPSDataType dt) {
            return [g negativeWithTensor:inv_sqrt_one_minus_sq(g, x, dt) name:nil];
        }));
        register_vjp_emitter(unary_deriv("arctan", [](MPSGraph* g, MPSGraphTensor* x, MPSDataType dt) {
            return [g reciprocalWithTensor:[g additionWithPrimaryTensor:cst(g, 1.0, dt)
                                                        secondaryTensor:[g squareWithTensor:x name:nil]
                                                                   name:nil]
                                      name:nil];
        }));
        register_vjp_emitter(unary_deriv("tan", [](MPSGraph* g, MPSGraphTensor* x, MPSDataType dt) {
            MPSGraphTensor* c = [g cosWithTensor:x name:nil];
            return [g reciprocalWithTensor:[g squareWithTensor:c name:nil] name:nil];
        }));
        register_vjp_emitter(unary_deriv("sinh", [](MPSGraph* g, MPSGraphTensor* x, MPSDataType) {
            return [g coshWithTensor:x name:nil];
        }));
        register_vjp_emitter(unary_deriv("cosh", [](MPSGraph* g, MPSGraphTensor* x, MPSDataType) {
            return [g sinhWithTensor:x name:nil];
        }));
        register_vjp_emitter(unary_deriv("erf", [](MPSGraph* g, MPSGraphTensor* x, MPSDataType dt) {
            MPSGraphTensor* e = [g exponentWithTensor:[g negativeWithTensor:[g squareWithTensor:x name:nil]
                                                                       name:nil]
                                                 name:nil];
            return [g multiplicationWithPrimaryTensor:cst(g, 1.1283791670955126, dt)
                                      secondaryTensor:e
                                                 name:nil];
        }));
        register_vjp_emitter(unary_deriv("log2", [](MPSGraph* g, MPSGraphTensor* x, MPSDataType dt) {
            return [g reciprocalWithTensor:[g multiplicationWithPrimaryTensor:x
                                                              secondaryTensor:cst(g, 0.6931471805599453, dt)
                                                                         name:nil]
                                      name:nil];
        }));
        register_vjp_emitter(std::make_unique<ExpVjp>());
        register_vjp_emitter(std::make_unique<SqrtVjp>());
        register_vjp_emitter(std::make_unique<RsqrtVjp>());
        register_vjp_emitter(std::make_unique<SquareVjp>());
        register_vjp_emitter(std::make_unique<AbsVjp>());
        register_vjp_emitter(std::make_unique<ReciprocalVjp>());
        register_vjp_emitter(std::make_unique<SinVjp>());
        register_vjp_emitter(std::make_unique<CosVjp>());
    }
};

[[maybe_unused]] static const MathVjpRegistrar g_math_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
