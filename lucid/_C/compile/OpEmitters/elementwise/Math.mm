// lucid/_C/compile/OpEmitters/UnaryMath.mm
//
// Single-tensor mathematical emitters: neg / abs / exp / log / sqrt /
// square / reciprocal / rsqrt.
//
// Op names match the engine schemas in:
//   - lucid/_C/ops/ufunc/Arith.cpp        ("neg", "abs", "reciprocal", "square")
//   - lucid/_C/ops/ufunc/Exponential.cpp  ("exp", "log", "sqrt", "rsqrt")
//
// All eight ops are 1-input → 1-output, no attribute payload required —
// MPSGraph exposes them directly as one-call builders (or, for
// ``square`` / ``rsqrt``, a single derived builder over the primitive).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <memory>
#include <optional>
#include <string_view>
#include <variant>

#include "../OpEmitter.h"
#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

template <class BuilderBlock>
inline bool emit_unary_math(BuilderContext& ctx, const OpNode& node, BuilderBlock builder) {
    if (node.inputs.size() != 1)
        return false;
    TensorId x_id = node.inputs[0];
    if (x_id < 0)
        return false;
    MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
    MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
    if (x_t == nil || graph == nil)
        return false;
    x_t = promote_to_float_output(graph, x_t, node);
    ctx.bind(node.outputs[0].id, (__bridge void*)(builder(graph, x_t)));
        return true;
}

class NegEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "neg"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g negativeWithTensor:x name:@"neg"];
        });
    }
};

class AbsEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "abs"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g absoluteWithTensor:x name:@"abs"];
        });
    }
};

class ExpEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "exp"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g exponentWithTensor:x name:@"exp"];
        });
    }
};

class LogEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "log"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g logarithmWithTensor:x name:@"log"];
        });
    }
};

class SqrtEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sqrt"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g squareRootWithTensor:x name:@"sqrt"];
        });
    }
};

class SquareEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "square"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            // ``squareWithTensor`` takes floating point only (throwing on
            // narrower integers) and has no ``square_i64`` kernel on 64-bit
            // integers.  ``x * x`` is the same answer on integers; for 64-bit
            // integers we wrap the two operands in ``maximum(x, x)`` and
            // ``minimum(x, x)`` so MPSGraph's algebraic simplifier cannot fold
            // the multiply back into ``square_i64``.
            if (x.dataType == MPSDataTypeInt64 || x.dataType == MPSDataTypeUInt64) {
                MPSGraphTensor* lhs = [g maximumWithPrimaryTensor:x secondaryTensor:x name:nil];
                MPSGraphTensor* rhs = [g minimumWithPrimaryTensor:x secondaryTensor:x name:nil];
                return [g multiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:@"square"];
            }
            if (!(x.dataType & MPSDataTypeFloatBit))
                return [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:@"square"];
            return [g squareWithTensor:x name:@"square"];
        });
    }
};

class ReciprocalEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "reciprocal"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g reciprocalWithTensor:x name:@"reciprocal"];
        });
    }
};

class RsqrtEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "rsqrt"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            if (x.dataType == MPSDataTypeFloat32) {
                MPSGraphTensor* root = [g squareRootWithTensor:x name:@"rsqrt_sqrt"];
                return [g reciprocalWithTensor:root name:@"rsqrt"];
            }
            return [g reciprocalSquareRootWithTensor:x name:@"rsqrt"];
        });
    }
};

// Phase 2 (R1) — additional unary math: trig inverses + rounding +
// cube + log2 + sign + erf + clip.  Each is a direct MPSGraph
// 1-builder so the per-op emitter body stays a single lambda.

class ArccosEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "arccos"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g acosWithTensor:x name:@"arccos"];
        });
    }
};

class ArcsinEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "arcsin"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g asinWithTensor:x name:@"arcsin"];
        });
    }
};

class ArctanEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "arctan"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g atanWithTensor:x name:@"arctan"];
        });
    }
};

class CeilEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "ceil"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g ceilWithTensor:x name:@"ceil"];
        });
    }
};

class FloorEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "floor"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g floorWithTensor:x name:@"floor"];
        });
    }
};

// ``trunc`` became an op of its own (it was a ``where`` over floor and ceil).
// An integer is already whole; MPSGraph's truncate is for floats.
class TruncEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "trunc"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            if ((x.dataType & MPSDataTypeFloatBit) == 0)
                return [g identityWithTensor:x name:@"trunc"];
            return [g truncateWithTensor:x name:@"trunc"];
        });
    }
};

class RoundEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "round"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g roundWithTensor:x name:@"round"];
        });
    }
};

class CubeEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cube"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            MPSGraphTensor* x2 = [g squareWithTensor:x name:@"cube_sq"];
            return [g multiplicationWithPrimaryTensor:x2
                                       secondaryTensor:x
                                                  name:@"cube"];
        });
    }
};

class CubeRootEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "cube_root"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            // Sign-preserving cbrt(x) = sign(x) · |x|^(1/3).
            MPSGraphTensor* a = [g absoluteWithTensor:x name:@"cbrt_abs"];
            MPSGraphTensor* third =
                [g constantWithScalar:(1.0 / 3.0) dataType:a.dataType];
            MPSGraphTensor* mag = [g powerWithPrimaryTensor:a
                                            secondaryTensor:third
                                                       name:@"cbrt_pow"];
            MPSGraphTensor* s = [g signWithTensor:x name:@"cbrt_sign"];
            return [g multiplicationWithPrimaryTensor:s
                                       secondaryTensor:mag
                                                  name:@"cube_root"];
        });
    }
};

class Log2Emitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "log2"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g logarithmBase2WithTensor:x name:@"log2"];
        });
    }
};

class SignEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "sign"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g signWithTensor:x name:@"sign"];
        });
    }
};

class ErfEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "erf"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            return [g erfWithTensor:x name:@"erf"];
        });
    }
};

// ``erfinv`` — the inverse error function, which MPSGraph does not have.
//
// Giles (2010), "Approximating the erfinv function": two polynomials in
// ``w = -log((1 - x)(1 + x))``, one for the central region and one for
// the tails, the single-precision coefficients.  Both branches are
// evaluated and one is selected, because a graph has no branch — which
// costs arithmetic and nothing in accuracy.
//
// Measured against Lucid's own on |x| < 1: ~3e-7 in the central region
// and ~5e-6 in the tails, the order of the pipeline's other float32
// error rather than a new kind of it.  The same approximation backs
// ``lucid.coreml``'s emitter, so the two exports agree.
namespace {

constexpr double kErfinvCentral[] = {
    2.81022636e-08,  3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
    -0.00125372503, -0.00417768164,  0.246640727,    1.50140941,
};
constexpr double kErfinvTail[] = {
    -0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
    -0.0076224613,   0.00943887047,  1.00167406,    2.83297682,
};

template <std::size_t N>
MPSGraphTensor* horner(MPSGraph* g,
                       const double (&coefficients)[N],
                       MPSGraphTensor* w,
                       MPSDataType dt) {
    MPSGraphTensor* value = [g constantWithScalar:coefficients[0] dataType:dt];
    for (std::size_t i = 1; i < N; ++i) {
        MPSGraphTensor* scaled =
            [g multiplicationWithPrimaryTensor:value secondaryTensor:w name:nil];
        value = [g additionWithPrimaryTensor:scaled
                             secondaryTensor:[g constantWithScalar:coefficients[i]
                                                          dataType:dt]
                                        name:nil];
    }
    return value;
}

}  // namespace

class ErfinvEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "erfinv"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        return emit_unary_math(ctx, node, [](MPSGraph* g, MPSGraphTensor* x) {
            const MPSDataType dt = x.dataType;
            MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:dt];

            // w = -log((1 - x) * (1 + x))
            MPSGraphTensor* lower =
                [g subtractionWithPrimaryTensor:one secondaryTensor:x name:nil];
            MPSGraphTensor* upper =
                [g additionWithPrimaryTensor:x secondaryTensor:one name:nil];
            MPSGraphTensor* product =
                [g multiplicationWithPrimaryTensor:lower secondaryTensor:upper name:nil];
            MPSGraphTensor* w = [g negativeWithTensor:[g logarithmWithTensor:product name:nil]
                                                 name:nil];

            MPSGraphTensor* central = horner(
                g, kErfinvCentral,
                [g subtractionWithPrimaryTensor:w
                                secondaryTensor:[g constantWithScalar:2.5 dataType:dt]
                                           name:nil],
                dt);
            MPSGraphTensor* tail = horner(
                g, kErfinvTail,
                [g subtractionWithPrimaryTensor:[g squareRootWithTensor:w name:nil]
                                secondaryTensor:[g constantWithScalar:3.0 dataType:dt]
                                           name:nil],
                dt);

            MPSGraphTensor* near =
                [g lessThanWithPrimaryTensor:w
                             secondaryTensor:[g constantWithScalar:5.0 dataType:dt]
                                        name:nil];
            MPSGraphTensor* chosen = [g selectWithPredicateTensor:near
                                             truePredicateTensor:central
                                            falsePredicateTensor:tail
                                                            name:nil];
            MPSGraphTensor* y = [g multiplicationWithPrimaryTensor:chosen
                                                   secondaryTensor:x
                                                              name:nil];
            // erfinv(±1) = ±inf.  There w is +inf, and the tail polynomial
            // evaluated at infinity takes its leading coefficient's sign —
            // which came back as erfinv(1) = -inf, erfinv(-1) = +inf.
            MPSGraphTensor* edge = [g equalWithPrimaryTensor:[g absoluteWithTensor:x name:nil]
                                             secondaryTensor:one
                                                        name:nil];
            MPSGraphTensor* signed_inf =
                [g multiplicationWithPrimaryTensor:x
                                   secondaryTensor:[g constantWithScalar:INFINITY dataType:dt]
                                              name:nil];
            return [g selectWithPredicateTensor:edge
                            truePredicateTensor:signed_inf
                           falsePredicateTensor:y
                                           name:@"erfinv"];
        });
    }
};

// ``clip(x, min, max)`` — Lucid records ``clip_min`` / ``clip_max``
// double attrs on the OpScopeFull (the underlying op may pass either
// or both bounds; missing bounds default to ±∞ via NaN/inf sentinels
// already filtered out of the attrs).  When both bounds are present
// we use ``clampWithTensor:min:max:``; when only one is present we
// compose `maximum` / `minimum` against a constant.
class ClipEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "clip"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1 || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;

        auto have = [&](const char* key) -> std::optional<double> {
            auto it = node.attrs.find(key);
            if (it == node.attrs.end())
                return std::nullopt;
            const auto* p = std::get_if<double>(&it->second);
            return p ? std::optional<double>(*p) : std::nullopt;
        };
        std::optional<double> lo = have("min");
        std::optional<double> hi = have("max");
        if (!lo && !hi) {
            // No bounds → clip is an identity; bind input verbatim
            // (the early-return previously here was outside the
            // braces, which silently skipped the maximum/minimum
            // emission below and left output unbound when either
            // bound was set — a latent bug that surfaced when the
            // compiled Rprop / LBFGS optimizers started using
            // ``Tensor.clip(lo, hi)`` with both ends).
            ctx.bind(node.outputs[0].id, (__bridge void*)(x_t));
            return true;
        }
        MPSGraphTensor* out = x_t;
        if (lo) {
            MPSGraphTensor* c =
                [graph constantWithScalar:*lo dataType:x_t.dataType];
            out = [graph maximumWithPrimaryTensor:out
                                   secondaryTensor:c
                                              name:@"clip_lo"];
        }
        if (hi) {
            MPSGraphTensor* c =
                [graph constantWithScalar:*hi dataType:x_t.dataType];
            out = [graph minimumWithPrimaryTensor:out
                                   secondaryTensor:c
                                              name:@"clip_hi"];
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)(out));
        return true;
    }
};

struct UnaryMathEmitterRegistrar {
    UnaryMathEmitterRegistrar() {
        register_emitter(std::make_unique<NegEmitter>());
        register_emitter(std::make_unique<AbsEmitter>());
        register_emitter(std::make_unique<ExpEmitter>());
        register_emitter(std::make_unique<LogEmitter>());
        register_emitter(std::make_unique<SqrtEmitter>());
        register_emitter(std::make_unique<SquareEmitter>());
        register_emitter(std::make_unique<ReciprocalEmitter>());
        register_emitter(std::make_unique<RsqrtEmitter>());
        // R1 additions.
        register_emitter(std::make_unique<ArccosEmitter>());
        register_emitter(std::make_unique<ArcsinEmitter>());
        register_emitter(std::make_unique<ArctanEmitter>());
        register_emitter(std::make_unique<CeilEmitter>());
        register_emitter(std::make_unique<FloorEmitter>());
        register_emitter(std::make_unique<TruncEmitter>());
        register_emitter(std::make_unique<RoundEmitter>());
        register_emitter(std::make_unique<CubeEmitter>());
        register_emitter(std::make_unique<CubeRootEmitter>());
        register_emitter(std::make_unique<Log2Emitter>());
        register_emitter(std::make_unique<SignEmitter>());
        register_emitter(std::make_unique<ErfEmitter>());
        register_emitter(std::make_unique<ErfinvEmitter>());
        register_emitter(std::make_unique<ClipEmitter>());
    }
};

[[maybe_unused]] static const UnaryMathEmitterRegistrar g_unary_math_registrar;

}  // namespace

}  // namespace lucid::compile
