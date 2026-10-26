// lucid/_C/random/Random.cpp
//
// Thin wrappers that route each random operation through the Storage-layer
// random helpers and wrap the result in a non-differentiable TensorImpl.
// All distribution-specific logic lives in core/Storage (CPU) or is
// delegated to mlx::core::random (GPU); this file is concerned only with
// profiling scope management, argument validation, and TensorImpl construction.

#include "Random.h"

#include <cmath>
#include <utility>

#include "../autograd/Helpers.h"
#include "../compile/RngFeeds.h"
#include "../compile/Tracer.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/Generator.h"
#include "../core/PhantomMode.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"

namespace lucid {

namespace {

// Fall back to the process-global default Generator when none is supplied.
inline Generator& resolve_gen(Generator* gen) {
    return gen ? *gen : default_generator();
}

// Construct a leaf TensorImpl from a freshly allocated Storage.
// requires_grad is always false for random tensors.
inline TensorImplPtr finalize(Storage&& storage, Shape shape, Dtype dt, Device device) {
    return std::make_shared<TensorImpl>(std::move(storage), std::move(shape), dt, device, false);
}

// Draws complex128 on the CPU as twice as many float64 parts, relabelled.
// Under phantom mode there are no parts to relabel: the draw below returns
// a phantom of the complex dtype itself.
bool draws_c128_as_parts(Dtype dt, Device device) {
    return dt == Dtype::C128 && device == Device::CPU && !PhantomMode::is_enabled();
}

Storage
uniform_storage(const Shape& shape, double lo, double hi, Dtype dt, Device device, Generator& gen) {
    if (draws_c128_as_parts(dt, device)) {
        const auto n = static_cast<std::int64_t>(shape_numel(shape));
        Storage s = random_uniform_storage(Shape{2 * n}, lo, hi, Dtype::F64, device, gen);
        std::get<CpuStorage>(s).dtype = Dtype::C128;
        return s;
    }
    return random_uniform_storage(shape, lo, hi, dt, device, gen);
}

Storage normal_storage(
    const Shape& shape, double mean, double std, Dtype dt, Device device, Generator& gen) {
    if (draws_c128_as_parts(dt, device)) {
        const std::size_t n = shape_numel(shape);
        Storage s = random_normal_storage(Shape{static_cast<std::int64_t>(2 * n)}, 0.0,
                                          std / std::sqrt(2.0), Dtype::F64, device, gen);
        auto& cpu = std::get<CpuStorage>(s);
        auto* parts = reinterpret_cast<double*>(cpu.ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            parts[2 * i] += mean;
        cpu.dtype = Dtype::C128;
        return s;
    }
    return random_normal_storage(shape, mean, std, dt, device, gen);
}

}  // namespace

// Tracer hook.  A draw from the default generator becomes a feed that
// ``run_executable`` draws again on every call — see
// :file:`compile/RngFeeds.h` for why a seeded op in the graph is not
// enough.  A draw from a caller's own generator stays an op, tagged so
// the compile path declines it: the executable could not advance that
// generator, and an eager fallback is the only way to keep its stream.
namespace {
inline void register_rng_output(const char* name,
                                const TensorImplPtr& out,
                                Generator* gen,
                                compile::RngRecipe recipe) {
    auto* trc = ::lucid::compile::current_tracer();
    if (trc == nullptr)
        return;
    const bool default_gen = gen == nullptr || gen == &default_generator();
    if (default_gen && !trc->redraw_rng()) {
        // Kept as an op: the consumer of this trace wants the draw itself.
        trc->on_op_io({}, out);
        return;
    }
    if (default_gen) {
        recipe.shape = out->shape();
        recipe.dtype = out->dtype();
        recipe.device = out->device();
        trc->on_rng_feed(name, out);
        compile::note_rng_feed(out, std::move(recipe));
        return;
    }
    trc->on_op_attr("own_generator", true);
    trc->on_op_io({}, out);
}

compile::RngRecipe recipe_of(compile::RngRecipe::Kind kind, double a, double b) {
    compile::RngRecipe r;
    r.kind = kind;
    r.a = a;
    r.b = b;
    return r;
}
}  // namespace

// Fill with U(0, 1) uniform samples.
TensorImplPtr rand_op(const Shape& shape, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"rand", device, dt, shape};
    check_concrete_shape(shape);
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    auto s = uniform_storage(shape, 0.0, 1.0, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output("rand", out, gen, recipe_of(compile::RngRecipe::Kind::Uniform, 0.0, 1.0));
    return out;
}

// Fill with U(low, high) uniform samples; validate that high > low first.
TensorImplPtr
uniform_op(const Shape& shape, double low, double high, Dtype dt, Device device, Generator* gen) {
    if (high <= low)
        ErrorBuilder("uniform").invalid_argument("high must be > low");
    OpScopeFull scope{"uniform", device, dt, shape};
    check_concrete_shape(shape);
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    scope.set_attr("low", low);
    scope.set_attr("high", high);
    auto s = uniform_storage(shape, low, high, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output("uniform", out, gen,
                        recipe_of(compile::RngRecipe::Kind::Uniform, low, high));
    return out;
}

// Fill with N(0, 1) standard-normal samples.
TensorImplPtr randn_op(const Shape& shape, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"randn", device, dt, shape};
    check_concrete_shape(shape);
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    auto s = normal_storage(shape, 0.0, 1.0, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output("randn", out, gen, recipe_of(compile::RngRecipe::Kind::Normal, 0.0, 1.0));
    return out;
}

// Fill with N(mean, std^2) samples; validate non-negative std.
TensorImplPtr
normal_op(const Shape& shape, double mean, double std, Dtype dt, Device device, Generator* gen) {
    if (std < 0.0)
        ErrorBuilder("normal").invalid_argument("std must be >= 0");
    OpScopeFull scope{"normal", device, dt, shape};
    check_concrete_shape(shape);
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    scope.set_attr("mean", mean);
    scope.set_attr("std", std);
    auto s = normal_storage(shape, mean, std, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output("normal", out, gen, recipe_of(compile::RngRecipe::Kind::Normal, mean, std));
    return out;
}

// Fill with integer samples drawn uniformly from [low, high).
TensorImplPtr randint_op(const Shape& shape,
                         std::int64_t low,
                         std::int64_t high,
                         Dtype dt,
                         Device device,
                         Generator* gen) {
    OpScopeFull scope{"randint", device, dt, shape};
    check_concrete_shape(shape);
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    scope.set_attr("low", low);
    scope.set_attr("high", high);
    auto s = random_randint_storage(shape, low, high, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    auto r = recipe_of(compile::RngRecipe::Kind::Randint, 0.0, 0.0);
    r.lo = low;
    r.hi = high;
    register_rng_output("randint", out, gen, std::move(r));
    return out;
}

// Fill with independent Bernoulli(p) samples.
TensorImplPtr bernoulli_op(const Shape& shape, double p, Dtype dt, Device device, Generator* gen) {
    OpScopeFull scope{"bernoulli", device, dt, shape};
    check_concrete_shape(shape);
    scope.set_attr("seed", static_cast<std::int64_t>(resolve_gen(gen).counter()));
    scope.set_attr("p", p);
    auto s = random_bernoulli_storage(shape, p, dt, device, resolve_gen(gen));
    auto out = finalize(std::move(s), shape, dt, device);
    register_rng_output("bernoulli", out, gen,
                        recipe_of(compile::RngRecipe::Kind::Bernoulli, p, 0.0));
    return out;
}

}  // namespace lucid
