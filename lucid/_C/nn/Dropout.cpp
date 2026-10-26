// lucid/_C/nn/Dropout.cpp
//
// Implementations of all five dropout variants: Dropout, DropoutNd,
// AlphaDropout, DropBlock, and DropPath.
//
// Common pattern for each variant:
//   1. Validate inputs and probability range.
//   2. If training==false (or p==0), clone the input and attach a no-op
//      backward node (empty mask, backward copies grad_out unchanged).
//   3. Draw a Bernoulli mask using the provided Generator or the global
//      default; check_schema_determinism() throws if neither is available
//      and set_deterministic(true) is active.
//   4. Apply the mask, build the TensorImpl, wire the backward node.
//
// Backward for all variants: multiply grad_out elementwise by the saved mask.

#include "Dropout.h"

#include <cmath>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../compile/RngFeeds.h"
#include "../compile/Tracer.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/Generator.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

namespace {

// A node built on the pass-through path (``training=false`` or ``p == 0``)
// holds no mask, and its gradient is the incoming one unchanged.
bool mask_absent(const Storage& mask) {
    if (const auto* cpu = std::get_if<CpuStorage>(&mask))
        return cpu->nbytes == 0;
    const auto* gpu = std::get_if<GpuStorage>(&mask);
    return gpu == nullptr || !gpu->arr;
}

// Graph-mode gradient of ``y = x * mask``: the saved mask is a constant
// of the forward draw, so the gradient is ``grad_out * mask`` — detached
// when ``grad_out`` is, as the reference's ``MulBackward`` leaves it — and
// the pass-through case hands ``grad_out`` straight back.
TensorImplPtr masked_grad_for_graph(const TensorImplPtr& grad_out,
                                    const Storage& mask,
                                    const Shape& shape,
                                    Dtype dt,
                                    Device device) {
    if (mask_absent(mask))
        return grad_out;
    auto m = std::make_shared<TensorImpl>(mask, shape, dt, device, false);
    return mul_op(grad_out, m);
}

}  // namespace

const OpSchema DropoutBackward::schema_v1{
    "dropout", 1, AmpPolicy::KeepInput, false,
    "uses Generator-driven Bernoulli mask; reproducible only with explicit seed"};

TensorImplPtr
DropoutBackward::forward(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    Validator::input(a, "dropout.a").non_null();
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("dropout").invalid_argument("p must be in [0, 1)");

    // Under deterministic mode an explicit generator is required.
    if (training && p > 0.0 && gen == nullptr)
        check_schema_determinism(schema_v1);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    // Expose training + p so the compile-path emitter can detect the
    // inference / p==0 passthrough case and emit an identity bind.
    scope.set_attr("training", training);
    scope.set_attr("p", p);
    const std::size_t numel = a->numel();

    // Pass-through path: no mask is applied; backward copies grad unchanged.
    if (!training || p == 0.0) {
        Storage clone = clone_storage(a->storage(), numel, a->dtype(), a->device());
        auto out = std::make_shared<TensorImpl>(std::move(clone), a->shape(), a->dtype(),
                                                a->device(), false);
        auto bwd = std::make_shared<DropoutBackward>();
        bwd->p_ = 0.0;

        kernel::NaryKernel<DropoutBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
        return out;
    }

    Generator& g = gen ? *gen : default_generator();
    // bernoulli_mask_storage_shape draws {0,1} at rate (1-p); scale = 1/(1-p)
    // gives inverted-dropout scaling so the expected value is preserved.
    Storage mask = bernoulli_mask_storage_shape(1.0 - p, a->shape(), a->dtype(), a->device(), g);
    const double scale = 1.0 / (1.0 - p);

    Storage scaled_mask = mul_scalar_storage(mask, scale, numel, a->dtype(), a->device());

    // Traced, from the engine's generator: the scaled mask becomes a feed
    // that ``run_executable`` draws again on every call, and the op an
    // ordinary multiply — whose VJP is dropout's backward.  A compiled
    // training step then drops out afresh each call, with exactly the mask
    // eager would have drawn; as an op it could only run eager, and every
    // model with dropout trained compiled not at all.  See
    // :file:`compile/RngFeeds.h`.
    if (auto* trc = compile::current_tracer();
        trc != nullptr && gen == nullptr && trc->redraw_rng()) {
        auto mask_t = std::make_shared<TensorImpl>(std::move(scaled_mask), a->shape(), a->dtype(),
                                                   a->device(), false);
        compile::RngRecipe r;
        r.kind = compile::RngRecipe::Kind::DropoutMask;
        r.a = 1.0 - p;

        r.b = scale;
        r.shape = a->shape();
        r.dtype = a->dtype();
        r.device = a->device();
        trc->on_rng_feed(schema_v1.name, mask_t);
        compile::note_rng_feed(mask_t, std::move(r));
        return mul_op(a, mask_t);
    }
    Storage y = multiply_storages(a->storage(), scaled_mask, numel, a->dtype(), a->device());

    auto out =
        std::make_shared<TensorImpl>(std::move(y), a->shape(), a->dtype(), a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    {
        auto bwd = std::make_shared<DropoutBackward>();
        bwd->p_ = p;
        // Store the already-scaled mask; backward needs only one multiply.
        bwd->mask_ = std::move(scaled_mask);
        kernel::NaryKernel<DropoutBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

std::vector<Storage> DropoutBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    if (mask_absent(mask_))
        return {clone_storage(grad_out, numel, dtype_, device_)};
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

std::vector<TensorImplPtr> DropoutBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    return {masked_grad_for_graph(grad_out, mask_, input_shapes_[0], dtype_, device_)};
}

TensorImplPtr dropout_op(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    return DropoutBackward::forward(a, p, training, gen);
}

std::pair<TensorImplPtr, TensorImplPtr> dropout_stateful_op(const TensorImplPtr& x,
                                                            const TensorImplPtr& state_in,
                                                            double p,
                                                            bool training,
                                                            Generator* gen) {
    Validator::input(x, "dropout_stateful.x").non_null();
    Validator::input(state_in, "dropout_stateful.state_in").non_null();
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("dropout_stateful").invalid_argument("p must be in [0, 1)");

    OpScopeFull scope{"dropout_stateful", x->device(), x->dtype(), x->shape()};
    scope.set_attr("training", training);
    scope.set_attr("p", p);

    // Suppress the nested ``dropout_op``'s OpScopeFull from polluting
    // the trace.  Without this the captured graph would contain a
    // ``"dropout"`` node nested under our ``"dropout_stateful"`` node
    // — the MPSGraph emitter would see both, double-emit the mask,
    // and the compile path would fail consistency checks.
    auto* saved_tracer = ::lucid::compile::current_tracer();
    ::lucid::compile::set_current_tracer(nullptr);
    TensorImplPtr y = dropout_op(x, p, training, gen);
    ::lucid::compile::set_current_tracer(saved_tracer);

    // ``state_out`` in eager mode is a verbatim clone of ``state_in`` —
    // Lucid's :class:`Generator` already advances per call, so the
    // state buffer is purely a trace-recording placeholder.  The
    // compile path is what mutates it in-place across dispatches via
    // ``randomTensorWithShape:descriptor:stateTensor:``.
    Storage cloned = clone_storage(state_in->storage(), state_in->numel(), state_in->dtype(),
                                   state_in->device());
    auto state_out = std::make_shared<TensorImpl>(std::move(cloned), state_in->shape(),
                                                  state_in->dtype(), state_in->device(),
                                                  /*requires_grad=*/false);

    // Wire 2 inputs + 2 outputs onto the just-recorded node.
    // The Tracer's repeat-call multi-output protocol (see Tracer.cpp:
    // 142-167) handles this — second call appends a fresh TensorMeta.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({x, state_in}, y);
        trc->on_op_io({x, state_in}, state_out);
    }

    return {y, state_out};
}

LUCID_REGISTER_OP(DropoutBackward)

namespace {

// Build the (N, C, 1, 1, ...) mask shape used by DropoutNd.
// All spatial dimensions become 1 so the mask broadcasts over them.
Shape channel_mask_shape(const Shape& in) {
    require_min_rank(in, 2, "dropoutnd: x", "(N, C, *)");
    Shape m(in.size(), 1);
    m[0] = in[0];
    m[1] = in[1];
    return m;
}

// Build the (N, 1, 1, ...) mask shape used by DropPath.
// Only the batch dimension is non-trivial; all other dims broadcast.
Shape sample_mask_shape(const Shape& in) {
    require_min_rank(in, 1, "drop_path: x", "(N, *)");
    Shape m(in.size(), 1);
    m[0] = in[0];
    return m;
}

// Whether a mask drawn now should become a feed of the active trace: the
// engine's generator, and a tracer that redraws (Core ML export does not).
bool traced_mask(Generator* gen) {
    auto* trc = compile::current_tracer();
    return trc != nullptr && gen == nullptr && trc->redraw_rng();
}

// Dropout's traced form for a broadcast mask — channel (DropoutNd) or sample
// (DropPath): the drawn, scaled mask becomes a feed ``run_executable`` draws
// again on every call, and the op a broadcasting multiply, whose VJP is the
// op's backward.  As ops they had no traced form, so every model that used
// them trained compiled not at all.  See ``DropoutBackward::forward``.
TensorImplPtr traced_mask_multiply(std::string_view name,
                                   const TensorImplPtr& a,
                                   Storage scaled,
                                   const Shape& mask_shape,
                                   double keep,
                                   double scale) {
    auto* trc = compile::current_tracer();
    auto mask_t =
        std::make_shared<TensorImpl>(std::move(scaled), mask_shape, a->dtype(), a->device(), false);
    compile::RngRecipe r;
    r.kind = compile::RngRecipe::Kind::DropoutMask;
    r.a = keep;

    r.b = scale;
    r.shape = mask_shape;
    r.dtype = a->dtype();
    r.device = a->device();
    trc->on_rng_feed(name, mask_t);
    compile::note_rng_feed(mask_t, std::move(r));
    return mul_op(a, mask_t);
}

}  // namespace

const OpSchema DropoutNdBackward::schema_v1{
    "dropoutnd", 1, AmpPolicy::KeepInput, false,
    "Bernoulli channel mask; reproducible only with explicit seed"};

TensorImplPtr
DropoutNdBackward::forward(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    Validator::input(a, "dropoutnd.a").non_null();
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("dropoutnd").invalid_argument("p must be in [0, 1)");

    if (training && p > 0.0 && gen == nullptr)
        check_schema_determinism(schema_v1);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    scope.set_attr("training", training);
    scope.set_attr("p", p);
    const std::size_t numel = a->numel();

    if (!training || p == 0.0) {
        Storage clone = clone_storage(a->storage(), numel, a->dtype(), a->device());
        auto out = std::make_shared<TensorImpl>(std::move(clone), a->shape(), a->dtype(),
                                                a->device(), false);
        auto bwd = std::make_shared<DropoutNdBackward>();
        bwd->p_ = 0.0;
        kernel::NaryKernel<DropoutNdBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
        return out;
    }

    // Generate a compact (N, C) mask and let the backend expand it to full shape.
    Shape mask_shape = channel_mask_shape(a->shape());
    const std::size_t mask_numel =
        static_cast<std::size_t>(mask_shape[0]) * static_cast<std::size_t>(mask_shape[1]);
    Generator& g = gen ? *gen : default_generator();
    Storage small_mask =
        bernoulli_mask_storage_shape(1.0 - p, mask_shape, a->dtype(), a->device(), g);
    const double scale = 1.0 / (1.0 - p);
    Storage scaled_small =
        mul_scalar_storage(small_mask, scale, mask_numel, a->dtype(), a->device());
    if (traced_mask(gen))
        return traced_mask_multiply(schema_v1.name, a, std::move(scaled_small), mask_shape, 1.0 - p,
                                    scale);
    auto& be = backend::Dispatcher::for_device(a->device());
    // expand_and_multiply broadcasts the (N,C) mask to the full input shape
    // and multiplies; returns the full mask and the masked output.
    auto [full_mask, y] =
        be.expand_and_multiply(scaled_small, a->storage(), mask_shape, a->shape(), a->dtype());

    auto out =
        std::make_shared<TensorImpl>(std::move(y), a->shape(), a->dtype(), a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    {
        auto bwd = std::make_shared<DropoutNdBackward>();
        bwd->p_ = p;
        bwd->mask_ = std::move(full_mask);
        kernel::NaryKernel<DropoutNdBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

std::vector<Storage> DropoutNdBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    if (mask_absent(mask_))
        return {clone_storage(grad_out, numel, dtype_, device_)};
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

std::vector<TensorImplPtr> DropoutNdBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    return {masked_grad_for_graph(grad_out, mask_, input_shapes_[0], dtype_, device_)};
}

TensorImplPtr dropoutnd_op(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    return DropoutNdBackward::forward(a, p, training, gen);
}

LUCID_REGISTER_OP(DropoutNdBackward)

namespace {
// SELU activation constants used by AlphaDropout.
constexpr double kSeluAlpha = 1.6732632423543772;
constexpr double kSeluLambda = 1.0507009873554805;
// kAlphaPrime = -lambda * alpha is the value assigned to dropped elements.
constexpr double kAlphaPrime = -kSeluLambda * kSeluAlpha;
}  // namespace

const OpSchema AlphaDropoutBackward::schema_v1{
    "alpha_dropout", 1, AmpPolicy::KeepInput, false,
    "Bernoulli mask + SELU rescaling; reproducible only with explicit seed"};

TensorImplPtr
AlphaDropoutBackward::forward(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    Validator::input(a, "alpha_dropout.a").non_null();
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("alpha_dropout").invalid_argument("p must be in [0, 1)");

    if (training && p > 0.0 && gen == nullptr)
        check_schema_determinism(schema_v1);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    scope.set_attr("training", training);
    scope.set_attr("p", p);
    const std::size_t numel = a->numel();

    if (!training || p == 0.0) {
        Storage clone = clone_storage(a->storage(), numel, a->dtype(), a->device());
        auto out = std::make_shared<TensorImpl>(std::move(clone), a->shape(), a->dtype(),
                                                a->device(), false);
        auto bwd = std::make_shared<AlphaDropoutBackward>();
        bwd->p_ = 0.0;
        bwd->a_coef_ = 1.0;
        kernel::NaryKernel<AlphaDropoutBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
        return out;
    }

    // Affine coefficients that restore mean and variance after masking.
    // a_coef = (keep * (1 + p * alpha'^2))^{-0.5}
    // b_coef = -a_coef * p * alpha'
    const double keep = 1.0 - p;
    const double a_coef = std::pow(keep * (1.0 + p * kAlphaPrime * kAlphaPrime), -0.5);
    const double b_coef = -a_coef * p * kAlphaPrime;

    Generator& g = gen ? *gen : default_generator();
    Storage mask = bernoulli_mask_storage_shape(keep, a->shape(), a->dtype(), a->device(), g);

    // Zero kept elements stay as-is; dropped elements receive kAlphaPrime.
    // Reconstruct: x_keep + (1-mask)*alpha' = x*mask + alpha'*(1-mask)
    //            = x*mask - mask*alpha' + alpha'
    Storage x_mask = multiply_storages(a->storage(), mask, numel, a->dtype(), a->device());

    Storage alpha_mask = mul_scalar_storage(mask, kAlphaPrime, numel, a->dtype(), a->device());
    Storage neg_alpha_mask = mul_scalar_storage(alpha_mask, -1.0, numel, a->dtype(), a->device());
    Storage one_m_mask_ap =
        add_scalar_storage(neg_alpha_mask, kAlphaPrime, numel, a->dtype(), a->device());

    Storage inner = add_storages(x_mask, one_m_mask_ap, numel, a->dtype(), a->device());
    Storage scaled = mul_scalar_storage(inner, a_coef, numel, a->dtype(), a->device());
    Storage y = add_scalar_storage(scaled, b_coef, numel, a->dtype(), a->device());

    auto out =
        std::make_shared<TensorImpl>(std::move(y), a->shape(), a->dtype(), a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    {
        auto bwd = std::make_shared<AlphaDropoutBackward>();
        bwd->p_ = p;
        bwd->a_coef_ = a_coef;
        bwd->mask_ = std::move(mask);
        kernel::NaryKernel<AlphaDropoutBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

std::vector<Storage> AlphaDropoutBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    if (mask_absent(mask_))
        return {clone_storage(grad_out, numel, dtype_, device_)};

    // Chain rule: d/dx[a*x + b] = a; mask selects which x were kept.
    Storage scaled_mask = mul_scalar_storage(mask_, a_coef_, numel, dtype_, device_);
    return {multiply_storages(grad_out, scaled_mask, numel, dtype_, device_)};
}

std::vector<TensorImplPtr> AlphaDropoutBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    if (mask_absent(mask_))
        return {grad_out};
    const std::size_t numel = shape_numel(input_shapes_[0]);
    return {masked_grad_for_graph(grad_out,
                                  mul_scalar_storage(mask_, a_coef_, numel, dtype_, device_),
                                  input_shapes_[0], dtype_, device_)};
}

TensorImplPtr alpha_dropout_op(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    return AlphaDropoutBackward::forward(a, p, training, gen);
}

LUCID_REGISTER_OP(AlphaDropoutBackward)

const OpSchema DropBlockBackward::schema_v1{
    "drop_block", 1, AmpPolicy::KeepInput, false,
    "Bernoulli + spatial dilation; reproducible only with explicit seed"};

TensorImplPtr DropBlockBackward::forward(
    const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen) {
    Validator::input(a, "drop_block.a").non_null();
    require_rank(a->shape(), 4, "drop_block: x", "(N, C, H, W)");
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("drop_block").invalid_argument("p must be in [0, 1)");
    if (block_size <= 0)
        ErrorBuilder("drop_block").invalid_argument("block_size must be > 0");

    if (p > 0.0 && gen == nullptr)
        check_schema_determinism(schema_v1);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    const std::size_t numel = a->numel();
    const std::int64_t H = a->shape()[2];
    const std::int64_t W = a->shape()[3];

    // gamma converts the block-level drop rate p into the per-element seed
    // probability that, after dilation, achieves the requested p on average.
    const double feat = static_cast<double>(H * W);
    const double valid = static_cast<double>((H - block_size + 1) * (W - block_size + 1));
    const double gamma = p * feat / (block_size * block_size) / (valid + eps);

    Generator& g = gen ? *gen : default_generator();
    Storage seed = bernoulli_mask_storage(gamma, numel, a->dtype(), a->device(), g);
    auto& be = backend::Dispatcher::for_device(a->device());
    // drop_block_mask dilates the seed into contiguous block masks.
    Storage keep_storage = be.drop_block_mask(seed, p, block_size, a->shape(), a->dtype());
    Storage y = multiply_storages(a->storage(), keep_storage, numel, a->dtype(), a->device());

    auto out =
        std::make_shared<TensorImpl>(std::move(y), a->shape(), a->dtype(), a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    {
        auto bwd = std::make_shared<DropBlockBackward>();
        bwd->p_ = p;
        bwd->mask_ = std::move(keep_storage);
        kernel::NaryKernel<DropBlockBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

std::vector<Storage> DropBlockBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

std::vector<TensorImplPtr> DropBlockBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    return {masked_grad_for_graph(grad_out, mask_, input_shapes_[0], dtype_, device_)};
}

TensorImplPtr drop_block_op(
    const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen) {
    return DropBlockBackward::forward(a, block_size, p, eps, gen);
}

LUCID_REGISTER_OP(DropBlockBackward)

const OpSchema DropPathBackward::schema_v1{
    "drop_path", 1, AmpPolicy::KeepInput, false,
    "Per-sample Bernoulli; reproducible only with explicit seed"};

TensorImplPtr
DropPathBackward::forward(const TensorImplPtr& a, double p, bool scale_by_keep, Generator* gen) {
    Validator::input(a, "drop_path.a").non_null();
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("drop_path").invalid_argument("p must be in [0, 1)");

    if (p > 0.0 && gen == nullptr)
        check_schema_determinism(schema_v1);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    const std::size_t numel = a->numel();

    if (p == 0.0) {
        Storage clone = clone_storage(a->storage(), numel, a->dtype(), a->device());
        auto out = std::make_shared<TensorImpl>(std::move(clone), a->shape(), a->dtype(),
                                                a->device(), false);
        kernel::NaryKernel<DropPathBackward, 1>::wire_autograd({a}, out, false);
        return out;
    }

    const double keep = 1.0 - p;
    const std::size_t B = static_cast<std::size_t>(a->shape()[0]);
    // One Bernoulli sample per batch element; shape (N, 1, 1, ...).
    Shape sshape = sample_mask_shape(a->shape());
    Generator& g = gen ? *gen : default_generator();
    Storage small = bernoulli_mask_storage_shape(keep, sshape, a->dtype(), a->device(), g);
    const double scale = (scale_by_keep && keep > 0.0) ? 1.0 / keep : 1.0;
    if (scale != 1.0) {
        // Scale before broadcast so the full-mask multiply also scales.
        small = mul_scalar_storage(small, scale, B, a->dtype(), a->device());
    }
    if (traced_mask(gen))
        return traced_mask_multiply(schema_v1.name, a, std::move(small), sshape, keep, scale);
    auto& be = backend::Dispatcher::for_device(a->device());
    auto [full_mask, y] =
        be.expand_and_multiply(small, a->storage(), sshape, a->shape(), a->dtype());

    auto out =
        std::make_shared<TensorImpl>(std::move(y), a->shape(), a->dtype(), a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    {
        auto bwd = std::make_shared<DropPathBackward>();
        bwd->mask_ = std::move(full_mask);
        kernel::NaryKernel<DropPathBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

std::vector<Storage> DropPathBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    if (mask_absent(mask_))
        return {clone_storage(grad_out, numel, dtype_, device_)};
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

std::vector<TensorImplPtr> DropPathBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    return {masked_grad_for_graph(grad_out, mask_, input_shapes_[0], dtype_, device_)};
}

TensorImplPtr drop_path_op(const TensorImplPtr& a, double p, bool scale_by_keep, Generator* gen) {
    return DropPathBackward::forward(a, p, scale_by_keep, gen);
}

LUCID_REGISTER_OP(DropPathBackward)

}  // namespace lucid
