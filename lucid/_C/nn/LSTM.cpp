// lucid/_C/nn/LSTM.cpp
//
// LSTM forward and BPTT backward implementation.
//
// forward() validates every operand against the input at the door, then
// decides between two backend paths based on whether any input requires a
// gradient:
//   Inference (no grad): IBackend::lstm_forward (returns just output/hn/cn).
//   Training   (grad):   IBackend::lstm_forward_train (saves gates/cells for BPTT).
//
// The training path returns 5 Storage objects:
//   res[0] – output sequence (T, B, H).
//   res[1] – final hidden state hn (1, B, H).
//   res[2] – final cell state cn (1, B, H).
//   res[3] – gates_all (T, B, 4H).
//   res[4] – cells_all (T+1, B, H).
//
// All three outputs share one LstmBackward node, at output slots 0 / 1 / 2.
// The node is a barrier: it waits for the gradient of every output it is
// going to receive and runs BPTT once, seeded with dh_n / dc_n.

#include "LSTM.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/GraphBarrier.h"
#include "../autograd/Helpers.h"
#include "../autograd/TensorHooks.h"
#include "../backend/Dispatcher.h"
#include "../compile/Tracer.h"
#include "../core/Allocator.h"
#include "../core/AmpPolicy.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../ops/ufunc/Astype.h"

namespace lucid {

namespace {

using LstmOpts = backend::IBackend::LstmOpts;

// {output, hn, cn} as the caller sees them: (T, B, Hrec), (1, B, Hrec),
// (1, B, H).  Hrec is proj_size when projection is on.
std::array<Shape, 3> lstm_output_shapes(const LstmOpts& opts) {
    const std::int64_t T = opts.seq_len, B = opts.batch_size, H = opts.hidden_size;
    const std::int64_t Hrec = opts.proj_size > 0 ? opts.proj_size : H;
    return {Shape{T, B, Hrec}, Shape{1, B, Hrec}, Shape{1, B, H}};
}

void check_operand(const ErrorBuilder& err,
                   const TensorImplPtr& t,
                   const std::string& what,
                   const Shape& want,
                   Dtype dt,
                   Device dev) {
    if (!t)
        err.invalid_argument(what + " is null");
    if (t->device() != dev)
        err.device_mismatch(dev, t->device(), what + " must be on the input's device");
    if (t->dtype() != dt)
        err.dtype_mismatch(dt, t->dtype(), what + " must have the input's dtype");
    if (t->shape() != want)
        err.shape_mismatch(want, t->shape(), what);
}

// The backends read every operand as a raw buffer sized from ``opts``, so an
// operand on another device, of another dtype or of another shape has to be
// refused here — past this point it is a bad_variant_access or a read past
// the end of a buffer.
void validate_operands(const TensorImplPtr& input,
                       const TensorImplPtr& h0,
                       const TensorImplPtr& c0,
                       const std::vector<TensorImplPtr>& weights,
                       const LstmOpts& opts) {
    const ErrorBuilder err("lstm");
    if (!input)
        err.invalid_argument("input is null");
    if (opts.num_layers != 1 || opts.bidirectional || opts.batch_first)
        err.not_implemented(
            "the engine runs one layer in one direction over a sequence-first input; "
            "stacking, directions and batch_first are composed by the caller");
    // The reference refuses an empty sequence, and the GPU kernel stacks its
    // per-step outputs — there are none to stack.  Same words as the Python
    // recurrent modules' door (``_sequence_major``).
    if (opts.seq_len < 1)
        err.invalid_argument("expected sequence length to be larger than 0, got " +
                             std::to_string(opts.seq_len));
    const Dtype dt = input->dtype();
    if (!is_floating_point(dt))
        throw DtypeMismatch("a floating-point dtype", std::string(dtype_name(dt)), "lstm");

    const Device dev = input->device();
    const std::int64_t I = opts.input_size, H = opts.hidden_size, G = 4 * H, P = opts.proj_size;
    const auto out = lstm_output_shapes(opts);
    check_operand(err, input, "input", Shape{opts.seq_len, opts.batch_size, I}, dt, dev);
    check_operand(err, h0, "h0", out[1], dt, dev);
    check_operand(err, c0, "c0", out[2], dt, dev);

    const std::size_t n_weights = P > 0 ? 5 : 4;
    if (weights.size() != n_weights)
        err.invalid_argument("expected " + std::to_string(n_weights) +
                             " weights {weight_ih, weight_hh, bias_ih, bias_hh" +
                             (P > 0 ? ", weight_hr}" : "}") + ", got " +
                             std::to_string(weights.size()));
    const std::array<std::string, 5> names{"weight_ih", "weight_hh", "bias_ih", "bias_hh",
                                           "weight_hr"};
    const std::array<Shape, 5> shapes{Shape{G, I}, Shape{G, out[1][2]}, Shape{G}, Shape{G},
                                      Shape{P, H}};
    for (std::size_t i = 0; i < n_weights; ++i)
        check_operand(err, weights[i], names[i], shapes[i], dt, dev);
}

}  // namespace

void LstmBackward::accumulate_barrier_grad(std::uint32_t input_nr, Storage grad) {
    if (input_nr >= grad_slots_.size())
        ErrorBuilder("LstmBackward")
            .fail("gradient for output slot " + std::to_string(input_nr) + " of a 3-output node");
    // Slots belong to one backward pass (PassSlots): what a pass delivered
    // and never ran (an exception, a pruned autograd.grad) must not seed the
    // next one, nor a pass nested in this one take what it holds.
    passes_.enter(grad_slots_, {});
    auto& slot = grad_slots_[input_nr];
    if (!slot.has_value()) {
        slot = std::move(grad);
        return;
    }
    // The first buffer may be one the engine also routed elsewhere; a CPU
    // add would write into it in place, so sum into a copy we own.
    Storage sum = own_grad_copy(*slot);
    accumulate_into(sum, grad);
    slot = std::move(sum);
}

std::vector<Storage> LstmBackward::apply_barrier() {
    passes_.enter(grad_slots_, {});
    const auto shapes = lstm_output_shapes(opts);
    std::array<Storage, 3> grads;
    for (std::size_t i = 0; i < grads.size(); ++i) {
        // An output the loss never reached contributes a zero gradient.
        grads[i] = grad_slots_[i].has_value() ? std::move(*grad_slots_[i])
                                              : make_zero_storage(shapes[i], dtype, device);
    }
    grad_slots_ = {};

    auto& be = backend::Dispatcher::for_device(device);
    return be.lstm_backward(grads[0], grads[1], grads[2], saved_input, saved_h0, saved_weights,
                            gates_all, cells_all, opts, dtype);
}

std::vector<Storage> LstmBackward::apply(Storage grad_out) {
    accumulate_barrier_grad(0, std::move(grad_out));
    return apply_barrier();
}

namespace {

[[noreturn]] void refuse_create_graph() {
    ErrorBuilder("LstmBackward")
        .not_implemented("create_graph=True is not yet supported for op 'LstmBackward'; use "
                         "retain_graph=True with multiple backward calls instead");
}

void autocast_lstm_operands(TensorImplPtr& input,
                            TensorImplPtr& h0,
                            TensorImplPtr& c0,
                            std::vector<TensorImplPtr>& weights,
                            std::optional<amp::AutocastGuard>& guard) {
    if (!input || !h0 || !c0)
        return;
    const auto autocast_dt = amp::active_dtype(input->device());
    if (!autocast_dt)
        return;
    const bool all_eligible =
        amp::is_autocast_eligible(input->dtype()) && amp::is_autocast_eligible(h0->dtype()) &&
        amp::is_autocast_eligible(c0->dtype()) &&
        std::all_of(weights.begin(), weights.end(), [](const TensorImplPtr& w) {
            return w && amp::is_autocast_eligible(w->dtype());
        });
    if (!all_eligible)
        return;
    const Dtype eff_dt = (c0->dtype() == *autocast_dt) ? *autocast_dt : Dtype::F32;
    guard.emplace(*autocast_dt, input->device(), /*enabled=*/false);
    input = astype_op(input, eff_dt);
    h0 = astype_op(h0, eff_dt);
    c0 = astype_op(c0, eff_dt);
    for (auto& w : weights)
        w = astype_op(w, eff_dt);
}

void record_saved_sources(LstmBackward& bwd,
                          const TensorImplPtr& input_in,
                          const TensorImplPtr& h0_in,
                          const TensorImplPtr& c0_in,
                          const std::vector<TensorImplPtr>& weights_in) {
    // Versions, by forward argument, of what BPTT reads by value — input,
    // h0 and the weight matrices — whether or not it takes a gradient: a
    // write into a weight that does not require grad still changes every
    // other input's gradient.  c0 and the biases are not read, so their
    // entries stay empty and a write into them after forward is allowed.
    std::vector<TensorImplPtr> source_tensors{input_in, h0_in, c0_in};
    for (const auto& w : weights_in)
        source_tensors.push_back(w);
    const auto read_by_bptt = [](std::size_t arg) { return arg != 2 && arg != 5 && arg != 6; };
    std::vector<std::int64_t> versions;
    versions.reserve(source_tensors.size());
    bwd.saved_sources.reserve(source_tensors.size());
    for (std::size_t arg = 0; arg < source_tensors.size(); ++arg) {
        const bool held = read_by_bptt(arg);
        if (held && source_tensors[arg])
            source_tensors[arg]->check_not_inference_saved("lstm");
        bwd.saved_sources.push_back(held ? source_tensors[arg] : TensorImplPtr{});
        versions.push_back(held ? source_tensors[arg]->version() : 0);
    }
    bwd.set_saved_versions(std::move(versions));
}

}  // namespace

void LstmBackward::accumulate_barrier_grad_for_graph(std::uint32_t /*input_nr*/,
                                                     TensorImplPtr /*grad*/) {
    refuse_create_graph();
}

std::vector<TensorImplPtr> LstmBackward::apply_barrier_for_graph() {
    refuse_create_graph();
}

void LstmBackward::release_saved() {
    saved_input = Storage{CpuStorage{}};
    saved_h0 = Storage{CpuStorage{}};
    saved_weights.clear();
    gates_all = Storage{CpuStorage{}};
    cells_all = Storage{CpuStorage{}};
    saved_sources.clear();
    grad_slots_ = {};
    passes_ = {};
}

void LstmBackward::validate_versions() {
    // ``saved_sources`` is indexed by forward argument; an entry for a
    // tensor that is not held is empty and passes.
    for (std::size_t k = 0; k < saved_sources.size(); ++k)
        check_saved_version(saved_sources[k], k, "lstm", k);
}

std::vector<TensorImplPtr> LstmBackward::forward(const TensorImplPtr& input_in,
                                                 const TensorImplPtr& h0_in,
                                                 const TensorImplPtr& c0_in,
                                                 const std::vector<TensorImplPtr>& weights_in,
                                                 const backend::IBackend::LstmOpts& opts) {
    TensorImplPtr input = input_in;
    TensorImplPtr h0 = h0_in;
    TensorImplPtr c0 = c0_in;
    std::vector<TensorImplPtr> weights = weights_in;
    std::optional<amp::AutocastGuard> autocast_off;
    autocast_lstm_operands(input, h0, c0, weights, autocast_off);
    validate_operands(input, h0, c0, weights, opts);

    auto& be = backend::Dispatcher::for_device(input->device());
    const Dtype dt = input->dtype();
    const Device dev = input->device();
    const int H = opts.hidden_size;

    std::vector<Storage> w_storages;
    w_storages.reserve(weights.size());
    for (const auto& w : weights)
        w_storages.push_back(w->storage());

    const bool needs_grad =
        GradMode::is_enabled() &&
        (input->requires_grad() || h0->requires_grad() || c0->requires_grad() ||
         std::any_of(weights.begin(), weights.end(),
                     [](const TensorImplPtr& w) { return w->requires_grad(); }));

    const auto shapes = lstm_output_shapes(opts);
    const Shape& out_shape = shapes[0];
    const Shape& hn_shape = shapes[1];
    const Shape& cn_shape = shapes[2];

    // Open an OpScope so the trace sees ``lstm`` as a single 3-output
    // op.  Attrs carry the shape parameters the compile-path emitter
    // needs to set up the MPSGraph LSTMDescriptor.  Without this the
    // C++ fused call would run invisible to the trace and downstream
    // ops would treat its outputs as fresh external feeds.
    OpScopeFull scope{"lstm", dev, dt, out_shape};
    scope.set_attr("hidden_size", static_cast<std::int64_t>(H));
    scope.set_attr("num_layers", static_cast<std::int64_t>(opts.num_layers));
    scope.set_attr("batch_first", opts.batch_first);
    scope.set_attr("bidirectional", opts.bidirectional);
    scope.set_attr("proj_size", static_cast<std::int64_t>(opts.proj_size));
    scope.set_attr("has_bias", weights.size() >= 4);

    auto register_trace_outputs = [&](const TensorImplPtr& out_t, const TensorImplPtr& hn_t,
                                      const TensorImplPtr& cn_t) {
        auto* trc = ::lucid::compile::current_tracer();
        if (trc == nullptr)
            return;
        // Pass all inputs on the first on_op_io call; the subsequent
        // calls pass empty input lists so the trace's input record
        // doesn't duplicate (see Tracer's first-vs-subsequent
        // detection logic).
        std::vector<TensorImplPtr> all_inputs{input, h0, c0};
        for (const auto& w : weights)
            all_inputs.push_back(w);
        trc->on_op_io(all_inputs, out_t);
        trc->on_op_io({}, hn_t);
        trc->on_op_io({}, cn_t);
    };

    // Backend storage convention: ``lstm_forward`` / ``lstm_forward_train``
    // emit hn / cn as 2-D ``(B, Hrec/H)`` arrays, while Lucid's Python
    // API contract is 3-D ``(num_layers, B, Hrec/H)``.  Without an
    // explicit reshape the TensorImpl claims the 3-D shape but the
    // underlying MLX storage still reports 2 dims — any downstream op
    // that walks the storage rank (``sum`` / ``flatten`` / …) then
    // fails with ``Invalid axis 2 for array with 2 dimensions``.  Add
    // the unsqueeze here so the storage rank matches what callers see.
    const Shape hn_storage_2d{hn_shape[1], hn_shape[2]};
    const Shape cn_storage_2d{cn_shape[1], cn_shape[2]};

    if (!needs_grad) {
        // Backends that don't implement projection in lstm_forward route
        // through the training kernel for proj_size > 0 and discard the
        // saved gates/cells outputs.
        auto res = opts.proj_size > 0
                       ? be.lstm_forward_train(input->storage(), h0->storage(), c0->storage(),
                                               w_storages, opts, dt)
                       : be.lstm_forward(input->storage(), h0->storage(), c0->storage(), w_storages,
                                         opts, out_shape, dt);
        if (res.size() < 3)
            ErrorBuilder("lstm").fail("backend returned < 3 outputs");
        auto hn_3d = be.reshape(res[1], hn_storage_2d, hn_shape, dt);
        auto cn_3d = be.reshape(res[2], cn_storage_2d, cn_shape, dt);
        auto out_t = std::make_shared<TensorImpl>(std::move(res[0]), out_shape, dt, dev, false);
        auto hn_t = std::make_shared<TensorImpl>(std::move(hn_3d), hn_shape, dt, dev, false);
        auto cn_t = std::make_shared<TensorImpl>(std::move(cn_3d), cn_shape, dt, dev, false);
        register_trace_outputs(out_t, hn_t, cn_t);
        return {out_t, hn_t, cn_t};
    }

    auto res =
        be.lstm_forward_train(input->storage(), h0->storage(), c0->storage(), w_storages, opts, dt);
    if (res.size() < 5)
        ErrorBuilder("lstm").fail("lstm_forward_train returned < 5 outputs");

    auto hn_3d = be.reshape(res[1], hn_storage_2d, hn_shape, dt);
    auto cn_3d = be.reshape(res[2], cn_storage_2d, cn_shape, dt);
    std::vector<TensorImplPtr> outputs{
        std::make_shared<TensorImpl>(std::move(res[0]), out_shape, dt, dev, true),
        std::make_shared<TensorImpl>(std::move(hn_3d), hn_shape, dt, dev, true),
        std::make_shared<TensorImpl>(std::move(cn_3d), cn_shape, dt, dev, true)};

    auto bwd = std::make_shared<LstmBackward>();
    bwd->saved_input = input->storage();
    bwd->saved_h0 = h0->storage();
    bwd->saved_weights = w_storages;
    // The biases (weights 2 and 3) are not read by BPTT: keep a zero-element
    // placeholder of the same device and dtype in their slots rather than
    // their bytes, so nothing holds them for backward.
    for (std::size_t k = 2; k < 4 && k < bwd->saved_weights.size(); ++k)
        bwd->saved_weights[k] = be.zeros(Shape{0}, dt);
    bwd->gates_all = std::move(res[3]);
    bwd->cells_all = std::move(res[4]);
    bwd->opts = opts;
    bwd->dtype = dt;
    bwd->device = dev;

    // Build the edge list manually: {input, h0, c0, wih, whh, bih, bhh}.
    // NaryKernel is not used here because the number of edges is dynamic
    // (it depends on the number of weight tensors supplied).
    std::vector<TensorImplPtr> edge_tensors{input, h0, c0};
    for (const auto& w : weights)
        edge_tensors.push_back(w);

    std::vector<Edge> edges;
    for (const auto& t : edge_tensors) {
        if (!t->requires_grad()) {
            // A null edge signals that this input does not participate in
            // gradient accumulation; the backward skips it.
            edges.emplace_back(nullptr, 0);
        } else {
            if (t->is_leaf() && !t->grad_fn())
                t->set_grad_fn(std::make_shared<AccumulateGrad>(t));
            edges.emplace_back(t->grad_fn(), t->grad_output_nr());
        }
    }
    bwd->set_next_edges(std::move(edges));
    record_saved_sources(*bwd, input_in, h0_in, c0_in, weights_in);

    // Every output shares the one node; ``grad_output_nr`` tells the engine
    // which barrier slot an arriving gradient belongs to.
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        outputs[i]->set_grad_fn(bwd);
        outputs[i]->set_grad_output_nr(static_cast<std::uint32_t>(i));
        outputs[i]->set_leaf(false);
    }
    return outputs;
}

std::vector<TensorImplPtr> lstm_op(const TensorImplPtr& input,
                                   const TensorImplPtr& h0,
                                   const TensorImplPtr& c0,
                                   const std::vector<TensorImplPtr>& weights,
                                   const backend::IBackend::LstmOpts& opts) {
    return LstmBackward::forward(input, h0, c0, weights, opts);
}

}  // namespace lucid
